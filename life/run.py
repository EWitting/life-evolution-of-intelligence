"""Generation loop, logging and recording (ADR-009, ADR-014).

run_evolution(exp, ruleset, fitness_fn, ...) simulates one shared world per generation with one agent per
genome, computes fitness from the per-agent stats, and evolves. It writes
runs/<name>/<timestamp>/{config.json, ruleset.json, layout.json, fitness.csv, best_genome.npz, population.npz,
recording.npz, dashboard.html}. A later experiment can warm-start from population.npz via load_population.
"""
from __future__ import annotations
import csv
import json
import os
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from . import brain
from .config import ExperimentConfig
from .evolution import next_generation
from .ruleset import Ruleset, RuleArrays
from .sensors import input_names, ray_offsets, observe_all, flatten_obs
from .world import init_world, step_world, take_agents, WorldState
from . import actions as A

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"
STOP_CHECK = 100   # ticks between checks whether anyone is still alive (unrecorded simulations stop when nobody is)
COMPACT = (2, 4, 8)   # an unrecorded simulation goes on with 1/2, 1/4, 1/8 of the arrays once that few agents live


def make_layout(exp: ExperimentConfig) -> brain.Layout:
    names = input_names(exp.vision, exp.body)
    return brain.build_layout(exp.brain, n_in=len(names), n_out=A.NUM_ACTIONS, in_names=names)


def make_tick(exp: ExperimentConfig, layout: brain.Layout):
    """One world tick for all agents: observe -> brains -> world. Shared by run_evolution and Lab.
    tick(rules, pop, world, bstate, sig, key, override) -> (world, bstate, sig, acts, obs, ev)
    sig[N, 3] are the world signals of the last tick (brain.WORLD_SIGNALS: reward, pain, food).
    `override[N]` replaces an agent's chosen action when >= 0 (Lab uses it; pass -1s otherwise).
    `ids` [n]: when the arrays hold only n of the N agents, their indices in the full population (every agent keeps
    the random draws of its index)."""
    offsets = jnp.asarray(ray_offsets(exp.vision))
    wcfg, vcfg, bcfg = exp.world, exp.vision, exp.brain
    N = wcfg.num_agents

    def tick(rules: RuleArrays, pop: brain.Genome, world: WorldState, bstate: brain.BrainState,
             sig: jnp.ndarray, key: jax.Array, override: jnp.ndarray, ids: jnp.ndarray | None = None):
        obs = flatten_obs(observe_all(vcfg, wcfg, rules, world, offsets, exp.body, jax.random.fold_in(key, 7), ids))
        keys = jax.random.split(key, N)
        if ids is not None:
            keys = keys[ids]
        bstate, acts = jax.vmap(brain.step, in_axes=(None, None, 0, 0, 0, 0, 0))(
            bcfg, layout, pop, bstate, obs, sig, keys)
        acts = jnp.where(override >= 0, override, acts)
        effort = bstate.x[:, layout.n_in:].mean(axis=1)
        world, ev = step_world(wcfg, rules, world, acts, key, effort)
        food = ev["gained"] / wcfg.food_scale
        sig = jnp.stack([food - ev["pain"], ev["pain"], food], axis=1)
        return world, bstate, sig, acts, obs, ev

    return tick


CHECKPOINT_EVERY = 10   # generations between checkpoint.npz writes (the population being evaluated)
SLOT_KEY = jax.random.PRNGKey(20261019)   # the random draws of the population slots (ProjectionSpec.per_slot)


def make_simulate(exp: ExperimentConfig, record: bool, compact: bool = True, record_obs: bool = False):
    """Returns a jitted simulate(rules, pop, key) -> (stats, recording_or_None) for one generation.
    With record=True the recording holds per-tick arrays plus weight snapshots every record_weights_every ticks.
    `record_obs` adds the raw observation and the world signals each brain received (for replays,
    scripts/probes/bridge.py). An unrecorded simulation drops dead agents from its arrays (COMPACT); `compact=False` keeps them, with the
    same result."""
    layout = make_layout(exp)
    tick = make_tick(exp, layout)
    T = exp.evolution.ticks_per_generation
    N = exp.world.num_agents
    every = exp.evolution.record_weights_every if record else T
    assert T % every == 0, f"ticks_per_generation ({T}) must be a multiple of record_weights_every ({every})"

    @jax.jit
    def simulate(rules: RuleArrays, pop: brain.Genome, key: jax.Array):
        # fewer genomes than agents: every genome lives as N / P siblings (agents i*k .. i*k+k-1 share genome i)
        P = pop.b.shape[0]
        assert N % P == 0, f"{N} agents cannot be split evenly over {P} genomes"
        # birth: the same for all siblings of a genome; per-slot synapses from the slot's fixed key (ProjectionSpec.per_slot)
        if layout.has_per_slot:
            bstate0 = jax.vmap(lambda g, k: brain.init_state(g, layout, key=k))(pop, jax.random.split(SLOT_KEY, P))
        else:
            bstate0 = jax.vmap(lambda g: brain.init_state(g, layout))(pop)
        pop = brain.block_genome(layout, pop)                             # a life reads only the plastic block
        if P != N:
            sib = lambda x: jnp.repeat(x, N // P, axis=0)
            pop, bstate0 = jax.tree_util.tree_map(sib, pop), jax.tree_util.tree_map(sib, bstate0)
        # rules with a leading axis of 2 = two phases of one life: phase 1 from tick WorldConfig.switch_tick on
        phased = rules.food_value.ndim == 2
        # (WorldConfig.switch_every > 0: back and forth every switch_every ticks from then on)
        sw, sw_every = exp.world.switch_tick, exp.world.switch_every
        phase_at = (lambda tk: (tk >= sw) & (((tk - sw) // sw_every) % 2 == 0)) if sw_every > 0 else (lambda tk: tk >= sw)
        rules_at = (lambda tk: jax.tree_util.tree_map(lambda a: a[phase_at(tk).astype(jnp.int32)], rules))             if phased else (lambda tk: rules)
        M = rules.food_value.shape[-1]
        half = exp.world.switch_tick if phased else T // 2

        def one_tick(carry, k, pop=pop, ids=None):
            world, bstate, sig, eats, tsum, fed, meal, fed_meal = carry
            tsum = tsum + world.temp * world.alive
            level = world.food / exp.world.max_food * world.alive
            fed = fed + level
            fed_meal = fed_meal + level * meal
            late = world.tick >= half
            r = rules_at(world.tick)
            world, bstate, sig, acts, obs, ev = tick(r, pop, world, bstate, sig, k, -jnp.ones(sig.shape[0], jnp.int32), ids)
            meal = jnp.maximum(meal, (ev["gained"] > 0).astype(jnp.float32))
            oh = jax.nn.one_hot(ev["ate"], M, dtype=jnp.float32).at[:, 0].set(0.0)
            oh = oh.at[:, 0].set(((ev["ate"] > 0) & (r.pain_value[ev["ate"]] > 0)).astype(jnp.float32))
            eats = eats + jnp.stack([oh * (1.0 - late), oh * late], axis=1)
            rec = None
            if record:
                rec = dict(grid=world.grid_obj.astype(jnp.int16), pos=world.pos.astype(jnp.int16),
                           dir=world.dir.astype(jnp.int8), alive=world.alive, held=world.held.astype(jnp.int16),
                           food=world.food, pain=world.pain, action=acts.astype(jnp.int8), mod=bstate.mod,
                           ate=ev["ate"].astype(jnp.int16),
                           x=bstate.x.astype(jnp.float16))
                if record_obs:
                    rec.update(obs=obs, sig_in=carry[2])
            return (world, bstate, sig, eats, tsum, fed, meal, fed_meal), rec

        def stats_of(carry):
            world, _, _, eats, tsum, fed, _, fed_meal = carry
            out = dict(alive_ticks=world.alive_ticks, eaten=world.eaten, pain=world.pain_total, alive=world.alive,
                       food=jnp.maximum(world.food, 0.0), eats=eats, fed=fed, fed_meal=fed_meal, tsum=tsum)
            if exp.world.snakes:
                out.update(bites=world.bites, rattles=world.rattles)
            if exp.world.thorn_food or exp.world.thorn_pain:
                out.update(pricks=world.pricks)
            return out

        def chunk(carry, keys):
            carry, recs = jax.lax.scan(one_tick, carry, keys)
            snap = carry[1].w.astype(jnp.float16) if record else None
            return carry, (recs, snap)

        kw, ks = jax.random.split(key)
        world = init_world(exp.world, rules_at(jnp.int32(0)), kw)
        bstate = bstate0
        sig = jnp.zeros((N, 3), jnp.float32)
        eats = jnp.zeros((N, 2, M), jnp.float32)
        tsum = jnp.zeros(N, jnp.float32)
        fed, meal, fed_meal = jnp.zeros(N, jnp.float32), jnp.zeros(N, jnp.float32), jnp.zeros(N, jnp.float32)
        carry = (world, bstate, sig, eats, tsum, fed, meal, fed_meal)
        if not record and T > STOP_CHECK and T % STOP_CHECK == 0:
            # nothing to record: stop once every agent is dead (no statistic changes after that). Long lives are
            # mostly empty ticks in the first generations of a stage; the random keys per tick are unchanged
            keys = jax.random.split(ks, T).reshape(T // STOP_CHECK, STOP_CHECK, -1)
            # The dead cost as much as the living, so once half the agents are dead the arrays are cut to the
            # living half (then a quarter, an eighth): `ids` maps a row to its agent, the order is kept (the USE
            # loop goes by index), the dead are seen by nobody, and what a dropped agent had is in `stats`.
            # (Pain still scheduled for a dead agent, WorldConfig.sickness_delay, is no longer counted.)
            sizes = [N // d for d in COMPACT if compact and N % d == 0]
            i, ids, pop_n, stats = jnp.int32(0), None, pop, None
            for nxt in sizes + [0]:
                def body(c, pop_n=pop_n, ids=ids):
                    i, carry = c
                    carry, _ = jax.lax.scan(lambda cc, k: one_tick(cc, k, pop_n, ids), carry, keys[i])
                    return i + 1, carry

                i, carry = jax.lax.while_loop(
                    lambda c, nxt=nxt: (c[0] < T // STOP_CHECK) & (c[1][0].alive.sum() > nxt), body, (i, carry))
                now = stats_of(carry)
                stats = now if ids is None else {k: v.at[ids].set(now[k]) for k, v in stats.items()}
                if nxt:
                    sel = jnp.argsort(~carry[0].alive, stable=True)[:nxt]   # the living first, in their order
                    ids = sel if ids is None else ids[sel]
                    pop_n = jax.tree_util.tree_map(lambda x: x[sel], pop_n)
                    carry = (take_agents(carry[0], sel),) + jax.tree_util.tree_map(lambda x: x[sel], carry[1:])
            recs = snaps = None
        else:
            keys = jax.random.split(ks, T).reshape(T // every, every, -1)
            carry, (recs, snaps) = jax.lax.scan(chunk, carry, keys)
            stats = stats_of(carry)
        # eats[N, 2, M]: objects eaten per local id in the first and second half of life (learning curves);
        # column 0 (empty) instead counts eating anything painful (poison), whatever its id in this world.
        # fed: sum over ticks alive of the food level as a fraction of max_food ('well-fed lifetime');
        # fed_meal: the same, counted only from the tick after the agent's first (positive) meal
        stats["temp_mean"] = stats.pop("tsum") / jnp.maximum(stats["alive_ticks"], 1)
        if record:
            recs = jax.tree_util.tree_map(lambda a: a.reshape((T,) + a.shape[2:]), recs)
            recs["w_snap"] = snaps
        return stats, recs

    return simulate


def latest_run(experiment_name: str) -> Path:
    runs = sorted(d for d in (RUNS_DIR / experiment_name).iterdir() if (d / "population.npz").exists())
    if not runs:
        raise FileNotFoundError(f"no finished runs under {RUNS_DIR / experiment_name}")
    return runs[-1]


def load_population(run_dir: Path | str, exp: ExperimentConfig | None = None, seed: int = 0,
                    file: str = "population.npz") -> brain.Genome:
    """The final population of an earlier run (ADR-012). With `exp`, the population is remapped onto exp's
    layout by region and input-feature names (brain.remap_genomes): new regions/projections start silent.
    If the population size differs from exp.world.num_agents it is tiled or truncated. file="checkpoint.npz": the
    population of a run's last checkpoint (CHECKPOINT_EVERY), for a run stopped early."""
    run_dir = Path(run_dir)
    old = make_layout(ExperimentConfig.from_json((run_dir / "config.json").read_text()))
    with np.load(run_dir / file) as f:
        pop = brain.genome_from_npz(f, old)
    if exp is None:
        return pop
    new = make_layout(exp)
    P, want = pop.b.shape[0], exp.world.num_agents // max(1, exp.evolution.siblings)
    if P != want:
        idx = jnp.arange(want) % P
        pop = jax.tree_util.tree_map(lambda a: a[idx], pop)
    return brain.remap_genomes(pop, old, new, jax.random.PRNGKey(seed))


@contextmanager
def recording_lock(verbose: bool = True):
    """Only one process on this machine records at a time: a run needs little memory while it evolves, but recording
    the last generation and writing the dashboard briefly needs several times that, and runs finishing together ran
    out of memory. A lock file in the system temp directory (per OS: WSL and Windows each have their own)."""
    path = Path(tempfile.gettempdir()) / "life-recording.lock"
    with open(path, "a+b") as f:
        t0 = time.time()
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            while True:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(2)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        if verbose and time.time() - t0 > 1:
            print(f"waited {time.time() - t0:.0f} s for another run's recording", flush=True)
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def save_recording(out_dir: Path, exp: ExperimentConfig, ruleset: Ruleset, layout: brain.Layout, pop: brain.Genome,
                   recs: dict, best: int, rows: list[dict] | None = None, dashboard: bool = True,
                   appearance=None, rules: RuleArrays | None = None):
    """Write recording.npz (+ genome arrays needed to interpret it) and the interactive dashboard. `rules`: the
    rules of the recorded life; their food and pain values are stored (per phase, [2, M], with a reversal), so that
    what is poison when is known exactly (dashboard.poison_phases)."""
    extra = {} if appearance is None else {"appearance": np.asarray(appearance, np.float32)}
    if rules is not None:
        extra.update(food_value=np.asarray(rules.food_value, np.float32), pain_value=np.asarray(rules.pain_value, np.float32))
    np.savez_compressed(out_dir / "recording.npz", best=best, w0=np.asarray(pop.w0, np.float16), **extra,
                        eta=np.asarray(pop.eta, np.float16), mask=np.asarray(pop.mask, np.uint8),
                        **{k: np.asarray(v) for k, v in recs.items()})
    (out_dir / "layout.json").write_text(json.dumps(layout.to_json_dict()))
    if dashboard:
        from .dashboard import export_dashboard
        export_dashboard(out_dir)


def run_evolution(exp: ExperimentConfig, ruleset: Ruleset, fitness_fn: Callable[[dict], jnp.ndarray],
                  rules_for_generation: Callable[[int, jax.Array], RuleArrays] | None = None,
                  init_population: brain.Genome | None = None,
                  out_dir: Path | None = None, verbose: bool = True, dashboard: bool = True,
                  row_extra: Callable[[dict, jnp.ndarray], dict] | None = None,
                  start_generation: int = 0, prior_rows: list[dict] | None = None) -> dict:
    """Run the full experiment. `rules_for_generation(gen, key)` may return different RuleArrays per generation
    (same shapes) to randomize the environment, or RuleArrays stacked along a leading axis of size
    `evolution.episodes` to vary it per episode; default uses ruleset.to_arrays throughout.
    `init_population` (from load_population) warm-starts from an earlier run instead of random genomes.
    `row_extra(stats, fitness)` adds experiment-specific columns (e.g. poison fraction) to fitness.csv.
    `start_generation`, `prior_rows`: resume a stopped run from its checkpoint (the population evaluated in that
    generation) in its own `out_dir`; the random keys of the earlier generations are drawn and skipped, so the run
    goes on as it would have."""
    ecfg = exp.evolution
    layout = make_layout(exp)
    key = jax.random.PRNGKey(ecfg.seed)
    key, kinit = jax.random.split(key)
    sib = max(1, ecfg.siblings)
    assert exp.world.num_agents % sib == 0, "num_agents must be a multiple of evolution.siblings"
    n_genomes = exp.world.num_agents // sib
    if init_population is None:
        pop = jax.vmap(lambda k: brain.init_genome(k, layout))(jax.random.split(kinit, n_genomes))
    else:
        pop = init_population
        assert pop.b.shape == (n_genomes, layout.n), \
            f"init population has shape {pop.b.shape}, experiment needs {(n_genomes, layout.n)}"
    base_rules = ruleset.to_arrays(exp.vision.appearance_dim)
    rules_fn = rules_for_generation or (lambda gen, k: base_rules)
    E = ecfg.episodes
    sim = jax.jit(jax.vmap(make_simulate(exp, record=False), in_axes=(0, None, 0)))
    ra = ecfg.record_agents
    exp_rec = exp
    if 0 < ra < exp.world.num_agents:   # the dashboard shows a sample of the population in a smaller world
        ra = min(ra, n_genomes)
        f = ra / exp.world.num_agents
        from dataclasses import replace as _replace
        exp_rec = _replace(exp, world=_replace(exp.world, num_agents=ra, height=max(8, round(exp.world.height * f ** 0.5)),
                                               width=max(8, round(exp.world.width * f ** 0.5)),
                                               patches=round(exp.world.patches * f) if exp.world.patches else 0))
    sim_rec = make_simulate(exp_rec, record=True)

    def per_episode(rules: RuleArrays) -> RuleArrays:
        if rules.food_value.ndim == 1 + (exp.world.switch_tick > 0):
            return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (E,) + x.shape), rules)
        assert rules.food_value.shape[0] == E, "rules_for_generation must return E stacked rulesets or one"
        return rules

    if out_dir is None:   # unique even when several runs start in the same second
        base = RUNS_DIR / exp.name / datetime.now().strftime("%Y%m%d-%H%M%S")
        base.parent.mkdir(parents=True, exist_ok=True)
        for k in range(100):
            out_dir = base if k == 0 else base.with_name(f"{base.name}-{k}")
            try:
                out_dir.mkdir()
                break
            except FileExistsError:
                continue
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(exp.to_json())
    (out_dir / "ruleset.json").write_text(json.dumps(ruleset.to_json_dict()))
    rows = list(prior_rows or [])
    t0 = time.time() - (rows[-1]["seconds"] if rows else 0.0)
    stats = None
    for _ in range(start_generation):
        key = jax.random.split(key, 4)[0]
    for gen in range(start_generation, ecfg.generations):
        key, kr, ks, ke = jax.random.split(key, 4)
        rules = per_episode(rules_fn(gen, kr))
        last = gen == ecfg.generations - 1
        ep_keys = jax.random.split(ks, E)
        stats, _ = sim(rules, pop, ep_keys)
        stats = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32).mean(axis=0), stats)
        fit = fitness_fn(stats).reshape(n_genomes, sib).mean(axis=1)   # a genome's fitness: the mean of its siblings
        row = dict(gen=gen, fit_mean=float(fit.mean()), fit_max=float(fit.max()), fit_median=float(jnp.median(fit)),
                   alive_ticks=float(stats["alive_ticks"].mean()), eaten=float(stats["eaten"].mean()),
                   pain=float(stats["pain"].mean()), fed=float(stats["fed"].mean()), fed_meal=float(stats["fed_meal"].mean()), survivors=float(stats["alive"].sum()),
                   seconds=round(time.time() - t0, 1))
        rows.append(row)
        if exp.world.temperature:
            row["temp_mean"] = float(stats["temp_mean"].mean())
        if row_extra is not None:
            row.update({k: float(v) for k, v in row_extra(stats, fit).items()})
        if verbose:
            print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if last:
            best = int(jnp.argmax(fit))
            rules0 = jax.tree_util.tree_map(lambda x: x[0], rules)
            pop_rec, best_rec = pop, best
            if exp_rec is not exp:   # every (N / ra)-th agent of the fitness ranking, best first
                pick = jnp.argsort(-fit)[:: max(1, n_genomes // ra)][:ra]   # one individual per genome shown
                pop_rec, best_rec = jax.tree_util.tree_map(lambda x: x[pick], pop), 0
            np.savez(out_dir / "best_genome.npz", **{k: np.asarray(v[best]) for k, v in pop._asdict().items()})
            np.savez_compressed(out_dir / "population.npz", **{k: np.asarray(v) for k, v in pop._asdict().items()})
        else:
            if out_dir is not None and gen % CHECKPOINT_EVERY == CHECKPOINT_EVERY - 1:
                # so that a run stopped early can still be looked into (not a finished run: latest_run and
                # load_population read population.npz; load this one with load_population(run, file=...))
                # written to a temporary file and renamed: a run killed while writing keeps its last checkpoint
                np.savez_compressed(out_dir / "checkpoint.tmp.npz", generation=gen,
                                    **{k: np.asarray(v) for k, v in pop._asdict().items()})
                os.replace(out_dir / "checkpoint.tmp.npz", out_dir / "checkpoint.npz")
                with open(out_dir / "fitness.csv", "w", newline="") as f:   # the rows so far (a resume reads them)
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)
            pop = next_generation(ke, pop, fit, ecfg, layout)
    with open(out_dir / "fitness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    rules_last = jax.tree_util.tree_map(lambda x: x[0], rules)
    app = rules_last.appearance if rules_last.appearance.ndim == 2 else rules_last.appearance[0]
    with recording_lock(verbose):   # the population is saved already; the recording needs much more memory
        _, recs = sim_rec(rules0, pop_rec, ep_keys[0])   # episode 0 of the last generation again, recording on
        save_recording(out_dir, exp, ruleset, layout, pop_rec, recs, best_rec, rows, dashboard=dashboard,
                       appearance=app, rules=rules_last)
        del recs
    if verbose:
        print(f"wrote {out_dir}" + ("  (open dashboard.html in a browser)" if dashboard else ""))
    return dict(out_dir=out_dir, rows=rows, pop=pop, stats=stats)
