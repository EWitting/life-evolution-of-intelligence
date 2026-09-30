"""Generation loop, logging and recording (ADR-009, ADR-014).

run_evolution(exp, ruleset, fitness_fn, ...) simulates one shared world per generation with one agent per
genome, computes fitness from the per-agent stats, and evolves. It writes
runs/<name>/<timestamp>/{config.json, ruleset.json, layout.json, fitness.csv, best_genome.npz, population.npz,
recording.npz, dashboard.html}. A later experiment can warm-start from population.npz via load_population.
"""
from __future__ import annotations
import csv
import json
import time
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
from .sensors import obs_size, ray_offsets, observe_all, flatten_obs
from .world import init_world, step_world, WorldState
from . import actions as A

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"


def make_layout(exp: ExperimentConfig) -> brain.Layout:
    return brain.build_layout(exp.brain, n_in=obs_size(exp.vision), n_out=A.NUM_ACTIONS)


def make_tick(exp: ExperimentConfig, layout: brain.Layout):
    """One world tick for all agents: observe -> brains -> world. Shared by run_evolution and Lab.
    tick(rules, pop, world, bstate, mod, key, override) -> (world, bstate, mod, acts, obs)
    `override[N]` replaces an agent's chosen action when >= 0 (Lab uses it; pass -1s otherwise)."""
    offsets = jnp.asarray(ray_offsets(exp.vision))
    wcfg, vcfg, bcfg = exp.world, exp.vision, exp.brain
    N = wcfg.num_agents

    def tick(rules: RuleArrays, pop: brain.Genome, world: WorldState, bstate: brain.BrainState,
             mod: jnp.ndarray, key: jax.Array, override: jnp.ndarray):
        obs = flatten_obs(observe_all(vcfg, wcfg, rules, world, offsets))
        keys = jax.random.split(key, N)
        bstate, acts = jax.vmap(brain.step, in_axes=(None, None, 0, 0, 0, 0, 0))(
            bcfg, layout, pop, bstate, obs, mod, keys)
        acts = jnp.where(override >= 0, override, acts)
        world, ev = step_world(wcfg, rules, world, acts, key)
        if bcfg.modulator == "reward":
            mod = ev["gained"] / wcfg.food_scale - ev["pain"]
        else:
            mod = jnp.ones(N, jnp.float32)
        return world, bstate, mod, acts, obs

    return tick


def make_simulate(exp: ExperimentConfig, record: bool):
    """Returns a jitted simulate(rules, pop, key) -> (stats, recording_or_None) for one generation.
    With record=True the recording holds per-tick arrays plus weight snapshots every record_weights_every ticks."""
    layout = make_layout(exp)
    tick = make_tick(exp, layout)
    T = exp.evolution.ticks_per_generation
    N = exp.world.num_agents
    every = exp.evolution.record_weights_every if record else T
    assert T % every == 0, f"ticks_per_generation ({T}) must be a multiple of record_weights_every ({every})"
    no_override = -jnp.ones(N, jnp.int32)

    @jax.jit
    def simulate(rules: RuleArrays, pop: brain.Genome, key: jax.Array):
        def one_tick(carry, k):
            world, bstate, mod = carry
            world, bstate, mod, acts, obs = tick(rules, pop, world, bstate, mod, k, no_override)
            rec = None
            if record:
                rec = dict(grid=world.grid_obj.astype(jnp.int16), pos=world.pos.astype(jnp.int16),
                           dir=world.dir.astype(jnp.int8), alive=world.alive, held=world.held.astype(jnp.int16),
                           food=world.food, pain=world.pain, action=acts.astype(jnp.int8), mod=mod,
                           x=bstate.x.astype(jnp.float16))
            return (world, bstate, mod), rec

        def chunk(carry, keys):
            carry, recs = jax.lax.scan(one_tick, carry, keys)
            snap = carry[1].w.astype(jnp.float16) if record else None
            return carry, (recs, snap)

        kw, ks = jax.random.split(key)
        world = init_world(exp.world, rules, kw)
        bstate = jax.vmap(brain.init_state)(pop)
        mod = jnp.ones(N, jnp.float32)
        keys = jax.random.split(ks, T).reshape(T // every, every, -1)
        (world, bstate, mod), (recs, snaps) = jax.lax.scan(chunk, (world, bstate, mod), keys)
        stats = dict(alive_ticks=world.alive_ticks, eaten=world.eaten, pain=world.pain_total, alive=world.alive,
                     food=world.food)
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


def load_population(run_dir: Path | str) -> brain.Genome:
    """The final population of an earlier run (ADR-012). Its brain layout must match the new experiment."""
    with np.load(Path(run_dir) / "population.npz") as f:
        return brain.Genome(**{k: jnp.asarray(f[k]) for k in brain.Genome._fields})


def save_recording(out_dir: Path, exp: ExperimentConfig, ruleset: Ruleset, layout: brain.Layout, pop: brain.Genome,
                   recs: dict, best: int, rows: list[dict] | None = None, dashboard: bool = True):
    """Write recording.npz (+ genome arrays needed to interpret it) and the interactive dashboard."""
    np.savez_compressed(out_dir / "recording.npz", best=best, w0=np.asarray(pop.w0, np.float16),
                        eta=np.asarray(pop.eta, np.float16), mask=np.asarray(pop.mask, np.uint8),
                        **{k: np.asarray(v) for k, v in recs.items()})
    (out_dir / "layout.json").write_text(json.dumps(layout.to_json_dict()))
    if dashboard:
        from .dashboard import export_dashboard
        export_dashboard(out_dir)


def run_evolution(exp: ExperimentConfig, ruleset: Ruleset, fitness_fn: Callable[[dict], jnp.ndarray],
                  rules_for_generation: Callable[[int, jax.Array], RuleArrays] | None = None,
                  init_population: brain.Genome | None = None,
                  out_dir: Path | None = None, verbose: bool = True, dashboard: bool = True) -> dict:
    """Run the full experiment. `rules_for_generation(gen, key)` may return different RuleArrays per generation
    (same shapes) to randomize the environment, or RuleArrays stacked along a leading axis of size
    `evolution.episodes` to vary it per episode; default uses ruleset.to_arrays throughout.
    `init_population` (from load_population) warm-starts from an earlier run instead of random genomes."""
    ecfg = exp.evolution
    layout = make_layout(exp)
    key = jax.random.PRNGKey(ecfg.seed)
    key, kinit = jax.random.split(key)
    if init_population is None:
        pop = jax.vmap(lambda k: brain.init_genome(k, layout))(jax.random.split(kinit, exp.world.num_agents))
    else:
        pop = init_population
        assert pop.b.shape == (exp.world.num_agents, layout.n), \
            f"init population has shape {pop.b.shape}, experiment needs {(exp.world.num_agents, layout.n)}"
    base_rules = ruleset.to_arrays(exp.vision.appearance_dim)
    rules_fn = rules_for_generation or (lambda gen, k: base_rules)
    E = ecfg.episodes
    sim = jax.jit(jax.vmap(make_simulate(exp, record=False), in_axes=(0, None, 0)))
    sim_rec = make_simulate(exp, record=True)

    def per_episode(rules: RuleArrays) -> RuleArrays:
        if rules.food_value.ndim == 1:
            return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (E,) + x.shape), rules)
        assert rules.food_value.shape[0] == E, "rules_for_generation must return E stacked rulesets or one"
        return rules

    out_dir = out_dir or RUNS_DIR / exp.name / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(exp.to_json())
    (out_dir / "ruleset.json").write_text(json.dumps(ruleset.to_json_dict()))
    rows = []
    t0 = time.time()
    stats = None
    for gen in range(ecfg.generations):
        key, kr, ks, ke = jax.random.split(key, 4)
        rules = per_episode(rules_fn(gen, kr))
        last = gen == ecfg.generations - 1
        ep_keys = jax.random.split(ks, E)
        stats, _ = sim(rules, pop, ep_keys)
        stats = jax.tree_util.tree_map(lambda x: x.astype(jnp.float32).mean(axis=0), stats)
        fit = fitness_fn(stats)
        row = dict(gen=gen, fit_mean=float(fit.mean()), fit_max=float(fit.max()),
                   alive_ticks=float(stats["alive_ticks"].mean()), eaten=float(stats["eaten"].mean()),
                   pain=float(stats["pain"].mean()), survivors=float(stats["alive"].sum()),
                   seconds=round(time.time() - t0, 1))
        rows.append(row)
        if verbose:
            print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
        if last:
            best = int(jnp.argmax(fit))
            rules0 = jax.tree_util.tree_map(lambda x: x[0], rules)
            _, recs = sim_rec(rules0, pop, ep_keys[0])   # replay episode 0 with recording on
            np.savez(out_dir / "best_genome.npz", **{k: np.asarray(v[best]) for k, v in pop._asdict().items()})
            np.savez_compressed(out_dir / "population.npz", **{k: np.asarray(v) for k, v in pop._asdict().items()})
        else:
            pop = next_generation(ke, pop, fit, ecfg, layout)
    with open(out_dir / "fitness.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    save_recording(out_dir, exp, ruleset, layout, pop, recs, best, rows, dashboard=dashboard)
    if verbose:
        print(f"wrote {out_dir}" + ("  (open dashboard.html in a browser)" if dashboard else ""))
    return dict(out_dir=out_dir, rows=rows, pop=pop, stats=stats)
