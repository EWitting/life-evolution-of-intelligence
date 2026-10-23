"""Running stages and measuring them: make_exp, run_stage, resume_stage, lesion(s), versus, respond."""
from __future__ import annotations
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from life.config import ExperimentConfig, BrainConfig, EvolutionConfig
from life.run import run_evolution, load_population, latest_run, RUNS_DIR
from life.experiments.base import Stage, STAGES, default_fitness


# ------------------------------------------------------------------ running

EVOLUTION_OVERRIDES: dict = {}   # set from the command line (--mutation-prob), applied to every stage
SIBLINGS = 4                     # 256 agents = 64 genomes x 4 siblings (EvolutionConfig.siblings)
CROSSOVER = 1.0                  # every child has two parents (whole neurons from either): in the recovery
                                 # benchmark this repaired a damaged population 6-7% further than asexual
                                 # reproduction in 40 generations, in every seed (STAGE_LOG, v20)


def make_exp(s: Stage, brain: BrainConfig, name: str, generations: int, seed: int) -> ExperimentConfig:
    return ExperimentConfig(name=name, world=s.world, vision=s.vision, body=s.body, brain=brain,
                            evolution=EvolutionConfig(generations=generations, ticks_per_generation=s.ticks,
                                                      eta_max=s.eta_max, **{"siblings": SIBLINGS, "crossover": CROSSOVER, **dict(s.evolution), **EVOLUTION_OVERRIDES},
                                                      plastic=s.plastic, seed=seed))


def run_stage(key: str, generations: int | None = None, control: bool = False, init_from: str | None = "auto",
              seed: int = 0, dashboard: bool = True, suffix: str = "", ticks: int | None = None, adapt: bool = True,
              resume: str | None = None):
    """`resume`: a run of this stage stopped early; it goes on from its checkpoint.npz in its own directory."""
    if resume:
        return resume_stage(key, resume, control=control, seed=seed, suffix=suffix, ticks=ticks, dashboard=dashboard)
    s = STAGES[key]
    if ticks:   # a trial with a longer or shorter life than the stage defines; use a suffix to keep it apart
        s = replace(s, ticks=ticks)
    gens = generations or s.generations
    if control:
        assert s.parent, "stage has no parent, so no control"
        brain, name = s.control_brain or STAGES[s.parent].brain, s.name + "_control"
    else:
        brain, name = s.brain, s.name
    name += suffix
    if init_from == "none" and s.parent:
        name += "_scratch"          # not part of the lineage (latest_run of the stage must stay the lineage run)
    exp = make_exp(s, brain, name, gens, seed)
    init = None
    if init_from == "auto" and s.parent:
        init_from = str(latest_run(STAGES[s.parent].name))
    if init_from and init_from not in ("auto", "none"):
        print(f"warm start from {init_from}")
        init = load_population(init_from, exp, seed=seed)
        from life.run import make_layout
        lay = make_layout(exp)
        if s.adapt_init and adapt:   # not when a run of this stage is continued: the change is already in it
            init = s.adapt_init(init, lay)
        if s.dense_plastic and not control:
            init = init._replace(mask=jnp.where(((np.asarray(lay.rule) > 0) & (np.asarray(lay.allowed) > 0))[None], 1.0, init.mask))
    rs, rules_fn = s.build(exp)
    if s.schedule:
        rules_fn = s.schedule(rules_fn)
    res = run_evolution(exp, rs, s.fitness or default_fitness, rules_for_generation=rules_fn, init_population=init,
                        row_extra=s.row_extra(rs) if s.row_extra else None, dashboard=dashboard)
    if init is not None:   # a run started from an earlier run of its own experiment continues it (book: one curve)
        from pathlib import Path
        src = Path(init_from).resolve()
        if src.parent == Path(res["out_dir"]).resolve().parent:
            (Path(res["out_dir"]) / "continues.txt").write_text(src.name)
    return res


def resume_stage(key: str, run_dir: str, control: bool = False, seed: int = 0, suffix: str = "",
                 ticks: int | None = None, dashboard: bool = False):
    """Continue a run stopped early (a kill) from its checkpoint: the population evaluated in the checkpoint's
    generation, the earlier rows of its fitness.csv, the same random keys and schedule from there on."""
    import csv
    from pathlib import Path
    run_dir = Path(run_dir)
    s = STAGES[key]
    if ticks:
        s = replace(s, ticks=ticks)
    brain = (s.control_brain or STAGES[s.parent].brain) if control else s.brain
    name = s.name + ("_control" if control else "") + suffix
    old = ExperimentConfig.from_json((run_dir / "config.json").read_text())
    exp = make_exp(s, brain, name, old.evolution.generations, seed)
    assert exp.to_json() == old.to_json(), "the stage's settings differ from the stopped run's config.json"
    with np.load(run_dir / "checkpoint.npz") as f:
        start = int(f["generation"])
    init = load_population(run_dir, file="checkpoint.npz")
    with open(run_dir / "fitness.csv", newline="") as f:
        prior = [{k: (int(v) if k == "gen" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    prior = [r for r in prior if r["gen"] < start]
    assert len(prior) == start, f"fitness.csv has {len(prior)} rows before generation {start}"
    print(f"resume {run_dir} at generation {start}")
    rs, rules_fn = s.build(exp)
    if s.schedule:
        rules_fn = s.schedule(rules_fn)
    return run_evolution(exp, rs, s.fitness or default_fitness, rules_for_generation=rules_fn, init_population=init,
                         out_dir=run_dir, row_extra=s.row_extra(rs) if s.row_extra else None, dashboard=dashboard,
                         start_generation=start, prior_rows=prior)


def stage_of_run(run_dir) -> Stage:
    from pathlib import Path
    name = Path(run_dir).resolve().parent.name
    found = [s for s in STAGES.values() if name.startswith(s.name)]   # the longest name: x_ch2_256_p70, not x_ch2_256
    if found:
        return max(found, key=lambda s: len(s.name))
    raise KeyError(f"no stage for experiment {name}")


def lesion(run_dir: str, regions: list[str] | None = None, worlds: int = 8, seed: int = 123,
           only: tuple | None = None, per_world: bool = False) -> dict:
    """Lesion study: the run's final population in its own world, intact and with each region silenced (all
    outgoing synapses removed). Averages over `worlds` independent worlds. Returns {label: stats means}.
    The worlds depend only on `seed`, so two runs of the same stage are measured in the same worlds.
    `only`: restrict to these labels (e.g. ("intact",) to re-evaluate a population). A region named
    "learn_off:<region>" switches off only the learning of the synapses onto that region."""
    from pathlib import Path
    from life.config import ExperimentConfig
    from life.run import make_layout, make_simulate
    run_dir = Path(run_dir)
    exp = ExperimentConfig.from_json((run_dir / "config.json").read_text())
    s = stage_of_run(run_dir)
    layout = make_layout(exp)
    pop = load_population(run_dir)
    rs, rules_fn = s.build(exp)
    extra = s.row_extra(rs) if s.row_extra else (lambda st, f: {})
    sim = jax.jit(make_simulate(exp, record=False))
    nodep = replace(exp, brain=replace(exp.brain, projections=tuple(replace(p, depression=()) for p in exp.brain.projections)))
    sim_nodep = jax.jit(make_simulate(nodep, record=False)) if layout.has_dep else None
    fit_fn = s.fitness or default_fitness
    regions = regions or [n for n in layout.names if n not in ("in", "out")]
    keys = jax.random.split(jax.random.PRNGKey(seed), worlds)
    out = {}
    labels = ["intact"] + (["no_plasticity"] if float(pop.eta.max()) > 0 else []) \
        + (["no_depression"] if layout.has_dep else []) + regions
    labels += [l for l in (only or ()) if l.startswith("learn_off:")]
    if only:
        labels = [l for l in labels if l in only]
    for label in labels:
        g = pop
        if label == "no_plasticity":
            g = pop._replace(eta=jnp.zeros_like(pop.eta))
        elif label.startswith("learn_off:"):   # only the learning of the synapses onto one region off
            g = pop._replace(eta=pop.eta.at[:, :, layout.region(label[10:])].set(0.0))
        elif label not in ("intact", "no_depression"):
            rows = layout.region(label)
            # silence the region: no outgoing synapses and no activity (so broadcast modulators read 0 too)
            g = pop._replace(w0=pop.w0.at[:, rows, :].set(0.0), mask=pop.mask.at[:, rows, :].set(0.0),
                             b=pop.b.at[:, rows].set(-10.0))
        acc = []
        for k in keys:
            kr, ks = jax.random.split(k)
            rules = rules_fn(0, kr)
            if rules.food_value.ndim > 1 + (exp.world.switch_tick > 0):
                rules = jax.tree_util.tree_map(lambda a: a[0], rules)
            st, _ = (sim_nodep if label == "no_depression" else sim)(rules, g, ks)
            fit = fit_fn(st)
            row = dict(fitness=float(fit.mean()), eaten=float(st["eaten"].mean()), pain=float(st["pain"].mean()),
                       alive=float(st["alive_ticks"].mean()), temp=float(st["temp_mean"].mean()))
            row.update({k2: float(v) for k2, v in extra(st, fit).items()})
            acc.append(row)
        out[label] = {k2: float(np.mean([r[k2] for r in acc])) for k2 in acc[0]}
        if per_world:   # the fitness in every world, for a standard error over worlds
            out[label]["fitness_worlds"] = [r["fitness"] for r in acc]
    return out


def seed_runs(s: Stage, control: bool = False) -> list:
    """Newest finished run of the stage's lineage directory and of every `_seedN` variant."""
    base = s.name + ("_control" if control else "")
    dirs = []
    for d in sorted(RUNS_DIR.iterdir()):
        rest = d.name[len(base):]
        if d.name.startswith(base) and (rest == "" or rest.startswith("_seed")):
            runs = sorted(r for r in d.iterdir() if (r / "population.npz").exists())
            if runs:
                dirs.append(runs[-1])
    return dirs


def lesions(key: str, worlds: int = 8) -> dict:
    """Lesion study over every seed of a stage: fitness with each region silenced, in % of the intact population.
    This is the test of whether a circuit is *used* (the stage criterion); one seed can mislead."""
    res = [lesion(str(r), worlds=worlds) for r in seed_runs(STAGES[key])]
    out = {}
    for label in res[0]:
        if label != "intact":
            out[label] = [100.0 * r[label]["fitness"] / r["intact"]["fitness"] for r in res]
    print(f"{key}: intact fitness per seed " + " ".join(f"{r['intact']['fitness']:.0f}" for r in res))
    print("lesion".ljust(18) + "% of intact per seed".rjust(26) + "mean".rjust(7))
    for label, rel in out.items():
        print(label.ljust(18) + " ".join(f"{v:7.0f}" for v in rel).rjust(26) + f"{np.mean(rel):7.0f}")
    return out


def versus(key: str, worlds: int = 8, seed: int = 321) -> list:
    """Head-to-head test: half of the stage's main population and half of its control population live in the same
    worlds, so both face the same depleted food supply. The control agents run the new brain with the regions
    their own brain lacks silenced and (if their own brain had none) plasticity off. Differences that live in the
    layout rather than in a region (depression, decay, a new receptor on an old region) are shared by both halves.
    Returns, per seed, mean fitness of the main half minus that of the control half."""
    from life.config import ExperimentConfig
    from life.run import make_layout, make_simulate
    s = STAGES[key]
    diffs = []
    for main_run, ctrl_run in zip(seed_runs(s), seed_runs(s, control=True)):
        exp = ExperimentConfig.from_json((main_run / "config.json").read_text())
        layout = make_layout(exp)
        old_layout = make_layout(ExperimentConfig.from_json((ctrl_run / "config.json").read_text()))
        main = load_population(main_run)
        own = load_population(ctrl_run)
        ctrl = load_population(ctrl_run, exp, seed=0)
        for r in [n for n in layout.names if n not in old_layout.names]:
            rows = layout.region(r)
            ctrl = ctrl._replace(w0=ctrl.w0.at[:, rows, :].set(0.0), mask=ctrl.mask.at[:, rows, :].set(0.0),
                                 b=ctrl.b.at[:, rows].set(-10.0))
        if float(own.eta.max()) == 0:
            ctrl = ctrl._replace(eta=jnp.zeros_like(ctrl.eta))
        n = main.b.shape[0]               # genomes; each lives as exp.evolution.siblings individuals
        half = n // 2
        sib = exp.world.num_agents // n
        mixed = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a[:half], b[:n - half]]), main, ctrl)
        rs, rules_fn = s.build(exp)
        sim = jax.jit(make_simulate(exp, record=False))
        fit_fn = s.fitness or default_fitness
        d = []
        for k in jax.random.split(jax.random.PRNGKey(seed), worlds):
            kr, ks = jax.random.split(k)
            rules = rules_fn(0, kr)
            if rules.food_value.ndim > 1 + (exp.world.switch_tick > 0):
                rules = jax.tree_util.tree_map(lambda a: a[0], rules)
            fit = fit_fn(sim(rules, mixed, ks)[0])
            d.append((float(fit[:half * sib].mean()), float(fit[half * sib:].mean())))
        m, c = np.mean(d, axis=0)
        diffs.append(m - c)
        print(f"{main_run.parent.name}: main half {m:.0f}, control half {c:.0f}, difference {m - c:+.0f}", flush=True)
    se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
    print(f"{key} head to head, main - control: {np.mean(diffs):+.1f} +-{se:.1f} (standard error over {len(diffs)} seeds)")
    return diffs


def respond(run_dir: str, regions: list[str] | None = None, steps: int = 4) -> list[dict]:
    """Stimulus-response readout: the run's final population (inherited weights) shown each object type straight
    ahead at distance 1, and held in hand, for `steps` brain steps with nothing else in view. Returns rows with
    the mean activity per region and the mean action probabilities (averaged over the population)."""
    from pathlib import Path
    from life.config import ExperimentConfig
    from life.run import make_layout
    from life import brain as B, actions as A
    run_dir = Path(run_dir)
    exp = ExperimentConfig.from_json((run_dir / "config.json").read_text())
    layout = make_layout(exp)
    pop = load_population(run_dir)
    rs, rules_fn = stage_of_run(run_dir).build(exp)
    rules = rules_fn(0, jax.random.PRNGKey(0))
    while rules.appearance.ndim > 2:
        rules = jax.tree_util.tree_map(lambda a: a[0], rules)
    app = np.asarray(rules.appearance)
    names = list(layout.in_names)
    regions = regions or [n for n in layout.names if n not in ("in", "out")]
    K = exp.vision.appearance_dim

    def obs_for(obj, held):
        o = np.zeros(layout.n_in, np.float32)
        o[names.index("food")] = 0.7
        if "temperature" in names:
            o[names.index("temperature")] = 0.5
        if held:
            o[names.index("held")] = 1.0
            for k in range(K):
                o[names.index(f"held_app{k}")] = app[obj, k]
        else:
            o[names.index("vis+0.hit")], o[names.index("vis+0.near")] = 1.0, 1.0 - 1.0 / exp.vision.range
            for k in range(K):
                o[names.index(f"vis+0.app{k}")] = app[obj, k]
        return jnp.asarray(o)

    def run_one(g, obs):
        st = B.init_state(g, layout)
        for _ in range(steps):
            st, _ = B.step(exp.brain, layout, g, st, obs, jnp.zeros(3), jax.random.PRNGKey(0))
        h = (st.x @ B.effective(layout, st.w) + g.b)[-layout.n_out:]
        p = jax.nn.softmax(h * exp.brain.logit_gain / max(exp.brain.action_temperature, 1e-3))
        return st.x, p

    g0 = pop._replace(eta=jnp.zeros_like(pop.eta))
    rows = []
    for obj in range(1, rs.size):
        for held in (False, True):
            if held and not rs.holdable[obj]:
                continue
            x, p = jax.vmap(lambda g: run_one(g, obs_for(obj, held)))(g0)
            row = {"object": rs.names[obj], "view": "held" if held else "ahead"}
            row.update({r: float(x[:, layout.region(r)].mean()) for r in regions})
            row.update({f"P({a})": float(p[:, i].mean()) for i, a in enumerate(A.NAMES[:6])})
            rows.append(row)
    return rows
