"""Does a population do better when there are fewer foragers around? The newest main and control populations of a
stage, each alone in the worlds of the head-to-head test (`stages versus`), at full and at half the number of
animals. Reads a head-to-head result: if the control half does better next to main animals than among its own,
is that because main animals leave food (then the control alone at half density does as well)?

    uv run python scripts/probes/density.py <stage key>     WORLDS=<n> (default 8)"""
import os, sys
from dataclasses import replace
import jax, numpy as np
from life.config import ExperimentConfig
from life.experiments import stages as S
from life.run import load_population, make_simulate

WORLDS = int(os.environ.get("WORLDS", 8))
s = S.STAGES[sys.argv[1]]
fit_fn = s.fitness or S.default_fitness
for main_run, ctrl_run in zip(S.seed_runs(s), S.seed_runs(s, control=True)):
    out = []
    for label, run in (("main", main_run), ("control", ctrl_run)):
        exp = ExperimentConfig.from_json((run / "config.json").read_text())
        pop = load_population(run)
        n = pop.b.shape[0]
        for share in (1.0, 0.5):
            k = int(n * share)
            e = replace(exp, world=replace(exp.world, num_agents=int(exp.world.num_agents * share)))
            g = jax.tree_util.tree_map(lambda a: a[:k], pop)
            rs, rules_fn = s.build(e)
            sim = jax.jit(make_simulate(e, record=False))
            acc = []
            for key in jax.random.split(jax.random.PRNGKey(321), WORLDS):
                kr, ks = jax.random.split(key)
                rules = rules_fn(0, kr)
                if rules.food_value.ndim > 1 + (e.world.switch_tick > 0):
                    rules = jax.tree_util.tree_map(lambda a: a[0], rules)
                acc.append(float(fit_fn(sim(rules, g, ks)[0]).mean()))
            out.append(f"{label} x{share:g}: {np.mean(acc):.0f}")
    print(f"{main_run.parent.name}: " + ", ".join(out), flush=True)
