"""Is fitness limited by the food supply or by skill? A stage's final population in worlds of several sizes (same
bush density, so more food per agent in a larger world): energy acquired, lifetime, and the share of the variance
between lives that is due to the genome (intraclass correlation over siblings; higher = selection sees more).
    uv run python scripts/probes/world_size.py <stage key> <size> [<size> ...]"""
import os, sys
from dataclasses import replace
import jax, numpy as np
from life.config import ExperimentConfig
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate

s = S.STAGES[sys.argv[1]]
run = latest_run(s.name)
exp0 = ExperimentConfig.from_json((run / "config.json").read_text())
pop = load_population(run)
P = pop.b.shape[0]; N = exp0.world.num_agents; k = N // P
rank = lambda v: np.argsort(np.argsort(v)).astype(np.float64)
for size in [int(a) for a in sys.argv[2:]]:
    dens = float(os.environ.get("DENSITY", exp0.world.spawn_density))
    exp = replace(exp0, world=replace(exp0.world, height=size, width=size, spawn_density=dens))
    rs, fn = s.build(exp)
    sim = make_simulate(exp, record=False)
    icc, eaten, alive = [], [], []
    for w in range(6):
        kr, ks = jax.random.split(jax.random.PRNGKey(300 + w))
        rules = fn(0, kr)
        if rules.food_value.ndim > 1 + (exp.world.switch_tick > 0):
            rules = jax.tree_util.tree_map(lambda a: a[0], rules)
        st = sim(rules, pop, ks)[0]
        e = np.asarray(st["eaten"]); g = rank(e).reshape(P, k)
        msb = k * g.mean(1).var(ddof=1); msw = g.var(axis=1, ddof=1).mean()
        icc.append((msb - msw) / (msb + (k - 1) * msw)); eaten.append(e.mean()); alive.append(float(st["alive_ticks"].mean()))
    print(f"{size} x {size}, bush density {dens}: energy {np.mean(eaten):6.1f}  lifetime {np.mean(alive):5.0f}  genome share of variance {np.mean(icc):.3f} +-{np.std(icc, ddof=1) / np.sqrt(6):.3f}", flush=True)
