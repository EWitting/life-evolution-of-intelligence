"""Does the life cap hide differences? The newest population of each run, in its own world, at several life lengths:
how many agents are still alive at the cap, and the fitness (well-fed lifetime from the first meal).

    uv run python scripts/probes/ceiling.py <stage key>[:control] [...]   TICKS=1000,2000 in the environment
"""
import os, sys
from dataclasses import replace
import jax, numpy as np
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate

TICKS = [int(x) for x in os.environ.get("TICKS", "1000,2000").split(",")]
WORLDS = 3
for arg in sys.argv[1:]:
    key, _, kind = arg.partition(":")
    s = S.STAGES[key]
    brain = S.STAGES[s.parent].brain if kind == "control" else s.brain
    name = s.name + ("_control" if kind == "control" else "")
    for ticks in TICKS:
        exp = S.make_exp(replace(s, ticks=ticks), brain, "ceiling", 1, 0)
        pop = load_population(latest_run(name), exp, seed=0)
        rs, fn = s.build(exp)
        sim = make_simulate(exp, record=False)
        acc = []
        for k in jax.random.split(jax.random.PRNGKey(11), WORLDS):
            kr, ks = jax.random.split(k)
            rules = fn(0, kr)
            if rules.food_value.ndim > 1 + (exp.world.switch_tick > 0):
                rules = jax.tree_util.tree_map(lambda a: a[0], rules)
            st, _ = sim(rules, pop, ks)
            acc.append([float(st["fed_meal"].mean()), float(st["alive_ticks"].mean()), float(st["alive"].mean()),
                        float(st["eaten"].mean())])
        m = np.mean(acc, 0)
        print(f"{arg:14s} life cap {ticks:5d}: well-fed lifetime {m[0]:5.0f}  lifetime {m[1]:5.0f}  alive at the cap {m[2]:4.0%}  "
              f"energy {m[3]:5.1f}", flush=True)
