"""No-cliff calibration (ADR-012): the parent's final population, with the parent's brain, in the stage's world at
several bush densities, against the same population in its own world.
    calib.py <stage key> <density> [<density> ...] [field=value ...]   (extra world overrides)"""
import sys
from dataclasses import replace
import jax, numpy as np
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate

key = sys.argv[1]
dens = [float(a) for a in sys.argv[2:] if "=" not in a]
over = {a.split("=")[0]: float(a.split("=")[1]) for a in sys.argv[2:] if "=" in a and not a[0].isupper()}
consts = {a.split("=")[0]: float(a.split("=")[1]) for a in sys.argv[2:] if "=" in a and a[0].isupper()}
for k_, v_ in consts.items():
    setattr(S, k_, v_)
s = S.STAGES[key]; parent = S.STAGES[s.parent]
run = latest_run(parent.name)


def evaluate(stage, world, label, worlds=4):
    st = replace(stage, world=world)
    exp = S.make_exp(st, parent.brain, "calib", 1, 0)
    pop = load_population(run, exp, seed=0)
    rs, fn = st.build(exp)
    sim = make_simulate(exp, record=False)
    acc = []
    for k in jax.random.split(jax.random.PRNGKey(11), worlds):
        kr, ks = jax.random.split(k)
        rules = fn(0, kr)
        if rules.food_value.ndim > 1 + (exp.world.switch_tick > 0):
            rules = jax.tree_util.tree_map(lambda a: a[0], rules)
        stt, _ = sim(rules, pop, ks)
        acc.append([float(stt[c].mean()) for c in ("fed_meal", "alive_ticks", "eaten")] + [float((stt["alive"]).sum())])
    m = np.mean(acc, 0)
    print(f"{label:46s} fitness {m[0]:5.0f}  lifetime {m[1]:5.0f} of {exp.evolution.ticks_per_generation}  eaten {m[2]:5.1f}  survivors {m[3]:4.1f}", flush=True)


evaluate(parent, parent.world, f"parent {s.parent} population in its own world")
for d in dens:
    w = replace(s.world, spawn_density=d, **{k: type(getattr(s.world, k))(v) for k, v in over.items()})
    evaluate(s, w, f"in the {key} world, density {d}" + "".join(f" {k}={v}" for k, v in {**over, **consts}.items()))
