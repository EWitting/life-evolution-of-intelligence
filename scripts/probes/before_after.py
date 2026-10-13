"""The newest run of a trial stage's parent against the newest run of the trial, in the trial's world, the same worlds for both.
    uv run python before_after.py <trial key> [worlds]     also checks the vmapped simulation of run_evolution"""
import sys
import jax, numpy as np
from life.experiments import stages as S
from life.freeze import load_frozen
from life.run import load_population, latest_run, make_simulate

s = S.STAGES[sys.argv[1]]
W = int(sys.argv[2]) if len(sys.argv) > 2 else 16
exp = S.make_exp(s, s.brain, "before_after", 1, 0)
rs, fn = s.build(exp)
sim = make_simulate(exp, record=False)
extra = S.poison_metrics(rs)
pops = {"before": load_population(latest_run(S.STAGES[s.parent].name), exp, seed=0), "after": load_population(latest_run(s.name), exp, seed=0)}
res = {k: [] for k in pops}
for n, k in enumerate(jax.random.split(jax.random.PRNGKey(5), W)):
    kr, ks = jax.random.split(k)
    rules1 = fn(0, kr)
    rules = jax.tree_util.tree_map(lambda a: a[0], rules1)
    for label, pop in pops.items():
        st, _ = sim(rules, pop, ks)
        e = extra(st, None)
        res[label].append([float(st["fed_meal"].mean()), float(st["alive_ticks"].mean()), float(e["berries"]), float(e["poison_frac"])])
    if n == 0:
        simv = jax.jit(jax.vmap(make_simulate(exp, record=False), in_axes=(0, None, 0)))
        sv, _ = simv(rules1, pops["after"], ks[None])
        print("vmapped simulation equal:", all(np.array_equal(np.asarray(sv[a][0]), np.asarray(st[a])) for a in st), flush=True)
a, b = np.array(res["before"]), np.array(res["after"])
for label, x in (("before", a), ("after", b)):
    m = x.mean(0)
    print(f"[{s.key}] {label:6s} well-fed lifetime {m[0]:5.0f} +-{x[:, 0].std(ddof=1) / np.sqrt(W):.0f}  lifetime {m[1]:5.0f}  berries {m[2]:5.1f}  poison share {m[3]:.3f}")
d = b[:, 0] - a[:, 0]
print(f"[{s.key}] after - before {d.mean():+.0f} +-{d.std(ddof=1) / np.sqrt(W):.0f}  (better in {int((d > 0).sum())} of {W} worlds)", flush=True)
