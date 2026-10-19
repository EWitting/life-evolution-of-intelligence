"""Inside one recorded life of the critic (variant td / tdp): what the dopamine signal does in each situation and
what the aversive cells make of poison and good bushes over the life. One world of 64 animals (the stage's world
scaled down, as record.py), generation 0 of the parent population.
    STAGE=x.critic DENSE=1 python scripts/probes/da_trace.py <variant> [NAME=value ...]
Situations per animal-tick: eating a good / a poison berry, a poison or good bush straight ahead, a poison or good
bush that was ahead a tick earlier and is not now (the animal turned away or emptied it), anything else. Printed per
situation: share of ticks, mean dopamine, and the summed positive and negative dopamine per animal and life."""
import os, sys
from dataclasses import replace
import jax, numpy as np
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate, make_layout
from life.world import DIRS
from variants import variant

V = sys.argv[1]
for a in sys.argv[2:]:
    k_, v_ = a.split("=")
    setattr(S, k_, float(v_))
s = S.STAGES[os.environ.get("STAGE", "x.critic")]
NA = 64
f = NA / s.world.num_agents
world = replace(s.world, num_agents=NA, height=round(s.world.height * f ** 0.5), width=round(s.world.width * f ** 0.5),
                patches=round(s.world.patches * f))
exp = S.make_exp(replace(s, world=world), variant(s.brain, V), "da_trace", 1, 0)
exp = replace(exp, evolution=replace(exp.evolution, siblings=1))
pop = load_population(os.environ.get("RUN") or latest_run(S.STAGES[s.parent].name), exp, seed=0)
L = make_layout(exp)
if os.environ.get("DENSE"):
    pl = (np.asarray(L.rule) > 0) & (np.asarray(L.allowed) > 0)
    pop = pop._replace(mask=jax.numpy.where(pl[None], 1.0, pop.mask))
pop = jax.tree_util.tree_map(lambda a: a[:NA], pop)
rs, fn = s.build(exp)
kr, ks = jax.random.split(jax.random.PRNGKey(31))
rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
st, rec = make_simulate(exp, record=True)(rules, pop, ks)
rec = {k: np.asarray(v) for k, v in rec.items() if k != "w_snap"}

# what is straight ahead: poison bush (its berry hurts), good bush (with berries), other
pv, fv = np.asarray(rules.pain_value), np.asarray(rules.food_value)
bush = {rs.local(S.variant(t)[S.BUSH]): rs.local(S.variant(t)[S.BERRY]) for t in range(6)}
cls = np.zeros(pv.shape[0], np.int8)                   # 0 other, 1 good bush, 2 poison bush
for b, berry in bush.items():
    cls[b] = 2 if pv[berry] > 0 else (1 if fv[berry] > 0 else 0)
T = rec["pos"].shape[0]
front = rec["pos"].astype(int) + np.asarray(DIRS)[rec["dir"].astype(int)]
fy = np.clip(front[..., 0], 0, world.height - 1); fx = np.clip(front[..., 1], 0, world.width - 1)
ahead = cls[rec["grid"][np.arange(T)[:, None], fy, fx]]
ate = rec["ate"].astype(int)
alive = rec["alive"]
MOD = os.environ.get("MOD") or next(m for m in ("da", "da_av", "us_av") if m in L.mod_names)   # MOD: the signal to show
da = rec["mod"][..., list(L.mod_names).index(MOD)]
mean = lambda r: rec["x"][..., L.offsets[L.names.index(r)]:L.offsets[L.names.index(r)] + L.sizes[L.names.index(r)]].astype(np.float32).mean(-1)
vav = mean("valence_av")
prev = np.concatenate([np.zeros_like(ahead[:1]), ahead[:-1]])
sit = {"eat good": (ate > 0) & (pv[ate] == 0), "eat poison": (ate > 0) & (pv[ate] > 0),
       "poison ahead": ahead == 2, "good ahead": ahead == 1,
       "poison gone": (prev == 2) & (ahead != 2), "good gone": (prev == 1) & (ahead != 1)}
sit["other"] = ~np.any(np.stack(list(sit.values())), 0)
n_alive = alive.sum()
print(f"{V}: lifetime {alive.sum(0).mean():.0f}, dopamine mean {da[alive].mean():+.3f}")
for name, m in sit.items():
    m = m & alive
    d = da[m]
    print(f"  {name:13s} {m.sum() / n_alive:6.1%} of ticks  dopamine {d.mean() if d.size else 0:+.3f}   "
          f"per life +{d[d > 0].sum() / NA:6.1f} {d[d < 0].sum() / NA:6.1f}")
q = np.array_split(np.arange(T), 4)
print("  aversive cells with a poison / good bush ahead, by quarter of life:",
      "  ".join(f"{vav[i][(ahead[i] == 2) & alive[i]].mean():.2f} / {vav[i][(ahead[i] == 1) & alive[i]].mean():.2f}" for i in q))
for r in ("value_av", "value_app"):
    if r in L.names:
        v = mean(r)
        print(f"  {r} with a poison / good bush ahead, by quarter:",
              "  ".join(f"{v[i][(ahead[i] == 2) & alive[i]].mean():.2f} / {v[i][(ahead[i] == 1) & alive[i]].mean():.2f}" for i in q))
