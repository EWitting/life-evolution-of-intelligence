"""How a life with hands is spent (stage 2.5, ADR-028), generation 0: the frozen 2.3s population on the stage's main
or control brain in one world, recorded for TICKS ticks. Per creature alive: picks, berries eaten, put down, share of
ticks holding something, how long a berry is held before it is eaten, and the actions taken while holding (by what
is ahead: nothing, a bush or other object).

    TICKS=1500 [RUN=runs/<stage run>/<time>] python scripts/probes/hands_life.py [main|control]"""
import os
import sys
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from life import actions as A, freeze
from life.experiments import stages as S
from life.run import make_layout, make_simulate

which = sys.argv[1] if len(sys.argv) > 1 else "main"
s = S.STAGES["2.5"]
s = replace(s, ticks=int(os.environ.get("TICKS", 1500)))
import os as _os
from dataclasses import replace as _replace
from life.experiments.chapter2 import hands as _hands, actor as _actor, B24S as _B24S, ACT_EXPLORE as _AE
if _os.environ.get("SWALLOW") or _os.environ.get("EXPLORE"):
    _sw = float(_os.environ["SWALLOW"]) if _os.environ.get("SWALLOW") else None
    _h = _hands(_B24S, swallow=_sw)
    s = _replace(s, brain=_actor(_h, explore=float(_os.environ.get("EXPLORE", _AE))), control_brain=_h)
    print("SWALLOW", _sw, "EXPLORE", _os.environ.get("EXPLORE", _AE), flush=True)
exp = S.make_exp(s, s.brain if which == "main" else s.control_brain, "life", 1, 0)
L = make_layout(exp)
if os.environ.get("RUN"):     # a run's checkpoint instead of the stage's start
    from life.run import load_population
    pop = load_population(os.environ["RUN"], exp, file="checkpoint.npz")
else:
    pop = s.adapt_init(freeze.load_frozen("2.3s", 0, exp), L)
if which == "main":
    pop = pop._replace(mask=jnp.where(((np.asarray(L.rule) > 0) & (np.asarray(L.allowed) > 0))[None], 1.0, pop.mask))
rs, fn = s.build(exp)
kr, ks = jax.random.split(jax.random.PRNGKey(31))
rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
st, rec = jax.jit(make_simulate(exp, record=True))(rules, pop, ks)
held, act, ate, alive = (np.asarray(rec[k]) for k in ("held", "action", "ate", "alive"))   # [T, N]
grid, pos, dr = np.asarray(rec["grid"]), np.asarray(rec["pos"]), np.asarray(rec["dir"])
T, N = held.shape
from life.world import DIRS
front = pos + np.asarray(DIRS)[dr.astype(int)]                                          # [T, N, 2]
H, W = grid.shape[1:]
inb = (front[..., 0] >= 0) & (front[..., 0] < H) & (front[..., 1] >= 0) & (front[..., 1] < W)
fc = np.clip(front, 0, [H - 1, W - 1])
ahead = np.where(inb, grid[np.arange(T)[:, None], fc[..., 0], fc[..., 1]], -1)
prev = np.vstack([np.zeros((1, N), held.dtype), held[:-1]])
live = alive > 0
picks = ((prev == 0) & (held > 0) & live).sum()
eaten = ((ate > 0) & live).sum()
drops = ((prev > 0) & (held == 0) & (ate == 0) & live).sum()
n_alive = live.any(0).sum()
holding = (held > 0) & live
print(f"{which}: {n_alive} creatures, {T} ticks; per creature: picks {picks / n_alive:.1f}, eaten {eaten / n_alive:.1f}, "
      f"put down {drops / n_alive:.1f}; share of living ticks holding {holding.sum() / live.sum():.2f}")
# how long a berry is held before it is eaten (completed holds only)
lens = []
for a in range(N):
    t0 = None
    for t in range(T):
        if holding[t, a] and t0 is None:
            t0 = t
        if t0 is not None and not holding[t, a]:
            lens.append((t - t0, ate[t, a] > 0 or (t > 0 and ate[t - 1, a] > 0)))
            t0 = None
lens = np.array(lens) if lens else np.zeros((0, 2))
if len(lens):
    print(f"completed holds {len(lens)}: median {np.median(lens[:, 0]):.0f} ticks, 90% {np.percentile(lens[:, 0], 90):.0f}")
for name, m in (("nothing ahead", holding & (ahead == 0)), ("something ahead", holding & (ahead > 0))):
    c = np.bincount(act[m].astype(int), minlength=A.NUM_ACTIONS)
    print(f"holding, {name} ({m.sum()} ticks): " + "  ".join(f"{A.NAMES[i]} {c[i] / max(m.sum(), 1):.2f}" for i in range(6)))
c = np.bincount(act[live & ~holding].astype(int), minlength=A.NUM_ACTIONS)
print("empty hand: " + "  ".join(f"{A.NAMES[i]} {c[i] / max((live & ~holding).sum(), 1):.2f}" for i in range(6)))
x = np.asarray(rec["x"], np.float32)                                                   # [T, N, cells]
for name, m in (("holding, nothing ahead", holding & (ahead == 0)), ("holding, something ahead", holding & (ahead > 0))):
    print(name + ": " + "  ".join(f"{r} {x[m][:, L.region(r)].mean():.2f}" for r in
                                  ("ingest", "no_feed", "no_hold", "grasp", "valence_av", "valence_app") + (("act", "gpi") if "act" in L.names else ())))
from life import brain as B
Wf = np.asarray(jax.vmap(lambda g: B.effective(L, g.w0 * g.mask))(pop))                 # [genomes, N, N] at birth
sib = exp.evolution.siblings
gidx = np.arange(N) // sib
out = L.region("out")
m = holding & (ahead > 0)
tt, aa = np.nonzero(m)
pick = np.random.default_rng(0).choice(len(tt), min(3000, len(tt)), replace=False)
tt, aa = tt[pick], aa[pick]
print("drive onto motor neurons while holding with something ahead (mean over ticks), by source:")
for a_name, j in (("FORWARD", 1), ("USE", 4), ("EAT", 5)):
    contrib = {}
    for r in L.names:
        if r == "out":
            continue
        sl = L.region(r)
        contrib[r] = np.mean([x[t, a, sl] @ Wf[gidx[a], sl, out.start + j] for t, a in zip(tt, aa)])
    top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:6]
    bias = np.mean(np.asarray(pop.b)[gidx[aa], out.start + j])
    print(f"  {a_name:8s} total {sum(contrib.values()) + bias:+.2f} (bias {bias:+.2f}): " + "  ".join(f"{k} {v:+.2f}" for k, v in top))
