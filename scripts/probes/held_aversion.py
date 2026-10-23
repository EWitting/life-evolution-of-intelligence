"""Why the cells about what is seen fire while a berry is held (stage 2.5): a recorded life from a run's checkpoint,
holding ticks split by what is held (good / poison, by the phase of the reversal) and what is ahead (nothing, a bush
with a berry, an empty bush, something else). Per class: rates of the valence cells, no_feed, grasp, ingest, no_eat,
act, the share of USE / EAT, and the drive onto the aversive cells about what is seen (valence_av 0-1) by source,
with the plastic weights of the nearest snapshot.

    RUN=runs/s2_5_actor/<time> [TICKS=1500] [PHASE=1|2] python scripts/probes/held_aversion.py [main|control]"""
import os
import sys
from dataclasses import replace

import jax
import numpy as np

from life import actions as A
from life.experiments import stages as S
from life.run import load_population, make_layout, make_simulate
from life.world import DIRS

which = sys.argv[1] if len(sys.argv) > 1 else "main"
s = S.STAGES["2.5"]
s = replace(s, ticks=int(os.environ.get("TICKS", 1500)))
exp = S.make_exp(s, s.brain if which == "main" else s.control_brain, "life", 1, 0)
L = make_layout(exp)
pop = load_population(os.environ["RUN"], exp, file="checkpoint.npz")
rs, fn = s.build(exp)
kr, ks = jax.random.split(jax.random.PRNGKey(31))
rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
st, rec = jax.jit(make_simulate(exp, record=True))(rules, pop, ks)
held, act, alive = (np.asarray(rec[k]) for k in ("held", "action", "alive"))
grid, pos, dr = np.asarray(rec["grid"]), np.asarray(rec["pos"]), np.asarray(rec["dir"])
x = np.asarray(rec["x"], np.float32)
T, N = held.shape
front = pos + np.asarray(DIRS)[dr.astype(int)]
H, W = grid.shape[1:]
inb = (front[..., 0] >= 0) & (front[..., 0] < H) & (front[..., 1] >= 0) & (front[..., 1] < W)
fc = np.clip(front, 0, [H - 1, W - 1])
ahead = np.where(inb, grid[np.arange(T)[:, None], fc[..., 0], fc[..., 1]], -1)
# the brain at tick t saw the state after tick t-1: what was held then and what was ahead then
seen_held = np.vstack([np.zeros((1, N), held.dtype), held[:-1]])
seen_ahead = np.vstack([np.full((1, N), -1), ahead[:-1]])
pv = np.asarray(rules.pain_value)                                   # [2 phases, M], or [M] without a reversal
pv = pv if pv.ndim == 2 else np.stack([pv, pv])
phase = (np.arange(T) >= exp.world.switch_tick).astype(int)[:, None]
bad = pv[phase, seen_held] > 0
bushes = {rs.local(S.variant(v)[S.BUSH]) for v in range(6)}
empties = {rs.local(S.variant(v)[S.EMPTY_BUSH]) for v in range(6)}
kind_ahead = np.where(seen_ahead <= 0, 0, np.where(np.isin(seen_ahead, list(bushes)), 1,
                                                   np.where(np.isin(seen_ahead, list(empties)), 2, 3)))
live = alive > 0
if os.environ.get("PHASE"):   # 1: before the reversal, 2: after
    live = live & ((np.arange(T)[:, None] >= exp.world.switch_tick) == (os.environ["PHASE"] == "2"))
snaps = np.asarray(rec["w_snap"], np.float32)                        # [T/every, N, cells, cells]
every = T // snaps.shape[0]
vav = L.region("valence_av")
cells = [c for c in ("valence_av", "valence_app", "no_feed", "grasp", "ingest", "no_eat", "act") if c in L.names]
names = ("nothing", "bush", "empty bush", "other")
print(f"{which}: holding ticks by held kind and what is ahead (brain's view)")
for held_bad, hk in ((False, "good"), (True, "poison")):
    for k, an in enumerate(names):
        m = live & (seen_held > 0) & (bad == held_bad) & (kind_ahead == k)
        if m.sum() < 50:
            continue
        tt, aa = np.nonzero(m)
        row = f"  held {hk:6s} ahead {an:10s} ({m.sum():6d}): "
        row += " ".join(f"{c} {x[m][:, L.region(c)].mean():.2f}" for c in cells)
        row += f" | av0-1 {x[m][:, vav.start:vav.start + 2].mean():.2f} av2 {x[m][:, vav.start + 2].mean():.2f}"
        row += f" | USE {(act[m] == A.USE).mean():.2f} EAT {(act[m] == A.EAT).mean():.2f}"
        print(row)
        pick = np.random.default_rng(0).choice(len(tt), min(2000, len(tt)), replace=False)
        contrib = {}
        for r in L.names:
            sl = L.region(r)
            v = [x[t, a, sl] @ snaps[min(t // every, snaps.shape[0] - 1), a, sl, vav.start:vav.start + 2].mean(-1)
                 for t, a in zip(tt[pick], aa[pick])]
            contrib[r] = float(np.mean(v))
        if "in" in L.names:   # split the input drive by feature group
            ins = L.region("in")
            groups = {"in[vis app]": [i for i, n in enumerate(L.in_names) if ".app" in n and n.startswith("vis")],
                      "in[held_app]": [i for i, n in enumerate(L.in_names) if n.startswith("held_app")],
                      "in[held]": [L.in_names.index("held")],
                      "in[pain]": [L.in_names.index("pain")]}
            for g, idx in groups.items():
                idx = np.array(idx) + ins.start
                contrib[g] = float(np.mean([x[t, a, idx] @ snaps[min(t // every, snaps.shape[0] - 1), a, idx,
                                                                  vav.start:vav.start + 2].mean(-1)
                                            for t, a in zip(tt[pick], aa[pick])]))
        top = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))[:6]
        print("      drive onto valence_av 0-1: " + "  ".join(f"{k} {v:+.2f}" for k, v in top))
