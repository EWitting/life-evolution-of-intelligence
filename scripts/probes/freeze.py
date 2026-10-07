"""The freeze: episodes in which an animal stands and grasps at nothing (USE with no full bush in front) for at
least MIN ticks. What started them, what the animal sees and what its cells do meanwhile, how they end.

    uv run python scripts/probes/freeze.py <experiment name> [MIN=50]"""
import json, sys
import numpy as np
from pathlib import Path

n = sys.argv[1]; MIN = int(sys.argv[2]) if len(sys.argv) > 2 else 50
d = sorted(p for p in Path("runs", n).iterdir() if (p / "population.npz").exists())[-1]
rec = np.load(d / "recording.npz"); rs = json.loads((d / "ruleset.json").read_text()); L = json.loads((d / "layout.json").read_text())
names = rs["names"]; inn = L["in_names"]
g, pos, dr, act, alive, ate, x, food = (rec[k] for k in ("grid", "pos", "dir", "action", "alive", "ate", "x", "food"))
alive = alive.astype(bool); T, N = act.shape; H, W = g.shape[1:]
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
f = pos.astype(int) + DIRS[dr.astype(int) % 4]
inb = (f >= 0).all(-1) & (f[..., 0] < H) & (f[..., 1] < W); fc = np.clip(f, 0, [H - 1, W - 1])
obj = np.where(inb, g[np.arange(T)[:, None], fc[..., 0], fc[..., 1]], -1)
full = [i for i, s in enumerate(names) if s.endswith("Wild Gooseberry Bush") and "Empty" not in s]
empty = [i for i, s in enumerate(names) if "Empty" in s]
off = dict(zip(L["names"], zip(L["offsets"], L["sizes"])))
reg = lambda r: x[..., off[r][0]:off[r][0] + off[r][1]].astype(np.float32).mean(-1)
kind = lambda o: "wall" if o < 0 else "open ground" if o == 0 else "full bush" if o in full else "empty bush" if o in empty else "other"
occupied = np.zeros((T, H, W), bool)
for i in range(N):
    occupied[np.arange(T)[alive[:, i]], pos[alive[:, i], i, 0], pos[alive[:, i], i, 1]] = True
agent_ahead = inb & occupied[np.arange(T)[:, None], fc[..., 0], fc[..., 1]]
idle = (act == 4) & alive & ~np.isin(obj, full)
eps = []
for i in range(N):
    t = 0
    while t < T:
        if idle[t, i]:
            t0 = t
            while t < T and alive[t, i] and (act[t, i] == 4 or (t + 1 < T and act[t + 1, i] == 4)) and (pos[t, i] == pos[t0, i]).all():
                t += 1
            if t - t0 >= MIN:
                eps.append((i, t0, t))
        else:
            t += 1
tot = sum(b - a for _, a, b in eps)
print(f"== {n} ({d.name}): {len(eps)} freezes of {MIN}+ ticks, {tot / alive.sum():.0%} of all ticks alive; median length {np.median([b - a for _, a, b in eps]) if eps else 0:.0f}")
if eps:
    cnt = lambda xs: {k: f"{v / len(xs):.0%}" for k, v in sorted({k: xs.count(k) for k in set(xs)}.items(), key=lambda kv: -kv[1])}
    print("   in front during the freeze:", cnt([("another animal" if agent_ahead[(a + b) // 2, i] else kind(int(obj[(a + b) // 2, i]))) for i, a, b in eps]))
    print("   what happened in the 5 ticks before:", cnt([("ate" if (ate[max(0, a - 5):a + 1, i] > 0).any() else "moved" if (act[max(0, a - 5):a, i] == 1).any() else "turned" if np.isin(act[max(0, a - 5):a, i], [2, 3]).any() else "born there" if a < 3 else "other") for i, a, b in eps]))
    print("   how it ended:", cnt([("death" if b >= T or not alive[min(b, T - 1), i] else "moved on") for i, a, b in eps]))
    mid = [(i, (a + b) // 2) for i, a, b in eps]; ii = np.array([m[0] for m in mid]); tt = np.array([m[1] for m in mid])
    feat = lambda nm: float(x[tt, ii, inn.index(nm)].astype(np.float32).mean())
    print(f"   stomach at the start {np.mean([food[a, i] for i, a, b in eps]) / 20:.0%}, at the end {np.mean([food[min(b, T - 1), i] for i, a, b in eps]) / 20:.0%}")
    print("   senses mid-freeze: " + ", ".join(f"{c}.hit {feat(c + '.hit'):.2f} near {feat(c + '.near'):.2f}" for c in ("vis-60", "vis-30", "vis+0", "vis+30", "vis+60")))
    print("   cells mid-freeze: " + ", ".join(f"{r} {float(reg(r)[tt, ii].mean()):.2f}" for r in L["names"] if r not in ("in", "out"))
          + "; motor: " + ", ".join(f"{a_} {float(x[tt, ii, off['out'][0] + k].astype(np.float32).mean()):.2f}" for k, a_ in enumerate(["NOOP", "FWD", "LEFT", "RIGHT", "USE", "EAT", "VOC"])))
    act_all = act[alive]
    print("   for comparison, cells over all ticks alive: " + ", ".join(f"{r} {float(reg(r)[alive].mean()):.2f}" for r in L["names"] if r not in ("in", "out")))
