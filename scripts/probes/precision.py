"""What recorded agents do in standard situations, split by what the object is in that life (good / poison / dud).
    precision.py <experiment name> [...]"""
import json, sys
import numpy as np
from pathlib import Path

ACT = ["NOOP", "FWD", "LEFT", "RIGHT", "USE", "EAT", "VOC"]
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
for name in sys.argv[1:]:
    d = sorted(p for p in Path("runs", name).iterdir() if (p / "population.npz").exists())[-1]
    rec = np.load(d / "recording.npz"); rs = json.loads((d / "ruleset.json").read_text())
    names = rs["names"]
    grid, pos, dr, held, act, alive, food = (rec[k] for k in ("grid", "pos", "dir", "held", "action", "alive", "food"))
    T, N = act.shape; H, W = grid.shape[1:]
    g, p, q, hd, al = grid[:-1], pos[:-1].astype(int), dr[:-1].astype(int), held[:-1].astype(int), alive[:-1].astype(bool)
    a = act[1:].astype(int); dfood = food[1:] - food[:-1]
    front = p + DIRS[q % 4]
    inb = (front >= 0).all(-1) & (front[..., 0] < H) & (front[..., 1] < W)
    fc = np.clip(front, 0, [H - 1, W - 1])
    obj = np.where(inb, g[np.arange(T - 1)[:, None], fc[..., 0], fc[..., 1]], -1)
    bush = [i for i, n in enumerate(names) if n.endswith("Wild Gooseberry Bush") and "Empty" not in n]
    cls = {}
    for b in bush:
        m = (a == 5) & (hd == b + 1) & al
        cls[b] = ("good" if np.median(dfood[m]) > 0 else "poison") if m.sum() >= 3 else "never eaten (dud or avoided)"
    print(f"\n== {name} ({d.name}), {N} recorded agents; lifetime mean {alive.sum(0).mean():.0f}")

    def show(label, m):
        m = m & al
        if m.sum() < 20:
            print(f"  {label:52s} n={m.sum()}"); return
        fr = np.bincount(a[m], minlength=7) / m.sum()
        print(f"  {label:52s} n={m.sum():6d}  FWD {fr[1]:.2f} turn {fr[2] + fr[3]:.2f} USE {fr[4]:.2f} EAT {fr[5]:.2f}")
    for c in ("good", "poison", "never eaten (dud or avoided)"):
        bs = [b for b in bush if cls[b] == c]
        if not bs:
            continue
        show(f"{c} bush in front, empty hand", np.isin(obj, bs) & (hd == 0))
        show(f"{c} berry in hand", np.isin(hd, [b + 1 for b in bs]))
    show("empty bush in front, empty hand", np.isin(obj, [i for i, n in enumerate(names) if "Empty" in n]) & (hd == 0))
    show("nothing in front, empty hand", (obj == 0) & (hd == 0))
    eats = (a == 5) & al
    good_b = [b + 1 for b in bush if cls[b] == "good"]; bad_b = [b + 1 for b in bush if cls[b] == "poison"]
    print(f"  berries eaten per agent: good {(eats & np.isin(hd, good_b)).sum() / N:.1f}, poison {(eats & np.isin(hd, bad_b)).sum() / N:.1f}")
