"""Follow individuals through a recorded life: what happens to picking, eating and valence for a berry type after
the agent's first meal of that type (poison types: after the first sickness)?
    events.py <experiment name> [agent index for a timeline]"""
import json, sys
import numpy as np
from pathlib import Path

ACT = ["NOOP", "FWD", "LEFT", "RIGHT", "USE", "EAT", "VOC"]
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
name = sys.argv[1]
d = sorted(p for p in Path("runs", name).iterdir() if (p / "population.npz").exists())[-1]
rec = np.load(d / "recording.npz"); rs = json.loads((d / "ruleset.json").read_text()); L = json.loads((d / "layout.json").read_text())
cfg = json.loads((d / "config.json").read_text())
names = rs["names"]
grid, pos, dr, held, act, alive, food = (rec[k] for k in ("grid", "pos", "dir", "held", "action", "alive", "food"))
x = rec["x"]
T, N = act.shape; H, W = grid.shape[1:]
o = dict(zip(L["names"], zip(L["offsets"], L["sizes"])))
reg = lambda r: x[:, :, o[r][0]:o[r][0] + o[r][1]].astype(np.float32).mean(-1)
vap, vav = reg("valence_app"), reg("valence_av")
bush = [i for i, n in enumerate(names) if n.endswith("Wild Gooseberry Bush") and "Empty" not in n]
berry_of = {b: b + 1 for b in bush}
tname = lambda b: names[b].replace(" Wild Gooseberry Bush", "").replace("Wild Gooseberry Bush", "Plain")
delay = cfg["world"]["sickness_delay"]

# state before the action taken at tick t is the recorded state at t-1
p, q, hd, al = pos[:-1].astype(int), dr[:-1].astype(int), held[:-1].astype(int), alive[:-1].astype(bool)
a = act[1:].astype(int)
front = p + DIRS[q % 4]
inb = (front >= 0).all(-1) & (front[..., 0] < H) & (front[..., 1] < W)
fc = np.clip(front, 0, [H - 1, W - 1])
obj = np.where(inb, grid[:-1][np.arange(T - 1)[:, None], fc[..., 0], fc[..., 1]], -1)
dfood = food[1:] - food[:-1]
va_p, va_v = vap[:-1], vav[:-1]     # valence before the action

# class of each type in this life, from what eating it did to the food level
cls = {}
for b in bush:
    m = (a == 5) & (hd == berry_of[b]) & al
    if m.sum():
        cls[b] = "good" if np.median(dfood[m]) > 0 else "poison"
print(f"== {name} ({d.name}); types in this life: " + ", ".join(f"{tname(b)}={c}" for b, c in cls.items()))

rows = {("poison", "before"): [], ("poison", "after"): [], ("good", "before"): [], ("good", "after"): []}
for i in range(N):
    for b, c in cls.items():
        eats = np.nonzero((a[:, i] == 5) & (hd[:, i] == berry_of[b]) & al[:, i])[0]
        if not len(eats):
            continue
        t0 = eats[0] + (delay if c == "poison" else 1)       # the lesson has arrived from here on
        face = (obj[:, i] == b) & (hd[:, i] == 0) & al[:, i]
        hold = (hd[:, i] == berry_of[b]) & al[:, i]
        tt = np.arange(T - 1)
        for when, sel in (("before", tt < t0), ("after", tt >= t0)):
            f, h = face & sel, hold & sel
            rows[(c, when)].append([f.sum(), (a[:, i][f] == 4).sum(), h.sum(), (a[:, i][h] == 5).sum(), len(eats[(eats >= t0)] if when == "after" else eats[eats < t0]),
                                    va_p[:, i][f].sum(), va_v[:, i][f].sum()])
print(f"{'':28s} {'bush in front, empty hand':>26s} {'P(USE)':>7s} {'berry in hand':>14s} {'P(EAT)':>7s} {'berries eaten/agent-type':>25s} {'valence app / av when facing':>29s}")
for (c, when), r in rows.items():
    r = np.array(r).sum(0) if len(r) else np.zeros(7)
    n = max(1, len(rows[(c, when)]))
    print(f"{c:7s} type, {when:6s} lesson: {int(r[0]):26d} {r[1] / max(r[0], 1):7.2f} {int(r[2]):14d} {r[3] / max(r[2], 1):7.2f} {r[4] / n:25.2f} "
          f"{r[5] / max(r[0], 1):14.2f} / {r[6] / max(r[0], 1):.2f}")

# the 12 ticks after each poison meal: what does the agent do, what do the valence cells do
if any(c == "poison" for c in cls.values()):
    win = 14; acc_a = np.zeros((win, 7)); acc_v = np.zeros((win, 2)); k = 0
    for i in range(N):
        for b, c in cls.items():
            if c != "poison": continue
            for t in np.nonzero((a[:, i] == 5) & (hd[:, i] == berry_of[b]) & al[:, i])[0]:
                if t + win < T - 1 and al[t + win, i]:
                    for j in range(win):
                        acc_a[j, a[t + j, i]] += 1
                    acc_v += np.stack([vap[t + 1:t + 1 + win, i], vav[t + 1:t + 1 + win, i]], 1); k += 1
    if k:
        print(f"after a poison meal ({k} meals; sickness arrives {delay} ticks later): ticks 0..{win - 1}")
        print("  aversive valence: " + " ".join(f"{v:.2f}" for v in acc_v[:, 1] / k))
        print("  appetitive:       " + " ".join(f"{v:.2f}" for v in acc_v[:, 0] / k))
        for j in (4, 5, 1, 2, 3):
            print(f"  P({ACT[j]:5s})          " + " ".join(f"{v:.2f}" for v in acc_a[:, j] / k))

if len(sys.argv) > 2:
    i = int(sys.argv[2])
    print(f"\ntimeline of agent {i} (lived {int(alive[:, i].sum())} ticks):")
    for t in range(T - 1):
        if not al[t, i]: break
        if a[t, i] == 5 and hd[t, i] in berry_of.values():
            b = hd[t, i] - 1
            print(f"  t={t:4d} eats {tname(b):7s} ({cls.get(b, '?'):6s}) food {food[t, i]:5.1f} -> {food[t + 1, i]:5.1f}   valence app/av before: {va_p[t, i]:.2f}/{va_v[t, i]:.2f}")
        elif obj[t, i] in bush and hd[t, i] == 0 and a[t, i] != 4:
            print(f"  t={t:4d} faces {tname(obj[t, i]):7s} ({cls.get(obj[t, i], '?'):6s}) bush and does {ACT[a[t, i]]:5s}             valence app/av: {va_p[t, i]:.2f}/{va_v[t, i]:.2f}")
