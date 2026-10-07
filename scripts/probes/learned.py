"""What did the animals of a learning stage learn within their life? From the recording of the newest run:

  - the inherited learning rates on the look -> aversive synapses;
  - per berry type, how strongly its look drives the aversive cells (weights of the eye pointing straight ahead times
    the look, mean over the aversive cells): inherited, and the change from the last weight snapshot before the
    animal's first bite of that type to its last snapshot alive;
  - the aversive cells' activity while the animal faces a bush of the type, before and after its first bite of it;
  - the teacher: how often the pain cell and the safety cell fire, and on what.

    uv run python scripts/probes/learned.py <experiment name>"""
import json, sys
import numpy as np
from pathlib import Path

name = sys.argv[1]
d = sorted(p for p in Path("runs", name).iterdir() if (p / "population.npz").exists())[-1]
rec = np.load(d / "recording.npz"); rs = json.loads((d / "ruleset.json").read_text()); L = json.loads((d / "layout.json").read_text())
names = rs["names"]
grid, pos, dr, act, alive, ate, x = (rec[k] for k in ("grid", "pos", "dir", "action", "alive", "ate", "x"))
w_snap, w0, eta, look = rec["w_snap"].astype(np.float32), rec["w0"].astype(np.float32), rec["eta"].astype(np.float32), np.asarray(rec["appearance"], np.float32)
T, N = act.shape; H, W = grid.shape[1:]; S = w_snap.shape[0]; every = T // S
alive = alive.astype(bool)
off = dict(zip(L["names"], zip(L["offsets"], L["sizes"])))
cells = lambda r: list(range(off[r][0], off[r][0] + off[r][1]))
av = cells("valence_av")
K = look.shape[1]
ahead = [L["in_names"].index(f"vis+0.app{k}") for k in range(K)]
ident = [i for i, n in enumerate(L["in_names"]) if ".app" in n]
bush = [i for i, n in enumerate(names) if n.endswith("Wild Gooseberry Bush") and "Empty" not in n]
tname = lambda b: names[b].replace(" Wild Gooseberry Bush", "").replace("Wild Gooseberry Bush", "Plain")
pain_v = np.asarray(rs["pain_value"]) if np.ndim(rs["pain_value"]) == 1 else None
bad = {b: bool((rec["pain"][1:][(ate[1:] == b + 1)] > 0.5).mean() > 0.5) if (ate == b + 1).any() else None for b in bush}

e = eta[:, ident][:, :, av]
print(f"== {name} ({d.name}), {N} animals, life {T} ticks, weight snapshots every {every}")
print(f"inherited learning rate, look -> aversive: mean {e.mean():.3f}, median over animals {np.median(e.mean((1, 2))):.3f}, "
      f"range {e.mean((1, 2)).min():.3f} to {e.mean((1, 2)).max():.3f}")

drive = lambda w, b: (w[..., ahead, :][..., av] * look[b][:, None]).sum(-2).mean(-1)   # [..., N]: net drive of look b
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
front = pos.astype(int) + DIRS[dr.astype(int) % 4]
inb = (front >= 0).all(-1) & (front[..., 0] < H) & (front[..., 1] < W)
fc = np.clip(front, 0, [H - 1, W - 1])
obj = np.where(inb, grid[np.arange(T)[:, None], fc[..., 0], fc[..., 1]], -1)
a_av = x[..., av].astype(np.float32).mean(-1)
last = np.maximum(alive.sum(0) // every - 1, 0)              # last snapshot taken while alive

print(f"\n{'type':8s} {'meaning':8s} {'animals that bit it':>20s} {'bites each':>10s} | drive of its look on the aversive cells: "
      f"{'inherited':>9s} {'change after the first bite':>28s} {'change, never bitten':>21s} | aversive activity facing it: before / after the first bite")
for b in bush:
    first = np.full(N, -1)
    for i in range(N):
        ts = np.nonzero(ate[:, i] == b + 1)[0]
        if len(ts):
            first[i] = ts[0]
    bit = first >= 0
    d0 = drive(w0, b)
    ch, ch_no = [], []
    for i in range(N):
        end = drive(w_snap[last[i], i], b)
        if bit[i]:
            s0 = first[i] // every - 1                        # last snapshot before the bite (-1: the inherited weights)
            start = drive(w_snap[s0, i], b) if s0 >= 0 else drive(w0[i], b)
            if last[i] > s0:
                ch.append(end - start)
        elif last[i] >= 0:
            ch_no.append(end - drive(w0[i], b))
    face = (obj == b) & alive
    tt = np.arange(T)[:, None]
    bef = face & bit[None] & (tt < first[None]); aft = face & bit[None] & (tt > first[None] + 3)
    m = lambda v: f"{np.mean(v):+.3f}" if len(v) else "    -"
    fa = lambda sel: f"{a_av[sel].mean():.2f}" if sel.any() else "  - "
    meaning = "?" if bad[b] is None else ("poison" if bad[b] else "good")
    print(f"{tname(b):8s} {meaning:8s} {int(bit.sum()):20d} {(ate == b + 1).sum() / max(1, bit.sum()):10.1f} | {'':41s} "
          f"{d0.mean():+9.3f} {m(ch):>28s} {m(ch_no):>21s} | {'':28s} {fa(bef)} / {fa(aft)}")

us, sf = x[..., cells("us_pain")].astype(np.float32).mean(-1), (x[..., cells("safety")].astype(np.float32).mean(-1) if "safety" in off else None)
bit_bad = np.zeros((T, N), bool); bit_good = np.zeros((T, N), bool)
for b in bush:
    if bad[b] is not None:
        (bit_bad if bad[b] else bit_good)[ate == b + 1] = True
nxt = lambda m: np.roll(m, 1, axis=0)                          # the teacher cells answer on the tick after the bite
print(f"\npain cell: {us[nxt(bit_bad)].mean():.2f} after a poison bite ({int(bit_bad.sum())} bites), {us[nxt(bit_good)].mean():.2f} after a good bite "
      f"({int(bit_good.sum())} bites)")
if sf is not None:
    g = sf[nxt(bit_good)]
    print(f"safety cell: mean {g.mean():.3f} after a good bite, above 0.05 after {np.mean(g > 0.05):.1%} of good bites; "
          f"{sf[nxt(bit_bad)].mean():.3f} after a poison bite; {sf[alive & ~nxt(bit_good) & ~nxt(bit_bad)].mean():.4f} otherwise")
    print(f"aversive activity on the tick of a good bite (what the safety cell can see): mean {a_av[bit_good].mean():.3f}, "
          f"above 0.2 on {np.mean(a_av[bit_good] > 0.2):.1%} of good bites")
