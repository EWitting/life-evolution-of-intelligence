"""How do the animals of a run's last generation spend their life? From the recording: lifetime and first meal,
what they do when a full bush is in front of them, how much they move, and how often they stand grasping at
nothing (the freeze).

    uv run python scripts/probes/forage.py <experiment name> [...]"""
import json, sys
import numpy as np
from pathlib import Path

DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
for n in sys.argv[1:]:
    d = sorted(p for p in Path("runs", n).iterdir() if (p / "population.npz").exists())[-1]
    rec = np.load(d / "recording.npz"); rs = json.loads((d / "ruleset.json").read_text()); L = json.loads((d / "layout.json").read_text())
    names = rs["names"]; g, pos, dr, act, alive, ate, x = (rec[k] for k in ("grid", "pos", "dir", "action", "alive", "ate", "x"))
    alive = alive.astype(bool); T, N = act.shape; H, W = g.shape[1:]
    f = pos.astype(int) + DIRS[dr.astype(int) % 4]
    inb = (f >= 0).all(-1) & (f[..., 0] < H) & (f[..., 1] < W); fc = np.clip(f, 0, [H - 1, W - 1])
    obj = np.where(inb, g[np.arange(T)[:, None], fc[..., 0], fc[..., 1]], -1)
    off = dict(zip(L["names"], zip(L["offsets"], L["sizes"])))
    m = lambda r: x[..., off[r][0]:off[r][0] + off[r][1]].astype(np.float32).mean(-1) if r in off else np.zeros((T, N), np.float32)
    full = [i for i, s in enumerate(names) if s.endswith("Wild Gooseberry Bush") and "Empty" not in s]
    empty = [i for i, s in enumerate(names) if "Empty" in s]
    facing = np.isin(obj, full)[:-1] & alive[:-1]; a = act[1:]
    idle = (a == 4) & alive[:-1] & ~np.isin(obj[:-1], full)
    life = alive.sum(0); first = np.array([np.nonzero(ate[:, i] > 0)[0][0] if (ate[:, i] > 0).any() else -1 for i in range(N)])
    meals = (ate > 0).sum(0)
    print(f"== {n} ({d.name}): lifetime median {np.median(life):.0f} of {T}, quartiles {np.percentile(life, [25, 75]).round()}, alive at the end "
          f"{alive[-1].mean():.0%}; never ate {(first < 0).mean():.0%}; first meal at tick {np.median(first[first >= 0]) if (first >= 0).any() else 0:.0f} (median); "
          f"berries per life {meals.mean():.1f}")
    print(f"   facing a full bush on {facing.sum() / alive[:-1].sum():.1%} of ticks alive; then USE {(a[facing] == 4).mean():.2f}, FWD {(a[facing] == 1).mean():.2f}, "
          f"turn {np.isin(a[facing], [2, 3]).mean():.2f}; appetitive {m('valence_app')[:-1][facing].mean():.2f}, aversive {m('valence_av')[:-1][facing].mean():.2f}, "
          f"grasp {m('grasp')[:-1][facing].mean():.2f}")
    al = alive[:-1]
    print(f"   moving on {(np.abs(np.diff(pos.astype(int), axis=0)).sum(-1) > 0)[al].mean():.0%} of ticks alive; grasping at nothing {idle.sum() / al.sum():.0%}; "
          f"FWD {(a[al] == 1).mean():.2f}, turn {np.isin(a[al], [2, 3]).mean():.2f}, USE {(a[al] == 4).mean():.2f}; "
          f"bushes full at tick 0 / middle / end: " + " / ".join(f"{np.isin(g[t], full).sum() / max(1, np.isin(g[t], full + empty).sum()):.0%}" for t in (0, T // 2, T - 1)))
