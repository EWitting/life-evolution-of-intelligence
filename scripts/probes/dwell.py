"""What could a lasting state after a meal (stay, search here) or after a poison bite (get away) be worth? From the
recording of a run's last generation: how a visit to a bush ends (bush emptied or berries left behind), what the
wait for the next meal is, whether food is clumped around a meal or poison around a poison bite, and what follows
a poison bite. A visit is a run of bites of one type with gaps under 6 ticks (as in lessons.py).

    uv run python scripts/probes/dwell.py <experiment name> [...]"""
import json, sys
import numpy as np
from pathlib import Path

DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
RADIUS = 3            # cells around a bite in which bushes are counted (a square of side 7)
med = lambda v: float(np.median(v)) if len(v) else float("nan")
mean = lambda v: float(np.mean(v)) if len(v) else float("nan")

for n in sys.argv[1:]:
    d = sorted(p for p in Path("runs", n).iterdir() if (p / "population.npz").exists())[-1]
    rec = np.load(d / "recording.npz"); rs = json.loads((d / "ruleset.json").read_text()); L = json.loads((d / "layout.json").read_text())
    names = rs["names"]
    g, pos, dr, alive, ate, x, food = (rec[k] for k in ("grid", "pos", "dir", "alive", "ate", "x", "food"))
    alive = alive.astype(bool); T, N = ate.shape; H, W = g.shape[1:]
    # which types are poison is drawn per life and not in the ruleset file: a type is poison when its bites hurt
    pain = np.array([rec["pain"][ate == o].mean() > 0.5 if (ate == o).any() else False for o in range(len(names))], float)
    off = dict(zip(L["names"], zip(L["offsets"], L["sizes"])))
    cell = lambda r: x[..., off[r][0]].astype(np.float32) if r in off else np.full((T, N), np.nan, np.float32)
    full = np.array([i for i, s in enumerate(names) if s.endswith("Wild Gooseberry Bush") and "Empty" not in s])
    good_full = np.array([i for i in full if pain[i + 1] == 0]); bad_full = np.array([i for i in full if pain[i + 1] > 0])
    front = np.clip(pos.astype(int) + DIRS[dr.astype(int) % 4], 0, [H - 1, W - 1])

    def around(t, c, ids):
        """Share of the cells within RADIUS of cell c at tick t that hold one of the objects ids (the cell itself not)."""
        a = g[t, max(0, c[0] - RADIUS):c[0] + RADIUS + 1, max(0, c[1] - RADIUS):c[1] + RADIUS + 1]
        return (np.isin(a, ids).sum() - np.isin(g[t, c[0], c[1]], ids)) / (a.size - 1)

    # visits per animal: [type, first tick, last tick, bites, bush cell]
    visits = []
    for i in range(N):
        ts = np.nonzero(ate[:, i] > 0)[0]; v = []
        for j, t in enumerate(ts):
            o = int(ate[t, i])
            if v and v[-1][0] == o and t - ts[j - 1] < 6 and (front[t, i] == v[-1][4]).all():
                v[-1][2] = t; v[-1][3] += 1
            else:
                v.append([o, t, t, 1, front[t, i].copy()])
        visits.append(v)

    left_full, bites, wait_full, wait_empty, dist, stomach, hungry_at_leave, back = [], [], [], [], [], [], [], []
    near_good, near_bad, after_bad, after_good, wait_bad, in_visit = [], [], [], [], [], []
    for i, v in enumerate(visits):
        for j, (o, t0, t1, b, c) in enumerate(v):
            nxt = v[j + 1] if j + 1 < len(v) else None
            if pain[o] > 0:
                near_bad.append(around(t1, c, bad_full))
                if nxt: after_bad.append(pain[nxt[0]] > 0); wait_bad.append(nxt[1] - t1)
                continue
            still = g[min(T - 1, t1 + 1), c[0], c[1]] == o - 1          # the bush still carries berries after the last bite
            left_full.append(still); bites.append(b); near_good.append(around(t1, c, good_full))
            stomach.append(food[t1, i]); hungry_at_leave.append(cell("hungry")[min(T - 1, t1 + 6), i] > 0.5)
            if b > 1: in_visit.append((t1 - t0) / (b - 1))
            if nxt:
                (wait_full if still else wait_empty).append(nxt[1] - t1); after_good.append(pain[nxt[0]] > 0)
                dist.append(np.abs(nxt[4] - c).sum()); back.append((nxt[4] == c).all())
    left_full = np.array(left_full); hungry_at_leave = np.array(hungry_at_leave)

    # how long an emptied bush stays empty
    isfull = np.isin(g, full); regrow = []
    for r, c in zip(*np.nonzero(isfull.any(0) & ~isfull.all(0))):
        s = isfull[:, r, c].astype(np.int8); ch = np.nonzero(np.diff(s))[0]
        regrow += [ch[k + 1] - ch[k] for k in range(len(ch) - 1) if s[ch[k]] == 1]
    al = alive[:-1]; moving = (np.abs(np.diff(pos.astype(int), axis=0)).sum(-1) > 0)
    base_good = np.isin(g, good_full).mean(); base_bad = np.isin(g, bad_full).mean()
    max_food = float(np.nanmax(food))

    print(f"== {n} ({d.name}): {N} animals, {sum(len(v) for v in visits) / N:.1f} visits per life, moving {moving[al].mean():.0%} of ticks alive; "
          f"hungry on {(cell('hungry')[alive] > 0.5).mean():.0%}, rest on {(cell('rest')[alive] > 0.5).mean():.0%}, "
          f"warm_seek on {(cell('warm_seek')[alive] > 0.5).mean():.0%}")
    print(f"   good visits: {mean(bites):.1f} bites each, one bite every {mean(in_visit):.1f} ticks; berries left behind in {left_full.mean():.0%} "
          f"(then stomach {np.mean(np.array(stomach)[left_full]) / max_food:.0%} full, still hungry in {hungry_at_leave[left_full].mean():.0%})")
    print(f"   wait for the next meal: {med(wait_full):.0f} ticks (median) after leaving berries, {med(wait_empty):.0f} after emptying the bush; "
          f"next bush {med(dist):.0f} cells away (median), the same bush in {mean(back):.0%}; an emptied bush is empty for {med(regrow):.0f} ticks (median)")
    print(f"   clumping, bushes with berries within {RADIUS} cells: good ones around a good meal {mean(near_good):.3f} (whole map {base_good:.3f}); "
          f"poison ones around a poison bite {mean(near_bad):.3f} (whole map {base_bad:.3f})")
    print(f"   the next visit is to poison: after a poison bite {mean(after_bad):.0%}, after a good meal {mean(after_good):.0%}; "
          f"wait after a poison bite {med(wait_bad):.0f} ticks (median)", flush=True)
