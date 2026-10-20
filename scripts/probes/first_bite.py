"""Does a creature judge a bush by its look before the first bite? Read from the recordings of the last generation
of the frozen 2.3 runs (main and control, three seeds; one world of 64 creatures each, no new simulation):
    python scripts/probes/first_bite.py
A visit is a run of ticks with the same full bush at the mouth (the world after the previous step, as the brain
sees it). Per kind (poison / good in that phase, read from the pain at each bite) and phase (before / after the
switch at tick 800): visits, P(any bite during the visit), P(bite on the first tick), bites per visit, and
P(a second bite | the first bite hurt / did not). Then the same split by experience: how many painful bites the
creature has had from that berry type earlier in the same phase (0, 1-2, 3+), the test of a learned aversion to
the look (a lesson stops the first bite on the next visit)."""
import json
from pathlib import Path
import numpy as np
from life import actions as A
from life.world import DIRS

D = json.load(open("book/data/2.3.json"))
SW = 800
BITE = [A.USE, A.EAT]


def visits(run: Path):
    rec = np.load(run / "recording.npz")
    names = json.load(open(run / "ruleset.json"))["names"]
    ate, pain, act, alive = rec["ate"].astype(int), rec["pain"], rec["action"], rec["alive"].astype(bool)
    grid, pos, dir_ = rec["grid"], rec["pos"].astype(int), rec["dir"].astype(int)
    T, N = act.shape
    H, W = grid.shape[1:]
    full = {i: names.index(n.replace("Wild Gooseberry Bush", "Gooseberry")) for i, n in enumerate(names)
            if n.endswith("Wild Gooseberry Bush") and "Empty" not in n}          # full bush -> its berry
    # poison per berry and phase: a bite of it hurt
    ph_of = lambda t: int(t >= SW)
    hurt = {}
    for ph, sl in ((0, slice(0, SW)), (1, slice(SW, T))):
        a, p = ate[sl], pain[sl]
        for o in np.unique(a[a > 0]):
            hurt[(int(o), ph)] = bool(np.mean(p[a == o] > 0.5) > 0.5)
    # what is at the mouth when the brain acts at t: the world after step t-1
    g = np.r_[grid[:1] * 0, grid[:-1]]
    ps, ds = np.r_[pos[:1], pos[:-1]], np.r_[dir_[:1], dir_[:-1]]
    fr = ps + np.asarray(DIRS)[ds]
    fy, fx = np.clip(fr[..., 0], 0, H - 1), np.clip(fr[..., 1], 0, W - 1)
    obj = g[np.arange(T)[:, None], fy, fx]
    lay = json.load(open(run / "layout.json"))
    reg = {n: slice(o, o + z) for n, o, z in zip(lay["names"], lay["offsets"], lay["sizes"])}
    cells = [n for n in ("valence_av", "valence_app", "value_av", "value_app") if n in reg]
    x = rec["x"]
    out = []
    for c in range(N):
        prior = {}                     # (berry, phase) -> painful bites so far
        t = 0
        while t < T:
            o = int(obj[t, c])
            if not alive[t, c] or o not in full:
                t += 1; continue
            t0, cell = t, (fy[t, c], fx[t, c])
            while t < T and alive[t, c] and obj[t, c] == o and (fy[t, c], fx[t, c]) == cell and ph_of(t) == ph_of(t0):
                t += 1
            berry, ph = full[o], ph_of(t0)
            if (berry, ph) not in hurt:
                continue
            bt = [u for u in range(t0, t) if act[u, c] in BITE]
            got = [u for u in range(t0, t + 1) if u < T and ate[u, c] == berry]
            first_hurt = bool(got) and pain[got[0], c] > 0.5
            out.append(dict(poison=hurt[(berry, ph)], ph=ph, len=t - t0, bite=len(bt) > 0, bite0=act[t0, c] in BITE,
                            n=len(got), first_hurt=first_hurt, prior=prior.get((berry, ph), 0),
                            **{n: float(x[t0, c, reg[n]].astype(np.float32).mean()) for n in cells}))
            prior[(berry, ph)] = prior.get((berry, ph), 0) + sum(pain[u, c] > 0.5 for u in got)
    return out


def table(vs, label):
    print(f"\n{label}")
    print(f"  {'':22s} {'visits':>6s} {'P(bite)':>8s} {'P(bite t0)':>10s} {'bites/visit':>11s} {'P(2nd|1st hurt)':>15s} {'P(2nd|1st ok)':>13s}")
    for ph in (0, 1):
        for pz in (True, False):
            v = [x for x in vs if x["ph"] == ph and x["poison"] == pz]
            if not v:
                continue
            h = [x for x in v if x["n"] >= 1 and x["first_hurt"]]
            ok = [x for x in v if x["n"] >= 1 and not x["first_hurt"]]
            sec = lambda xs: f"{np.mean([x['n'] >= 2 for x in xs]):.2f}" if xs else "  -"
            print(f"  {['before', 'after'][ph]:6s} {'poison' if pz else 'good':6s} now        {len(v):6d} {np.mean([x['bite'] for x in v]):8.2f} "
                  f"{np.mean([x['bite0'] for x in v]):10.2f} {np.mean([x['n'] for x in v]):11.2f} {sec(h):>15s} {sec(ok):>13s}")
    print("  by painful bites from that type earlier in the phase (poison now):  P(bite) per visit [visits]")
    for ph in (0, 1):
        row = []
        for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 99)):
            v = [x for x in vs if x["ph"] == ph and x["poison"] and lo <= x["prior"] <= hi]
            row.append(f"{lo}-{hi}: {np.mean([x['bite'] for x in v]):.2f} [{len(v)}]" if v else f"{lo}-{hi}: -")
        print(f"    {['before', 'after'][ph]:6s} " + "   ".join(row))
    cells = [n for n in ("valence_av", "valence_app", "value_av", "value_app") if n in vs[0]]
    print("  cells on the first tick of a visit (mean rate), by kind now and painful bites from that type earlier:")
    for ph in (0, 1):
        for pz in (True, False):
            for lo, hi in ((0, 0), (1, 2), (3, 99)):
                v = [x for x in vs if x["ph"] == ph and x["poison"] == pz and lo <= x["prior"] <= hi]
                if v:
                    print(f"    {['before', 'after'][ph]:6s} {'poison' if pz else 'good':6s} {lo}-{hi:<2d} [{len(v):4d}] "
                          + "  ".join(f"{n} {np.mean([x[n] for x in v]):.3f}" for n in cells))


for side in ("main", "control"):
    allv = []
    for r in D["runs"][side]:
        v = visits(Path("runs") / r["dir"])
        allv += v
        table(v, f"{side} seed {r['seed']}")
    table(allv, f"{side}, three seeds pooled")
