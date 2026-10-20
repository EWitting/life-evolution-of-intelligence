"""Mean activity of named cells around events in recorded lives (runs with recording.npz and layout.json):
    python scripts/probes/cell_trace.py <run dir> [<run dir> ...]
Events: painful bites and good bites (ticks t-1 .. t+3, t = the tick the berry is eaten); and the first tick of a
visit to a full bush that is poison / good at that tick, before and after WorldConfig.switch_tick (the kinds read
from the pain of the bites in each phase). Cells: the critic's and the aversion path's, those present."""
import json, sys
from pathlib import Path
import numpy as np
from life.dashboard import phase_of_ticks
from life.world import DIRS

CELLS = ("us_pain", "habenula", "gp_habenular", "ventral_pallidum", "dopamine", "value_app", "value_av", "valence_av",
         "valence_app", "no_feed", "grasp")
for run in map(Path, sys.argv[1:]):
    rec = np.load(run / "recording.npz"); L = json.load(open(run / "layout.json"))
    cfg = json.load(open(run / "config.json")); names = json.load(open(run / "ruleset.json"))["names"]
    reg = {n: slice(o, o + z) for n, o, z in zip(L["names"], L["offsets"], L["sizes"])}
    cells = [c for c in CELLS if c in reg]
    x = rec["x"].astype(np.float32); ate, pain, alive = rec["ate"].astype(int), rec["pain"], rec["alive"].astype(bool)
    T, N = ate.shape
    da = rec["mod"][..., L["mod_names"].index("da")] if "da" in L["mod_names"] else None
    hurt, good = (ate > 0) & (pain > 0.5), (ate > 0) & (pain <= 0.5)
    print(f"\n{run}  (painful bites {hurt.sum()}, good bites {good.sum()})")
    print(f"  {'':18s} " + " ".join(f"{c[:11]:>11s}" for c in cells) + ("  da" if da is not None else ""))
    for lab, ev in (("painful", hurt), ("good", good)):
        for k in (-1, 0, 1, 2, 3):
            m = np.zeros_like(ev)
            if k >= 0: m[k:] = ev[:T - k]
            else: m[:k] = ev[-k:]
            m &= alive
            print(f"  {lab:8s} t{k:+d}       " + " ".join(f"{x[..., reg[c]][m].mean():11.3f}" for c in cells)
                  + (f"  {da[m].mean():+.2f}" if da is not None else ""))
    # arrivals at full bushes, by kind in the phase
    grid, pos, dr = rec["grid"], rec["pos"].astype(int), rec["dir"].astype(int)
    H, W = grid.shape[1:]
    g = np.r_[grid[:1] * 0, grid[:-1]]
    fr = np.r_[pos[:1], pos[:-1]] + np.asarray(DIRS)[np.r_[dr[:1], dr[:-1]]]
    obj = g[np.arange(T)[:, None], np.clip(fr[..., 0], 0, H - 1), np.clip(fr[..., 1], 0, W - 1)]
    full = {i: names.index(n.replace("Wild Gooseberry Bush", "Gooseberry")) for i, n in enumerate(names)
            if n.endswith("Wild Gooseberry Bush") and "Empty" not in n}
    ph = phase_of_ticks(T, cfg["world"])
    hurt_of = {}
    for p in (0, 1):
        a, q = ate[ph == p], pain[ph == p]
        for o in np.unique(a[a > 0]):
            hurt_of[(int(o), p)] = bool(np.mean(q[a == o] > 0.5) > 0.5)
    arr = {}
    for c in range(N):
        for t in range(1, T):
            o = int(obj[t, c])
            if alive[t, c] and o in full and obj[t - 1, c] != o and (full[o], ph[t]) in hurt_of:
                arr.setdefault((ph[t], hurt_of[(full[o], ph[t])]), []).append((t, c))
    for (p, kind), v in sorted(arr.items()):
        ts, cs = np.array(v).T
        print(f"  arrive {'before' if p == 0 else 'after '} {'poison' if kind else 'good':6s} [{len(v):5d}] "
              + " ".join(f"{x[ts, cs, reg[c]].mean():11.3f}" for c in cells) + (f"  {da[ts, cs].mean():+.2f}" if da is not None else "")
              + f"  P(bite) {np.isin(rec['action'][ts, cs], (4, 5)).mean():.2f}, near {x[ts, cs, L['in_names'].index('vis+0.near')].mean():.2f}")
