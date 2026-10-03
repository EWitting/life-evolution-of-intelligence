"""Economy and behaviour of the last generation of runs: last-25-generation means from fitness.csv plus statistics of
the recording (a 64-agent sample for large populations).

    uv run python scripts/probes/eco.py <experiment name> [...]

energy   food eaten per life (the fitness since v19), last 25 generations
alive    ticks alive; surv = agents alive at the end
meals    eating events per life (recording)
actions  share of ticks alive spent on each action
cells    distinct cells visited per 100 ticks; camp% = share of 100-tick windows spent within 3 cells
empty%   share of all bushes that are empty, averaged over the life: high = the agents eat all there is, so
         fitness is limited by the food supply and not by skill
"""
import sys, json, csv, numpy as np
from pathlib import Path

print(f"{'run':24s} {'energy':>6s} {'alive':>5s} {'surv':>5s} | {'meals':>5s} {'NOOP':>5s} {'FWD':>5s} {'TURN':>5s} {'USE':>5s} {'EAT':>5s} | "
      f"{'cells':>5s} {'camp%':>5s} {'empty%':>6s} | {'s/gen':>5s}")
for name in sys.argv[1:]:
    d = sorted(p for p in Path("runs", name).iterdir() if (p / "population.npz").exists())[-1]
    allrows = list(csv.DictReader(open(d / "fitness.csv"))); rows = allrows[-25:]
    m = lambda k: np.mean([float(r[k]) for r in rows])
    rs = json.loads((d / "ruleset.json").read_text())
    rec = np.load(d / "recording.npz")
    alive = rec["alive"].astype(bool); food = rec["food"].astype(np.float64); act = rec["action"].astype(int); pos = rec["pos"].astype(int)
    T, N = alive.shape
    meals = (rec["ate"] > 0).sum(0) if "ate" in rec else (np.diff(food, axis=0) > 0).sum(0)
    fr = lambda a: (act[alive] == a).mean()
    cell = pos[..., 0] * 1000 + pos[..., 1]
    distinct, camp = [], []
    for a in range(N):
        for t0 in range(0, alive[:, a].sum() - 99, 100):
            n = len(np.unique(cell[t0:t0 + 100, a])); distinct.append(n); camp.append(n <= 3)
    names = rs["names"]
    full = [i for i, n in enumerate(names) if n.endswith("Wild Gooseberry Bush") and "Empty" not in n]
    empty = [i for i, n in enumerate(names) if "Empty" in n]
    grid = rec["grid"]
    n_full, n_empty = np.isin(grid, full).sum(), np.isin(grid, empty).sum()
    spg = float(allrows[-1]["seconds"]) / len(allrows)
    print(f"{name:24s} {m('eaten'):6.1f} {m('alive_ticks'):5.0f} {m('survivors'):5.1f} | {meals.mean():5.1f} {fr(0):5.2f} {fr(1):5.2f} "
          f"{fr(2) + fr(3):5.2f} {fr(4):5.2f} {fr(5):5.2f} | {np.mean(distinct):5.1f} {np.mean(camp):5.0%} {n_empty / max(1, n_full + n_empty):6.0%} | {spg:5.2f}")
