"""Economy and behaviour statistics of the last generation of runs (recording) plus last-25 csv means."""
import sys, json, csv, numpy as np
from pathlib import Path
print(f"{'run':24s} {'fed':>5s} {'alive':>5s} {'surv':>5s} | {'meals':>5s} {'absorb%':>7s} {'NOOP':>5s} {'FWD':>5s} {'TURN':>5s} {'USE':>5s} {'EAT':>5s} | {'cells/100t':>10s} {'camp%':>5s} | {'ge>.95':>6s} {'s/gen':>5s}")
for name in sys.argv[1:]:
    d = sorted(p for p in Path("runs", name).iterdir() if (p / "population.npz").exists())[-1]
    allrows = list(csv.DictReader(open(d / "fitness.csv"))); rows = allrows[-25:]
    m = lambda k: np.mean([float(r[k]) for r in rows])
    cfg = json.loads((d / "config.json").read_text()); L = json.loads((d / "layout.json").read_text())
    rec = np.load(d / "recording.npz")
    alive = rec["alive"].astype(bool); food = rec["food"].astype(np.float64); act = rec["action"].astype(int); pos = rec["pos"].astype(int)
    T, N = alive.shape
    df = np.diff(food, axis=0); meals = (df > 0).sum(0)
    absorbed = np.where(df > 0, df, 0).sum()
    fr = lambda a: (act[alive] == a).mean()
    cell = pos[..., 0] * 1000 + pos[..., 1]
    distinct, camp = [], []
    for a in range(N):
        life = alive[:, a].sum()
        for t0 in range(0, life - 99, 100):
            n = len(np.unique(cell[t0:t0 + 100, a])); distinct.append(n); camp.append(n <= 3)
    o = dict(zip(L["names"], zip(L["offsets"], L["sizes"]))); x = rec["x"]
    ge = x[:, :, o["ganglion_e"][0]:o["ganglion_e"][0] + o["ganglion_e"][1]][alive]
    spg = float(allrows[-1]["seconds"]) / len(allrows)
    print(f"{name:24s} {m('fed'):5.0f} {m('alive_ticks'):5.0f} {m('survivors'):5.1f} | {meals.mean():5.1f} {absorbed / max(m('eaten') * N, 1e-9):7.0%} {fr(0):5.2f} {fr(1):5.2f} {fr(2) + fr(3):5.2f} {fr(4):5.2f} {fr(5):5.2f} | {np.mean(distinct):10.1f} {np.mean(camp):5.0%} | {(ge > .95).mean():6.2f} {spg:5.2f}")
