"""Why a creature bites a bush it has been hurt by: the cells on the path to the bite, read from a run's recording.
    python scripts/probes/bite_trace.py <run dir> [PRIOR=2]
First tick of every visit to a poison bush (before the switch, tick 800) after at least PRIOR painful bites from that
type: mean rates of the cells that decide the bite, split by whether the creature bit on that tick; then the inputs
to the bite outputs (USE, EAT) by source region (rate x weight, summed per region), and to grasp and no_feed. Good
visits are shown for comparison. Weights: the recording's snapshot nearest the tick."""
import json, sys
from pathlib import Path
import numpy as np
from life import actions as A

run = Path(sys.argv[1])
PRIOR = int(next((a[6:] for a in sys.argv[2:] if a.startswith("PRIOR=")), 2))
sys.argv = sys.argv[:1]
src = open(Path(__file__).parent / "first_bite.py").read().split("\nif sys.argv[1:]:")[0]
ns = {}
exec(compile(src, "first_bite", "exec"), ns)
rec = np.load(run / "recording.npz")
L = json.load(open(run / "layout.json"))
reg = {n: slice(o, o + z) for n, o, z in zip(L["names"], L["offsets"], L["sizes"])}
SIGN = dict(zip(L["names"], L["sign"]))
x, ws, act = rec["x"].astype(np.float32), rec["w_snap"], rec["action"]
n, n_out = L["n"], L["n_out"]
out0 = n - n_out
bite_cols = [out0 + A.USE, out0 + A.EAT]
CELLS = [c for c in ("valence_av", "no_feed", "grasp", "no_touch", "valence_app", "hungry", "value_av", "value_app",
                     "flee", "rest", "settle", "fed") if c in reg]

# visits with their tick and creature: rerun first_bite's visit finder, keeping t0 and c

def visits_with_t(run):
    out = []
    names = json.load(open(run / "ruleset.json"))["names"]
    ate, pain, alive = rec["ate"].astype(int), rec["pain"], rec["alive"].astype(bool)
    grid, pos, dir_ = rec["grid"], rec["pos"].astype(int), rec["dir"].astype(int)
    T, N = act.shape
    H, W = grid.shape[1:]
    from life.world import DIRS
    full = {i: names.index(nm.replace("Wild Gooseberry Bush", "Gooseberry")) for i, nm in enumerate(names)
            if nm.endswith("Wild Gooseberry Bush") and "Empty" not in nm}
    hurt = {int(o): bool(np.mean(pain[:800][ate[:800] == o] > 0.5) > 0.5) for o in np.unique(ate[:800][ate[:800] > 0])}
    g = np.r_[grid[:1] * 0, grid[:-1]]
    fr = np.r_[pos[:1], pos[:-1]] + np.asarray(DIRS)[np.r_[dir_[:1], dir_[:-1]]]
    obj = g[np.arange(T)[:, None], np.clip(fr[..., 0], 0, H - 1), np.clip(fr[..., 1], 0, W - 1)]
    for c in range(N):
        prior = {}
        for t in range(1, 800):
            if ate[t, c] > 0 and pain[t, c] > 0.5:
                prior[int(ate[t, c])] = prior.get(int(ate[t, c]), 0) + 1
            o = int(obj[t, c])
            if alive[t, c] and o in full and obj[t - 1, c] != o and full[o] in hurt:
                out.append((t, c, hurt[full[o]], prior.get(full[o], 0)))
    return out


V = visits_with_t(run)
groups = {"poison, bit": [], "poison, did not bite": [], "good, bit": [], "good, did not bite": []}
for t, c, pz, pr in V:
    if pz and pr < PRIOR:
        continue
    b = act[t, c] in (A.USE, A.EAT)
    groups[f"{'poison' if pz else 'good'}, {'bit' if b else 'did not bite'}"].append((t, c))
print(f"{run}: first tick of visits before tick 800 (poison: after {PRIOR}+ painful bites from that type)")
print(f"  {'':22s} {'n':>5s} " + " ".join(f"{c[:11]:>11s}" for c in CELLS))
for k, v in groups.items():
    if not v:
        continue
    m = [np.mean([x[t, c, reg[cl]].mean() for t, c in v]) for cl in CELLS]
    print(f"  {k:22s} {len(v):5d} " + " ".join(f"{a:11.3f}" for a in m))


def inputs_to(cols, v):
    tot = {}
    for t, c in v:
        w = ws[min(t // 100, len(ws) - 1), c].astype(np.float32)
        look = [i for i, nm in enumerate(L["in_names"]) if ".app" in nm]
        other = [i for i in range(L["n_in"]) if i not in look]
        for r2, ix in (("in: looks", look), ("in: other", other)):
            tot[r2] = tot.get(r2, 0.0) + float((x[t, c, ix] @ w[ix][:, cols]).sum()) / len(cols)
        for r, sl in reg.items():
            if r in ("out", "in"):
                continue
            sg = SIGN[r] if SIGN[r] != 0 else 1.0        # Dale's law: the stored weights of a signed region are magnitudes
            tot[r] = tot.get(r, 0.0) + sg * float((x[t, c, sl] @ w[sl][:, cols]).sum()) / len(cols)
    return {r: s / len(v) for r, s in tot.items()}


for target, cols in (("bite outputs (USE, EAT)", bite_cols), ("grasp", list(range(reg["grasp"].start, reg["grasp"].stop))),
                     ("no_feed", list(range(reg["no_feed"].start, reg["no_feed"].stop))),
                     ("valence_av", list(range(reg["valence_av"].start, reg["valence_av"].stop)))):
    print(f"\n  input to {target} by source (rate x weight), largest first:")
    res = {k: inputs_to(cols, v) for k, v in groups.items() if v}
    order = sorted(res["poison, bit"], key=lambda r: -max(abs(res[k].get(r, 0)) for k in res))[:9]
    print(f"  {'':22s} " + " ".join(f"{r[:11]:>11s}" for r in order))
    for k, d in res.items():
        print(f"  {k:22s} " + " ".join(f"{d.get(r, 0):+11.3f}" for r in order))
