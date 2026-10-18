"""Napkin model, no brain and no world: how small must the transfer of a lesson be for learning to pay?

A life is a random sequence of bush visits with the counts of the 1.8 world (`lessons.py`): 24 good visits (two good
types), 15 poison visits (two poison types, 1.3 bites each). Each bitten poison visit is a lesson: the aversion to
type j grows by eta * bites * T(poison type, j), with T = 1 for the same type, T_o for the other poison, T_g for a
good type. Lessons add up and never fade (the learned synapses do not decay). A visit is bitten with probability
sigmoid(SLOPE * (a - aversion)); a is the appetite, which evolution can raise to offset a growing aversion.
A bitten good visit is worth V_g, a bitten poison visit costs 1.3 * 2 food units; a refused visit is worth 0.
V_g = 7.6 bites * 2 = 15.2 food units, times a factor below 1 when food is plentiful (71% of bites in 1.8 land on a
stomach over 90% full: a refused bush costs some search, not its whole content).
SD: the look differs from bite to bite, so a lesson's overlap with a later visit is T + SD * N(0, 1) per lesson and
visit (relative to the mean overlap with the same poison; in 1.8 about 1.0, `pallium_real.py`: 0.31 +- 0.33 on the
same poison, 0.08 +- 0.29 on good food).
Best learner: grid over a and eta; best non-learner: eta = 0 (it bites everything). Gain in % of the non-learner's
intake.

    uv run python scripts/probes/learn_worth.py [SLOPE=3] [SD=0] [TO=all | T_g | <value>]"""
import sys
import numpy as np

ARGS = dict(a.split("=") for a in sys.argv[1:])
SLOPE, SD = float(ARGS.get("SLOPE", 3)), float(ARGS.get("SD", 0))
NG, NP, PB, VP = 24, 15, 1.3, 2.6
LIVES = 400
rng = np.random.default_rng(0)
# visit sequences: type 0, 1 poison, 2, 3 good
seqs = np.array([rng.permutation([0] * 8 + [1] * 7 + [2] * 12 + [3] * 12) for _ in range(LIVES)])
A_GRID = np.linspace(-1, 6, 29); ETA_GRID = np.concatenate([[0], np.geomspace(0.02, 5, 30)])


def life_value(T, vg, a, eta):
    """Mean value over the lives, vectorised over (a, eta) grids."""
    aa, ee = np.meshgrid(a, eta, indexing="ij")
    av = np.zeros((LIVES, 4) + aa.shape); sq = np.zeros_like(av)   # sum of lesson strengths and of their squares
    total = np.zeros(aa.shape)
    for t in range(seqs.shape[1]):
        j = seqs[:, t]
        noise = SD * np.sqrt(sq[np.arange(LIVES), j]) * rng.normal(size=(LIVES,) + aa.shape)
        p = 1 / (1 + np.exp(-SLOPE * (aa - av[np.arange(LIVES), j] - noise)))
        bad = j < 2
        total += np.where(bad[:, None, None], -VP * p, vg * p).mean(0)
        # expected lesson (bite probability times bites) onto every type
        les = (bad[:, None, None] * p * PB)[:, None] * T[j][:, :, None, None] * ee
        av = av + les
        sq = sq + (bad[:, None, None] * p * PB * ee)[:, None] ** 2
    return total


def table(t_other):
    print(f"\nother poison T_o = {t_other}; rows T_g (transfer to a good type), columns worth of a good visit:"
          f" gain of the best learner over the best non-learner, % of intake (best eta, appetite a)")
    facs = (1.0, 0.5, 0.25)
    print(f"  {'T_g':>5s}" + "".join(f"{'V_g x ' + str(f):>26s}" for f in facs))
    for tg in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        to = tg if t_other == "T_g" else t_other
        T = np.array([[1, to, tg, tg], [to, 1, tg, tg], [0, 0, 0, 0], [0, 0, 0, 0]], float)
        row = ""
        for f in facs:
            vg = 15.2 * f
            v = life_value(T, vg, A_GRID, ETA_GRID)
            base = v[:, 0].max()
            i, k = np.unravel_index(v.argmax(), v.shape)
            row += f"{100 * (v[i, k] - base) / (NG * vg):10.1f}% (eta {ETA_GRID[k]:4.2f}, a {A_GRID[i]:4.1f})"
        print(f"  {tg:5.1f}" + row, flush=True)


print(f"bite slope {SLOPE}, spread {SD}; a perfect learner (one bite per poison type) would save "
      f"{100 * (NP - 2) * PB * 2 / (NG * 15.2):.0f}% of intake at V_g x 1")
for to in ("T_g", 0.0, 1.0) if ARGS.get("TO", "all") == "all" else [ARGS["TO"] if ARGS["TO"] == "T_g" else float(ARGS["TO"])]:
    table(to)
