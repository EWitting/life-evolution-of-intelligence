"""Before the pallium stage, brain-free: how far would a sparse random expansion of the look (the pallium) keep a
lesson about one novel bush type off the others? A lesson of the association rule is dW ~ code of the poison's look,
so the aversion it leaves on another type is (code_P . code_G) / (code_P . code_P): the transfer. 1 = the other type
is as suspect as the poison, 0 = untouched.

Looks as in the world (berry_world, novel_looks): one eye column, 8 features, novel bush looks
sim * gooseberry + sqrt(1 - sim^2) * random direction. Three inputs to the learned synapses:
  raw          the look itself
  adapted 0.5  the look minus half the mean of the four novel looks (an animal with a bush in view half the time)
  adapted 1    the look minus the mean of the four novel looks (an ideal habituation)
Pallium: N cells, each reading `fan` random look features with weights N(0, 1) (Dale's law for the input is not
imposed: an adapted feature below its average already acts as inhibition), k-WTA keeps the k strongest, rate
tanh(h) of the winners.

    uv run python scripts/probes/pallium_sep.py [sim ...]        default 0.8 0.5"""
import sys
import numpy as np
from life.experiments import stages as S

K = S.APPEARANCE
SIMS = [float(a) for a in sys.argv[1:]] or [S.NOVEL_SIM, 0.5]
LIVES, NETS = 200, 8
base, rest = S.colour_basis(S.BUSH, K)
rng = np.random.default_rng(0)


def looks(sim, n=4):
    u = rng.normal(size=(n, K - 1)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    return sim * base + np.sqrt(1 - sim ** 2) * (u @ rest)


def pallium(n, fan, k):
    w = np.zeros((n, K))
    for i in range(n):
        w[i, rng.choice(K, fan, replace=False)] = rng.normal(size=fan)
    def code(x):
        h = x @ w.T
        out = np.zeros_like(h)
        idx = np.argsort(-h, axis=-1)[..., :k]
        np.put_along_axis(out, idx, np.maximum(np.tanh(np.take_along_axis(h, idx, -1)), 0), -1)
        return out
    return code


def transfer(c):
    """Mean over ordered pairs of types (lesson on i, effect on j != i) of (c_i . c_j) / (c_i . c_i)."""
    g = c @ c.T; d = np.diag(g)
    t = g / np.maximum(d[:, None], 1e-9)
    return t[~np.eye(len(c), dtype=bool)].mean()


INPUTS = {"raw": 0.0, "adapted 0.5": 0.5, "adapted 1": 1.0}
NETWORKS = [("look itself", None)] + [(f"N={n:3d} fan={f} k={k:2d}", (n, f, k))
                                       for n in (32, 64, 128) for f in (2, 3, 4) for k in (n // 16, n // 8, n // 4)]
for sim in SIMS:
    lv = [looks(sim) for _ in range(LIVES)]
    print(f"\nnovel similarity {sim}: transfer of a lesson to another novel type (mean over {LIVES} lives"
          f"{f', {NETS} random pallia' if NETS else ''})")
    print(f"  {'':24s}" + "".join(f"{nm:>13s}" for nm in INPUTS))
    for name, spec in NETWORKS:
        row = []
        for a in INPUTS.values():
            xs = [x - a * x.mean(0) for x in lv]
            if spec is None:
                row.append(np.mean([transfer(x) for x in xs]))
            else:
                codes = [pallium(*spec) for _ in range(NETS)]
                row.append(np.mean([transfer(cd(x)) for cd in codes for x in xs[:LIVES // 4]]))
        print(f"  {name:24s}" + "".join(f"{v:13.2f}" for v in row))
