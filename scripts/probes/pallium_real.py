"""The transfer of pallium_sep.py measured on real lives: the newest population of a stage in its own world (one
recorded life, learning as evolved). For every poison bite (a lesson: pain on the next tick teaches what the look
inputs carried at the bite) and every later bite by the same animal, the overlap (v_lesson . v_later) /
(v_lesson . v_lesson) of the adapted look inputs (all eye columns) at the two bites. Mean by what the later bite
was: the same poison type, the other poison type, a good type. A good learner needs a high first and a low last.
The same through a random pallium per eye column (N cells, fan-in, k-WTA; see pallium_sep.py).

    STAGE=1.8 [RAW=1] [LIVES=2] uv run python scripts/probes/pallium_real.py"""
import os, fnmatch
import jax, numpy as np
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate, make_layout

s = S.STAGES[os.environ.get("STAGE", "1.8")]
exp = S.make_exp(s, s.brain, "pallium_real", 1, 0)
L = make_layout(exp); names = list(L.in_names)
pop = load_population(os.environ.get("RUN") or latest_run(s.name), exp, seed=0)
rs, fn = s.build(exp)
sim = make_simulate(exp, record=True)
cols = sorted({n.split(".")[0] for n in names if fnmatch.fnmatch(n, "vis*.app*")})
idx = np.array([[names.index(f"{c}.app{k}") for k in range(exp.vision.appearance_dim)] for c in cols])   # [C, K]
K = idx.shape[1]
rng = np.random.default_rng(0)
RAW = os.environ.get("RAW") == "1"   # RAW=1: the look before adaptation (reconstructed from the recording)


def pallium(n, fan, k):
    w = np.zeros((len(cols), n, K))
    for c in range(len(cols)):
        for i in range(n):
            w[c, i, rng.choice(K, fan, replace=False)] = rng.normal(size=fan)
    def code(v):                     # v [..., C, K] -> [..., C * n]
        h = np.einsum("...ck,cnk->...cn", v, w)
        out = np.zeros_like(h)
        top = np.argsort(-h, axis=-1)[..., :k]
        np.put_along_axis(out, top, np.maximum(np.tanh(np.take_along_axis(h, top, -1)), 0), -1)
        return out.reshape(*v.shape[:-2], -1)
    return code


NETS = {"look itself": lambda v: v.reshape(*v.shape[:-2], -1)}
for spec in ((64, 3, 4), (64, 3, 8), (64, 4, 8), (128, 3, 16)):
    NETS[f"pallium N={spec[0]} fan={spec[1]} k={spec[2]}"] = pallium(*spec)

res = {nm: {"same poison": [], "other poison": [], "good": []} for nm in NETS}
counts = np.zeros(3)
for life in range(int(os.environ.get("LIVES", "2"))):
    kr, ks = jax.random.split(jax.random.PRNGKey(100 + life))
    rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
    pain = np.asarray(rules.pain_value)
    st, rec = sim(rules, pop, ks)
    ate = np.asarray(rec["ate"]); T, N = ate.shape
    x = np.asarray(rec["x"][:, :, :L.n_in], np.float32)
    if RAW:   # undo the adaptation: obs = raw - tr, tr' = tr + (1 - tau) obs, tr = 0 at birth
        tr = np.cumsum(np.concatenate([np.zeros_like(x[:1]), x[:-1]]), 0) * (1 - exp.brain.in_trace_tau)
        x = x + tr * np.asarray(L.in_adapt)
    for i in range(N):
        ts = np.nonzero(ate[:, i] > 0)[0]
        if not len(ts):
            continue
        v = x[ts, i][:, idx]                         # [bites, C, K]
        typ = ate[ts, i]; bad = pain[typ] > 0
        for nm, f in NETS.items():
            c = f(v)
            for a in np.nonzero(bad)[0]:
                later = np.arange(a + 1, len(ts))
                if not len(later):
                    continue
                d = c[a] @ c[a]
                if d < 1e-9:
                    continue
                t = c[later] @ c[a] / d
                same = typ[later] == typ[a]
                for key, m in (("same poison", same), ("other poison", bad[later] & ~same), ("good", ~bad[later])):
                    res[nm][key] += list(t[m])
                    if nm == "look itself":
                        counts[["same poison", "other poison", "good"].index(key)] += m.sum()
print(f"stage {s.key}{' (look before adaptation)' if RAW else ''}: later bites after a poison bite, same poison {counts[0]:.0f}, other poison {counts[1]:.0f}, "
      f"good {counts[2]:.0f}")
print(f"  {'':34s}{'same poison':>13s}{'other poison':>14s}{'good':>8s}{'good / same':>13s}")
for nm, r in res.items():
    m = {k_: np.mean(v_) for k_, v_ in r.items()}
    print(f"  {nm:34s}{m['same poison']:13.2f}{m['other poison']:14.2f}{m['good']:8.2f}{m['good'] / m['same poison']:13.2f}")
