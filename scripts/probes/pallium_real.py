"""The transfer of pallium_sep.py measured on real lives: the newest population of a stage in its own world (one
recorded life, learning as evolved). For every poison bite (a lesson: pain on the next tick teaches what the look
inputs carried at the bite) and every later bite by the same animal, the overlap (v_lesson . v_later) /
(v_lesson . v_lesson) of the adapted look inputs (all eye columns) at the two bites. Mean by what the later bite
was: the same poison type, the other poison type, a good type. A good learner needs a high first and a low last.
The same through a random pallium per eye column (N cells, fan-in, k-WTA; see pallium_sep.py).
Lessons add up: 'summed' reads every bite, once both poison types have hurt the animal, against the sum of all
lessons so far (codes at the earlier poison bites); good / poison = mean read-out on good bites relative to that on
poison bites (0 = perfect, 1 = none, < 0 good looks safer), sd = spread of the read-out on poison bites relative to
its mean. With XOR looks single lessons barely touch good food, but two lessons cancel.

World and input options (environment): SIM=<x> novel similarity, XOR=1 XOR looks (berry_world novel_xor), both
only for the 1.8 world; STRENGTHS=0,0.5,1 the adapted look recomputed offline at several adaptation strengths
(look minus strength x its slow average; the animals behave as recorded, with their own strength).

    STAGE=1.8 [COLS=vis+0] [RAW=1] [LIVES=2] [SIM=0.6] [XOR=1] [STRENGTHS=0,0.5,1] uv run python scripts/probes/pallium_real.py"""
import os, fnmatch
from dataclasses import replace
import jax, numpy as np
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate, make_layout

s = S.STAGES[os.environ.get("STAGE", "1.8")]
if os.environ.get("SIM") or os.environ.get("XOR"):
    assert s.key == "1.8"
    s = replace(s, build=lambda exp: S.learning_world(exp, springs=S.SPRINGS_17, berries=S.LEARN_BERRIES,
                                                      novel_sim=float(os.environ.get("SIM", S.NOVEL_SIM)),
                                                      novel_xor=os.environ.get("XOR") == "1"))
exp = S.make_exp(s, s.brain, "pallium_real", 1, 0)
L = make_layout(exp); names = list(L.in_names)
pop = load_population(os.environ.get("RUN") or latest_run(s.name), exp, seed=0)
rs, fn = s.build(exp)
sim = make_simulate(exp, record=True)
cols = sorted({n.split(".")[0] for n in names if fnmatch.fnmatch(n, os.environ.get("COLS", "vis*") + ".app*")})   # COLS=vis+0: the eye ahead
idx = np.array([[names.index(f"{c}.app{k}") for k in range(exp.vision.appearance_dim)] for c in cols])   # [C, K]
K = idx.shape[1]
rng = np.random.default_rng(0)
RAW = os.environ.get("RAW") == "1"   # RAW=1: the look before adaptation (reconstructed from the recording)
STRENGTHS = [float(a) for a in os.environ["STRENGTHS"].split(",")] if os.environ.get("STRENGTHS") else [None]


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

res = {(st_, nm): {"same poison": [], "other poison": [], "good": [], "sum poison": [], "sum good": []}
       for st_ in STRENGTHS for nm in NETS}
counts = np.zeros(3)


def overlaps(x, ate, pain, st_):
    """Adds the overlaps of every poison bite with the later bites of the same animal to res[(st_, net)]."""
    for i in range(ate.shape[1]):
        ts = np.nonzero(ate[:, i] > 0)[0]
        if not len(ts):
            continue
        v = x[ts, i][:, idx]                         # [bites, C, K]
        typ = ate[ts, i]; bad = pain[typ] > 0
        for nm, f in NETS.items():
            c = f(v)
            w = np.cumsum(c * bad[:, None], 0)                     # lessons up to and including each bite
            first = [np.nonzero(typ == o)[0][0] for o in np.unique(typ[bad])]
            if len(first) == 2:
                for b in range(max(first) + 1, len(ts)):
                    res[(st_, nm)]["sum poison" if bad[b] else "sum good"].append(w[b - 1] @ c[b])
            for a in np.nonzero(bad)[0]:
                later = np.arange(a + 1, len(ts))
                d = c[a] @ c[a]
                if not len(later) or d < 1e-9:
                    continue
                t = c[later] @ c[a] / d
                same = typ[later] == typ[a]
                for key, m in (("same poison", same), ("other poison", bad[later] & ~same), ("good", ~bad[later])):
                    res[(st_, nm)][key] += list(t[m])
                    if nm == "look itself" and st_ == STRENGTHS[0]:
                        counts[["same poison", "other poison", "good"].index(key)] += m.sum()


for life in range(int(os.environ.get("LIVES", "2"))):
    kr, ks = jax.random.split(jax.random.PRNGKey(100 + life))
    rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
    pain = np.asarray(rules.pain_value)
    st, rec = sim(rules, pop, ks)
    ate = np.asarray(rec["ate"]); T, N = ate.shape
    x0 = np.asarray(rec["x"][:, :, :L.n_in], np.float32)
    # undo the adaptation: obs = raw - a tr, tr' = tau tr + (1 - tau) raw (tr = 0 at birth), a = the brain's strength
    a0, tau = float(np.asarray(L.in_adapt).max()), exp.brain.in_trace_tau
    tr = np.zeros_like(x0[0]); raw = np.zeros_like(x0)
    for t in range(T):
        raw[t] = x0[t] + a0 * tr * (np.asarray(L.in_adapt) > 0)
        tr = tau * tr + (1 - tau) * raw[t]
    trs = raw - x0 if a0 == 0 else (raw - x0) / a0
    for st_ in STRENGTHS:
        overlaps(raw if RAW else x0 if st_ is None else raw - st_ * trs, ate, pain, st_)

print(f"stage {s.key}{' (look before adaptation)' if RAW else ''}: later bites after a poison bite, same poison {counts[0]:.0f}, other poison {counts[1]:.0f}, "
      f"good {counts[2]:.0f}")
print(f"  {'':34s}{'same poison':>13s}{'other poison':>14s}{'good':>8s}{'good / same':>13s}   sd same / other / good"
      f"   summed: good / poison, sd")
for (st_, nm), r in res.items():
    if st_ is not None and nm == "look itself":
        print(f"  adaptation strength {st_}")
    m = {k_: np.mean(v_) for k_, v_ in r.items()}; sd = {k_: np.std(v_) for k_, v_ in r.items()}
    print(f"  {nm:34s}{m['same poison']:13.2f}{m['other poison']:14.2f}{m['good']:8.2f}{m['good'] / m['same poison']:13.2f}"
          f"   {sd['same poison']:.2f} / {sd['other poison']:.2f} / {sd['good']:.2f}"
          f"   {m['sum good'] / m['sum poison']:8.2f}, {sd['sum poison'] / m['sum poison']:.2f}")
