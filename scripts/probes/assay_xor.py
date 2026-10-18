"""The assay (assay.py) for the pallium stage, brain only: XOR looks. Each agent of the parent's final population, on
this stage's brain, eats from four novel bushes whose looks are the corners of two colour directions drawn for it:
poison ++ and --, good +- and -+ (as berry_world novel_xor), three berries per meal, pain or taste on the tick after
each bite, in the order poison ++, good +-, poison --, good -+, ROUNDS times. Readouts as in assay.py (a rested brain
that keeps only what it learned, the bush shown adjacent for three ticks): mean aversive activity and P(bite) per
type, before and after, with the learning of the look synapses and of the pallium synapses on or off.

    STAGE=2.2 [PAL_ETA=0.03] uv run python scripts/probes/assay_xor.py [rounds] [SIM]"""
import os, sys
import jax, jax.numpy as jnp, numpy as np
from life import brain as B
from life.experiments import stages as S
from life.run import load_population, latest_run, make_layout

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
NA = 24
s = S.STAGES[os.environ.get("STAGE", "2.2")]
SIM = float(sys.argv[2]) if len(sys.argv) > 2 else S.PALLIUM_SIM
exp = S.make_exp(s, s.brain, "assay", 1, 0)
L = make_layout(exp); names = list(L.in_names); K = exp.vision.appearance_dim
pop = load_population(latest_run(S.STAGES[s.parent].name), exp, seed=0)
if os.environ.get("PAL_ETA"):   # a starting learning rate of the pallium synapses other than the stage's
    _pid = np.asarray(make_layout(exp).proj_id); _i = list(make_layout(exp).proj_names).index("pallium->valence_av")
    pop = pop._replace(eta=jnp.where(jnp.asarray(_pid == _i)[None], float(os.environ["PAL_ETA"]), pop.eta))
base, comp = S.colour_basis(S.BUSH, K)
pid = np.asarray(L.proj_id)
PROJ = {n: i for i, n in enumerate(L.proj_names)}
LOOK = [i for n, i in PROJ.items() if n.startswith("in->valence_av") and (np.asarray(L.rule)[pid == i] > 0).any()]
PAL = [i for n, i in PROJ.items() if n == "pallium->valence_av"]
TYPES = ["poison ++", "poison --", "good +-", "good -+"]
SIGNS = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]]) / np.sqrt(2)


def obs(seen=None, taste=0.0, pain=0.0):
    o = np.zeros(L.n_in, np.float32)
    o[names.index("food")] = 0.5
    if "temperature" in names:
        o[names.index("temperature")] = 0.5
    if seen is not None:
        o[names.index("vis+0.hit")], o[names.index("vis+0.near")] = 1.0, 0.8
        for k in range(K):
            o[names.index(f"vis+0.app{k}")] = seen[k]
    o[names.index("taste")] = taste
    o[names.index("pain")] = pain
    return jnp.asarray(o)


def meal(look, good):
    out = dict(taste=3.0) if good else dict(pain=1.0)
    return [obs(look)] + [obs(look, **out) for _ in range(3)] + ([] if good else [obs(pain=0.3)]) + [obs()] * 4


step = jax.jit(lambda g, st, o: B.step(exp.brain, L, g, st, o, jnp.zeros(3), jax.random.PRNGKey(0))[0])


def readout(g, st, o):
    frozen = g._replace(eta=jnp.zeros_like(g.eta))
    s2 = B.init_state(g, L)._replace(w=st.w)
    for _ in range(40):
        s2 = step(frozen, s2, obs())
    for _ in range(3):
        s2 = step(frozen, s2, o)
    h = (s2.x @ B.effective(L, s2.w) + g.b)[-L.n_out:]
    p = np.asarray(jax.nn.softmax(h * exp.brain.logit_gain / exp.brain.action_temperature))
    return np.array([float(s2.x[L.region("valence_av")].mean()), p[4] + p[5]])


def off(g, projs):
    eta = np.asarray(g.eta).copy()
    for i in projs:
        eta[pid == i] = 0.0
    return g._replace(eta=jnp.asarray(eta))


rng = np.random.default_rng(0)
CONDS = {"both learn": [], "look only": PAL, "pallium only": LOOK, "neither": LOOK + PAL}
res = {(c, w, t): [] for c in CONDS for w in ("before", "after") for t in TYPES}
for a in range(NA):
    g0 = jax.tree_util.tree_map(lambda x: x[a], pop)
    q, _ = np.linalg.qr(rng.normal(size=(K - 1, 2)))
    looks = {t: SIM * base + np.sqrt(1 - SIM ** 2) * ((SIGNS[i] @ q.T) @ comp) for i, t in enumerate(TYPES)}
    for c, projs in CONDS.items():
        g = off(g0, projs)
        st = B.init_state(g, L)
        for t in TYPES:
            res[(c, "before", t)].append(readout(g, st, obs(looks[t])))
        for _ in range(ROUNDS):
            for t in ("poison ++", "good +-", "poison --", "good -+"):
                for o in meal(looks[t], t.startswith("good")):
                    st = step(g, st, o)
        for t in TYPES:
            res[(c, "after", t)].append(readout(g, st, obs(looks[t])))
eta = np.asarray(pop.eta)
print(f"stage {s.key}, XOR looks at similarity {SIM}, {ROUNDS} rounds, {NA} agents; learning rates: look "
      f"{eta[:, np.isin(pid, LOOK)].mean():.2f}, pallium {eta[:, np.isin(pid, PAL)].mean():.2f}")
print(f"  {'':14s}{'':8s}" + "".join(f"{t:>20s}" for t in TYPES) + "      (aversive activity / P(bite))")
for c in CONDS:
    for w in ("before", "after"):
        print(f"  {c:14s}{w:8s}" + "".join(f"{np.mean(res[(c, w, t)], 0)[0]:12.2f} / {np.mean(res[(c, w, t)], 0)[1]:.2f}"
                                          for t in TYPES))
