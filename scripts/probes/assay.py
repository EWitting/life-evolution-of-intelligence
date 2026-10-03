"""Step 1 of the learning chain, brain only: after scripted experience, what does an agent DO in a given situation?
Agents: the stage-1.4 population on the 1.5 brain (inherited behaviour, learning at the starting rate).
Experience per round: 3 staple berries (taste), 3 novel-good berries (taste), 3 novel-poison berries (sickness after
`delay` ticks). Readout with learning frozen: action probabilities when a bush of each type is straight ahead with an
empty hand, and when its berry is in hand. 'innate poison' is the ancestral poison type that evolution already avoids.
    STAGE=x.learn uv run python scripts/probes/assay.py [variant] [delay] [rounds]      variants: see variants.py"""
import os, sys
from dataclasses import replace
import jax, jax.numpy as jnp, numpy as np
from life import brain as B
from life.experiments import stages as S
from life.run import load_population, latest_run, make_layout
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "base"
DELAY = int(sys.argv[2]) if len(sys.argv) > 2 else S.SICK_DELAY
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
NA = 24

from variants import variant

s = S.STAGES[os.environ.get("STAGE", "1.5")]
exp = S.make_exp(s, variant(s.brain, VARIANT), "assay", 1, 0)
L = make_layout(exp); names = list(L.in_names); K = exp.vision.appearance_dim
pop = load_population(latest_run(S.STAGES[s.parent].name), exp, seed=0)
ACT = ["NOOP", "FWD", "LEFT", "RIGHT", "USE", "EAT", "VOC"]
rs, _ = s.build(exp)
look4 = S.lookalike_appearance(rs, 6, K)        # the fixed looks of the ancestral colour variants
bases = {o: S.colour_basis(o, K) for o in (S.BUSH, S.BERRY)}


def novel(u):
    return {o: S.NOVEL_SIM * bases[o][0] + np.sqrt(1 - S.NOVEL_SIM ** 2) * (u @ bases[o][1]) for o in bases}


def obs(seen=None, held=None, taste=0.0, pain=0.0):
    o = np.zeros(L.n_in, np.float32)
    o[names.index("food")] = 0.5
    if "temperature" in names:
        o[names.index("temperature")] = 0.5
    if seen is not None:
        o[names.index("vis+0.hit")], o[names.index("vis+0.near")] = 1.0, 0.8
        for k in range(K):
            o[names.index(f"vis+0.app{k}")] = seen[k]
    if held is not None:
        o[names.index("held")] = 1.0
        for k in range(K):
            o[names.index(f"held_app{k}")] = held[k]
    o[names.index("taste")] = taste
    o[names.index("pain")] = pain
    return jnp.asarray(o)


def meal(look, good):
    """three berries from one bush: see bush / hold berry (bush still ahead) / eaten"""
    seq = []
    for _ in range(3):
        seq += [obs(look[S.BUSH]), obs(look[S.BUSH], look[S.BERRY]), obs(look[S.BUSH], taste=3.0 if good else 0.0)]
    if good:
        return seq + [obs()] * 4
    return seq + [obs()] * max(0, DELAY - 1) + [obs(pain=1.0), obs(pain=0.3)] + [obs()] * 4


step = jax.jit(lambda g, st, o: B.step(exp.brain, L, g, st, o, jnp.zeros(3), jax.random.PRNGKey(0))[0])


PROGS = [r for r in ("approach", "grasp", "ingest", "reject") if r in L.names]   # motor programmes, if the brain has them


def probs(g, st, o):
    """Readout from a rested brain that keeps only what was learned (weights, depression): activity and
    neuromodulators start from zero and settle for 40 ticks with nothing in view, then the stimulus is shown."""
    frozen = g._replace(eta=jnp.zeros_like(g.eta))
    s2 = B.init_state(g, L)._replace(w=st.w)
    for _ in range(40):
        s2 = step(frozen, s2, obs())
    for _ in range(3):
        s2 = step(frozen, s2, o)
    h = (s2.x @ B.effective(L, s2.w) + g.b)[-L.n_out:]
    p = np.asarray(jax.nn.softmax(h * exp.brain.logit_gain / exp.brain.action_temperature))
    cells = lambda r, a, b: float(s2.x[L.region(r)][a:b].mean())
    out = [p[1], p[2] + p[3], p[4], p[5], cells("valence_app", 0, 2), cells("valence_av", 0, 2),
           cells("valence_app", 2, 3), cells("valence_av", 2, 3)]
    return np.array(out + [float(s2.x[L.region(r)].mean()) for r in PROGS])


w0 = np.asarray(pop.w0); design = np.nan_to_num(np.asarray(L.w_init)); pidm = np.asarray(L.proj_id)
print("evolved strength of hard-wired projections, as a multiple of the designed value (population mean):")
for i, n in enumerate(L.proj_names):
    m = (pidm == i) & (np.asarray(L.tune) > 0)
    if m.any():
        print(f"  {n:44s} designed {design[m].mean():+.1f}   x{(np.abs(w0[:, m]).mean() / abs(design[m].mean())):.2f}")
eta = np.asarray(pop.eta)
rng = np.random.default_rng(0)
types = ["staple", "novel good", "novel poison", "innate poison"]
SITS = ("bush ahead", "berry in hand, bush ahead", "berry in hand, nothing ahead")
res = {(w, t, sit): [] for w in ("before", "after") for t in types for sit in SITS}
for a in range(NA):
    g = jax.tree_util.tree_map(lambda x: x[a], pop)
    u = rng.normal(size=(2, K - 1)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    looks = {"staple": {o: bases[o][0] for o in bases}, "novel good": novel(u[0]), "novel poison": novel(u[1]),
             "innate poison": {o: look4[rs.local(S.variant(4)[o])] for o in bases}}
    st = B.init_state(g, L)
    for when in ("before", "after"):
        if when == "after":
            for _ in range(ROUNDS):
                for t, good in (("staple", True), ("novel poison", False), ("novel good", True), ("staple", True)):
                    for o in meal(looks[t], good):
                        st = step(g, st, o)
        for t in types:
            res[(when, t, "bush ahead")].append(probs(g, st, obs(looks[t][S.BUSH])))
            res[(when, t, SITS[1])].append(probs(g, st, obs(looks[t][S.BUSH], looks[t][S.BERRY])))
            res[(when, t, SITS[2])].append(probs(g, st, obs(None, looks[t][S.BERRY])))
print(f"variant {VARIANT}, sickness delay {DELAY}, {ROUNDS} rounds of experience, {NA} agents (mean)")
print("value = appetitive / aversive activity of the value cells for what is seen (cells 0-1) and for what is in hand (cell 2)")
print(f"{'':50s} {'P(FWD)':>7s} {'P(turn)':>8s} {'P(USE)':>7s} {'P(EAT)':>7s}   {'value seen':>11s} {'value held':>11s}   " + " ".join(f"{r:>8s}" for r in PROGS))
for sit in SITS:
    for t in types:
        for when in ("before", "after"):
            m = np.mean(res[(when, t, sit)], 0)
            print(f"{sit:30s} {t:13s} {when:6s} {m[0]:7.2f} {m[1]:8.2f} {m[2]:7.2f} {m[3]:7.2f}   {m[4]:5.2f}/{m[5]:.2f}  {m[6]:5.2f}/{m[7]:.2f}   "
                  + " ".join(f"{v:8.2f}" for v in m[8:]))
    print()
