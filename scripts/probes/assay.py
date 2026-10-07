"""Step 1 of checking a learning rule, brain only: after scripted experience, what does an agent DO in a given situation?

Agents: the parent stage's final population on this stage's brain (inherited behaviour, learning at its starting
rate unless a variant says otherwise). Experience per round: a meal from a staple bush, from a novel poison bush,
from a novel good bush and from a staple bush again (three berries each; taste or pain follows each bite). Readouts
with learning frozen, from a rested brain that keeps only what was learned:
  before any experience, after the rounds, and after four more staple meals (does suspicion of the staple go away
  while the aversion to the poison stays?).
'innate poison' is the ancestral poison type that evolution already avoids: it shows what avoidance looks like.

    STAGE=1.5 uv run python scripts/probes/assay.py [variant] [rounds]      variants: see variants.py"""
import os, sys
import jax, jax.numpy as jnp, numpy as np
from life import brain as B
from life.experiments import stages as S
from life.run import load_population, latest_run, make_layout
from variants import variant

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "base"
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
NA = 24

s = S.STAGES[os.environ.get("STAGE", "1.5")]
exp = S.make_exp(s, variant(s.brain, VARIANT), "assay", 1, 0)
L = make_layout(exp); names = list(L.in_names); K = exp.vision.appearance_dim
pop = load_population(latest_run(S.STAGES[s.parent].name), exp, seed=0)
HANDS = "ingest" in L.names                      # the animal carries what it picks (pick, hold, eat)
rs, _ = s.build(exp)
look4 = S.lookalike_appearance(rs, 6, K)         # the fixed looks of the ancestral colour variants
bases = {o: S.colour_basis(o, K) for o in (S.BUSH, S.BERRY)}
PROGS = [r for r in ("approach", "grasp", "ingest", "reject") if r in L.names]


def novel(u):
    return {o: S.NOVEL_SIM * bases[o][0] + np.sqrt(1 - S.NOVEL_SIM ** 2) * (u @ bases[o][1]) for o in bases}


def obs(seen=None, near=0.8, held=None, taste=0.0, pain=0.0):
    o = np.zeros(L.n_in, np.float32)
    o[names.index("food")] = 0.5
    if "temperature" in names:
        o[names.index("temperature")] = 0.5
    if seen is not None:
        o[names.index("vis+0.hit")], o[names.index("vis+0.near")] = 1.0, near
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
    """Three berries from one bush; the outcome (taste or pain) arrives the tick after each bite."""
    out = dict(taste=3.0) if good else dict(pain=1.0)
    if HANDS:   # see bush / berry in hand / eaten, outcome
        seq = []
        for _ in range(3):
            seq += [obs(look[S.BUSH]), obs(look[S.BUSH], held=look[S.BERRY]), obs(look[S.BUSH], **out)]
    else:       # bite / outcome and next bite / ...
        seq = [obs(look[S.BUSH])] + [obs(look[S.BUSH], **out) for _ in range(3)]
    return seq + ([] if good else [obs(pain=0.3)]) + [obs()] * 4


step = jax.jit(lambda g, st, o: B.step(exp.brain, L, g, st, o, jnp.zeros(3), jax.random.PRNGKey(0))[0])


def readout(g, st, o):
    """Activity and neuromodulators start from zero and settle for 40 ticks with nothing in view; only the learned
    weights are kept. Then the stimulus is shown for three ticks."""
    frozen = g._replace(eta=jnp.zeros_like(g.eta))
    s2 = B.init_state(g, L)._replace(w=st.w)
    for _ in range(40):
        s2 = step(frozen, s2, obs())
    for _ in range(3):
        s2 = step(frozen, s2, o)
    h = (s2.x @ B.effective(L, s2.w) + g.b)[-L.n_out:]
    p = np.asarray(jax.nn.softmax(h * exp.brain.logit_gain / exp.brain.action_temperature))
    cells = lambda r, a, b: float(s2.x[L.region(r)][a:b].mean())
    out = [p[1], p[2] + p[3], p[4], p[5], cells("valence_app", 0, 2), cells("valence_av", 0, 2)]
    if HANDS:
        out += [cells("valence_app", 2, 3), cells("valence_av", 2, 3)]
    return np.array(out + [float(s2.x[L.region(r)].mean()) for r in PROGS])


eta = np.asarray(pop.eta); pid = np.asarray(L.proj_id)
print("learning rates (population mean): " + ", ".join(
    f"{n}: {eta[:, pid == i].mean():.3f}" for i, n in enumerate(L.proj_names) if (np.asarray(L.rule)[pid == i] > 0).any()))
rng = np.random.default_rng(0)
types = ["staple", "novel good", "novel poison", "innate poison"]
SITS = ["bush at a distance", "bush adjacent"] + (["berry in hand, bush adjacent", "berry in hand, nothing ahead"] if HANDS else [])
WHEN = ["before", "after", "then 4 staple meals"]
res = {(w, t, sit): [] for w in WHEN for t in types for sit in SITS}
for a in range(NA):
    g = jax.tree_util.tree_map(lambda x: x[a], pop)
    u = rng.normal(size=(2, K - 1)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    looks = {"staple": {o: bases[o][0] for o in bases}, "novel good": novel(u[0]), "novel poison": novel(u[1]),
             "innate poison": {o: look4[rs.local(S.variant(4)[o])] for o in bases}}
    st = B.init_state(g, L)
    for when in WHEN:
        if when == "after":
            for _ in range(ROUNDS):
                for t, good in (("staple", True), ("novel poison", False), ("novel good", True), ("staple", True)):
                    for o in meal(looks[t], good):
                        st = step(g, st, o)
        elif when == "then 4 staple meals":
            for _ in range(4):
                for o in meal(looks["staple"], True):
                    st = step(g, st, o)
        for t in types:
            res[(when, t, "bush at a distance")].append(readout(g, st, obs(looks[t][S.BUSH], near=0.4)))
            res[(when, t, "bush adjacent")].append(readout(g, st, obs(looks[t][S.BUSH])))
            if HANDS:
                res[(when, t, SITS[2])].append(readout(g, st, obs(looks[t][S.BUSH], held=looks[t][S.BERRY])))
                res[(when, t, SITS[3])].append(readout(g, st, obs(None, held=looks[t][S.BERRY])))
print(f"stage {s.key}, variant {VARIANT}, {ROUNDS} rounds of experience, {NA} agents (mean)")
print("value = appetitive / aversive activity of the value cells" + (" for what is seen and for what is in hand" if HANDS else ""))
head = f"{'':30s} {'':13s} {'':20s} {'P(FWD)':>7s} {'P(turn)':>8s} {'P(USE)':>7s} {'P(EAT)':>7s}   {'value':>10s}"
print(head + ("  value held" if HANDS else "") + "   " + " ".join(f"{r:>8s}" for r in PROGS))
for sit in SITS:
    for t in types:
        for when in WHEN:
            m = np.mean(res[(when, t, sit)], 0)
            line = f"{sit:30s} {t:13s} {when:20s} {m[0]:7.2f} {m[1]:8.2f} {m[2]:7.2f} {m[3]:7.2f}   {m[4]:5.2f}/{m[5]:.2f}"
            k = 6
            if HANDS:
                line += f"  {m[6]:5.2f}/{m[7]:.2f}"; k = 8
            print(line + "   " + " ".join(f"{v:8.2f}" for v in m[k:]))
    print()
