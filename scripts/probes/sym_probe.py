"""Appetite learning from dopamine (stages.appetite_learning, B24S) on evolved 2.3c populations: lab tests, then
generation 0 in the 2.3c world, the brain as frozen against the symmetric one. The new synapses (look and pallium ->
valence_app, taught by `da`) start with the learning genes of their aversive mirror (look and pallium -> valence_av,
taught by `da_neg`), synapse by synapse, times ETA (default 1).
    python scripts/probes/sym_probe.py <run dir> [...] [ETA=f] [WORLDS=16] [LAB=0] [WORLD=0] [VARIANTS=frozen,appetite]
VARIANTS: frozen (2.3c), relay (pain -> us_pain -> habenula, stages.pain_relay), appetite (stages.appetite_learning),
both (B24S); the first is the reference of the paired differences.
Lab (24 creatures, world-like lessons: one tick of outcome, approach of 8 ticks; readout on the first tick at the
mouth after 8 from afar): extinction of a pain (P(bite) and aversive drive after 6 painful meals and 12 without);
appetite (6 good meals of a new look, then 12 without taste); XOR look-alikes (pain on X++ and X--, 3 good meals of
X+- and X-+ per round, 6 rounds). World: fitness, lifetime and good / poison berries per creature before and after
the switch, paired over the same worlds."""
import sys
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
from life.config import ExperimentConfig
from life.experiments import stages as S
from life.labtests import conditioning
from life.run import load_population, make_layout, make_simulate

arg = lambda k, d: next((a[len(k) + 1:] for a in sys.argv[1:] if a.startswith(k + "=")), d)
ETA, WORLDS = float(arg("ETA", 1.0)), int(arg("WORLDS", 16))
ST = S.STAGES["2.3c"]


BRAINS = {"frozen": ST.brain, "relay": S.pain_relay(ST.brain), "appetite": S.appetite_learning(ST.brain),
          "both": S.B24S}


def population(run, name):
    sym = name in ("appetite", "both")
    s = replace_brain(ST, BRAINS[name])
    exp = S.make_exp(s, s.brain, "sym", 1, 0)
    pop = load_population(run, exp)
    L = make_layout(exp)
    plastic = (np.asarray(L.rule) > 0) & (np.asarray(L.allowed) > 0)       # dense_plastic, as the stage runs
    pop = pop._replace(mask=jnp.where(jnp.asarray(plastic)[None], 1.0, pop.mask))
    if sym:
        idx = lambda r: np.arange(L.n)[L.region(r)]
        app, av = idx("valence_app"), idx("valence_av")
        rows = np.concatenate([[L.in_names.index(f) for f in L.in_names if f.startswith("vis") and ".app" in f],
                               idx("pallium")])
        genes = {k: np.array(getattr(pop, k)) for k in ("eta", "A", "B", "C", "D")}
        for k, g in genes.items():
            g[np.ix_(np.arange(g.shape[0]), rows, app)] = g[np.ix_(np.arange(g.shape[0]), rows, av)] * (ETA if k == "eta" else 1)
        pop = pop._replace(**{k: jnp.asarray(v) for k, v in genes.items()})
    return exp, pop, s


def replace_brain(s, b):
    from dataclasses import replace
    return replace(s, brain=b)


WL = dict(outcome_ticks=1, linger=0.0, lead=8, approach=True, test_far=8, test_ticks=1)


def lab(exp, pop):
    A = {"vis+0": "A"}
    r = conditioning(exp, pop, dict(train=[("pain", A, 6), ("none", A, 12)], tests={"A": A}, **WL), S.PALLIUM_SIM)
    a = r["tests"]["A"]
    ext = f"P(bite) {a[0, :, 1].mean():.2f} -> {a[6, :, 1].mean():.2f} -> {a[18, :, 1].mean():.2f}"
    r = conditioning(exp, pop, dict(train=[("taste", A, 6), ("none", A, 12)], tests={"A": A}, **WL), S.PALLIUM_SIM)
    a = r["tests"]["A"]
    appetite = f"P(bite) {a[0, :, 1].mean():.2f} -> {a[6, :, 1].mean():.2f} -> {a[18, :, 1].mean():.2f}"
    rnd = [("pain", {"vis+0": "X++"}, 1), ("taste", {"vis+0": "X+-"}, 3), ("pain", {"vis+0": "X--"}, 1),
           ("taste", {"vis+0": "X-+"}, 3)]
    looks = ("X++", "X--", "X+-", "X-+")
    r = conditioning(exp, pop, dict(train=rnd * 6, tests={c: {"vis+0": c} for c in looks}, **WL), S.PALLIUM_SIM)
    t = {c: r["tests"][c][:, :, 1].mean(1) for c in looks}
    ends = np.cumsum([0] + [e[2] for e in rnd] * 6)[::len(rnd)]
    xor = " ".join(f"{(t['X++'][i] + t['X--'][i]) / 2:.2f}/{(t['X+-'][i] + t['X-+'][i]) / 2:.2f}" for i in ends[[0, 1, 3, 6]])
    return ext, appetite, xor


def world(exp, pop, s):
    rs, fn = s.build(exp)
    if s.schedule:
        fn = s.schedule(fn)
    sim = jax.jit(make_simulate(exp, record=False))
    berry = [rs.local(S.variant(t)[S.BERRY]) for t in range(6)]
    acc = []
    for k in jax.random.split(jax.random.PRNGKey(31), WORLDS):
        kr, ks = jax.random.split(k)
        rules = jax.tree_util.tree_map(lambda a: a[0], fn(99, kr))           # a generation after the calm start
        pv = np.asarray(rules.pain_value)
        pv = pv if pv.ndim == 2 else np.stack([pv, pv])
        st, _ = sim(rules, pop, ks)
        e = np.asarray(st["eats"]); n = e.shape[0]
        eat = lambda h, bad: sum(e[:, h, berry[t]].sum() for t in range(6) if (pv[h, berry[t]] > 0) == bad) / n
        acc.append([float((s.fitness or S.default_fitness)(st).mean()), float(st["alive_ticks"].mean()),
                    eat(0, False), eat(0, True), eat(1, False), eat(1, True)])
    return np.array(acc)


for run in [Path(a) for a in sys.argv[1:] if "=" not in a]:
    print(f"\n{run}  (new synapses' learning rates x{ETA})", flush=True)
    base = None
    for name in arg("VARIANTS", "frozen,appetite").split(","):
        exp, pop, s = population(run, name)
        if arg("LAB", "1") != "0":
            ext, appetite, xor = lab(exp, pop)
            print(f"  {name:9s} lab: pain extinction {ext};  appetite {appetite};  XOR poison/good rounds 0,1,3,6: {xor}",
                  flush=True)
        if arg("WORLD", "1") != "0":
            acc = world(exp, pop, s)
            m = acc.mean(0)
            d = "" if base is None else f"  vs frozen {(acc[:, 0] - base[:, 0]).mean():+.0f} +-{(acc[:, 0] - base[:, 0]).std(ddof=1) / np.sqrt(len(acc)):.0f}"
            base = acc if base is None else base
            print(f"  {name:9s} world: fitness {m[0]:.0f}{d}  lifetime {m[1]:.0f}  good / poison before the switch "
                  f"{m[2]:.1f} / {m[3]:.1f}, after {m[4]:.1f} / {m[5]:.1f}", flush=True)
