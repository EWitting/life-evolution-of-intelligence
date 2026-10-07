"""Generation-0 test of a learning stage: the parent stage's final population on this stage's brain (or a variant of
it), in this stage's world, with learning on and off in the same worlds. No evolution: a sound rule should help
straight away, or at least not hurt.
    STAGE=1.5 uv run python scripts/probes/gen0.py [FIELD=value ...] <variant> [...]
variants: see variants.py. lower-case field=value overrides a WorldConfig field (hunger_per_tick=0.02), upper-case
NAME=value a constant of life.experiments.stages (NOVEL_WEIGHT=1.0). STAGE defaults to 1.5."""
import os, sys
from dataclasses import replace
import jax, jax.numpy as jnp, numpy as np
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate
from variants import variant

s = S.STAGES[os.environ.get("STAGE", "1.5")]
WORLDS = int(os.environ.get("WORLDS", 6))
args = [a for a in sys.argv[1:] if "=" not in a]
for a in sys.argv[1:]:
    if "=" in a:
        k_, v_ = a.split("=")
        if k_[0].isupper():
            setattr(S, k_, float(v_))
        else:
            s = replace(s, world=replace(s.world, **{k_: type(getattr(s.world, k_))(float(v_))}))
print("settings:", [a for a in sys.argv[1:] if "=" in a], flush=True)


for v in args:
    exp = S.make_exp(s, variant(s.brain, v), "gen0", 1, 0)
    pop = load_population(os.environ.get("RUN") or latest_run(S.STAGES[s.parent].name), exp, seed=0)   # RUN: another run
    rs, fn = s.build(exp)
    sim = make_simulate(exp, record=False)
    berry = [rs.local(S.variant(t)[S.BERRY]) for t in range(6)]
    for label, g in (("learning on ", pop), ("learning off", pop._replace(eta=jnp.zeros_like(pop.eta)))):
        acc = []
        for k in jax.random.split(jax.random.PRNGKey(31), WORLDS):
            kr, ks = jax.random.split(k)
            rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
            pv = np.asarray(rules.pain_value)
            bad = [t for t in range(6) if pv[berry[t]] > 0]; good = [t for t in range(6) if pv[berry[t]] == 0]
            st, _ = sim(rules, g, ks)
            e = np.asarray(st["eats"]); n = e.shape[0]
            tot = lambda h, ts: sum(e[:, h, berry[t]].sum() for t in ts) / n
            acc.append([float((s.fitness or S.default_fitness)(st).mean()), float(st["alive_ticks"].mean()), tot(0, good), tot(0, bad), tot(1, good), tot(1, bad),
                        float(st["pain"].mean())])
        m = np.mean(acc, 0); se = np.std(acc, 0, ddof=1) / np.sqrt(WORLDS)
        print(f"{v:22s} {label}: fitness {m[0]:4.0f} +-{se[0]:.0f}  lifetime {m[1]:4.0f}  good / poison berries per agent: first half "
              f"{m[2]:5.1f} / {m[3]:4.1f}, second half {m[4]:5.1f} / {m[5]:4.1f}  pain {m[6]:.1f}", flush=True)
