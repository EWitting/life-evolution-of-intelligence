"""How much is there to learn in a life? The parent stage's population in a learning stage's world, learning off:
per animal, the number of visits to a bush (a run of bites of one type with gaps under 6 ticks), bites per visit,
and how often an animal comes back to a poison type it has already been hurt by. A lesson can only pay on those
later visits.

    STAGE=1.5 uv run python scripts/probes/lessons.py [FIELD=value ...]     as gen0.py: lower-case = WorldConfig field,
                                                                            upper-case = constant of stages.py
TICKS in the environment overrides the length of a life."""
import os, sys
from dataclasses import replace
import jax, jax.numpy as jnp, numpy as np
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate

s = S.STAGES[os.environ.get("STAGE", "1.5")]
for a in sys.argv[1:]:
    k_, v_ = a.split("=")
    if k_[0].isupper():
        setattr(S, k_, float(v_))
    else:
        s = replace(s, world=replace(s.world, **{k_: type(getattr(s.world, k_))(float(v_))}))
if os.environ.get("TICKS"):
    s = replace(s, ticks=int(os.environ["TICKS"]))
exp = S.make_exp(s, s.brain, "lessons", 1, 0)
pop = load_population(os.environ.get("RUN") or latest_run(S.STAGES[s.parent].name), exp, seed=0)   # RUN: another run
pop = pop._replace(eta=jnp.zeros_like(pop.eta))
rs, fn = s.build(exp)
sim = make_simulate(exp, record=True)
acc = []
for k in jax.random.split(jax.random.PRNGKey(31), 2):
    kr, ks = jax.random.split(k)
    rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
    pain = np.asarray(rules.pain_value)
    st, rec = sim(rules, pop, ks)
    ate = np.asarray(rec["ate"]); T, N = ate.shape
    life = np.asarray(st["alive_ticks"])
    for i in range(N):
        ts = np.nonzero(ate[:, i] > 0)[0]
        visits = []                                  # (type, bites)
        for j, t in enumerate(ts):
            o = int(ate[t, i])
            if visits and visits[-1][0] == o and t - ts[j - 1] < 6:
                visits[-1][1] += 1
            else:
                visits.append([o, 1])
        good = [v for v in visits if pain[v[0]] == 0]; bad = [v for v in visits if pain[v[0]] > 0]
        seen, again = set(), 0
        for o, _ in bad:
            again += o in seen; seen.add(o)
        acc.append([life[i], life[i] >= T, len(good), np.mean([b for _, b in good]) if good else np.nan, len(bad),
                    np.mean([b for _, b in bad]) if bad else np.nan, len(seen), again])
m = np.nanmean(np.array(acc, float), 0)
print(f"settings {sys.argv[1:]}: lifetime {m[0]:.0f} of {exp.evolution.ticks_per_generation}, alive at the cap {m[1]:.0%}")
print(f"  visits to a good bush per life {m[2]:.1f} ({m[3]:.1f} bites each); to a poison bush {m[4]:.1f} ({m[5]:.1f} bites each)")
print(f"  poison types an animal is hurt by {m[6]:.1f}; later visits to a type that already hurt it {m[7]:.2f} per life", flush=True)
