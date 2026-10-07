"""How much of a life's fitness is the genome and how much is luck? A stage's newest population lives in several
different worlds; reported: the correlation between worlds of a genome's fitness (mean of its siblings), which is
what selection can see, and how many of the elite genomes of one world are elite again in another.

    uv run python scripts/probes/repeat.py <stage key> [field=value ...] [+ field=value ...]
each group after a '+' is one more variant of the WorldConfig (start_food=1.0); upper-case = constant of stages.py.
RUN=<run directory> for another population, WORLDS=<n> (default 4)."""
import os, sys
from dataclasses import replace
import jax, numpy as np
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate

key = sys.argv[1]; WORLDS = int(os.environ.get("WORLDS", 4))
groups = [[]]
for a in sys.argv[2:]:
    if a == "+":
        groups.append([])
    else:
        groups[-1].append(a)
s0 = S.STAGES[key]
for g in groups:
    over = {}
    for a in g:
        k_, v_ = a.split("=")
        if k_[0].isupper():
            setattr(S, k_, float(v_)); continue
        over[k_] = type(getattr(s0.world, k_))(float(v_))
    s = replace(s0, world=replace(s0.world, **over))
    exp = S.make_exp(s, s.brain, "repeat", 1, 0)
    pop = load_population(os.environ.get("RUN") or latest_run(s.name), exp, seed=0)
    rs, fn = s.build(exp)
    sim = make_simulate(exp, record=False)
    sib = exp.evolution.siblings; P = pop.b.shape[0]
    fits, lifes = [], []
    for k in jax.random.split(jax.random.PRNGKey(77), WORLDS):
        kr, ks = jax.random.split(k)
        rules = fn(0, kr)
        if rules.food_value.ndim > 1 + (exp.world.switch_tick > 0):
            rules = jax.tree_util.tree_map(lambda a: a[0], rules)
        st, _ = sim(rules, pop, ks)
        f = np.asarray((s.fitness or S.default_fitness)(st))
        fits.append(f.reshape(P, sib).mean(1)); lifes.append(float(np.asarray(st["alive_ticks"]).mean()))
    F = np.array(fits)
    rr = [np.corrcoef(F[i], F[j])[0, 1] for i in range(WORLDS) for j in range(i + 1, WORLDS)]
    ne = max(1, P // 8)
    again = [len(set(np.argsort(-F[i])[:ne]) & set(np.argsort(-F[j])[:ne])) for i in range(WORLDS) for j in range(WORLDS) if i != j]
    print(f"{key} {' '.join(g) or 'as defined'}: fitness {F.mean():.0f}, lifetime {np.mean(lifes):.0f}; between worlds, a genome's fitness correlates "
          f"r = {np.mean(rr):+.2f} (pairs {' '.join(f'{v:+.2f}' for v in rr)}); of {ne} elite genomes {np.mean(again):.1f} are elite again "
          f"(chance {ne * ne / P:.1f})", flush=True)
