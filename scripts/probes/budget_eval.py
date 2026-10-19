"""The GPU-budget test (STAGE_LOG v53): the final populations of 2.2 (100 generations, half the budget) and of the
side stages x.b1 (200 generations), x.b4 (episodes 4, 87 generations) and x.b256 (256 genomes in one large world, 114
generations), every seed, in the stage's own world (256 x 256, 64 genomes x 4 siblings) and the same WORLDS worlds.
A population of 256 genomes is represented by 64 of them drawn at random. Also with the pallium silenced and with
learning off (in the first LESION_WORLDS of those worlds), and the GPU time every run took.

    [WORLDS=64] [LESION_WORLDS=16] [BATCH=4] python scripts/probes/budget_eval.py [key ...]   default: all four"""
import os, sys, csv
import jax, jax.numpy as jnp, numpy as np
from life.experiments import stages as S
from life.run import make_simulate, load_population, make_layout

WORLDS, LW, BATCH = int(os.environ.get("WORLDS", 64)), int(os.environ.get("LESION_WORLDS", 16)), int(os.environ.get("BATCH", 4))
keys_ = sys.argv[1:] or ["2.2", "x.b1", "x.b4", "x.b256"]
s = S.STAGES["2.2"]
exp = S.make_exp(s, s.brain, "budget", 1, 0)
lay = make_layout(exp)
rs, fn = s.build(exp)
sim = jax.jit(jax.vmap(make_simulate(exp, record=False), in_axes=(0, None, 0)))
keys = jax.random.split(jax.random.PRNGKey(4242), WORLDS)
rules_all = [jax.tree_util.tree_map(lambda a: a[0], fn(0, jax.random.split(k)[0])) for k in keys]
G = exp.world.num_agents // exp.evolution.siblings


def fitness(pop, n):
    out = []
    for i in range(0, n, BATCH):
        rules = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *rules_all[i:i + BATCH])
        ks = jnp.stack([jax.random.split(k)[1] for k in keys[i:i + BATCH]])
        st, _ = sim(rules, pop, ks)
        out.append(np.asarray(jax.vmap(s.fitness or S.default_fitness)(st)).mean(1))
    return np.concatenate(out)


def silenced(pop, region):
    rows = lay.region(region)
    return pop._replace(w0=pop.w0.at[:, rows, :].set(0.0), mask=pop.mask.at[:, rows, :].set(0.0), b=pop.b.at[:, rows].set(-10.0))


res = {}
for key in keys_:
    st = S.STAGES[key]
    for run in S.seed_runs(st):
        own = S.make_exp(st, st.brain, "budget", 1, 0)
        pop = load_population(run, own)
        if pop.b.shape[0] != G:   # 64 genomes at random
            pick = np.random.default_rng(0).choice(pop.b.shape[0], G, replace=False)
            pop = jax.tree_util.tree_map(lambda a: a[pick] if getattr(a, "ndim", 0) and a.shape[0] == pop.b.shape[0] else a, pop)
        rows = list(csv.DictReader(open(run / "fitness.csv")))
        intact = fitness(pop, WORLDS)
        pal = fitness(silenced(pop, "pallium"), LW)
        nop = fitness(pop._replace(eta=jnp.zeros_like(pop.eta)), LW)
        r = dict(fit=intact.mean(), pallium=100 * pal.mean() / intact[:LW].mean(), no_plasticity=100 * nop.mean() / intact[:LW].mean(),
                 gens=len(rows), minutes=float(rows[-1]["seconds"]) / 60, per_world=intact)
        res.setdefault(key, []).append(r)
        print(f"{key:7s} {run.parent.name}/{run.name}: fitness {r['fit']:.0f} (64 worlds), pallium silenced {r['pallium']:.0f}%, "
              f"learning off {r['no_plasticity']:.0f}%; {r['gens']} generations in {r['minutes']:.0f} min", flush=True)

print(f"\nsummary ({WORLDS} shared worlds; +- s.e. over seeds; paired difference from 2.2 per seed, same worlds)")
ref = res.get("2.2")
for key, rr in res.items():
    f = np.array([r["fit"] for r in rr])
    line = f"  {key:7s} fitness {f.mean():.0f} +-{f.std(ddof=1) / np.sqrt(len(f)):.0f} ({', '.join(f'{v:.0f}' for v in f)})"
    if ref and key != "2.2":
        d = np.array([(a["per_world"] - b["per_world"]).mean() for a, b in zip(rr, ref)])
        line += f"; vs 2.2 {d.mean():+.0f} +-{d.std(ddof=1) / np.sqrt(len(d)):.0f}"
    line += (f"; pallium silenced {np.mean([r['pallium'] for r in rr]):.0f}%, learning off {np.mean([r['no_plasticity'] for r in rr]):.0f}%; "
             f"{np.mean([r['minutes'] for r in rr]):.0f} min per run")
    print(line, flush=True)
