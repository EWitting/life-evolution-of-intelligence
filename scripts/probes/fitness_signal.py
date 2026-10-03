"""Which fitness definition lets selection see skill? Candidate fitness measures compared on the same simulated lives.

 A. Detectability: 32 genomes of the final population live in the same worlds as slightly damaged copies of
    themselves (one round of mutation). For each candidate measure: d = (mean intact - mean damaged) / sd of one
    life. A measure with a larger d separates better from worse genomes more easily, whatever its scale.
 B. Intraclass correlation in the undamaged population (64 genomes x 4 siblings): the share of the variance between
    individual lives that is due to the genome.
    signal.py <stage key> [worlds]"""
import sys
from dataclasses import replace
import jax, jax.numpy as jnp, numpy as np
from life.config import ExperimentConfig
from life.evolution import mutate
from life.experiments import stages as S
from life.run import load_population, latest_run, make_layout, make_simulate

key = sys.argv[1]
WORLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 6
s = S.STAGES[key]
run = latest_run(s.name)
exp = ExperimentConfig.from_json((run / "config.json").read_text())
layout = make_layout(exp)
pop = load_population(run)
P = pop.b.shape[0]; N = exp.world.num_agents; k = N // P
rs, fn = s.build(exp)
sim = jax.jit(make_simulate(exp, record=False))
T = exp.evolution.ticks_per_generation


def measures(st):
    eaten, alive = np.asarray(st["eaten"]), np.asarray(st["alive_ticks"]).astype(np.float64)
    return {"well-fed lifetime from first meal (current)": np.asarray(st["fed_meal"]),
            "well-fed lifetime from birth": np.asarray(st["fed"]),
            "ticks alive": alive,
            "energy acquired (food eaten, uncapped)": eaten,
            "energy acquired per tick alive": eaten / np.maximum(alive, 1.0),
            "energy acquired + 0.02 x ticks alive": eaten + 0.02 * alive}


def lives(g, seed):
    kr, ks = jax.random.split(jax.random.PRNGKey(seed))
    rules = fn(0, kr)
    if rules.food_value.ndim > 1 + (exp.world.switch_tick > 0):
        rules = jax.tree_util.tree_map(lambda a: a[0], rules)
    return measures(sim(rules, g, ks)[0])


rank = lambda v: np.argsort(np.argsort(v)).astype(np.float64)
print(f"== stage {key} ({run.name}): {P} genomes x {k} siblings, {WORLDS} worlds, {T} ticks")

# ---- B. intraclass correlation of the undamaged population
acc = {}
for w in range(WORLDS):
    for name, v in lives(pop, 100 + w).items():
        for tag, x in (("", v), (" [ranks]", rank(v))):
            g = x.reshape(P, k)
            msb = k * g.mean(1).var(ddof=1); msw = g.var(axis=1, ddof=1).mean()
            acc.setdefault(name + tag, []).append((msb - msw) / (msb + (k - 1) * msw))
print("B. share of the variance between lives that is due to the genome (intraclass correlation), final population:")
for name, v in acc.items():
    if "[ranks]" in name:
        print(f"   {name:58s} {np.mean(v):+.3f} +-{np.std(v, ddof=1) / np.sqrt(len(v)):.3f}")

# ---- A. detectability of damaged genomes
half = P // 2
base = jax.tree_util.tree_map(lambda a: a[:half], pop)
print("A. intact genomes vs mutated copies of themselves in the same worlds: d = difference / sd of one life (ranks)")
for label, cfg in (("mutation as in warm stages (10% of weights, std 0.1)", replace(exp.evolution, weight_mutation_prob=0.1, mutation_std=0.1)),
                   ("every weight, std 0.1 (as in stage 1.0)", replace(exp.evolution, weight_mutation_prob=1.0, mutation_std=0.1)),
                   ("every weight, std 0.25", replace(exp.evolution, weight_mutation_prob=1.0, mutation_std=0.25))):
    keys = jax.random.split(jax.random.PRNGKey(5), half)
    dam = jax.vmap(lambda kk, g: mutate(kk, g, cfg, layout))(keys, base)
    mixed = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a, b]), base, dam)
    ds = {}
    for w in range(WORLDS):
        for name, v in lives(mixed, 200 + w).items():
            r = rank(v)
            a, b = r[:half * k], r[half * k:]
            sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
            ds.setdefault(name, []).append((a.mean() - b.mean()) / sd)
    print(f"   {label}:")
    for name, v in ds.items():
        print(f"      {name:55s} d = {np.mean(v):+.2f} +-{np.std(v, ddof=1) / np.sqrt(len(v)):.2f}", flush=True)
