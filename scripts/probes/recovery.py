"""Recovery benchmark: which evolution settings repair a damaged population fastest?

A stage's final population is damaged by one heavy round of mutation (every weight, std 0.25) and then evolved for G
generations under each setting. Every setting is scored with the same yardsticks in the same fixed worlds:
recovery = (value - damaged) / (intact - damaged), for energy acquired and for ticks alive. This isolates how well
selection works; it does not need a setting's own fitness numbers to be comparable.

    uv run python scripts/probes/recovery.py <stage key> <lane> <lanes> [generations] [seeds]

DENSITY=<x> in the environment overrides the bush density of the run's world; ONLY=<a>,<b> keeps the settings
whose name starts with one of these.
Run one process per lane (lane = 0 .. lanes-1); together they cover all settings x seeds. Lines starting with
'ROW' are results: setting, seed, generation, energy acquired, ticks alive, recovery of each in %."""
import os, sys
from dataclasses import replace
import jax, jax.numpy as jnp, numpy as np
from life.config import ExperimentConfig
from life.evolution import mutate, next_generation
from life.experiments import stages as S
from life.run import load_population, latest_run, make_layout, make_simulate

key, lane, lanes = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
G = int(sys.argv[4]) if len(sys.argv) > 4 else 40
SEEDS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
YARD_WORLDS = 3

# name: (fitness stat, crossover probability, crossover mode, weight mutation prob, mutation std, siblings)
SETTINGS = {
    "current (well-fed lifetime, asexual)":        ("fed_meal", 0.0, "neuron", 0.10, 0.10, 4),
    "well-fed lifetime + recombination":           ("fed_meal", 1.0, "neuron", 0.10, 0.10, 4),
    "energy acquired":                             ("eaten",    0.0, "neuron", 0.10, 0.10, 4),
    "energy + recombination (whole neurons)":      ("eaten",    1.0, "neuron", 0.10, 0.10, 4),
    "energy + recombination (blending)":           ("eaten",    1.0, "blend",  0.10, 0.10, 4),
    "energy + recombination + few large mutations": ("eaten",   1.0, "neuron", 0.02, 0.30, 4),
    "energy + recombination, 256 genomes x 1":     ("eaten",    1.0, "neuron", 0.10, 0.10, 1),
    "energy, every weight mutates (as stage 1.0)": ("eaten",    0.0, "neuron", 1.00, 0.10, 4),
}

if "ONLY" in os.environ:
    SETTINGS = {n: v for n, v in SETTINGS.items() if n.startswith(tuple(os.environ["ONLY"].split(",")))}
s = S.STAGES[key]
run = latest_run(s.name)
exp = ExperimentConfig.from_json((run / "config.json").read_text())
if "DENSITY" in os.environ:   # a richer or poorer world than the run's own
    exp = replace(exp, world=replace(exp.world, spawn_density=float(os.environ["DENSITY"])))
layout = make_layout(exp)
intact = load_population(run)
N = exp.world.num_agents
rs, fn = s.build(exp)
sim = jax.jit(make_simulate(exp, record=False))


def rules_for(k):
    rules = fn(0, k)
    if rules.food_value.ndim > 1 + (exp.world.switch_tick > 0):
        rules = jax.tree_util.tree_map(lambda a: a[0], rules)
    return rules


def yardstick(pop):
    out = []
    for w in range(YARD_WORLDS):
        kr, ks = jax.random.split(jax.random.PRNGKey(7000 + w))
        st = sim(rules_for(kr), pop, ks)[0]
        out.append([float(st["eaten"].mean()), float(st["alive_ticks"].mean())])
    return np.mean(out, axis=0)


def damaged(P):
    idx = jnp.arange(P) % intact.b.shape[0]
    base = jax.tree_util.tree_map(lambda a: a[idx], intact)
    cfg = replace(exp.evolution, weight_mutation_prob=1.0, mutation_std=0.25, mask_flip_prob=0.0, tune_prob=0.0, plastic=False)
    return jax.vmap(lambda k, g: mutate(k, g, cfg, layout))(jax.random.split(jax.random.PRNGKey(4242), P), base)


y_intact = yardstick(intact)
print(f"stage {key} ({run.name}), {N} agents, {G} generations; intact population: energy {y_intact[0]:.1f}, alive {y_intact[1]:.0f}", flush=True)
runs = [(name, seed) for name in SETTINGS for seed in range(SEEDS)]
start = {}
for i, (name, seed) in enumerate(runs):
    if i % lanes != lane:
        continue
    stat, cx, mode, mprob, mstd, sib = SETTINGS[name]
    P = N // sib
    ecfg = replace(exp.evolution, crossover=cx, crossover_mode=mode, weight_mutation_prob=mprob, mutation_std=mstd,
                   siblings=sib, plastic=False)
    pop = damaged(P)
    if P not in start:
        start[P] = yardstick(pop)
        print(f"damaged start with {P} genomes: energy {start[P][0]:.1f}, alive {start[P][1]:.0f}", flush=True)
    y0 = start[P]
    k = jax.random.PRNGKey(100 + seed)
    for gen in range(1, G + 1):
        k, kr, ks, ke = jax.random.split(k, 4)
        st = sim(rules_for(kr), pop, ks)[0]
        fit = st[stat].astype(jnp.float32).reshape(P, sib).mean(axis=1)
        pop = next_generation(ke, pop, fit, ecfg, layout)
        if gen in (G // 2, G):
            y = yardstick(pop)
            rec = 100 * (y - y0) / (y_intact - y0)
            print(f"ROW|{name}|{seed}|{gen}|{y[0]:.1f}|{y[1]:.0f}|{rec[0]:.0f}|{rec[1]:.0f}", flush=True)
