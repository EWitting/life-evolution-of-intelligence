"""Small (or large) worlds as episodes: the stage's world shrunk by FACTOR in area (below 1 enlarges it; animals and
patches scale with it, so animal, food and patch density stay the same), SIBLINGS per genome (default 1; the number of
genomes stays the stage's), many worlds per generation, against the stage's own world.
Generation 0 of the stage and its final seed-0 populations (main and control, as noise_gen0.py), WORLDS small worlds
run BATCH at a time (one batch = one generation with `episodes` BATCH).

- the world's economy: fitness, lifetime and meals per animal, small against large;
- repeatability of a genome's score from n lives (= n small worlds), as noise_gen0.py;
- main vs control, paired: sd of the difference per world and the standard error of a re-evaluation;
- speed: seconds per generation (one batch) against one large world.

    [STAGE=2.2] [FACTOR=4] [SIBLINGS=1] [WORLDS=128] [BATCH=16] python scripts/probes/small_world.py
    one large world, 16 siblings:  FACTOR=0.25 SIBLINGS=16 WORLDS=16 BATCH=1"""
import os, time, math
from dataclasses import replace
import jax, jax.numpy as jnp, numpy as np
from life.experiments import stages as S
from life.run import make_simulate, load_population, latest_run, make_layout, RUNS_DIR

s0 = S.STAGES[os.environ.get("STAGE", "2.2")]
SIB = int(os.environ.get("SIBLINGS", 1))
FACTOR, WORLDS, BATCH = float(os.environ.get("FACTOR", 4)), int(os.environ.get("WORLDS", 128)), int(os.environ.get("BATCH", 16))
w = s0.world
side = int(round(w.height / math.sqrt(FACTOR)))
small = replace(s0, world=replace(w, num_agents=round(w.num_agents / FACTOR), height=side, width=side,
                                  patches=round(w.patches / FACTOR)))
print(f"stage {s0.key}: large {w.height} x {w.width}, {w.num_agents} animals, {w.patches} patches; small {side} x {side}, "
      f"{small.world.num_agents} animals, {small.world.patches} patches, {SIB} siblings; {WORLDS} worlds, {BATCH} per generation", flush=True)


def population(s, tag, control, siblings):
    brain = (s.control_brain or S.STAGES[s.parent].brain) if control else s.brain
    exp = S.make_exp(s, brain, "small", 1, 0)
    exp = replace(exp, evolution=replace(exp.evolution, siblings=siblings))
    if tag == "final":
        return exp, load_population(S.seed_runs(s0, control=control)[0], exp)
    pop = load_population(latest_run(S.STAGES[s.parent].name), exp, seed=0)
    lay = make_layout(exp)
    if s.adapt_init:
        pop = s.adapt_init(pop, lay)
    if s.dense_plastic and not control:
        pop = pop._replace(mask=jnp.where(((np.asarray(lay.rule) > 0) & (np.asarray(lay.allowed) > 0))[None], 1.0, pop.mask))
    return exp, pop


def run(s, exp, pop, n, batch, seed):
    """stats per world: fitness [n, animals], alive ticks and meals per world"""
    rs, fn = s.build(exp)
    sim = jax.jit(jax.vmap(make_simulate(exp, record=False), in_axes=(0, None, 0)))
    keys = jax.random.split(jax.random.PRNGKey(seed), n)
    F, life, meals, secs = [], [], [], []
    for i in range(0, n, batch):
        kr, ks = jax.vmap(jax.random.split, out_axes=1)(keys[i:i + batch])
        rules = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *[jax.tree_util.tree_map(lambda a: a[0], fn(0, k)) for k in kr])
        t0 = time.time()
        st, _ = sim(rules, pop, ks)
        jax.block_until_ready(st)
        secs.append(time.time() - t0)
        F.append(np.asarray(jax.vmap(s.fitness or S.default_fitness)(st)))
        life.append(np.asarray(st["alive_ticks"]).mean(1)); meals.append(np.asarray(st["eaten"]).mean(1))
    return np.concatenate(F), np.concatenate(life), np.concatenate(meals), secs


def repeatability(G):
    """G [worlds, genomes]: {lives: r} between disjoint groups of worlds"""
    out = {}
    for k in (1, 2, 4, 8, 16, 32):
        if 2 * k <= len(G):
            g = np.array([G[j * k:(j + 1) * k].mean(0) for j in range(len(G) // k)])
            c = np.corrcoef(g)
            out[k] = c[np.triu_indices(len(g), 1)].mean()
    return out


res = {}
for tag in ("gen0", "final"):
    for control in (False, True):
        label = f"{tag} {'control' if control else 'main'}"
        exp, pop = population(small, tag, control, SIB)
        F, life, meals, secs = run(small, exp, pop, WORLDS, BATCH, 2026)
        res[label] = F.reshape(len(F), -1, SIB).mean(2)   # [worlds, genomes]
        print(f"{label}: small worlds fitness {F.mean():.0f}, lifetime {life.mean():.0f}, meals {meals.mean():.1f}; "
              f"{np.median(secs[1:]) if len(secs) > 1 else secs[0]:.1f} s per {BATCH} worlds (first with compilation {secs[0]:.0f} s)", flush=True)
        if tag == "final" or control:
            continue
        exp, pop = population(s0, tag, control, 4)   # the large world once, for the economy and the speed
        F, life, meals, secs = run(s0, exp, pop, 8, 4, 2027)
        print(f"{label}: large world  fitness {F.mean():.0f}, lifetime {life.mean():.0f}, meals {meals.mean():.1f}; "
              f"{np.median(secs[1:]):.1f} s per 4 worlds (episodes 4)", flush=True)
np.savez(RUNS_DIR / "logs" / f"small_world_{s0.key}_f{FACTOR:g}_s{SIB}.npz", **{k.replace(' ', '_'): v for k, v in res.items()})

print(f"\nRepeatability with n worlds x {SIB} siblings")
for label, F in res.items():
    print(f"  {label:14s} " + ", ".join(f"{n * SIB} lives ({n} world{'s' if n > 1 else ''}) {v:.2f}" for n, v in repeatability(F).items()))
print("\nMain vs control, paired")
for tag in ("gen0", "final"):
    m, c = res[f"{tag} main"].mean(1), res[f"{tag} control"].mean(1)
    d = m - c; sd = d.std(ddof=1)
    print(f"  {tag}: difference {d.mean():+.0f}, sd per world {sd:.0f}; re-evaluation s.e. "
          + ", ".join(f"{W} worlds {sd / np.sqrt(W):.0f} ({100 * 2 * sd / np.sqrt(W) / m.mean():.1f}%)" for W in (4, 8, 16, 32, 64, 128, 256)))
