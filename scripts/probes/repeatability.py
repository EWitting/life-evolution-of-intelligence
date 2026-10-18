"""How much of a genome's fitness is the genome? The final population of a stage's newest runs (seed 0, main and
control) lives in WORLDS shared worlds; per world, the fitness of every genome (mean of its siblings). Repeatability
= the mean correlation of the genome scores between two worlds: the share of the spread between genomes that is
genetic. From it, by the Spearman-Brown formula, the repeatability with n lives per genome: n r1 / (1 + (n-1) r1),
r1 = the repeatability of one life.

    [STAGE=2.2] [WORLDS=8] python scripts/probes/repeatability.py"""
import os
import jax, numpy as np
from life.experiments import stages as S
from life.run import make_simulate, load_population

s = S.STAGES[os.environ.get("STAGE", "2.2")]
WORLDS = int(os.environ.get("WORLDS", 8))

for label, run, brain in (("main", S.seed_runs(s)[0], s.brain),
                          ("control", S.seed_runs(s, control=True)[0], s.control_brain or S.STAGES[s.parent].brain)):
    exp = S.make_exp(s, brain, "repeat", 1, 0)
    pop = load_population(run, exp)
    rs, fn = s.build(exp)
    sim = make_simulate(exp, record=False)
    sib = exp.evolution.siblings
    F = []
    for k in jax.random.split(jax.random.PRNGKey(99), WORLDS):
        kr, ks = jax.random.split(k)
        rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
        F.append(np.asarray((s.fitness or S.default_fitness)(sim(rules, pop, ks)[0])).reshape(-1, sib).mean(1))
    F = np.array(F)                                   # [worlds, genomes]
    c = np.corrcoef(F)
    r = c[np.triu_indices(WORLDS, 1)].mean()          # repeatability of a score from `sib` lives
    r1 = r / (sib - (sib - 1) * r)                    # Spearman-Brown backwards: one life
    lives = {n: n * r1 / (1 + (n - 1) * r1) for n in (1, 4, 8, 16, 32)}
    print(f"{label} ({run.parent.name}): sd between genomes in a world {F.std(1, ddof=1).mean():.0f}, "
          f"of the genome means over {WORLDS} worlds {F.mean(0).std(ddof=1):.0f}; repeatability with {sib} siblings "
          f"{r:.2f}, one life {r1:.2f}; with n lives: " + ", ".join(f"{n}: {v:.2f}" for n, v in lives.items()), flush=True)
