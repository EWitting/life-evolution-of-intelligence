"""How well can two populations be told apart? For the newest main and control run of every seed of a stage:

- training curve: the mean fitness per generation over the last 50 generations (new worlds every generation): its
  standard deviation, and the standard error of the window mean;
- re-evaluation: the final population in WORLDS shared worlds (the same for main and control): per world the mean
  fitness; spread over worlds, the standard error of main - control unpaired and paired (same worlds), and the
  smallest difference that two standard errors resolve, in % of the main mean;
- within a world: the spread between animals (genomes, siblings averaged), against the spread between worlds.

    [STAGE=2.2] [WORLDS=16] python scripts/probes/noise.py"""
import os, csv
import jax, numpy as np
from life.experiments import stages as S
from life.run import make_simulate, load_population, latest_run

s = S.STAGES[os.environ.get("STAGE", "2.2")]
WORLDS = int(os.environ.get("WORLDS", 16))
keys = jax.random.split(jax.random.PRNGKey(777), WORLDS)


def evaluate(run, brain):
    exp = S.make_exp(s, brain, "noise", 1, 0)
    pop = load_population(run, exp)
    rs, fn = s.build(exp)
    sim = make_simulate(exp, record=False)
    sib = exp.evolution.siblings
    per_world, between_animals = [], []
    for k in keys:
        kr, ks = jax.random.split(k)
        rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
        st, _ = sim(rules, pop, ks)
        f = np.asarray((s.fitness or S.default_fitness)(st)).reshape(-1, sib).mean(1)   # per genome
        per_world.append(f.mean()); between_animals.append(f.std(ddof=1))
    return np.array(per_world), float(np.mean(between_animals))


def curve(run):
    rows = list(csv.DictReader(open(run / "fitness.csv")))[-50:]
    f = np.array([float(r["fit_mean"]) for r in rows])
    return f.mean(), f.std(ddof=1)


print(f"stage {s.key}, {WORLDS} shared worlds per re-evaluation")
for main_run, ctrl_run in zip(S.seed_runs(s), S.seed_runs(s, control=True)):
    cm, csd = curve(main_run); cc, ccsd = curve(ctrl_run)
    wm, am = evaluate(main_run, s.brain)
    wc, ac = evaluate(ctrl_run, s.control_brain or S.STAGES[s.parent].brain)
    d = wm - wc
    se_unp = np.sqrt(wm.var(ddof=1) / WORLDS + wc.var(ddof=1) / WORLDS); se_p = d.std(ddof=1) / np.sqrt(WORLDS)
    print(f"\n{main_run.parent.name} / {ctrl_run.parent.name}")
    print(f"  training curve, last 50 generations: main {cm:.0f} (sd per generation {csd:.0f}, s.e. {csd / np.sqrt(50):.0f}), "
          f"control {cc:.0f} (sd {ccsd:.0f}, s.e. {ccsd / np.sqrt(50):.0f}); difference {cm - cc:+.0f}")
    print(f"  re-evaluated: main {wm.mean():.0f} (sd over worlds {wm.std(ddof=1):.0f}), control {wc.mean():.0f} "
          f"(sd {wc.std(ddof=1):.0f}); difference {d.mean():+.0f}, s.e. unpaired {se_unp:.0f}, paired {se_p:.0f}; "
          f"resolvable (2 s.e., paired) {100 * 2 * se_p / wm.mean():.1f}% of main")
    print(f"  within a world, sd between genomes: main {am:.0f}, control {ac:.0f} (against sd between worlds "
          f"{wm.std(ddof=1):.0f} / {wc.std(ddof=1):.0f})", flush=True)
