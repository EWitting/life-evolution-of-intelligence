"""Noise at the start of a stage, where selection has something to act on. Generation 0 of a stage (the parent's seed-0
population on the stage's brain, as `run_stage` builds it, main and control) and, for contrast, the stage's final
seed-0 populations, all in the same WORLDS worlds. Per world the fitness of every animal.

- Repeatability of a genome's score from n lives (siblings x episodes): the correlation between scores of the same
  genomes from disjoint sets of worlds, measured, not extrapolated. Its square root is roughly how well a score
  ranks genomes by what they are worth (selection accuracy): the response to selection grows with it.
- Comparing main and control: per world the population means and the paired difference; its standard deviation
  is the noise of one generation of the training curve with one episode, and of one world in a re-evaluation.
  Standard errors with E episodes x 50 generations and with W re-evaluation worlds follow.

    [STAGE=2.2] [WORLDS=32] [BATCH=4] python scripts/probes/noise_gen0.py      (raw fitness kept in runs/logs/)"""
import os
import jax, jax.numpy as jnp, numpy as np
from life.experiments import stages as S
from life.run import make_simulate, load_population, latest_run, make_layout, RUNS_DIR

s = S.STAGES[os.environ.get("STAGE", "2.2")]
WORLDS, BATCH = int(os.environ.get("WORLDS", 32)), int(os.environ.get("BATCH", 4))
keys = jax.random.split(jax.random.PRNGKey(2026), WORLDS)


def gen0(control):
    brain = (s.control_brain or S.STAGES[s.parent].brain) if control else s.brain
    exp = S.make_exp(s, brain, "noise0", 1, 0)
    pop = load_population(latest_run(S.STAGES[s.parent].name), exp, seed=0)
    lay = make_layout(exp)
    if s.adapt_init:
        pop = s.adapt_init(pop, lay)
    if s.dense_plastic and not control:
        pop = pop._replace(mask=jnp.where(((np.asarray(lay.rule) > 0) & (np.asarray(lay.allowed) > 0))[None], 1.0, pop.mask))
    return exp, pop


def final(control):
    brain = (s.control_brain or S.STAGES[s.parent].brain) if control else s.brain
    exp = S.make_exp(s, brain, "noise0", 1, 0)
    return exp, load_population(S.seed_runs(s, control=control)[0], exp)


def fitness(exp, pop):
    """[worlds, animals]"""
    rs, fn = s.build(exp)
    sim = jax.jit(jax.vmap(make_simulate(exp, record=False), in_axes=(0, None, 0)))
    out = []
    for i in range(0, WORLDS, BATCH):
        kk = keys[i:i + BATCH]
        kr, ks = jax.vmap(jax.random.split, out_axes=1)(kk)
        rules = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *[jax.tree_util.tree_map(lambda a: a[0], fn(0, k)) for k in kr])
        st, _ = sim(rules, pop, ks)
        out.append(np.asarray(jax.vmap(s.fitness or S.default_fitness)(st)))
        print(f"  worlds {i + len(kk)}/{WORLDS}", flush=True)
    return np.concatenate(out)


def repeatability(F, sib):
    """{(lives, worlds): r}: scores of the same genomes from disjoint worlds (siblings within a world, episodes = worlds)."""
    G = F.reshape(F.shape[0], -1, sib)                     # [worlds, genomes, siblings]
    out = {}
    for n_sib in (1, sib):
        for k in (1, 2, 4, 8, 16):
            if 2 * k > F.shape[0]:
                continue
            groups = np.array([G[j * k:(j + 1) * k, :, :n_sib].mean((0, 2)) for j in range(F.shape[0] // k)])
            c = np.corrcoef(groups)
            out[(n_sib * k, k)] = c[np.triu_indices(len(groups), 1)].mean()
    return out


res = {}
for tag, make in (("gen0", gen0), ("final", final)):
    for control in (False, True):
        label = f"{tag} {'control' if control else 'main'}"
        print(label, flush=True)
        exp, pop = make(control)
        res[label] = fitness(exp, pop)
        sib = exp.evolution.siblings
np.savez(RUNS_DIR / "logs" / f"noise_gen0_{s.key}.npz", **{k.replace(' ', '_'): v for k, v in res.items()})

print(f"\nstage {s.key}, {WORLDS} shared worlds, {sib} siblings per genome")
print("\nRepeatability of a genome's score (correlation between disjoint worlds); selection accuracy ~ its square root")
for label, F in res.items():
    r = repeatability(F, sib)
    G = F.reshape(WORLDS, -1, sib).mean(2)
    print(f"  {label:14s} mean {F.mean():5.0f}, sd between genomes in a world {G.std(1, ddof=1).mean():4.0f}: "
          + ", ".join(f"{n} lives ({'1 world' if k == 1 else f'{k} worlds'}) {v:.2f}" for (n, k), v in sorted(r.items())))

print("\nMain vs control, per world (paired: the same worlds)")
for tag in ("gen0", "final"):
    m, c = res[f"{tag} main"].mean(1), res[f"{tag} control"].mean(1)
    d = m - c
    sd = d.std(ddof=1)
    print(f"  {tag}: main {m.mean():.0f} (sd over worlds {m.std(ddof=1):.0f}), control {c.mean():.0f} ({c.std(ddof=1):.0f}); "
          f"difference {d.mean():+.0f}, sd of the paired difference per world {sd:.0f} (unpaired {np.sqrt(m.var(ddof=1) + c.var(ddof=1)):.0f})")
    print("    training curve, s.e. of the 50-generation difference: "
          + ", ".join(f"{E} episode{'s' if E > 1 else ''} {sd / np.sqrt(50 * E):.0f}" for E in (1, 2, 4, 8))
          + "  (world noise only; seeds differ more)")
    print("    re-evaluation, s.e. of the difference: "
          + ", ".join(f"{W} worlds {sd / np.sqrt(W):.0f} ({100 * 2 * sd / np.sqrt(W) / m.mean():.1f}% at 2 s.e.)" for W in (8, 16, 32, 64)))
