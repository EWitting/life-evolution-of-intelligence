"""Are the number of generations and the mutation settings sensible? Per stage (lineage run):
 A. the fitness curve: noise between generations, gain per block of generations, is it still rising at the end;
 B. repeatability: the final population lives twice in different worlds; correlation of an agent's two fitnesses
    (how much of one life's fitness is the genome rather than luck), and how many elites would be elite again;
 C. mutational load: the final population against one-step mutated copies of itself in the same worlds, for
    several mutation settings (fitness lost per generation to mutation, before selection).
    evo_check.py <stage key> [...]"""
import csv, sys
from dataclasses import replace
import jax, jax.numpy as jnp, numpy as np
from life.config import ExperimentConfig
from life.evolution import mutate
from life.experiments import stages as S
from life.run import load_population, latest_run, make_layout, make_simulate

for key in sys.argv[1:]:
    s = S.STAGES[key]
    for control in (False, True):
        name = s.name + ("_control" if control else "")
        try:
            run = latest_run(name)
        except Exception:
            continue
        rows = list(csv.DictReader(open(run / "fitness.csv")))
        f = np.array([float(r["fit_mean"]) for r in rows]); G = len(f)
        k = 10
        smooth = np.convolve(f, np.ones(k) / k, mode="valid")
        resid = f[k // 2: k // 2 + len(smooth)] - smooth
        noise = resid.std() * np.sqrt(k / (k - 1))
        b = max(10, G // 5)
        blocks = [f[i:i + b].mean() for i in range(0, G - b + 1, b)]
        last, prev = f[-b:], f[-2 * b:-b]
        se = np.sqrt(last.var(ddof=1) / b + prev.var(ddof=1) / b)
        total = blocks[-1] - blocks[0]
        print(f"\n== {key} {name}: {G} generations, population {ExperimentConfig.from_json((run / 'config.json').read_text()).world.num_agents}")
        print(f"A. block means ({b} gens): " + " ".join(f"{x:.0f}" for x in blocks) + f" | noise between generations (sd) {noise:.0f}"
              f" | last block - previous: {last.mean() - prev.mean():+.0f} +-{se:.0f}"
              f" | share of the total gain made in the last block: {(blocks[-1] - blocks[-2]) / total:.0%}" if abs(total) > 1e-6 else "")
        if control:
            continue
        exp = ExperimentConfig.from_json((run / "config.json").read_text())
        layout = make_layout(exp)
        pop = load_population(run)
        rs, fn = s.build(exp)
        sim = jax.jit(make_simulate(exp, record=False))
        fit_fn = s.fitness or S.default_fitness

        def world(seed):
            kr, ks = jax.random.split(jax.random.PRNGKey(seed))
            rules = fn(0, kr)
            if rules.food_value.ndim > 1 + (exp.world.switch_tick > 0):
                rules = jax.tree_util.tree_map(lambda a: a[0], rules)
            return rules, ks

        def fitness(g, seed):
            rules, ks = world(seed)
            return np.asarray(fit_fn(sim(rules, g, ks)[0]))

        fa, fb = fitness(pop, 1), fitness(pop, 2)
        r = np.corrcoef(fa, fb)[0, 1]
        n = len(fa); ne = max(1, round(exp.evolution.elite_frac * n))
        top_a = set(np.argsort(-fa)[:ne]); top_b = set(np.argsort(-fb)[:ne]); half_b = set(np.argsort(-fb)[:n // 2])
        print(f"B. repeatability of one life: r = {r:.2f} (population sd of fitness {fa.std():.0f}, mean {fa.mean():.0f}); "
              f"of the {ne} elites of world 1, {len(top_a & top_b)} are elite again in world 2 (chance {ne * ne / n:.1f}), "
              f"{len(top_a & half_b)} are in its top half")
        base = np.mean([fa.mean(), fb.mean()])
        ecfg = exp.evolution
        out = []
        for label, cfg in (("as run", ecfg),
                           ("prob x0.3", replace(ecfg, weight_mutation_prob=ecfg.weight_mutation_prob * 0.3)),
                           ("prob x3", replace(ecfg, weight_mutation_prob=min(1.0, ecfg.weight_mutation_prob * 3))),
                           ("std x0.5", replace(ecfg, mutation_std=ecfg.mutation_std * 0.5)),
                           ("std x2", replace(ecfg, mutation_std=ecfg.mutation_std * 2))):
            keys = jax.random.split(jax.random.PRNGKey(7), n)
            mut = jax.vmap(lambda kk, g: mutate(kk, g, cfg, layout))(keys, pop)
            fm = np.mean([fitness(mut, 1).mean(), fitness(mut, 2).mean()])
            out.append(f"{label}: {fm - base:+.0f}")
        gain = (blocks[-1] - blocks[0]) / max(1, G - b)
        print(f"C. mutational load (fitness change of the whole population after one round of mutation; weights mutate with "
              f"prob {ecfg.weight_mutation_prob}, std {ecfg.mutation_std}): " + ", ".join(out)
              + f" | average gain per generation over the run {gain:+.2f}", flush=True)
