"""Food supply over a life: the newest population of a stage in its own world (or with world fields overridden):
share of the edible bushes that carry berries, animals alive and the mean stomach, at points of the life. Shows
whether the food is a stock that is raced down (everything full at the start, empty by mid-life) or a flow.

    uv run python scripts/probes/supply.py <stage key> [field=value ...] [+ field=value ...]
each group of field=value after a '+' is one more variant of the WorldConfig (start_spent=0.5 eat_cost=0.2)."""
import os, sys
from dataclasses import replace
import jax, numpy as np
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate

key = sys.argv[1]
groups = [[]]
for a in sys.argv[2:]:
    if a == "+":
        groups.append([])
    else:
        groups[-1].append(a)
s0 = S.STAGES[key]
if os.environ.get("TICKS"):          # a longer or shorter life than the stage defines
    s0 = replace(s0, ticks=int(os.environ["TICKS"]))
for g in groups:
    over = {}
    for a in g:
        k_, v_ = a.split("=")
        if k_[0].isupper():          # a constant of stages.py (BUSH_BERRIES=2); stays set for the later variants
            setattr(S, k_, float(v_))
            continue
        over[k_] = type(getattr(s0.world, k_))(float(v_))
    s = replace(s0, world=replace(s0.world, **over))
    exp = S.make_exp(s, s.brain, "supply", 1, 0)
    pop = load_population(os.environ.get("RUN") or latest_run(s.name), exp, seed=0)   # RUN: another run directory
    rs, fn = s.build(exp)
    sim = make_simulate(exp, record=True)
    kr, ks = jax.random.split(jax.random.PRNGKey(5))
    rules = fn(0, kr)
    if rules.food_value.ndim > 1 + (exp.world.switch_tick > 0):
        rules = jax.tree_util.tree_map(lambda a: a[0], rules)
    st, rec = sim(rules, pop, ks)
    names = rs.names
    pain = np.asarray(rules.pain_value)
    full = [i for i, n in enumerate(names) if n.endswith("Wild Gooseberry Bush") and "Empty" not in n and pain[i + 1] == 0]
    spent = {i: next(j for j, n in enumerate(names) if "Empty" in n and np.asarray(rules.decay_new)[j] == i) for i in full}
    grid = np.asarray(rec["grid"]); alive = np.asarray(rec["alive"]); food = np.asarray(rec["food"]); T = grid.shape[0]
    n_full = np.isin(grid, full).reshape(T, -1).sum(1); n_spent = np.isin(grid, list(spent.values())).reshape(T, -1).sum(1)
    ate = np.asarray(rec["ate"]) > 0
    wasted = (ate[1:] & (food[:-1] >= 0.9 * exp.world.max_food)).sum() / max(1, ate[1:].sum())
    print(f"\n{key} {' '.join(g) or 'as defined'}: well-fed lifetime {float(st['fed_meal'].mean()):.0f}, lifetime "
          f"{float(st['alive_ticks'].mean()):.0f} of {T}, alive at the end {int(alive[-1].sum())} of {alive.shape[1]}, "
          f"food eaten {float(st['eaten'].mean()):.1f}, bites on a stomach over 90% full {wasted:.0%}")
    print("  tick                 " + " ".join(f"{t:5d}" for t in range(0, T, T // 10)))
    print("  edible bushes full % " + " ".join(f"{100 * n_full[t] / max(1, n_full[t] + n_spent[t]):5.0f}" for t in range(0, T, T // 10)))
    print("  alive %              " + " ".join(f"{100 * alive[t].mean():5.0f}" for t in range(0, T, T // 10)))
    print("  mean stomach %       " + " ".join(
        f"{100 * (food[t] * alive[t]).sum() / max(1, alive[t].sum()) / exp.world.max_food:5.0f}" for t in range(0, T, T // 10)), flush=True)
