"""Is the adaptation of the inputs used? The newest population of a stage with adapting inputs (BrainConfig.in_adapt)
lives in the same worlds with the adaptation on and off; there is no region to silence, so this is its lesion.

    uv run python scripts/probes/adapt_off.py <stage key> [...]     WORLDS=<n> (default 4)
The result is also written to book/data/adapt_off_<key>.json for the stage's book page."""
import json, os, sys
from pathlib import Path
from dataclasses import replace
import jax, numpy as np
from life.experiments import stages as S
from life.run import load_population, make_simulate

WORLDS = int(os.environ.get("WORLDS", 4))
for key in sys.argv[1:]:
    s = S.STAGES[key]
    on, off = [], []
    for run in S.seed_runs(s):
        res = {}
        for label, brain in (("on", s.brain), ("off", replace(s.brain, in_adapt=()))):
            exp = S.make_exp(s, brain, "adapt_off", 1, 0)
            pop = load_population(run, exp, seed=0)
            rs, fn = s.build(exp)
            sim = make_simulate(exp, record=False)
            acc = []
            for k in jax.random.split(jax.random.PRNGKey(91), WORLDS):
                kr, ks = jax.random.split(k)
                rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
                st, _ = sim(rules, pop, ks)
                acc.append([float((s.fitness or S.default_fitness)(st).mean()), float(st["alive_ticks"].mean())])
            res[label] = np.mean(acc, 0)
        on.append(float(res["on"][0])); off.append(float(res["off"][0]))
        print(f"{key} {run.parent.name}: adaptation on {res['on'][0]:.0f} (lifetime {res['on'][1]:.0f}), off {res['off'][0]:.0f} "
              f"(lifetime {res['off'][1]:.0f}): {100 * res['off'][0] / res['on'][0]:.0f}% of intact", flush=True)
    st = lambda v: {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None, "seeds": [round(x, 1) for x in v]}
    out = {"columns": ["Fitness of the final population", "main"],
           "rows": [["Adaptation on (as evolved)", st(on)], ["Adaptation switched off", st(off)],
                    ["Off, % of on", st([100 * b / a for a, b in zip(on, off)])]],
           "caption": f"The same animals in the same {WORLDS} worlds with the adaptation of the look inputs on and off."}
    (Path(__file__).resolve().parents[2] / "book" / "data" / f"adapt_off_{key}.json").write_text(json.dumps(out))
