"""Is a stage's world crowded? The stage's starting population (the parent's frozen seed 0, with the stage's
`adapt_init`) and its newest evolved population, in the stage's world at several sizes with the same number of
animals and the same density of bushes. If animals take food from each other, both do better in a larger world, and
a population that forages better can look worse over the generations of a run. Worlds as in `stages.lesion`.

    uv run python scripts/probes/crowding.py <stage key> [size ...]     sizes in cells per side (default: the stage's)
    WORLDS=<n> (default 4), BOOK=1: also write the table book/data/crowding_<key>.json for the stage's page"""
import json, os, sys, time
from dataclasses import replace
import jax, jax.numpy as jnp, numpy as np
from life.experiments import stages as S
from life.freeze import load_frozen
from life.run import load_population, latest_run, make_layout, make_simulate

s0 = S.STAGES[sys.argv[1]]
WORLDS = int(os.environ.get("WORLDS", 4))
table = []
for size in [int(a) for a in sys.argv[2:]] or [s0.world.height]:
    table.append([f"{size} x {size}" + (" (this stage)" if size == s0.world.height else "")])
    s = replace(s0, world=replace(s0.world, height=size, width=size,   # the same share of the map in patches
                                  patches=round(s0.world.patches * size ** 2 / s0.world.height ** 2)))
    exp = S.make_exp(s, s.brain, "crowding_probe", 1, 0)
    lay = make_layout(exp)
    start = load_frozen(s.parent, 0, exp)
    if s.adapt_init:
        start = s.adapt_init(start, lay)
    if s.dense_plastic:   # as a run of the stage starts (stages.run_stage)
        start = start._replace(mask=jnp.where(((np.asarray(lay.rule) > 0) & (np.asarray(lay.allowed) > 0))[None], 1.0, start.mask))
    rs, fn = s.build(exp)
    sim = make_simulate(exp, record=False)
    extra = S.poison_metrics(rs)
    for label, pop in (("start", start), ("evolved", load_population(latest_run(s.name), exp, seed=0))):
        acc, t0 = [], time.time()
        for k in jax.random.split(jax.random.PRNGKey(123), WORLDS):
            kr, ks = jax.random.split(k)
            rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
            st, _ = sim(rules, pop, ks)
            e = extra(st, None)
            life = np.asarray(st["alive_ticks"])
            acc.append([float(st["fed_meal"].mean()), life.mean(), np.median(life), float(e["berries"]), float(e["poison_frac"]),
                        float((life >= exp.evolution.ticks_per_generation).mean())])
        acc = np.array(acc); m = acc.mean(0)
        table[-1] += [round(float(m[0])), f"{m[5]:.0%}"]
        print(f"[{s.key} {size}x{size}] {label:8s} well-fed lifetime {m[0]:5.0f} +-{acc[:, 0].std(ddof=1) / np.sqrt(WORLDS):.0f}  lifetime mean {m[1]:5.0f} "
              f"median {m[2]:5.0f}  alive at the cap {m[5]:.0%}  berries {m[3]:5.1f}  poison share {m[4]:.3f}  "
              f"({(time.time() - t0) / WORLDS:.0f} s per world)", flush=True)
if os.environ.get("BOOK"):
    out = {"columns": ["World, cells per side", "starting population", "reaching the end of life", "after evolution (seed 0)",
                       "reaching the end of life"], "rows": table,
           "caption": f"Well-fed lifetime, mean of {WORLDS} worlds, the same {s0.world.num_agents} creatures and the same density "
                      "of bushes in every size. The starting population is the population of the previous stage as this stage "
                      "begins."}
    with open(os.path.join("book", "data", f"crowding_{s0.key}.json"), "w", encoding="utf-8") as f:
        json.dump(out, f)
