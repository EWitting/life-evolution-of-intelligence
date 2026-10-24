"""Lab tests of the hands + actor stage (2.5, ADR-028), brain only, before generation 0: operant boxes
(life/labtests.py OPERANT) on the frozen 2.3s population (seed 0) moved onto the stage's layouts.

    main          hands + actor
    actor off     main with the striatum's learning off (the actor stays as it is born: silent)
    control       hands without the actor (the stage's control)

Per block of trials, for poison and good berries: the share eaten, put down, or neither (still held when the box is
cleared), and for `pick` also the share picked. Escape learning shows as 'put down' rising for poison in main only.

    .venv/Scripts/python.exe scripts/probes/lab_actor.py [hold|pick ...] [--agents 24] [--seed 0] [--runs MAIN,CONTROL]
        [--stage x.thorns]   (thorny bushes; with --runs of 2.5: its evolved creatures in the thorn box)"""
import argparse

import jax.numpy as jnp
import numpy as np

from life import freeze
from life import labtests as T
from life.experiments import stages as S
from life.run import make_layout

ap = argparse.ArgumentParser()
ap.add_argument("protocols", nargs="*", default=["hold", "pick"])
ap.add_argument("--agents", type=int, default=24)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--block", type=int, default=12)
ap.add_argument("--explore", type=float, default=None, help="start of held -> Go (USE), instead of ACT_EXPLORE")
ap.add_argument("--only", default="", help="comma-separated conditions to run (main, actor off, control)")
ap.add_argument("--stage", default="2.5", help="the stage whose brains and world are used (e.g. x.thorns)")
ap.add_argument("--runs", default="", help="'<main run>,<control run>': evolved populations instead of the stage's start")
a = ap.parse_args()

s = S.STAGES[a.stage]
import os as _os
from dataclasses import replace as _replace
from life.experiments.chapter2 import hands as _hands, actor as _actor, B24S as _B24S, ACT_EXPLORE as _AE
if _os.environ.get("SWALLOW") or _os.environ.get("EXPLORE"):
    _sw = float(_os.environ["SWALLOW"]) if _os.environ.get("SWALLOW") else None
    _h = _hands(_B24S, swallow=_sw)
    s = _replace(s, brain=_actor(_h, explore=float(_os.environ.get("EXPLORE", _AE))), control_brain=_h)
    print("SWALLOW", _sw, "EXPLORE", _os.environ.get("EXPLORE", _AE), flush=True)
if a.explore is not None:
    from life.experiments.chapter2 import actor, B25H
    from dataclasses import replace
    s = replace(s, brain=actor(B25H, explore=a.explore))


def population(brain, learn_off=()):
    exp = S.make_exp(s, brain, "lab", 1, 0)
    L = make_layout(exp)
    if a.runs:
        from life.run import load_population
        pop = load_population(a.runs.split(",")[0 if brain is s.brain else 1], exp)
    else:
        pop = freeze.load_frozen("2.3s", a.seed, exp)
        pop = s.adapt_init(pop, L) if s.adapt_init else pop
    # as run_stage does for main (dense_plastic): every synapse of a learned projection exists
    pop = pop._replace(mask=jnp.where(((np.asarray(L.rule) > 0) & (np.asarray(L.allowed) > 0))[None], 1.0, pop.mask))
    if learn_off:
        off = np.isin(np.asarray(L.proj_id), [i for i, n in enumerate(L.proj_names) if n.split("->")[1].startswith(learn_off)])
        pop = pop._replace(eta=jnp.where(jnp.asarray(off)[None], 0.0, pop.eta))
    return exp, pop


conds = {"main": population(s.brain), "actor off": population(s.brain, ("striatum",)),
         "control": population(s.control_brain)}
if a.only:
    conds = {k: v for k, v in conds.items() if k in a.only.split(",")}
for proto in a.protocols:
    print(f"\n== {proto} ({T.OPERANT[proto]})")
    for c, (exp, pop) in conds.items():
        rs, rules_fn = s.build(exp)
        r = T.operant(exp, pop, rs, rules_fn, T.OPERANT[proto], n_agents=a.agents, seed=a.seed)
        n = r["event"].shape[0]
        print(f"-- {c}")
        print("  trials     " + "poison: ate  put down  neither".ljust(34) + "good: ate  put down  neither"
              + ("      picked poison / good" if proto == "pick" else ""))
        for b0 in range(0, n, a.block):
            sl = slice(b0, b0 + a.block)
            ev, po, pk = r["event"][sl], r["poison"][sl], r["picked"][sl]
            row = f"  {b0 + 1:3d}-{min(b0 + a.block, n):3d}  "
            for m in (po, ~po):
                if proto == "pick":
                    m = m & pk
                tot = max(m.sum(), 1)
                row += "       " + "  ".join(f"{(ev[m] == e).sum() / tot:7.2f}" for e in (1, 2, 0)) + "  "
            if proto == "pick":
                row += f"       {pk[po].mean():.2f} / {pk[~po].mean():.2f}"
            print(row, flush=True)
