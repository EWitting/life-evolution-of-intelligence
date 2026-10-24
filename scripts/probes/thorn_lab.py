"""Dopamine at the pick in the thorny pick box (x.thorns, life/labtests.py OPERANT 'pick'): per block of trials and
berry kind, the share picked, the dopamine on the tick before the pick (the bush ahead) and the two ticks after it
(the prick, the berry in hand), and the share of picked berries eaten. A NoGo lesson for poison bushes needs the
dip after a poison pick to be deeper than after a good one.

    .venv/Scripts/python.exe scripts/probes/thorn_lab.py [--runs MAIN,CONTROL] [--trials 96] [--agents 24]"""
import argparse
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from life import freeze
from life import labtests as T
from life.experiments import stages as S
from life.run import make_layout

ap = argparse.ArgumentParser()
ap.add_argument("--stage", default="x.thorns")
ap.add_argument("--runs", default="")
ap.add_argument("--trials", type=int, default=96)
ap.add_argument("--agents", type=int, default=24)
ap.add_argument("--block", type=int, default=24)
a = ap.parse_args()
s = S.STAGES[a.stage]


def population(brain, learn_off=()):
    exp = S.make_exp(s, brain, "lab", 1, 0)
    L = make_layout(exp)
    if a.runs:
        from life.run import load_population
        pop = load_population(a.runs.split(",")[0 if brain is s.brain else 1], exp)
    else:
        pop = s.adapt_init(freeze.load_frozen("2.3s", 0, exp), L)
    pop = pop._replace(mask=jnp.where(((np.asarray(L.rule) > 0) & (np.asarray(L.allowed) > 0))[None], 1.0, pop.mask))
    if learn_off:
        off = np.isin(np.asarray(L.proj_id), [i for i, n in enumerate(L.proj_names) if n.split("->")[1].startswith(learn_off)])
        pop = pop._replace(eta=jnp.where(jnp.asarray(off)[None], 0.0, pop.eta))
    return exp, pop


proto = dict(T.OPERANT["pick"], trials=a.trials)
for c, (exp, pop) in {"main": population(s.brain), "actor off": population(s.brain, ("striatum",)),
                      "control": population(s.control_brain)}.items():
    rs, fn = s.build(exp)
    r = T.operant(exp, pop, rs, fn, proto, n_agents=a.agents)
    da, pt, pk, po, ev = r["da"], r["pick_tick"], r["picked"], r["poison"], r["event"]
    print(f"-- {c}\n  trials    picked p/g     da before p/g     da +1 p/g      da +2 p/g      eaten of picked p/g")
    for b0 in range(0, len(pk), a.block):
        sl = slice(b0, b0 + a.block)
        row = f"  {b0 + 1:3d}-{min(b0 + a.block, len(pk)):3d}"
        row += f"   {pk[sl][po[sl]].mean():.2f}/{pk[sl][~po[sl]].mean():.2f}"
        for off in (-1, 1, 2):
            vals = []
            for m in (po[sl] & pk[sl], ~po[sl] & pk[sl]):
                tr, ag = np.nonzero(m)
                t = pt[sl][tr, ag] + off
                ok = t >= 0
                vals.append(da[sl][tr[ok], ag[ok], t[ok]].mean() if ok.any() else np.nan)
            row += f"     {vals[0]:+.2f}/{vals[1]:+.2f}"
        row += "     " + "/".join(f"{(ev[sl][m & pk[sl]] == 1).mean():.2f}" for m in (po[sl], ~po[sl]))
        print(row, flush=True)
