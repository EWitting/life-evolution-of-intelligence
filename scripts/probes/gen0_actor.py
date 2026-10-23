"""Generation 0 of the hands + actor stage (2.5, ADR-028): the frozen 2.3s population (seed 0) on the stage's brains,
in the stage's world, the same worlds for every condition (paired). A diagnostic, not a gate.

    main          hands + actor (as run_stage starts it: every learned synapse exists)
    actor off     main with the striatum's learning off
    control       hands without the actor (the stage's control)
    2.3s          the parent brain in its own world (eating on the bite), for scale

Per condition: well-fed lifetime (fitness), lifetime, good and poison berries eaten per creature in each phase of the
reversal, and the paired difference from main with its standard error over worlds.

    WORLDS=16 python scripts/probes/gen0_actor.py"""
import os

import jax
import jax.numpy as jnp
import numpy as np

from life import freeze
from life.experiments import stages as S
from life.run import make_layout, make_simulate

WORLDS = int(os.environ.get("WORLDS", 16))
s = S.STAGES["2.5"]
import os as _os
from dataclasses import replace as _replace
from life.experiments.chapter2 import hands as _hands, actor as _actor, B24S as _B24S, ACT_EXPLORE as _AE
if _os.environ.get("SWALLOW") or _os.environ.get("EXPLORE"):
    _sw = float(_os.environ["SWALLOW"]) if _os.environ.get("SWALLOW") else None
    _h = _hands(_B24S, swallow=_sw)
    s = _replace(s, brain=_actor(_h, explore=float(_os.environ.get("EXPLORE", _AE))), control_brain=_h)
    print("SWALLOW", _sw, "EXPLORE", _os.environ.get("EXPLORE", _AE), flush=True)


def setup(stage, brain, dense, learn_off=()):
    exp = S.make_exp(stage, brain, "gen0", 1, 0)
    L = make_layout(exp)
    pop = freeze.load_frozen("2.3s", 0, exp)
    if stage is s and stage.adapt_init:   # the 2.5 start (not 2.3s's own, which its frozen population has had)
        pop = stage.adapt_init(pop, L)
    if dense:
        pop = pop._replace(mask=jnp.where(((np.asarray(L.rule) > 0) & (np.asarray(L.allowed) > 0))[None], 1.0, pop.mask))
    if learn_off:
        off = np.isin(np.asarray(L.proj_id), [i for i, n in enumerate(L.proj_names) if n.split("->")[1].startswith(learn_off)])
        pop = pop._replace(eta=jnp.where(jnp.asarray(off)[None], 0.0, pop.eta))
    return stage, exp, pop


parent = S.STAGES["2.3s"]
conds = {"main": setup(s, s.brain, True), "actor off": setup(s, s.brain, True, ("striatum",)),
         "control": setup(s, s.control_brain, False), "2.3s": setup(parent, parent.brain, False)}
res = {}
for label, (stage, exp, pop) in conds.items():
    rs, fn = stage.build(exp)
    sim = jax.jit(make_simulate(exp, record=False))
    berry = [rs.local(S.variant(t)[S.BERRY]) for t in S.NOVEL]
    rows = []
    for k in jax.random.split(jax.random.PRNGKey(31), WORLDS):
        kr, ks = jax.random.split(k)
        rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
        pv = np.asarray(rules.pain_value)                       # [2 phases, M]
        st, _ = sim(rules, pop, ks)
        e = np.asarray(st["eats"]); n = e.shape[0]
        per = lambda h, bad: sum(e[:, h, b].sum() for b in berry if (pv[h, b] > 0) == bad) / n
        rows.append([float(S.default_fitness(st).mean()), float(st["alive_ticks"].mean()),
                     per(0, False), per(0, True), per(1, False), per(1, True)])
        print(f"  {label}: world {len(rows)}/{WORLDS} fitness {rows[-1][0]:.0f}", flush=True)
    res[label] = np.array(rows)

cols = ["fitness", "alive", "good 1st", "poison 1st", "good 2nd", "poison 2nd"]
print("\ncondition    " + "".join(f"{c:>12s}" for c in cols) + "   fitness - main (paired, s.e.)")
for label, r in res.items():
    d = r[:, 0] - res["main"][:, 0]
    print(f"{label:12s} " + "".join(f"{v:12.1f}" for v in r.mean(0))
          + (f"   {d.mean():+.0f} +-{d.std(ddof=1) / np.sqrt(len(d)):.0f}" if label != "main" else ""))
