"""Where the hold box's lesson stops (stage 2.5): rates on the first tick of each trial (a berry in hand, nothing
ahead) of the cells between the held look and the hand (aversion to the held item `valence_av[2]`, `no_eat`,
`ingest`, the actor's Go / NoGo for USE, gpi, act), per block and berry kind, and the learning rates of the
projections that teach the held cells.

    SWALLOW=2.5 python scripts/probes/hold_trace.py [main|control]"""
import os
import sys
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from life import freeze
from life import labtests as T
from life.experiments import stages as S
from life.experiments.chapter2 import hands, actor, B24S
from life.run import make_layout

which = sys.argv[1] if len(sys.argv) > 1 else "main"
s = S.STAGES["2.5"]
if os.environ.get("SWALLOW"):
    h = hands(B24S, swallow=float(os.environ["SWALLOW"]))
    s = replace(s, brain=actor(h), control_brain=h)
brain = s.brain if which == "main" else s.control_brain
exp = S.make_exp(s, brain, "lab", 1, 0)
L = make_layout(exp)
pop = s.adapt_init(freeze.load_frozen("2.3s", 0, exp), L)
pop = pop._replace(mask=jnp.where(((np.asarray(L.rule) > 0) & (np.asarray(L.allowed) > 0))[None], 1.0, pop.mask))
pid, eta, w0 = np.asarray(L.proj_id), np.asarray(pop.eta), np.asarray(pop.w0 * pop.mask)
for i, n in enumerate(L.proj_names):
    if n.split("->")[1].startswith(("valence_av", "valence_app")) and (np.asarray(L.rule) > 0)[pid == i].any():
        m = pid == i
        print(f"  {n:36s} eta {eta[:, m].mean():.3f} (genomes with eta 0: {(eta[:, m].max(1) == 0).mean():.2f})  w0 {w0[:, m].mean():+.3f}")
cells = ["valence_av[2]", "valence_app[2]", "no_eat", "ingest"] + (["striatum_d1[0]", "striatum_d2[0]", "gpi[0]", "act[0]"]
                                                                  if which == "main" else []) + ["out[1]", "out[3]", "out[4]", "out[5]"]
rs, rules_fn = s.build(exp)
r = T.operant(exp, pop, rs, rules_fn, T.OPERANT["hold"], n_agents=24, seed=0, cells=tuple(cells))
po = r["poison"]
print(f"-- {which}: first-tick rates, poison / good")
print("  trials  " + "".join(f"{c:>16s}" for c in cells) + "   eaten p/g   first act tick p/g")
n = po.shape[0]
for b0 in range(0, n, 8):
    sl = slice(b0, b0 + 8)
    row = f"  {b0 + 1:3d}-{b0 + 8:3d}"
    for c in cells:
        x = r["cells"][c][sl]
        row += f"    {x[po[sl]].mean():5.2f}/{x[~po[sl]].mean():5.2f}"
    ev, tk = r["event"][sl], r["tick"][sl]
    row += f"   {(ev[po[sl]] == 1).mean():.2f}/{(ev[~po[sl]] == 1).mean():.2f}   {tk[po[sl] & (ev > 0)].mean():.1f}/{tk[~po[sl] & (ev > 0)].mean():.1f}"
    print(row, flush=True)
