"""Lab probe (not adopted): does the actor's learned NoGo hold back picking a poison bush if the basal ganglia's
output also gates the motor neurons themselves (gpi -| USE, EAT, offset in their bias at rest), as SNr/GPi gate
brainstem motor centres? Pick box on the stage's start population (frozen 2.3s + hands_init), main vs main with
the gate; striatum learning on in both.

    W=4 python scripts/probes/gpi_gate_lab.py"""
import os
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from life import freeze
from life import labtests as T
from life.experiments import stages as S
from life.experiments.base import fixed
from life.experiments.chapter2 import HAND_ACTS
from life.run import make_layout

W = float(os.environ.get("W", 4.0))
s = S.STAGES["2.5"]
gated = replace(s.brain, projections=s.brain.projections + (
    fixed("gpi", "out", W, dst_range=HAND_ACTS, topology="one_to_one"),))
for label, brain in (("main", s.brain), (f"gpi gates USE/EAT (W {W})", gated)):
    exp = S.make_exp(s, brain, "lab", 1, 0)
    L = make_layout(exp)
    pop = s.adapt_init(freeze.load_frozen("2.3s", 0, exp), L)
    pop = pop._replace(mask=jnp.where(((np.asarray(L.rule) > 0) & (np.asarray(L.allowed) > 0))[None], 1.0, pop.mask))
    if brain is gated:   # gpi rests at 0.5: the motor neurons get the inhibition back in their bias
        out = L.region("out")
        pop = pop._replace(b=pop.b.at[:, out.start + 4:out.start + 6].add(W * 0.5))
    rs, rules_fn = s.build(exp)
    print(f"-- {label}")
    for proto in ("pick",):
        r = T.operant(exp, pop, rs, rules_fn, T.OPERANT[proto], n_agents=24, seed=0)
        for b0 in range(0, 48, 12):
            sl = slice(b0, b0 + 12)
            ev, po, pk = r["event"][sl], r["poison"][sl], r["picked"][sl]
            pe = (ev[po & pk] == 1).mean() if (po & pk).any() else np.nan
            ge = (ev[~po & pk] == 1).mean() if (~po & pk).any() else np.nan
            print(f"  trials {b0 + 1:2d}-{b0 + 12:2d}: picked poison {pk[po].mean():.2f} good {pk[~po].mean():.2f}"
                  f" | of picked eaten: poison {pe:.2f} good {ge:.2f}", flush=True)
