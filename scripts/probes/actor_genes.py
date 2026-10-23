"""What evolution does with the actor (stage 2.5): population means of the start strengths (w0 * mask, mean over a
projection's synapses), learning rates and biases of the hands and actor cells, at the start (frozen 2.3s with the
stage's start) and in a run's checkpoint.

    python scripts/probes/actor_genes.py runs/s2_5_actor/<run> [runs/s2_5_actor_control/<run>]"""
import sys
from pathlib import Path

import numpy as np

from life import brain as B, freeze
from life.experiments import stages as S
from life.run import load_population, make_layout

KEYS = ("striatum", "gpe", "gpi", "act", "ingest", "no_eat", "grasp", "no_feed", "in->striatum")
s = S.STAGES["2.5"]


def show(label, pop, L):
    pid, w = np.asarray(L.proj_id), np.asarray(pop.w0 * pop.mask)
    eta = np.asarray(pop.eta)
    print(f"-- {label}")
    for i, n in enumerate(L.proj_names):
        if not (any(k in n for k in KEYS) or n.endswith("->out")):
            continue
        m = pid == i
        if n.endswith("->out") and not any(k in n.split("->")[0] for k in KEYS + ("hungry", "valence")):
            continue
        cells = w[:, m]                                          # [genomes, synapses]
        print(f"  {n:40s} w {cells.mean():+.2f} (sd over genomes {cells.mean(1).std():.2f})  eta {eta[:, m].mean():.3f}")
    out = L.region("out")
    b = np.asarray(pop.b)
    print("  bias USE {:+.2f} EAT {:+.2f}".format(b[:, out.start + 4].mean(), b[:, out.start + 5].mean()))


for run in sys.argv[1:]:
    run = Path(run)
    control = "control" in str(run)
    exp = S.make_exp(s, s.control_brain if control else s.brain, "probe", 1, 0)
    L = make_layout(exp)
    start = s.adapt_init(freeze.load_frozen("2.3s", 0, exp), L)
    show(f"{'control' if control else 'main'} start", start, L)
    show(f"{run} checkpoint", load_population(run, exp, file="checkpoint.npz"), L)
