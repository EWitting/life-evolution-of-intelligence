"""Classical conditioning lab tests (life.labtests.PROTOCOLS: acquisition, extinction, blocking) for brain variants,
on the parent population at generation 0:
    STAGE=x.critic DENSE=1 python scripts/probes/lab_conditioning.py base td tdp [NAME=value ...]
Per protocol and variant: the aversive cells (rate, and input drive, which does not saturate) for each test cue before training and after chosen trials (t0 =
before, tN = after trial N), P(bite) and critic value before and at the end, and the teaching signal summed over
the tick of the cue alone, summed over the three outcome ticks, and on the tick after the cue is gone, on the first
and last trial of each phase. PROTOCOLS=a,b picks protocols."""
import os, sys
import jax, numpy as np
from life.experiments import stages as S
from life.labtests import PROTOCOLS, conditioning
from life.run import load_population, latest_run, make_layout
from variants import variant

vs = [a for a in sys.argv[1:] if "=" not in a] or ["base"]
for a in sys.argv[1:]:
    if "=" in a:
        k_, v_ = a.split("=")
        setattr(S, k_, float(v_))
s = S.STAGES[os.environ.get("STAGE", "x.critic")]
which = os.environ.get("PROTOCOLS", ",".join(PROTOCOLS)).split(",")
print("settings:", [a for a in sys.argv[1:] if "=" in a])
for name in which:
    proto = PROTOCOLS[name]
    print(f"\n== {name}: " + ", then ".join(f"{n} x {kind} {'+'.join(cues.values())}" for kind, cues, n in proto["train"]))
    ends = np.cumsum([n for _, _, n in proto["train"]])
    show = sorted({0, 1, 2, *ends.tolist()})
    for v in vs:
        exp = S.make_exp(s, variant(s.brain, v), "lab", 1, 0)
        pop = load_population(os.environ.get("RUN") or latest_run(S.STAGES[s.parent].name), exp, seed=0)
        L = make_layout(exp)
        if os.environ.get("DENSE"):
            pl = (np.asarray(L.rule) > 0) & (np.asarray(L.allowed) > 0)
            pop = pop._replace(mask=jax.numpy.where(pl[None], 1.0, pop.mask))
        r = conditioning(exp, pop, proto, S.PALLIUM_SIM)
        for k, a in r["tests"].items():
            m, se = a.mean(1), a.std(1) / np.sqrt(a.shape[1])
            print(f"  {v:6s} {k:7s} aversion " + "  ".join(f"t{t}:{m[t, 0]:.2f}" for t in show)
                  + f"   P(bite) {m[0, 1]:.2f}->{m[-1, 1]:.2f}   value " + " ".join(f"{m[t, 2]:+.2f}" for t in show))
            print(f"  {'':6s} {'':7s} drive    " + "  ".join(f"t{t}:{m[t, 3]:+.2f}" for t in show) + f"   (s.e. {se[-1, 3]:.2f})")
        sig = r["signal"]
        line = []
        for (kind, cues, n), a, b in zip(proto["train"], np.r_[0, ends[:-1]], ends):
            for t in (a, b - 1):
                line.append(f"trial {t + 1}: cue {sig[t, :, 0].mean():+.2f} outcome {sig[t, :, 1:4].sum(-1).mean():+.2f} "
                            f"after {sig[t, :, 4].mean():+.2f}")
        print(f"  {v:6s} signal  " + " | ".join(line))
