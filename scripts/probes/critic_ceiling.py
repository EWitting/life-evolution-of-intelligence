"""How much of the pain can the 2.3 critic predict away in the lab? The frozen 2.3 populations (last generation, three
seeds, 24 creatures each), a new look A followed by pain on every trial (labtests `conditioning`), 60 trials:
    python scripts/probes/critic_ceiling.py [x1 x5 x20] [lead=1,4,8,16]
lead=a,b,..: the cue is in view for that many ticks before the first bite (labtests protocol "lead"; default 1),
as when a creature walks up to a bush it can see. xK multiplies the learning rates of pallium -> value_app / value_av by K (x1 = as evolved), to tell too few trials
from a ceiling. Per trial: the error on the cue tick, mean over the three bites, on the tick after the cue is gone
(pain 0.3 lingers), and the aversive value (value_av - value_app).
Expected fixed point (constant value c while the cue is in view, da = -pain - 1.5 c' + 3 c, the error of tick t
crediting the pallium of t-1): bites -1 + 1.5c, after -0.3 + 3c (the cue is gone), and the three bites plus the
tick after credit the cue: 3(-1 + 1.5c) + (-0.3 + 3c) = 0 -> c = 0.44, bites at -0.34 at best."""
import json, sys
from pathlib import Path
import jax.numpy as jnp, numpy as np
from life.experiments import stages as S
from life.labtests import conditioning
from life.run import load_population, make_layout

s = S.STAGES["2.3"]
D = json.load(open("book/data/2.3.json"))
N = 60
proto = dict(train=[("pain", {"vis+0": "A"}, N)], tests={"A": {"vis+0": "A"}})
show = [1, 2, 6, 12, 24, 40, 60]
leads = [int(v) for a in sys.argv[1:] if a.startswith("lead=") for v in a[5:].split(",")] or [1]
mults = [a for a in sys.argv[1:] if a.startswith("x")] or ["x1", "x5", "x20"]
for arg, lead in [(m, l) for m in mults for l in leads]:
    K = float(arg[1:])
    proto["lead"] = lead
    sig, val = [], []
    for r in D["runs"]["main"]:
        run = Path("runs") / r["dir"]
        exp = S.make_exp(s, s.brain, "lab", 1, 0)
        pop = load_population(run, exp)
        L = make_layout(exp)
        sl = lambda n: np.arange(L.region(n).start, L.region(n).stop)
        rows = sl("pallium")
        cols = np.r_[sl("value_app"), sl("value_av")]
        eta = pop.eta.at[:, rows[:, None], cols[None, :]].multiply(K)
        if r["seed"] == 0:
            print(f"\n{arg}, cue {lead} tick(s) before the bites: evolved pallium -> value eta (mean over creatures, nonzero) "
                  f"{float(pop.eta[:, rows[:, None], cols[None, :]].max(axis=(1, 2)).mean()):.3f}, now x{K:g}")
        res = conditioning(exp, pop._replace(eta=eta), proto, S.PALLIUM_SIM)
        sig.append(res["signal"].mean(1))                 # trials x ticks
        val.append(res["tests"]["A"][:, :, 2].mean(1))    # trials + 1
    sig, val = np.mean(sig, 0), np.mean(val, 0)
    for t in show:
        q = sig[t - 1, lead - 1:]                         # from the last cue-only tick on
        print(f"  trial {t:2d}: cue onset {sig[t - 1, 0]:+.2f}  cue, mean of later ticks {sig[t - 1, 1:lead].mean() if lead > 1 else 0:+.2f}"
              f"  bites {q[1:4].mean():+.2f}  after {q[4]:+.2f}   aversive value after the trial {val[t]:+.2f}")
