"""Lab tests of evolved critic populations (labtests.conditioning), from a run's own config and population:
    python scripts/probes/lab_critic.py <run dir> [<run dir> ...] [FILE=checkpoint.npz]
Per run (24 creatures): extinction (6 painful meals, then 12 without pain) with the lab's trials (three ticks of pain,
0.3 after) and with world-like ones (one tick, nothing after): input drive of the aversive cells and P(bite) after 6
and after 18; the error (da) on every tick of the first, sixth and seventh trial (a run with a critic); and the XOR
meals of lab_mixed.py with world-like lessons (pain on X++ and X--, three good meals of X+- and X-+ per lesson, six
rounds; readout after an approach, first tick at the mouth): P(bite) on poison / good corners."""
import sys
from pathlib import Path
import numpy as np
from life.config import ExperimentConfig
from life.experiments import stages as S
from life.labtests import PROTOCOLS, conditioning
from life.run import load_population

FILE = next((a[5:] for a in sys.argv[1:] if a.startswith("FILE=")), "population.npz")
WORLD = dict(outcome_ticks=1, linger=0.0)
for run in [Path(a) for a in sys.argv[1:] if "=" not in a]:
    exp = ExperimentConfig.from_json((run / "config.json").read_text())
    pop = load_population(run, exp, file=FILE)
    critic = any(m.name == "da" for m in exp.brain.modulators)
    print(f"\n{run} ({FILE}){' critic' if critic else ''}")
    for label, extra in (("lab trials", {}), ("world-like trials", WORLD)):
        r = conditioning(exp, pop, {**PROTOCOLS["extinction"], **extra}, S.PALLIUM_SIM)
        a = r["tests"]["A"]
        print(f"  extinction, {label:17s}: drive {a[0, :, 3].mean():+.2f} -> {a[6, :, 3].mean():+.2f} after 6 painful -> "
              f"{a[18, :, 3].mean():+.2f} after 12 without;  P(bite) {a[0, :, 1].mean():.2f} -> {a[6, :, 1].mean():.2f} -> "
              f"{a[18, :, 1].mean():.2f}")
        if critic and not extra:
            sig = r["signal"].mean(1)
            for t, name in ((0, "first painful"), (5, "sixth painful"), (6, "first without pain")):
                print(f"    da, {name:18s}: " + " ".join(f"{v:+.2f}" for v in sig[t]))
    G, R = 3, 6
    rnd = [("pain", {"vis+0": "X++"}, 1), ("taste", {"vis+0": "X+-"}, G), ("pain", {"vis+0": "X--"}, 1),
           ("taste", {"vis+0": "X-+"}, G)]
    looks = ("X++", "X--", "X+-", "X-+")
    proto = dict(train=rnd * R, tests={c: {"vis+0": c} for c in looks}, lead=8, approach=True, test_far=8,
                 test_ticks=1, **WORLD)
    r = conditioning(exp, pop, proto, S.PALLIUM_SIM)
    t = {c: r["tests"][c][:, :, 1].mean(1) for c in looks}
    ends = np.cumsum([0] + [e[2] for e in rnd] * R)[::len(rnd)]
    print("  XOR meals, world-like lessons: P(bite) poison / good corners after rounds 0-6: " + "  ".join(
        f"{(t['X++'][i] + t['X--'][i]) / 2:.2f}/{(t['X+-'][i] + t['X-+'][i]) / 2:.2f}" for i in ends))
