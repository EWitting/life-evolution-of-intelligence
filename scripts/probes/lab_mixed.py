"""Lab: pain lessons on A interleaved with good meals of B (`good` meals per lesson, as in the world, where a good bush
gives several bites and a poison one bite), trials with an approach (cue 7 ticks from afar, then at the mouth); the
readout as the world decides a bite: the cue 8 ticks from afar, P(bite) on the first tick at the mouth.
    python scripts/probes/lab_mixed.py [v=pnt] [good=3] [rounds=8] [xor=1] [outcome=1] [linger=0]
xor=1: the looks of the world (XOR corners): pain on X++ and X--, good meals of X+- and X-+; P(bite) on poison / good
corners (mean of the two) and a look never met. outcome=N: ticks of pain or taste per meal (default 3; a world bite
is one).
Frozen 2.3 populations (last generation, main runs, three seeds x 24 creatures)."""
import json, sys
from pathlib import Path
import numpy as np
from life.experiments import stages as S
from life.labtests import conditioning
from life.run import load_population
sys.path.insert(0, str(Path(__file__).parent))
from variants import variant

arg = dict(a.split("=") for a in sys.argv[1:] if "=" in a)
V, G, R, X = arg.get("v", "base"), int(arg.get("good", 3)), int(arg.get("rounds", 8)), arg.get("xor") == "1"
s = S.STAGES["2.3"]
D = json.load(open("book/data/2.3.json"))
rnd = ([("pain", {"vis+0": "X++"}, 1), ("taste", {"vis+0": "X+-"}, G), ("pain", {"vis+0": "X--"}, 1),
        ("taste", {"vis+0": "X-+"}, G)] if X else [("pain", {"vis+0": "A"}, 1), ("taste", {"vis+0": "B"}, G)])
looks = ("X++", "X--", "X+-", "X-+", "C") if X else ("A", "B", "C")
proto = dict(train=[e for e in rnd if e[2]] * R, tests={c: {"vis+0": c} for c in looks},
             lead=8, approach=True, test_far=8, test_ticks=1, outcome_ticks=int(arg.get("outcome", 3)), linger=float(arg.get("linger", 0.3)))
pb, dr = [], []
for r in D["runs"]["main"]:
    exp = S.make_exp(s, variant(s.brain, V), "lab", 1, 0)
    res = conditioning(exp, load_population(Path("runs") / r["dir"], exp), proto, S.PALLIUM_SIM)
    t = [res["tests"][c][:, :, 1].mean(1) for c in looks]
    pb.append([(t[0] + t[1]) / 2, (t[2] + t[3]) / 2, t[4]] if X else t)
    d = [res["tests"][c][:, :, 3].mean(1) for c in looks]   # input drive of the aversive cells
    dr.append([(d[0] + d[1]) / 2, (d[2] + d[3]) / 2, d[4]] if X else d)
pb, dr = np.mean(pb, 0), np.mean(dr, 0)  # look x (trials + 1)
ends = np.cumsum([0] + [e[2] for e in rnd if e[2]] * R)[::len([e for e in rnd if e[2]]) // (2 if X else 1)]
ends = np.cumsum([0] + [e[2] for e in rnd if e[2]] * R)
per = len([e for e in rnd if e[2]])     # training entries per round
print(f"{V}{' XOR looks' if X else ''}, {G} good meals per pain lesson; P(bite) painful look(s) / good look(s) / never met, after each round:")
for k in range(R + 1):
    i = ends[k * per]
    print(f"  round {k}: {pb[0, i]:.2f} / {pb[1, i]:.2f} / {pb[2, i]:.2f}    drive of the aversive cells {dr[0, i]:+.2f} / {dr[1, i]:+.2f} / {dr[2, i]:+.2f}")
