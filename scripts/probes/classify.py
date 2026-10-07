"""Step 0 of checking a learning rule, no brain and no world: can the rule, in principle, turn bites into a
good / poison classifier of the looks used in the learning stages?

One aversive cell a = max(0, tanh(w . look)) reads the 8 appearance features; the animal meets bushes at random
(staple and ancestral poison weight 1, the four novel types NOVEL_WEIGHT each; two novel types are poison, drawn
per life) and bites with probability 1 - a (aversion blocks the bite). After a bite the rule changes w:

    additive   dw = eta * pain * look                      the 1.5 rule: pain only, weights only grow
    signed     dw = eta * (pain - good) * look             pain minus taste: every good meal lowers the weights
    safety     dw = eta * (pain - tanh(a) * good) * look   the v19-v20 safety cell: a good meal of something mistrusted
    delta      dw = eta * (pain - a) * look                prediction error: the cell's own activity is the prediction
    centred    dw = eta * pain * (look - usual look)       pain only, but the eyes have adapted to the look shared by
                                                           everything met (a running mean), so the lesson is about
                                                           what sets this food apart
    centred+safety                                         both

BLOCK in the environment: 'graded' (default; bites with probability 1 - a) or 'hard' (bites only while a < 0.1, as
the evolved 1.5 animals do: the aversive cells are silent at every good bite, so mistrusted food is never tasted).

Inherited weights: the smallest w that avoids the ancestral poison (w . look = 1.5) and leaves the staple alone.
Reported: probability of biting each class after n encounters, mean over lives.

    python scripts/probes/classify.py [NOVEL_SIM]"""
import os, sys
import numpy as np
from life.experiments import stages as S

SIM = float(sys.argv[1]) if len(sys.argv) > 1 else S.NOVEL_SIM
K = S.APPEARANCE
base, comp = S.colour_basis(S.BUSH, K)
dirs = [sgn * c for sgn in (1, -1) for c in comp]
look = lambda v, sim: base if v == 0 else sim * base + np.sqrt(1 - sim ** 2) * dirs[(v - 1) % len(dirs)]
X = np.stack([look(v, S.LOOKALIKE_SIMILARITY if v == 4 else SIM) for v in range(6)])   # type 4: ancestral poison
NOVEL = list(S.NOVEL)
P_MEET = np.array([S.NOVEL_WEIGHT if v in NOVEL else 1.0 for v in range(6)]); P_MEET /= P_MEET.sum()
W0 = np.linalg.pinv(X[[4, 0]]) @ np.array([1.5, 0.0])
act = lambda w, x: np.maximum(0.0, np.tanh(x @ w))
CHECK = (0, 20, 60, 200, 600)
HARD = os.environ.get("BLOCK", "graded") == "hard"


def life(rule, eta, rng, always_bite=False):
    poison = set(rng.choice(NOVEL, 2, replace=False)) | {4}
    w = W0.copy(); out = []; usual = P_MEET @ X
    for n in range(CHECK[-1] + 1):
        if n in CHECK:
            a = act(w, X)
            cls = lambda ts: float(np.mean([(a[t] < 0.1) if HARD else 1 - a[t] for t in ts]))
            out.append([cls([0]), cls([t for t in NOVEL if t not in poison]), cls([t for t in NOVEL if t in poison]), cls([4])])
        v = rng.choice(6, p=P_MEET); a = act(w, X[v])
        if always_bite or (a < 0.1 if HARD else rng.random() < 1 - a):
            pain = 1.0 if v in poison else 0.0
            safe = np.tanh(a) * (1 - pain)
            teach = {"additive": pain, "signed": pain - (1 - pain), "safety": pain - safe, "delta": pain - a,
                     "centred": pain, "centred+safety": pain - safe}[rule]
            w = w + eta * teach * (X[v] - usual if rule.startswith("centred") else X[v])
    return out


print(f"novel similarity {SIM}; cosine between looks (staple, 3 novel, ancestral poison, novel):")
print(np.round(X @ X.T, 2))
print("\nprobability of biting: staple / novel good / novel poison / ancestral poison")
print("block:", "hard" if HARD else "graded")
for rule in ("additive", "signed", "safety", "delta", "centred", "centred+safety"):
    for eta in (0.05, 0.3):
        r = np.mean([life(rule, eta, np.random.default_rng(s)) for s in range(200)], 0)
        print(f"{rule:14s} eta {eta:4.2f}  " + "   ".join(
            f"n={n:3d}: " + " ".join(f"{p:4.2f}" for p in row) for n, row in zip(CHECK, r)))
