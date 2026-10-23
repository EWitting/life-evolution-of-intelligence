"""Brain-free probe for the hands + actor stage (ADR-028): can the pallium (128 cells, k-WTA 2) also read the item in
hand and still tell the berries apart? Numpy only, no JAX run.

Looks as in the 2.3s world (the stage's own builder, XOR novel looks at similarity 0.5, drawn per life). A lesson of
the critic's rule is dW ~ code (presynaptic), so after lessons the learned value of a state is code . (sum of lesson
codes). Lessons: a pain while holding each poison berry with its own empty bush ahead (after a pick the one-berry bush
is empty). Read-out: the aversion to each held berry in two places, ahead its own empty bush (where it was picked) and
ahead nothing (where it can be put down), relative to the aversion to a poison held in front of its bush (= 1). Wanted:
good berries near 0 in both places, poison in front of nothing high (it is the same berry). And whether the codes with nothing in hand stay as they are (by construction they do: the held
input is 0 when the hand is empty).

Variants: the held look adds to the eye's input of the same cells (scale 1 or 2, 128 or 256 cells), or a separate
region of its own (`own`, 64 cells, k 2) next to the look-only pallium.

    .venv/Scripts/python.exe scripts/probes/pallium_held.py"""
import jax
import numpy as np

from life.experiments import stages as S

K = S.APPEARANCE
rng = np.random.default_rng(0)
s = S.STAGES["2.3s"]
exp = S.make_exp(s, s.brain, "probe", 1, 0)
rs, rules_fn = s.build(exp)
ids = lambda o: [rs.local(S.variant(v)[o]) for v in S.NOVEL]


def kwta(h, k):
    out = np.zeros_like(h)
    idx = np.argsort(-h, axis=-1)[..., :k]
    np.put_along_axis(out, idx, np.maximum(np.tanh(np.take_along_axis(h, idx, -1)), 0), -1)
    return out


def net(n, fan_in, k):
    """Per-slot random input weights as brain.init_state draws them: N(0, 1) / sqrt(fan-in), every synapse present."""
    w = rng.normal(size=(fan_in, n)) / np.sqrt(fan_in)
    return lambda x: kwta(x @ w, k)


def one_life(key, variant):
    rules = rules_fn(0, key)
    app = np.asarray(rules.appearance)[0, 0]          # [objects, K], first episode, first phase
    fv = np.asarray(rules.food_value)[0, 0]
    bush, empty, berry = (app[ids(o)] for o in (S.BUSH, S.EMPTY_BUSH, S.BERRY))
    poison = fv[ids(S.BERRY)] < 0                      # which novel types are poison this life
    z = np.zeros(K)
    n, scale, own = variant
    if own:
        eye, hand = net(128, K, 2), net(64, K, 2)
        code = lambda ahead, held: np.concatenate([eye(ahead), hand(held)])
    else:
        both = net(n, 2 * K, 2)
        code = lambda ahead, held: both(np.concatenate([ahead, scale * held]))
    lessons = sum(code(empty[i], berry[i]) for i in range(4) if poison[i])
    av = lambda ahead, held: code(ahead, held) @ lessons
    at_bush = np.array([av(empty[i], berry[i]) for i in range(4)])
    at_none = np.array([av(z, berry[i]) for i in range(4)])
    p, g = poison, ~poison
    unit = max(at_bush[p].mean(), 1e-9)   # the aversion to a poison held in front of its bush = 1
    return at_bush[g].mean() / unit, at_none[p].mean() / unit, at_none[g].mean() / unit


VARIANTS = {"same cells, 128, held x1": (128, 1.0, False), "same cells, 128, held x2": (128, 2.0, False),
            "same cells, 256, held x1": (256, 1.0, False), "own region, 64 cells": (0, 0.0, True)}
keys = jax.random.split(jax.random.PRNGKey(0), 200)
print("aversion relative to a poison held in front of its own empty bush (= 1)")
print(f"{'variant':28s} {'good, at its bush':>18s} {'poison, at nothing':>19s} {'good, at nothing':>18s}")
for name, v in VARIANTS.items():
    r = np.array([one_life(k, v) for k in keys])
    m, se = r.mean(0), r.std(0) / np.sqrt(len(r))
    print(f"{name:28s} " + " ".join(f"{a:11.2f} +-{b:4.2f}" for a, b in zip(m, se)))
