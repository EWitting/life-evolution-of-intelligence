"""Lab experiments on single animals in scripted situations (ADR-025: they take the place of the assay for the
learning stages; results go into the book's data files).

xor_meals: four novel bushes whose looks are the corners of two colour directions (poison ++ and --, good +- and
-+, as berry_world novel_xor), three berries per meal, pain or taste on the tick after each bite, in the order
poison ++, good +-, poison --, good -+, `rounds` times. Read out from a rested brain that keeps only what it learned,
the bush shown adjacent for three ticks: the aversive cells' activity and P(bite) per type, before and after, with
the learning of chosen projections switched off."""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

from . import brain as B
from .run import make_layout

XOR_TYPES = ("poison ++", "poison --", "good +-", "good -+")
_SIGNS = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]]) / np.sqrt(2)


def xor_meals(exp, pop, sim: float, conds: dict, rounds: int = 2, n_agents: int = 24, seed: int = 0) -> dict:
    """conds: {label: [projection names whose learning is off]}. Returns {(label, 'before'|'after', type):
    [[aversive activity, P(bite)] per agent]}."""
    from .experiments import stages as S
    L = make_layout(exp); names = list(L.in_names); K = exp.vision.appearance_dim
    base, comp = S.colour_basis(S.BUSH, K)
    pid = np.asarray(L.proj_id)

    def obs(seen=None, taste=0.0, pain=0.0):
        o = np.zeros(L.n_in, np.float32)
        o[names.index("food")] = 0.5
        if "temperature" in names:
            o[names.index("temperature")] = 0.5
        if seen is not None:
            o[names.index("vis+0.hit")], o[names.index("vis+0.near")] = 1.0, 0.8
            for k in range(K):
                o[names.index(f"vis+0.app{k}")] = seen[k]
        o[names.index("taste")] = taste
        o[names.index("pain")] = pain
        return jnp.asarray(o)

    def meal(look, good):
        out = dict(taste=3.0) if good else dict(pain=1.0)
        return [obs(look)] + [obs(look, **out) for _ in range(3)] + ([] if good else [obs(pain=0.3)]) + [obs()] * 4

    step = jax.jit(lambda g, st, o: B.step(exp.brain, L, g, st, o, jnp.zeros(3), jax.random.PRNGKey(0))[0])

    def readout(g, st, o):
        frozen = g._replace(eta=jnp.zeros_like(g.eta))
        s2 = B.init_state(g, L)._replace(w=st.w)
        for _ in range(40):
            s2 = step(frozen, s2, obs())
        for _ in range(3):
            s2 = step(frozen, s2, o)
        h = (s2.x @ B.effective(L, s2.w) + g.b)[-L.n_out:]
        p = np.asarray(jax.nn.softmax(h * exp.brain.logit_gain / exp.brain.action_temperature))
        return [float(s2.x[L.region("valence_av")].mean()), float(p[4] + p[5])]

    rng = np.random.default_rng(seed)
    res = {(c, w, t): [] for c in conds for w in ("before", "after") for t in XOR_TYPES}
    for a in range(min(n_agents, pop.b.shape[0])):
        g0 = jax.tree_util.tree_map(lambda x: x[a], pop)
        q, _ = np.linalg.qr(rng.normal(size=(K - 1, 2)))
        looks = {t: sim * base + np.sqrt(1 - sim ** 2) * ((_SIGNS[i] @ q.T) @ comp) for i, t in enumerate(XOR_TYPES)}
        for c, off in conds.items():
            eta = np.asarray(g0.eta).copy()
            eta[np.isin(pid, [list(L.proj_names).index(n) for n in off])] = 0.0
            g = g0._replace(eta=jnp.asarray(eta))
            st = B.init_state(g, L)
            for t in XOR_TYPES:
                res[(c, "before", t)].append(readout(g, st, obs(looks[t])))
            for _ in range(rounds):
                for t in ("poison ++", "good +-", "poison --", "good -+"):
                    for o in meal(looks[t], t.startswith("good")):
                        st = step(g, st, o)
            for t in XOR_TYPES:
                res[(c, "after", t)].append(readout(g, st, obs(looks[t])))
    return res
