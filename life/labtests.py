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
        s2 = B.init_state(g, L)._replace(w=st.w, ws=st.ws, wg=st.wg)
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
            st = B.init_state(g, L, key=jax.random.PRNGKey(1000 * seed + a))   # per-life wiring: one draw per animal
            for t in XOR_TYPES:
                res[(c, "before", t)].append(readout(g, st, obs(looks[t])))
            for _ in range(rounds):
                for t in ("poison ++", "good +-", "poison --", "good -+"):
                    for o in meal(looks[t], t.startswith("good")):
                        st = step(g, st, o)
            for t in XOR_TYPES:
                res[(c, "after", t)].append(readout(g, st, obs(looks[t])))
    return res


# ---------------------------------------------------------------- classical conditioning (chapter 2, the critic)
# Textbook protocols on single brains: a cue is a look in one eye column (the column straight ahead, 'vis+0', is the
# one the pallium reads; 'vis+30' is a side column, read only by the look synapses), shown on the tick before the
# outcome. A trial: the cue, the cue with three bites (pain 1 / taste 3 / nothing on the tick after each), then
# four blank ticks. Cues per animal: novel looks (A, B, C, A2 80% like A, -A the opposite colour direction).
#   acquisition   A -> pain, 12 trials: the aversion to A, its look-alike and its opposite after each trial
#   extinction    A -> pain 6 trials, then A -> nothing 12 trials
#   blocking      A -> pain 6, then A + B (side column) -> pain 6; control: C -> pain 6, then A + B -> pain 6.
#                 Kamin (1969): a cue added to one that already predicts the outcome is learned less
PROTOCOLS = {
    "acquisition": dict(train=[("pain", {"vis+0": "A"}, 12)],
                        tests={"A": {"vis+0": "A"}, "A2": {"vis+0": "A2"}, "-A": {"vis+0": "-A"}, "B": {"vis+0": "B"}}),
    "extinction": dict(train=[("pain", {"vis+0": "A"}, 6), ("none", {"vis+0": "A"}, 12)],
                       tests={"A": {"vis+0": "A"}}),
    "blocking": dict(train=[("pain", {"vis+0": "A"}, 6), ("pain", {"vis+0": "A", "vis+30": "B"}, 6)],
                     tests={"B side": {"vis+30": "B"}, "A": {"vis+0": "A"}}),
    # retroactive interference: does learning that B hurts lower the aversion to A? (the reversal world: the new
    # poisons are the other two XOR corners, X+- and X-+, of the old ones, X++ and X--)
    "interference": dict(train=[("pain", {"vis+0": "A"}, 6), ("pain", {"vis+0": "B"}, 6)], tests={"A": {"vis+0": "A"}}),
    "interference control": dict(train=[("pain", {"vis+0": "A"}, 6), ("none", {}, 6)], tests={"A": {"vis+0": "A"}}),
    "xor interference": dict(train=[("pain", {"vis+0": "X++"}, 1), ("pain", {"vis+0": "X--"}, 1)] * 4
                             + [("pain", {"vis+0": "X+-"}, 1), ("pain", {"vis+0": "X-+"}, 1)] * 4,
                             tests={"X++": {"vis+0": "X++"}, "X+-": {"vis+0": "X+-"}}),
    "xor interference control": dict(train=[("pain", {"vis+0": "X++"}, 1), ("pain", {"vis+0": "X--"}, 1)] * 4
                                     + [("none", {}, 8)], tests={"X++": {"vis+0": "X++"}, "X+-": {"vis+0": "X+-"}}),
    "blocking control": dict(train=[("pain", {"vis+0": "C"}, 6), ("pain", {"vis+0": "A", "vis+30": "B"}, 6)],
                             tests={"B side": {"vis+30": "B"}, "A": {"vis+0": "A"}}),
}


def conditioning(exp, pop, protocol: dict, sim: float, n_agents: int = 24, seed: int = 0, signal: str | None = None,
                 cells: tuple = ()) -> dict:
    """Runs one protocol (PROTOCOLS) on the first n_agents genomes at once. Returns
        tests: {label: [trials + 1, agents, 4]}  readout before training and after every trial: aversive cells,
               P(bite) (use or eat), critic value (aversive minus appetitive value cells, 0 without a critic), and
               the aversive cells' input drive (does not saturate as their rate does)
        signal: [trials, agents, ticks]          the teaching signal (`signal`, default da, else us_av) per tick
        cells:  {region: [trials, agents, ticks]} with `cells` (region names): their mean rate per tick
    A readout is taken as in xor_meals: a rested brain with only the learned weights, learning off, the cue shown
    for three ticks."""
    from .experiments import stages as S
    L = make_layout(exp); names = list(L.in_names); K = exp.vision.appearance_dim
    base, comp = S.colour_basis(S.BUSH, K)
    NA = min(n_agents, pop.b.shape[0])
    pop = jax.tree_util.tree_map(lambda x: x[:NA], pop)
    rng = np.random.default_rng(seed)
    looks = []
    for _ in range(NA):
        q, _ = np.linalg.qr(rng.normal(size=(K - 1, 4)))
        u = {"A": q[:, 0], "B": q[:, 1], "C": q[:, 2], "A2": 0.8 * q[:, 0] + 0.6 * q[:, 3], "-A": -q[:, 0],
             **{f"X{a}{b}": (int(f"{a}1") * q[:, 0] + int(f"{b}1") * q[:, 1]) / np.sqrt(2) for a in "+-" for b in "+-"}}
        looks.append({c: sim * base + np.sqrt(1 - sim ** 2) * (v @ comp) for c, v in u.items()})

    def obs(cues=None, taste=0.0, pain=0.0, near=0.8, command=False):
        o = np.zeros((NA, L.n_in), np.float32)
        if command:   # protocol option `command`: a copy of the bite command (needs BodyConfig.efference)
            o[:, names.index("efc_USE")] = 1.0
        o[:, names.index("food")] = 0.5
        if "temperature" in names:
            o[:, names.index("temperature")] = 0.5
        for col, c in (cues or {}).items():
            o[:, names.index(f"{col}.hit")], o[:, names.index(f"{col}.near")] = 1.0, near
            idx = [names.index(f"{col}.app{k}") for k in range(K)]
            for a in range(NA):
                o[a, idx] = looks[a][c]
        o[:, names.index("taste")] = taste
        o[:, names.index("pain")] = pain
        return jnp.asarray(o)

    def trial(kind, cues):
        out = dict(pain=dict(pain=1.0), taste=dict(taste=3.0), none={})[kind]
        # lead: ticks with the cue in view before the outcome; approach: all but the last of them seen from afar
        # (vis+0.near 0.4, two cells off; at the mouth 0.8), as a creature walking up to a bush
        # command: the tick on which each bite is commanded (one before its outcome) carries a copy of the command
        n = protocol.get("lead", 1)
        far = n - 1 if protocol.get("approach") else 0
        k, c = protocol.get("outcome_ticks", 3), bool(protocol.get("command"))
        return [obs(cues, near=0.4)] * far + [obs(cues)] * (n - far - 1) + [obs(cues, command=c)] \
            + [obs(cues, **out, command=c and i < k - 1) for i in range(k)] \
            + ([obs(pain=protocol.get("linger", 0.3))] if kind == "pain" else [obs()]) + [obs()] * 4

    one = lambda g, st, o: B.step(exp.brain, L, g, st, o, jnp.zeros(3), jax.random.PRNGKey(0))[0]
    step = jax.jit(jax.vmap(one))
    init = jax.vmap(lambda g, k: B.init_state(g, L, key=k))
    keys = jax.random.split(jax.random.PRNGKey(seed), NA)   # per-life wiring: one draw per animal
    frozen = pop._replace(eta=jnp.zeros_like(pop.eta))
    blank = obs()
    mi = list(L.mod_names).index(signal or ("da" if "da" in L.mod_names else "us_av"))
    av = L.region("valence_av")
    val = (L.region("value_av"), L.region("value_app")) if "value_av" in L.names else None

    # the test cue: test_far ticks seen from afar (as a creature walking up to it: the look adapts meanwhile), then
    # test_ticks at the mouth (default 3; 1 = the first tick at the mouth, where the world's bite is decided)
    t_far, t_mouth = protocol.get("test_far", 0), protocol.get("test_ticks", 3)

    def readout(st, o, o_far):
        s2 = init(frozen, keys)._replace(w=st.w, ws=st.ws, wg=st.wg)
        for _ in range(40):
            s2 = step(frozen, s2, blank)
        for _ in range(t_far):
            s2 = step(frozen, s2, o_far)
        for _ in range(t_mouth):
            s2 = step(frozen, s2, o)
        hall = jax.vmap(lambda x, w, b: x @ B.effective(L, w) + b)(s2.x, s2.w, frozen.b)
        p = np.asarray(jax.nn.softmax(hall[:, -L.n_out:] * exp.brain.logit_gain / exp.brain.action_temperature, axis=-1))
        x = np.asarray(s2.x)
        v = x[:, val[0]].mean(-1) - x[:, val[1]].mean(-1) if val else np.zeros(NA)
        return np.stack([x[:, av].mean(-1), p[:, 4] + p[:, 5], v, np.asarray(hall)[:, av].mean(-1)], -1)

    tests = {k: (obs(c), obs(c, near=0.4)) for k, c in protocol["tests"].items()}
    st = init(pop, keys)
    for _ in range(4):
        st = step(pop, st, blank)
    res = {k: [readout(st, *o)] for k, o in tests.items()}
    sig, rates = [], {c: [] for c in cells}
    for kind, cues, n in protocol["train"]:
        for _ in range(n):
            tr, cr = [], {c: [] for c in cells}
            for o in trial(kind, cues):
                st = step(pop, st, o)
                tr.append(np.asarray(st.mod[:, mi]))
                for c in cells:
                    cr[c].append(np.asarray(st.x[:, L.region(c)].mean(-1)))
            sig.append(np.stack(tr, -1))
            for c in cells:
                rates[c].append(np.stack(cr[c], -1))
            for k, o in tests.items():
                res[k].append(readout(st, *o))
    out = dict(tests={k: np.stack(v) for k, v in res.items()}, signal=np.stack(sig))
    if cells:
        out["cells"] = {c: np.stack(v) for c, v in rates.items()}
    return out


# ---------------------------------------------------------------- operant tests (hands and the actor, ADR-028)
# A box in which the creature's own actions decide what happens, as in a lever-pressing rat: per trial it starts
# holding a berry with nothing ahead (`hold`) or with an empty hand at a full bush (`pick`). The four berry types are
# the creature's novel XOR types (looks drawn per life by the stage's world, poison ++ and --). Per tick the brain
# chooses an action (sampled as in the world); the box applies it:
#   USE   empty hand at a full bush: the berry is in hand, the bush ahead is empty (with thorns,
#         WorldConfig.thorn_pain, the pick pricks: pain on the next tick); empty hand at a berry on the ground: picked up; holding something with nothing ahead: put down (the berry lies ahead)
#   EAT   holding a berry: eaten; taste 3 or pain 1 on the next tick (one berry, as in the world)
#   step or turn: nothing is ahead any more (it walked or turned away from the bush or the berry)
# A trial lasts `ticks`, then the box is cleared (`iti` blank ticks). Escape learning (Mowrer 1947): a creature that has
# learned that a held berry hurts can learn to put it down from the relief alone; nothing innate does it.
OPERANT = {
    "hold": dict(start="hold", trials=48, ticks=12, iti=6),
    "pick": dict(start="pick", trials=48, ticks=16, iti=6),
}
NONE, BUSH, EMPTY, BERRY = 0, 1, 2, 3


def operant(exp, pop, rs, rules_fn, protocol: dict, n_agents: int = 24, seed: int = 0, cells: tuple = ()) -> dict:
    """Runs an OPERANT protocol on the first n_agents genomes of `pop` (on exp's layout). `rules_fn`: the stage's world
    (its looks and poisons, one life per agent), `rs` its ruleset. Returns per trial and agent: the berry type (0-3), poison (0/1), the
    first event (0 none, 1 ate, 2 put down) and its tick, and whether it was picked at all (pick) and when first. `cells`: region names
    (or single cells, 'name[i]') whose mean rate on the trial's first tick is returned as well ({name: [trials, agents]}), and the dopamine signal
    on every tick (`da`: [trials, agents, ticks]) when the brain has one."""
    from .experiments import stages as S
    L = make_layout(exp); names = list(L.in_names); K = exp.vision.appearance_dim
    NA = min(n_agents, pop.b.shape[0])
    pop = jax.tree_util.tree_map(lambda x: x[:NA], pop)
    ids = [[rs.local(S.variant(v)[o]) for o in (S.BUSH, S.BERRY, S.EMPTY_BUSH)] for v in S.NOVEL]
    looks, poison = np.zeros((NA, 4, 3, K), np.float32), np.zeros((NA, 4), bool)
    phased = exp.world.switch_tick > 0   # rules [episodes, 2 phases, ...]: the box uses the first phase
    for a in range(NA):
        r = jax.tree_util.tree_map(lambda x: x[0, 0] if phased else x[0], rules_fn(0, jax.random.PRNGKey(7919 * seed + a)))
        app, fv = np.asarray(r.appearance), np.asarray(r.food_value)
        for i in range(4):
            looks[a, i] = app[ids[i]]
            poison[a, i] = fv[ids[i][1]] < 0
    ix = {n: names.index(n) for n in names}
    vis = [ix[f"vis+0.app{k}"] for k in range(K)]
    hand = [ix[f"held_app{k}"] for k in range(K)]
    efc = [ix[f"efc_{n}"] for n in ("NOOP", "FORWARD", "TURN_LEFT", "TURN_RIGHT", "USE", "EAT", "VOCALIZE")]

    def obs(held, ahead, kind, taste, pain, last):
        o = np.zeros((NA, L.n_in), np.float32)
        o[:, ix["food"]] = 0.5
        o[:, ix["skin"]] = 0.6           # warm: no cold, no warmth seeking (the box is not a cold world)
        o[:, ix["temperature"]] = 0.6
        for a in range(NA):
            if held[a] >= 0:
                o[a, ix["held"]] = 1.0
                o[a, hand] = looks[a, held[a], 1]
            if ahead[a] != NONE:
                o[a, ix["vis+0.hit"]], o[a, ix["vis+0.near"]] = 1.0, 0.8
                o[a, vis] = looks[a, kind[a], {BUSH: 0, BERRY: 1, EMPTY: 2}[ahead[a]]]
            if last[a] >= 0:
                o[a, efc[last[a]]] = 1.0
        o[:, ix["taste"]], o[:, ix["pain"]] = taste, pain
        return jnp.asarray(o)

    one = lambda g, st, o, k: B.step(exp.brain, L, g, st, o, jnp.zeros(3), k)
    step = jax.jit(jax.vmap(one))
    keys = jax.random.split(jax.random.PRNGKey(seed), NA)
    st = jax.vmap(lambda g, k: B.init_state(g, L, key=k))(pop, keys)
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed + 1)
    last = -np.ones(NA, int)
    out = {k: [] for k in ("type", "poison", "event", "tick", "picked", "pick_tick")}
    rec = {c: [] for c in cells}
    mi = list(L.mod_names).index("da") if "da" in L.mod_names else None
    das = []
    taste, pain = np.zeros(NA, np.float32), np.zeros(NA, np.float32)
    for trial in range(protocol["trials"]):
        kind = rng.permutation(np.tile(np.arange(4), NA // 4 + 1))[:NA] if trial % 4 == 0 else (kind + 1) % 4
        if protocol["start"] == "hold":
            held, ahead = kind.copy(), np.full(NA, NONE)
        else:
            held, ahead = -np.ones(NA, int), np.full(NA, BUSH)
        event, when, picked, pick_t = np.zeros(NA, int), np.full(NA, -1), np.zeros(NA, bool), np.full(NA, -1)
        da = []
        for t in range(protocol["ticks"] + protocol["iti"]):
            if t == protocol["ticks"]:     # the box is cleared
                held, ahead = -np.ones(NA, int), np.full(NA, NONE)
            key, k = jax.random.split(key)
            st, act = step(pop, st, obs(held, ahead, kind, taste, pain, last), jax.random.split(k, NA))
            act = np.asarray(act)
            if t == 0:
                for c in cells:   # a region's mean rate, or one cell as 'name[i]'
                    r, i = (c[:-1].split("[") if c.endswith("]") else (c, None))
                    sl = L.region(r) if i is None else L.region(r).start + int(i)
                    rec[c].append(np.asarray(st.x[:, sl].mean(-1) if i is None else st.x[:, sl]))
            if mi is not None:
                da.append(np.asarray(st.mod[:, mi]))
            taste, pain = np.zeros(NA, np.float32), np.zeros(NA, np.float32)
            if t >= protocol["ticks"]:
                last = act
                continue
            for a in range(NA):
                e = 0
                if act[a] == 4:
                    if held[a] < 0 and ahead[a] == BUSH:
                        held[a], ahead[a], e = kind[a], EMPTY, 3
                        pain[a] = exp.world.thorn_pain
                    elif held[a] < 0 and ahead[a] == BERRY:
                        held[a], ahead[a], e = kind[a], NONE, 3
                    elif held[a] >= 0 and ahead[a] == NONE:
                        held[a], ahead[a], e = -1, BERRY, 2
                elif act[a] == 5 and held[a] >= 0:
                    pain[a], taste[a] = (1.0, 0.0) if poison[a, held[a]] else (0.0, 3.0)
                    held[a], e = -1, 1
                elif act[a] in (1, 2, 3):
                    ahead[a] = NONE
                if e == 3:
                    pick_t[a] = t if not picked[a] else pick_t[a]
                    picked[a] = True
                elif e and event[a] == 0:
                    event[a], when[a] = e, t
            last = act
        out["type"].append(kind.copy()); out["poison"].append(poison[np.arange(NA), kind])
        out["event"].append(event); out["tick"].append(when); out["picked"].append(picked); out["pick_tick"].append(pick_t)
        if mi is not None:
            das.append(np.stack(da, -1))
    res = {k: np.stack(v) for k, v in out.items()}
    if cells:
        res["cells"] = {c: np.stack(v) for c, v in rec.items()}
    if das:
        res["da"] = np.stack(das)
    return res
