"""Worlds of the stages: berry-bush types sliced from OHOL, their looks (fixed or drawn per life), poison,
patches and schedules. A stage's own world (its WorldConfig and the builder that sets its berry types) is
defined next to its brain in chapter1.py, chapter2.py or side_paths.py."""
from __future__ import annotations
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from life.config import (ExperimentConfig, WorldConfig, VisionConfig, BodyConfig, BrainConfig, EvolutionConfig,
                         RegionSpec as R, ProjectionSpec as P, ModulatorSpec as Mod)

from life import ohol


# ------------------------------------------------------------------ OHOL objects used by the stages
BUSH, BERRY, EMPTY_BUSH = 30, 31, 279          # Wild Gooseberry Bush -> Gooseberry (6 uses) -> Empty bush
ONION_PLANT, ONION = 805, 808                   # Wild Onion (single pick: the plant disappears)
GARLIC_PLANT, GARLIC = 4251, 4252
HOT_SPRING = 2140                               # natural heat source (heatValue 3)
REGROW_FOOD = 24.0                              # an empty bush regrows after the time in which a resting agent burns
                                                # this much food (2 x what a bush holds), so waiting at a bush
                                                # never pays, whatever the metabolic rate of the stage


BUSH_BERRIES = 6                                # berries a bush carries (the OHOL gooseberry bush: 6). Probes vary it:
                                                # smaller bushes, with the density raised to match, mean more
                                                # decisions per life for the same food


def regrow_ticks(exp, berries: float | None = None) -> int:
    berries = BUSH_BERRIES if berries is None else berries
    return int(min(exp.world.max_decay_ticks, round(REGROW_FOOD * berries / 6 / exp.world.hunger_per_tick)))



# ------------------------------------------------------------------ worlds

COLOURS = ["Red", "Blue", "Yellow", "Purple", "White", "Black", "Pink", "Orange"]


def variant(v: int) -> dict:
    """OHOL ids of berry-bush type v: 0 = the real gooseberry bush, v >= 1 = a look-alike with its own appearance."""
    if v == 0:
        return {BUSH: BUSH, BERRY: BERRY, EMPTY_BUSH: EMPTY_BUSH}
    return {o: 100000 + 1000 * v + o for o in (BUSH, BERRY, EMPTY_BUSH)}


def berry_world(exp: ExperimentConfig, n_types: int = 4, poison: tuple = (), per_life_pool: tuple = (),
                per_life_k: int = 1, onions: bool = True, springs: float = 0.0, poison_food: float = -1.0,
                poison_pain: float = 1.0, reverse: bool = False, per_life_sets: tuple = (),
                appearance_mode: str = "lookalike", novel_looks: tuple = (), novel_sim: float | None = None,
                weights: dict | None = None, duds: tuple = (), berries: int | None = None,
                shared_colour: bool = False, novel_xor: bool = False, one_berry: float = 0.0, thorns: bool = False):
    """Several berry-bush types (the OHOL gooseberry and colour look-alikes; all behave like the gooseberry:
    6 berries, then empty, regrowing after regrow_ticks(exp)), optional wild onions and hot springs.
    poison: types whose berries always drain food and hurt (an inheritable fact). per_life_pool/per_life_k:
    per world, k types drawn from the pool are poison as well (only lifetime learning can know which).
    reverse: with a pool of two and k=1, the poison swaps to the other type at WorldConfig.switch_tick.
    per_life_sets: explicit alternatives instead of pool/k, e.g. ((1, 4), (2, 3)) for the XOR world.
    appearance_mode 'xor': types 1-4 look like the original plus (+-d1 +-d2)/sqrt2 (two binary features), types
    5 and 6 plus +-d3 (see lookalike_appearance).
    novel_looks: types whose colour is drawn anew for every life (a random direction, shared by the type's bush,
    berry and empty bush; similarity novel_sim to the original, default LOOKALIKE_SIMILARITY), so no inherited
    weight can know them (ADR-017). Combine with per_life_pool so that their meaning is drawn per life as well.
    weights: {type: spawn weight} overriding the default of 1 per bush type.
    duds: types whose bush looks like a bush but yields nothing (USE does nothing): a meaningless stimulus. With a
    look drawn per life (novel_looks) evolution cannot learn to ignore it by inheritance.
    berries: berries per bush (default BUSH_BERRIES); the regrowth time scales with it, so the food per bush and
    tick stays the same and a smaller bush only means more, smaller visits.
    shared_colour (chapter 2): a novel type's colour is one vector, the same for its bush, berry and empty bush
    (drawn in the directions that none of the three original looks uses). Without it the three get the same
    coefficients in three different bases, so a bush and its own empty bush are no more alike than chance.
    one_berry (> 0): every bush carries one berry worth `one_berry` berries of food (a poison berry stays one bite
    and poison_food), regrowing in the time that many berries take: a bite is a whole visit, so stopping a visit at
    the first painful berry no longer helps and only judging the look does (STAGE_LOG v58, v59).
    thorns: every bush that carries berries is thorny: a pick pricks (WorldConfig.thorn_food, thorn_pain)."""
    berries = int(BUSH_BERRIES if berries is None else berries)
    if one_berry:
        berries = 1
    data = ohol.load()
    ids = [BUSH, BERRY, EMPTY_BUSH] + ([ONION_PLANT, ONION] if onions else []) + ([HOT_SPRING] if springs else [])
    sets = [(variant(v), COLOURS[v - 1] + " ") for v in range(1, n_types)]
    rs = ohol.slice_ruleset(data, ids, ticks_per_second=exp.world.ticks_per_ohol_second,
                            max_decay_ticks=exp.world.max_decay_ticks, clone_sets=sets,
                            extra_decays={EMPTY_BUSH: (BUSH, regrow_ticks(exp, one_berry or berries))})
    for v in range(n_types):
        b = rs.local(variant(v)[BUSH])
        rs.num_uses[b] = berries
        if one_berry:   # OHOL counts uses only above 1: a bush of one use would never take its last-use transition
            rs.use_table[:, b] = rs.last_use_table[:, b]
            rs.food_value[rs.local(variant(v)[BERRY])] *= one_berry
    for v in duds:   # no transition from USE on this bush
        b = rs.local(variant(v)[BUSH])
        rs.use_table[:, b] = -1
        rs.last_use_table[:, b] = -1
    spawn = np.zeros(rs.size, np.float32)
    for v in range(n_types):
        spawn[rs.local(variant(v)[BUSH])] = (weights or {}).get(v, 1.0)
    if onions:
        spawn[rs.local(ONION_PLANT)] = 0.5
    if springs:
        spawn[rs.local(HOT_SPRING)] = springs
        # an animal can stand in a spring. As a blocking object (the OHOL default, and the only one in these worlds)
        # it was a trap: the animals walked into it and stayed, facing a spring on a third of their ticks, and
        # springs alone cut fitness by 60% before any cold (STAGE_LOG 2026-10-08)
        rs.blocks[rs.local(HOT_SPRING)] = False
    berry = [rs.local(variant(v)[BERRY]) for v in range(n_types)]

    app = lookalike_appearance(rs, n_types, exp.vision.appearance_dim, mode=appearance_mode)
    thorny = np.zeros(rs.size, bool)
    if thorns:
        thorny[[rs.local(variant(v)[BUSH]) for v in range(n_types)]] = True

    def arrays(bad: tuple):
        fv, pv = rs.food_value.copy(), np.zeros(rs.size, np.float32)
        for v in bad:
            fv[berry[v]], pv[berry[v]] = poison_food, poison_pain
        return rs.to_arrays(exp.vision.appearance_dim, spawn_weight=spawn, food_value=fv, pain_value=pv,
                            appearance=app, thorny=thorny)

    if not per_life_pool and not per_life_sets and not novel_looks:
        rules = arrays(tuple(poison))
        return rs, (lambda gen, key: rules)
    import itertools
    alts = per_life_sets or (tuple(itertools.combinations(per_life_pool, per_life_k)) if per_life_pool else ((),))
    combos = [tuple(poison) + tuple(c) for c in alts]
    variants = [arrays(c) for c in combos]
    stacked = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *variants)
    E = exp.evolution.episodes

    K = exp.vision.appearance_dim
    objs = (BUSH, BERRY, EMPTY_BUSH)
    basis = [colour_basis(o, K) for o in objs]
    bases, comps = jnp.asarray(np.stack([b for b, _ in basis])), jnp.asarray(np.stack([c for _, c in basis]))
    novel_ids = np.array([[rs.local(variant(v)[o]) for o in objs] for v in novel_looks], np.int32).reshape(-1, 3)
    nsim = LOOKALIKE_SIMILARITY if novel_sim is None else novel_sim

    shared = jnp.asarray(np.linalg.qr(np.column_stack([b for b, _ in basis] + [np.eye(K)]))[0][:, 3:K].T)   # [K-3, K]

    if novel_xor:   # corner signs [variant, novel type, 2]: poisons of the variant ++ and --, the others +- and -+
        assert len(novel_looks) == 4 and all(len(set(c) & set(novel_looks)) == 2 for c in combos)
        corners = np.zeros((len(combos), 4, 2), np.float32)
        for vi, c in enumerate(combos):
            bad = [i for i, v in enumerate(novel_looks) if v in c]; good = [i for i in range(4) if i not in bad]
            for i, sg in zip(bad + good, ((1, 1), (-1, -1), (1, -1), (-1, 1))):
                corners[vi, i] = sg
        corners = jnp.asarray(corners)

    def novel_appearance(key, pick):   # [E, types, 3 objects, K]
        if novel_xor:
            q, _ = jnp.linalg.qr(jax.random.normal(key, (E, K - 3 if shared_colour else K - 1, 2)))   # [E, D, 2]
            u = jnp.einsum("etj,edj->etd", corners[pick], q) / np.sqrt(2)
        else:
            u = jax.random.normal(key, (E, len(novel_looks), K - 3 if shared_colour else K - 1))
            u = u / jnp.linalg.norm(u, axis=-1, keepdims=True)
        if shared_colour:
            return nsim * bases[None, None] + np.sqrt(1 - nsim ** 2) * (u @ shared)[:, :, None, :]
        return nsim * bases[None, None] + np.sqrt(1 - nsim ** 2) * jnp.einsum("enc,ock->enok", u, comps)

    def rules_fn(gen, key):
        kp, ka = jax.random.split(key)
        pick = jax.random.randint(kp, (E,), 0, len(variants))
        if reverse:   # [E, 2 phases, ...]: phase 2 = the complementary choice (combinations list them mirrored)
            assert len(variants) % 2 == 0
            idx = jnp.stack([pick, len(variants) - 1 - pick], axis=1)
            rules = jax.tree_util.tree_map(lambda a: a[idx], stacked)
        else:
            rules = jax.tree_util.tree_map(lambda a: a[pick], stacked)
        if novel_looks:
            new = novel_appearance(ka, pick)
            app = rules.appearance
            app = app.at[:, :, novel_ids].set(new[:, None]) if reverse else app.at[:, novel_ids].set(new)
            rules = rules._replace(appearance=app)
        return rules
    return rs, rules_fn


# In stages 1.1-1.4 a poison berry costs what a good one gives (OHOL food points). At a third of that the innate
# aversive reflex was not worth keeping and evolution turned it down.
POISON_INNATE = -3.0
LOOKALIKE_SIMILARITY = 0.8   # cosine similarity of a colour variant's appearance to the original object


def colour_basis(orig: int, k: int):
    """(appearance of the original object, orthonormal basis [k-1, k] of its complement = the colour space)."""
    from life.ruleset import appearance_for
    base = appearance_for(orig, k)
    q, _ = np.linalg.qr(np.column_stack([base, np.eye(k)]))
    return base, q[:, 1:k].T


def lookalike_appearance(rs, n_types: int, k: int, sim: float = LOOKALIKE_SIMILARITY, mode: str = "lookalike") -> dict:
    """Colour variants look *like* the original: appearance = sim * original + sqrt(1 - sim^2) * colour, where the
    colour vector (one per variant, orthogonal to the original) is shared by the variant's bush, berry and empty
    bush. Hash-based appearances (ADR-006) would make every variant an unrelated object."""
    out = {}
    for orig in (BUSH, BERRY, EMPTY_BUSH):
        base, comp = colour_basis(orig, k)   # colours = +-basis directions (maximally spread)
        dirs = [sgn * c for sgn in (1, -1) for c in comp]
        if mode == "xor":
            r = 1 / np.sqrt(2)
            dirs = [r * (comp[0] + comp[1]), r * (comp[0] - comp[1]), r * (-comp[0] + comp[1]), r * (-comp[0] - comp[1]),
                    comp[2], -comp[2]]
        for v in range(1, n_types):
            out[rs.local(variant(v)[orig])] = sim * base + np.sqrt(1 - sim ** 2) * dirs[(v - 1) % len(dirs)]
    return out


def berry_ids(rs) -> list[int]:
    return [i for i, n in enumerate(rs.names) if n.endswith("Gooseberry")]


def poison_metrics(rs):
    """row_extra factory: fraction of berries eaten that were poison (painful), whole life and per half."""
    berries = berry_ids(rs)

    def extra(stats, fit):
        e = stats["eats"]                       # [N, 2, M]; column 0 = painful things eaten
        out = {}
        for tag, sl in (("", slice(None)), ("_early", 0), ("_late", 1)):
            n = e[:, sl][..., berries].sum()
            out["poison_frac" + tag] = e[:, sl][..., 0].sum() / jnp.maximum(n, 1.0)
        out["berries"] = e[..., berries].sum() / e.shape[0]
        return out
    return extra


def calm_first(generations: int):
    """Schedule: for the first `generations` generations both phases of a life are the first one (no reversal), then
    the world as built (the poison and good kinds swap at WorldConfig.switch_tick)."""
    def wrap(fn):
        def rules(gen, key):
            r = fn(gen, key)
            return r if gen >= generations else jax.tree_util.tree_map(lambda a: jnp.repeat(a[:, :1], 2, axis=1), r)
        return rules
    return wrap
