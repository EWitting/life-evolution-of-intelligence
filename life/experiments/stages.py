"""The brain-evolution sequence of docs/BRAIN_EVOLUTION.md as runnable stages.

Every stage = a world (OHOL slice + world mechanics), a brain (regions/projections added on top of the previous
stage) and a fitness. Stages form one lineage: by default a stage warm-starts from the newest run of its parent
stage, remapped onto the new layout (new parts start silent, brain.remap_genomes). The optional control runs the
*parent's* brain in the *new* world from the same starting population, so the new module is the only difference.

    python -m life.experiments.stages list
    python -m life.experiments.stages 1.0 --generations 150
    python -m life.experiments.stages 1.1 --generations 100 [--control] [--init-from <run dir> | --init-from none]
    python -m life.experiments.stages replicate 1.1                     # two more seeds of main and control
    python -m life.experiments.stages summary 1.1                       # main vs control over the seeds
    python -m life.experiments.stages versus 1.1                        # main and control agents in the same worlds
    python -m life.experiments.stages lesions 1.1                       # is the circuit used? lesions over the seeds
    python -m life.experiments.stages lesion runs/s1_1_valence/<ts>     # silence each region, measure behaviour
    python -m life.experiments.stages respond runs/s1_1_valence/<ts>    # region activity/actions per object seen or held

Results and interpretation: docs/STAGE_LOG.md.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass, field, replace
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from life import ohol
from life.config import (ExperimentConfig, WorldConfig, VisionConfig, BodyConfig, BrainConfig, EvolutionConfig,
                         RegionSpec as R, ProjectionSpec as P, ModulatorSpec as Mod)
from life.run import run_evolution, load_population, latest_run, RUNS_DIR

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



# chapter 1 senses: a few coarse, short-range "eyes" (early bilaterians had simple photoreceptors/chemosensing)
# appearance: 8 features per object (1 shared by a family of look-alikes + 7 for its 'colour'), so that different
# kinds are nearly orthogonal; with 4, an aversion learned for one berry type spread to all of them
APPEARANCE = 8
VISION_CH1 = VisionConfig(columns=5, fov_degrees=120.0, range=5, appearance_dim=APPEARANCE)


@dataclass
class Stage:
    key: str                      # "1.0"
    name: str                     # run directory name
    parent: str | None            # stage this one builds on (warm start + control architecture)
    brain: BrainConfig
    world: WorldConfig
    vision: VisionConfig
    body: BodyConfig
    build: Callable               # (exp) -> (ruleset, rules_for_generation)
    fitness: Callable = None
    row_extra: Callable | None = None   # (ruleset) -> (stats, fitness) -> {column: value}
    generations: int = 100
    ticks: int = 1000
    plastic: bool = False
    eta_max: float = 0.5          # evolution clips learning rates to [0, eta_max]
    evolution: tuple = ()         # ((EvolutionConfig field, value), ...) for this stage only (trials)
    dense_plastic: bool = False   # warm start: every synapse of a learned projection exists (the inherited ones keep
                                  # their weight, the others start at 0); an absent synapse could not learn
    adapt_init: Callable | None = None   # (population, layout) -> population: a change to the warm-start population
                                  # after it is remapped onto this stage's layout (e.g. interpolate_eyes)
    control_brain: BrainConfig | None = None   # the control's brain if not the parent's (e.g. with a new gene)
    schedule: Callable | None = None   # (rules_for_generation) -> rules_for_generation, applied during evolution only:
                                  # a world that changes over the generations (lesions, the book and probes, which ask
                                  # for generation 0, get the stage's world as built)
    notes: str = ""


STAGES: dict[str, Stage] = {}


def stage(s: Stage) -> Stage:
    STAGES[s.key] = s
    return s


def default_fitness(stats):
    """Well-fed lifetime from the first meal (v6-v18, and again from v23): the stomach level, as a fraction of full,
    summed over the ticks alive after the first meal. An animal that never eats scores zero. Survival is what counts,
    so a bite on a full stomach earns nothing and cold, which burns the reserve faster, costs in proportion. (v19-v22
    used energy acquired, `fitness_energy`: with recombination the two repair a damaged population equally well,
    STAGE_LOG v23, but energy acquired pays for every bite, hungry or not, so no need state can be selected.)"""
    return stats["fed_meal"]


def fitness_energy(stats):
    """Energy acquired (the fitness of v19-v22), kept for comparisons: all food eaten, not capped by the stomach."""
    return stats["eaten"]


def fitness_v4(stats):
    """The v4 fitness, kept for comparisons: gross food eaten (also on a full stomach) - pain + survival bonus."""
    return stats["eaten"] - stats["pain"] + 0.01 * stats["alive_ticks"]


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
                shared_colour: bool = False, novel_xor: bool = False, one_berry: float = 0.0):
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
    the first painful berry no longer helps and only judging the look does (STAGE_LOG v58, v59)."""
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

    def arrays(bad: tuple):
        fv, pv = rs.food_value.copy(), np.zeros(rs.size, np.float32)
        for v in bad:
            fv[berry[v]], pv[berry[v]] = poison_food, poison_pain
        return rs.to_arrays(exp.vision.appearance_dim, spawn_weight=spawn, food_value=fv, pain_value=pv,
                            appearance=app)

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


# ------------------------------------------------------------------ brains (each stage adds to the previous)

# 1.0 steering: a ganglion of excitatory and inhibitory interneurons between sensors and motor neurons,
# plus direct sensor->motor reflex arcs. All weights evolved, no plasticity. Both populations have divisive
# normalisation lagging one step (ADR-018): without it most interneurons sit at their ceiling.
# alpha 1: no blending with the previous step, so a path through an interneuron lags one tick, not several.
GANGLION = dict(alpha=1.0, norm=2.0, norm_lag=True, group="ganglion")
# Within a tick activity runs through the layers in order (RegionSpec.phase): senses -> ganglion, value and gate
# cells (phase 0) -> no_feed (1) -> motor programmes (2) -> motor neurons (3). One tick is about a second of
# behaviour; a synapse takes milliseconds. Updating everything at once cost a tick per layer, so a programme
# reacted to what was sensed two ticks earlier and repeated its action after the situation had changed.
OUT_PHASE = 3
# The ganglion and the motor neurons see the outside world only. The body's own state (food level, temperature)
# reaches the brain through the drive cells of stage 1.2 and nowhere else: before 1.2 the animal is a pure
# reflex animal, and any state-dependent behaviour after it has to use the drive cells.
# Taste and pain are not in this list either: they reach the brain through the valence cells of stage 1.1 (gustatory
# and nociceptive neurons synapse onto appetitive and aversive interneurons), so every reaction to tasting or
# being hurt goes through those cells. What an object looks like stays available to the ganglion as well.
EXTERO = ("vis*", "held*", "age", "sound")
B10 = BrainConfig(
    out_phase=OUT_PHASE,
    regions=(R("ganglion_e", 16, sign="exc", **GANGLION), R("ganglion_i", 8, sign="inh", **GANGLION)),
    projections=(P("in", "ganglion_e", src_select=EXTERO), P("in", "ganglion_i", src_select=EXTERO),
                 P("ganglion_e", "ganglion_e"), P("ganglion_e", "ganglion_i"), P("ganglion_i", "ganglion_e"),
                 P("ganglion_e", "out"), P("ganglion_i", "out"), P("in", "out", src_select=EXTERO)))

# Food economy (v6): many small meals. A berry is 2 food units (10% of a stomach), a bush 12; bushes regrow after
# regrow_ticks(exp), during which a waiting agent burns 24 units, so camping at one bush does not pay. The world
# is 128 x 128 for 256 agents (the same density as 64 agents on 64 x 64), at a lower bush density than before,
# and holds roughly 1.5 x what the agents need for 1000 ticks. A population of 256 lets selection see small
# advantages that drift hides at 64.
# Walking costs +50% hunger for the tick, turning +25%. (With a fitness that pays out the birth reserve, evolution
# from random brains then stands still; the first-meal fitness, ADR-018, removes that.)
# Chapter-1 animals have a mouth, not hands: they eat what they grasp (eat_on_pick). Picking something up, holding
# it and then eating it is OHOL's formulation; carrying arrives later as its own stage (with_hands, x.hands).
# Bush density 0.14: at 0.07 the agents ate nearly all there was (57% of bushes empty), fitness was a race for a
# fixed supply, and the genome explained 8% of the variance between lives; at 0.14 it explains 35%.
# Born a quarter full (from v23): a full stomach lasts 400-800 ticks at rest, most of a 1000-tick life, so under a
# survival fitness an animal that sat still reached the life cap almost unfed and could not be told from a forager.
# Food as a flow (v25): with every bush full at birth and a regrowth as long as a life, the food was a stock: eaten
# down in about 500 ticks, then famine, and nobody could gain by holding back. Now 70% of the bushes start empty at
# a random point of their regrowth, which is where a grazed world settles (about a fifth of the edible bushes full
# all life, `scripts/probes/supply.py`). The regrowth itself is unchanged (all berries back at once), so camping at
# a bush still does not pay. A bite costs a tenth of a berry, so one on a full stomach is a loss.
W10 = WorldConfig(height=128, width=128, num_agents=256, spawn_density=0.14, max_decay_ticks=2000, food_scale=2 / 3,
                  move_cost=0.5, turn_cost=0.25, eat_on_pick=True, start_food=0.25, start_spent=0.7, eat_cost=0.2)

stage(Stage("1.0", "s1_0_steering", None, B10, W10, VISION_CH1, BodyConfig(), lambda exp: berry_world(exp, 4),
            generations=400, notes="evolved reflexive steering to 4 kinds of berry bushes and onions; movement costs energy"))


def extend(base: BrainConfig, regions=(), projections=(), modulators=(), **kw) -> BrainConfig:
    """New brain = old brain + new regions (inserted before 'out' in the given order) + projections."""
    return replace(base, regions=base.regions + tuple(regions), projections=base.projections + tuple(projections),
                   modulators=base.modulators + tuple(modulators), **kw)


def fixed(src, dst, w, **kw):
    """A hard-wired projection (every synapse present, designed weight w): innate structure. Evolution cannot
    rewire it or flip its sign, but it may scale the whole projection (tune; ADR-016 amended)."""
    return P(src, dst, density=1.0, w_init=w, evolve=False, **{"tune": True, **kw})


# Action indices on the motor region 'out' (actions.py): NOOP 0, FORWARD 1, TURN_LEFT 2, TURN_RIGHT 3, USE 4, EAT 5
FWD, TURNS, FEED, USE_, EAT_ = (1, 2), (2, 4), (4, 6), (4, 5), (5, 6)

# 1.1 valence: cell types with a fixed meaning (ADR-020). Value cells: appetitive and aversive neurons with innate
# (hard-wired) input from the unconditioned senses (taste -> appetitive, pain -> aversive; these senses enter the
# brain only here) and evolved input from what is seen. Value does not drive the motor neurons itself. As in small
# animals it acts on a motor programme with its own trigger:
#   grasp (-> USE): a contact reflex: fires only when something is directly ahead, in proportion to its value.
# The 'only when' is disinhibition: the inhibitory gate cell 'no_touch', active when nothing is adjacent ahead,
# holds the programme shut. So a learned value becomes the right action for the situation; a value added straight
# onto the motor neurons came out as FORWARD whatever the context (assay 2026-10-03). Aversion turns the animal
# away and, through the inhibitory 'no_feed' pair, blocks grasping. (An 'approach' programme with a contact gate
# was tried and removed: silencing it changed nothing in any of three seeds; steering towards food is done by the
# evolved reflexes.)
# The animal eats what it grasps (W10: eat_on_pick). Carrying things arrives later with `with_hands` below.
VAL = ("valence_app", "valence_av")
# Identity features (the object's appearance: the analogue of an odour) are kept apart from generic features
# ('something is there', how near, agent, wall), because only identity can become a conditioned stimulus later
# (1.5); conditioning 'something is there' would make every object aversive.
CS_SEL = ("vis*.app*", "held_app*")
CS_VIS, CS_HELD = ("vis*.app*",), ("held_app*",)
GENERIC_SEL = ("vis*.hit", "vis*.near", "vis*.agent", "vis*.wall", "held")
VALENCE_MOTOR_W = 3.0
GATE = dict(sign="inh", alpha=1.0, evolve_bias=False, group="valence/gates")
PROG = dict(sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, phase=2, group="valence/programmes")
GATE_W = 6.0          # an active gate cell closes its programme completely
B11 = extend(B10,
             regions=(R("valence_app", 2, sign="exc", alpha=1.0, group="valence"),
                      R("valence_av", 2, sign="exc", alpha=1.0, group="valence"),
                      R("no_feed", 2, sign="inh", alpha=1.0, bias=0.0, evolve_bias=False, phase=1, group="valence"),
                      R("no_touch", 1, bias=10.5, **GATE),     # max(0, tanh(10.5 - 15 near)): nothing adjacent ahead
                      R("grasp", 1, **PROG)),
             projections=(fixed("in", "valence_app", 3.0, src_select=("taste",)),
                          fixed("in", "valence_av", 3.0, src_select=("pain",)),
                          *(P("in", v, src_select=CS_VIS) for v in VAL),
                          *(P("in", v, src_select=GENERIC_SEL) for v in VAL),
                          # context
                          P("in", "no_touch", src_select=("vis+0.near",), density=1.0, w_init=-15.0, evolve=False),
                          # value -> programme, gated; programme -> motor neuron
                          fixed("valence_app", "grasp", 1.5), fixed("no_touch", "grasp", GATE_W),
                          fixed("no_feed", "grasp", VALENCE_MOTOR_W),
                          fixed("grasp", "out", VALENCE_MOTOR_W, dst_range=USE_),
                          # aversion: turn away, do not bite
                          fixed("valence_av", "out", VALENCE_MOTOR_W, dst_range=TURNS),
                          fixed("valence_av", "no_feed", VALENCE_MOTOR_W),
                          fixed("no_feed", "out", VALENCE_MOTOR_W, dst_range=USE_),
                          *(P(v, t) for v in VAL for t in ("ganglion_e", "ganglion_i"))))

SEEN, HELD = (0, 2), (2, 3)   # with hands: value cells 0-1 are about what is seen, cell 2 about what is in hand


def with_hands(brain: BrainConfig) -> BrainConfig:
    """The valence stage for an animal that carries things (pick, hold, then eat: WorldConfig.eat_on_pick off). One
    more value cell per sign for the thing in hand and one more programme:
      ingest (-> EAT): fires only with something in hand ('no_hold' holds it shut otherwise), in proportion to the
      value of the thing in hand. Aversion now blocks eating instead of USE (USE is also how a thing is put down).
    Works on any brain that contains the 1.1 valence stage; identity synapses from the held item get the same rule
    as those from what is seen (so they are plastic where those are). A 'reject' programme (put down what is bad)
    and a full-hand gate on grasp were tried and removed: unused in the x.hands run (lesions 100%)."""
    regions = tuple(replace(r, size=3) if r.name in VAL else r for r in brain.regions)
    proj, held = [], []
    for p in brain.projections:
        if p.src == "in" and p.dst in VAL and p.src_select == CS_VIS:
            held.append(replace(p, src_select=CS_HELD, dst_range=HELD))
            p = replace(p, dst_range=SEEN)
        elif p.src == "in" and p.dst == "valence_app" and p.src_select == ("taste",):
            p = replace(p, dst_range=SEEN)
        elif p.src == "valence_app" and p.dst == "grasp":
            p = replace(p, src_range=SEEN)
        elif p.src == "no_feed" and p.dst == "out":
            p = replace(p, dst_range=EAT_)
        proj.append(p)
    brain = replace(brain, regions=regions, projections=tuple(proj))
    return extend(brain,
                  regions=(R("no_hold", 1, bias=3.0, **GATE),       # max(0, tanh(3 - 6 held)): empty hand
                           R("ingest", 1, **PROG)),
                  projections=(*held,
                               P("in", "no_hold", src_select=("held",), density=1.0, w_init=-6.0, evolve=False),
                               fixed("valence_app", "ingest", 3.0, src_range=HELD), fixed("no_hold", "ingest", GATE_W),
                               fixed("no_feed", "ingest", VALENCE_MOTOR_W),
                               fixed("ingest", "out", VALENCE_MOTOR_W, dst_range=EAT_)))


BODY_TASTE = BodyConfig(taste=True)

# same number of good bushes as 1.0 plus 2 poisonous types on top (density 0.07 * 6/4)
# Each new hardship is offset by a lower metabolic rate, chosen so that the parent population keeps its lifetime
# when it enters the new world (ADR-012: an increment, not a cliff). More bushes do not help: time per meal, not
# food, is the limit.
W11 = replace(W10, hunger_per_tick=0.025, pain_decay=0.3)   # same bush density; pain is a brief signal
# Lives are 4000 ticks from the settling run of 1.0 on (`stages 1.0 --ticks 4000 --init-from <the scratch run>`): a
# stomach lasts 400 ticks at rest in 1.0 and 800 in 1.1, so in a life of 1000 ticks an animal that ate once and then
# stood still reached the cap. Stage 1.0 from scratch keeps 1000 ticks (nearly everything dies young at first).
LIFE = 4000
stage(Stage("1.1", "s1_1_valence", "1.0", B11, W11, VISION_CH1, BODY_TASTE,
            lambda exp: berry_world(exp, 6, poison=(4, 5), poison_food=POISON_INNATE), row_extra=poison_metrics,
            generations=50, ticks=LIFE, notes="value cells acting on a contact-gated grasp programme; 2 of 6 berry types poison"))

# x.hands (side stage; the condition for eating on grasp, user 2026-10-03): can a lineage that evolved with a mouth
# adapt when food has to be picked up, held and then eaten? The 1.1 population with the hands extension in the 1.1
# world without eat_on_pick. Control: the plain 1.1 brain in that world.
BXH = with_hands(B11)
WXH = replace(W11, eat_on_pick=False)
stage(Stage("x.hands", "sx_hands", "1.1", BXH, WXH, VISION_CH1, BODY_TASTE,
            lambda exp: berry_world(exp, 6, poison=(4, 5), poison_food=POISON_INNATE), row_extra=poison_metrics,
            generations=100, notes="side test: carrying returns (pick, hold, eat) with an ingest programme"))


# ---- 1.2-1.4 are a parked side branch (2026-10-03): defined on top of 1.1 but not on the path to chapter 2 until
# ---- learning works. Their circuits were not used in the last runs (STATUS.md).
# 1.2 drives (hypothalamus): named drive neurons read the body. 'hunger' is broadcast as a neuropeptide-like
# modulator: receptors on valence_app raise appetite with need. 'cold' gates an innate thermotaxis circuit
# (run-and-tumble on temperature changes, as C. elegans AFD -> AIY/AIZ): when cold and getting warmer, keep going
# forward; when cold and getting colder, turn. No knowledge of what a heat source looks like is needed. 'cold'
# acts through synapses only (it is not broadcast).
#   hunger    = max(0, tanh(2 - 3 * food))                     fires below ~2/3 full
#   cold      = max(0, tanh(3.3 - 7 * temperature))            fires below ~0.47
#   warm_run  = max(0, tanh(-2.5 + 2.5 cold + 3 skin_change))  -> FORWARD      (cold and moving up the gradient)
#   warm_turn = max(0, tanh(-2.5 + 2.5 cold - 3 skin_change))  -> TURN_LEFT/RIGHT  (cold and moving down it)
# skin_change is the change of the temperature at the agent's cell (one cell up a hot spring's gradient = +0.9).
# The body warms and cools slowly (temp_rate 0.03), so warmth can be banked for a foraging trip, and the cold
# costs up to +75% hunger: warming up competes with feeding, which is what a drive is for.
# World: a cold world (OHOL-style temperature) with hot springs; being cold multiplies hunger.
THERMO = dict(sign="exc", alpha=1.0, bias=-2.5, evolve_bias=False, group="thermotaxis")
B12 = extend(B11,
             regions=(R("hunger", 1, sign="exc", alpha=0.5, bias=2.0, evolve_bias=False, group="hypothalamus"),
                      R("cold", 1, sign="exc", alpha=0.5, bias=3.3, evolve_bias=False, group="hypothalamus"),
                      R("warm_run", 1, **THERMO), R("warm_turn", 1, **THERMO)),
             projections=(fixed("in", "hunger", -3.0, src_select=("food",)),
                          fixed("in", "cold", -7.0, src_select=("temperature",)),
                          fixed("cold", "warm_run", 2.5), fixed("cold", "warm_turn", 2.5),
                          fixed("in", "warm_run", 3.0, src_select=("skin_change",)),
                          fixed("in", "warm_turn", -3.0, src_select=("skin_change",)),
                          fixed("warm_run", "out", 3.0, dst_range=FWD), fixed("warm_turn", "out", 3.0, dst_range=TURNS),
                          P("hunger", "ganglion_e"), P("cold", "ganglion_e"),
                          P("hunger", "ganglion_i"), P("cold", "ganglion_i")),
             modulators=(Mod("hunger", pos="hunger"),))
B12 = replace(B12, regions=tuple(replace(r, receptors=r.receptors + (("hunger", "gain", 1.0),)) if r.name == "valence_app"
                                 else r for r in B12.regions))
BODY_12 = BodyConfig(taste=True, temperature=True, skin_change=True)
# a few more bushes; the metabolic rate is set so that the 1.1 population keeps its lifetime (ADR-012)
W12 = replace(W11, temperature=True, ambient_temp=0.25, heat_scale=0.15, heat_radius=4, temp_rate=0.03, temp_hunger=1.5,
              spawn_density=0.12, hunger_per_tick=0.019)
stage(Stage("1.2", "s1_2_drives", "1.1", B12, W12, VISION_CH1, BODY_12,
            lambda exp: berry_world(exp, 6, poison=(4, 5), springs=0.6, poison_food=POISON_INNATE), row_extra=poison_metrics,
            generations=150, notes="PARKED side branch: hunger (broadcast) and cold-gated thermotaxis; cold world with hot springs"))

# 1.3 affect: two slow, antagonistic neuromodulatory states, as in C. elegans (Flavell et al. 2013):
#   raphe (serotonin, '5ht'): fires on taste (food found); the released serotonin is cleared slowly (decay 0.97
#   per tick, about 30 ticks), so the state outlasts the meal; 5ht receptors on 'dwell', which
#   drives turning -> local search after food (dwelling).
#   pdf (roaming neuropeptide): driven by hunger, slow; pdf receptors on 'roam', which drives FORWARD -> long
#   straight runs when food has not been found for a while (roaming).
# The two nuclei inhibit each other (Flavell et al.: mutual inhibition makes dwelling and roaming two stable
# states): inhibitory 5ht receptors on pdf and pdf receptors on raphe, hard-wired. World: food in a few dense
# patches.
AFFECT_INHIB = 2.0   # input removed from one nucleus per unit mean activity of the other
B13 = extend(B12,
             regions=(R("raphe", 2, sign="exc", alpha=1.0, receptors=(("pdf", "bias", -AFFECT_INHIB),), group="affect"),
                      R("pdf", 2, sign="exc", alpha=0.03, receptors=(("5ht", "bias", -AFFECT_INHIB),), group="affect"),
                      R("dwell", 2, sign="exc", bias=0.0, evolve_bias=False, receptors=(("5ht", "bias", 2.0),), group="affect"),
                      R("roam", 2, sign="exc", bias=0.0, evolve_bias=False, receptors=(("pdf", "bias", 2.0),), group="affect")),
             projections=(fixed("in", "raphe", 2.0, src_select=("taste",)), fixed("hunger", "pdf", 2.0),
                          P("valence_app", "raphe"), P("valence_av", "pdf"),
                          fixed("dwell", "out", 1.5, dst_range=TURNS), fixed("roam", "out", 1.5, dst_range=FWD)),
             modulators=(Mod("5ht", pos="raphe", decay=0.97), Mod("pdf", pos="pdf")))
W13 = replace(W12, patches=80, patch_radius=6)   # the same number of objects, on at most half of the map
stage(Stage("1.3", "s1_3_affect", "1.2", B13, W13, VISION_CH1, BODY_12,
            lambda exp: berry_world(exp, 6, poison=(4, 5), springs=0.6, poison_food=POISON_INNATE), row_extra=poison_metrics,
            generations=150, notes="PARKED side branch: serotonin (dwell) and PDF (roam) broadcast states; patchy food"))

# 1.4 habituation: short-term depression on the sensory -> appetitive synapses, so the pull of something that
# stays in view fades (and recovers after about 150 ticks). World: dud bushes, which look like a bush with a
# colour drawn per life but yield nothing, so no inherited weight can ignore them. Without habituation an
# agent that happens to find the dud's look attractive keeps trying; with it the agent gives up and moves on.
B14 = replace(B13, projections=tuple(
    replace(p, depression=(0.1, 150.0)) if (p.src == "in" and p.dst == "valence_app" and p.src_select == CS_VIS)
    else p for p in B13.projections))
DUD = 6               # berry-bush type used as the dud
DUD_WEIGHT = 2.0      # spawn weight (the six real types: 1 each)
W14 = replace(W13, spawn_density=round(W13.spawn_density * (7.1 + DUD_WEIGHT) / 7.1, 3))   # the duds come on top


def dud_world(exp):
    return berry_world(exp, 7, poison=(4, 5), springs=0.6, duds=(DUD,), novel_looks=(DUD,), weights={DUD: DUD_WEIGHT},
                       poison_food=POISON_INNATE)


stage(Stage("1.4", "s1_4_habituation", "1.3", B14, W14, VISION_CH1, BODY_12, dud_world,
            row_extra=poison_metrics, generations=150,
            notes="PARKED side branch: habituating appetitive synapses; dud bushes with a per-life look"))

# 1.5 associative learning (ADR-017, ADR-021), built directly on 1.1. The world must be one that evolution cannot
# memorise: besides one ancestral good type and one ancestral poison type (fixed looks, so innate preferences
# still pay), four *novel* types get a new look every life and two of them are poison. Familiar food keeps a
# non-learner alive; novel food is what a learner can add.
# Rule: dW = eta * teacher * trace(pre) on the identity -> aversive synapses, with pain as the teacher. Pain is
# immediate and the eligibility trace short, so the teacher writes only about what was just bitten.
# (A second teacher, taste on the identity -> appetitive synapses, was part of v19-v22 and removed: silencing it
# left fitness at or above intact in all six seeds of 1.5 and 1.6. Appetitive learning returns with the reward
# prediction error of chapter 2; CS_MOD keeps both names for that.)
# (A 'safety' cell that subtracted from the aversive teacher after a good meal of something mistrusted was tried:
# silencing it changed nothing, 98% of intact, so it is not part of the brain.) Sickness that arrives several ticks after eating needs
# a longer aversive trace (CS_ELIG_SICK) and is left for the prediction error of 2.5.
CS_ETA = 0.05         # starting learning rate of both pathways; evolvable per projection
CS_ELIG = {"valence_app": 0.5, "valence_av": 0.5}   # eligibility decay per tick
CS_ELIG_SICK = 0.92   # aversive trace for delayed sickness (not used on the main path)
CS_MOD = {"valence_app": "us_app", "valence_av": "us_av"}
CS_MODS = tuple(CS_MOD.values())
CS_TAUGHT = ("valence_av",)   # value cells whose identity synapses learn in chapter 1


def _split_cs(projections, taught=CS_TAUGHT, eta=None, centred=False, only=()):
    """in[identity features] -> the taught value cells becomes plastic and US-gated: dW = eta * teacher * trace(pre).
    centred: pre is the look minus the slow average of each look input (ProjectionSpec.centred).
    only: input-feature patterns that learn (e.g. the eye pointing ahead); the rest of the projection stays as
    inherited."""
    out = []
    for p in projections:
        if p.src == "in" and p.src_select in (CS_VIS, CS_HELD) and p.dst in taught:
            learned = replace(p, rule="hebb", modulator=CS_MOD[p.dst], eta_init=CS_ETA if eta is None else eta,
                              elig_tau=CS_ELIG[p.dst], abcd=(0.0, 1.0, 0.0, 0.0), centred=centred)
            if only and p.src_select == CS_VIS:
                out.append(replace(learned, src_select=tuple(only)))
                out.append(replace(p, src_select=SIDE_EYES))
            else:
                out.append(learned)
        else:
            out.append(p)
    return tuple(out)


AHEAD_EYE, SIDE_EYES = ("vis+0.app*",), ("vis-*.app*", "vis+[36]0.app*")


# 1.1l long lives (side path since the rerun of 2026-10-08, in which 1.1 itself has long lives): the 1.1 brain in
# the 1.1 world with lives of 4000 ticks after a 1.1 of 1000 ticks. No new circuit and so no control.
stage(Stage("1.1l", "s1_1_valence_long", "1.1", B11, W11, VISION_CH1, BODY_TASTE,
            lambda exp: berry_world(exp, 6, poison=(4, 5), poison_food=POISON_INNATE), row_extra=poison_metrics,
            generations=60, ticks=4000, notes="SIDE PATH: the 1.1 brain under lives of 4000 ticks (adaptation step, no control)"))


# 1.2h habituation by sensory adaptation (v30, v31; the user's preferred form, 2026-10-07): no new cells. Every look
# input (`vis*.app*`) passes on its input minus the slow average of that input, so everything downstream sees what
# is unusual, and a look that stays in view fades: the pull of a bush that yields nothing wears off (`freeze.py`).
# The inherited synapses were shaped on the raw look, so the population has to re-tune. Control: the 1.1 brain.
# There is no region to silence: the lesion is `scripts/probes/adapt_off.py`.
ADAPT_TAU = 0.98      # the look inputs' average runs over about 50 ticks
B12H = replace(B11, in_adapt=CS_VIS, in_trace_tau=ADAPT_TAU)
# World: the 1.1 world plus a dud, a bush type that never carries a berry and whose colour is drawn anew every life,
# so no inherited weight can know it (the world of the parked stage 1.4). It is the case habituation is for: an
# animal that finds the dud's look attractive keeps biting at it unless the pull fades. In the 1.1 world alone long
# lives had already selected most standing still away and the adaptation was used weakly (84% of intact with it off,
# main level with control); with the dud 34%, main ahead (one seed each, STAGE_LOG v32). Giving the good types novel
# colours instead did not help (72-78%, main behind control).
DUD_12 = 1.0          # spawn weight of the dud (a berry type: 1); the density is raised so that the food stays the same
W12 = replace(W11, spawn_density=round(W11.spawn_density * (6.5 + DUD_12) / 6.5, 3))


def habituation_world(exp, novel: tuple = (), dud: float = DUD_12):
    """The 1.1 world (four good types, two innately known poison types, onions) plus a dud bush type with a look
    drawn per life at spawn weight `dud`. novel (trials): good types that get a look drawn per life as well."""
    if not dud:
        return berry_world(exp, 6, poison=(4, 5), poison_food=POISON_INNATE, novel_looks=tuple(novel), novel_sim=NOVEL_SIM)
    return berry_world(exp, 7, poison=(4, 5), poison_food=POISON_INNATE, novel_looks=tuple(novel) + (DUD,),
                       novel_sim=NOVEL_SIM, duds=(DUD,), weights={DUD: dud})


stage(Stage("1.2h", "s1_2_habituation", "1.1", B12H, W12, VISION_CH1, BODY_TASTE, habituation_world,
            row_extra=poison_metrics, generations=40, ticks=LIFE,
            notes="the look inputs adapt to their slow average: what stays in view fades; a dud bush type with a "
                  "colour drawn per life"))


# What the lesson is about (v27, STAGE_LOG 2026-10-07). Pain alone, on the whole look, only ever raises the aversive
# weights, and every look shares part of the gooseberry look, so suspicion of a poison ends on good food and is
# never taken back; evolution then sets the learning rate to zero. Three things make the lesson land on the poison:
#   adapted    the look inputs adapt since the habituation stage, so the synapse learns from the look minus the slow
#              average of each look input: what sets this food apart, not what all berries share. (Before the rerun
#              of 2026-10-08 association came first and its synapses did this themselves, `centred`: side stage 1.5c.)
#   one tick   pain lasts one tick in this world (pain_decay 0). While it faded over three ticks the teacher was
#              still at half strength when the animal had turned to a neighbouring bush, and taught about that one;
#   dense      every learned synapse exists (Stage.dense_plastic); only 42% did, and an absent synapse cannot learn.
# Tried and left out: a 'safety' cell (a good meal of something mistrusted lowers the aversion). It never fires in
# a living animal, which bites only while its aversive cells are silent (variant `safety` in scripts/probes).
# Variant 1.5f: only the eye pointing straight ahead learns (what is at the mouth); as defined every eye does, and
# at a bite the side eyes mostly see other bushes.
ETA_15 = 0.3          # starting learning rate of stage 1.5 (evolvable up to Stage.eta_max)
_B15 = extend(B11,
              regions=(R("us_pain", 1, sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, group="us"),),
              projections=(P("in", "us_pain", src_select=("pain",), density=1.0, w_init=2.0, evolve=False),),
              modulators=(Mod("us_av", pos="us_pain"),), in_trace_tau=0.99)
B15C = replace(_B15, projections=_split_cs(_B15.projections, eta=ETA_15, centred=True))   # side path (old order)
# the association brain: the habituation brain + the pain teacher; the learned synapses are plain
B15 = replace(_B15, in_adapt=CS_VIS, in_trace_tau=ADAPT_TAU, projections=_split_cs(_B15.projections, eta=ETA_15))
B15F = replace(_B15, projections=_split_cs(_B15.projections, eta=ETA_15, centred=True, only=AHEAD_EYE))
# The world (v28): nothing familiar. Four novel types, a new look and a new meaning every life, two of them poison;
# no staple and no ancestral poison. With familiar food as half of the bushes the animals ate eleven good berries
# for every poison one and a lesson had to be near perfect to pay; here whoever cannot learn pays one bite on every
# visit to a poison bush, half of all bushes, for life, and 'never touch anything novel' is starving. A life is
# 4000 ticks (five stomachs; the 1.1 animals froze after a meal and still reached a 1000-tick cap), with fewer
# generations to pay for it.
# Born half full: with a quarter (5 food units) the first bite, poison with one chance in two and costing 4, killed
# the animal within 40 ticks whatever its genome, and no population climbed in 40 generations.
W15 = replace(W11, spawn_density=0.19, pain_decay=0.0, start_food=0.5)
LIFE_15 = 4000        # ticks per life in stage 1.5
LEARN_BERRIES = 6     # berries per bush in stage 1.5
LIFE_LEARN = 2000     # ticks per life in the later learning stages (older designs, to be revisited)
NOVEL_SIM = 0.8       # novel types look as much like the gooseberry as the 1.1 types do: a new kind of berry bush is
                      # recognisable as a berry bush, and differs in its colour. At 0.45 (until v27) the 1.1
                      # animals hardly took them for food (one good bush per life)
POISON_FOOD = -3.0    # OHOL food points lost per poison berry in the learning stages (a berry gives +3): a mistake
                      # costs one berry and the pain reflex ends the visit. At two berries (4 food units) a bite on
                      # a stomach under a fifth full killed, a coin flip for any hungry animal in a world where
                      # half of the bushes are poison. Generation 0: a perfect learner would save 23% of intake


POISON_PAIN = 1.0     # pain of one poison berry in the learning stages. A world lesson is this one tick of pain (pain_decay
                      # 0), three to five times weaker than a lab trial of three (STAGE_LOG v61)


NOVEL = (1, 2, 3, 5)  # berry types whose look and meaning are drawn per life; type 0 is always good, 4 always poison
NOVEL_WEIGHT = 1.0    # spawn weight of each novel type
STAPLE_WEIGHT = 0.0   # spawn weight of the ancestral good type (0: no familiar food)
ANC_POISON_WEIGHT = 0.0   # spawn weight of the ancestral poison type (0: none)


def learning_world(exp, reverse: bool = False, springs: float = 0.0, berries: int | None = None,
                   shared_colour: bool = False, novel_sim: float | None = None, novel_xor: bool = False,
                   one_berry: float = 0.0):
    """Four novel berry types (two of them poison, drawn per life, with a look drawn per life), one ancestral good
    and one ancestral poison type, no onions. NOVEL_WEIGHT sets how much of the supply is novel: familiar food
    should keep a non-learner alive, novel food should be worth trying."""
    return berry_world(exp, 6, poison=(4,), per_life_pool=NOVEL, per_life_k=2, springs=springs, reverse=reverse,
                       novel_looks=NOVEL, novel_sim=NOVEL_SIM if novel_sim is None else novel_sim, poison_food=POISON_FOOD, onions=False,
                       poison_pain=POISON_PAIN,
                       weights={0: STAPLE_WEIGHT, 4: ANC_POISON_WEIGHT, **{v: NOVEL_WEIGHT for v in NOVEL}},
                       berries=berries, shared_colour=shared_colour, novel_xor=novel_xor, one_berry=one_berry)


_S15 = dict(row_extra=poison_metrics, plastic=True, generations=40, ticks=LIFE_15, eta_max=1.0, dense_plastic=True)
stage(Stage("1.5", "s1_5_association", "1.2h", B15, W15, VISION_CH1, BODY_TASTE,
            lambda exp: learning_world(exp, berries=LEARN_BERRIES), **_S15,
            notes="aversive conditioning: pain teaches the (adapted) look -> aversive synapses; "
                  "only novel foods, half of them poison"))
# Side paths: the order that passed with three seeds before the rerun (1.1l -> 1.5 with the centred rule -> 1.6h) and
# the forward-eye variant. The runs of the old 1.5 are in runs/s1_5_association*, older than those of the rerun.
stage(Stage("1.5c", "s1_5c_association", "1.1l", B15C, W15, VISION_CH1, BODY_TASTE,
            lambda exp: learning_world(exp, berries=LEARN_BERRIES), **_S15,
            notes="SIDE PATH: association before habituation, the learned synapses centre the look themselves"))
stage(Stage("1.5f", "s1_5_association_fwd", "1.1l", B15F, W15, VISION_CH1, BODY_TASTE,
            lambda exp: learning_world(exp, berries=LEARN_BERRIES), **_S15,
            notes="SIDE PATH: 1.5c with only the eye pointing ahead learning"))


# 1.6h (side path, the old order): habituation after the centred association 1.5c; the same brain as 1.5 is now.
stage(Stage("1.6h", "s1_6_habituation", "1.5c", B15, W15, VISION_CH1, BODY_TASTE,
            lambda exp: learning_world(exp, berries=LEARN_BERRIES), **_S15,
            notes="SIDE PATH: habituation after association (the old order)"))


# 1.7 drives (v29, v30): two need cells and a warmth mode. The circuit is the one that won
# the generation-0 probe (`scripts/probes/drives.py`, STAGE_LOG v23 and 2026-10-08):
#   cold       fires while the skin is colder than about 0.47 (max(0, tanh(3.3 - 7 skin)));
#   warm_seek  cold -> FORWARD: keep moving while it is cold (orthokinesis; no gradient, no sight of the spring);
#   rest       an inhibitory cell that fires where it is warm and holds FORWARD back: stay;
#   hungry     an inhibitory cell that fires below two thirds of a stomach and shuts warm_seek and rest: forage when
#              hungry, look after warmth when fed. Without it the warmth seeker starves.
# New here: `hungry` also gets evolvable synapses (starting at zero) onto the grasp programme, the bite block, the
# appetitive cells and the ganglion, so evolution can make feeding depend on hunger (user, 2026-10-07: not
# hard-wired). The cell is inhibitory, so it can only release or hold back: for example release the bite block.
# World: the 1.5 world made cold (ambient 0.25), with a few hot springs; the cold makes the body burn more.
DRIVE_HYP = dict(alpha=0.5, evolve_bias=False, group="hypothalamus")
B17 = extend(B15,
             regions=(R("cold", 1, sign="exc", bias=3.3, **DRIVE_HYP),
                      R("warm_seek", 1, sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, group="thermotaxis"),
                      R("rest", 1, sign="inh", bias=-3.3, **DRIVE_HYP),
                      R("hungry", 1, sign="inh", bias=6.0 * 0.67, **DRIVE_HYP)),
             projections=(fixed("in", "cold", -7.0, src_select=("skin",)),
                          fixed("cold", "warm_seek", 3.0), fixed("warm_seek", "out", 3.0, dst_range=FWD),
                          fixed("in", "rest", 7.0, src_select=("skin",)), fixed("rest", "out", 9.0, dst_range=FWD),
                          fixed("in", "hungry", -6.0, src_select=("food",)),
                          fixed("hungry", "warm_seek", 6.0), fixed("hungry", "rest", 6.0),
                          *(P("hungry", d, density=1.0, w_init=0.0) for d in
                            ("grasp", "no_feed", "valence_app", "ganglion_e", "ganglion_i"))))
BODY_17 = BodyConfig(taste=True, temperature=True, skin_change=True, skin=True)
W17 = replace(W15, temperature=True, ambient_temp=0.25, heat_scale=0.15, heat_radius=4, temp_rate=0.03, temp_hunger=1.5)
SPRINGS_17 = 0.1      # spawn weight of hot springs (a bush type: 1)
stage(Stage("1.7", "s1_7_drives", "1.5", B17, W17, VISION_CH1, BODY_17,
            lambda exp: learning_world(exp, springs=SPRINGS_17, berries=LEARN_BERRIES), **_S15,
            notes="cold and hungry need cells, warmth by kinesis, hunger decides between foraging and warmth; "
                  "the 1.5 world made cold with hot springs"))


# 1.8 affect (2026-10-12): two states that outlast what set them off. The drives stage lets a need that is sensed
# now select a mode; here an event leaves a modulator behind that is released at once and cleared slowly
# (ModulatorSpec.decay), and that pushes the animal's existing modes for a while:
#   alarm      pain (the `us_pain` cell of the association stage) -> the modulator `alarm` (about 10 ticks) -> a
#              `flee` cell -> FORWARD: after a poison bite, keep going. Short: a long alarm runs the animal out of
#              the good patches as well (-30 at about 30 ticks, generation 0);
#   serotonin  taste -> a `fed` cell -> the modulator `serotonin` (about 100 ticks) -> a `settle` cell -> `rest`:
#              after a meal the animal stays where it ate, also where it is cold. `hungry` still shuts `rest`.
# Neither modulator teaches anything (dopamine, the learning signal, is chapter 2). Evolution tunes every strength
# (the hard-wired projections) and both fade times (Genome.clear). The roam nucleus of the old stage 1.3 is gone:
# `hungry` does that job. Generation 0 (`scripts/probes/affect.py`, STAGE_LOG v33): +46 and +200, together +252.
# World: for a lasting state to have a job, what just happened must say something about what is near. Every berry
# type grows in its own patches, so a good meal means more good bushes close by and a poison bite means the
# neighbours are poison too. Hot springs stay spread evenly. In the drives world a bush is emptied in one visit,
# fills the stomach, and food and poison are spread evenly: nothing to stay for or to flee from.
AFFECT = dict(sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, group="affect")
B18 = extend(B17,
             regions=(R("flee", 1, receptors=(("alarm", "bias", 3.0),), **AFFECT),
                      R("fed", 1, **AFFECT),
                      R("settle", 1, receptors=(("serotonin", "bias", 2.0),), **AFFECT)),
             projections=(fixed("flee", "out", 3.0, dst_range=FWD),
                          fixed("in", "fed", 2.0, src_select=("taste",)),
                          fixed("settle", "rest", 3.0)),
             modulators=(Mod("alarm", pos="us_pain", decay=0.9), Mod("serotonin", pos="fed", decay=0.99)))
W18 = replace(W17, patches=80, patch_radius=6, patch_types=True)
stage(Stage("1.8old", "s1_8_affect", "1.7", B18, W18, VISION_CH1, BODY_17,
            lambda exp: learning_world(exp, springs=SPRINGS_17, berries=LEARN_BERRIES), **_S15,
            notes="two lasting states: alarm after pain (keep going), serotonin after a meal (stay); "
                  "the drives world with every berry type in its own patches"))

# Trials for the world of chapter 2 (2026-10-12, STAGE_LOG v34): the affect brain and population, nothing new, in
# worlds with poison standing between good bushes again. The question: does the lineage pick learning up again
# where the world pays for it? Not stages of the chain.
for _key, _name, _world, _note in (
        ("x.w60", "x_world_p60", replace(W18, patch_purity=0.6), "patches 60% their own type"),
        ("x.w30", "x_world_p30", replace(W18, patch_purity=0.3), "patches with a third poison in a good patch"),
        ("x.w30f", "x_world_p30_food", replace(W18, patch_purity=0.3, spawn_density=0.3), "the same, with more food"),
        ("x.weven", "x_world_even", replace(W17, spawn_density=0.3), "no patches, more food (the drives population "
                                                                     "gains from learning here)")):
    stage(Stage(_key, _name, "1.8old", B18, _world, VISION_CH1, BODY_17,
                lambda exp: learning_world(exp, springs=SPRINGS_17, berries=LEARN_BERRIES), **_S15,
                notes="TRIAL: the affect population in a chapter 2 world candidate: " + _note))


# 1.6 extinction and reversal: the learned weights now relax back toward their inherited values (a fast,
# forgetting component), so an association that stops being renewed fades and a new one can take over. World: the
# novel types swap their meaning halfway through life.
B16 = replace(B15C, projections=tuple(replace(p, decay=0.003) if p.modulator in CS_MODS else p for p in B15C.projections))
W16 = replace(W15, switch_tick=LIFE_LEARN // 2)
stage(Stage("1.6", "s1_6_reversal", "1.5c", B16, W16, VISION_CH1, BODY_TASTE, lambda exp: learning_world(exp, reverse=True),
            row_extra=poison_metrics, plastic=True, generations=150, ticks=LIFE_LEARN,
            notes="reversal learning: learned weights decay toward w0; the novel types swap meaning mid-life"))


# Chapter 2, the first design (2026-09; keys old2.x, not on the path): replaced stage by stage by the definitions
# under "chapter 2 as built" below (plan revised 2026-10-12, ADR-025). It is written for a brain with both teachers: the taste teacher that chapter 1
# dropped is added back here, on the reversal brain.
B16_TASTE = extend(B16, regions=(R("us_taste", 1, sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, group="us"),),
                   projections=(P("in", "us_taste", src_select=("taste",), density=1.0, w_init=2.0, evolve=False),),
                   modulators=(Mod("us_app", pos="us_taste"),))
B16_TASTE = replace(B16_TASTE, projections=tuple(
    replace(q, decay=0.003) if q.modulator in CS_MODS else q for q in _split_cs(B16_TASTE.projections, taught=VAL)))


# ================================================================== chapter 2: reinforcing (early vertebrates)

# camera eyes: 9 columns over 120 degrees, range 6. Columns at -60, -30, 0, 30, 60 keep their names (and so their
# inherited weights); the new in-between columns start silent.
VISION_CH2 = VisionConfig(columns=9, fov_degrees=120.0, range=6, appearance_dim=APPEARANCE)

# 2.1 optic tectum: a retinotopic map (2 neurons per vision column) with a pool of inhibitory interneurons for
# competition between targets, projecting to the motor neurons (orienting) and the ganglion.
B21 = extend(B16_TASTE,
             regions=(R("tectum", 18, sign="exc", group="midbrain/tectum"), R("tectum_i", 3, sign="inh", group="midbrain/tectum")),
             projections=(P("in", "tectum", src_select=("vis*",), topology="topographic", groups=9, density=1.0),
                          P("tectum", "tectum_i", density=1.0), P("tectum_i", "tectum", density=1.0),
                          P("tectum", "out", density=1.0), P("tectum", "ganglion_e")))
stage(Stage("old2.1", "old_s2_1_tectum", "1.6", B21, W16, VISION_CH2, BODY_TASTE, lambda exp: learning_world(exp, reverse=True),
            row_extra=poison_metrics, plastic=True, generations=200, ticks=LIFE_LEARN,
            notes="retinotopic target selection with lateral inhibition; camera eyes (9 columns)"))


def cs_plastic(src, dst, sel=None, **kw):
    """A US-gated CS-trace projection onto a valence population (the 1.5 rule)."""
    extra = {"src_select": sel} if sel else {}
    return P(src, dst, rule="hebb", modulator=CS_MOD[dst], eta_init=CS_ETA, elig_tau=CS_ELIG[dst], decay=0.003,
             abcd=(0.0, 1.0, 0.0, 0.0), **extra, **kw)


# 2.2 pallium as an expansion layer: a large population with fixed, sparse, random input from vision and the held
# item and k-winners-take-all inhibition (piriform cortex / mushroom-body style pattern separation). The
# US-gated learning of 1.5 now also runs from the pallium onto the valence neurons. World: the XOR world, where
# which pair of look-alikes is poison ({++, --} or {+-, -+} of two appearance features) is decided per life.
# (Still the v4 design: before running, move it to per-life looks and ADR-017's random-per-life pallium input.)
B22 = extend(B21,
             regions=(R("pallium", 48, sign="exc", kwta=6, group="forebrain/pallium"),),
             projections=(P("in", "pallium", src_select=("vis*", "held*"), density=0.15, evolve=False),
                          cs_plastic("pallium", "valence_app"), cs_plastic("pallium", "valence_av")))
W22 = replace(W16, spawn_density=W16.spawn_density * 7 / 6)   # 7 berry types instead of 6


def xor_world(exp):
    return berry_world(exp, 7, poison=(5, 6), per_life_sets=((1, 4), (2, 3)), appearance_mode="xor",
                       reverse=True)


stage(Stage("old2.2", "old_s2_2_pallium_expansion", "old2.1", B22, W22, VISION_CH2, BODY_TASTE, xor_world,
            row_extra=poison_metrics, plastic=True, generations=200, ticks=LIFE_LEARN,
            notes="sparse random expansion (k-WTA) before US-gated learning; XOR poison rule per life"))

# 2.3 pallium pattern completion and clustering: the input to the pallium becomes plastic (Oja, unsupervised,
# no modulator) and the pallium gets recurrent Hebbian connections, so it forms noise-robust categories within
# life. World: noisy perception of appearance (std 0.3 per feature and tick).
B23 = replace(B22, projections=tuple(
    replace(p, rule="oja", eta_init=0.01, evolve=True) if (p.src == "in" and p.dst == "pallium") else p
    for p in B22.projections) + (P("pallium", "pallium", rule="hebb", eta_init=0.005, decay=0.01,
                                   abcd=(1.0, 0.0, 0.0, 0.0)),))
VISION_23 = replace(VISION_CH2, appearance_noise=0.3)
stage(Stage("old2.3", "old_s2_3_pallium_clustering", "old2.2", B23, W22, VISION_23, BODY_TASTE, xor_world,
            row_extra=poison_metrics, plastic=True, generations=200, ticks=LIFE_LEARN,
            notes="unsupervised Oja input + recurrent Hebb in pallium; noisy appearance"))

# 2.4 basal ganglia, fixed: inhibitory striatal channels (one per action) inhibit a tonically active GPi, which
# inhibits the motor neurons one-to-one; an action is released by disinhibition. Striatum is driven by pallium,
# tectum and valence (evolved).
B24 = extend(B23,
             regions=(R("striatum", 7, sign="inh", group="forebrain/basal_ganglia"), R("gpi", 7, sign="inh", bias=1.0, evolve_bias=False, group="forebrain/basal_ganglia")),
             projections=(*(P(s, "striatum") for s in ("pallium", "tectum", "valence_app", "valence_av", "ganglion_e")),
                          P("striatum", "gpi", topology="one_to_one", w_init=2.0, evolve=False),
                          P("gpi", "out", topology="one_to_one", w_init=2.0, evolve=False)))
stage(Stage("old2.4", "old_s2_4_basal_ganglia", "old2.3", B24, W22, VISION_23, BODY_TASTE, xor_world,
            row_extra=poison_metrics, plastic=True, generations=200, ticks=LIFE_LEARN,
            notes="action selection by disinhibition (striatum -| GPi -| motor), fixed"))


# 2.5 dopamine as a temporal-difference prediction error (the critic). An opponent pair of value populations in the
# ventral striatum (value_app: expected good, value_av: expected bad; rates cannot be negative) learns from
# identity features and the pallium; copies delayed by one step (value_*_prev) let the midbrain compute
#   da = (us_taste - us_pain) + gamma * (V_app - V_av) - (V_app_prev - V_av_prev)
# (region means). The value populations drive valence (incentive salience, fixed), and the CS -> valence learning
# of 1.5 / 2.2 is now taught by dopamine instead of the raw US: error-driven, so a fully predicted outcome stops
# teaching and only the discriminating features keep changing (look-alikes stop generalising).
GAMMA = 0.9
TD = dict(rule="hebb", modulator="da", eta_init=0.05, elig_tau=0.8, decay=0.002)


def td_plastic(src, dst, sel=None):
    b = 1.0 if dst.endswith("_app") else -1.0
    extra = {"src_select": sel} if sel else {}
    return P(src, dst, abcd=(0.0, b, 0.0, 0.0), **TD, **extra)


def to_dopamine(projections):
    """Every US-gated projection is taught by the signed TD error instead (aversive synapses with the sign flipped)."""
    return tuple(replace(p, modulator="da", elig_tau=TD["elig_tau"],
                         abcd=(0.0, 1.0 if p.modulator == "us_app" else -1.0, 0.0, 0.0)) if p.modulator in CS_MODS else p
                 for p in projections)


VALUE = dict(sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, group="forebrain/basal_ganglia")
B25 = extend(B24,
             regions=(R("value_app", 3, **VALUE), R("value_av", 3, **VALUE),
                      R("value_app_prev", 3, **VALUE), R("value_av_prev", 3, **VALUE)),
             projections=(td_plastic("in", "value_app", CS_SEL), td_plastic("in", "value_av", CS_SEL),
                          td_plastic("pallium", "value_app"), td_plastic("pallium", "value_av"),
                          P("value_app", "value_app_prev", topology="one_to_one", w_init=1.0, evolve=False),
                          P("value_av", "value_av_prev", topology="one_to_one", w_init=1.0, evolve=False),
                          fixed("value_app", "valence_app", 1.5), fixed("value_av", "valence_av", 1.5)),
             modulators=(Mod("da", terms=(("us_taste", 1.0), ("us_pain", -1.0), ("value_app", GAMMA),
                                          ("value_av", -GAMMA), ("value_app_prev", -1.0), ("value_av_prev", 1.0))),))
B25 = replace(B25, projections=to_dopamine(B25.projections))
stage(Stage("old2.5", "old_s2_5_dopamine_td", "old2.4", B25, W22, VISION_23, BODY_TASTE, xor_world,
            row_extra=poison_metrics, plastic=True, generations=200, ticks=LIFE_LEARN,
            notes="TD critic (opponent value populations) and dopamine teaching instead of the raw US"))


# x.td (side experiment, not part of the lineage): does error-driven TD learning fix the over-generalisation of
# 1.5/1.6 conditioning? The 1.6 brain plus only the TD critic of 2.5 (value populations from identity features,
# dopamine replacing the raw US as teacher), in the 1.6 world; control = the 1.6 brain.
BXTD = extend(B16_TASTE,
              regions=(R("value_app", 3, **VALUE), R("value_av", 3, **VALUE),
                       R("value_app_prev", 3, **VALUE), R("value_av_prev", 3, **VALUE)),
              projections=(td_plastic("in", "value_app", CS_SEL), td_plastic("in", "value_av", CS_SEL),
                           P("value_app", "value_app_prev", topology="one_to_one", w_init=1.0, evolve=False),
                           P("value_av", "value_av_prev", topology="one_to_one", w_init=1.0, evolve=False),
                           fixed("value_app", "valence_app", 1.5), fixed("value_av", "valence_av", 1.5)),
              modulators=(Mod("da", terms=(("us_taste", 1.0), ("us_pain", -1.0), ("value_app", GAMMA),
                                           ("value_av", -GAMMA), ("value_app_prev", -1.0), ("value_av_prev", 1.0))),))
BXTD = replace(BXTD, projections=to_dopamine(BXTD.projections))
stage(Stage("x.td", "sx_td_test", "1.6", BXTD, W16, VISION_CH1, BODY_TASTE, STAGES["1.6"].build,
            row_extra=poison_metrics, plastic=True, generations=200, ticks=LIFE_LEARN,
            notes="side test: 1.6 + TD critic, dopamine teaches instead of the raw US"))


# ================================================================== chapter 2 as built (ADR-025)

# The world of chapter 2 (STAGE_LOG v34): the drives world without patches and with more food. Of the worlds tried,
# only this one gave learning a job again (plasticity off 87% of intact after 40 generations, trial `x.weven`);
# every world with patches left it neutral. The stay-after-a-meal state keeps a smaller job here, the alarm and
# warmth seeking little (user, 2026-10-12: no patches). The looks stay as in chapter 1. One colour shared by a type's
# bush, berry and empty bush (`shared_colour`) was tried and dropped: a full bush then looks more like its own empty
# bush (cosine 0.55 against 0.22) and the adapted population lost 42% at generation 0 (1144 -> 661, eight worlds).
# Twice the area for the same 256 animals (181 x 181; user, 2026-10-12): at 128 x 128 the animals took food from each
# other in the first few hundred ticks, when all are alive and 70% of the bushes are still regrowing, and the more
# so the better they foraged: the evolved camera-eyes population scored 788 at 128, 1247 at 181 and 1722 at 256,
# its starting population 487, 676 and 765 (`scripts/probes/crowding.py`). A stage that forages better must not
# look worse for it. At 256 a simulation takes about twice as long and 37% reach the life cap.
W20 = replace(W17, spawn_density=0.3, height=181, width=181)


def world_ch2(exp, shared_colour: bool = False):
    return learning_world(exp, springs=SPRINGS_17, berries=LEARN_BERRIES, shared_colour=shared_colour)


def camera_eyes(brain: BrainConfig) -> BrainConfig:
    """The adjacent-ahead gate of the grasp programme, set again for the longer range of the camera eyes
    (near = 1 - d / 6): shut from two cells on, open at one, as before."""
    return replace(brain,
                   regions=tuple(replace(r, bias=13.5) if r.name == "no_touch" else r for r in brain.regions),
                   projections=tuple(replace(p, w_init=-18.0) if p.dst == "no_touch" and p.src == "in" else p
                                     for p in brain.projections))


def interpolate_eyes(pop, layout, old: VisionConfig = VISION_CH1, scale: float = 1.0):
    """A head start for eyes of higher resolution (user, 2026-10-12): every evolved or learned synapse from a new eye
    column starts as the mean of the same synapse from its two neighbours among the old columns, instead of silent.
    scale: all synapses from the eyes are multiplied by this afterwards (more columns = more input in total).
    Hard-wired synapses are left as designed."""
    from life.sensors import column_angles
    names = list(layout.in_names)
    off = layout.offsets[layout.names.index("in")]
    angles = column_angles(old)
    col = lambda a: f"vis{a:+.0f}"
    old_cols = {col(a) for a in angles}
    free = (np.asarray(layout.evolve_w) > 0) & (np.asarray(layout.allowed) > 0)
    w0, mask = np.array(pop.w0), np.array(pop.mask)
    for i, n in enumerate(names):
        c, _, f = n.partition(".")
        if not c.startswith("vis"):
            continue
        r = off + i
        if c not in old_cols:
            a = float(c[3:])
            j = off + names.index(f"{col(angles[angles < a].max())}.{f}")
            k = off + names.index(f"{col(angles[angles > a].min())}.{f}")
            w0[:, r] = np.where(free[r], 0.5 * (w0[:, j] * mask[:, j] + w0[:, k] * mask[:, k]), w0[:, r])
            mask[:, r] = np.where(free[r], np.maximum(mask[:, j], mask[:, k]), mask[:, r])
    if scale != 1.0:
        for i, n in enumerate(names):
            if n.startswith("vis"):
                w0[:, off + i] = np.where(free[off + i], scale * w0[:, off + i], w0[:, off + i])
    return pop._replace(w0=jnp.asarray(w0), mask=jnp.asarray(mask))


# 2.0 camera eyes in open country: 9 columns over 120 degrees, range 6, in the open world at 181 x 181. The first
# of the two steps that prepare the animal for chapter 2 (the second is 2.1; the book tells them as one stage).
# Frozen as 2.1 on 2026-10-12, renamed 2.0 on 2026-10-13. What it does for the lineage: in open country, where
# nothing but its look tells a poison bush from a good one, learning comes back (plasticity off 53, 44, 41% of
# intact; the affect population had let it go, 99, 95, 99). No new cells: every projection
# that read the five old columns reads all nine, each synapse its own (no copies of a circuit per column; what is
# worth what, and in which column it counts, stays for evolution and the pain rule to set). The tectum has no stage
# (user, 2026-10-12): its innate half answers to moving and looming things, which this world does not have, and its
# other half, value choosing a place, is a gate that arrives with the basal ganglia. A first design (a 9-cell map
# with one winner driving orient and approach programmes) was neutral to harmful at generation 0 (STAGE_LOG v35).
# A short adaptation step like hands: no control.
B20 = camera_eyes(B18)
# Generation 0 (`scripts/probes/eyes.py`, the population of trial x.weven, eight worlds, well-fed lifetime): old eyes
# 1144; new columns silent 1199; interpolated 75 (twice the input from the eyes); interpolated and all eye synapses
# times 5/9 1027, with the poison share at 0.06 against 0.10.
EYES_SCALE = 5 / 9    # nine columns instead of five: the same total input from the eyes
_S2 = dict(row_extra=poison_metrics, plastic=True, generations=40, ticks=LIFE_15, eta_max=1.0, dense_plastic=True)
stage(Stage("2.0", "s2_1_eyes", "1.8old", B20, W20, VISION_CH2, BODY_17, world_ch2, **_S2,
            adapt_init=lambda pop, layout: interpolate_eyes(pop, layout, scale=EYES_SCALE),
            notes="camera eyes (9 columns, range 6) in open country at 181 x 181, where learning comes back; the new "
                  "columns start as the mean of their neighbours"))
stage(Stage("x.eyes0", "x_eyes_silent", "1.8old", B20, W20, VISION_CH2, BODY_17, world_ch2, **_S2,
            notes="TRIAL: 2.1 with the new eye columns starting silent"))

# Trials (2026-10-13, STAGE_LOG v40): is there a patch purity at which both learning and the lasting states pay,
# now that the lineage learns again? The frozen camera-eyes population, nothing new, at four times the area of
# chapter 1 (256 x 256), with the same share of the map in patches as the affect world. Not stages of the chain.
W20_256 = replace(W20, height=256, width=256)
for _key, _name, _world, _note in (
        ("x.c256", "x_ch2_256", W20_256, "no patches (the world of chapter 2, larger)"),
        ("x.cmix", "x_ch2_256_pmix", replace(W20_256, patches=320, patch_radius=6),
         "patches of bushes with every type mixed in them (rich and poor ground, not good and bad ground)"),
        ("x.c60", "x_ch2_256_p60", replace(W20_256, patches=320, patch_radius=6, patch_types=True, patch_purity=0.6),
         "patches 60% their own type"),
        ("x.c70", "x_ch2_256_p70", replace(W20_256, patches=320, patch_radius=6, patch_types=True, patch_purity=0.7),
         "patches 70% their own type"),
        ("x.c80", "x_ch2_256_p80", replace(W20_256, patches=320, patch_radius=6, patch_types=True, patch_purity=0.8),
         "patches 80% their own type")):
    stage(Stage(_key, _name, "2.0", B20, _world, VISION_CH2, BODY_17, world_ch2, **_S2,
                notes="TRIAL: the camera-eyes population at 256 x 256: " + _note))


# Trials (2026-10-13, user's idea, STAGE_LOG v41): lasting states that add up. In stage 1.8 a modulator is released
# at once (one poison bite is a full alarm, one good bite a full stay). Here each is a leaky sum of the events and
# the cell that reads it has a threshold counted in bites, so that in a patch of mixed kinds the animal leaves or
# stays on several bites. Start values from generation 0 (`scripts/probes/sum_affect.py`: neutral for the population
# of x.c70): alarm above 1.5 poison bites, fading over about 33 ticks; stay above 9 good bites (a bush has 6),
# fading over about 100. Evolution moves the thresholds (the biases of `flee` and `settle`) and the fade times.
SUM_ALARM, SUM_FED = (1.5, 0.97), (9.0, 0.99)   # (threshold in bites, fraction left after a tick)
B20S = replace(B20,
               modulators=tuple(replace(m, accumulate=1.0) if m.name in ("alarm", "serotonin") else m for m in B20.modulators),
               regions=tuple(replace(r, bias=-3.0 * SUM_ALARM[0], evolve_bias=True) if r.name == "flee" else
                             replace(r, bias=-2.0 * SUM_FED[0], evolve_bias=True) if r.name == "settle" else r
                             for r in B20.regions))


def sum_affect_init(pop, layout):
    """The designed thresholds and fade times (an inherited bias or fade time would be that of release at once)."""
    if "flee" not in layout.names:   # the control (the parent's brain) has no lasting states
        return pop
    b, clear = pop.b, pop.clear
    for cell, mod, (t, d), sens in (("flee", "alarm", SUM_ALARM, 3.0), ("settle", "serotonin", SUM_FED, 2.0)):
        b = b.at[:, layout.region(cell)].set(-sens * t)
        clear = clear.at[:, layout.mod_names.index(mod)].set(d)
    return pop._replace(b=b, clear=clear)


for _key, _name, _purity in (("x.s60", "x_ch2_256_sum_p60", 0.6), ("x.s80", "x_ch2_256_sum_p80", 0.8)):   # 70%: stage 2.1
    stage(Stage(_key, _name, "2.0", B20S,
                replace(W20_256, patches=320, patch_radius=6, patch_types=True, patch_purity=_purity),
                VISION_CH2, BODY_17, world_ch2, **_S2, adapt_init=sum_affect_init,
                notes=f"TRIAL: lasting states that add up, with thresholds; patches {_purity:.0%} their own type"))


# 2.1 tuning for chapter 2 (user, 2026-10-13; STAGE_LOG v40-v42). No new cells and no new circuit: after 2.0 the
# animal is moved into the world of the chapter and its lasting states are made to fit it, in one short adaptation
# step without control.
#   world    four times the area of chapter 1 for the same 256 animals (256 x 256; at 128 crowding hid most of what
#            a better forager gains), more food (density 0.3), and the berry types in patches that are 70% their
#            own kind: mixed enough that a bush has to be known by its look, pure enough that a meal or a poison
#            bite says something about the neighbours;
#   affect   the two lasting states of stage 1.8 add up over bites instead of being released in full by one
#            (`B20S`): in mixed patches one bite says little, several say what kind of patch this is.
# Three seeds, each from its own seed of 2.0, 16 worlds, % of intact: plasticity off 86 78 86, fed off 74 77 86,
# rest off 66 76 69, flee off 92 86 91. With release at once the seeds went three ways at this purity; in the open
# world and in patches of mixed types the lasting states were not used; in purer patches learning was not.
# The order matters (trial `x.direct`, the same stage straight from the affect population, three seeds): the
# lasting states are used (fed off 66 79 66) but learning weakly or not at all (plasticity off 90 84 102; seed 2
# turned its learning rate to 0). The affect population arrives with a learning rate that drifted up while
# learning was idle (0.36); in patches it is easier to switch such learning off than to tune it. Open country
# first (2.0) brings the rate down and learning back; the patches then keep it.
W21 = replace(W20_256, patches=320, patch_radius=6, patch_types=True, patch_purity=0.7)
stage(Stage("2.1", "s2_1_tuning", "2.0", B20S, W21, VISION_CH2, BODY_17, world_ch2, **_S2, adapt_init=sum_affect_init,
            notes="tuning for chapter 2: a wider world with patches 70% their own type, lasting states that add up "
                  "over bites"))
stage(Stage("x.direct", "x_2_1_direct", "1.8old", B20S, W21, VISION_CH2, BODY_17, world_ch2, **_S2,
            adapt_init=lambda pop, layout: sum_affect_init(interpolate_eyes(pop, layout, scale=EYES_SCALE), layout),
            notes="TRIAL: stage 2.1 straight from the affect population (camera eyes, patches and lasting states "
                  "that add up at once): learning does not reliably come back"))

# Trial (user, 2026-10-17): the affect stage itself in the world of 2.1, with states that add up, from the drives
# population, where learning is still used (plasticity off 71 66 76), and with the eyes of chapter 1. If it works,
# it replaces 1.8, and 2.0 and 2.1 leave the chain.
B18S = replace(B18,
               modulators=tuple(replace(m, accumulate=1.0) if m.name in ("alarm", "serotonin") else m for m in B18.modulators),
               regions=tuple(replace(r, bias=-3.0 * SUM_ALARM[0], evolve_bias=True) if r.name == "flee" else
                             replace(r, bias=-2.0 * SUM_FED[0], evolve_bias=True) if r.name == "settle" else r
                             for r in B18.regions))
stage(Stage("x.affect2", "x_affect_ch2world", "1.7", B18S, W21, VISION_CH1, BODY_17,
            lambda exp: learning_world(exp, springs=SPRINGS_17, berries=LEARN_BERRIES), **_S15, adapt_init=sum_affect_init,
            notes="TRIAL: affect from the drives population in the world of 2.1 (256 x 256, patches 70% their own "
                  "type), lasting states that add up, chapter 1 eyes"))
# The same with the learning rates of the drives population times 0.4 (about 0.16, where 2.0 ended): is the inherited
# rate why learning costs more than it saves at the start?
stage(Stage("x.affect2lr", "x_affect_ch2world_lr", "1.7", B18S, W21, VISION_CH1, BODY_17,
            lambda exp: learning_world(exp, springs=SPRINGS_17, berries=LEARN_BERRIES), **_S15,
            adapt_init=lambda pop, layout: sum_affect_init(pop._replace(eta=pop.eta * 0.4), layout),
            notes="TRIAL: x.affect2 with the inherited learning rates times 0.4"))
# The same as x.affect2 with learning rates that mutate multiplicatively (EvolutionConfig.eta_log_std): is the
# additive step, which clips small rates to 0 and leaves 0 only by a jump, why lineages drop learning? (v44)
stage(Stage("x.affect2log", "x_affect_ch2world_log", "1.7", B18S, W21, VISION_CH1, BODY_17,
            lambda exp: learning_world(exp, springs=SPRINGS_17, berries=LEARN_BERRIES), **_S15, adapt_init=sum_affect_init,
            evolution=(("eta_log_std", 0.3),),
            notes="TRIAL: x.affect2 with learning rates that mutate multiplicatively"))
# The same with novel types less alike (two novel types share about 0.25 of their look instead of 0.64): at generation 80
# of x.affect2log, learning gave up good food for less poison at 0.8 and gained at 0.5 (`novel_sim.py`, v44)
stage(Stage("x.affect2sim", "x_affect_ch2world_sim05", "1.7", B18S, W21, VISION_CH1, BODY_17,
            lambda exp: learning_world(exp, springs=SPRINGS_17, berries=LEARN_BERRIES, novel_sim=0.5), **{**_S15, "generations": 80},
            adapt_init=sum_affect_init, evolution=(("eta_log_std", 0.3),),
            notes="TRIAL: x.affect2log with novel types less alike (similarity 0.5)"))

# 1.8 affect (redefined 2026-10-17, STAGE_LOG v43-v45; the first affect stage is `1.8old`, a side path with 2.0 and
# 2.1 after it). The two lasting states of `1.8old`, now adding up over bites (B18S: `flee` above about 1.5 poison
# bites, `settle` above about 9 good bites; thresholds and fade times evolve), from the drives population, with the
# eyes of chapter 1, in the world of chapter 2: 256 x 256 (at 128 crowding hides most of what a better forager gains),
# more food (density 0.3), berry types in patches 70% their own kind. Learning rates mutate multiplicatively
# (`eta_log_std`): with the additive step a lineage that leaves learning idle loses it for good (v44). 100 generations,
# the standard from here on (user): long enough to tell whether a circuit is used, short of full convergence (about
# 160 generations, v44). In patches of one type, learning (by look) and the lasting states (by place) do the same job,
# so a lineage leans on one of them; here it leans on the lasting states, and learning stays weak but recoverable.
W18S = W21
stage(Stage("1.8", "s1_8_affect2", "1.7", B18S, W18S, VISION_CH1, BODY_17,
            lambda exp: learning_world(exp, springs=SPRINGS_17, berries=LEARN_BERRIES), **{**_S15, "generations": 100},
            adapt_init=sum_affect_init, evolution=(("eta_log_std", 0.3),),
            notes="two lasting states that add up over bites: alarm after poison bites (keep going), serotonin after "
                  "good bites (stay); 256 x 256, berry types in patches 70% their own kind"))

# 2.2 pallium (STAGE_LOG v46; ADR-025): a sparse random expansion of the look, as the oldest pallium (piriform-like
# cortex) and the insect mushroom body do it. 128 cells read the eye ahead only (a fovea after orienting: at a bite
# the side eyes see the neighbouring bushes, and whatever they see is taught too; even a few side cells undo most of
# the gain, v46), each from all 8 look features, with fixed random weights (drawn per genome, inherited, never
# mutated); k-WTA keeps the 2 strongest (1.6%: sparser codes spread a lesson less to good food; in real lives a lesson
# reaches good bushes at 0.14 of its strength on the poison with 2 cells, 0.21 with 4, v48). The pain teacher of the association stage teaches pallium -> aversive
# as it teaches look -> aversive. World: the 1.8 world with XOR looks: the four novel types are the corners
# (+-u1 +-u2) / sqrt2 of two colour directions drawn per life, poison ++ and --, good +- and -+, so no weighted sum of
# the look tells them apart and a lesson on one poison clears the other; the pallium can. Novel similarity lowered to
# make up for it (PALLIUM_SIM). From here on the adaptation strength is a gene (user, 2026-10-18), in main and control.
PALLIUM_N, PALLIUM_K, PALLIUM_FAN = 128, 2, 8
PALLIUM_ETA = 0.2     # starting learning rate of pallium -> aversive (2 cells carry a lesson; with 4 cells 0.1, at 0.3
                      # the aversive cells saturated for good food as well after two rounds of meals, assay_xor.py)
PALLIUM_SIM = 0.5     # novel similarity of the pallium world (0.8 before); XOR at 0.8 was too hard even for the pallium,
                      # at 0.6 no learning was used (v48)
_GENE = dict(in_adapt_evolve=True)
B22P = extend(B18S, regions=(R("pallium", PALLIUM_N, sign="exc", alpha=1.0, bias=0.0, evolve_bias=False,
                               kwta=PALLIUM_K, group="forebrain/pallium"),),
              projections=(P("in", "pallium", src_select=("vis+0.app*",), density=PALLIUM_FAN / APPEARANCE, evolve=False),
                           P("pallium", "valence_av", rule="hebb", modulator=CS_MOD["valence_av"], eta_init=PALLIUM_ETA,
                             elig_tau=CS_ELIG["valence_av"], abcd=(0.0, 1.0, 0.0, 0.0), density=1.0)), **_GENE)


def pallium_world(exp, novel_sim: float | None = None, xor: bool = True, reverse: bool = False, one_berry: float = 0.0):
    return learning_world(exp, springs=SPRINGS_17, berries=LEARN_BERRIES, novel_xor=xor, reverse=reverse, one_berry=one_berry,
                          novel_sim=PALLIUM_SIM if novel_sim is None else novel_sim)


# World: patches mixed evenly (purity 0.25 with four novel types: where a bush grows says nothing about its kind).
# At purity 0.7 a poison bite told the animal about the bushes around it, the alarm carried it off, and a lesson by
# look saved only the first bite at the next bush of a kind: learning was used in one seed of three. Paired, on the
# evolved seed-0 population: all learning off costs 24% at purity 0.25 (pallium learning off 19%), against 6% at
# 0.7 (STAGE_LOG v48). The lasting states lose part of their job; accepted (user, 2026-10-18).
W22 = replace(W18S, patch_purity=0.25)
stage(Stage("2.2", "s2_2_pallium", "1.8", B22P, W22, VISION_CH1, BODY_17, pallium_world, **{**_S15, "generations": 100},
            evolution=(("eta_log_std", 0.3),), control_brain=replace(B18S, **_GENE),
            notes="pallium: a sparse random expansion of the eye ahead (128 cells, k-WTA 2) taught by pain onto the "
                  "aversive cells; XOR looks; the adaptation strength evolves"))

EPISODES = 4                     # worlds per generation: 16 lives per genome (4 siblings x 4 worlds) nearly double the
                                 # selection accuracy for 2.3x the GPU time; one large world or many small ones do no
                                 # better (STAGE_LOG v52). Frozen stages ran with 1 and keep it.
EVOLUTION_NEW = (("eta_log_std", 0.3), ("episodes", EPISODES))   # Stage.evolution of 2.3 (frozen) and its drafts
# Stage.evolution of new stages from 2026-10-20 on (user): 2 worlds per generation, 8 lives per genome. The time of a
# generation grows with the lives simulated (a generation ends when every creature is dead), and the harsh-world
# populations live 2-3x longer than those of v52 (more time per life to tell genomes apart): 4 episodes took 30-40 s
# per generation, 2 take about half (cost probe 2026-10-20: 8.7 s vs 4.1 s at generation 0)
EVOLUTION_2EP = (("eta_log_std", 0.3), ("episodes", 2))

# Draft of the dopamine critic (2026-10-19): the 2.2 brain and world, for the generation-0 probes on the frozen 2.2
# population (`gen0.py` with variant `td`); the stage proper is defined once the probes settle the design.
stage(Stage("x.critic", "x_critic", "2.2", B22P, W22, VISION_CH1, BODY_17, pallium_world, **{**_S15, "generations": 100},
            evolution=EVOLUTION_NEW, notes="DRAFT: the 2.2 brain and world as the base of the critic probes"))
# Draft: the same with a reversal: at switch_tick the two poison novel types turn good and the two good ones poison
# (the looks stay; berry_world reverse), so an aversion must be unlearned. Life is 4000 ticks but the incoming
# population lives about 1400 at generation 0, so the switch comes early (gen0.py switch_tick=... overrides it).
REV_SWITCH = 800
stage(Stage("x.reversal", "x_reversal", "2.2", B22P, replace(W22, switch_tick=REV_SWITCH), VISION_CH1, BODY_17,
            lambda exp: pallium_world(exp, reverse=True), **{**_S15, "generations": 100},
            evolution=EVOLUTION_NEW, notes="DRAFT: x.critic with a reversal of the novel types at switch_tick"))

# 2.3 dopamine critic (STAGE_LOG v54-v57; ADR-025): a TD error as a formula, the first step before dopamine cells.
# Value cells (three appetitive, three aversive, rates) read the pallium through synapses that the error teaches;
# one-tick copies of them give the error of each tick
#     da = taste - pain + k (gamma (V_app - V_av) - (V_app' - V_av'))           (' = a tick earlier)
# (Schultz, Dayan & Montague 1997: one signed error, as in mammals). The same error, negated (da_neg), teaches every
# synapse that pain taught in 2.2 (look -> aversive, pallium -> aversive): a predicted pain teaches nothing more, and
# pain that does not come (better than expected) unteaches aversion, which a pain teacher cannot do. Eligibility
# traces take the activity of a tick earlier (elig_delay: the error of tick t is about the move from t-1), and the
# value cells read the pallium of the same tick (one phase later). A short horizon (gamma 0.5) and value scale 3,
# so that a pain that lasts while the bush is in view can be predicted (the state does not say when it ends);
# taste weighs 0.3, as the creature cannot see whether it will eat (taste is half predictable from the pallium).
# Settings from a generation-0 sweep in this world: gamma 0.5 / scale 3 +142 +- 18, gamma 0.7 +124, gamma 0.9 / scale
# 1 +84 (v57). The pallium's wiring belongs to the population slot (per_slot_pallium), in main and control alike.
# World: the reversal: at tick 800 the two poison novel types turn good and the two good ones poison (looks
# unchanged); what was learned stops being true, and only a prediction error unlearns it. Generation 0: the critic
# +144 +- 17 over the pain teacher, which does worse than no learning there (v56).
CRITIC_GAMMA, CRITIC_SCALE, CRITIC_TASTE = 0.5, 3.0, 0.3
CRITIC_ETA, CRITIC_ELIG, CRITIC_VALENCE = 0.1, 0.5, 1.5


def critic(brain: BrainConfig, gamma=None, scale=None, taste=None) -> BrainConfig:
    """`brain` with the formula critic: value cells, their one-tick copies, the error `da` and its negation `da_neg`,
    which replaces the pain teacher (us_av) of every synapse it taught."""
    g, k = CRITIC_GAMMA if gamma is None else gamma, CRITIC_SCALE if scale is None else scale
    t = CRITIC_TASTE if taste is None else taste
    terms = (("us_taste", t), ("us_pain", -1.0), ("value_app", g * k), ("value_av", -g * k),
             ("value_app_prev", -k), ("value_av_prev", k))
    taught = lambda dst, b: P("pallium", dst, rule="hebb", modulator="da", eta_init=CRITIC_ETA, elig_tau=CRITIC_ELIG,
                              abcd=(0.0, b, 0.0, 0.0), density=1.0, w_init=0.0, elig_delay=True)
    pal = next(r.phase for r in brain.regions if r.name == "pallium")
    value = lambda n: replace(R(n, 3, **VALUE), phase=0 if n.endswith("_prev") else pal + 1)
    brain = extend(brain,
                   regions=(R("us_taste", 1, sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, group="us"),
                            *(value(n) for n in ("value_app", "value_av", "value_app_prev", "value_av_prev"))),
                   projections=(P("in", "us_taste", src_select=("taste",), density=1.0, w_init=2.0, evolve=False),
                                taught("value_app", 1.0), taught("value_av", -1.0),
                                P("value_app", "value_app_prev", topology="one_to_one", w_init=1.0, evolve=False),
                                P("value_av", "value_av_prev", topology="one_to_one", w_init=1.0, evolve=False),
                                fixed("value_app", "valence_app", CRITIC_VALENCE),
                                fixed("value_av", "valence_av", CRITIC_VALENCE)),
                   modulators=(Mod("da", terms=terms), Mod("da_neg", terms=tuple((r, -w) for r, w in terms))))
    # the pain-taught synapses keep their inherited plasticity genes (remap_genomes carries A-D over for a projection
    # whose rule stays): so they take the negated error rather than a flipped sign in abcd
    return replace(brain, projections=tuple(replace(p, modulator="da_neg", elig_delay=True) if p.modulator == "us_av"
                                            else p for p in brain.projections))


def per_slot_pallium(brain: BrainConfig) -> BrainConfig:
    """The pallium's random input wiring belongs to the population slot, not to the genome (ProjectionSpec.per_slot:
    no lineage inherits or selects a lucky code; user, 2026-10-19), and what reads the pallium starts at 0 and is
    learned only, as it could only read a code that is not inherited. Generation 0: no cost (v55)."""
    return replace(brain, projections=tuple(
        replace(p, per_slot=True) if p.src == "in" and p.dst == "pallium" else
        replace(p, w_init=0.0, evolve=False) if p.src == "pallium" else p for p in brain.projections))


W23 = replace(W22, switch_tick=REV_SWITCH)
B23 = per_slot_pallium(critic(B22P))
stage(Stage("2.3", "s2_3_critic", "2.2", B23, W23, VISION_CH1, BODY_17, lambda exp: pallium_world(exp, reverse=True),
            **{**_S15, "generations": 100}, evolution=EVOLUTION_NEW, control_brain=per_slot_pallium(B22P),
            notes="dopamine critic as a formula: value cells on the pallium, one TD error teaching them and, negated, "
                  "every synapse the pain teacher taught; reversal world (poison and good novel types swap at tick 800)"))

# Draft (not on the chain): the 2.3 brain and world with one-berry bushes (berry_world one_berry), for generation-0
# probes of the next stage. ONE_BERRY: the food of the one berry, in berries (STAGE_LOG v59)
ONE_BERRY = 3.0
stage(Stage("x.one", "x_one_berry", "2.3", B23, W23, VISION_CH1, BODY_17,
            lambda exp: pallium_world(exp, reverse=True, one_berry=ONE_BERRY),
            **{**_S15, "generations": 100}, evolution=EVOLUTION_NEW, control_brain=per_slot_pallium(B22P),
            notes="DRAFT: 2.3 with one berry per bush (a bite is a whole visit)"))


def pallium_touch(brain: BrainConfig) -> BrainConfig:
    """The pallium also reads `no_touch` (about 0 with something at the mouth, 0.9 otherwise; random weights like its
    look input, per slot where that is), so its code is look x at-the-mouth: a bush seen from afar and the same bush
    at the mouth are two states for the critic. The pallium and the value cells move one phase later, so `no_touch`
    and the look describe the same tick and the value cells still read this tick's code (probe variant `pnt`, v55).
    Lab (v60): with the bush in view from afar for 15 ticks the error at the bites falls to -0.65 (without: -0.87)."""
    pal = next(p for p in brain.projections if p.src == "in" and p.dst == "pallium")
    return replace(brain, regions=tuple(replace(r, phase=r.phase + 1) if r.name in ("pallium", "value_app", "value_av")
                                        else r for r in brain.regions),
                   projections=(*brain.projections, P("no_touch", "pallium", density=1.0, evolve=False,
                                                        per_slot=pal.per_slot)))


# Draft: x.one with the touch-gated pallium; control the 2.3 brain in the same world. World tuned at generation 0
# (STAGE_LOG v60): born three quarters full, a lower metabolic rate (regrowth follows it): lives of about 2500
# ticks and about 26 painful bites at generation 0
ONE_T_WORLD = dict(start_food=0.75, hunger_per_tick=0.015)
stage(Stage("x.onet", "x_one_touch", "2.3", pallium_touch(B23), replace(W23, **ONE_T_WORLD), VISION_CH1, BODY_17,
            lambda exp: pallium_world(exp, reverse=True, one_berry=ONE_BERRY),
            **{**_S15, "generations": 100}, evolution=EVOLUTION_NEW, control_brain=B23,
            notes="DRAFT: one berry per bush; the pallium reads no_touch (look x at the mouth); control: the 2.3 brain"))

# Draft: a harsh world of small meals, no reversal (STAGE_LOG v62-v63). One berry per bush worth one berry (3 food), a
# poison berry costs as much (POISON_FOOD -3), so trying every bush once gains nothing; a quarter of the metabolic
# rate of 2.3 (regrowth follows it), born three quarters full, so one bad bite never kills. The aversion learning rates
# (look and pallium -> valence_av) start x10 in main and control (a start value; evolution moves them from there).
# Generation 0 (frozen 2.3 seed 0, x.onet brain): base 882, learning x4 +240 +- 34, x10 +443 +- 66 (poison share
# 0.46 -> 0.40); lives of about 1400 ticks with about 17 painful bites.
HARSH_WORLD = dict(start_food=0.75, hunger_per_tick=0.005, switch_tick=0)
AV_LEARN = ("in->valence_av[vis*.app*]", "pallium->valence_av")
AV_ETA_START = 10.0
HARSH_REVERSE = 0.0   # 1: the poison and good kinds swap at WorldConfig.switch_tick (probes set both)


def av_eta_init(pop, layout):
    """Start the aversion learning rates AV_ETA_START times their inherited value."""
    pid, names = np.asarray(layout.proj_id), list(layout.proj_names)
    sel = jnp.asarray(np.isin(pid, [names.index(n) for n in AV_LEARN if n in names]))[None]
    return pop._replace(eta=jnp.where(sel, pop.eta * AV_ETA_START, pop.eta))


stage(Stage("x.harsh", "x_harsh", "2.3", pallium_touch(B23), replace(W23, **HARSH_WORLD), VISION_CH1, BODY_17,
            lambda exp: pallium_world(exp, reverse=bool(HARSH_REVERSE), one_berry=1.0),
            **{**_S15, "generations": 100}, evolution=EVOLUTION_NEW, control_brain=B23, adapt_init=av_eta_init,
            notes="DRAFT: one berry worth one, poison as costly as a berry is good, low metabolic rate, no reversal; "
                  "aversion learning starts x10; control: the 2.3 brain"))

# Draft: the harsh world with the reversal at tick 800 (STAGE_LOG v64). Main: the 2.3 brain (formula critic, without the
# touch-gated pallium, which cost about 300 in x.harsh); control: the 2.2 brain (pain-taught, no critic). Start from the
# x.harsh control run (caution by look evolved; run with --init-from). Generation 0 (that population, 16 worlds): aversion
# learning off -706 +- 73, critic value learning off -287 +- 34 (good berries after the switch 9.0 vs 18.8).
HARSH_REV_WORLD = dict(HARSH_WORLD, switch_tick=REV_SWITCH)
stage(Stage("x.harshrev", "x_harsh_rev", "x.harsh", B23, replace(W23, **HARSH_REV_WORLD), VISION_CH1, BODY_17,
            lambda exp: pallium_world(exp, reverse=True, one_berry=1.0),
            **{**_S15, "generations": 100}, evolution=EVOLUTION_NEW, control_brain=per_slot_pallium(B22P),
            notes="DRAFT: the harsh world with a reversal at 800; main the 2.3 critic, control the 2.2 brain"))

def calm_first(generations: int):
    """Schedule: for the first `generations` generations both phases of a life are the first one (no reversal), then
    the world as built (the poison and good kinds swap at WorldConfig.switch_tick)."""
    def wrap(fn):
        def rules(gen, key):
            r = fn(gen, key)
            return r if gen >= generations else jax.tree_util.tree_map(lambda a: jnp.repeat(a[:, :1], 2, axis=1), r)
        return rules
    return wrap


# The critic on the chain (frozen 2026-10-20, replacing 2.3; STAGE_LOG v65): the 2.3 design in the harsh world with the reversal, straight
# from the pallium stage (no intermediate harsh stage). Main the formula critic (B23), control the 2.2 brain; the
# aversion learning rates start x10 in both (av_eta_init), a start value that gives generation 0 learners to select.
HARSH_CALM = 50   # 2.3h: generations without the reversal first (user, 2026-10-20: one stage, no intermediate one)
stage(Stage("2.3h", "s2_3h_critic", "2.2", B23, replace(W23, **HARSH_REV_WORLD), VISION_CH1, BODY_17,
            lambda exp: pallium_world(exp, reverse=True, one_berry=1.0),
            **{**_S15, "generations": 100}, evolution=EVOLUTION_2EP, control_brain=per_slot_pallium(B22P),
            adapt_init=av_eta_init, schedule=calm_first(HARSH_CALM),
            notes="dopamine critic as a formula in the harsh world (one berry worth one, poison as costly, "
                  "low metabolic rate); the reversal at 800 from generation HARSH_CALM on; control the 2.2 brain"))

# ------------------------------------------------------------------ the critic built from cells (STAGE_LOG v66)
# The simplest dopamine circuit (user, 2026-10-20): one dopamine cell with a resting rate; its rate minus that rest is
# the error (a burst is better than predicted, a pause worse). Taste excites it; pain excites the habenula, which
# inhibits it. The value cells are striatal and inhibitory, learned from the pallium: `value_app` (D1-type rule,
# strengthened by bursts) inhibits the dopamine cell, so a predicted reward cancels its burst; `value_av` (D2-type
# rule, strengthened by pauses) inhibits the habenula, so a predicted pain cancels its pause and an omitted one gives
# a burst. Without a delayed path the error is r - V of this tick (Rescorla-Wagner), not TD: a fast pallidal path
# can add the change of value later (BRAIN_EVOLUTION, Dopamine). Both cells rest at 0.5 (fixed biases); the links
# are hard-wired with strengths evolution may tune (`fixed`). The habenula rests above zero so that a predicted pain
# can push it down: a rectified cell at rest could not signal the relief of an omitted pain.
DA_REST, HAB_REST = 0.5, 0.5
DA_SCALE = 2.0          # modulator = DA_SCALE * (rate - DA_REST): from -1 (silent) to +1 (saturated)
DA_TASTE = 0.23         # us_taste -> dopamine: a good bite about +0.3, the taste weight of the formula critic
HAB_PAIN, HAB_DA = 2.0, 1.0   # pain -> habenula, habenula -| dopamine: a full pain about -0.9
VAL_DA, VAL_HAB = 1.0, 3.0    # value_app -| dopamine, value_av -| habenula


def dopamine_cells(brain: BrainConfig, delay: bool = False) -> BrainConfig:
    """`brain` (with the formula critic, `critic()`) with the error computed by cells instead of the formula. `delay`:
    the error teaches the synapses active a tick earlier (ProjectionSpec.elig_delay, as the formula critic does);
    without it, those active on the tick it is about (the pairing Rescorla-Wagner asks for)."""
    at = float(np.arctanh(DA_REST)) + HAB_DA * HAB_REST   # dopamine at rest with the habenula at rest
    # the strengths are per unit of mean value: every value cell has its own synapse, so each gets 1 / n of it
    # (with the full strength per synapse the value scale was 3x the formula's; STAGE_LOG v68)
    nv = next(r.size for r in brain.regions if r.name == "value_app")
    gone = ("value_app_prev", "value_av_prev")
    regions = tuple(replace(r, sign="inh") if r.name in ("value_app", "value_av") else r
                    for r in brain.regions if r.name not in gone)
    pal = next(r.phase for r in brain.regions if r.name == "pallium")
    projections = tuple(replace(p, elig_delay=delay) if p.modulator in ("da", "da_neg") else p
                        for p in brain.projections
                        if p.src not in gone and p.dst not in gone
                        and not (p.src in ("value_app", "value_av") and p.dst.startswith("valence")))
    return replace(brain, regions=regions + (
        R("habenula", 1, sign="inh", alpha=1.0, bias=float(np.arctanh(HAB_REST)), evolve_bias=False, phase=pal + 2,
          group="epithalamus"),
        R("dopamine", 1, sign="exc", alpha=1.0, bias=at, evolve_bias=False, phase=pal + 3, group="midbrain/dopamine")),
        projections=projections + (
            P("in", "habenula", src_select=("pain",), density=1.0, w_init=HAB_PAIN, evolve=False, tune=True),
            # the outputs of tonically active cells are not tuned: a stronger one would move the dopamine cell's
            # rest away from DA_REST, a standing error (STAGE_LOG v67)
            fixed("habenula", "dopamine", HAB_DA, tune=False), fixed("us_taste", "dopamine", DA_TASTE),
            fixed("value_app", "dopamine", VAL_DA / nv), fixed("value_av", "habenula", VAL_HAB / nv)),
        modulators=tuple(m for m in brain.modulators if m.name not in ("da", "da_neg")) + (
            Mod("da", pos="dopamine", baseline=DA_REST, scale=DA_SCALE),
            Mod("da_neg", pos="dopamine", baseline=DA_REST, scale=-DA_SCALE)))


# The TD form (STAGE_LOG v67; agreed with the user 2026-10-20): each value reaches the dopamine signal twice.
#   fast, this tick: value_app -| ventral_pallidum -| dopamine (disinhibition: a rising good value is a burst);
#                    value_av -| gp_habenular -| habenula (a rising bad value releases the habenula: a pause)
#   slow, a tick late: value_app -| dopamine, value_av -| habenula through slow synapses (ProjectionSpec.lag: the
#                    value of the tick before)
# error = taste - pain + gamma (V_app - V_av)(t) - (V_app - V_av)(t-1), gamma = fast / slow strength (tunable). The
# slow path is the striosomes' direct, slower inhibition of the dopamine cells (the Houk-Adams-Barto critic); the
# habenula-projecting pallidum (GPh; Stephenson-Jones et al. 2013 in the lamprey, 2016 in mice) carries the bad-news
# path; a slow striosomal path onto the habenula is the plausible analogue, a modelling choice. The pallidal cells
# rest at 0.5. Strengths from the formula's value scale 3 and gamma 0.5 at the cells' working point (tanh' = 0.75 at
# rest 0.5): the error is about 1.5 x the dopamine cell's change of drive.
TD_SLOW_DA, TD_SLOW_HAB = 2.0, 2.67     # value_app -| dopamine, value_av -| habenula (slow synapses)
# the fast path's input strengths set gamma: x1.8 of the formula's 0.5, about 0.9 (generation 0 on frozen 2.3h, 16
# worlds, the value scale corrected: gamma 0.5 1041, 0.75 +34 +- 16, 0.9 +56 +- 18 (1098; the formula 1072); STAGE_LOG
# v68; tunable)
TD_VAL_VP, TD_VP_DA = 2.4, 1.0          # value_app -| ventral_pallidum, ventral_pallidum -| dopamine
TD_VAL_GPH, TD_GPH_HAB = 3.2, 1.0       # value_av -| gp_habenular, gp_habenular -| habenula
VSTR = "forebrain/basal_ganglia/ventral_striatum"
PALLIDUM = "forebrain/basal_ganglia/pallidum"


def dopamine_td(brain: BrainConfig) -> BrainConfig:
    """`brain` (with the formula critic) with the TD error computed by cells: dopamine_cells (delay on) with the
    direct value links made slow and fast pallidal paths added."""
    b = dopamine_cells(brain, delay=True)
    rest = float(np.arctanh(0.5))
    nv = next(r.size for r in brain.regions if r.name == "value_app")   # strengths per unit of mean value (above)
    pal = next(r.phase for r in b.regions if r.name == "pallium")
    regions = []
    for r in b.regions:
        if r.name in ("value_app", "value_av"):
            r = replace(r, group=VSTR)
        elif r.name == "habenula":   # rests at HAB_REST with the GP-habenular cell at its rest
            r = replace(r, bias=rest + TD_GPH_HAB * 0.5, phase=pal + 3)
        elif r.name == "dopamine":   # rests at DA_REST with the habenula and the ventral pallidum at theirs
            r = replace(r, bias=rest + HAB_DA * HAB_REST + TD_VP_DA * 0.5, phase=pal + 4)
        regions.append(r)
    tonic = lambda n: R(n, 1, sign="inh", alpha=1.0, bias=rest, evolve_bias=False, phase=pal + 2, group=PALLIDUM)
    slow = {("value_app", "dopamine"): TD_SLOW_DA / nv, ("value_av", "habenula"): TD_SLOW_HAB / nv}
    projections = tuple(replace(p, w_init=slow[(p.src, p.dst)], lag=True) if (p.src, p.dst) in slow else p
                        for p in b.projections)
    return replace(b, regions=tuple(regions) + (tonic("ventral_pallidum"), tonic("gp_habenular")),
                   projections=projections + (
                       fixed("value_app", "ventral_pallidum", TD_VAL_VP / nv),
                       fixed("ventral_pallidum", "dopamine", TD_VP_DA, tune=False),
                       fixed("value_av", "gp_habenular", TD_VAL_GPH / nv),
                       fixed("gp_habenular", "habenula", TD_GPH_HAB, tune=False)))


B24 = dopamine_cells(B23)
B24TD = dopamine_td(B23)
B24D = dopamine_cells(B23, delay=True)
# Draft for the probes (lab tests, generation 0): the 2.3h world as frozen (reversal at 800), from the 2.3h population
stage(Stage("x.dacells", "x_dopamine_cells", "2.3h", B24, replace(W23, **HARSH_REV_WORLD), VISION_CH1, BODY_17,
            lambda exp: pallium_world(exp, reverse=True, one_berry=1.0),
            **{**_S15, "generations": 100}, evolution=EVOLUTION_2EP, control_brain=B23,
            notes="DRAFT: the critic built from cells (one dopamine cell with a resting rate, habenula, inhibitory "
                  "striatal value cells; Rescorla-Wagner form); control the formula critic"))
stage(Stage("x.datd", "x_dopamine_td", "2.3h", B24TD, replace(W23, **HARSH_REV_WORLD), VISION_CH1, BODY_17,
            lambda exp: pallium_world(exp, reverse=True, one_berry=1.0),
            **{**_S15, "generations": 100}, evolution=EVOLUTION_2EP, control_brain=B23,
            notes="DRAFT: the TD critic built from cells (pallidal fast paths, striosomal slow paths); control the "
                  "formula critic"))
stage(Stage("x.dacellsd", "x_dopamine_cells_delay", "2.3h", B24D, replace(W23, **HARSH_REV_WORLD), VISION_CH1, BODY_17,
            lambda exp: pallium_world(exp, reverse=True, one_berry=1.0),
            **{**_S15, "generations": 100}, evolution=EVOLUTION_2EP, control_brain=B23,
            notes="DRAFT: x.dacells with the error teaching the synapses of a tick earlier (elig_delay)"))

# Stage 2.4: the critic built from cells (TD form), on top of 2.3h in its world (the reversal from generation 0: the
# population has lived with it). Control: the formula critic of 2.3h. Judged by use and by main not clearly worse.
stage(Stage("2.4", "s2_4_dopamine", "2.3h", B24TD, replace(W23, **HARSH_REV_WORLD), VISION_CH1, BODY_17,
            lambda exp: pallium_world(exp, reverse=True, one_berry=1.0),
            **{**_S15, "generations": 100}, evolution=EVOLUTION_2EP, control_brain=B23,
            notes="dopamine critic built from cells: a dopamine cell with a resting rate, habenula, inhibitory "
                  "ventral striatal value cells with fast pallidal and slow striosomal paths (TD); control the "
                  "formula critic"))

# ------------------------------------------------------------------ running

EVOLUTION_OVERRIDES: dict = {}   # set from the command line (--mutation-prob), applied to every stage
SIBLINGS = 4                     # 256 agents = 64 genomes x 4 siblings (EvolutionConfig.siblings)
CROSSOVER = 1.0                  # every child has two parents (whole neurons from either): in the recovery
                                 # benchmark this repaired a damaged population 6-7% further than asexual
                                 # reproduction in 40 generations, in every seed (STAGE_LOG, v20)


def make_exp(s: Stage, brain: BrainConfig, name: str, generations: int, seed: int) -> ExperimentConfig:
    return ExperimentConfig(name=name, world=s.world, vision=s.vision, body=s.body, brain=brain,
                            evolution=EvolutionConfig(generations=generations, ticks_per_generation=s.ticks,
                                                      eta_max=s.eta_max, **{"siblings": SIBLINGS, "crossover": CROSSOVER, **dict(s.evolution), **EVOLUTION_OVERRIDES},
                                                      plastic=s.plastic, seed=seed))


def run_stage(key: str, generations: int | None = None, control: bool = False, init_from: str | None = "auto",
              seed: int = 0, dashboard: bool = True, suffix: str = "", ticks: int | None = None, adapt: bool = True):
    s = STAGES[key]
    if ticks:   # a trial with a longer or shorter life than the stage defines; use a suffix to keep it apart
        s = replace(s, ticks=ticks)
    gens = generations or s.generations
    if control:
        assert s.parent, "stage has no parent, so no control"
        brain, name = s.control_brain or STAGES[s.parent].brain, s.name + "_control"
    else:
        brain, name = s.brain, s.name
    name += suffix
    if init_from == "none" and s.parent:
        name += "_scratch"          # not part of the lineage (latest_run of the stage must stay the lineage run)
    exp = make_exp(s, brain, name, gens, seed)
    init = None
    if init_from == "auto" and s.parent:
        init_from = str(latest_run(STAGES[s.parent].name))
    if init_from and init_from not in ("auto", "none"):
        print(f"warm start from {init_from}")
        init = load_population(init_from, exp, seed=seed)
        from life.run import make_layout
        lay = make_layout(exp)
        if s.adapt_init and adapt:   # not when a run of this stage is continued: the change is already in it
            init = s.adapt_init(init, lay)
        if s.dense_plastic and not control:
            init = init._replace(mask=jnp.where(((np.asarray(lay.rule) > 0) & (np.asarray(lay.allowed) > 0))[None], 1.0, init.mask))
    rs, rules_fn = s.build(exp)
    if s.schedule:
        rules_fn = s.schedule(rules_fn)
    res = run_evolution(exp, rs, s.fitness or default_fitness, rules_for_generation=rules_fn, init_population=init,
                        row_extra=s.row_extra(rs) if s.row_extra else None, dashboard=dashboard)
    if init is not None:   # a run started from an earlier run of its own experiment continues it (book: one curve)
        from pathlib import Path
        src = Path(init_from).resolve()
        if src.parent == Path(res["out_dir"]).resolve().parent:
            (Path(res["out_dir"]) / "continues.txt").write_text(src.name)
    return res


def stage_of_run(run_dir) -> Stage:
    from pathlib import Path
    name = Path(run_dir).resolve().parent.name
    found = [s for s in STAGES.values() if name.startswith(s.name)]   # the longest name: x_ch2_256_p70, not x_ch2_256
    if found:
        return max(found, key=lambda s: len(s.name))
    raise KeyError(f"no stage for experiment {name}")


def lesion(run_dir: str, regions: list[str] | None = None, worlds: int = 8, seed: int = 123,
           only: tuple | None = None, per_world: bool = False) -> dict:
    """Lesion study: the run's final population in its own world, intact and with each region silenced (all
    outgoing synapses removed). Averages over `worlds` independent worlds. Returns {label: stats means}.
    The worlds depend only on `seed`, so two runs of the same stage are measured in the same worlds.
    `only`: restrict to these labels (e.g. ("intact",) to re-evaluate a population)."""
    from pathlib import Path
    from life.config import ExperimentConfig
    from life.run import make_layout, make_simulate
    run_dir = Path(run_dir)
    exp = ExperimentConfig.from_json((run_dir / "config.json").read_text())
    s = stage_of_run(run_dir)
    layout = make_layout(exp)
    pop = load_population(run_dir)
    rs, rules_fn = s.build(exp)
    extra = s.row_extra(rs) if s.row_extra else (lambda st, f: {})
    sim = jax.jit(make_simulate(exp, record=False))
    nodep = replace(exp, brain=replace(exp.brain, projections=tuple(replace(p, depression=()) for p in exp.brain.projections)))
    sim_nodep = jax.jit(make_simulate(nodep, record=False)) if layout.has_dep else None
    fit_fn = s.fitness or default_fitness
    regions = regions or [n for n in layout.names if n not in ("in", "out")]
    keys = jax.random.split(jax.random.PRNGKey(seed), worlds)
    out = {}
    labels = ["intact"] + (["no_plasticity"] if float(pop.eta.max()) > 0 else []) \
        + (["no_depression"] if layout.has_dep else []) + regions
    if only:
        labels = [l for l in labels if l in only]
    for label in labels:
        g = pop
        if label == "no_plasticity":
            g = pop._replace(eta=jnp.zeros_like(pop.eta))
        elif label not in ("intact", "no_depression"):
            rows = layout.region(label)
            # silence the region: no outgoing synapses and no activity (so broadcast modulators read 0 too)
            g = pop._replace(w0=pop.w0.at[:, rows, :].set(0.0), mask=pop.mask.at[:, rows, :].set(0.0),
                             b=pop.b.at[:, rows].set(-10.0))
        acc = []
        for k in keys:
            kr, ks = jax.random.split(k)
            rules = rules_fn(0, kr)
            if rules.food_value.ndim > 1 + (exp.world.switch_tick > 0):
                rules = jax.tree_util.tree_map(lambda a: a[0], rules)
            st, _ = (sim_nodep if label == "no_depression" else sim)(rules, g, ks)
            fit = fit_fn(st)
            row = dict(fitness=float(fit.mean()), eaten=float(st["eaten"].mean()), pain=float(st["pain"].mean()),
                       alive=float(st["alive_ticks"].mean()), temp=float(st["temp_mean"].mean()))
            row.update({k2: float(v) for k2, v in extra(st, fit).items()})
            acc.append(row)
        out[label] = {k2: float(np.mean([r[k2] for r in acc])) for k2 in acc[0]}
        if per_world:   # the fitness in every world, for a standard error over worlds
            out[label]["fitness_worlds"] = [r["fitness"] for r in acc]
    return out


def seed_runs(s: Stage, control: bool = False) -> list:
    """Newest finished run of the stage's lineage directory and of every `_seedN` variant."""
    base = s.name + ("_control" if control else "")
    dirs = []
    for d in sorted(RUNS_DIR.iterdir()):
        rest = d.name[len(base):]
        if d.name.startswith(base) and (rest == "" or rest.startswith("_seed")):
            runs = sorted(r for r in d.iterdir() if (r / "population.npz").exists())
            if runs:
                dirs.append(runs[-1])
    return dirs


def lesions(key: str, worlds: int = 8) -> dict:
    """Lesion study over every seed of a stage: fitness with each region silenced, in % of the intact population.
    This is the test of whether a circuit is *used* (the stage criterion); one seed can mislead."""
    res = [lesion(str(r), worlds=worlds) for r in seed_runs(STAGES[key])]
    out = {}
    for label in res[0]:
        if label != "intact":
            out[label] = [100.0 * r[label]["fitness"] / r["intact"]["fitness"] for r in res]
    print(f"{key}: intact fitness per seed " + " ".join(f"{r['intact']['fitness']:.0f}" for r in res))
    print("lesion".ljust(18) + "% of intact per seed".rjust(26) + "mean".rjust(7))
    for label, rel in out.items():
        print(label.ljust(18) + " ".join(f"{v:7.0f}" for v in rel).rjust(26) + f"{np.mean(rel):7.0f}")
    return out


def versus(key: str, worlds: int = 8, seed: int = 321) -> list:
    """Head-to-head test: half of the stage's main population and half of its control population live in the same
    worlds, so both face the same depleted food supply. The control agents run the new brain with the regions
    their own brain lacks silenced and (if their own brain had none) plasticity off. Differences that live in the
    layout rather than in a region (depression, decay, a new receptor on an old region) are shared by both halves.
    Returns, per seed, mean fitness of the main half minus that of the control half."""
    from life.config import ExperimentConfig
    from life.run import make_layout, make_simulate
    s = STAGES[key]
    diffs = []
    for main_run, ctrl_run in zip(seed_runs(s), seed_runs(s, control=True)):
        exp = ExperimentConfig.from_json((main_run / "config.json").read_text())
        layout = make_layout(exp)
        old_layout = make_layout(ExperimentConfig.from_json((ctrl_run / "config.json").read_text()))
        main = load_population(main_run)
        own = load_population(ctrl_run)
        ctrl = load_population(ctrl_run, exp, seed=0)
        for r in [n for n in layout.names if n not in old_layout.names]:
            rows = layout.region(r)
            ctrl = ctrl._replace(w0=ctrl.w0.at[:, rows, :].set(0.0), mask=ctrl.mask.at[:, rows, :].set(0.0),
                                 b=ctrl.b.at[:, rows].set(-10.0))
        if float(own.eta.max()) == 0:
            ctrl = ctrl._replace(eta=jnp.zeros_like(ctrl.eta))
        n = main.b.shape[0]               # genomes; each lives as exp.evolution.siblings individuals
        half = n // 2
        sib = exp.world.num_agents // n
        mixed = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a[:half], b[:n - half]]), main, ctrl)
        rs, rules_fn = s.build(exp)
        sim = jax.jit(make_simulate(exp, record=False))
        fit_fn = s.fitness or default_fitness
        d = []
        for k in jax.random.split(jax.random.PRNGKey(seed), worlds):
            kr, ks = jax.random.split(k)
            rules = rules_fn(0, kr)
            if rules.food_value.ndim > 1 + (exp.world.switch_tick > 0):
                rules = jax.tree_util.tree_map(lambda a: a[0], rules)
            fit = fit_fn(sim(rules, mixed, ks)[0])
            d.append((float(fit[:half * sib].mean()), float(fit[half * sib:].mean())))
        m, c = np.mean(d, axis=0)
        diffs.append(m - c)
        print(f"{main_run.parent.name}: main half {m:.0f}, control half {c:.0f}, difference {m - c:+.0f}", flush=True)
    se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else float("nan")
    print(f"{key} head to head, main - control: {np.mean(diffs):+.1f} +-{se:.1f} (standard error over {len(diffs)} seeds)")
    return diffs


def respond(run_dir: str, regions: list[str] | None = None, steps: int = 4) -> list[dict]:
    """Stimulus-response readout: the run's final population (inherited weights) shown each object type straight
    ahead at distance 1, and held in hand, for `steps` brain steps with nothing else in view. Returns rows with
    the mean activity per region and the mean action probabilities (averaged over the population)."""
    from pathlib import Path
    from life.config import ExperimentConfig
    from life.run import make_layout
    from life import brain as B, actions as A
    run_dir = Path(run_dir)
    exp = ExperimentConfig.from_json((run_dir / "config.json").read_text())
    layout = make_layout(exp)
    pop = load_population(run_dir)
    rs, rules_fn = stage_of_run(run_dir).build(exp)
    rules = rules_fn(0, jax.random.PRNGKey(0))
    while rules.appearance.ndim > 2:
        rules = jax.tree_util.tree_map(lambda a: a[0], rules)
    app = np.asarray(rules.appearance)
    names = list(layout.in_names)
    regions = regions or [n for n in layout.names if n not in ("in", "out")]
    K = exp.vision.appearance_dim

    def obs_for(obj, held):
        o = np.zeros(layout.n_in, np.float32)
        o[names.index("food")] = 0.7
        if "temperature" in names:
            o[names.index("temperature")] = 0.5
        if held:
            o[names.index("held")] = 1.0
            for k in range(K):
                o[names.index(f"held_app{k}")] = app[obj, k]
        else:
            o[names.index("vis+0.hit")], o[names.index("vis+0.near")] = 1.0, 1.0 - 1.0 / exp.vision.range
            for k in range(K):
                o[names.index(f"vis+0.app{k}")] = app[obj, k]
        return jnp.asarray(o)

    def run_one(g, obs):
        st = B.init_state(g, layout)
        for _ in range(steps):
            st, _ = B.step(exp.brain, layout, g, st, obs, jnp.zeros(3), jax.random.PRNGKey(0))
        h = (st.x @ B.effective(layout, st.w) + g.b)[-layout.n_out:]
        p = jax.nn.softmax(h * exp.brain.logit_gain / max(exp.brain.action_temperature, 1e-3))
        return st.x, p

    g0 = pop._replace(eta=jnp.zeros_like(pop.eta))
    rows = []
    for obj in range(1, rs.size):
        for held in (False, True):
            if held and not rs.holdable[obj]:
                continue
            x, p = jax.vmap(lambda g: run_one(g, obs_for(obj, held)))(g0)
            row = {"object": rs.names[obj], "view": "held" if held else "ahead"}
            row.update({r: float(x[:, layout.region(r)].mean()) for r in regions})
            row.update({f"P({a})": float(p[:, i].mean()) for i, a in enumerate(A.NAMES[:6])})
            rows.append(row)
    return rows


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "chain":
        # chain <first> <last> [--generations N] [--no-control]: each stage (main and control in parallel,
        # both from the parent's newest main run), then the next. Logs in runs/logs/<name>.log.
        import subprocess
        from pathlib import Path
        q = argparse.ArgumentParser(prog="stages chain")
        q.add_argument("first")
        q.add_argument("last")
        q.add_argument("--generations", type=int, default=None)
        q.add_argument("--no-control", action="store_true")
        q.add_argument("--mutation-prob", default=None)
        b = q.parse_args(argv[1:])
        keys = list(STAGES)
        todo = keys[keys.index(b.first): keys.index(b.last) + 1]
        logs = RUNS_DIR / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        for k in todo:
            s = STAGES[k]
            parent = str(latest_run(STAGES[s.parent].name)) if s.parent else "none"
            base = [sys.executable, "-m", "life.experiments.stages", k, "--init-from", parent]
            if b.generations:
                base += ["--generations", str(b.generations)]
            if b.mutation_prob:
                base += ["--mutation-prob", str(b.mutation_prob)]
            jobs = [(s.name, base)] + ([] if b.no_control or not s.parent else [(s.name + "_control", base + ["--control"])])
            procs = [subprocess.Popen(cmd, stdout=open(logs / f"{n}.log", "w"), stderr=subprocess.STDOUT)
                     for n, cmd in jobs]
            codes = [p.wait() for p in procs]
            print(f"stage {k}: exit codes {codes}", flush=True)
            if any(codes):
                return
        return
    if argv and argv[0] == "replicate":
        # replicate <key> --seeds 1,2 [--generations N] [--mutation-prob p]: main + control per seed, in parallel,
        # from the parent's newest lineage run; experiment names get the suffix _seed<S>
        import subprocess
        q = argparse.ArgumentParser(prog="stages replicate")
        q.add_argument("key")
        q.add_argument("--seeds", default="1,2")
        q.add_argument("--generations", type=int, default=None)
        q.add_argument("--mutation-prob", default="0.1")
        b = q.parse_args(argv[1:])
        s = STAGES[b.key]
        parent = str(latest_run(STAGES[s.parent].name))
        logs = RUNS_DIR / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        for seed in b.seeds.split(","):
            base = [sys.executable, "-m", "life.experiments.stages", b.key, "--init-from", parent, "--seed", seed,
                    "--suffix", f"_seed{seed}", "--mutation-prob", b.mutation_prob]
            if b.generations:
                base += ["--generations", str(b.generations)]
            jobs = [(f"{s.name}_seed{seed}", base), (f"{s.name}_control_seed{seed}", base + ["--control"])]
            procs = [subprocess.Popen(cmd, stdout=open(logs / f"{n}.log", "w"), stderr=subprocess.STDOUT) for n, cmd in jobs]
            print(f"stage {b.key} seed {seed}: exit codes {[p.wait() for p in procs]}", flush=True)
        return
    if argv and argv[0] == "summary":
        # summary <key> [--window 50] [--worlds 8]: main and control over every seed (run name variants).
        # Two cheap ways to cut measurement noise: the mean over the last `window` generations, and the final
        # population re-evaluated in `worlds` fresh worlds that are the same for every run (0 = skip).
        # What remains, the spread between seeds, is how far lineages drift apart; only more seeds average that.
        from life.compare import load, window_means
        q = argparse.ArgumentParser(prog="stages summary")
        q.add_argument("key")
        q.add_argument("--window", type=int, default=50)
        q.add_argument("--worlds", type=int, default=8)
        b = q.parse_args(argv[1:])
        s = STAGES[b.key]
        se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        res = {}
        for label, base in (("main", s.name), ("control", s.name + "_control")):
            dirs = []
            for d in sorted(RUNS_DIR.iterdir()):
                rest = d.name[len(base):]
                if d.name.startswith(base) and (rest == "" or rest.startswith("_seed")):
                    runs = sorted(r for r in d.iterdir() if (r / "population.npz").exists())
                    if runs:
                        dirs.append(runs[-1])
            if not dirs:
                continue
            lasts = [window_means(load(r), b.window)[1] for r in dirs]
            keys = [k for k in ("fit_mean", "alive_ticks", "eaten", "pain", "temp_mean", "poison_frac") if k in lasts[0]]
            vals = {k: [x[k] for x in lasts] for k in keys}
            if b.worlds > 0:
                vals["re-evaluated"] = [lesion(str(r), worlds=b.worlds, only=("intact",))["intact"]["fitness"] for r in dirs]
            res[label] = vals
            print(f"{label:8s} n={len(dirs)}  " + "  ".join(
                f"{k} {np.mean(v):.3g} +-{se(v):.2g} [{' '.join(f'{x:.3g}' for x in v)}]" for k, v in vals.items()))
        if len(res) == 2:
            for k in ("fit_mean", "re-evaluated"):
                if k in res["main"] and len(res["main"][k]) == len(res["control"][k]):
                    diff = np.array(res["main"][k]) - np.array(res["control"][k])
                    print(f"main - control, {k}: {diff.mean():+.3g} +-{se(diff):.2g} (standard error over {len(diff)} seeds)")
        return res
    if argv and argv[0] == "respond":
        q = argparse.ArgumentParser(prog="stages respond")
        q.add_argument("run_dir")
        q.add_argument("--regions", default="")
        b = q.parse_args(argv[1:])
        rows = respond(b.run_dir, [r for r in b.regions.split(",") if r] or None)
        cols = [c for c in rows[0] if c not in ("object", "view")]
        print("object".ljust(34) + "view ".ljust(7) + "".join(c[:11].rjust(12) for c in cols))
        for r in rows:
            print(r["object"][:33].ljust(34) + r["view"].ljust(7) + "".join(f"{r[c]:12.3f}" for c in cols))
        return rows
    if argv and argv[0] == "versus":
        q = argparse.ArgumentParser(prog="stages versus")
        q.add_argument("key")
        q.add_argument("--worlds", type=int, default=8)
        b = q.parse_args(argv[1:])
        return versus(b.key, b.worlds)
    if argv and argv[0] == "lesions":
        q = argparse.ArgumentParser(prog="stages lesions")
        q.add_argument("key")
        q.add_argument("--worlds", type=int, default=8)
        b = q.parse_args(argv[1:])
        return lesions(b.key, b.worlds)
    if argv and argv[0] == "lesion":
        q = argparse.ArgumentParser(prog="stages lesion")
        q.add_argument("run_dir")
        q.add_argument("--regions", default="")
        q.add_argument("--worlds", type=int, default=8)
        b = q.parse_args(argv[1:])
        res = lesion(b.run_dir, [r for r in b.regions.split(",") if r] or None, b.worlds)
        cols = list(next(iter(res.values())))
        print("lesioned region".ljust(18) + "".join(c[:14].rjust(15) for c in cols))
        for k, row in res.items():
            print(k.ljust(18) + "".join(f"{row[c]:15.3f}" for c in cols))
        return res
    p = argparse.ArgumentParser()
    p.add_argument("stage", help="stage key, e.g. 1.0, 'list', or 'lesion <run dir>'")
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--control", action="store_true", help="parent's brain in this stage's world, same start")
    p.add_argument("--init-from", default="auto", help="'auto' (newest parent run), 'none', or a run dir")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mutation-prob", type=float, default=None, help="per-synapse chance that a weight mutates")
    p.add_argument("--suffix", default="", help="appended to the run's experiment name (variants)")
    p.add_argument("--ticks", type=int, default=None, help="ticks per life, instead of the stage's own (trials)")
    p.add_argument("--continue", dest="cont", action="store_true",
                   help="--init-from is a run of this stage: do not apply the stage's change to the start population again")
    p.add_argument("--dashboard", action="store_true",
                   help="also write the run's full dashboard.html (about 150 MB and a peak in memory at the end of the "
                        "run; the book's light dashboards are made from the recording: `python -m life.book dashboards`)")
    a = p.parse_args(argv)
    if a.stage == "list":
        for s in STAGES.values():
            print(f"{s.key:5s} {s.name:32s} parent={s.parent}  {s.notes}")
        return
    if a.mutation_prob is None and a.init_from != "none" and STAGES[a.stage].parent:
        # a warm-started population is near its optimum: with every weight mutating in every child it loses ground
        # (mutational load; the trial x.c256 of 2026-10-13 lost a quarter in 40 generations at 1.0, STAGE_LOG v40)
        a.mutation_prob = 0.1
    if a.mutation_prob is not None:
        EVOLUTION_OVERRIDES["weight_mutation_prob"] = a.mutation_prob
    return run_stage(a.stage, a.generations, a.control, a.init_from, a.seed, suffix=a.suffix, ticks=a.ticks,
                     dashboard=a.dashboard, adapt=not a.cont)


if __name__ == "__main__":
    main()
