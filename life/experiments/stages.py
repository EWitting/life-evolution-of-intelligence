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
    dense_plastic: bool = False   # warm start: every synapse of a learned projection exists (the inherited ones keep
                                  # their weight, the others start at 0); an absent synapse could not learn
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
                weights: dict | None = None, duds: tuple = (), berries: int | None = None):
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
    tick stays the same and a smaller bush only means more, smaller visits."""
    berries = int(BUSH_BERRIES if berries is None else berries)
    data = ohol.load()
    ids = [BUSH, BERRY, EMPTY_BUSH] + ([ONION_PLANT, ONION] if onions else []) + ([HOT_SPRING] if springs else [])
    sets = [(variant(v), COLOURS[v - 1] + " ") for v in range(1, n_types)]
    rs = ohol.slice_ruleset(data, ids, ticks_per_second=exp.world.ticks_per_ohol_second,
                            max_decay_ticks=exp.world.max_decay_ticks, clone_sets=sets,
                            extra_decays={EMPTY_BUSH: (BUSH, regrow_ticks(exp, berries))})
    for v in range(n_types):
        rs.num_uses[rs.local(variant(v)[BUSH])] = berries
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

    def novel_appearance(key):   # [E, types, 3 objects, K]
        u = jax.random.normal(key, (E, len(novel_looks), K - 1))
        u = u / jnp.linalg.norm(u, axis=-1, keepdims=True)
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
            new = novel_appearance(ka)
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
    return P(src, dst, density=1.0, w_init=w, evolve=False, tune=True, **kw)


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
stage(Stage("1.1", "s1_1_valence", "1.0", B11, W11, VISION_CH1, BODY_TASTE,
            lambda exp: berry_world(exp, 6, poison=(4, 5), poison_food=POISON_INNATE), row_extra=poison_metrics,
            generations=150, notes="value cells acting on approach and grasp programmes; 2 of 6 berry types poison"))

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


# What the lesson is about (v27, STAGE_LOG 2026-10-07). Pain alone, on the whole look, only ever raises the aversive
# weights, and every look shares part of the gooseberry look, so suspicion of a poison ends on good food and is
# never taken back; evolution then sets the learning rate to zero. Three things make the lesson land on the poison:
#   centred    the synapse learns from the look minus the slow average of each look input (about 100 ticks): what
#              sets this food apart, not what all berries share (a covariance rule; the average sits in the input
#              neuron);
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
B15 = replace(_B15, projections=_split_cs(_B15.projections, eta=ETA_15, centred=True))
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


NOVEL = (1, 2, 3, 5)  # berry types whose look and meaning are drawn per life; type 0 is always good, 4 always poison
NOVEL_WEIGHT = 1.0    # spawn weight of each novel type
STAPLE_WEIGHT = 0.0   # spawn weight of the ancestral good type (0: no familiar food)
ANC_POISON_WEIGHT = 0.0   # spawn weight of the ancestral poison type (0: none)


def learning_world(exp, reverse: bool = False, springs: float = 0.0, berries: int | None = None):
    """Four novel berry types (two of them poison, drawn per life, with a look drawn per life), one ancestral good
    and one ancestral poison type, no onions. NOVEL_WEIGHT sets how much of the supply is novel: familiar food
    should keep a non-learner alive, novel food should be worth trying."""
    return berry_world(exp, 6, poison=(4,), per_life_pool=NOVEL, per_life_k=2, springs=springs, reverse=reverse,
                       novel_looks=NOVEL, novel_sim=NOVEL_SIM, poison_food=POISON_FOOD, onions=False,
                       weights={0: STAPLE_WEIGHT, 4: ANC_POISON_WEIGHT, **{v: NOVEL_WEIGHT for v in NOVEL}},
                       berries=berries)


_S15 = dict(row_extra=poison_metrics, plastic=True, generations=60, ticks=LIFE_15, eta_max=1.0, dense_plastic=True)
stage(Stage("1.5", "s1_5_association", "1.1", B15, W15, VISION_CH1, BODY_TASTE,
            lambda exp: learning_world(exp, berries=LEARN_BERRIES), **_S15,
            notes="aversive conditioning: pain teaches the look -> aversive synapses what sets the bitten food apart; "
                  "only novel foods, half of them poison; long lives"))
stage(Stage("1.5f", "s1_5_association_fwd", "1.1", B15F, W15, VISION_CH1, BODY_TASTE,
            lambda exp: learning_world(exp, berries=LEARN_BERRIES), **_S15,
            notes="1.5 with only the eye pointing ahead learning (variant)"))


# 1.6 extinction and reversal: the learned weights now relax back toward their inherited values (a fast,
# forgetting component), so an association that stops being renewed fades and a new one can take over. World: the
# novel types swap their meaning halfway through life.
B16 = replace(B15, projections=tuple(replace(p, decay=0.003) if p.modulator in CS_MODS else p for p in B15.projections))
W16 = replace(W15, switch_tick=LIFE_LEARN // 2)
stage(Stage("1.6", "s1_6_reversal", "1.5", B16, W16, VISION_CH1, BODY_TASTE, lambda exp: learning_world(exp, reverse=True),
            row_extra=poison_metrics, plastic=True, generations=150, ticks=LIFE_LEARN,
            notes="reversal learning: learned weights decay toward w0; the novel types swap meaning mid-life"))


# Chapter 2 (older design, to be reviewed) is written for a brain with both teachers: the taste teacher that chapter 1
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
stage(Stage("2.1", "s2_1_tectum", "1.6", B21, W16, VISION_CH2, BODY_TASTE, lambda exp: learning_world(exp, reverse=True),
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


stage(Stage("2.2", "s2_2_pallium_expansion", "2.1", B22, W22, VISION_CH2, BODY_TASTE, xor_world,
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
stage(Stage("2.3", "s2_3_pallium_clustering", "2.2", B23, W22, VISION_23, BODY_TASTE, xor_world,
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
stage(Stage("2.4", "s2_4_basal_ganglia", "2.3", B24, W22, VISION_23, BODY_TASTE, xor_world,
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
stage(Stage("2.5", "s2_5_dopamine_td", "2.4", B25, W22, VISION_23, BODY_TASTE, xor_world,
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


# ------------------------------------------------------------------ running

EVOLUTION_OVERRIDES: dict = {}   # set from the command line (--mutation-prob), applied to every stage
SIBLINGS = 4                     # 256 agents = 64 genomes x 4 siblings (EvolutionConfig.siblings)
CROSSOVER = 1.0                  # every child has two parents (whole neurons from either): in the recovery
                                 # benchmark this repaired a damaged population 6-7% further than asexual
                                 # reproduction in 40 generations, in every seed (STAGE_LOG, v20)


def make_exp(s: Stage, brain: BrainConfig, name: str, generations: int, seed: int) -> ExperimentConfig:
    return ExperimentConfig(name=name, world=s.world, vision=s.vision, body=s.body, brain=brain,
                            evolution=EvolutionConfig(generations=generations, ticks_per_generation=s.ticks,
                                                      eta_max=s.eta_max, **{"siblings": SIBLINGS, "crossover": CROSSOVER, **EVOLUTION_OVERRIDES},
                                                      plastic=s.plastic, seed=seed))


def run_stage(key: str, generations: int | None = None, control: bool = False, init_from: str | None = "auto",
              seed: int = 0, dashboard: bool = True, suffix: str = "", ticks: int | None = None):
    s = STAGES[key]
    if ticks:   # a trial with a longer or shorter life than the stage defines; use a suffix to keep it apart
        s = replace(s, ticks=ticks)
    gens = generations or s.generations
    if control:
        assert s.parent, "stage has no parent, so no control"
        brain, name = STAGES[s.parent].brain, s.name + "_control"
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
        if s.dense_plastic and not control:
            from life.run import make_layout
            lay = make_layout(exp)
            init = init._replace(mask=jnp.where(((np.asarray(lay.rule) > 0) & (np.asarray(lay.allowed) > 0))[None], 1.0, init.mask))
    rs, rules_fn = s.build(exp)
    return run_evolution(exp, rs, s.fitness or default_fitness, rules_for_generation=rules_fn, init_population=init,
                         row_extra=s.row_extra(rs) if s.row_extra else None, dashboard=dashboard)


def stage_of_run(run_dir) -> Stage:
    from pathlib import Path
    name = Path(run_dir).resolve().parent.name
    for s in STAGES.values():
        if name.startswith(s.name):
            return s
    raise KeyError(f"no stage for experiment {name}")


def lesion(run_dir: str, regions: list[str] | None = None, worlds: int = 8, seed: int = 123,
           only: tuple | None = None) -> dict:
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
    a = p.parse_args(argv)
    if a.stage == "list":
        for s in STAGES.values():
            print(f"{s.key:5s} {s.name:32s} parent={s.parent}  {s.notes}")
        return
    if a.mutation_prob is not None:
        EVOLUTION_OVERRIDES["weight_mutation_prob"] = a.mutation_prob
    return run_stage(a.stage, a.generations, a.control, a.init_from, a.seed, suffix=a.suffix, ticks=a.ticks)


if __name__ == "__main__":
    main()
