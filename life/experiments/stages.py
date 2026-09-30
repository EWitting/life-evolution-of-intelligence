"""The brain-evolution sequence of docs/BRAIN_EVOLUTION.md as runnable stages.

Every stage = a world (OHOL slice + world mechanics), a brain (regions/projections added on top of the previous
stage) and a fitness. Stages form one lineage: by default a stage warm-starts from the newest run of its parent
stage, remapped onto the new layout (new parts start silent, brain.remap_genomes). The optional control runs the
*parent's* brain in the *new* world from the same starting population, so the new module is the only difference.

    python -m life.experiments.stages list
    python -m life.experiments.stages 1.0 --generations 150
    python -m life.experiments.stages 1.1 --generations 100 [--control] [--init-from <run dir> | --init-from none]
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
REGROW_TICKS = 500                              # rule patch: empty bushes regrow by themselves (OHOL: watering)

# chapter 1 senses: a few coarse, short-range "eyes" (early bilaterians had simple photoreceptors/chemosensing)
VISION_CH1 = VisionConfig(columns=5, fov_degrees=120.0, range=5)


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
    notes: str = ""


STAGES: dict[str, Stage] = {}


def stage(s: Stage) -> Stage:
    STAGES[s.key] = s
    return s


def default_fitness(stats):
    """Energy intake is the proxy for reproductive success; a small bonus for staying alive."""
    return stats["eaten"] + 0.01 * stats["alive_ticks"]


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
                appearance_mode: str = "lookalike"):
    """Several berry-bush types (the OHOL gooseberry and colour look-alikes; all behave like the gooseberry:
    6 berries, then empty, regrowing after REGROW_TICKS), optional wild onions and hot springs.
    poison: types whose berries always drain food and hurt (an inheritable fact). per_life_pool/per_life_k:
    per world, k types drawn from the pool are poison as well (only lifetime learning can know which).
    reverse: with a pool of two and k=1, the poison swaps to the other type at WorldConfig.switch_tick.
    per_life_sets: explicit alternatives instead of pool/k, e.g. ((1, 4), (2, 3)) for the XOR world.
    appearance_mode 'xor': types 1-4 look like the original plus (+-d1 +-d2)/sqrt2 (two binary features), types
    5 and 6 plus +-d3 (see lookalike_appearance)."""
    data = ohol.load()
    ids = [BUSH, BERRY, EMPTY_BUSH] + ([ONION_PLANT, ONION] if onions else []) + ([HOT_SPRING] if springs else [])
    sets = [(variant(v), COLOURS[v - 1] + " ") for v in range(1, n_types)]
    rs = ohol.slice_ruleset(data, ids, ticks_per_second=exp.world.ticks_per_ohol_second,
                            max_decay_ticks=exp.world.max_decay_ticks, clone_sets=sets,
                            extra_decays={EMPTY_BUSH: (BUSH, REGROW_TICKS)})
    spawn = np.zeros(rs.size, np.float32)
    for v in range(n_types):
        spawn[rs.local(variant(v)[BUSH])] = 1.0
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

    if not per_life_pool and not per_life_sets:
        rules = arrays(tuple(poison))
        return rs, (lambda gen, key: rules)
    import itertools
    alts = per_life_sets or tuple(itertools.combinations(per_life_pool, per_life_k))
    combos = [tuple(poison) + tuple(c) for c in alts]
    variants = [arrays(c) for c in combos]
    stacked = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *variants)
    E = exp.evolution.episodes

    def rules_fn(gen, key):
        pick = jax.random.randint(key, (E,), 0, len(variants))
        if reverse:   # [E, 2 phases, ...]
            assert len(variants) == 2
            idx = jnp.stack([pick, 1 - pick], axis=1)
            return jax.tree_util.tree_map(lambda a: a[idx], stacked)
        return jax.tree_util.tree_map(lambda a: a[pick], stacked)
    return rs, rules_fn


LOOKALIKE_SIMILARITY = 0.8   # cosine similarity of a colour variant's appearance to the original object


def lookalike_appearance(rs, n_types: int, k: int, sim: float = LOOKALIKE_SIMILARITY, mode: str = "lookalike") -> dict:
    """Colour variants look *like* the original: appearance = sim * original + sqrt(1 - sim^2) * colour, where the
    colour vector (one per variant, orthogonal to the original) is shared by the variant's bush, berry and empty
    bush. Hash-based appearances (ADR-006) would make every variant an unrelated object."""
    from life.ruleset import appearance_for
    out = {}
    for orig in (BUSH, BERRY, EMPTY_BUSH):
        base = appearance_for(orig, k)
        # orthonormal basis of the complement of base; colours = +-basis directions (maximally spread)
        q, _ = np.linalg.qr(np.column_stack([base, np.eye(k)]))
        comp = q[:, 1:k].T
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


def fitness_pain(stats):
    return stats["eaten"] - stats["pain"] + 0.01 * stats["alive_ticks"]


# ------------------------------------------------------------------ brains (each stage adds to the previous)

# 1.0 steering: a ganglion of excitatory and inhibitory interneurons between sensors and motor neurons,
# plus direct sensor->motor reflex arcs. All weights evolved, no plasticity.
B10 = BrainConfig(
    regions=(R("ganglion_e", 16, sign="exc", group="ganglion"), R("ganglion_i", 8, sign="inh", group="ganglion")),
    projections=(P("in", "ganglion_e"), P("in", "ganglion_i"),
                 P("ganglion_e", "ganglion_e"), P("ganglion_e", "ganglion_i"), P("ganglion_i", "ganglion_e"),
                 P("ganglion_e", "out"), P("ganglion_i", "out"), P("in", "out")))

W10 = WorldConfig(height=32, width=32, num_agents=64, spawn_density=0.08, max_decay_ticks=1000)

stage(Stage("1.0", "s1_0_steering", None, B10, W10, VISION_CH1, BodyConfig(), lambda exp: berry_world(exp, 4),
            generations=200, notes="baseline: evolved reflexive steering to 4 kinds of berry bushes and onions"))


def extend(base: BrainConfig, regions=(), projections=(), modulators=(), **kw) -> BrainConfig:
    """New brain = old brain + new regions (inserted before 'out' in the given order) + projections."""
    return replace(base, regions=base.regions + tuple(regions), projections=base.projections + tuple(projections),
                   modulators=base.modulators + tuple(modulators), **kw)


def fixed(src, dst, w, **kw):
    """A hard-wired projection (every synapse present, weight w, never mutated): innate structure."""
    return P(src, dst, density=1.0, w_init=w, evolve=False, **kw)


# Action indices on the motor region 'out' (actions.py): NOOP 0, FORWARD 1, TURN_LEFT 2, TURN_RIGHT 3, USE 4, EAT 5
FWD, TURNS, FEED = (1, 2), (2, 4), (4, 6)

# 1.1 valence: cell types with a fixed meaning. Appetitive and aversive neurons receive innate (hard-wired) input
# from the unconditioned senses (taste -> appetitive, pain -> aversive) and evolved input from vision and the held
# item. Their motor meaning is innate as in C. elegans (aversive interneurons drive turns/reversals and suppress
# feeding, appetitive ones drive forward movement and feeding): valence_av -> TURN_LEFT/RIGHT and, through an
# inhibitory 'no_feed' pair, -| USE/EAT; valence_app -> FORWARD, USE, EAT. Because their meaning is fixed, later
# stages (learning in 1.5, drives in 1.2) can target them.
VAL = ("valence_app", "valence_av")
# Identity features (the object's appearance, seen or held: the analogue of an odour) are kept apart from generic
# features ('something is there', how near, agent, wall), because only identity can become a conditioned
# stimulus later (1.5); conditioning 'something is there' would make every object aversive.
CS_SEL = ("vis*.app*", "held_app*")
GENERIC_SEL = ("vis*.hit", "vis*.near", "vis*.agent", "vis*.wall", "held")
VALENCE_MOTOR_W = 3.0
B11 = extend(B10,
             regions=(R("valence_app", 3, sign="exc", group="valence"), R("valence_av", 3, sign="exc", group="valence"),
                      R("no_feed", 2, sign="inh", bias=0.0, evolve_bias=False, group="valence")),
             projections=(fixed("in", "valence_app", 3.0, src_select=("taste",)),
                          fixed("in", "valence_av", 3.0, src_select=("pain",)),
                          *(P("in", v, src_select=CS_SEL) for v in VAL),
                          *(P("in", v, src_select=GENERIC_SEL) for v in VAL),
                          fixed("valence_av", "out", VALENCE_MOTOR_W, dst_range=TURNS),
                          fixed("valence_av", "no_feed", VALENCE_MOTOR_W),
                          fixed("no_feed", "out", VALENCE_MOTOR_W, dst_range=FEED),
                          fixed("valence_app", "out", VALENCE_MOTOR_W, dst_range=FWD),
                          fixed("valence_app", "out", VALENCE_MOTOR_W, dst_range=FEED),
                          *(P(v, t) for v in VAL for t in ("ganglion_e", "ganglion_i"))))
BODY_TASTE = BodyConfig(taste=True)

# same number of good bushes as 1.0 plus 2 poisonous types on top (density 0.08 * 6/4)
W11 = replace(W10, spawn_density=0.12)
stage(Stage("1.1", "s1_1_valence", "1.0", B11, W11, VISION_CH1, BODY_TASTE,
            lambda exp: berry_world(exp, 6, poison=(4, 5)), fitness=fitness_pain, row_extra=poison_metrics,
            generations=150, notes="innate good/bad cell types with fixed motor meaning; 2 of 6 berry types poison"))


# 1.2 drives (hypothalamus): named drive neurons read the body. 'hunger' is broadcast as a neuropeptide-like
# modulator: receptors on valence_app raise appetite with need. 'cold' gates an innate thermotaxis circuit
# (run-and-tumble on temperature changes, as C. elegans AFD -> AIY/AIZ): when cold and getting warmer, keep going
# forward; when cold and getting colder, turn. No knowledge of what a heat source looks like is needed.
#   hunger    = max(0, tanh(2 - 3 * food))                     fires below ~2/3 full
#   cold      = max(0, tanh(3.3 - 7 * temperature))            fires below ~0.47
#   warm_run  = max(0, tanh(-2 + 2.5 cold + 3 temp_change))    -> FORWARD
#   warm_turn = max(0, tanh(-2 + 2.5 cold - 3 temp_change))    -> TURN_LEFT/RIGHT
# World: a cold world (OHOL-style temperature) with hot springs; being cold multiplies hunger.
THERMO = dict(sign="exc", alpha=1.0, bias=-2.0, evolve_bias=False, group="thermotaxis")
B12 = extend(B11,
             regions=(R("hunger", 1, sign="exc", alpha=0.5, bias=2.0, evolve_bias=False, group="hypothalamus"),
                      R("cold", 1, sign="exc", alpha=0.5, bias=3.3, evolve_bias=False, group="hypothalamus"),
                      R("warm_run", 1, **THERMO), R("warm_turn", 1, **THERMO)),
             projections=(fixed("in", "hunger", -3.0, src_select=("food",)),
                          fixed("in", "cold", -7.0, src_select=("temperature",)),
                          fixed("cold", "warm_run", 2.5), fixed("cold", "warm_turn", 2.5),
                          fixed("in", "warm_run", 3.0, src_select=("temp_change",)),
                          fixed("in", "warm_turn", -3.0, src_select=("temp_change",)),
                          fixed("warm_run", "out", 3.0, dst_range=FWD), fixed("warm_turn", "out", 3.0, dst_range=TURNS),
                          P("hunger", "ganglion_e"), P("cold", "ganglion_e")),
             modulators=(Mod("hunger", pos="hunger"), Mod("cold", pos="cold")))
B12 = replace(B12, regions=tuple(replace(r, receptors=r.receptors + (("hunger", "gain", 1.0),)) if r.name == "valence_app"
                                 else r for r in B12.regions))
BODY_12 = BodyConfig(taste=True, temperature=True, temp_change=True)
# calibrated 2026-09-30 so the inherited 1.1 behaviour keeps its food economy (lifetime ~440, food ~49): more
# bushes compensate the cost of the cold (an increment, not a cliff, ADR-012)
W12 = replace(W11, temperature=True, ambient_temp=0.25, heat_scale=0.15, heat_radius=4, temp_rate=0.1, temp_hunger=1.0,
              spawn_density=0.16)
stage(Stage("1.2", "s1_2_drives", "1.1", B12, W12, VISION_CH1, BODY_12,
            lambda exp: berry_world(exp, 6, poison=(4, 5), springs=0.6), fitness=fitness_pain, row_extra=poison_metrics,
            generations=150, notes="hunger (broadcast, receptors on appetite) and cold-gated thermotaxis; cold world with hot springs"))

# 1.3 affect: two slow, antagonistic neuromodulatory states, as in C. elegans (Flavell et al. 2013):
#   raphe (serotonin, '5ht'): driven by taste (food found), slow (alpha 0.03); 5ht receptors on 'dwell', which
#   drives turning -> local search after food (dwelling).
#   pdf (roaming neuropeptide): driven by hunger, slow; pdf receptors on 'roam', which drives FORWARD -> long
#   straight runs when food has not been found for a while (roaming).
# The two nuclei may inhibit each other only through evolved connections. World: food in a few dense patches.
B13 = extend(B12,
             regions=(R("raphe", 2, sign="exc", alpha=0.03, group="affect"),
                      R("pdf", 2, sign="exc", alpha=0.03, group="affect"),
                      R("dwell", 2, sign="exc", bias=0.0, evolve_bias=False, receptors=(("5ht", "bias", 2.0),), group="affect"),
                      R("roam", 2, sign="exc", bias=0.0, evolve_bias=False, receptors=(("pdf", "bias", 2.0),), group="affect")),
             projections=(fixed("in", "raphe", 2.0, src_select=("taste",)), fixed("hunger", "pdf", 2.0),
                          P("valence_app", "raphe"), P("valence_av", "pdf"), P("raphe", "pdf"), P("pdf", "raphe"),
                          fixed("dwell", "out", 1.5, dst_range=TURNS), fixed("roam", "out", 1.5, dst_range=FWD)),
             modulators=(Mod("5ht", pos="raphe"), Mod("pdf", pos="pdf")))
W13 = replace(W12, patches=10, patch_radius=5, spawn_density=0.18)   # calibrated: lifetime ~424, food ~54
stage(Stage("1.3", "s1_3_affect", "1.2", B13, W13, VISION_CH1, BODY_12,
            lambda exp: berry_world(exp, 6, poison=(4, 5), springs=0.6), fitness=fitness_pain, row_extra=poison_metrics,
            generations=150, notes="serotonin (dwell) and PDF (roam) broadcast states; patchy food"))

# 1.4 habituation: short-term depression on the sensory -> appetitive synapses, so a food eaten repeatedly
# loses appeal (sensory-specific satiety). World: OHOL's 'yum' variety bonus (fresh foods worth 1.5x, repeated
# ones 0.5x), so switching between food types pays.
B14 = replace(B13, projections=tuple(
    replace(p, depression=(0.1, 150.0)) if (p.src == "in" and p.dst == "valence_app" and p.src_select == CS_SEL)
    else p for p in B13.projections))
W14 = replace(W13, variety_bonus=0.3, variety_tau=150.0)
stage(Stage("1.4", "s1_4_habituation", "1.3", B14, W14, VISION_CH1, BODY_12,
            lambda exp: berry_world(exp, 6, poison=(4, 5), springs=0.6), fitness=fitness_pain, row_extra=poison_metrics,
            generations=150, notes="habituating appetitive synapses; OHOL variety bonus"))

# 1.5 associative learning: which of two berry types is poison is decided per life (on top of the two that
# are always poison). Dedicated US neurons (hard-wired from taste and pain) write the 'us' modulator; the
# sensory -> valence synapses become plastic, gated by 'us', with an eligibility trace that bridges the delay
# between eating and sickness (Aplysia-style CS-trace rule: dW = eta * us * trace(pre)).
B15 = extend(B14,
             regions=(R("us_taste", 1, sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, group="us"),
                      R("us_pain", 1, sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, group="us")),
             projections=(P("in", "us_taste", src_select=("taste",), density=1.0, w_init=2.0, evolve=False),
                          P("in", "us_pain", src_select=("pain",), density=1.0, w_init=2.0, evolve=False)),
             modulators=(Mod("us", pos="us_taste", neg="us_pain", scale=1.0),))


CS_ETA = 0.05  # CS learning rate (probe 2026-09-30: 0.02 too slow; >= 0.1 over-generalises aversion); evolvable


def _split_cs(projections):
    """in[identity features] -> valence_* becomes plastic and US-gated (CS-trace rule, sign by target)."""
    out = []
    for p in projections:
        if p.src == "in" and p.src_select == CS_SEL and p.dst in VAL:
            out.append(replace(p, rule="hebb", modulator="us", eta_init=CS_ETA, elig_tau=0.8,
                               abcd=(0.0, 1.0 if p.dst == "valence_app" else -1.0, 0.0, 0.0)))
        else:
            out.append(p)
    return tuple(out)


B15 = replace(B15, projections=_split_cs(B15.projections))
W15 = replace(W14, sickness_delay=2, pain_decay=0.3, spawn_density=0.22)   # extra bushes for the extra poison type
stage(Stage("1.5", "s1_5_association", "1.4", B15, W15, VISION_CH1, BODY_12,
            lambda exp: berry_world(exp, 6, poison=(4, 5), per_life_pool=(2, 3), per_life_k=1, springs=0.6),
            fitness=fitness_pain, row_extra=poison_metrics, plastic=True, generations=200,
            notes="classical conditioning: US-gated plasticity with eligibility traces; poison identity per life"))


# 1.6 extinction and reversal: the learned (US-gated) weights now relax back toward their inherited values
# (a fast, forgetting component), so an association that stops being renewed fades and a new one can take
# over. World: the per-life poison swaps to the other look-alike halfway through life.
B16 = replace(B15, projections=tuple(replace(p, decay=0.003) if p.modulator == "us" else p for p in B15.projections))
W16 = replace(W15, switch_tick=500)
stage(Stage("1.6", "s1_6_reversal", "1.5", B16, W16, VISION_CH1, BODY_12,
            lambda exp: berry_world(exp, 6, poison=(4, 5), per_life_pool=(2, 3), per_life_k=1, springs=0.6, reverse=True),
            fitness=fitness_pain, row_extra=poison_metrics, plastic=True, generations=200,
            notes="reversal learning: learned weights decay toward w0; poison identity swaps mid-life"))


# ================================================================== chapter 2: reinforcing (early vertebrates)

# camera eyes: 9 columns over 120 degrees, range 6. Columns at -60, -30, 0, 30, 60 keep their names (and so their
# inherited weights); the new in-between columns start silent.
VISION_CH2 = VisionConfig(columns=9, fov_degrees=120.0, range=6)

# 2.1 optic tectum: a retinotopic map (2 neurons per vision column) with a pool of inhibitory interneurons for
# competition between targets, projecting to the motor neurons (orienting) and the ganglion.
B21 = extend(B16,
             regions=(R("tectum", 18, sign="exc", group="midbrain/tectum"), R("tectum_i", 3, sign="inh", group="midbrain/tectum")),
             projections=(P("in", "tectum", src_select=("vis*",), topology="topographic", groups=9, density=1.0),
                          P("tectum", "tectum_i", density=1.0), P("tectum_i", "tectum", density=1.0),
                          P("tectum", "out", density=1.0), P("tectum", "ganglion_e")))
stage(Stage("2.1", "s2_1_tectum", "1.6", B21, W16, VISION_CH2, BODY_12,
            lambda exp: berry_world(exp, 6, poison=(4, 5), per_life_pool=(2, 3), per_life_k=1, springs=0.6, reverse=True),
            fitness=fitness_pain, row_extra=poison_metrics, plastic=True, generations=200,
            notes="retinotopic target selection with lateral inhibition; camera eyes (9 columns)"))


def cs_plastic(src, dst, sel=None, **kw):
    """A US-gated CS-trace projection onto a valence population (the 1.5 rule), sign by target."""
    b = 1.0 if dst == "valence_app" else -1.0
    extra = {"src_select": sel} if sel else {}
    return P(src, dst, rule="hebb", modulator="us", eta_init=CS_ETA, elig_tau=0.8, decay=0.003,
             abcd=(0.0, b, 0.0, 0.0), **extra, **kw)


# 2.2 pallium as an expansion layer: a large population with fixed, sparse, random input from vision and the held
# item and k-winners-take-all inhibition (piriform cortex / mushroom-body style pattern separation). The
# US-gated learning of 1.5 now also runs from the pallium onto the valence neurons. World: the XOR world, where
# which pair of look-alikes is poison ({++, --} or {+-, -+} of two appearance features) is decided per life.
B22 = extend(B21,
             regions=(R("pallium", 48, sign="exc", kwta=6, group="forebrain/pallium"),),
             projections=(P("in", "pallium", src_select=("vis*", "held*"), density=0.15, evolve=False),
                          cs_plastic("pallium", "valence_app"), cs_plastic("pallium", "valence_av")))
W22 = replace(W16, spawn_density=W16.spawn_density * 7 / 6)   # 7 berry types instead of 6


def xor_world(exp):
    return berry_world(exp, 7, poison=(5, 6), per_life_sets=((1, 4), (2, 3)), appearance_mode="xor",
                       springs=0.6, reverse=True)


stage(Stage("2.2", "s2_2_pallium_expansion", "2.1", B22, W22, VISION_CH2, BODY_12, xor_world,
            fitness=fitness_pain, row_extra=poison_metrics, plastic=True, generations=200,
            notes="sparse random expansion (k-WTA) before US-gated learning; XOR poison rule per life"))

# 2.3 pallium pattern completion and clustering: the input to the pallium becomes plastic (Oja, unsupervised,
# no modulator) and the pallium gets recurrent Hebbian connections, so it forms noise-robust categories within
# life. World: noisy perception of appearance (std 0.3 per feature and tick).
B23 = replace(B22, projections=tuple(
    replace(p, rule="oja", eta_init=0.01, evolve=True) if (p.src == "in" and p.dst == "pallium") else p
    for p in B22.projections) + (P("pallium", "pallium", rule="hebb", eta_init=0.005, decay=0.01,
                                   abcd=(1.0, 0.0, 0.0, 0.0)),))
VISION_23 = replace(VISION_CH2, appearance_noise=0.3)
stage(Stage("2.3", "s2_3_pallium_clustering", "2.2", B23, W22, VISION_23, BODY_12, xor_world,
            fitness=fitness_pain, row_extra=poison_metrics, plastic=True, generations=200,
            notes="unsupervised Oja input + recurrent Hebb in pallium; noisy appearance"))

# 2.4 basal ganglia, fixed: inhibitory striatal channels (one per action) inhibit a tonically active GPi, which
# inhibits the motor neurons one-to-one; an action is released by disinhibition. Striatum is driven by pallium,
# tectum, valence and hypothalamus (evolved).
B24 = extend(B23,
             regions=(R("striatum", 7, sign="inh", group="forebrain/basal_ganglia"), R("gpi", 7, sign="inh", bias=1.0, evolve_bias=False, group="forebrain/basal_ganglia")),
             projections=(*(P(s, "striatum") for s in ("pallium", "tectum", "valence_app", "valence_av",
                                                        "hunger", "cold", "ganglion_e")),
                          P("striatum", "gpi", topology="one_to_one", w_init=2.0, evolve=False),
                          P("gpi", "out", topology="one_to_one", w_init=2.0, evolve=False)))
stage(Stage("2.4", "s2_4_basal_ganglia", "2.3", B24, W22, VISION_23, BODY_12, xor_world,
            fitness=fitness_pain, row_extra=poison_metrics, plastic=True, generations=200,
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
B25 = replace(B25, projections=tuple(replace(p, modulator="da") if p.modulator == "us" else p for p in B25.projections))
stage(Stage("2.5", "s2_5_dopamine_td", "2.4", B25, W22, VISION_23, BODY_12, xor_world,
            fitness=fitness_pain, row_extra=poison_metrics, plastic=True, generations=200,
            notes="TD critic (opponent value populations) and dopamine teaching instead of the raw US"))


# x.td (side experiment, not part of the lineage): does error-driven TD learning fix the over-generalisation of
# 1.5/1.6 conditioning? The 1.6 brain plus only the TD critic of 2.5 (value populations from identity features,
# dopamine replacing the raw US as teacher), in the 1.6 world; control = the 1.6 brain.
BXTD = extend(B16,
              regions=(R("value_app", 3, **VALUE), R("value_av", 3, **VALUE),
                       R("value_app_prev", 3, **VALUE), R("value_av_prev", 3, **VALUE)),
              projections=(td_plastic("in", "value_app", CS_SEL), td_plastic("in", "value_av", CS_SEL),
                           P("value_app", "value_app_prev", topology="one_to_one", w_init=1.0, evolve=False),
                           P("value_av", "value_av_prev", topology="one_to_one", w_init=1.0, evolve=False),
                           fixed("value_app", "valence_app", 1.5), fixed("value_av", "valence_av", 1.5)),
              modulators=(Mod("da", terms=(("us_taste", 1.0), ("us_pain", -1.0), ("value_app", GAMMA),
                                           ("value_av", -GAMMA), ("value_app_prev", -1.0), ("value_av_prev", 1.0))),))
BXTD = replace(BXTD, projections=tuple(replace(p, modulator="da") if p.modulator == "us" else p for p in BXTD.projections))
stage(Stage("x.td", "sx_td_test", "1.6", BXTD, W16, VISION_CH1, BODY_12, STAGES["1.6"].build,
            fitness=fitness_pain, row_extra=poison_metrics, plastic=True, generations=200,
            notes="side test: 1.6 + TD critic, dopamine teaches instead of the raw US"))


# ------------------------------------------------------------------ running

EVOLUTION_OVERRIDES: dict = {}   # set from the command line (--mutation-prob), applied to every stage


def make_exp(s: Stage, brain: BrainConfig, name: str, generations: int, seed: int) -> ExperimentConfig:
    return ExperimentConfig(name=name, world=s.world, vision=s.vision, body=s.body, brain=brain,
                            evolution=EvolutionConfig(generations=generations, ticks_per_generation=s.ticks,
                                                      eta_max=s.eta_max, **EVOLUTION_OVERRIDES,
                                                      plastic=s.plastic, seed=seed))


def run_stage(key: str, generations: int | None = None, control: bool = False, init_from: str | None = "auto",
              seed: int = 0, dashboard: bool = True, suffix: str = ""):
    s = STAGES[key]
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


def lesion(run_dir: str, regions: list[str] | None = None, worlds: int = 8, seed: int = 123) -> dict:
    """Lesion study: the run's final population in its own world, intact and with each region silenced (all
    outgoing synapses removed). Averages over `worlds` independent worlds. Returns {label: stats means}."""
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
    fit_fn = s.fitness or default_fitness
    regions = regions or [n for n in layout.names if n not in ("in", "out")]
    keys = jax.random.split(jax.random.PRNGKey(seed), worlds)
    out = {}
    labels = ["intact"] + (["no_plasticity"] if float(pop.eta.max()) > 0 else []) + regions
    for label in labels:
        g = pop
        if label == "no_plasticity":
            g = pop._replace(eta=jnp.zeros_like(pop.eta))
        elif label != "intact":
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
            st, _ = sim(rules, g, ks)
            fit = fit_fn(st)
            row = dict(fitness=float(fit.mean()), eaten=float(st["eaten"].mean()), pain=float(st["pain"].mean()),
                       alive=float(st["alive_ticks"].mean()), temp=float(st["temp_mean"].mean()))
            row.update({k2: float(v) for k2, v in extra(st, fit).items()})
            acc.append(row)
        out[label] = {k2: float(np.mean([r[k2] for r in acc])) for k2 in acc[0]}
    return out


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
        q.add_argument("--seeds", default="1")
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
        # summary <key> [--window 25]: last-window means of main and control over every seed (run name variants)
        from life.compare import load, window_means
        q = argparse.ArgumentParser(prog="stages summary")
        q.add_argument("key")
        q.add_argument("--window", type=int, default=25)
        b = q.parse_args(argv[1:])
        s = STAGES[b.key]
        for label, base in (("main", s.name), ("control", s.name + "_control")):
            dirs = []
            for d in sorted(RUNS_DIR.iterdir()):
                rest = d.name[len(base):]
                if d.name.startswith(base) and (rest == "" or rest.startswith("_seed")):
                    runs = sorted(r for r in d.iterdir() if (r / "fitness.csv").exists())
                    if runs:
                        dirs.append(runs[-1])
            if not dirs:
                continue
            lasts = [window_means(load(r), b.window)[1] for r in dirs]
            keys = [k for k in ("fit_mean", "eaten", "pain", "alive_ticks", "temp_mean", "poison_frac") if k in lasts[0]]
            vals = {k: [x[k] for x in lasts] for k in keys}
            print(f"{label:8s} n={len(dirs)}  " + "  ".join(
                f"{k} {np.mean(v):.3g} [{' '.join(f'{x:.3g}' for x in v)}]" for k, v in vals.items()))
        return
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
    a = p.parse_args(argv)
    if a.stage == "list":
        for s in STAGES.values():
            print(f"{s.key:5s} {s.name:32s} parent={s.parent}  {s.notes}")
        return
    if a.mutation_prob is not None:
        EVOLUTION_OVERRIDES["weight_mutation_prob"] = a.mutation_prob
    return run_stage(a.stage, a.generations, a.control, a.init_from, a.seed, suffix=a.suffix)


if __name__ == "__main__":
    main()
