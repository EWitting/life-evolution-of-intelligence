"""Side paths, parked branches, trials and the first designs that the chain replaced: defined so that
their runs can still be read, lesioned and compared, not on the chain (frozen/README.md lists them with
reasons). Names that a chain stage also uses carry a suffix here (_PARKED, _OLD)."""
from __future__ import annotations
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from life.config import (ExperimentConfig, WorldConfig, VisionConfig, BodyConfig, BrainConfig, EvolutionConfig,
                         RegionSpec as R, ProjectionSpec as P, ModulatorSpec as Mod)

from life.experiments.base import *  # noqa: F401,F403
from life.experiments.worlds import *  # noqa: F401,F403
from life.experiments.chapter1 import *  # noqa: F401,F403
from life.experiments.chapter1 import _S15, _B15, _split_cs, _GENE
from life.experiments.chapter2 import *  # noqa: F401,F403


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
W12_PARKED = replace(W11, temperature=True, ambient_temp=0.25, heat_scale=0.15, heat_radius=4, temp_rate=0.03, temp_hunger=1.5,
              spawn_density=0.12, hunger_per_tick=0.019)
stage(Stage("1.2", "s1_2_drives", "1.1", B12, W12_PARKED, VISION_CH1, BODY_12,
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
W13 = replace(W12_PARKED, patches=80, patch_radius=6)   # the same number of objects, on at most half of the map
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

DUD_WEIGHT = 2.0      # spawn weight (the six real types: 1 each)
W14 = replace(W13, spawn_density=round(W13.spawn_density * (7.1 + DUD_WEIGHT) / 7.1, 3))   # the duds come on top


def dud_world(exp):
    return berry_world(exp, 7, poison=(4, 5), springs=0.6, duds=(DUD,), novel_looks=(DUD,), weights={DUD: DUD_WEIGHT},
                       poison_food=POISON_INNATE)


stage(Stage("1.4", "s1_4_habituation", "1.3", B14, W14, VISION_CH1, BODY_12, dud_world,
            row_extra=poison_metrics, generations=150,
            notes="PARKED side branch: habituating appetitive synapses; dud bushes with a per-life look"))



# 1.1l long lives (side path since the rerun of 2026-10-08, in which 1.1 itself has long lives): the 1.1 brain in
# the 1.1 world with lives of 4000 ticks after a 1.1 of 1000 ticks. No new circuit and so no control.
stage(Stage("1.1l", "s1_1_valence_long", "1.1", B11, W11, VISION_CH1, BODY_TASTE,
            lambda exp: berry_world(exp, 6, poison=(4, 5), poison_food=POISON_INNATE), row_extra=poison_metrics,
            generations=60, ticks=4000, notes="SIDE PATH: the 1.1 brain under lives of 4000 ticks (adaptation step, no control)"))



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
W22_OLD = replace(W16, spawn_density=W16.spawn_density * 7 / 6)   # 7 berry types instead of 6


def xor_world(exp):
    return berry_world(exp, 7, poison=(5, 6), per_life_sets=((1, 4), (2, 3)), appearance_mode="xor",
                       reverse=True)


stage(Stage("old2.2", "old_s2_2_pallium_expansion", "old2.1", B22, W22_OLD, VISION_CH2, BODY_TASTE, xor_world,
            row_extra=poison_metrics, plastic=True, generations=200, ticks=LIFE_LEARN,
            notes="sparse random expansion (k-WTA) before US-gated learning; XOR poison rule per life"))

# 2.3 pallium pattern completion and clustering: the input to the pallium becomes plastic (Oja, unsupervised,
# no modulator) and the pallium gets recurrent Hebbian connections, so it forms noise-robust categories within
# life. World: noisy perception of appearance (std 0.3 per feature and tick).
B23_OLD = replace(B22, projections=tuple(
    replace(p, rule="oja", eta_init=0.01, evolve=True) if (p.src == "in" and p.dst == "pallium") else p
    for p in B22.projections) + (P("pallium", "pallium", rule="hebb", eta_init=0.005, decay=0.01,
                                   abcd=(1.0, 0.0, 0.0, 0.0)),))
VISION_23 = replace(VISION_CH2, appearance_noise=0.3)
stage(Stage("old2.3", "old_s2_3_pallium_clustering", "old2.2", B23_OLD, W22_OLD, VISION_23, BODY_TASTE, xor_world,
            row_extra=poison_metrics, plastic=True, generations=200, ticks=LIFE_LEARN,
            notes="unsupervised Oja input + recurrent Hebb in pallium; noisy appearance"))

# 2.4 basal ganglia, fixed: inhibitory striatal channels (one per action) inhibit a tonically active GPi, which
# inhibits the motor neurons one-to-one; an action is released by disinhibition. Striatum is driven by pallium,
# tectum and valence (evolved).
B24_OLD = extend(B23_OLD,
             regions=(R("striatum", 7, sign="inh", group="forebrain/basal_ganglia"), R("gpi", 7, sign="inh", bias=1.0, evolve_bias=False, group="forebrain/basal_ganglia")),
             projections=(*(P(s, "striatum") for s in ("pallium", "tectum", "valence_app", "valence_av", "ganglion_e")),
                          P("striatum", "gpi", topology="one_to_one", w_init=2.0, evolve=False),
                          P("gpi", "out", topology="one_to_one", w_init=2.0, evolve=False)))
stage(Stage("old2.4", "old_s2_4_basal_ganglia", "old2.3", B24_OLD, W22_OLD, VISION_23, BODY_TASTE, xor_world,
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



B25 = extend(B24_OLD,
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
stage(Stage("old2.5", "old_s2_5_dopamine_td", "old2.4", B25, W22_OLD, VISION_23, BODY_TASTE, xor_world,
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



B20S = replace(B20,
               modulators=tuple(replace(m, accumulate=1.0) if m.name in ("alarm", "serotonin") else m for m in B20.modulators),
               regions=tuple(replace(r, bias=-3.0 * SUM_ALARM[0], evolve_bias=True) if r.name == "flee" else
                             replace(r, bias=-2.0 * SUM_FED[0], evolve_bias=True) if r.name == "settle" else r
                             for r in B20.regions))




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

stage(Stage("2.1", "s2_1_tuning", "2.0", B20S, W21, VISION_CH2, BODY_17, world_ch2, **_S2, adapt_init=sum_affect_init,
            notes="tuning for chapter 2: a wider world with patches 70% their own type, lasting states that add up "
                  "over bites"))
stage(Stage("x.direct", "x_2_1_direct", "1.8old", B20S, W21, VISION_CH2, BODY_17, world_ch2, **_S2,
            adapt_init=lambda pop, layout: sum_affect_init(interpolate_eyes(pop, layout, scale=EYES_SCALE), layout),
            notes="TRIAL: stage 2.1 straight from the affect population (camera eyes, patches and lasting states "
                  "that add up at once): learning does not reliably come back"))


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
