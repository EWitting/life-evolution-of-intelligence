"""Chapter 1 on the chain: steering, valence, habituation, association, drives, affect and the pallium
(keys 1.0, 1.1, 1.2h, 1.5, 1.7, 1.8, 2.2; all frozen). Each brain adds to the one before. Side paths and
trials of the same period are in side_paths.py."""
from __future__ import annotations
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from life.config import (ExperimentConfig, WorldConfig, VisionConfig, BodyConfig, BrainConfig, EvolutionConfig,
                         RegionSpec as R, ProjectionSpec as P, ModulatorSpec as Mod)

from life.experiments.base import *  # noqa: F401,F403
from life.experiments.worlds import *  # noqa: F401,F403


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


DUD = 6               # berry-bush type used as the dud

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
                   one_berry: float = 0.0, thorns: bool = False):
    """Four novel berry types (two of them poison, drawn per life, with a look drawn per life), one ancestral good
    and one ancestral poison type, no onions. NOVEL_WEIGHT sets how much of the supply is novel: familiar food
    should keep a non-learner alive, novel food should be worth trying."""
    return berry_world(exp, 6, poison=(4,), per_life_pool=NOVEL, per_life_k=2, springs=springs, reverse=reverse,
                       novel_looks=NOVEL, novel_sim=NOVEL_SIM if novel_sim is None else novel_sim, poison_food=POISON_FOOD, onions=False,
                       poison_pain=POISON_PAIN,
                       weights={0: STAPLE_WEIGHT, 4: ANC_POISON_WEIGHT, **{v: NOVEL_WEIGHT for v in NOVEL}},
                       berries=berries, shared_colour=shared_colour, novel_xor=novel_xor, one_berry=one_berry,
                       thorns=thorns)


_S15 = dict(row_extra=poison_metrics, plastic=True, generations=40, ticks=LIFE_15, eta_max=1.0, dense_plastic=True)
stage(Stage("1.5", "s1_5_association", "1.2h", B15, W15, VISION_CH1, BODY_TASTE,
            lambda exp: learning_world(exp, berries=LEARN_BERRIES), **_S15,
            notes="aversive conditioning: pain teaches the (adapted) look -> aversive synapses; "
                  "only novel foods, half of them poison"))

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

# The world of the affect stage was found in two steps that are side paths now (2.0, 2.1 in side_paths.py), hence the
# names W20, W20_256, W21; 1.8 uses W21 (W18S).
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

W20_256 = replace(W20, height=256, width=256)   # four times the area of chapter 1 (trials x.c*, STAGE_LOG v40)

# Trials (2026-10-13, user's idea, STAGE_LOG v41): lasting states that add up. In stage 1.8 a modulator is released
# at once (one poison bite is a full alarm, one good bite a full stay). Here each is a leaky sum of the events and
# the cell that reads it has a threshold counted in bites, so that in a patch of mixed kinds the animal leaves or
# stays on several bites. Start values from generation 0 (`scripts/probes/sum_affect.py`: neutral for the population
# of x.c70): alarm above 1.5 poison bites, fading over about 33 ticks; stay above 9 good bites (a bush has 6),
# fading over about 100. Evolution moves the thresholds (the biases of `flee` and `settle`) and the fade times.
SUM_ALARM, SUM_FED = (1.5, 0.97), (9.0, 0.99)   # (threshold in bites, fraction left after a tick)

def sum_affect_init(pop, layout):
    """The designed thresholds and fade times (an inherited bias or fade time would be that of release at once)."""
    if "flee" not in layout.names:   # the control (the parent's brain) has no lasting states
        return pop
    b, clear = pop.b, pop.clear
    for cell, mod, (t, d), sens in (("flee", "alarm", SUM_ALARM, 3.0), ("settle", "serotonin", SUM_FED, 2.0)):
        b = b.at[:, layout.region(cell)].set(-sens * t)
        clear = clear.at[:, layout.mod_names.index(mod)].set(d)
    return pop._replace(b=b, clear=clear)


# patches 70% their own type (the world of side path 2.1; why there, in side_paths.py)
W21 = replace(W20_256, patches=320, patch_radius=6, patch_types=True, patch_purity=0.7)

# The affect brain with states that add up (first tried as trial x.affect2, user 2026-10-17: the affect stage itself
# in the world of 2.1, from the drives population, where learning is still used, with the eyes of chapter 1)
B18S = replace(B18,
               modulators=tuple(replace(m, accumulate=1.0) if m.name in ("alarm", "serotonin") else m for m in B18.modulators),
               regions=tuple(replace(r, bias=-3.0 * SUM_ALARM[0], evolve_bias=True) if r.name == "flee" else
                             replace(r, bias=-2.0 * SUM_FED[0], evolve_bias=True) if r.name == "settle" else r
                             for r in B18.regions))

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


def pallium_world(exp, novel_sim: float | None = None, xor: bool = True, reverse: bool = False, one_berry: float = 0.0,
                  thorns: bool = False):
    return learning_world(exp, springs=SPRINGS_17, berries=LEARN_BERRIES, novel_xor=xor, reverse=reverse, one_berry=one_berry,
                          thorns=thorns,
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
