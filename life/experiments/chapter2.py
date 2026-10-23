"""Chapter 2 (reinforcing): the dopamine critic, its drafts and the stages that were frozen on the way
(2.3, 2.3h, 2.3c, 2.3s on the chain), and hands, the start of the actor stage."""
from __future__ import annotations
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from life.config import (ExperimentConfig, WorldConfig, VisionConfig, BodyConfig, BrainConfig, EvolutionConfig,
                         RegionSpec as R, ProjectionSpec as P, ModulatorSpec as Mod)

from life import actions as A
from life.experiments.base import *  # noqa: F401,F403
from life.experiments.worlds import *  # noqa: F401,F403
from life.experiments.chapter1 import *  # noqa: F401,F403
from life.experiments.chapter1 import _S15


VALUE = dict(sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, group="forebrain/basal_ganglia")

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



# The formula critic (frozen 2026-10-20, off the chain since 2026-10-21; STAGE_LOG v65): the 2.3 design in the harsh world with the reversal, straight
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


def appetite_learning(brain: BrainConfig) -> BrainConfig:
    """The appetitive side learns as the aversive one (user, 2026-10-21; STAGE_LOG v74): look and pallium ->
    `valence_app` taught by `da` (a burst strengthens), the mirror of look and pallium -> `valence_av` taught by
    `da_neg` (a pause strengthens). Learning from pain only was meant until dopamine came (ADR-024)."""
    av = {p.src: p for p in brain.projections if p.dst == "valence_av" and p.modulator == "da_neg"}
    look = replace(av["in"], dst="valence_app", modulator="da")
    return replace(brain, projections=tuple(
        look if (p.src, p.dst, p.src_select) == ("in", "valence_app", look.src_select) else p
        for p in brain.projections) + (replace(av["pallium"], dst="valence_app", modulator="da"),))


def pain_relay(brain: BrainConfig) -> BrainConfig:
    """Pain reaches the habenula through its relay cell `us_pain`, as taste reaches the dopamine cell through
    `us_taste` (user, 2026-10-21), instead of straight from the sense. The strength is set so that a full pain (1,
    `us_pain` tanh(2) = 0.96) drives the habenula as the direct line did."""
    return replace(brain, projections=tuple(p for p in brain.projections if (p.src, p.dst) != ("in", "habenula"))
                   + (fixed("us_pain", "habenula", HAB_PAIN / float(np.tanh(2.0))),))


B24 = dopamine_cells(B23)
B24TD = dopamine_td(B23)
B24D = dopamine_cells(B23, delay=True)
B24S = pain_relay(appetite_learning(B24TD))
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

# The critic on the chain (frozen 2026-10-21, replacing 2.3h; STAGE_LOG v70). The critic page's stage (user, 2026-10-21:
# the cell version replaces the formula on the page, without an intermediate
# stage): 2.3h's design with the critic built from cells (B24TD), straight from the pallium stage; the harsh world,
# fifty calm generations, the aversion learning rates x10 at the start; control the 2.2 brain (the pain teacher).
# 2.4 (from 2.3h) checked first that the cells work: re-evaluated (16 worlds) cells 1178, formula 1252, pain teacher
# 943 (STAGE_LOG v69).
stage(Stage("2.3c", "s2_3c_critic", "2.2", B24TD, replace(W23, **HARSH_REV_WORLD), VISION_CH1, BODY_17,
            lambda exp: pallium_world(exp, reverse=True, one_berry=1.0),
            **{**_S15, "generations": 100}, evolution=EVOLUTION_2EP, control_brain=per_slot_pallium(B22P),
            adapt_init=av_eta_init, schedule=calm_first(HARSH_CALM),
            notes="dopamine critic built from cells (dopamine cell with a resting rate, habenula, ventral striatal value "
                  "cells, fast pallidal and slow striosomal paths: TD) in the harsh world; the reversal at 800 from "
                  "generation HARSH_CALM on; control the 2.2 brain"))

# Candidate to replace 2.3c on the chain and the critic page (user, 2026-10-21: get it right in one stage rather than
# fix it in the next): 2.3c with the appetitive synapses taught by dopamine as the aversive ones are
# (`appetite_learning`) and pain reaching the habenula through `us_pain` (`pain_relay`); B24S. Generation 0 on the
# 2.3c populations neutral (STAGE_LOG v74). The control (the 2.2 brain, same world, seeds and start) is 2.3c's.
stage(Stage("2.3s", "s2_3s_critic", "2.2", B24S, replace(W23, **HARSH_REV_WORLD), VISION_CH1, BODY_17,
            lambda exp: pallium_world(exp, reverse=True, one_berry=1.0),
            **{**_S15, "generations": 100}, evolution=EVOLUTION_2EP, control_brain=per_slot_pallium(B22P),
            adapt_init=av_eta_init, schedule=calm_first(HARSH_CALM),
            notes="2.3c with appetite learning from dopamine (look and pallium -> valence_app under da, the mirror of "
                  "the aversive side) and pain reaching the habenula through us_pain; control the 2.2 brain"))




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


# ------------------------------------------------------------------ hands and the actor (ADR-028)
# One stage (user, 2026-10-23): the critic brain picks, holds and eats (`hands`), and main adds the basal ganglia's
# actor (`actor`); the control is the critic brain with hands. Both meet the hands world at once, from 2.3s.
# Why hands give the actor a job: USE picks with an empty hand at a bush and puts down what is held at an empty cell;
# held poison blocks the next pick (a full hand cannot pick), so putting it down pays, and nothing innate does it.
SWALLOW = 2.5         # ingest's resting drive: something in the mouth is swallowed unless aversion blocks it
PROG_MOTOR_W = 8.0    # a programme the brain releases (ingest; the actor's act) onto its motor neuron. The 2.3s ganglion
                      # drives FORWARD about 4 units above every other action when nothing is close ahead and holds EAT
                      # down (EAT did nothing useful in 2.3s); at 3 (grasp's strength) a released programme lost to
                      # walking on (lab, 2026-10-23): a selected action has to take the motor neurons over
HELD_SCALE = 2.0      # the held item's weight on the pallium relative to the eye's (scripts/probes/pallium_held.py: a
                      # lesson on a poison held at its bush reaches it held in front of nothing at 0.23 with x1, 0.50
                      # with x2, as much as a region of its own; good berries stay at 0.10 of a poison's aversion)


def hands(brain: BrainConfig, swallow: float | None = None) -> BrainConfig:
    """The critic brain (2.3s) for a creature that carries things: `with_hands` (a value cell per sign for the thing in
    hand, an `ingest` programme onto EAT, aversion blocking EAT) with three changes for the critic brain:
      - taste and pain reach the cells about what is seen only (with_hands sent pain to the held cell as well, taste
        not: an asymmetry no stage needed);
      - `ingest` swallows by default (SWALLOW): in 2.3s eating followed the bite at once, and the held cells learn
        per life, so without it nothing would be eaten at generation 0 (x.hands lost all its food at first);
      - the pallium reads the item in hand as well as the eye ahead (held at HELD_SCALE, per slot like the eye), so
        the critic and the actor know what is held; with an empty hand the code is the 2.3s one;
      - `hungry` reaches `ingest` and `no_eat` as it reaches `grasp` and `no_feed` (evolvable, from 0): eating is
        decided at ingest now;
      - ingest drives EAT at PROG_MOTOR_W;
      - aversion is split as appetite is: aversion to what is seen stops picking (`no_feed` -> grasp and USE, as in
        2.3s), aversion to what is held stops swallowing (`no_eat` -> ingest and EAT). with_hands let every aversive
        cell stop eating, so anything disliked in view kept a held berry from being swallowed (generation 0: `no_feed`
        at 0.5 while holding, 4 berries eaten per life);
      - the valence cells read the pallium of the same tick (phase after the pallium, then no_feed, then the
        programmes): in 2.3s they shared its phase and read it a tick late, which did not matter at a bush seen from
        afar, but a held berry is swallowed on its first tick in hand, before a learned aversion arrived (lab,
        2026-10-23: the drive onto the aversive cells was +1 on that tick, their rate 0)."""
    b = with_hands(brain)
    eye = next(p for p in b.projections if p.src == "in" and p.dst == "pallium")
    sq2 = float(np.sqrt(2.0))   # the fan-in doubles: the eye's weights keep their 2.3s size (ProjectionSpec.slot_scale)
    proj = []
    for p in b.projections:
        if p.src == "in" and p.dst == "valence_av" and p.src_select == ("pain",):
            p = replace(p, dst_range=SEEN)
        elif p is eye:
            p = replace(p, slot_scale=sq2)
        elif (p.src, p.dst) == ("valence_av", "no_feed"):
            p = replace(p, src_range=SEEN)
        elif (p.src, p.dst) == ("no_feed", "out"):
            p = replace(p, dst_range=USE_)
        elif (p.src, p.dst) == ("no_feed", "ingest"):
            continue
        proj.append(p)
    proj += [fixed("valence_av", "no_eat", VALENCE_MOTOR_W, src_range=HELD), fixed("no_eat", "ingest", VALENCE_MOTOR_W),
             fixed("no_eat", "out", VALENCE_MOTOR_W, dst_range=EAT_)]
    proj.append(replace(eye, src_select=("held_app*",), slot_scale=HELD_SCALE * sq2))
    proj = [replace(p, w_init=PROG_MOTOR_W) if (p.src, p.dst) == ("ingest", "out") else p for p in proj]
    proj += [P("hungry", "ingest", density=1.0, w_init=0.0), P("hungry", "no_eat", density=1.0, w_init=0.0)]
    pal = next(r.phase for r in b.regions if r.name == "pallium")
    phase = {"valence_app": pal + 1, "valence_av": pal + 1, "no_feed": pal + 2, "no_eat": pal + 2, "grasp": pal + 3,
             "ingest": pal + 3}
    no_feed = next(r for r in b.regions if r.name == "no_feed")
    b = replace(b, regions=b.regions + (replace(no_feed, name="no_eat", size=1),))
    sw = SWALLOW if swallow is None else swallow
    regions = tuple(replace(r, phase=phase.get(r.name, r.phase), **({"bias": sw} if r.name == "ingest" else {}))
                    for r in b.regions)
    return replace(b, projections=tuple(proj), regions=regions, out_phase=max(b.out_phase, pal + 4))


# The actor (BRAIN_EVOLUTION, "Basal ganglia with the actor"; Redgrave, Prescott & Gurney 1999; Frank 2005). One channel
# per hand action, USE and EAT (cell 0 and 1 of every region below); the meaning of an action (pick or put down) comes
# from the state the striatum reads, not from a programme per situation (user, 2026-10-23):
#   striatum_d1 (Go) -| gpi -| act -> out          a burst strengthens state -> D1: the action is released
#   striatum_d2 (NoGo) -| gpe -| gpi               a pause strengthens state -> D2: the action is held back
# gpi and gpe rest at 0.5 (tonic). `act` is the action's brainstem programme as the basal ganglia drive it, shut while
# gpi rests; gpi also holds back the old programmes of the same action (grasp, ingest) a little, compensated in their
# rest, so that the old paths and the actor act on one programme each (principle 8) and NoGo can stop them.
# The striatum reads the pallium (what is ahead and in hand) through synapses learned within a life (from 0), and a
# full hand drives Go for USE a little (ACT_EXPLORE). Credit: an efference copy of the action taken excites both
# cells of its channel through slow synapses (the copy of tick t-1 arrives with the error about it), so the pre x post
# trace marks the state in which that action was taken, also when an old path chose it (Fee 2014). Both cells get it
# equally and their paths are balanced (W_D1 = 0.75 W_EG W_D2 at rest), so the copy itself does not repeat the action.
ACT_ETA, ACT_ELIG = 0.1, 0.5   # as the critic's value cells
W_EFC = 0.5                    # efference copy -> D1, D2 (about 0.46 of the cell's rate)
W_D1, W_D2, W_EG = 1.5, 1.0, 2.0
W_GA, B_ACT = 4.0, 1.5         # gpi -| act; act's rest: shut at gpi 0.5, 0.9 at gpi 0
W_GP = 1.0                     # gpi -| grasp, ingest (offset in their bias at rest)
ACT_EXPLORE = 0.2              # held -> Go (USE), hard-wired, tuned by evolution: with something in hand USE is now and
                               # then released, so putting down happens and can be learned (user, 2026-10-23: a start
                               # strength is fine to leave a dead end at the start of a stage). Without it the action
                               # was never taken in the lab (the 2.3s ganglion walks on; sampling is sharp); at 0.4 and
                               # more everything held was put down at once. Learned from `held` (and `vis+0.near`)
                               # instead, it took the average value of an action over every berry (Go-USE and NoGo-EAT
                               # for good berries too, lab 2026-10-23): what is held is the pallium's to say
DSTR = "forebrain/basal_ganglia/dorsal_striatum"
HAND_ACTS = (4, 6)             # USE, EAT on the motor neurons (actions.py)


def actor(brain: BrainConfig, explore: float = ACT_EXPLORE) -> BrainConfig:
    """`brain` (with hands) with the basal ganglia's actor for USE and EAT. Phases: striatum after the pallium, then
    gpe, gpi, the programmes (act, grasp and ingest move after gpi) and the motor neurons last."""
    rest = float(np.arctanh(0.5))
    pal = next(r.phase for r in brain.regions if r.name == "pallium")
    ph = dict(d=pal + 1, gpe=pal + 2, gpi=pal + 3, prog=pal + 4)
    regions = tuple(replace(r, phase=ph["prog"], bias=r.bias + W_GP * 0.5) if r.name in ("grasp", "ingest") else r
                    for r in brain.regions)
    spn = lambda n: R(n, 2, sign="inh", alpha=1.0, bias=0.0, evolve_bias=False, phase=ph["d"], group=DSTR)
    tonic = lambda n, b, p: R(n, 2, sign="inh", alpha=1.0, bias=b, evolve_bias=False, phase=p, group=PALLIDUM)
    learn = lambda src, dst, mod, **kw: P(src, dst, rule="hebb", modulator=mod, eta_init=ACT_ETA, elig_tau=ACT_ELIG,
                                          abcd=(1.0, 0.0, 0.0, 0.0), density=1.0, evolve=False, elig_delay=True,
                                          **{"w_init": 0.0, **kw})
    o2o = dict(topology="one_to_one", density=1.0, evolve=False)
    b = replace(brain, regions=regions, out_phase=ph["prog"] + 1)
    return extend(b,
                  regions=(spn("striatum_d1"), spn("striatum_d2"),
                           tonic("gpe", rest, ph["gpe"]), tonic("gpi", rest + W_EG * 0.5, ph["gpi"]),
                           R("act", 2, sign="exc", alpha=1.0, bias=B_ACT, evolve_bias=False, phase=ph["prog"],
                             group="valence/programmes")),
                  projections=(learn("pallium", "striatum_d1", "da"), learn("pallium", "striatum_d2", "da_neg"),
                               P("in", "striatum_d1", src_select=("held",), dst_range=(0, 1), density=1.0, w_init=explore,
                                 evolve=False, tune=True),
                               P("in", "striatum_d1", src_select=("efc_USE", "efc_EAT"), w_init=W_EFC, lag=True, **o2o),
                               P("in", "striatum_d2", src_select=("efc_USE", "efc_EAT"), w_init=W_EFC, lag=True, **o2o),
                               P("striatum_d1", "gpi", w_init=W_D1, tune=True, **o2o),
                               P("striatum_d2", "gpe", w_init=W_D2, tune=True, **o2o),
                               P("gpe", "gpi", w_init=W_EG, **o2o),
                               P("gpi", "act", w_init=W_GA, **o2o),
                               fixed("gpi", "grasp", W_GP, src_range=(0, 1), tune=False),
                               fixed("gpi", "ingest", W_GP, src_range=(1, 2), tune=False),
                               P("act", "out", w_init=PROG_MOTOR_W, tune=True, dst_range=HAND_ACTS, **o2o)))


def hands_init(pop, layout):
    """Start of the hands stage (main and control): the evolved synapses from what is held (`held`, `held_app*`) start
    silent. In 2.3s the hand was always empty (eating on the bite), so these inputs were 0 and their synapses drifted
    unselected; with a full hand they fired into the aversive cells and the ganglion (generation 0: the eating gate
    `no_feed` at 0.4-0.6 while holding, no berry swallowed). A new sense starts silent, as remap_genomes does for new
    synapses. Designed synapses (the `no_hold` gate, the pallium's per-slot input) keep their weights.
    Likewise the EAT motor neuron starts neutral (its evolved inputs and bias at 0): in 2.3s EAT did nothing (eating
    came with the bite), so evolution held it down by about 5 units (bias -1.5, ganglion -2.2, sensor reflexes -1.1;
    generation 0: a held berry was hardly ever eaten, the ganglion walking on at +5.4). With hands the action gets
    its meaning; what evolution makes of it starts from neutral, as a new sense does."""
    names = list(layout.in_names)
    rows = [i for i, n in enumerate(names) if n == "held" or n.startswith("held_app")]
    off = layout.offsets[layout.names.index("in")]
    free = np.asarray(layout.evolve_w) > 0
    sel = np.zeros(free.shape, bool)
    sel[[off + r for r in rows]] = True
    eat = layout.offsets[layout.names.index("out")] + A.EAT
    sel[:, eat] = True
    sel &= free
    b = pop.b.at[:, eat].set(0.0) if bool(np.asarray(layout.evolve_b)[eat]) else pop.b
    return pop._replace(w0=jnp.where(jnp.asarray(sel)[None], 0.0, pop.w0), b=b)


B25H = hands(B24S)
B25 = actor(B25H)
BODY_25 = replace(BODY_17, efference=True)
W25 = replace(W23, **HARSH_REV_WORLD, eat_on_pick=False)
stage(Stage("2.5", "s2_5_actor", "2.3s", B25, W25, VISION_CH1, BODY_25,
            lambda exp: pallium_world(exp, reverse=True, one_berry=1.0),
            **{**_S15, "generations": 100}, evolution=EVOLUTION_2EP, control_brain=B25H, adapt_init=hands_init,
            notes="hands (pick, hold, eat; the pallium reads the item in hand) and the basal ganglia's actor: Go and "
                  "NoGo channels for USE and EAT learned from dopamine, credit by an efference copy; control the "
                  "critic brain with hands"))
