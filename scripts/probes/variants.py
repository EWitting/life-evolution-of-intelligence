"""Brain variants for the learning probes (gen0.py, assay.py): small changes to a stage's brain, combined with '+'.

    base          the stage's brain as defined
    eta<x>        both conditioning learning rates start at x (e.g. eta0.2)
    decay<x>      learned weights relax toward the inherited ones by this fraction per tick (e.g. decay0.01)
    centred       the learned look synapses learn from the look minus the slow average of each look input
                  (ProjectionSpec.centred, BrainConfig.in_trace_tau 0.99); centred<tau> sets the trace decay
    ahead         only the eye pointing straight ahead learns (what is at the mouth); the side columns keep their
                  inherited look synapses. As defined, every column learns, and at a bite the side columns mostly
                  see other bushes
    pulse         the pain teacher fires on the tick the pain starts and not while it fades (us_pain with a
                  threshold: bias -1, input weight 3). As defined it is still at 0.54 and 0.18 on the next two ticks,
                  when the animal has turned and is looking at another bush
    short         the sickness teacher gets the same short eligibility trace as the taste teacher (0.5)
    td            the critic as a formula (old2.5 on the 2.2 brain): value cells from the pallium, dopamine = TD error
                  teaches every pain-gated synapse (TD_GAMMA, TD_SCALE, TD_ETA, TD_ELIG, TD_TASTE); tdp: only those
                  from the pallium, the look synapses stay pain-gated
    pslot         the pallium's input wiring belongs to the population slot (the same draw in every generation, not
                  inherited); what reads the pallium starts at 0, not evolved. (plife until 2026-10-19: a new draw for
                  every creature at every birth; logs of then)
    pnt           the pallium also reads the no_touch cell (nothing at the mouth), one phase later; put it before
                  td / td2 in a combination (their value cells go one phase after the pallium)
    noadapt       no input adaptation (BrainConfig.in_adapt off: the looks are not high-passed)
    pphase        control for pnt: the pallium one phase later, without the no_touch input
    pk<k>, pn<n>  pallium k-WTA winners, pallium size (e.g. pk4+pn512)
    pnear         the pallium also reads the distance of the object ahead (vis+0.near), so it codes look x distance;
                  combine with the others, e.g. td+pnear
    td2           one error per valence: taste teaches appetitive value, pain aversive value and pallium -> aversive
    safety        a 'safety' cell (tastes good although the aversive cells expected bad one tick earlier) is
                  subtracted from the pain teacher, so a safe meal undoes suspicion of what was just eaten (the
                  v19-v20 design); safety<k> sets its weight in the teacher (default 1)
"""
from dataclasses import replace
from life.experiments import stages as S

# the formula critic of `td` (constants of stages, so gen0.py can set them: TD_GAMMA=0.8)
S.TD_GAMMA = 0.9      # discount per tick
S.TD_SCALE = 1.0      # weight of value (mean rate of a value population) against taste and pain in the error
S.TD_ETA = 0.1        # starting learning rate of pallium -> value cells
S.TD_ELIG = 0.5       # eligibility decay of the value synapses (that of the pain-taught synapses, CS_ELIG)
S.TD_TASTE = 1.0      # weight of taste in the error
S.TD_VALENCE = 1.5    # td, td2: weight of value cells -> valence cells (incentive salience, fixed)
S.TD_DELAY = 1.0      # td, td2: 1 = the value cells read this tick's pallium and the dopamine-taught traces take the pallium
                      # of a tick earlier (ProjectionSpec.elig_delay), so the error teaches the moment it is about


def td_critic(brain, looks: bool = True):
    """The critic as a formula (old2.5 on the 2.2 brain, no memory): opponent value cells read the pallium,
    starting silent (at birth the error is taste - pain), one-tick copies give
        da = us_taste - us_pain + k (gamma (V_app - V_av) - (V_app' - V_av'))     (k = TD_SCALE, ' = a tick earlier)
    and every pain-gated synapse is taught by -da instead (same trace, modulator da_neg): a good meal now unteaches aversion, and a
    predicted outcome teaches nothing. looks=False (variant tdp): the look synapses stay pain-gated, only those
    from the pallium switch to dopamine."""
    g, k, dl = S.TD_GAMMA, S.TD_SCALE, bool(S.TD_DELAY)
    terms = (("us_taste", S.TD_TASTE), ("us_pain", -1.0), ("value_app", g * k), ("value_av", -g * k),
             ("value_app_prev", -k), ("value_av_prev", k))
    plastic = lambda dst, b: S.P("pallium", dst, rule="hebb", modulator="da", eta_init=S.TD_ETA, elig_tau=S.TD_ELIG,
                                 abcd=(0.0, b, 0.0, 0.0), density=1.0, w_init=0.0, elig_delay=dl)
    pal = next(r.phase for r in brain.regions if r.name == "pallium")
    value = lambda n: replace(S.R(n, 3, **S.VALUE), phase=pal + 1 if dl and not n.endswith("_prev") else 0)
    brain = S.extend(brain,
                     regions=(S.R("us_taste", 1, sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, group="us"),
                              *(value(n) for n in ("value_app", "value_av", "value_app_prev", "value_av_prev"))),
                     projections=(S.P("in", "us_taste", src_select=("taste",), density=1.0, w_init=2.0, evolve=False),
                                  plastic("value_app", 1.0), plastic("value_av", -1.0),
                                  S.P("value_app", "value_app_prev", topology="one_to_one", w_init=1.0, evolve=False),
                                  S.P("value_av", "value_av_prev", topology="one_to_one", w_init=1.0, evolve=False),
                                  S.fixed("value_app", "valence_app", S.TD_VALENCE), S.fixed("value_av", "valence_av", S.TD_VALENCE)),
                     modulators=(S.Mod("da", terms=terms), S.Mod("da_neg", terms=tuple((r, -w) for r, w in terms))))
    # the pain-gated synapses keep their inherited plasticity genes (remap_genomes carries A-D over for a projection
    # with the same rule, so an abcd set here is not used): they are taught by the negated error, da_neg = -da
    return replace(brain, projections=tuple(
        replace(p, modulator="da_neg", elig_delay=dl) if p.modulator == "us_av" and (looks or p.src == "pallium") else p
        for p in brain.projections))


def td_split(brain):
    """The critic with one error per valence (variant td2), as the reward and punishment dopamine cells of the
    mushroom body (PAM, PPL1) or the two dopamine cell types of Matsumoto & Hikosaka (2009):
        da_app = us_taste + k (gamma V_app - V_app')      teaches pallium -> value_app
        da_av  = us_pain  + k (gamma V_av  - V_av')       teaches pallium -> value_av and pallium -> valence_av
    so the end of a meal (appetitive value falls) is no lesson about danger. The look synapses stay pain-gated."""
    g, k, dl = S.TD_GAMMA, S.TD_SCALE, bool(S.TD_DELAY)
    plastic = lambda dst, mod: S.P("pallium", dst, rule="hebb", modulator=mod, eta_init=S.TD_ETA, elig_tau=S.TD_ELIG,
                                   abcd=(0.0, 1.0, 0.0, 0.0), density=1.0, w_init=0.0, elig_delay=dl)
    pal = next(r.phase for r in brain.regions if r.name == "pallium")
    value = lambda n: replace(S.R(n, 3, **S.VALUE), phase=pal + 1 if dl and not n.endswith("_prev") else 0)
    brain = S.extend(brain,
                     regions=(S.R("us_taste", 1, sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, group="us"),
                              *(value(n) for n in ("value_app", "value_av", "value_app_prev", "value_av_prev"))),
                     projections=(S.P("in", "us_taste", src_select=("taste",), density=1.0, w_init=2.0, evolve=False),
                                  plastic("value_app", "da_app"), plastic("value_av", "da_av"),
                                  S.P("value_app", "value_app_prev", topology="one_to_one", w_init=1.0, evolve=False),
                                  S.P("value_av", "value_av_prev", topology="one_to_one", w_init=1.0, evolve=False),
                                  S.fixed("value_app", "valence_app", S.TD_VALENCE), S.fixed("value_av", "valence_av", S.TD_VALENCE)),
                     modulators=(S.Mod("da_app", terms=(("us_taste", S.TD_TASTE), ("value_app", g * k), ("value_app_prev", -k))),
                                 S.Mod("da_av", terms=(("us_pain", 1.0), ("value_av", g * k), ("value_av_prev", -k)))))
    return replace(brain, projections=tuple(
        replace(p, modulator="da_av", elig_delay=dl) if p.modulator == "us_av" and p.src == "pallium" else p
        for p in brain.projections))


def variant(brain, v):
    for part in v.split("+"):
        if part == "base":
            pass
        elif part == "pslot":   # the pallium's input wiring belongs to the population slot (ProjectionSpec.per_slot);
            # what reads the pallium starts at 0 and is not evolved (it could only read a code drawn per slot)
            brain = replace(brain, projections=tuple(
                replace(p, per_slot=True) if p.dst == "pallium" else
                replace(p, w_init=0.0, evolve=False) if p.src == "pallium" else p for p in brain.projections))
        elif part == "noadapt":   # no input adaptation (the looks are not high-passed; BrainConfig.in_adapt)
            brain = replace(brain, in_adapt=())
        elif part == "pphase":   # control for pnt: only the pallium's move one phase later, without no_touch
            brain = replace(brain, regions=tuple(replace(r, phase=r.phase + 1) if r.name == "pallium" else r
                                                 for r in brain.regions))
        elif part == "pnt":   # the pallium also reads the no_touch cell (nothing at the mouth): look x at-the-mouth.
            # The pallium moves one phase later, so no_touch (phase 0, from this tick's inputs) and the look describe
            # the same tick; the synapses are random like the pallium's input (and per life where that one is)
            # on a brain that has the critic already (2.3), its value cells move along, so they still read the
            # pallium of the same tick
            pal = next(p for p in brain.projections if p.src == "in" and p.dst == "pallium")
            brain = replace(brain, regions=tuple(replace(r, phase=r.phase + 1)
                                                 if r.name in ("pallium", "value_app", "value_av") else r
                                                 for r in brain.regions),
                            projections=(*brain.projections,
                                         S.P("no_touch", "pallium", density=1.0, evolve=False, per_slot=pal.per_slot)))
        elif part.startswith("pk") and part[2:].isdigit():   # pallium k-WTA winners
            brain = replace(brain, regions=tuple(replace(r, kwta=int(part[2:])) if r.name == "pallium" else r
                                                 for r in brain.regions))
        elif part.startswith("pn") and part[2:].isdigit():   # pallium size
            brain = replace(brain, regions=tuple(replace(r, size=int(part[2:])) if r.name == "pallium" else r
                                                 for r in brain.regions))
        elif part == "pnear":   # the pallium also reads how near the object ahead is (vis+0.near): look x distance
            brain = replace(brain, projections=tuple(
                replace(p, src_select=(*p.src_select, "vis+0.near")) if p.src == "in" and p.dst == "pallium" else p
                for p in brain.projections))
        elif part == "short":
            brain = replace(brain, projections=tuple(
                replace(p, elig_tau=0.5) if p.modulator == "us_av" else p for p in brain.projections))
        elif part.startswith("safety"):
            k = float(part[6:]) if part[6:] else 1.0
            if not any(r.name == "safety" for r in brain.regions):   # stage 1.5 has the cell itself from v26 on
                brain = S.extend(brain,
                                 regions=(S.R("safety", 1, sign="exc", alpha=1.0, bias=-2.0, evolve_bias=False, group="us"),),
                                 projections=(S.P("in", "safety", src_select=("taste",), density=1.0, w_init=2.0 / 3.0, evolve=False),
                                              S.fixed("valence_av", "safety", 1.0)))
            brain = replace(brain, modulators=tuple(   # safety0 takes the cell out of the teacher
                replace(m, pos="", terms=(("us_pain", 1.0), ("safety", -k))) if m.name == "us_av" else m
                for m in brain.modulators))
        elif part == "pulse":
            brain = replace(brain, regions=tuple(replace(r, bias=-1.0) if r.name == "us_pain" else r for r in brain.regions),
                            projections=tuple(replace(p, w_init=3.0) if p.dst == "us_pain" else p for p in brain.projections))
        elif part == "ahead":
            out = []
            for p in brain.projections:
                if p.modulator in S.CS_MODS and p.src == "in" and p.src_select == S.CS_VIS:
                    out.append(replace(p, src_select=("vis+0.app*",)))
                    out.append(replace(p, src_select=("vis-*.app*", "vis+[36]0.app*"), rule="fixed", modulator="",
                                       eta_init=0.0, elig_tau=0.0, abcd=(1.0, 0.0, 0.0, 0.0), centred=False))
                else:
                    out.append(p)
            brain = replace(brain, projections=tuple(out))
        elif part.startswith("centred"):
            brain = replace(brain, in_trace_tau=float(part[7:]) if part[7:] else 0.99, projections=tuple(
                replace(p, centred=True) if p.modulator in S.CS_MODS else p for p in brain.projections))
        elif part.startswith("eta"):
            brain = replace(brain, projections=tuple(
                replace(p, eta_init=float(part[3:])) if p.modulator in S.CS_MODS else p for p in brain.projections))
        elif part.startswith("decay"):
            brain = replace(brain, projections=tuple(
                replace(p, decay=float(part[5:])) if p.modulator in S.CS_MODS else p for p in brain.projections))
        elif part == "td2":
            brain = td_split(brain)
        elif part in ("td", "tdp"):
            brain = td_critic(brain, looks=part == "td")
        else:
            raise SystemExit(f"unknown variant {part!r}; see variants.py")
    return brain
