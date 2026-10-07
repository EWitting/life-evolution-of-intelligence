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
    safety        a 'safety' cell (tastes good although the aversive cells expected bad one tick earlier) is
                  subtracted from the pain teacher, so a safe meal undoes suspicion of what was just eaten (the
                  v19-v20 design); safety<k> sets its weight in the teacher (default 1)
"""
from dataclasses import replace
from life.experiments import stages as S


def variant(brain, v):
    for part in v.split("+"):
        if part == "base":
            pass
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
        else:
            raise SystemExit(f"unknown variant {part!r}; see variants.py")
    return brain
