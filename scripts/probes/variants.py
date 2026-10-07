"""Brain variants for the learning probes (gen0.py, assay.py): small changes to a stage's brain, combined with '+'.

    base          the stage's brain as defined
    eta<x>        both conditioning learning rates start at x (e.g. eta0.2)
    decay<x>      learned weights relax toward the inherited ones by this fraction per tick (e.g. decay0.01)
    short         the sickness teacher gets the same short eligibility trace as the taste teacher (0.5)
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
        elif part.startswith("eta"):
            brain = replace(brain, projections=tuple(
                replace(p, eta_init=float(part[3:])) if p.modulator in S.CS_MODS else p for p in brain.projections))
        elif part.startswith("decay"):
            brain = replace(brain, projections=tuple(
                replace(p, decay=float(part[5:])) if p.modulator in S.CS_MODS else p for p in brain.projections))
        else:
            raise SystemExit(f"unknown variant {part!r}; see variants.py")
    return brain
