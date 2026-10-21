"""Does the critic cancel a predicted pain when its state changes as the pain nears? Lab extinction (labtests
.conditioning) on a run's population (2.3c), the brain as frozen and three variants (new links, the value cells'
learning rates copied from their pallium synapses, genome by genome):
    base     as frozen
    touch    the pallium also reads `no_touch` (look x at the mouth; `pallium_touch`, v60), the critic one phase later
    near     the value cells also read the distance of the bush ahead (`vis+0.near`), learned like the pallium's input
    command  the value cells also read a copy of the bite command (`efc_USE`), set on the tick each bite is commanded,
             one before its outcome (a stand-in for a same-tick copy of the motor command: in the world the
             efference sense arrives a tick late, with the pain)
Protocols: lab trials (3 ticks of pain, 0.3 after), world-like (1 tick, nothing after); `approach` (8 ticks in view,
7 from afar) for touch and near. Per variant: error on the tick before the first bite (bush) and at the bites, on the
first and sixth painful meal and the first without pain; aversion drive after 6 painful meals and 12 without.
    python scripts/probes/lab_state.py <run dir> [<run dir> ...] [HAB=f] [VARIANTS=base,command]
HAB=f: the pain -> habenula strength times f (does the habenula's saturation hide the prediction?)."""
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from life.experiments import stages as S
from life.labtests import PROTOCOLS, conditioning
from life.run import load_population, make_layout

ST = S.STAGES["2.3c"]
CRITIC = ("value_app", "value_av", "ventral_pallidum", "gp_habenular", "habenula", "dopamine")


def variant(name):
    b, body = ST.brain, ST.body
    if name == "touch":
        pal = next(p for p in b.projections if p.src == "in" and p.dst == "pallium")
        b = replace(b, regions=tuple(replace(r, phase=r.phase + 1) if r.name in ("pallium",) + CRITIC else r
                                     for r in b.regions),
                    projections=b.projections + (S.P("no_touch", "pallium", density=1.0, evolve=False,
                                                     per_slot=pal.per_slot),))
    elif name in ("near", "command"):
        feat = "vis+0.near" if name == "near" else "efc_USE"
        new = tuple(replace(p, src="in", src_select=(feat,)) for p in b.projections
                    if p.src == "pallium" and p.dst in ("value_app", "value_av"))
        b = replace(b, projections=b.projections + new)
        if name == "command":
            body = replace(body, efference=True)
    return replace(ST, brain=b, body=body), feat if name in ("near", "command") else None


def population(run, name):
    s, feat = variant(name)
    exp = S.make_exp(s, s.brain, "lab", 1, 0)
    pop = load_population(run, exp)
    if feat:   # the new synapses learn as the pallium's onto the same value cells, genome by genome
        L = make_layout(exp)
        pal, i = L.region("pallium"), L.in_names.index(feat)
        genes = {k: np.array(getattr(pop, k)) for k in ("eta", "A", "B", "C", "D")}
        for dst in ("value_app", "value_av"):
            cols = L.region(dst)
            for k, g in genes.items():
                g[:, i, cols] = g[:, pal][:, :, cols].mean(axis=1)
        pop = pop._replace(**{k: jnp.asarray(v) for k, v in genes.items()})
    if HAB != 1.0:
        L = make_layout(exp)
        i, cols = L.in_names.index("pain"), L.region("habenula")
        pop = pop._replace(w0=pop.w0.at[:, i, cols].multiply(HAB))
    return exp, pop


HAB = float(next((a[4:] for a in sys.argv[1:] if a.startswith("HAB=")), 1.0))
VARIANTS = next((a[9:].split(",") for a in sys.argv[1:] if a.startswith("VARIANTS=")), ("base", "touch", "near", "command"))
PROTO = {"lab": {}, "world-like": dict(outcome_ticks=1, linger=0.0)}
for run in [Path(a) for a in sys.argv[1:] if "=" not in a]:
    print(f"\n{run}")
    print(f"  {'variant':8s} {'trials':22s} {'meal 1: bush, bites':>20s} {'meal 6: bush, bites':>20s} "
          f"{'no pain: bush, bites':>21s} {'aversion 6 -> 18':>17s}")
    for name in VARIANTS:
        exp, pop = population(run, name)
        for pl, pe in PROTO.items():
            for appr in ((False, True) if name in ("touch", "near") else (False,)):
                extra = {**pe, **(dict(lead=8, approach=True) if appr else {}), **(dict(command=True) if name == "command" else {})}
                pr = {**PROTOCOLS["extinction"], **extra}
                r = conditioning(exp, pop, pr, S.PALLIUM_SIM)
                lead, k = pr.get("lead", 1), pr.get("outcome_ticks", 3)
                sig = r["signal"].mean(1)                                   # [trials, ticks]
                cell = lambda t: f"{sig[t, lead - 1]:+.2f}, {sig[t, lead:lead + k].mean():+.2f}"
                a = r["tests"]["A"][:, :, 3].mean(1)
                label = pl + (", approach" if appr else "")
                print(f"  {name:8s} {label:22s} {cell(0):>20s} {cell(5):>20s} {cell(6):>21s} "
                      f"{a[6]:>8.2f} -> {a[18]:.2f}", flush=True)
