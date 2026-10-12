"""Generation-0 test for the affect stage: can a hard-wired lasting state help the drives population in the world
where every berry type grows in its own patches? The parent population (drives, seed 0), unchanged, on the drives
brain ('base') and on variants, in the same worlds. A variant is a '+'-joined list of parts:

    alarm<d>[w<s>]   pain -> the modulator 'alarm', cleared by the factor d per tick (alarm0.9: about 20 ticks) ->
                     a `flee` cell -> FORWARD (strength s, default 3): after a poison bite, keep going
    stay<d>[w<s>]    taste -> a `fed` cell -> the modulator 'serotonin', cleared by the factor d per tick -> a receptor
                     on `rest` (strength s, default 4): after a meal `rest` holds FORWARD back also where it is cold,
                     so a fed animal stays in the patch; `hungry` still shuts `rest`

    uv run python scripts/probes/affect.py [field=value ...] [variant ...]    WORLDS=<n> (default 4)"""
import os, re, sys
from dataclasses import replace
import jax, numpy as np
from life.experiments import stages as S
from life.experiments.stages import R, Mod, fixed, extend, FWD
from life.run import load_population, latest_run, make_simulate

s = S.STAGES["1.8"]
WORLDS = int(os.environ.get("WORLDS", 4))
for a in sys.argv[1:]:
    if "=" in a:
        k_, v_ = a.split("=")
        s = replace(s, world=replace(s.world, **{k_: type(getattr(s.world, k_))(float(v_))}))
VARIANTS = [a for a in sys.argv[1:] if "=" not in a]
DEFAULT = ["alarm0.9", "alarm0.97", "stay0.97", "stay0.99", "alarm0.9+stay0.97"]
PART = re.compile(r"(alarm|stay)([0-9.]+)(?:w([0-9.]+))?$")


def affect(brain, v):
    regions, proj, mods = list(brain.regions), [], []
    for part in v.split("+"):
        kind, d, w = PART.match(part).groups()
        if kind == "alarm":
            mods.append(Mod("alarm", pos="us_pain", decay=float(d)))
            regions.append(R("flee", 1, sign="exc", alpha=1.0, bias=0.0, evolve_bias=False,
                             receptors=(("alarm", "bias", 3.0),), group="affect"))
            proj.append(fixed("flee", "out", float(w or 3.0), dst_range=FWD))
        else:
            mods.append(Mod("serotonin", pos="fed", decay=float(d)))
            regions.append(R("fed", 1, sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, group="affect"))
            proj.append(fixed("in", "fed", 2.0, src_select=("taste",)))
            regions = [replace(r, receptors=r.receptors + (("serotonin", "bias", float(w or 4.0)),)) if r.name == "rest" else r
                       for r in regions]
    new = tuple(r for r in regions if r.name not in {q.name for q in brain.regions})
    out = extend(replace(brain, regions=tuple(r for r in regions if r not in new)), regions=new, projections=tuple(proj))
    return replace(out, modulators=out.modulators + tuple(mods))


def measure(label, brain):
    exp = S.make_exp(s, brain, "affect_probe", 1, 0)
    pop = load_population(latest_run(S.STAGES[s.parent].name), exp, seed=0)
    rs, fn = s.build(exp)
    sim = make_simulate(exp, record=False)
    extra = S.poison_metrics(rs)
    acc = []
    for k in jax.random.split(jax.random.PRNGKey(53), WORLDS):
        kr, ks = jax.random.split(k)
        rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
        st, _ = sim(rules, pop, ks)
        e = extra(st, None)
        acc.append([float(st["fed_meal"].mean()), float(st["alive_ticks"].mean()), float(e["berries"]),
                    float(e["poison_frac"]), float(st["temp_mean"].mean())])
    acc = np.array(acc); m = acc.mean(0); se = acc.std(0, ddof=1) / np.sqrt(WORLDS)
    print(f"{label:22s} well-fed lifetime {m[0]:4.0f} +-{se[0]:.0f}  lifetime {m[1]:5.0f}  berries {m[2]:5.1f}  "
          f"poison share {m[3]:.3f}  body temperature {m[4]:.2f}   per world: " + " ".join(f"{x:.0f}" for x in acc[:, 0]), flush=True)
    return acc[:, 0]


print(f"world: {s.world.patches} patches of radius {s.world.patch_radius}, one type each; {WORLDS} worlds", flush=True)
base = measure("base (drives brain)", s.brain)
for v in VARIANTS or DEFAULT:
    f = measure(v, affect(s.brain, v))
    d = f - base
    print(f"{'':22s} paired difference to base {d.mean():+.0f} +-{d.std(ddof=1) / np.sqrt(WORLDS):.0f}", flush=True)
