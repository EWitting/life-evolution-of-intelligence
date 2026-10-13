"""Generation-0 test of lasting states that add up (user, 2026-10-13). As built in stage 1.8 a modulator is released
at once: one poison bite is a full alarm, one good bite a full stay. Here it is a leaky sum of the events
(`ModulatorSpec.accumulate`), and the cell that reads it (`flee`, `settle`) has a threshold counted in bites, so
that in a patch of mixed kinds the animal leaves or stays on the evidence of several bites. An evolved population,
unchanged, as it is and with variants, in the same worlds. A variant is a '+'-joined list of

    alarm<t>[d<decay>]    the alarm adds up; `flee` fires above t poison bites (0 = no threshold); decay per tick
    fed<t>[d<decay>]      serotonin adds up; `settle` fires above t good bites        instead of the evolved one

    uv run python scripts/probes/sum_affect.py <trial key> [variant ...]     WORLDS=<n> (default 8)"""
import os, re, sys
from dataclasses import replace
import jax, numpy as np
from life.experiments import stages as S
from life.run import load_population, latest_run, make_layout, make_simulate

s = S.STAGES[sys.argv[1]]
WORLDS = int(os.environ.get("WORLDS", 8))
PART = re.compile(r"(alarm|fed)([0-9.]+)(?:d([0-9.]+))?$")
CELL = {"alarm": ("flee", "us_pain"), "fed": ("settle", "fed")}      # the cell with the receptor, the cell that releases
MOD = {"alarm": "alarm", "fed": "serotonin"}
run = latest_run(s.name)

# the size of one bite: the activity of the releasing cell on a tick it fires (a short recorded life)
short = S.make_exp(replace(s, ticks=400), s.brain, "sum_affect_probe", 1, 0)
lay = make_layout(short)
rs, fn = s.build(short)
kr, ks = jax.random.split(jax.random.PRNGKey(7))
_, rec = make_simulate(short, record=True)(jax.tree_util.tree_map(lambda a: a[0], fn(0, kr)), load_population(run, short, seed=0), ks)
x = np.asarray(rec["x"], np.float32)
BITE = {}
for kind, (_, src) in CELL.items():
    v = x[:, :, lay.region(src)].reshape(-1)
    BITE[kind] = float(np.median(v[v > 0.05]))
    print(f"one bite: {src} fires at {BITE[kind]:.2f} (quartiles {np.percentile(v[v > 0.05], 25):.2f}-{np.percentile(v[v > 0.05], 75):.2f}), "
          f"on {100 * (v > 0.05).mean():.2f}% of ticks", flush=True)
del rec, x


def variant(v):
    """(brain, {modulator: decay or None})"""
    brain, decays = s.brain, {}
    for part in v.split("+"):
        kind, t, d = PART.match(part).groups()
        cell, _ = CELL[kind]
        sens = dict((m, w) for r in brain.regions if r.name == cell for m, _, w in r.receptors)[MOD[kind]]
        brain = replace(brain,
                        modulators=tuple(replace(m, accumulate=1.0) if m.name == MOD[kind] else m for m in brain.modulators),
                        regions=tuple(replace(r, bias=-sens * float(t) * BITE[kind]) if r.name == cell else r for r in brain.regions))
        decays[MOD[kind]] = float(d) if d else None
    return brain, decays


def measure(label, brain, decays):
    exp = S.make_exp(s, brain, "sum_affect_probe", 1, 0)
    layout = make_layout(exp)
    pop = load_population(run, exp, seed=0)
    for m, d in decays.items():
        if d:
            pop = pop._replace(clear=pop.clear.at[:, layout.mod_names.index(m)].set(d))
    _, fn_ = s.build(exp)
    sim = make_simulate(exp, record=False)
    extra = S.poison_metrics(rs)
    acc = []
    for k in jax.random.split(jax.random.PRNGKey(53), WORLDS):
        kr, ks = jax.random.split(k)
        st, _ = sim(jax.tree_util.tree_map(lambda a: a[0], fn_(0, kr)), pop, ks)
        e = extra(st, None)
        acc.append([float(st["fed_meal"].mean()), float(st["alive_ticks"].mean()), float(e["berries"]), float(e["poison_frac"])])
    acc = np.array(acc); m = acc.mean(0)
    print(f"[{s.key}] {label:22s} well-fed lifetime {m[0]:5.0f} +-{acc[:, 0].std(ddof=1) / np.sqrt(WORLDS):.0f}  lifetime {m[1]:5.0f}  "
          f"berries {m[2]:5.1f}  poison share {m[3]:.3f}", end="", flush=True)
    return acc[:, 0]


pop0 = load_population(run, S.make_exp(s, s.brain, "sum_affect_probe", 1, 0), seed=0)
names = make_layout(short).mod_names
print("evolved fade times, ticks: " + ", ".join(f"{m} {1 / (1 - float(np.median(np.asarray(pop0.clear)[:, names.index(m)]))):.0f}"
                                                 for m in MOD.values()), flush=True)
base = measure("as evolved", s.brain, {})
print(flush=True)
for v in sys.argv[2:]:
    d = measure(v, *variant(v)) - base
    print(f"   against as evolved {d.mean():+5.0f} +-{d.std(ddof=1) / np.sqrt(WORLDS):.0f}", flush=True)
