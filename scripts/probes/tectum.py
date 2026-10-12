"""Generation-0 test for the first stage of chapter 2: what do the camera eyes, the shared colour and the tectum do
to a population that has not evolved with them? The same worlds for every variant (so lines of separate processes
can be compared world by world). Variants:

    eyes5            the affect brain with the old eyes, old colours (the population as it evolved)
    eyes5+colour     the same in the world of chapter 2 (a type's bush, berry and empty bush share one colour)
    eyes9            camera eyes (9 columns, range 6), chapter 2 world: the control of the tectum stage
    tectum           stage 2.1 as defined
    tectum:<parts>[:<strength>]   parts 'orient', 'approach' or 'orient+approach'; strength of programme -> motor

    uv run python scripts/probes/tectum.py [variant ...]
    WORLDS=<n> (default 4), POP=even (default: the newest run of trial x.weven, adapted to this world) or frozen"""
import os, sys
from dataclasses import replace
import jax, numpy as np
from life.experiments import stages as S
from life.freeze import load_frozen
from life.run import load_population, latest_run, make_simulate

WORLDS, POP = int(os.environ.get("WORLDS", 4)), os.environ.get("POP", "even")
s21 = S.STAGES["2.1"]


def measure(v):
    name, *opt = v.split(":")
    shared = name != "eyes5"
    if name.startswith("eyes5"):
        s = replace(s21, brain=S.B18, vision=S.VISION_CH1)
    elif name == "eyes9":
        s = replace(s21, brain=S.B20)
    else:
        s = s21 if not opt else replace(s21, brain=S.with_tectum(S.B20, parts=opt[0], out=float(opt[1]) if opt[1:] else S.TECT_OUT))
    exp = S.make_exp(s, s.brain, "tectum_probe", 1, 0)
    pop = load_frozen("1.8", 0, exp) if POP == "frozen" else load_population(latest_run("x_world_even"), exp, seed=0)
    rs, fn = S.world_ch2(exp, shared_colour=shared)
    sim = make_simulate(exp, record=False)
    extra = S.poison_metrics(rs)
    acc = []
    for k in jax.random.split(jax.random.PRNGKey(91), WORLDS):
        kr, ks = jax.random.split(k)
        rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
        st, _ = sim(rules, pop, ks)
        e = extra(st, None)
        acc.append([float(st["fed_meal"].mean()), float(st["alive_ticks"].mean()), float(e["berries"]), float(e["poison_frac"])])
    acc = np.array(acc); m = acc.mean(0)
    print(f"[{POP}] {v:28s} well-fed lifetime {m[0]:4.0f}  lifetime {m[1]:5.0f}  berries {m[2]:5.1f}  poison share {m[3]:.3f}"
          f"   per world: " + " ".join(f"{x:.0f}" for x in acc[:, 0]), flush=True)


for v in sys.argv[1:] or ["eyes5", "eyes5+colour", "eyes9", "tectum"]:
    measure(v)
