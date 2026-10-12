"""Generation-0 test for the camera eyes of chapter 2: what do nine eye columns and the shared colour do to a
population that has not evolved with them? The same worlds for every variant (so lines of separate processes can be
compared world by world; with '+colour' the looks are drawn differently, the places are the same). Variants:

    eyes5            the old eyes (5 columns, range 5): the population as it evolved
    silent           camera eyes (9 columns, range 6), the synapses of the four new columns start at zero
    interp           the new columns start as the mean of their neighbours (`stages.interpolate_eyes`)
    interp<s>        the same, then every synapse from the eyes times s (e.g. interp0.56: 5/9, the same total input)
    <any>+colour     a type's bush, berry and empty bush share one colour

    uv run python scripts/probes/eyes.py [variant ...]
    WORLDS=<n> (default 4), POP=even (default: the newest run of trial x.weven, adapted to this world) or frozen"""
import os, sys
from dataclasses import replace
import jax, numpy as np
from life.experiments import stages as S
from life.freeze import load_frozen
from life.run import load_population, latest_run, make_layout, make_simulate

WORLDS, POP = int(os.environ.get("WORLDS", 4)), os.environ.get("POP", "even")
s21 = S.STAGES["2.1"]


def measure(v):
    name, *opt = v.split("+")
    s = replace(s21, brain=S.B18, vision=S.VISION_CH1) if name == "eyes5" else s21
    exp = S.make_exp(s, s.brain, "eyes_probe", 1, 0)
    pop = load_frozen("1.8", 0, exp) if POP == "frozen" else load_population(latest_run("x_world_even"), exp, seed=0)
    if name.startswith("interp"):
        pop = S.interpolate_eyes(pop, make_layout(exp), scale=float(name[6:] or 1.0))
    rs, fn = S.world_ch2(exp, shared_colour="colour" in opt)
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
    print(f"[{POP}] {v:18s} well-fed lifetime {m[0]:4.0f} +-{acc[:, 0].std(ddof=1) / np.sqrt(WORLDS):.0f}  lifetime {m[1]:5.0f}  "
          f"berries {m[2]:5.1f}  poison share {m[3]:.3f}   per world: " + " ".join(f"{x:.0f}" for x in acc[:, 0]), flush=True)


for v in sys.argv[1:] or ["eyes5", "silent", "interp"]:
    measure(v)
