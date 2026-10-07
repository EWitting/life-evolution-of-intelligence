"""Generation-0 test for the drives stage: what does the cold cost the learning population, and can a hard-wired
warmth-seeking mode win it back, with and without hunger deciding between foraging and warming up?

The parent population (stage 1.5 by default) is put, unchanged, into its own world made cold (few hot springs), on
the parent brain ('blind': cold is not sensed) and on variants of a kinesis circuit, in the same worlds. A variant is
a '+'-joined list of parts:

    run            cold -> FORWARD: keep moving while the body is cold (orthokinesis; needs no gradient)
    turn           cold and the skin getting colder -> turn (klinokinesis, as C. elegans AFD -> AIY/AIZ)
    climb          cold and the skin getting warmer -> FORWARD (the old stage 1.2 rule)
    rest[<w>]      body warm -> FORWARD suppressed (strength w, default 3): stay where it is warm
    skin           'run' and 'rest' read the temperature of the skin (the place) instead of the body's, which lags
    hungry<f>      an inhibitory cell that fires below stomach level f (fraction of full) and shuts all of the
                   above: forage when hungry, look after warmth when fed (e.g. hungry0.67)

The reference is the blind brain in the same world without cold. Default variants: see DEFAULT below.

    uv run python scripts/probes/drives.py [field=value ...] [variant ...]   (WorldConfig fields, e.g. temp_rate=0.005)
    SPRINGS=<spawn weight of hot springs, default 0.1>  PARENT=<stage key, default 1.5>  in the environment
"""
import os, sys
from dataclasses import replace
import jax, numpy as np
from life.config import BodyConfig
from life.experiments import stages as S
from life.experiments.stages import R, P, fixed, extend, FWD, TURNS
from life.run import load_population, latest_run, make_simulate

parent = S.STAGES[os.environ.get("PARENT", "1.5")]
SPRINGS = float(os.environ.get("SPRINGS", "0.1"))
WORLDS = 4
world = replace(parent.world, temperature=True, ambient_temp=0.25, heat_scale=0.15, heat_radius=4, temp_rate=0.03,
                temp_hunger=1.5)
VARIANTS = [a for a in sys.argv[1:] if "=" not in a]
for a in sys.argv[1:]:
    if "=" in a:
        k_, v_ = a.split("=")
        world = replace(world, **{k_: type(getattr(world, k_))(float(v_))})
body = BodyConfig(taste=True, temperature=True, skin_change=True, skin=True)
DEFAULT = ["run+rest+skin", "run+turn+rest+skin", "run+turn+rest+skin+hungry0.67", "run+turn+rest+hungry0.67"]
HYP = dict(alpha=0.5, evolve_bias=False, group="hypothalamus")
PROG = dict(sign="exc", alpha=1.0, evolve_bias=False, group="thermotaxis")


def kinesis(brain, v):
    parts = v.split("+")
    hungry = next((float(q[6:]) for q in parts if q.startswith("hungry")), None)
    place = ("skin",) if "skin" in parts else ("temperature",)
    regions = [R("cold", 1, sign="exc", bias=3.3, **HYP)]                     # max(0, tanh(3.3 - 7 T)): below 0.47
    proj = [fixed("in", "cold", -7.0, src_select=place)]
    cells = []
    if "run" in parts:
        regions.append(R("warm_seek", 1, bias=0.0, **PROG)); cells.append("warm_seek")
        proj += [fixed("cold", "warm_seek", 3.0), fixed("warm_seek", "out", 3.0, dst_range=FWD)]
    if "climb" in parts:
        regions.append(R("warm_run", 1, bias=-2.5, **PROG)); cells.append("warm_run")
        proj += [fixed("cold", "warm_run", 2.5), fixed("in", "warm_run", 3.0, src_select=("skin_change",)),
                 fixed("warm_run", "out", 3.0, dst_range=FWD)]
    if "turn" in parts:
        regions.append(R("warm_turn", 1, bias=-2.5, **PROG)); cells.append("warm_turn")
        proj += [fixed("cold", "warm_turn", 2.5), fixed("in", "warm_turn", -3.0, src_select=("skin_change",)),
                 fixed("warm_turn", "out", 3.0, dst_range=TURNS)]
    rest = next((float(q[4:] or 3.0) for q in parts if q.startswith("rest")), None)
    if rest is not None:
        regions.append(R("rest", 1, sign="inh", bias=-3.3, **HYP)); cells.append("rest")   # body above 0.47
        proj += [fixed("in", "rest", 7.0, src_select=place), fixed("rest", "out", rest, dst_range=FWD)]
    if hungry is not None:
        regions.append(R("hungry", 1, sign="inh", bias=6.0 * hungry, **HYP))
        proj += [fixed("in", "hungry", -6.0, src_select=("food",))] + [fixed("hungry", c, 6.0) for c in cells]
    return extend(brain, regions=tuple(regions), projections=tuple(proj))


def measure(label, brain, w):
    s = replace(parent, world=w, body=body, build=lambda exp: S.learning_world(exp, springs=SPRINGS))
    exp = S.make_exp(s, brain, "drives_probe", 1, 0)
    pop = load_population(latest_run(parent.name), exp, seed=0)
    rs, fn = s.build(exp)
    sim = make_simulate(exp, record=False)
    acc = []
    for k in jax.random.split(jax.random.PRNGKey(53), WORLDS):
        kr, ks = jax.random.split(k)
        rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
        st, _ = sim(rules, pop, ks)
        acc.append([float(st["eaten"].mean()), float(st["alive_ticks"].mean()),
                    float(st["temp_mean"].mean()) if "temp_mean" in st else float("nan"), float(st["pain"].mean()),
                    float(st["fed_meal"].mean())])
    m = np.mean(acc, 0); se = np.std(acc, 0, ddof=1) / np.sqrt(WORLDS)
    print(f"{label:28s} energy {m[0]:5.1f} +-{se[0]:.1f}  lifetime {m[1]:5.0f} +-{se[1]:.0f}  well-fed lifetime {m[4]:4.0f} +-{se[4]:.0f}  "
          f"body temperature {m[2]:.2f}  pain {m[3]:.1f}", flush=True)


print(f"parent {parent.key}; springs {SPRINGS}; ambient {world.ambient_temp}, temp_rate {world.temp_rate}, "
      f"temp_hunger {world.temp_hunger}, hunger_per_tick {world.hunger_per_tick}", flush=True)
measure("blind, world not cold", parent.brain, replace(world, temperature=False))
measure("blind", parent.brain, world)
for v in VARIANTS or DEFAULT:
    measure(v, kinesis(parent.brain, v), world)
