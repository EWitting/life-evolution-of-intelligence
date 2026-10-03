"""Generation-0 test for the drives stage: what does the cold cost the learning population, and can a hard-wired
warmth-seeking mode win it back, with and without hunger deciding between foraging and warming up?

The parent population (stage 1.5 by default) is put, unchanged, into its own world made cold (hot springs added),
on three brains, in the same worlds:

    blind          the parent brain; cold is not sensed
    taxis          + cold cell gating thermotaxis (run up the skin-temperature gradient, turn when it falls)
    taxis+hunger   + an inhibitory 'hungry' cell that shuts thermotaxis: forage when hungry, warm up when fed

and, as the reference, the blind brain in the same world without cold.

    uv run python scripts/probes/drives.py [field=value ...]      (WorldConfig fields, e.g. temp_rate=0.005)
    SPRINGS=<spawn weight of hot springs, default 0.6>  PARENT=<stage key, default 1.5>  in the environment
"""
import os, sys
from dataclasses import replace
import jax, numpy as np
from life.config import BodyConfig
from life.experiments import stages as S
from life.experiments.stages import R, P, fixed, extend, FWD, TURNS
from life.run import load_population, latest_run, make_simulate

parent = S.STAGES[os.environ.get("PARENT", "1.5")]
SPRINGS = float(os.environ.get("SPRINGS", "0.6"))
WORLDS = 4
world = replace(parent.world, temperature=True, ambient_temp=0.25, heat_scale=0.15, heat_radius=4, temp_rate=0.03,
                temp_hunger=1.5)
for a in sys.argv[1:]:
    k_, v_ = a.split("=")
    world = replace(world, **{k_: type(getattr(world, k_))(float(v_))})
body = BodyConfig(taste=True, temperature=True, skin_change=True)
THERMO = dict(sign="exc", alpha=1.0, bias=-2.5, evolve_bias=False, group="thermotaxis")
GATE = dict(sign="inh", alpha=0.5, evolve_bias=False, group="hypothalamus")


def taxis(brain, hunger=None):
    regions = [R("cold", 1, sign="exc", alpha=0.5, bias=3.3, evolve_bias=False, group="hypothalamus"),
               R("warm_run", 1, **THERMO), R("warm_turn", 1, **THERMO)]
    proj = [fixed("in", "cold", -7.0, src_select=("temperature",)),
            fixed("cold", "warm_run", 2.5), fixed("cold", "warm_turn", 2.5),
            fixed("in", "warm_run", 3.0, src_select=("skin_change",)),
            fixed("in", "warm_turn", -3.0, src_select=("skin_change",)),
            fixed("warm_run", "out", 3.0, dst_range=FWD), fixed("warm_turn", "out", 3.0, dst_range=TURNS)]
    if hunger:   # hungry = max(0, tanh(2 - 3 food)): fires below about 2/3 full, and closes the warming mode
        regions.append(R("hungry", 1, bias=2.0, **GATE))
        proj += [fixed("in", "hungry", -3.0, src_select=("food",)),
                 fixed("hungry", "warm_run", 6.0), fixed("hungry", "warm_turn", 6.0)]
    return extend(brain, regions=tuple(regions), projections=tuple(proj))


BRAINS = {"blind": parent.brain, "taxis": taxis(parent.brain), "taxis+hunger": taxis(parent.brain, hunger=True)}


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
measure("blind, world not cold", BRAINS["blind"], replace(world, temperature=False))
for name, b in BRAINS.items():
    measure(name, b, world)
