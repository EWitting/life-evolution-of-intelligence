"""Experiment 02: Hebbian association of appearance with pain (evolved plastic network).

World: two look-alike berry bushes: the OHOL gooseberry bush (30/31/279) and a synthetic clone with its own
appearance (100030/100031/100279). In each world a coin flip decides which berry is poison: eating it
causes pain and drains food. Because the poison identity differs between worlds, evolution cannot
hard-wire avoidance; a genome that wins must use lifetime plasticity (eta > 0 on the right synapses) to
associate the berry's appearance with the pain signal and then suppress EAT/USE for that berry.
Brain: same layout as exp01 but every projection uses the 'hebb' (ABCD) rule; evolution may switch
plasticity on per synapse (EvolutionConfig.plastic=True).
Fitness: net food eaten minus pain, plus survival bonus.
Payoff design: good berry = +6 food, poison = -2 food and -1 fitness for pain, so random eating still pays
(+1.5 per berry) and discriminating doubles it. Harsher poison collapsed the population to "never eat".

Warm start: from a random population nobody forages well enough to ever taste poison, so start from the
evolved foragers of exp01 (ADR-012). Always run the --no-plastic control alongside: on 2026-09-18 the
control also avoided poison (23%) by reacting to the lingering pain input, i.e. the task did not yet
require synaptic memory. See docs/ROADMAP.md for what to change; this is left to the user to tune.

Run: .venv\\Scripts\\python.exe -m life.experiments.exp02_hebbian_association --init-from latest [--generations N]
     add --no-plastic for the control run (same start, no lifetime learning)
"""
import argparse
import jax
import jax.numpy as jnp
import numpy as np

from life import ohol
from life.config import ExperimentConfig, WorldConfig, VisionConfig, BrainConfig, EvolutionConfig, ProjectionSpec
from life.run import run_evolution, load_population, latest_run

BUSH, BERRY, EMPTY_BUSH = 30, 31, 279
CLONE = {BUSH: 100030, BERRY: 100031, EMPTY_BUSH: 100279}
REGROW_TICKS = 300   # rule patch: empty bushes regrow by themselves (OHOL needs watering)
POISON_FOOD = -1.0   # OHOL foodValue units (scaled by food_scale) lost when eating the poison berry
POISON_PAIN = 1.0

HEBB_PROJECTIONS = tuple(ProjectionSpec(s, d, rule="hebb") for s, d in
                         (("in", "hidden"), ("hidden", "hidden"), ("hidden", "out"), ("in", "out")))


def build_ruleset(world: WorldConfig):
    data = ohol.load()
    return ohol.slice_ruleset(data, [BUSH, BERRY, EMPTY_BUSH], ticks_per_second=world.ticks_per_ohol_second,
                              max_decay_ticks=world.max_decay_ticks, clones=CLONE, clone_name_prefix="Blue ",
                              extra_decays={EMPTY_BUSH: (BUSH, REGROW_TICKS)})


def fitness(stats):
    return stats["eaten"] - 1.0 * stats["pain"] + 0.01 * stats["alive_ticks"]


def make_config(generations: int = 60, quick: bool = False, plastic: bool = True) -> ExperimentConfig:
    return ExperimentConfig(
        name="exp02_hebbian_association" + ("" if plastic else "_control"),
        world=WorldConfig(height=32, width=32, num_agents=64, spawn_density=0.10, max_decay_ticks=1000),
        vision=VisionConfig(),
        brain=BrainConfig(projections=HEBB_PROJECTIONS),
        evolution=EvolutionConfig(generations=3 if quick else generations, ticks_per_generation=100 if quick else 800,
                                  plastic=plastic, seed=1),
    )


def poison_variants(rs, appearance_dim: int):
    """Two RuleArrays: poison = gooseberry, poison = blue gooseberry."""
    spawn = np.zeros(rs.size, np.float32)
    spawn[rs.local(BUSH)] = 1.0
    spawn[rs.local(CLONE[BUSH])] = 1.0
    variants = []
    for poison in (rs.local(BERRY), rs.local(CLONE[BERRY])):
        fv = rs.food_value.copy()
        pv = np.zeros_like(fv)
        fv[poison] = POISON_FOOD
        pv[poison] = POISON_PAIN
        variants.append(rs.to_arrays(appearance_dim, spawn_weight=spawn, food_value=fv, pain_value=pv))
    return variants


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-plastic", action="store_true", help="control run without lifetime plasticity")
    p.add_argument("--init-from", default=None,
                   help="run dir (or 'latest' = newest exp01 run) whose final population warm-starts this experiment")
    args = p.parse_args(argv)
    exp = make_config(args.generations, args.quick, plastic=not args.no_plastic)
    init_pop = None
    if args.init_from:
        src = latest_run("exp01_evolved_forager") if args.init_from == "latest" else args.init_from
        print(f"warm start from {src}")
        init_pop = load_population(src)
    rs = build_ruleset(exp.world)
    print(rs.describe())
    variants = poison_variants(rs, exp.vision.appearance_dim)

    def rules_for_generation(gen, key):
        # one coin flip per episode (world); with episodes=1 that is one per generation
        picks = jax.random.bernoulli(key, shape=(exp.evolution.episodes,)).astype(jnp.int32)
        return jax.tree_util.tree_map(lambda x0, x1: jnp.stack([x0, x1])[picks], variants[0], variants[1])

    return run_evolution(exp, rs, fitness, rules_for_generation=rules_for_generation, init_population=init_pop)


if __name__ == "__main__":
    main()
