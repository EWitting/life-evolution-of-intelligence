"""Experiment 01: evolved forager (no lifetime learning).

World: OHOL gooseberry bushes (id 30) that yield gooseberries (31) six times, then become an empty bush
(279). Rule patch: in OHOL the empty bush only regrows after watering; here it regrows by itself after
REGROW_TICKS (extra_decays). Agents must find bushes, USE to pick a berry, and EAT it.
Brain: default architecture (in -> hidden -> out, all projections fixed), weights evolved.
Fitness: net food eaten plus a small survival bonus.
Observed 2026-09-18: with one world per generation the fitness signal is dominated by spawn luck; progress
still appears over ~40 generations but is noisy. Tuning that is left to the user (Intention.md).

Run: .venv\\Scripts\\python.exe -m life.experiments.exp01_evolved_forager [--generations N] [--quick]
"""
import argparse
import numpy as np

from life import ohol
from life.config import ExperimentConfig, WorldConfig, VisionConfig, BrainConfig, EvolutionConfig
from life.run import run_evolution

BUSH, BERRY, EMPTY_BUSH = 30, 31, 279
REGROW_TICKS = 500


def build_ruleset(world: WorldConfig):
    data = ohol.load()
    return ohol.slice_ruleset(data, [BUSH, BERRY, EMPTY_BUSH], ticks_per_second=world.ticks_per_ohol_second,
                              max_decay_ticks=world.max_decay_ticks, extra_decays={EMPTY_BUSH: (BUSH, REGROW_TICKS)})


def spawn_weights(rs):
    """Spawn only bushes; berries and empty bushes appear through interaction and decay."""
    spawn = np.zeros(rs.size, np.float32)
    spawn[rs.local(BUSH)] = 1.0
    return spawn


def fitness(stats):
    return stats["eaten"] + 0.01 * stats["alive_ticks"]


def make_config(generations: int = 30, quick: bool = False) -> ExperimentConfig:
    return ExperimentConfig(
        name="exp01_evolved_forager",
        world=WorldConfig(height=32, width=32, num_agents=64, spawn_density=0.08, max_decay_ticks=1000),
        vision=VisionConfig(),
        brain=BrainConfig(),
        evolution=EvolutionConfig(generations=3 if quick else generations, ticks_per_generation=100 if quick else 1000,
                                  plastic=False, seed=0),
    )


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--generations", type=int, default=30)
    p.add_argument("--quick", action="store_true", help="tiny run to check that everything works")
    args = p.parse_args(argv)
    exp = make_config(args.generations, args.quick)
    rs = build_ruleset(exp.world)
    rules = rs.to_arrays(exp.vision.appearance_dim, spawn_weight=spawn_weights(rs))
    print(rs.describe())
    return run_evolution(exp, rs, fitness, rules_for_generation=lambda gen, k: rules)


if __name__ == "__main__":
    main()
