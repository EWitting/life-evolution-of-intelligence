"""What a stage is (`Stage`), the registry of stages, fitness functions and the helpers every brain
definition uses (`extend`, `fixed`, action ranges, evolution settings of new stages)."""
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Callable

from life.config import WorldConfig, VisionConfig, BodyConfig, BrainConfig, ProjectionSpec as P


# chapter 1 senses: a few coarse, short-range "eyes" (early bilaterians had simple photoreceptors/chemosensing)
# appearance: 8 features per object (1 shared by a family of look-alikes + 7 for its 'colour'), so that different
# kinds are nearly orthogonal; with 4, an aversion learned for one berry type spread to all of them
APPEARANCE = 8
VISION_CH1 = VisionConfig(columns=5, fov_degrees=120.0, range=5, appearance_dim=APPEARANCE)


@dataclass
class Stage:
    key: str                      # "1.0"
    name: str                     # run directory name
    parent: str | None            # stage this one builds on (warm start + control architecture)
    brain: BrainConfig
    world: WorldConfig
    vision: VisionConfig
    body: BodyConfig
    build: Callable               # (exp) -> (ruleset, rules_for_generation)
    fitness: Callable = None
    row_extra: Callable | None = None   # (ruleset) -> (stats, fitness) -> {column: value}
    generations: int = 100
    ticks: int = 1000
    plastic: bool = False
    eta_max: float = 0.5          # evolution clips learning rates to [0, eta_max]
    evolution: tuple = ()         # ((EvolutionConfig field, value), ...) for this stage only (trials)
    dense_plastic: bool = False   # warm start: every synapse of a learned projection exists (the inherited ones keep
                                  # their weight, the others start at 0); an absent synapse could not learn
    adapt_init: Callable | None = None   # (population, layout) -> population: a change to the warm-start population
                                  # after it is remapped onto this stage's layout (e.g. interpolate_eyes)
    control_brain: BrainConfig | None = None   # the control's brain if not the parent's (e.g. with a new gene)
    schedule: Callable | None = None   # (rules_for_generation) -> rules_for_generation, applied during evolution only:
                                  # a world that changes over the generations (lesions, the book and probes, which ask
                                  # for generation 0, get the stage's world as built)
    notes: str = ""


STAGES: dict[str, Stage] = {}


def stage(s: Stage) -> Stage:
    STAGES[s.key] = s
    return s


def default_fitness(stats):
    """Well-fed lifetime from the first meal (v6-v18, and again from v23): the stomach level, as a fraction of full,
    summed over the ticks alive after the first meal. An animal that never eats scores zero. Survival is what counts,
    so a bite on a full stomach earns nothing and cold, which burns the reserve faster, costs in proportion. (v19-v22
    used energy acquired, `fitness_energy`: with recombination the two repair a damaged population equally well,
    STAGE_LOG v23, but energy acquired pays for every bite, hungry or not, so no need state can be selected.)"""
    return stats["fed_meal"]


def fitness_energy(stats):
    """Energy acquired (the fitness of v19-v22), kept for comparisons: all food eaten, not capped by the stomach."""
    return stats["eaten"]


def fitness_v4(stats):
    """The v4 fitness, kept for comparisons: gross food eaten (also on a full stomach) - pain + survival bonus."""
    return stats["eaten"] - stats["pain"] + 0.01 * stats["alive_ticks"]


def extend(base: BrainConfig, regions=(), projections=(), modulators=(), **kw) -> BrainConfig:
    """New brain = old brain + new regions (inserted before 'out' in the given order) + projections."""
    return replace(base, regions=base.regions + tuple(regions), projections=base.projections + tuple(projections),
                   modulators=base.modulators + tuple(modulators), **kw)


def fixed(src, dst, w, **kw):
    """A hard-wired projection (every synapse present, designed weight w): innate structure. Evolution cannot
    rewire it or flip its sign, but it may scale the whole projection (tune; ADR-016 amended)."""
    return P(src, dst, density=1.0, w_init=w, evolve=False, **{"tune": True, **kw})


# Action indices on the motor region 'out' (actions.py): NOOP 0, FORWARD 1, TURN_LEFT 2, TURN_RIGHT 3, USE 4, EAT 5
FWD, TURNS, FEED, USE_, EAT_ = (1, 2), (2, 4), (4, 6), (4, 5), (5, 6)

EPISODES = 4                     # worlds per generation: 16 lives per genome (4 siblings x 4 worlds) nearly double the
                                 # selection accuracy for 2.3x the GPU time; one large world or many small ones do no
                                 # better (STAGE_LOG v52). Frozen stages ran with 1 and keep it.
EVOLUTION_NEW = (("eta_log_std", 0.3), ("episodes", EPISODES))   # Stage.evolution of 2.3 (frozen) and its drafts
# Stage.evolution of new stages from 2026-10-20 on (user): 2 worlds per generation, 8 lives per genome. The time of a
# generation grows with the lives simulated (a generation ends when every creature is dead), and the harsh-world
# populations live 2-3x longer than those of v52 (more time per life to tell genomes apart): 4 episodes took 30-40 s
# per generation, 2 take about half (cost probe 2026-10-20: 8.7 s vs 4.1 s at generation 0)
EVOLUTION_2EP = (("eta_log_std", 0.3), ("episodes", 2))
