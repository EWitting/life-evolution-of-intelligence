"""All tunable knobs. Every field states its unit and default. Frozen so they can be static jit args."""
from dataclasses import dataclass, field, asdict
import json


@dataclass(frozen=True)
class WorldConfig:
    height: int = 32                 # cells
    width: int = 32                  # cells
    num_agents: int = 32             # agents = genomes per generation (ADR-008)
    max_food: float = 20.0           # food units; agents start full
    hunger_per_tick: float = 0.05    # food units lost per tick while alive (20/0.05 = 400 ticks to starve)
    food_scale: float = 2.0          # food units gained per OHOL foodValue point
    max_age: int = 100000            # ticks; death of old age (large = effectively off)
    ticks_per_ohol_second: float = 1.0   # conversion of OHOL decay seconds to ticks (ADR-002)
    max_decay_ticks: int = 600       # ticks; clip for very long OHOL decays (hours)
    hear_radius: int = 6             # cells (Chebyshev) within which VOCALIZE is heard
    pain_decay: float = 0.9          # multiplier per tick on the pain body signal
    spawn_density: float = 0.08      # fraction of cells that get an object at world init


@dataclass(frozen=True)
class VisionConfig:
    columns: int = 9                 # W: retina width in ray columns
    fov_degrees: float = 120.0       # field of view spread of the columns
    range: int = 6                   # cells; maximum ray length
    appearance_dim: int = 4          # K: size of the per-object appearance vector (ADR-006)


@dataclass(frozen=True)
class RegionSpec:
    """A named block of neurons (ADR-013). 'in' and 'out' are reserved and created automatically."""
    name: str
    size: int
    alpha: float = 0.5               # leak of this region: x <- (1-alpha) x + alpha tanh(...)
    trace_tau: float = 0.8           # decay per step of the activity trace used by the 'trace' rule


@dataclass(frozen=True)
class ProjectionSpec:
    """Synapses from every neuron of region src to every neuron of region dst, present with prob density.
    rule: 'fixed' (no plasticity), 'hebb' (ABCD: A pre post + B pre + C post + D), 'oja' (A post (pre - post w)),
    'trace' (STDP-like: A pre_trace post - C pre post_trace). modulated=True multiplies the update by the
    modulator (three-factor learning, ADR-013). eta_init is the starting learning rate (evolution changes it)."""
    src: str
    dst: str
    density: float = 0.3
    rule: str = "fixed"
    modulated: bool = False
    eta_init: float = 0.0


DEFAULT_REGIONS = (RegionSpec("hidden", 32),)
DEFAULT_PROJECTIONS = (ProjectionSpec("in", "hidden"), ProjectionSpec("hidden", "hidden"),
                       ProjectionSpec("hidden", "out"), ProjectionSpec("in", "out"))


@dataclass(frozen=True)
class BrainConfig:
    regions: tuple = DEFAULT_REGIONS          # tuple[RegionSpec, ...] excluding the reserved 'in' and 'out'
    projections: tuple = DEFAULT_PROJECTIONS  # tuple[ProjectionSpec, ...]
    out_alpha: float = 0.5           # leak of the output region
    w_max: float = 4.0               # weight clip after plasticity
    steps_per_tick: int = 1          # brain updates per world tick
    logit_gain: float = 4.0          # multiplier on output pre-activations before sampling
    action_temperature: float = 0.5  # softmax temperature; 0 means argmax
    modulator: str = "none"          # 'none': mod = 1. 'reward': mod = food gained (OHOL foodValue units)
                                     # minus pain received in the previous tick (ADR-013)


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int = 30
    ticks_per_generation: int = 400  # ticks each generation lives
    episodes: int = 1                # independent worlds per generation. 1 = one life per genome (default,
                                     # chosen for biological plausibility, see ADR-012). >1 averages stats.
    elite_frac: float = 0.125        # fraction copied unchanged
    tournament: int = 3              # tournament size for parent selection
    mutation_std: float = 0.1        # Gaussian std on weights and biases
    mask_flip_prob: float = 0.005    # per allowed synapse: toggle presence
    plastic: bool = False            # if False, plasticity genes (eta, A, B, C, D) are never mutated
    eta_max: float = 0.05            # clip for learning rates when plastic
    eta_mutation_prob: float = 0.01  # per plastic-capable synapse: chance to switch plasticity on (random eta)
                                     # or off (eta=0). Dense eta noise destroyed inherited behaviour (2026-09-18).
    record_weights_every: int = 100  # ticks between weight snapshots in the recording (must divide ticks)
    seed: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "unnamed"
    world: WorldConfig = field(default_factory=WorldConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
