"""All tunable knobs. Every field states its unit and default. Frozen so they can be static jit args."""
from dataclasses import dataclass, field, asdict, fields
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
    patches: int = 0                 # > 0: objects spawn only inside this many discs (patchy food)
    patch_radius: int = 3            # cells
    variety_bonus: float = 0.0       # OHOL 'yum': food not eaten recently is worth (1 + bonus) x, food eaten
                                     # repeatedly down to (1 - bonus) x; 0 = off
    variety_tau: float = 100.0       # ticks over which the memory of what was eaten fades
    switch_tick: int = 0             # > 0: rules_for_generation returns two phases per life; phase 2 from this tick
                                     # (e.g. the poison identity reverses mid-life). Life halves in stats split here.
    sickness_delay: int = 0          # ticks between eating something with pain_value and feeling the pain
    eat_on_pick: bool = False        # True: food that a USE puts into an empty hand is eaten in the same tick (an
                                     # animal with a mouth, not hands; nothing edible is ever carried)
    # --- temperature (OHOL-style): body temp drifts toward the local temperature = ambient_temp + heat of
    # objects nearby (heatValue * heat_scale, falling off linearly to 0 beyond heat_radius cells).
    # Deviation from 0.5 multiplies hunger: hunger * (1 + temp_hunger * 2|T - 0.5|), as in OHOL.
    temperature: bool = False
    ambient_temp: float = 0.5        # 0 = freezing, 0.5 = comfortable, 1 = scorching
    heat_scale: float = 0.1          # temperature added per heatValue point at the heat source
    heat_radius: int = 3             # cells
    temp_rate: float = 0.05          # fraction per tick by which body temperature approaches the local one
    temp_hunger: float = 2.0
    move_cost: float = 0.0           # extra hunger for a tick spent on FORWARD, as a fraction of hunger_per_tick
    turn_cost: float = 0.0           # same for TURN_LEFT / TURN_RIGHT (0 = moving is as cheap as standing still)
    brain_cost: float = 0.0          # metabolic cost of neural activity: hunger * (1 + brain_cost * mean rate of
                                     # all non-input neurons). A mean, so it prices dense firing, not brain size.


@dataclass(frozen=True)
class VisionConfig:
    columns: int = 9                 # W: retina width in ray columns
    fov_degrees: float = 120.0       # field of view spread of the columns
    range: int = 6                   # cells; maximum ray length
    appearance_dim: int = 4          # K: size of the per-object appearance vector (ADR-006)
    appearance_noise: float = 0.0    # std of Gaussian noise added to every seen/held appearance feature per tick


@dataclass(frozen=True)
class BodyConfig:
    """Optional body-sense features appended after the fixed ones [food, age, pain, held, held_app(K)].
    Every feature is named (see sensors.input_names) so genomes can be remapped when features are added."""
    taste: bool = False              # sweetness of what was eaten last tick: max(food gained, 0) / food_scale
    efference: bool = False          # one-hot copy of the agent's own previous action (NUM_ACTIONS features)
    temperature: bool = False        # body temperature in [0, 1], 0.5 = comfortable (needs WorldConfig.temperature)
    temp_change: bool = False        # change of body temperature over the last tick x 50
    skin_change: bool = False        # change over the last tick of the temperature *at the agent's cell* x 10: the
                                     # spatial gradient along its path (thermosensory neurons such as C. elegans
                                     # AFD respond to changes of the ambient temperature)


@dataclass(frozen=True)
class RegionSpec:
    """A named block of neurons (ADR-013). 'in' and 'out' are reserved and created automatically.
    Rates are non-negative: x <- (1-alpha) x + alpha max(0, tanh(h)) (ADR-002)."""
    name: str
    size: int
    alpha: float = 0.5               # leak per step (1 = no memory; 0.05 = time constant of ~20 steps)
    trace_tau: float = 0.8           # decay per step of the activity trace used by the 'trace' (STDP-like) rule
    sign: str = "mixed"              # Dale's law: 'exc' (all outgoing weights >= 0), 'inh' (all <= 0) or 'mixed'
    kwta: int = 0                    # > 0: only the k most active neurons of the region stay active each step
    norm: float = 0.0                # > 0: divisive normalisation (shunting inhibition by an implicit pool of fast
                                     # interneurons): input h / (1 + norm * mean over the region of max(h, 0))
    norm_lag: bool = False           # True: the pool of the previous step divides (inhibition lags one step, so
                                     # the onset of a stimulus passes at full strength)
    bias: float | None = None        # initial bias of every neuron; None = small random. > 0 = tonically active
    evolve_bias: bool = True         # False: bias is hard-wired (never mutated)
    receptors: tuple = ()            # broadcast neuromodulation of activity: (modulator, effect, sensitivity) with
                                     # effect 'gain' (input x exp(sensitivity * m), clipped e^+-2) or 'bias'
                                     # (input + sensitivity * m); the modulator of the previous step is used
    phase: int = 0                   # order of updating within a tick: regions of phase p read the activity that
                                     # regions of lower phase have *this* tick (and everything else from the
                                     # previous tick), so a feedforward chain senses -> ... -> motor runs within
                                     # one tick instead of one tick per layer. All 0 = fully synchronous.
    group: str = ""                  # visual grouping only (dashboard), e.g. 'hypothalamus' or
                                     # 'forebrain/basal_ganglia' for nested groups


@dataclass(frozen=True)
class ProjectionSpec:
    """Synapses from region src to region dst.
    rule: 'fixed' (no plasticity), 'hebb' (ABCD: A pre post + B pre + C post + D; with rates >= 0, A=1, B=-theta
    gives the covariance rule), 'oja' (A post (pre - post w)), 'trace' (STDP-like: A pre_trace post - C pre
    post_trace), 'delta' (A pre (teacher - post): error-driven, teacher = a region of the same size as dst).
    modulator: name of a BrainConfig.modulators entry that multiplies the update (three-factor learning);
    modulated=True is the legacy spelling for 'the first modulator'. elig_tau > 0 accumulates the rule into a
    per-synapse eligibility trace e <- elig_tau e + rule, and the modulator converts e into a weight change.
    kind='gain' makes the synapses multiplicative: the target's input is scaled by exp(sum w x) (clipped to
    e^-2..e^2), the way neuromodulators and drives change the excitability of a whole population.
    Plasticity genes (eta, A, B, C, D) are one value per projection (a cell-type-pair rule, not per synapse)."""
    src: str
    dst: str
    density: float = 0.3             # fraction of possible synapses present at init ('full' topology)
    rule: str = "fixed"
    modulated: bool = False
    eta_init: float = 0.0            # starting learning rate (evolution changes it when plastic)
    modulator: str = ""
    elig_tau: float = 0.0            # eligibility trace decay per step; 0 = no trace (immediate update)
    kind: str = "add"                # 'add' or 'gain'
    topology: str = "full"           # 'full' (random with density) or 'one_to_one' (src i -> dst i), or 'topographic' (see groups)
    w_init: float | None = None      # None: random N(0,1)/sqrt(fan_in). A value: every synapse starts there
    evolve: bool = True              # False: hard-wired: w0 and presence are never mutated
    tune: bool = False               # hard-wired (evolve=False) projections only: evolution may scale the whole
                                     # projection by one factor (EvolutionConfig.tune_*); the wiring and the sign
                                     # stay as designed, the strength is tuned
    teacher: str = ""                # rule 'delta': region providing the target activity for dst
    abcd: tuple = (1.0, 0.0, 0.0, 0.0)   # initial rule coefficients A, B, C, D
    groups: int = 0                  # topology 'topographic': src and dst split into this many aligned groups
                                     # (e.g. one group per vision column -> one tectum neuron group per column)
    dst_range: tuple = ()            # (start, stop): only these neurons of dst receive the projection (topography)
    src_range: tuple = ()            # (start, stop): only these neurons of src send it (not for src 'in': use src_select)
    decay: float = 0.0               # per step, plastic weights relax by this fraction back toward w0 (forgetting;
                                     # lifetime learning then fades unless renewed: extinction, reversal)
    src_select: tuple = ()           # src 'in' only: input-feature name patterns (fnmatch, e.g. 'vis*', 'pain')
                                     # restricting which input neurons project; several projections from 'in'
                                     # to the same region may use disjoint selections
    depression: tuple = ()           # (U, tau_rec): short-term depression; each unit of pre activity uses up a
                                     # fraction U of the synapse's resource, which recovers with tau_rec steps


@dataclass(frozen=True)
class ModulatorSpec:
    """A named, globally broadcast signal per agent (dopamine, serotonin, a raw US, ...), computed after every
    brain step and used by the next one. source 'region': scale * (mean(pos) - mean(neg) - baseline), from
    region activity (neg optional). 'world:reward' (food gained / food_scale - pain), 'world:pain',
    'world:food': taken directly from world events of the last tick (a shortcut for hard-wired taste)."""
    name: str
    source: str = "region"
    pos: str = ""
    neg: str = ""
    baseline: float = 0.0
    scale: float = 1.0
    decay: float = 0.0               # > 0: the modulator is max(new value, decay * previous value): released at
                                     # once, cleared slowly (a persistent state that outlasts its trigger)
    terms: tuple = ()                # ((region, weight), ...): adds sum(weight * mean(region)) to pos - neg; e.g.
                                     # a TD error r + gamma V(t) - V(t-1) from reward, value and value-copy regions


DEFAULT_REGIONS = (RegionSpec("hidden", 32),)
DEFAULT_PROJECTIONS = (ProjectionSpec("in", "hidden"), ProjectionSpec("hidden", "hidden"),
                       ProjectionSpec("hidden", "out"), ProjectionSpec("in", "out"))


@dataclass(frozen=True)
class BrainConfig:
    regions: tuple = DEFAULT_REGIONS          # tuple[RegionSpec, ...] excluding the reserved 'in' and 'out'
    projections: tuple = DEFAULT_PROJECTIONS  # tuple[ProjectionSpec, ...]
    out_alpha: float = 0.5           # leak of the output region
    out_phase: int = 0               # RegionSpec.phase of the output region (set above every other phase so the
                                     # action uses this tick's activity)
    w_max: float = 4.0               # plastic weights are clipped to [-w_max, w_max] (inherited, non-plastic
                                     # weights are used as they are)
    steps_per_tick: int = 1          # brain updates per world tick
    logit_gain: float = 4.0          # multiplier on output pre-activations before sampling
    action_temperature: float = 0.5  # softmax temperature; 0 means argmax
    modulator: str = "none"          # legacy: 'reward' adds ModulatorSpec("reward", "world:reward") first
    modulators: tuple = ()           # tuple[ModulatorSpec, ...]


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int = 30
    ticks_per_generation: int = 400  # ticks each generation lives
    episodes: int = 1                # independent worlds per generation. 1 = one life per genome (default,
                                     # chosen for biological plausibility, see ADR-012). >1 averages stats.
    siblings: int = 1                # individuals per genome in the world (num_agents = genomes x siblings). Each
                                     # lives one life, scattered independently; the genome's fitness is their
                                     # mean. One life is mostly luck (repeatability ~0.1), so a parent is judged
                                     # by several offspring, as in a real lineage.
    elite_frac: float = 0.125        # fraction copied unchanged
    tournament: int = 3              # tournament size for parent selection
    crossover_mode: str = "neuron"   # 'neuron': see crossover; 'blend': the child is the average of its parents
                                     # (the quantitative-genetics picture of a trait built from many small genes)
    crossover: float = 0.0           # chance that a child has two parents (each chosen by tournament). It then
                                     # takes every neuron (its incoming weights, their presence and its bias) from
                                     # one parent or the other, and every projection's rule genes and tunable
                                     # strength likewise. 0 = asexual. With noisy fitness, recombination lets a
                                     # population average out luck and combine gains from different lineages.
    mutation_std: float = 0.1        # Gaussian std on weights and biases
    weight_mutation_prob: float = 1.0  # per synapse/neuron and child: chance that its weight/bias mutates
    mask_flip_prob: float = 0.005    # per allowed synapse: toggle presence
    plastic: bool = False            # if False, plasticity genes (eta, A, B, C, D) are never mutated
    eta_max: float = 0.05            # clip for learning rates when plastic
    eta_mutation_prob: float = 0.1   # per plastic projection and child: chance that its rule genes (eta, A..D)
                                     # mutate. eta moves by eta_max * N(0, 0.3), clipped to [0, eta_max].
    record_weights_every: int = 100  # ticks between weight snapshots in the recording (must divide ticks)
    record_agents: int = 64          # the recording (dashboard) of the last generation shows this many agents,
                                     # sampled evenly over the fitness ranking (best first), in a world scaled
                                     # down to the same density; 0 or >= num_agents: the whole population
    tune_prob: float = 0.1           # per tunable projection and child: chance that its strength mutates
    tune_std: float = 0.2            # std of the log of the factor it is multiplied by
    tune_range: float = 3.0          # the strength stays within [1/range, range] x the designed value (at 10 evolution
                                     # turned the valence -> motor wiring down to ~0.15 x: free to disconnect)
    seed: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "unnamed"
    world: WorldConfig = field(default_factory=WorldConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    body: BodyConfig = field(default_factory=BodyConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(text: str) -> "ExperimentConfig":
        """Inverse of to_json (used to rebuild an earlier run's layout for warm starts). Unknown keys are dropped."""
        d = json.loads(text)
        def tup(x):
            return tuple(tup(y) for y in x) if isinstance(x, list) else x

        def mk(cls, v):
            names = {f.name for f in fields(cls)}
            return cls(**{k: tup(x) for k, x in (v or {}).items() if k in names})
        b = dict(d.get("brain", {}))
        b["regions"] = tuple(mk(RegionSpec, r) for r in b.get("regions", [])) if "regions" in b else DEFAULT_REGIONS
        b["projections"] = tuple(mk(ProjectionSpec, p) for p in b["projections"]) if "projections" in b else DEFAULT_PROJECTIONS
        b["modulators"] = tuple(mk(ModulatorSpec, m) for m in b.get("modulators", []))
        brain = BrainConfig(**{k: v for k, v in b.items() if k in {f.name for f in fields(BrainConfig)}})
        return ExperimentConfig(name=d.get("name", "unnamed"), world=mk(WorldConfig, d.get("world")),
                                vision=mk(VisionConfig, d.get("vision")), body=mk(BodyConfig, d.get("body")),
                                brain=brain, evolution=mk(EvolutionConfig, d.get("evolution")))
