# Architecture Decision Records

Read this file first. Every entry is a decision that later work must respect unless a new
entry explicitly supersedes it. Add new entries at the bottom; never edit the meaning of an
old one silently. Entries are written for a reader (human or model) with no other context.

Status legend: **Fixed** = do not change without a new ADR. **Default** = safe to change per experiment via config.

**Note (user, 2026-09-30):** these ADRs guided the initial framework build and are not sacred. New additions may
override them freely; when one does, add or amend an entry so the record stays current (see also `docs/BRAIN_EVOLUTION.md`).

---

## ADR-001 Stack: Python 3.12 + JAX, managed by uv

- **Fixed.** Code is Python. Simulation math is JAX (`jax.numpy`, `jit`, `vmap`, `lax.scan`).
- The venv lives in `.venv/` and is created with `uv` (see `docs/SETUP.md`). No conda, no poetry.
- Development runs on Windows CPU JAX. Scale runs use WSL2 + CUDA on the RTX 5060 (Blackwell, needs CUDA 12.8+ wheels).
- Why: the whole world step and all brains must be batched into a few array operations. JAX `jit` over the entire step, and `vmap` over agents, is the proven way to get millions of agent-steps per second (Craftax, JaxLife).
- Do not introduce PyTorch, Nengo, Brian2 or any spiking simulator into the core.

## ADR-002 Time is discrete ticks; neurons are rate coded; no spikes

- **Fixed.** The world advances in integer ticks. One tick is the unit of everything: hunger drain, decay timers, ages.
- The brain runs `brain_steps_per_tick` (**Default** 1) updates of a rate-coded network per world tick.
- Neurons hold a continuous, non-negative firing rate in [0, 1): `max(0, tanh(h))` (amended 2026-09-30; was signed tanh). Signed quantities live in weights and modulators, not in rates. There is no spike generation anywhere.
- Learning rules are functions of pre-activity, post-activity, per-synapse traces and a scalar modulator. STDP, if ever needed, is expressed as a trace rule on rates, not as spike timing.
- OHOL decay times are given in seconds (negative numbers mean hours). They convert to ticks by `ticks_per_ohol_second` (**Default** 1.0) and are clipped to `max_decay_ticks` (**Default** 600).

## ADR-003 The world is an OHOL-style transition system on a 2D grid

- **Fixed.** A world is: a grid of object ids, one object per cell (0 = empty ground), plus per-cell `uses_left` and `decay_timer`; and a set of agents with position, facing (4 directions), held object id, food, age, alive flag, pain, emitted sound, parent index.
- All interactions are transitions `(actor, target) -> (new_actor, new_target)` where actor is the held object (0 = empty hand) and target is the object in the cell the agent faces. This is exactly the One Hour One Life rule model, so OHOL data plugs in directly.
- Extra columns not in OHOL (for example `pain_value`) are allowed on the ruleset but must default to "no effect" so OHOL data loads unchanged.
- Agents may overlap on a cell (OHOL allows this). Objects with `blocks_walking` block movement. The grid has hard walls at its edges.
- Cell interactions from many agents in one tick are applied sequentially in agent order via `lax.scan` (deterministic, conflicts resolved first-come). Movement is applied in parallel.

## ADR-004 Rulesets are OHOL slices remapped to compact local ids

- **Fixed.** A `Ruleset` (see `life/ruleset.py`) is a table of arrays indexed by *local id* 0..M-1. Local id 0 is always "empty" (empty hand and empty ground). Each local id remembers its `ohol_id` (or a synthetic id >= 100000 for objects we invent) for names, appearance and sprites.
- The dense lookup `use_table[actor_local, target_local]` gives a transition index or -1. This is O(1) at runtime and fits in memory for slices up to a few thousand objects.
- Complexity is increased by slicing a bigger subgraph of OHOL (`life/ohol.py: slice_ruleset`), never by writing a new environment. Small hand-made rulesets exist only for tests.
- OHOL conventions we rely on (verified against `data/ohol` on 2026-09-18, see `docs/OHOL_FORMAT.md`): actor -1 = time decay; actor 0 = empty hand; target -1 is treated as empty ground (local 0); `_LT`/`_L` suffix = transition that applies on the target's last use; `_LA` = last use of actor (currently ignored; documented gap).
- Eating is not a transition in OHOL. It is our `EAT` action: consumes a held object with `food_value > 0`. Any leftover object OHOL would produce (bowl, plate) is dropped for now (documented simplification).

## ADR-005 The observation/action contract is fixed and append-only

- **Fixed.** Every brain receives `obs = {"vision": [W, F], "body": [B], "sound": [S]}` and returns one integer action from `life/actions.py`.
- Actions: `NOOP, FORWARD, TURN_LEFT, TURN_RIGHT, USE, EAT, VOCALIZE`. `USE` does pick-up, drop and crafting depending on the transition table. New actions may only be **appended**, never reordered.
- Vision is egocentric: `W` columns spread over a field of view centred on the facing direction; each column reports the first non-empty cell along its ray up to `range`. Features per column: `[hit, 1-dist/range, is_agent, is_wall, appearance(K)]`. See ADR-006 for appearance.
- Body: `[food/max_food, age/max_age, pain, held_present, held_appearance(K)]`.
- Sound: `[heard]` (sum of vocalizations of agents within `hear_radius` last tick). Present from day one even when unused so later experiments do not change the input size.
- Channels may grow in width (more columns, bigger K) per experiment config, but the *names and order* of channels and features are fixed.

## ADR-006 Object appearance is a deterministic pseudo-random vector per OHOL id

- **Fixed.** Brains never see object ids. They see `appearance[K]`, a unit vector generated from a hash of the object's `ohol_id`, so the same object looks the same in every experiment and slice. Agents have one reserved appearance. `K` is **Default** 4.
- A sprite-derived appearance (colour histogram of the OHOL sprite) can be added later as a *second* option behind the same interface. Raw pixels are a possible third sensor. None of these change the brain contract.

## ADR-007 Brain substrate: one masked recurrent rate network, extended by adding blocks

- **Fixed.** A brain is a state vector `x[N]` and a weight matrix `W[N,N]` with a binary `mask[N,N]`. Neuron order: inputs (clamped to obs), hidden, outputs (action logits). One step: `x <- (1-a) x + a max(0, tanh(W^T x + b))` on non-input neurons.
- Plasticity is per synapse: `dW = eta * mod * (A pre post + B pre + C post + D)`, where `eta, A, B, C, D` are genome arrays of shape `[N,N]` (the ABCD Hebbian family) and `mod` is a scalar modulator. `mod = 1` gives pure Hebbian learning; a reward-prediction-error signal gives three-factor reinforcement learning. Weights are clipped to `[-w_max, w_max]`.
- Genome = `W0, mask, b, eta, A, B, C, D` (all arrays). Evolution mutates them (`life/evolution.py`). Lifetime learning changes `W` only; `W` resets to `W0` at birth.
- **How "building on top" works:** the next complexity level adds neurons (a block of rows/columns) and possibly a *module* that owns a sub-range of neurons and provides its own update or modulator (e.g. a basal-ganglia module that outputs `mod`). The previous level's neurons, weights and rules are kept untouched. Never replace the substrate; extend the vector.
- The `Brain` interface in `life/brain.py` is: `init(genome) -> state`, `step(genome, state, obs, mod, key) -> (state, action)`. Modules must respect it.

## ADR-008 Population, worlds and evolution

- **Fixed.** A population of `N` genomes is simulated as `N` agents living in one shared world (multi-agent from day one). Independent replicate worlds, if wanted, are an extra `vmap` axis on top.
- Generational evolution (all agents born at tick 0, evaluated for `T` ticks, then selection) is the **Default** for early experiments. Continuous birth near a parent (OHOL-style lineages) is planned; the `parent` field exists in the state from day one so cross-generational experiments do not change the state layout.
- Fitness is defined per experiment as a function of the recorded per-agent stats (ticks alive, food eaten, pain).
- Selection: elitism + tournament + Gaussian mutation with per-array mutation rates (`life/evolution.py`). Deterministic given the PRNG key.

## ADR-009 Recording and viewing are first class

- **Fixed.** Every experiment run writes to `runs/<experiment>/<timestamp>/`: `config.json`, `fitness.csv`, `best_genome.npz`, and a `recording.npz` of the final generation (grid, agents and the best agent's neuron activations per tick).
- `python -m life.viewer <run dir>` replays a recording with matplotlib. Rendering uses appearance colours; OHOL sprites are optional and only for the viewer, never for the brain.

## ADR-010 Experiments are config plus a fitness function, nothing else

- **Fixed.** An experiment file in `life/experiments/` may only: build a ruleset slice, set config dataclasses, define fitness, and call `life.run.run_evolution`. If an experiment needs a new mechanic, the mechanic goes into the core with a config switch that defaults to off, plus a test.

## ADR-012 One life per genome; experiments warm-start from earlier runs

- **Fixed (amended 2026-09-18 by the user).** Each genome lives **one** life per generation (`EvolutionConfig.episodes` **Default** 1). Averaging fitness over several worlds (`episodes > 1`, still supported as a vmap axis) was tried and helps evolution, but the user rejected it as biologically implausible and accepts the noisier signal; noise is to be handled by other means (population size, lifespan, environment design) which the user will tune.
- **Fixed.** Every run saves its final population (`population.npz`). `run_evolution(init_population=load_population(run_dir))` starts the next experiment from it. This is the evolutionary counterpart of ADR-007's "build on top": each level of complexity starts from the population of the previous level, never from scratch. The brain layout (inputs, hidden, outputs) must match; widen it by padding, not by re-initializing.
- Experiments should also keep the payoff such that the *previous* level's behaviour still pays (exp02 keeps random foraging profitable), so that the new skill is an increment, not a cliff.
- Anything an experiment randomizes to force lifetime learning (which berry is poison, where food is) must be randomized **per episode within a generation**, never per generation, so that a fixed inherited preference cannot score well on average. `rules_for_generation` may return rulesets stacked along the episode axis for this.
- Plasticity genes mutate sparsely (`eta_mutation_prob`): dense learning-rate noise turned on half the synapses at once and destroyed inherited behaviour.

## ADR-013 Brain architecture: named regions and projections with their own learning rules

- **Fixed.** Supersedes the "one hidden block" part of ADR-007; the single-matrix substrate stays. A `BrainConfig` declares `regions` (name, size, leak `alpha`, `trace_tau`) and `projections` (src, dst, connection `density`, `rule`, `modulated`, `eta_init`). `brain.build_layout` compiles them into constant per-synapse arrays (`allowed`, `rule`, `modulated`, ...) that sit beside the evolvable genome. Regions `in` and `out` are reserved and sized by the contract.
- Rules, selected per synapse by the projection: `fixed` (no plasticity), `hebb` (A pre post + B pre + C post + D), `oja` (A post (pre - post w), self-normalizing), `trace` (A pre_trace post - C pre post_trace; STDP-like order sensitivity using low-pass activity traces). `modulated=True` multiplies the update by the modulator (three-factor learning).
- Modulator: `BrainConfig.modulator` is `none` (mod = 1) or `reward` (food gained minus pain in the previous tick, a world-provided taste signal). A basal-ganglia module that computes its own reward prediction error will be a region whose update writes `mod`; the hook is the `mod` argument of `brain.step`.
- Evolution only toggles plasticity on synapses whose rule is not `fixed`, and only flips mask bits inside declared projections. Genome shapes depend only on the total neuron count, so a population can be reused when regions are appended at the end (pad the arrays; do not reorder).
- This is how "building on top" is done: keep the old regions and projections, add new regions and projections, warm-start from the previous population.

## ADR-014 Recording, dashboard and lab are the primary tools for understanding

- **Fixed.** A recording holds, per tick, the grid, every agent's position, facing, alive flag, held object, food, pain, action, modulator and full activation vector, plus weight snapshots of all agents every `record_weights_every` ticks and the genome's `w0`, `eta`, `mask`. Activations of input neurons *are* the observation, so the retina and body inputs can be reconstructed from the recording.
- `life/dashboard.py` writes a self-contained `dashboard.html` (plain HTML + JS, data embedded as base64 typed arrays, OHOL sprites embedded as PNG) into the run directory. It shows the world with sprites, the focused agent's retina, body inputs, activations over time by region, and its weight matrix (w at snapshot, w - w0, w0, eta) with region boundaries and hover details, plus the fitness curve. No server, no build step; open the file in a browser.
- `life/lab.py` (`Lab`) steps a world by hand from Python or Jupyter, overrides actions, exposes observations and brain state per agent, and exports the stepped history to the same dashboard.
- The dashboard's first panel is an **architecture view**: one block per region (cells = the focused agent's activations at the current tick), one arrow per projection (colour = rule, dashed = modulated, width = summed |w| per target neuron at the current snapshot, loops = recurrent). Clicking a block shows that region's activations over time and its in/out projections with stats; clicking an arrow shows its weight block (w, w - w0, w0, eta). Blocks can be dragged; positions are remembered per experiment in browser localStorage. Runs without `brain.projections` in `config.json` get projections inferred from nonzero weight blocks.
- OHOL sprites are composited approximately (`life/sprites.py`) for the viewer only; the brain never sees them (ADR-006).
- The world background is the OHOL grassland ground texture (`data/ohol/ground/ground_0.tga`), tiled one texture per 4x4 cells as in the game. Single biome only for now; viewer only. Falls back to a flat colour when `ground/` is not checked out.
- Agents are drawn as OHOL player bodies (the 22 spawnable `person>0` objects composited at age 25, agent i uses body i mod 22). OHOL bodies only face left/right, so the sprite is flipped for left, keeps its last horizontal facing while moving up/down, and a chevron at the cell edge (navy, red = focused) shows the true facing. Dead agents are faded. The held item is drawn at hand height in front of the body (approximating OHOL heldOffset), or as a legend-coloured square inside the triangle. Unchecking "sprites" restores the triangle view.
- World, activations and weights zoom and pan (wheel at the cursor, drag, double-click resets; activations zoom time by default, shift+wheel zooms neurons). Heatmaps are rendered once at one pixel per value to an offscreen canvas and scaled up without smoothing; the weight image is only rebuilt when mode, snapshot or focus changes.
- Playback runs on wall-clock time: speed Nx = N game seconds per real second (one game second = `ticks_per_ohol_second` ticks, so 1x = 1 tick/s by default). During playback and stepping, zoomed views follow: the world keeps the focused agent out of the outer 20% of the view, the activation plot does the same for the current tick.

## ADR-011 Documentation duties for every change

- **Fixed.** Any change to a **Fixed** decision needs a new ADR here. Any new config field needs a docstring stating its default and unit (ticks, cells, fraction). `docs/STATUS.md` tracks stage status (overview) and `docs/STAGE_LOG.md` the details.

## ADR-015 Brain v2: rectified rates, Dale's law, per-projection options, named modulators (2026-09-30)

- Supersedes the relevant parts of ADR-002, ADR-007 and ADR-013. Rates are `max(0, tanh(h))`. Regions may be
  `exc`/`inh`/`mixed` (Dale's law; w stores magnitudes for signed regions). Projections choose rule (`fixed`,
  `hebb`, `oja`, `trace`, `delta`), modulator (by name), eligibility trace, decay toward w0, short-term depression,
  kind (`add`/`gain`), topology (`full`/`one_to_one`/`topographic`), input-feature selection (`src_select`),
  target sub-range (`dst_range`), and hard-wiring (`w_init`, `evolve=False`).
- Modulators are named and computed from region activity after each step (or from the world as a shortcut).
- Plasticity genes (eta, A, B, C, D) are one value per projection, not per synapse.
- Genomes carry across layouts by region and input-feature names (`brain.remap_genomes`); new evolvable synapses
  start at 0. See `tests/test_brain_v2.py`, `docs/STAGE_LOG.md`.

## ADR-016 New cell types get a fixed meaning; neuromodulation is broadcast through receptors (2026-09-30)

- Decided with the user: a module added by a stage has hard-wired defining inputs and outputs (its meaning), so
  later stages can build on it; evolution tunes the rest. Hard-wiring individual neurons is fine when the stage is
  about that neuron.
- Neuromodulators act on activity through `RegionSpec.receptors` (gain or bias, one sensitivity per region) and
  on plasticity through `ProjectionSpec.modulator`. `gain` projections remain for point-to-point cases.
- `RegionSpec.group` groups regions for visualisation only (nested with '/').
