# Stage log: building the brain up, stage by stage

Progress through the sequence of `docs/BRAIN_EVOLUTION.md`. Newest at the bottom of each section.
Each stage lists: what was added, the world, the runs, the numbers that matter, and what I conclude.
Run a stage yourself with `python -m life.experiments.stages <key> [--control]`; compare runs with
`python -m life.compare <experiment or run dir> ...`. Every run directory has a `dashboard.html` whose
first panel shows the architecture of that stage.

How to read the comparisons: "main" = the stage's new brain, "control" = the parent stage's brain in the same
new world, started from the *same* population (the parent's final population). Both have one life per genome
(ADR-012), so single generations are noisy; I compare averages over the last 25 generations. "first/last"
are averages over the first and last 25 generations of a run.

---

## Core changes made before the stages (2026-09-30)

All documented in `docs/BRAIN_EVOLUTION.md` section 2 (M1-M10) and tested in `tests/test_brain_v2.py`.

- **Rates are non-negative**: `max(0, tanh(h))` (0 at rest). Signed quantities live in weights and modulators.
- **Dale's law** per region: `RegionSpec(sign="exc" | "inh" | "mixed")`. For signed regions w stores the
  magnitude and the effective weight is sign * w; evolution and plasticity keep magnitudes >= 0.
- **Hard-wiring**: `ProjectionSpec(w_init=..., evolve=False, topology="one_to_one")`, `RegionSpec(bias=...,
  evolve_bias=False)`. The user's rule of thumb: structure may be hard-wired, individual neurons only when the
  stage is about that neuron.
- **Input-feature selection**: `ProjectionSpec(src_select=("pain", "vis*"))` so that e.g. only pain nerves
  reach a nociceptive region. Every input neuron is named (`sensors.input_names`); vision columns are named by
  angle, so retinas of different resolution line up.
- **Named neuromodulators** (M2): `BrainConfig(modulators=(ModulatorSpec("da", pos="snc", baseline=...),))`,
  each computed from region activity (or taken from the world as a shortcut) after every step; each
  projection picks which one gates its plasticity (`modulator="da"`).
- **Eligibility traces** (M1): `elig_tau` per projection; the modulator converts the trace into a weight change.
- **Gain projections** (M3): `kind="gain"` multiplies the target's input by exp(sum w x), clipped to e^+-2.
- **Short-term depression** (M4): `depression=(U, tau_rec)` for habituation.
- **k-WTA** (M5): `RegionSpec(kwta=k)`. **Delta rule** (M6): `rule="delta", teacher=<region>`.
- **Efference copy, taste, temperature** as optional body senses (`BodyConfig`).
- **Plasticity genes are per projection** (one eta/A/B/C/D per cell-type pair), not per synapse. This is both
  more realistic and far less destructive to mutate (exp02's per-synapse eta noise wrecked inherited behaviour).
- **Warm starts across layouts**: `load_population(run, exp)` remaps a population by region and input names;
  every new evolvable synapse starts at 0, so inherited behaviour is unchanged until evolution uses the new part.
- **World**: delayed sickness (`sickness_delay`), taste, OHOL-style temperature (body temperature drifts toward
  ambient + nearby heatValue; deviation from comfortable multiplies hunger, as in OHOL), per-object eat counts
  per life half (`stats["eats"]`) for learning curves.
- **Dashboard**: architecture view (blocks = regions, green = excitatory, red = inhibitory; arrows = projections,
  dashed = modulated plasticity, dotted = gain), all modulators per agent, named body inputs.

---

## World fix (2026-09-30): look-alikes must look alike

The user pointed out that colour variants of the gooseberry bush had *unrelated* appearances: appearance is a
hash of the object id (ADR-006), so "Blue Gooseberry Bush" shared nothing with the gooseberry bush. That made
generalisation impossible and is unrealistic for look-alikes. Now (`stages.lookalike_appearance`) a variant's
appearance = 0.8 x original + 0.6 x a colour direction, with colour directions spread orthogonally: cosine
similarity 0.8 to the original, 0.64 between most variants, 0.28 between "opposite" colours. Everything below
the line "v2" uses this world; the earlier chapter-1 runs are archived in `runs/_archive_v1/` and summarised
as v1. Also fixed: two runs starting in the same second wrote into the same run directory.

## Chapter 1: steering (v1 = hash appearances, archived; v2 = look-alike appearances)

### 1.0 Reflexive steering (baseline, from scratch)

- **Brain**: `ganglion_e` (16 excitatory) and `ganglion_i` (8 inhibitory) interneurons between the senses and 7
  motor neurons, plus direct sensor->motor reflex arcs. All evolved, no plasticity.
- **Senses**: 5 vision columns over 120 degrees, range 5 (coarse "eyes"), body state, sound.
- **World**: 32x32, 64 agents, four kinds of berry bush (the OHOL gooseberry plus three colour look-alikes; 6
  berries each, regrowing 500 ticks after being emptied) and wild onions; 1000-tick lives. Economics check:
  standing at one bush yields about 12 berries per life, walking between bushes 30+, so foraging pays and the
  "stand still at a bush" trap of early exp01 does not apply.
- **Runs**: `runs/s1_0_steering/20260930-145742` (200 generations from random genomes) continued as
  `20260930-150008` (200 more). About 2 minutes per 200 generations on the CPU.

| | mean fitness | food eaten | survivors of 64 |
|---|---|---|---|
| generations 0-19 | 7.5 | 3.2 | 0.4 |
| generations 180-199 of the first run | 41.6 | 36.0 | 4.9 |
| last 20 of the continuation (generation 400) | 51.4 | 45.6 | 7.2 |

- **Conclusion**: reflexive foraging evolves with the new substrate (rectified rates, Dale's law). This
  population is the root of the lineage.
- **Lesson from a first attempt**: with only one food type (the real gooseberry) the population became a
  *specialist* that ignores everything else. Adding a poisonous look-alike in 1.1 then had nothing to teach,
  because the look-alike was never eaten in the first place (0 poison eaten), and swapping half the good bushes
  for poison ones halved the food supply and crashed foraging (59 -> 12 food eaten). The fix: evolve a
  generalist on several edible types first, and *add* harmful types without removing good food.

### 1.1 Valence neurons

- **Added**: `valence_app` and `valence_av` (3 excitatory neurons each). Hard-wired, innate input from the
  unconditioned senses (taste -> appetitive, pain -> aversive, weight 3, not evolvable); evolved input from vision
  and the held item; output to the ganglion and the motor neurons. New body sense: `taste`.
- **World**: 6 berry types; types 4 and 5 (Purple, White) are *always* poison (-2 food and 1 pain per berry), so
  avoidance can be inherited. Spawn density raised 0.08 -> 0.12 so the number of good bushes equals 1.0.
  Fitness: food eaten - pain + survival bonus.
- **Runs** (150 generations each, both warm-started from the 1.0 population):
  `runs/s1_1_valence/20260930-150226` and `runs/s1_1_valence_control/20260930-150226`.

| last 25 generations | fitness | food eaten | pain | poison fraction |
|---|---|---|---|---|
| main (valence) | 39.6 | 35.5 | 0.72 | 0.111 |
| control (1.0 brain) | 40.2 | 35.5 | 0.57 | 0.089 |

- **Result**: no difference. The inherited foragers already avoid the new types quite well (11-12% of berries
  eaten are poison where indiscriminate eating gives 33%), because their evolved preference for the four known
  appearances generalises poorly to new ones. There is little left for valence to add.
- This matches the plan's expectation that a valence bottleneck helps *evolvability* rather than final
  performance. Test running: both architectures evolved from scratch in the 1.1 world, 2 seeds each.

### v2 results (look-alike world)

#### 1.0 v2
- `runs/s1_0_steering/20260930-152119`, 400 generations from random genomes. Food eaten 10.0 (first 25
  generations) -> 46.7 (last 25), survivors 3.0 -> 6.3 of 64. Progress is much faster than in v1 (fitness 31.7 by
  generation 40 vs about 7), because similar-looking bushes now generalise.

#### 1.1 v2 valence
- `runs/s1_1_valence/20260930-152716` and `..._control/20260930-152716`, 150 generations each from the 1.0 v2
  population.

| last 25 generations | fitness | food eaten | pain | poison fraction |
|---|---|---|---|---|
| main (valence) | 50.1 | 46.7 | 1.89 | 0.187 |
| control (1.0 brain) | 38.4 | 35.4 | 1.82 | 0.221 |

- The poison look-alikes are now tempting: both runs start at a poison fraction of about 0.26 (eating blind
  gives 0.33) and both learn to avoid them partly.
- Lesion study of the main population (6 worlds): silencing `valence_app` or `valence_av` costs only ~2 fitness
  points (51.8 -> 49.8 / 50.0), while silencing the ganglion destroys foraging (-> 8-10). So the valence module
  is used a little, and most of the 12-point main/control gap is probably run-to-run noise (one seed each).
  Replicate seeds are queued.

#### 1.2 v2 drives (first cold world, archived in `runs/_archive_v2a/`)
- Ambient temperature 0.30, springs heat 0.12/radius 3, cold multiplies hunger by at most 1.4.

| last 25 of 150 generations | fitness | food eaten | pain | poison fraction |
|---|---|---|---|---|
| main (hypothalamus + neurosecretory) | 36.6 | 34.0 | 1.29 | 0.179 |
| control (1.1 brain) | 33.5 | 30.7 | 0.97 | 0.157 |

- Neither run improved over its 150 generations. Lesions: silencing `hypothalamus` or `neurosecretory` changes
  nothing, and mean body temperature stays at 0.38 (ambient 0.30): the agents do not thermoregulate. The cold
  costs the inherited population about 140 ticks of life, but warming up needed ~20 ticks of standing next to a
  spring, so the behaviour was hard to discover. Retuned (ambient 0.25, heat 0.15 over radius 4, warming rate
  0.1/tick, hunger x1.75 when cold): now a quarter of the cells are comfortable and passing near a spring warms
  quickly. Stages 1.3 and 1.4 were stopped because they built on the old 1.2.

#### 1.5 associative learning: debugging before evolving (2026-09-30)
Before spending evolution time, I probed the learning rule directly: the same population run with learning
switched off vs on (no evolution in between). Findings, in order:
1. **No effect at all at first** (late-life poison fraction ~0.23-0.29 with learning off, 1x, 3x, 10x).
   Weight inspection: the aversive drive for this life's poison rose by only ~+0.05 over a life, against inherited
   differences of ~0.25. Also nothing inherited turned valence activity into movement (lesions of the valence
   neurons cost ~2 fitness points). 
2. **Fix, structural (1.1 changed)**: an innate valence -> motor pathway, as in *C. elegans* (aversive
   interneurons drive turns/reversals, appetitive ones forward movement and feeding): `valence_av -> TURN_LEFT,
   TURN_RIGHT`, `valence_app -> FORWARD, USE, EAT`, hard-wired weight 2. With that, strong learning halves pain
   (3.2 -> 1.4) but *also* cuts food eaten (26.6 -> 19): the aversion over-generalised.
3. **Cause of over-generalisation #1**: the plastic synapses included generic vision features ("something is
   there", nearness, wall). Pain after eating taught "avoid everything". Fix (1.1 changed): identity features
   (appearance, seen or held: the analogue of an odour) and generic features are separate projections, and only
   identity can become a conditioned stimulus.
4. **Learning rate**: the eligibility trace keeps integrating while a bush stays in view, so a single poisoning
   changed the weights by a total of 7.5; a rate of 0.2 slams weights into their limits, 0.01-0.08 still shows no
   benefit. Evolution may tune it (learning-rate cap raised to 0.5 per stage).
5. **Hypothesis being tested now**: all berry look-alikes share 80% of their appearance. US-gated Hebbian learning
   is not error-driven, so it accumulates the *shared* component: good berries push everything berry-like toward
   appetitive, poison pushes everything toward aversive, and the discriminating colour component is a small
   remainder. Probe with much more distinct look-alikes is running. If confirmed, this is a result in itself:
   plain conditioning handles distinctive stimuli (as in real conditioning experiments, a distinctive odour),
   while telling similar stimuli apart needs error-driven learning (Rescorla-Wagner / dopamine prediction error,
   chapter 2). The 1.5 world would then use distinct stimuli, and the look-alike discrimination becomes the
   test case for stage 2.5.

After this, the whole chapter-1 chain (1.1-1.6) is re-run in one go on the corrected 1.1.

#### Design change after discussion (2026-09-30): cell types with fixed meaning, receptors, groups
- **Why hard-wire meaning.** A region with only evolvable connections is mathematically just more hidden neurons;
  evolution routes behaviour through whatever neurons exist (lesions of 1.1 v2: valence barely used, ganglion
  essential), and the resulting "good/bad" code is anonymous and lineage-specific. A genome cannot specify a
  learning rule onto "whichever ganglion neurons mean bad". So a new cell type gets a *fixed meaning*: innate
  inputs from what defines it and innate outputs to what it does. Evolution tunes everything else. This is how
  genomes work too: cell types and their connection rules, not individual synapses.
- **Receptors (broadcast neuromodulation).** Neuromodulators were broadcast only for plasticity; effects on
  activity used point-to-point `gain` synapses. Now `RegionSpec(receptors=(("5ht", "gain", 2.0),))`: a named
  modulator (mean activity of its source nucleus) scales the gain or bias of every neuron of the target region,
  with one sensitivity per cell type, like volume transmission onto receptors.
- **Groups.** `RegionSpec(group="forebrain/basal_ganglia")` is visual only: the dashboard draws labelled boxes
  around modules and lays out each top-level group as a card; modulators are drawn as diamonds (dashed line from
  the source nucleus, dotted lines to receptor-bearing regions).
- **Chapter 1 now** (see `life/experiments/stages.py` for exact wiring):
  - 1.1 valence: `valence_app` (taste in; FORWARD, USE, EAT out), `valence_av` (pain in; turns out, and an
    inhibitory `no_feed` pair that suppresses USE/EAT), evolved input from identity and generic features.
  - 1.2 hypothalamus: `hunger` (fires below ~2/3 food) with hunger receptors on `valence_app` (appetite rises
    with need); `cold` (fires below temperature ~0.4) with cold receptors on `warm_seek`, an evolvable
    heat-seeking pathway that is nearly silent when warm.
  - 1.3 affect: `raphe` (serotonin, driven by taste, slow) -> 5ht receptors on `dwell` (turning: local search);
    `pdf` (roaming neuropeptide, driven by hunger, slow) -> pdf receptors on `roam` (forward: long runs).
- **1.5 probe with this circuit** (1.0 population, no evolution): learning now clearly acts (pain 2.93 -> 1.58 at
  eta 0.2) but over-generalises (food eaten 16.1 -> 11.8, fitness 13.1 -> 10.3; the poison fraction barely drops).
  This is the predicted weakness of non-error-driven conditioning on look-alikes. Evolution gets to tune the
  per-projection learning rates and rule coefficients (start 0.05, cap 0.5) in the chain run now in progress.

### v3 results (cell types with fixed meaning)

#### 1.1 v3 valence: now clearly better than the control
- `runs/s1_1_valence/20260930-171406` and `..._control/20260930-171406`, 200 generations from the 1.0 v2 population.

| last 25 of 200 generations | fitness | food eaten | pain | poison fraction |
|---|---|---|---|---|
| main (valence with innate motor meaning) | 53.1 | 48.9 | 1.18 | 0.124 |
| control (1.0 brain) | 39.7 | 36.0 | 1.47 | 0.188 |

- The main run starts *lower* (fitness 35 vs 44 in the first 25 generations: the new innate wiring disturbs the
  inherited behaviour) and overtakes the control.
- Lesions (6 worlds): intact 49.5; without `valence_app` 7.9, without `valence_av` 7.1, without `no_feed` 8.6,
  without `ganglion_e` 15.0. The valence cell types are now essential (before the fixed meaning: -2 points).
- New tool used for this: `python -m life.experiments.stages respond <run>` (activity of each region and action
  probabilities for each object seen ahead or held).

#### 1.2 v3 drives, first attempt: failed, redesigned
- With `hunger` (receptors on appetite) and `cold` gating an evolvable `warm_seek` pathway: main 35.1 vs control
  39.8 fitness, mean body temperature 0.38 in both: no thermoregulation. `warm_seek` would have had to evolve
  "hot spring appearance -> approach" from nothing.
- Redesign, as C. elegans does thermotaxis: a new body sense `temp_change` (AFD-like neurons respond to changes)
  and two innate cold-gated interneurons: `warm_run` (cold AND warming -> FORWARD), `warm_turn` (cold AND
  cooling -> turn), i.e. run-and-tumble up the temperature gradient. The cold threshold was raised so that `cold`
  fires below ~0.47. Probe without evolution (1.1 population, 6 worlds): lifetime 355 -> 387 ticks, food eaten
  and fitness unchanged, mean temperature 0.42 -> 0.40 (runs keep agents moving through warm zones instead of
  parking at a spring). Chain 1.2 -> 1.6 is running on this design.

#### 1.2 v3b drives with thermotaxis
- `runs/s1_2_drives/20260930-172708` vs `..._control/20260930-172707`, 200 generations from the 1.1 v3 population.

| last 25 of 200 generations | fitness | food eaten | lifetime | mean body temp | poison fraction |
|---|---|---|---|---|---|
| main (hunger, cold, thermotaxis) | 37.2 | 34.5 | 347 | 0.390 | 0.116 |
| control (1.1 brain) | 36.6 | 34.4 | 333 | 0.372 | 0.158 |

- A small edge within noise; neither run improves over its 200 generations (the cold world is harsh: lives last
  about a third of the 1000 ticks).
- Lesions (6 worlds): intact 35.7; without `cold` 26.4, without `warm_run` 24.7, without `warm_turn` 35.2, without
  `hunger` 34.8. So the cold-gated run neuron became essential, but the lesioned agents are *not* colder (0.40 vs
  0.38). Evolution co-opted "cold -> keep running" as a general forward drive for foraging (agents are nearly always
  somewhat cold, so it is almost always on). The hunger modulator is barely used. An honest co-option story, not
  the designed thermoregulation.

#### 1.3 v3b affect (serotonin dwell / PDF roam, broadcast via receptors), patchy food
| last 25 of 200 generations | fitness | food eaten | lifetime | poison fraction |
|---|---|---|---|---|
| main | 33.0 | 30.8 | 328 | 0.165 |
| control (1.2 brain) | 31.0 | 28.0 | 335 | 0.081 |

- Lesions (activity silenced, 6 worlds): intact 31.1; without `raphe` 25.5, `dwell` 26.3, `pdf` 29.0, `roam` 29.0.
  The dwell/roam system is used, serotonin/dwelling most. (Earlier lesion tool bug: it only removed outgoing
  synapses, so a broadcast nucleus kept acting through its modulator; lesions now also silence activity.)
- **Both runs got worse over the 200 generations** (38.7 -> 33.0 and 36.0 -> 31.0). Hypothesis: mutational load:
  every weight of every child mutated by N(0, 0.1) each generation, which with noisy one-life fitness erodes
  fine-tuned behaviour faster than selection keeps it. Test (`runs/s1_3_affect_mp01/20260930-173909`, same start,
  only 10% of weights mutate per child): fitness 36.6 -> 36.3 (no decline), lifetime 326 -> 365. So from here on
  warm-started stages use `weight_mutation_prob=0.1` (`--mutation-prob`); scratch runs keep 1.0.

#### 1.4 v3b habituation (depressing appetitive synapses), OHOL 'yum' variety bonus
| last 25 of 200 generations | fitness | food eaten | lifetime | poison fraction |
|---|---|---|---|---|
| main | 26.8 | 24.5 | 349 | 0.172 |
| control (1.3 brain) | 25.2 | 22.2 | 351 | 0.097 |

- Small edge (+1.6 fitness, +2.3 food); within noise with one seed. The variety world is harder for both
  (repeated foods are worth half).

#### 1.5 / 1.6 v3b, and why the worlds were rebalanced
| last 25 of 200 generations | fitness | food eaten | lifetime | poison fraction |
|---|---|---|---|---|
| 1.5 main (US-gated conditioning) | 16.1 | 14.5 | 305 | 0.283 |
| 1.5 control (1.4 brain) | 12.6 | 11.0 | 304 | 0.339 |
| 1.6 main (+ forgetting, mid-life reversal) | 16.7 | 15.1 | 305 | 0.277 |
| 1.6 control (1.5 brain) | 13.9 | 12.5 | 295 | 0.312 |

- Main beats control in both, but the lesion test says it is not the learning: the 1.5 population with all
  learning rates set to 0 scores the same (17.0 vs 16.9). Evolution *raised* the CS learning rates
  (0.05 -> 0.12 appetitive, 0.10 aversive) and weakened aversive learning (B -1 -> -0.43): learning is kept but is
  neutral in this world.
- **Diagnosis: the worlds had become too harsh to learn in.** Every stage added a hardship (cold, patches, yum
  penalty, an extra poison) and nothing compensated: fitness fell 53 (1.1) -> 37 -> 33 -> 26 -> 16 (1.5), lives
  shrank to ~300 of 1000 ticks, and an agent ate only ~5 berries per life, leaving almost no chance to learn from
  a poisoning and then use it. This violates the plan's own principle (ADR-012: the previous behaviour should keep
  paying, so a new skill is an increment, not a cliff).
- **Calibration** (`scratchpad calib.py` logic: the 1.1 population, no evolution, 6 worlds per variant):

| world | lifetime | food eaten | berries per agent |
|---|---|---|---|
| 1.1 | 532 | 49.0 | 9.3 |
| 1.2 before (density 0.12, cold x1.5) | 347 | 36.6 | 6.9 |
| 1.2 now (density 0.16, cold x1.0) | 442 | 49.3 | 9.0 |
| 1.3 now (10 patches of radius 5, density 0.18) | 424 | 54.2 | 9.8 |
| 1.5 now (+ yum 0.3, density 0.22) | 343 | 26.8 | 9.2 |

  (1.5 stays lower for a naive population because a third of the berries are this life's unknown poison: that is
  the learning opportunity.) Chain 1.2 -> 1.6 re-running on these worlds with `--mutation-prob 0.1`.

### v4: calibrated worlds, 10% weight mutation per child (seed 0)
Chain `runs/logs/chain_v4.log`, 200 generations per run; `python -m life.experiments.stages summary <key>`.
With the lower mutation rate both main and control now *improve* under selection (e.g. 1.2 control 50.8 -> 58.3),
which confirms the mutational-load diagnosis.

| last 25 of 200 generations | main fitness | control fitness | main / control poison | main / control lifetime |
|---|---|---|---|---|
| 1.2 drives | 51.9 | 58.3 | 0.165 / 0.092 | 373 / 420 |
| 1.3 affect | 59.1 | 58.3 | 0.126 / 0.147 | 421 / 384 |
| 1.4 habituation | 48.8 | 53.6 | 0.124 / 0.095 | 416 / 470 |

- 1.2: the drive brain keeps agents warmer (0.433 vs 0.411) but eats more poison. Plausible cause: hunger raises
  the gain of appetite for everything, poison look-alikes included. 1.3 ties; 1.4 control ahead.
- Single seeds; the gaps are of the size that flipped sign between earlier versions. Second seeds are queued for
  every chapter-1 comparison (`stages replicate <key> --seeds 1`).

#### 1.5 v4 association (calibrated world)
| last 25 of 200 generations | fitness | food eaten | pain | poison fraction | lifetime |
|---|---|---|---|---|---|
| main (US-gated conditioning) | 32.8 | 32.2 | 3.08 | 0.296 | 365 |
| control (1.4 brain) | 32.3 | 32.2 | 3.43 | 0.310 | 348 |

- Tie. Decisive lesion (8 worlds): the main population scores 30.4 intact and **33.8 with learning switched
  off**. Bilaterian-style US-gated Hebbian conditioning does not pay on look-alikes even in a world with enough
  learning opportunities (10 berries per life) and after evolution tuned its rates. Consistent with the
  over-generalisation diagnosis.
- Next: side experiment `x.td` (1.6 brain + the TD critic of 2.5, dopamine teaching instead of the raw US), same
  world, against the 1.6 brain, to test whether error-driven learning fixes it before building chapter 2.

#### 1.6 v4 reversal
| last 25 of 200 generations | fitness | food eaten | pain | poison fraction | lifetime |
|---|---|---|---|---|---|
| main (conditioning + forgetting) | 30.8 | 31.1 | 3.75 | 0.334 | 345 |
| control (1.5 brain) | 30.4 | 29.4 | 2.49 | 0.271 | 344 |

- Tie in fitness; the main brain eats *more* poison. Same picture as 1.5: in this world the bilaterian-style
  conditioning does not help. Chapter 1 summary: the fixed-meaning valence circuit (1.1) is the one clear win;
  drives, affect and habituation give ties or small losses with one seed; conditioning fails on look-alikes.

#### x.td side test: TD critic + dopamine teaching (1.6 world)
| last 25 of 200 generations | fitness | food eaten | pain | poison fraction |
|---|---|---|---|---|
| main (1.6 + TD critic, dopamine teaches) | 30.0 | 29.6 | 2.96 | 0.302 |
| control (1.6 brain) | 30.7 | 30.4 | 3.18 | 0.310 |

- Tie. Lesion: learning on 33.4 vs off 32.9 (TD made learning neutral instead of harmful, not useful).

#### Why conditioning does not pay here (probe on the 1.5 population, learning on vs off, 12 worlds)
| per-life poison looks | learning | fitness | pain | poison early -> late |
|---|---|---|---|---|
| like the other berries (0.8) | on | 28.5 | 3.10 | 0.32 -> 0.04 |
| like the other berries (0.8) | off | 28.3 | 3.45 | 0.35 -> 0.09 |
| unrelated (0.0) | on | 25.2 | 0.55 | 0.12 -> 0.06 |
| unrelated (0.0) | off | 25.0 | 0.58 | 0.12 -> 0.03 |

- With look-alikes, learning works a little (pain -10%, late poison halved) but gains only ~0.2 fitness. With
  distinct stimuli it is useless because the population never eats unfamiliar-looking food anyway (inherited
  preferences act like neophobia). The early -> late drop also happens *without* learning (poison bushes are
  emptied early and regrow slowly), so the within-life poison metric is confounded.
- Conclusion: the rules work; the **task** does not reward within-life learning enough: inherited preferences
  and the reactive pain response already capture most of the value, and a single-life lesson about one of six
  berry types is worth a point or two against fitness noise of tens.

---

## v5 (2026-10-01): honest fitness, normalised ganglion, chapter 1 re-run

Decisions: ADR-017 (innate vs learned compartments) and ADR-018 (fitness, normalisation).

### What the v4 recordings showed (final generations of 1.0, 1.3, 1.6)
| | 1.0 | 1.3 | 1.6 |
|---|---|---|---|
| `ganglion_i` above 0.95 (share of time) | 84% | 75% | 90% |
| `ganglion_e` above 0.95 | 47% | 51% | 67% |
| `ganglion_e` neurons stuck on all life | 15% | 31% | 45% |
| `no_feed` above 0.95 | n/a | 98% | 78% |
| action entropy (nats; uniform 1.95) | 0.06 | 0.05 | 0.08 |
| food "eaten" (v4 fitness term) / actually absorbed | 76 / 18 | 51 / 8 | 29 / 10 |
| survival term 0.01 x ticks | 6.9 | 3.8 | 3.3 |

- Actions are sampled from the motor *pre-activations* x 8 (`logit_gain / action_temperature`), so they are
  nearly deterministic, not random; the saturated `out` block in the dashboard is display only.
- One berry is 6 food units, a bush 36, the stomach 20: agents strip a bush and v4 credited all of it.
- `raphe` and `pdf` were both excitatory with evolved cross-connections, so they could only excite each other
  (activity correlation +0.73 in 1.6); the `cold` modulator had no receptor.

### Saturation probes: stage 1.0 from scratch, well-fed fitness, 250 generations (last 25)
| variant | fitness (fed) | lifetime | `ganglion_e` > 0.95 | `ganglion_i` > 0.95 |
|---|---|---|---|---|
| none, seeds 0 / 1 | 464 / 436 | 716 / 688 | 0.51 / 0.23 | 0.61 / 0.48 |
| normalisation 2, instantaneous, seeds 0 / 1 | 261 / 466 | 485 / 725 | 0.00 / 0.01 | 0.00 / 0.01 |
| normalisation 2, lagging one step, seeds 0 / 1 | 409 / 550 | 648 / 799 | 0.00 / 0.00 | 0.00 / 0.00 |
| metabolic cost 0.3 | 416 | 649 | 0.18 | 0.36 |
| metabolic cost 1.0 | 187 | 372 | 0.01 | 0.05 |
| lagging normalisation + cost 0.3 | 372 | 616 | 0.03 | 0.00 |

- Cost 1.0: evolution silences the brain and nothing is eaten (0.1 food). Cost fitness values are not directly
  comparable (the cost itself shortens life).
- Seed spread is about +-60, so only the saturation columns separate the variants. Adopted: lagging
  normalisation on the ganglion; metabolic cost implemented but off.

### v5 chain (seed 0; 1.0: 400 generations from scratch, 1.1-1.4: 200 generations, 10% weight mutation)
Fitness = well-fed lifetime. Log `runs/logs/chain_v5.log`.

| last 25 generations | main | control | main / control lifetime | main / control poison |
|---|---|---|---|---|
| 1.0 steering | 408 | | 649 | |
| 1.1 valence, seeds 0 / 1 / 2 | 334 / 376 / 446 (mean 386) | 465 / 324 / 370 (mean 387) | 619 / 604 | 0.19 / 0.20 |
| 1.2 drives | 232 | 281 | 405 / 468 | 0.17 / 0.14 |
| 1.3 affect | 284 | 288 | 481 / 476 | 0.10 / 0.09 |
| 1.4 habituation | 320 | 307 | 528 / 508 | 0.11 / 0.08 |

- Saturation is gone along the lineage (ganglion above 0.95: 0-3% of the time; `no_feed` 1% in 1.3).
- **1.1 valence is a tie over three seeds.** The v4 "clear win" (53 vs 40) came from the fitness: the pain term
  paid the pain pathway directly, and gross eating paid the taste -> feed reflex for gorging. Lesions still show
  the valence cells carry the behaviour (intact 335; `valence_av` 205, `no_feed` 209, `valence_app` 252).
  Halving the hard-wired valence -> motor weights (3 -> 1) changes nothing (336).
- 1.2-1.4 are single seeds and within the +-60 seed spread: no conclusion without replicates.
- 1.3 lesions (8 worlds): intact 280; `pdf` 175 and `roam` 186 (essential), `raphe` 274 and `dwell` 274 (unused;
  `raphe` mean activity 0.04). With the hard-wired mutual inhibition the nuclei are now anticorrelated (-0.48).

### Learning world (1.5/1.6 as now defined) is not run yet
New world (`stages.learning_world`): two ancestral good types, two ancestral poison types, two novel types whose
look is drawn per life (similarity 0.6 to the gooseberry) and one of which is poison per life; poison costs a
whole berry (-3 food points); lives of 2000 ticks. The v5 1.4 population dropped into it (4 worlds):

| world | learning | fitness | lifetime | berries eaten per life | poison share |
|---|---|---|---|---|---|
| density 0.22, novel similarity 0.6, poison -3 | on / off | 211 / 214 | 367 / 372 | 4.8 / 5.2 | 0.17 / 0.17 |
| density 0.22, similarity 0.8, poison -3 | on / off | 208 / 211 | 357 / 356 | 4.7 / 5.1 | 0.18 / 0.19 |
| density 0.30, similarity 0.6, poison -3 | on / off | 203 / 210 | 348 / 362 | 5.5 / 5.5 | 0.16 / 0.17 |
| density 0.22, similarity 0.6, poison -1 | on / off | 239 / 236 | 417 / 414 | 6.1 / 6.2 | 0.20 / 0.20 |

- The blocker is the **food economy**, not the learning rule: an agent eats 5-7 berries in its whole life and
  dies near the never-eating baseline, so it meets a novel food once or twice. No rule can pay on that. Initial
  food in the 1.0 world (about 73 bushes x 36 units = 2600) is below what 64 agents need to live 1000 ticks (3200
  units), and a berry is 30% of a stomach. Learning needs many small meals: smaller berries, a richer world,
  or fewer agents per world.


---

## v6 (2026-10-01, later): many small meals, movement cost, alpha 1, three seeds

### Food economy and movement cost: stage 1.0 probes from scratch (250 generations, fitness = well-fed lifetime)
| world | movement cost (step / turn) | fitness | lifetime | meals per life | forward share of actions |
|---|---|---|---|---|---|
| 32 x 32, density 0.08, berry 6 units (v5) | none | 408 | 649 | 10 | 0.31 |
| 64 x 64, density 0.07, berry 2 units | none, seeds 0 / 1 / 2 | 539 / 465 / 506 | 812 / 734 / 770 | 21 / 15 / 21 | 0.40 |
| 64 x 64, density 0.07, berry 2 units | +50% / +25% | 201 | 402 | 1 | 0.00 |
| 64 x 64, density 0.07, berry 2 units | +20% / +10% | 202 | 403 | 1 | 0.00 |
| 48 x 48, density 0.12, berry 2 units | +50% / +25% | 205 | 408 | 1 | 0.00 |
| 64 x 64, warm start from the no-cost run, 150 generations | none | 595 | 851 | 33 | 0.64 |
| 64 x 64, warm start from the no-cost run, 150 generations | +50% / +25% | 454 | 688 | 20 | 0.41 |

- With any movement cost, evolution from random brains settles on standing still for the whole life. Cause: the
  fitness paid out the birth reserve (a full stomach = 400 ticks), so a sitter that never eats (about 200) ranks
  above a walker that never eats (about 150), and a random brain almost never completes approach + USE + EAT.
  Selection is by rank, so the size of the gap does not matter.
- Staging works: evolve foraging with free movement (stage 0.9), then switch the cost on (1.0).
- Camping (share of 100-tick windows in which an agent stays within 3 cells): 2% without cost, 19% with it;
  14-25% in the old world. Regrowth stays at 500 ticks: waiting costs 25 food for a 12-food bush.

### Time constants: 64 x 64 no-cost world from scratch, 250 generations
| ganglion | fitness (seeds) | meals per life | s per generation |
|---|---|---|---|
| alpha 0.5 (v5) | 539 / 465 / 506 | 21 / 15 / 21 | 0.93 |
| alpha 1.0 | 524 / 574 | 30 / 32 | 0.85 |
| alpha 0.5, two brain steps per tick | 480 / 543 | 19 / 26 | 1.14 |

Adopted alpha 1 for the ganglion and the valence cells (the valence part untested on its own).

### v6 chain, three seeds (0.9: 250 generations from scratch; 1.0-1.4: 200 generations, 10% weight mutation)
Fitness = well-fed lifetime (counted from birth). `stages summary <key>`: mean of the last 50 generations, and the
final populations re-evaluated in 8 shared worlds; +- = standard error over seeds.

| stage | main | control | main - control, last 50 gens | main - control, re-evaluated | meals per life (main) |
|---|---|---|---|---|---|
| 0.9 bootstrap | 587 | | | | 28 |
| 1.0 steering (cost on) | 522 | | | | 32 |
| 1.1 valence | 440 +-22 | 452 +-15 | -12 +-7 | +9 +-4 | 22 |
| 1.2 drives | 309 +-6 | 325 +-4 | -16 +-5 | -8 +-12 | 16 |
| 1.3 affect | 276 +-9 | 285 +-5 | -10 +-4 | -9 +-19 | 18 |
| 1.4 habituation | 282 +-5 | 273 +-8 | +9 +-13 | +8 +-13 | 16 |

- Seed spread is far smaller than in v5 (the larger world and the many meals average out luck).
- No module beats its control. Valence eats less poison (0.14 vs 0.19 of berries) and feels less pain (3.5 vs
  6.1) but that is worth nothing in fitness at the current poison cost (2/3 food unit per berry).
- The step from the 1.1 world to the 1.2 world costs a quarter of the fitness (440 -> ~320): still a cliff.
- **Bug found afterwards: `cold` fired almost always.** Every weight was clipped to +-`w_max` (4) during life,
  so the hard-wired temperature -> `cold` weight of -7 acted as -4 and the neuron fired below body temperature
  0.82 instead of 0.47. 1.2-1.4 above (and all of v4, v5) ran with it. Fixed: only plastic weights are clipped.

### Speed
Plasticity, eligibility, decay and depression are now computed only on the block of synapses that can change
(`Layout.pl_rows/pl_cols`), with identical results (max difference 2e-7 over 60 steps on the 1.4, 1.6, 2.3 and
2.5 brains). One process: learning stage 1.6 at 2000 ticks 9.2 -> 2.3 s per generation; 1.3: 1.4 -> 0.9 s.

### Not finished
The fitness now counts the well-fed lifetime **from the first meal** (`fed_meal`), so non-eaters tie at zero and
standing still can no longer outrank walking. The probe that should show whether stage 1.0 then evolves from
scratch *with* movement cost (which would make stage 0.9 unnecessary) was stopped when the machine ran low on
memory; no result yet. Command: `stages 1.0 --init-from none` (two seeds).


---

## v7-v9 (2026-10-01, evening): first-meal fitness, no-cliff worlds, drive and serotonin fixes

**Stage criterion from now on (user, 2026-10-01):** a stage succeeds when its circuit is *used* (lesions of the new
regions clearly lower fitness) and main is not clearly worse than control. Main >> control is not required.
Tools: `stages summary <key>` (main vs control over seeds) and `stages lesions <key>` (lesions over seeds).

### First-meal fitness removes the standing-still trap
Stage 1.0 from random brains *with* movement cost, 300 generations, fitness = well-fed lifetime from the first
meal: seeds 0 / 1 reach well-fed lifetime (from birth) 436 / 410, lifetime 671 / 658, 23 / 19 meals per life.
Under the from-birth fitness the same setup stood still (fitness 201). Stage 0.9 (bootstrap) is removed again.

### v7: the worlds were cliffs
Chain with the v6 worlds (1.1: density 0.105; 1.2: cold; 1.3: patches), three seeds, fitness from the first meal:
main / control 1.1: 352 / 332, 1.2: 249 / 273, 1.3: 177 / 177, 1.4: 216 / 198; lifetimes 610, 501, 421, 462 of
1000. Each world is harsher than the last and the lineage ends up barely surviving. More bushes do not help
(1.0 population in the 1.1 world: fitness 258 at density 0.105, 243 at 0.12; 1.1 population in the 1.2 world: 257,
279, 280 at 0.12, 0.15, 0.18): the limit is time per meal, not food.

### v8: worlds calibrated with the metabolic rate
`hunger_per_tick` is lowered per stage so that the parent population keeps its lifetime when it enters the new
world: 1.0: 0.05, 1.1: 0.035, 1.2-1.4: 0.025 (v9: 0.019), 1.5: 0.012.

| v8, 3 seeds | main | control | main - control (re-evaluated) | lifetime (main) |
|---|---|---|---|---|
| 1.1 valence | 539 | 537 | +15 +-11 | 838 |
| 1.2 drives | 586 | 593 | +14 +-26 | 899 |
| 1.3 affect | 519 | 531 | +6 +-37 | 864 |
| 1.4 habituation | 517 | 511 | +25 +-35 | 881 |

Lesions, fitness in % of intact, mean of three seeds (per seed in `runs/logs/lesion_v8.log`):
1.1: valence_app 87, valence_av 73, no_feed 78. 1.2: hunger 100, cold 99, warm_run 97, warm_turn 101.
1.3: raphe 99, dwell 99, pdf 62, roam 68. Lesions differ a lot between seeds (e.g. valence_av 83 / 49 / 88), so one
seed can mislead.

### v9: two design faults fixed
- **Thermotaxis sensed the wrong thing.** `temp_change` was the change of *body* temperature, which mostly says
  "this place is warmer than I am", not "I am moving up the gradient"; near a spring the reflex walked the agent
  away. New sense `skin_change`: the change of the temperature at the agent's cell (x 10; one cell up a spring's
  gradient = +0.9). The body now warms and cools slowly (`temp_rate` 0.03) and the cold costs up to +75% hunger.
- **Serotonin was never released in quantity.** `raphe` integrated one-tick taste pulses with alpha 0.03 and stayed
  near 0.04. Now `raphe` fires on taste (alpha 1) and the modulator persists: `ModulatorSpec.decay` 0.97 (released
  at once, cleared over about 30 ticks).
- `stages lesion` got a `no_depression` row (short-term depression switched off).

| v9, 3 seeds | main | control | main - control (re-evaluated) |
|---|---|---|---|
| 1.2 drives | 609 | 634 | -36 +-53 |
| 1.3 affect | 539 | 567 | -39 +-34 |
| 1.4 habituation | 557 | 567 | -12 +-40 |

Lesions (% of intact; per seed, then mean):

| region | 1.2 | 1.3 | 1.4 |
|---|---|---|---|
| valence_app | 68 93 94 (85) | 69 89 101 (86) | 58 99 84 (81) |
| valence_av | 84 50 77 (70) | 60 49 96 (68) | 82 67 85 (78) |
| no_feed | 83 66 90 (80) | 79 64 102 (82) | 96 89 92 (92) |
| hunger | 100 101 103 (102) | 99 102 107 (103) | 98 104 101 (101) |
| cold | 100 102 103 (102) | 97 100 102 (100) | 101 100 99 (100) |
| warm_run | 98 100 103 (101) | 99 103 107 (103) | 102 102 100 (101) |
| warm_turn | 102 100 101 (101) | 96 100 97 (98) | 98 96 94 (96) |
| raphe | | 93 73 95 (87) | 93 92 94 (93) |
| dwell | | 93 80 101 (91) | 98 96 98 (97) |
| pdf | | 71 100 83 (85) | 43 69 51 (54) |
| roam | | 71 101 83 (85) | 48 76 55 (60) |
| no_depression | | | 104 98 97 (100) |

- **Used:** valence (both), no_feed, PDF/roam, and now serotonin/dwell weakly.
- **Not used: hunger, cold, thermotaxis.** The 1.2 recording shows the thermotaxis reflex itself works (P(FORWARD |
  warm_run on) = 0.88 against 0.30 overall), but it fires on 4% of ticks: agents are in a gradient and moving on
  only 9% of ticks. Body temperature is the same in main and control (0.406), 81% of the time below the comfort
  threshold. Likely reasons: (1) a one-tick run-or-tumble reflex cannot hold an agent near a spring against the
  foraging drives; (2) the calibrated worlds are easy (lifetime 915 of 1000), so a 30-40% saving in hunger buys
  little; (3) hunger only scales appetite, and with small meals the animal should eat whenever it can.
- **Not used: habituation** (depression off = 100%).
- Camping (agent stays within 3 cells for 100 ticks): 18-36% of windows in 1.2-1.4 (v9), up from 9-21% in v8.

### Stage 1.5 (learning) in the new world
World (`stages.learning_world`): two ancestral good types, two ancestral poison types, two novel types whose look
is drawn per life (similarity 0.6 to the gooseberry) and one of which is poison per life; a poison berry costs as
much as a good one gives; sickness arrives 2 ticks after eating; 2000-tick lives; `hunger_per_tick` 0.012 (the
1.4 population entering this world: lifetime 694 / 842 / 938 of 2000 at 0.019 / 0.015 / 0.012, net food eaten
about 11 units per life, against 39 per 1000 ticks in its own world).

One seed, 200 generations, last 50:

| | fitness | lifetime (of 2000) | net food | poison share, whole life / first half / second half |
|---|---|---|---|---|
| main (plastic identity -> valence) | 735 | 1313 | 18.5 | 0.19 / 0.25 / 0.06 |
| control (the 1.4 brain) | 883 | 1505 | 24.2 | 0.14 / 0.18 / 0.04 |

Lesions of the main population (8 worlds): intact 692; `no_plasticity` 675 (97.5%); `no_depression` 651 (94%);
valence_app 526 (76%); valence_av 367 (53%); us_taste 643 (93%); us_pain 685 (99%).

- **Learning is still not used.** Switching plasticity off costs 2.5% (inside the noise) and the main brain is
  below its control in this seed. The poison share falls from the first to the second half of life just as much
  without plasticity (confounded, as in v4: poison bushes are emptied early, and the second half contains only
  the survivors).
- An agent eats about 19 berries in a life, so it meets each novel food only a handful of times. Lowering the
  metabolic rate to remove the cliff also lowered how often an agent needs to eat: the calibration knob works
  against the "many small meals" the learning stage needs.
- Not tested yet: whether the rule discriminates the two novel foods at all at similarity 0.6 (in v4 it
  over-generalised across look-alikes); a cleaner metric (poison eaten after the first poisoning).

#### Why learning is not used: what the agents actually eat (12 worlds, the 1.5 main population, learning on vs off)
| sickness delay | learning | novel poison / novel good berries eaten per agent, first half | second half | ancestral good, first / second half | fitness |
|---|---|---|---|---|---|
| 2 ticks | on | 1.09 / 2.22 | 0.08 / 1.01 | 7.7 / 3.8 | 725 |
| 2 ticks | off | 1.05 / 2.04 | 0.07 / 0.99 | 8.0 / 4.0 | 761 |
| 12 ticks (re-evolved, elig decay 0.92) | on | 1.25 / 2.04 | 0.13 / 1.05 | 9.3 / 4.7 | 837 |
| 12 ticks | off | 1.37 / 2.15 | 0.11 / 1.03 | 9.5 / 4.5 | 817 |

- An agent eats about **one** novel-poison berry in its whole life, with or without plasticity. There is nothing
  for learning to save. Novel foods are a small part of the diet (about 3 of 11 berries in the first half).
- Hypothesis tested and rejected: that the innate pain reflex limits the damage because sickness (2 ticks) arrives
  while the agent is still at the bush. With sickness after 12 ticks (`SICK_DELAY`, kept) and a longer eligibility
  trace (`CS_ELIG` 0.92) the numbers are the same: main 788 vs control 805, `no_plasticity` 696 vs intact 700.
- The cause is the meal rate. At `hunger_per_tick` 0.012 one 2-unit berry lasts 167 ticks, so an agent needs about
  12 berries in 2000 ticks and takes one or two berries per bush. Calibrating the cliff away with the metabolic
  rate removed the many small meals that the v6 economy was built for. A calibration that keeps the meal rate
  (scale berry size and poison cost together with the metabolic rate, or make novel foods most of the supply) is
  the next thing to try.


---

## v11-v18 (2026-10-02 and 2026-10-03): why learning did not pay, and a redesign of the valence stage

Probes used below live in `scripts/probes/` (see its README). All evolution results here are **one seed** unless
stated; the measurements under "Selection signal" say how little one seed means.

### Learning rule: one signed teacher erased its own lessons
- v11 world (novel foods 3/4 of the supply, sickness after 12 ticks): both main and control stuck at lifetime ~350
  of 2000, half of what they eat is poison, plasticity off = intact (117.8 vs 117.7).
- In that population evolution had set the learning rate onto the aversive cells to 0.013 and onto the appetitive
  cells to 0.33: aversive learning switched off.
- Tick-by-tick trace on one brain: one poisoning raises the weights from the poison's look onto the aversive cells
  by +0.26; the next good meals remove it again. Cause: one teacher (taste - pain) and one long eligibility trace
  (long because sickness is delayed), so a good meal acts on everything still in the trace, including the poison
  bush just left.
- **Two teachers** (adopted): taste teaches the appetitive synapses with a short trace (0.5), sickness teaches the
  aversive synapses with a long one (0.92). Scripted probe on fresh brains (3 poison + 3 good trials, change of net
  valence of the poison look / the good look):

| pain timing, trace | one teacher | two teachers |
|---|---|---|
| immediate, short trace (8 appearance features) | -0.04 / +0.18 | -0.02 / +0.11 |
| after 6 ticks, long trace (8 features) | -0.06 / +0.46 | -0.26 / +0.01 |
| after 6 ticks, long trace (4 features, similarity 0.6) | -0.06 / +0.31 | -0.19 / -0.05 |

  With immediate pain one teacher is as good as two; with delayed sickness only two teachers work. Eight appearance
  features (adopted, `APPEARANCE = 8`) keep the aversion from spreading to the good look.

### Infrastructure added (v12-v13, v18)
Tunable hard-wired strengths (`ProjectionSpec.tune`), population 256 on 128 x 128 with a 64-agent recording,
`stages versus` (head to head), dud bushes for 1.4, siblings (64 genomes x 4), `RegionSpec.phase` (ordered
propagation), `ProjectionSpec.src_range`, regrowth tied to the metabolic rate, side stage `x.learn`.

### Population 256, chain on the v13 definitions (seed 0, 150 generations; 1.0: 400)
| stage | main | control | note |
|---|---|---|---|
| 1.0 | lifetime 688 of 1000, 79 of 256 survive, 41 meals per life | | 4.4 s per generation |
| 1.1 | 506 | 551 | |
| 1.2 | 643 | 666 | body temperature 0.44 vs 0.39 |
| 1.3 | 620 | 603 | |
| 1.4 (dud bushes) | 629 | 594 | |
| 1.5 (two teachers, novel foods 3/4) | 306 | 333 | flat over 150 generations; plasticity off = 94% of intact, sickness teacher off = 93% |

### Following individuals (1.5 recording, 64 agents): learning changes behaviour but does not pay
| before -> after the first lesson about a type | learning brain | control |
|---|---|---|
| poison type: USE when its bush is in front | 0.49 -> 0.27 | 0.40 -> 0.34 |
| poison type: EAT when its berry is in hand | 0.68 -> 0.32 | 0.73 -> 0.65 |
| good type: EAT when its berry is in hand | 0.19 -> 0.32 | 0.85 -> 0.93 |
| aversive valence at a good bush | 0.53 -> 0.56 | 0.07 -> 0.09 |
| ticks with a (good-type) berry in hand | 3085 | 1122 |

The aversion is not specific (aversive cells at ~0.5 for good bushes too), and an agent holding a berry it mistrusts
could neither eat it nor put it down (aversion blocked USE as well as EAT).

### Generation-0 tests on the v13 lineage (stage-1.4 population, learning on vs off, 6 shared worlds)
| variant | fitness on / off | poison share of berries eaten, on / off |
|---|---|---|
| delayed sickness, rule as defined | 253 / 295 | 0.32 / 0.34 |
| aversion blocks EAT only | 249 / 273 | 0.36 / 0.36 |
| sickness blames only the item in hand | 293 / 295 | 0.34 / 0.34 |
| rebalanced world (novel half of supply), delayed | 334 / 376 | 0.31 / 0.33 |
| same, immediate pain, short traces | 379 / 406 | 0.31 / 0.31 |
| same, 6 x learning rate, with decay 0.01 | 351 / 348 | 0.34 / 0.35 |
| immediate pain, mutual inhibition between the valence cells | 375 / 363 | 0.33 / 0.34 |

No variant of the *rule* gave selectivity in the world: learning lowered eating of everything.

### The broken link was valence -> action (brain-only action assay, `scripts/probes/assay.py`)
- v13 lineage: evolution had scaled the hard-wired valence wiring to 0.12-0.39 x the designed value (the range
  allowed 0.1 x). After training at learning rate 0.5 the valence cells discriminate (poison 0.57 / 0.90
  appetitive / aversive, good 0.84 / 0.41, staple 0.78 / 0.67) but at the staple bush USE fell from 63% to 19%.
- With one motor meaning per appetitive cell (v15) a learned "good" came out as FORWARD in every context: at the
  staple bush FORWARD / USE went from 64% / 34% to 100% / 0%, EAT with the berry in hand from 84% to 1%.
- Small animals do it differently: value acts on motor *programmes*; approach is driven from a distance, grasping
  and swallowing are contact reflexes that value permits or blocks (worm command neurons, *Aplysia* feeding, fly
  proboscis reflex).

### Stage 1.1 redesign, step by step (one seed each; lesion = fitness with the region silenced, % of intact)
| version | change | main / control | head to head | lesions | at a good bush: USE / EAT (empty hand); EAT (berry in hand) |
|---|---|---|---|---|---|
| v14 | poison costs a berry, tuning range 0.33-3 x, siblings | 657 / 681 | -54 | app 56, av 60, no_feed 66 | 0.36 / 0.47; 0.67 |
| v15 | one motor meaning per appetitive cell; aversion blocks EAT only | 689 / 681 | +9 | app 98, av 97, no_feed 99 | 0.68 / 0.01; 0.25 |
| v16 | taste and pain reach the brain only through the valence cells | - / - | +8 | app 79, av 91, no_feed 97 | 0.57 / 0.12; 0.51 |
| v17 | value cells (seen / held) + gated programmes approach, grasp, ingest, reject | 646 / 697 | -40 | app 39, approach 53, ingest 69, grasp 102, reject 100, av 94 | 0.50 / 0.03; 0.61 |
| v18 | + ordered propagation, hand-full gate, regrowth tied to metabolism | 518 / 564 | +54 | app 37, av 61 (poison share 0.06 -> 0.25), no_feed 81, grasp 37, approach 100, ingest 104, reject 100 | 0.78 / 0.00; 0.81 |

- v14's high lesion numbers were dependence, not benefit: main lost to control.
- v17 (two ticks from sense to programme-driven action): one berry every ~6 ticks through the programme path, and
  for single-pick items the stale second USE puts the item down again. v18 lets activity pass through the layers in
  order within a tick (`RegionSpec.phase`); behaviour at the bush becomes clean.
- v18: main and control disagree between separate worlds (-46) and shared worlds (+54): one seed, and a better
  forager population also depletes its own world.
- v17/v18 agents spend 45-55% of their time pressing USE at empty bushes (25-32 thousand of ~57 thousand
  agent-ticks). Resting costs little at the calibrated metabolic rate and the fitness stops rewarding an agent whose
  stomach is full; the contact reflex has nothing that stops it (what habituation is for).

### Learning test bed `x.learn` (conditioning on the 1.1 brain, immediate pain, short traces), generation 0
| stage-1.1 population | fitness on / off | note |
|---|---|---|
| v16 (cells drive the motor neurons) | 440 / 549 | second-half eating collapses: learned value comes out as FORWARD |
| v17 (gated programmes) | 646 / 632 | first time learning does not hurt; poison share in the first half of life 0.23 / 0.25 |
| v17, 4 x learning rate | 529 / 632 | poison share in the first half 0.21 / 0.25, less eating overall |
| v18 (ordered propagation) | 460 / 469 | poison share over the whole life 0.16 / 0.19, pain 2.6 / 3.9, berries eaten 16 / 21, lifetime 927 / 900 of 2000 |
| v18, 4 x learning rate | 431 / 469 | poison share 0.14 / 0.19, pain 2.0 / 3.9, berries eaten 14 / 21 |

At v18 learning is selective from generation 0 (a third less pain, poison share down by a sixth to a quarter) and
no longer harmful, but it still lowers the eating of good food by a quarter, so it does not pay yet.

### Evolution settings (population 256, v13 chain)
- Curves: most of the gain comes in the first 60-90 generations, then a plateau; 150 generations are enough.
- One life is mostly luck: the correlation between an agent's fitness in two worlds is 0.00-0.13; of 32 elites 4-7
  are elite again (chance 4). Mutational load per round: -23 ... +5, not resolved (noise ~10).
- Siblings (64 genomes x 4) did not change stage 1.0 visibly (lifetime 650, 30 meals, USE at a good bush 79%).

### Selection signal (`scripts/probes/fitness_signal.py`, stage 1.0, v18, 8 worlds)
Intact genomes and mutated copies of themselves in the same worlds; d = difference in rank / spread of one life.

| damage | well-fed lifetime from first meal (current) | energy acquired | ticks alive |
|---|---|---|---|
| one round of warm-stage mutation (10% of weights, std 0.1) | 0.04 +-0.05 | 0.04 +-0.04 | -0.11 |
| one round of stage-1.0 mutation (every weight, std 0.1) | 0.16 +-0.07 | 0.20 +-0.07 | -0.01 |
| every weight, std 0.25 | 0.67 +-0.04 | 0.77 +-0.03 | 0.47 |
| share of the variance between lives due to the genome (intraclass correlation) | 0.18 | 0.22 (per tick alive: 0.24) | 0.17 |

- A normal mutation is invisible to selection; warm stages are mostly drift. Rough yardstick: an advantage below
  about 10% of fitness cannot be selected for with this machinery.
- "Energy acquired" (food eaten, not capped by the stomach: surplus becomes offspring) separates genomes 15-30%
  better than the current fitness and has no ceiling.

### What the two days taught (for every later stage)
A new capability pays only if all four hold: (1) the mechanism computes the right thing, (2) its output reaches
behaviour in a form that fits the situation, (3) the world offers something only it can exploit, (4) selection can
see the benefit. For learning we found and fixed (1) (two teachers) and (2) (programmes), built (3), and measured
that (4) is weak. Still open in the learning stage: specificity (aversion spreads to the staple, which shares 45% of
its look with novel foods); animals solve it with latent inhibition and blocking.


---

## v19-v21 (2026-10-03, afternoon): five decisions and the first learning stage that is used

Decided with the user after the reflection at v18: fitness = energy acquired; recombination, chosen on a recovery
benchmark; chapter-1 animals eat what they grasp (holding returns later); 1.2-1.4 parked; the learning rule kept
simple. Later the same day: leave the food economy alone; a generation-0 result is a diagnostic, not a gate; drop
cells that lesions show unused.

### Selection signal and the food supply (`scripts/probes/world_size.py`)
The stage-1.0 population (eat on grasp, bush density 0.07) left 57% of all bushes empty: fitness was a race for a
fixed supply. The same population with more food per agent:

| world | energy per life | lifetime | share of the variance between lives due to the genome |
|---|---|---|---|
| 128 x 128, density 0.07 | 69 | 614 | 0.08 |
| 181 x 181, density 0.07 | 118 | 820 | 0.27 |
| 256 x 256, density 0.07 | 128 | 805 | 0.39 |
| 128 x 128, density 0.14 (adopted) | 139 | 853 | 0.35 |
| 128 x 128, density 0.28 | 120 | 729 | 0.34 |

A population evolved at 0.14 empties 60% of the bushes again and its genome share is 0.18 in its own world: better
foragers always push to the supply limit. Accepted (user): no more tuning of the economy.

### Recovery benchmark (`scripts/probes/recovery.py`): damaged stage-1.0 population, 40 generations, 3 seeds
| setting | energy, poor world (0.07) | energy, rich world (0.14) | lifetime (rich) |
|---|---|---|---|
| well-fed lifetime, asexual (v18) | 65.2 | 163.7 | 871 |
| energy acquired, asexual | 67.5 | 160.9 | 837 |
| energy + recombination, whole neurons (adopted) | 68.6 | 171.4 [170 173 172] | 827 |
| energy + recombination + few large mutations | 68.4 | 170.7 | 809 |
| energy + recombination, blending | 71.0 | 169.8 | 816 |
| energy + recombination, 256 genomes x 1 | 67.6 | 168.2 | 832 |
| energy, every weight mutates (stage 1.0 from scratch) | 59.7 | 159.5 | 835 |

- In the poor world every setting hits the supply ceiling. In the rich world recombination is 6-7% ahead of asexual
  reproduction in every seed (seeds within +-2). The energy fitness is not better than the well-fed lifetime at
  recovery; it is kept because it has no ceiling.
- Mutating every weight is the worst setting: stage 1.0 now ends with 60 generations at 10% (`--init-from <run>
  --mutation-prob 0.1`), which took it from 155 to 172 energy.

### Chapter 1 on the new footing (64 genomes x 4 siblings, recombination, energy fitness, density 0.14, eat on grasp)
| stage | main | control | head to head | note |
|---|---|---|---|---|
| 1.0 | 172 energy, lifetime 801, 88 meals per life | | | USE at a bush 90%, EAT never pressed, 6% camping |
| 1.1, 3 seeds | 60.1 +-2.5 | 29.5 +-1.9 | +81 +-3 | poison share 0.09 vs 0.29 |

1.1 lesions over three seeds (% of intact): valence_app 46, valence_av 52, no_feed 55, grasp 46, no_touch 59;
ganglion_e 99, ganglion_i 102 (the ganglion is idle in this stage: steering is done by the sensor -> motor
reflexes). An approach programme with a contact gate was unused in all three seeds (100, 100) and removed; energy
unchanged (60.0 before, 60.1 after).

- The control (no cells that receive pain or taste) eats poison at chance (0.29, 0.29, 0.30) in all three seeds.
  It could learn to avoid poison by look, since looks are fixed, and an earlier control did (v16); under the current
  conditions it does not within 150 generations. Not investigated further (user: move on).
- In main most avoidance is by look before any bite (USE at 5% of poison bushes faced); the pain reflex is the
  backup after a bite.
- The 1.1 world is hard on the incoming population (24 energy on entry, 175 in its own world).

### x.hands: can a mouth-feeding lineage adapt to pick, hold, eat?
Two runs of 100 generations, the 1.1 population moved into a pick-then-eat world. With the hands extension: energy 2 -> 45 after ten generations -> 65 (first run, before trimming) and 10 -> 56 -> 67 (trimmed); the lineage reaches what it had with eat on grasp (60-63). With the plain 1.1 brain: 0 throughout in the first run, 2 -> 38 -> 66 in the second. Lesions of the trimmed extension: ingest 0%, grasp 0%, valence_app 0%, no_hold 55%, valence_av 39%. So the lineage can adapt; the ingest programme makes it reliable and fast, and is what the adapted animal uses.

### Learning (1.5 on the 1.1 brain; immediate pain, short traces, two teachers)
Assay on the 1.1 population (`scripts/probes/assay.py`, learning rate 0.3, two rounds of meals), USE when the bush
is adjacent, before -> after: staple 79% -> 87% (100% after four more staple meals), novel good 29% -> 90%, novel
poison 17% -> 0% (turns away 62%), innately avoided poison 0% -> 49% (it looks 80% like the staple; appetite learned
for the staple spreads to it). Value cells after: staple 0.99 / 0.05, novel good 0.83 / 0.01, poison 0.86 / 0.66.
At the starting rate 0.05 the same directions, small.

Generation 0 (6 shared worlds): rate 0.05: energy 52 on / 52 off, poison berries 4.3 / 5.1; rate 0.3: 47 / 52,
poison 2.4 / 5.1, good berries 25.0 / 29.5.

Evolution, three seeds, 150 generations (trimmed brain, no safety cell):

| | main | control |
|---|---|---|
| energy acquired | 43.6 +-0.8 [43.5 45.1 42.2] | 40.9 +-0.5 [41.9 40.8 40.0] |
| head to head | +9.7 +-2.8 | |
| poison share / pain / lifetime (of 2000) | 0.23 / 8.1 / 720 | 0.25 / 8.9 / 677 |

Lesions (% of intact, per seed, mean): no_plasticity 103 96 92 (97); us_pain 94 92 90 (92); us_taste 105 106 101
(104); valence_app 53, valence_av 34, no_feed 48, grasp 54, no_touch 48; ganglion 101 / 98.
Evolved learning rates (start 0.05, cap 0.5): appetitive 0.28 / 0.36 / 0.35, aversive 0.05 / 0.11 / 0.08; genomes
within a population range from 0 to the cap.

- Main is ahead of control in both comparisons in every seed, by a small margin (about 7% energy).
- The aversive half of the rule helps consistently (silencing the pain teacher costs 6-10% in every seed). The
  appetitive half does not: silencing the taste teacher *raises* energy by 1-6%, although evolution raised its
  learning rate sixfold. Likely the spread of learned appetite onto look-alike poison seen in the assay; a few
  percent is below what selection can see.
- Plasticity as a whole is therefore worth only about 3% (97% without it).
- An earlier single seed with the safety cell (v20): main 45.1 vs control 39.2, head to head +20.6, no_plasticity
  89%, us_pain 84%, us_taste 101%, safety 98%, rates 0.13 / 0.16.

## v22 (2026-10-03, evening): reversal left out, and what a drive could win

### Stage 1.6 reversal (learned weights relax toward the inherited ones; novel types swap meaning mid-life), 3 seeds

| | main | control (the 1.5 brain) |
|---|---|---|
| energy, last 50 generations | 42.8 +-1.2 [45.1 40.9 42.5] | 41.5 +-0.9 [42.6 42.2 39.7] |
| re-evaluated in shared worlds | 46.6 +-0.7 | 43.9 +-1.3 |
| head to head | -6.5 +-1.6 (-3, -9, -8) | |

Lesions (% of intact, per seed, mean): no_plasticity 100 103 101 (101); us_taste 104 108 103 (105); us_pain 98 96 97
(97); valence_app 7, grasp 5, valence_av 34, no_feed 44, no_touch 27; ganglion 100.

- Learning is not used in the reversal world, and in direct competition the relaxing weights lose in every seed.
- The taste teacher is at or above 100 in all six seeds of 1.5 and 1.6.
- Decision (user): reversal stays out of the chain. Chapter 1 learning ends at 1.5.

### Bites on a full stomach (recordings of the last generation)

| stage | meals per life | meals within 1 unit of full | stomach rise per meal (berry = 2) |
|---|---|---|---|
| 1.0 | 88 | 80% | 0.42 |
| 1.1 | 32 | 84% | 0.28 |
| 1.5 | 30 | 59% | 0.44 |

Energy acquired counts every bite at face value; of the 172 per life in stage 1.0 about 37 enter the body. Counting
only absorbed energy would be the lifetime fitness again (absorbed = burned - starting store), so the fitness stays
(user). Consequence: a hunger signal that holds back appetite cannot be selected in this chapter.

### What the cold costs (`scripts/probes/drives.py`, the 1.5 population at generation 0, 4 worlds)

The old cold world (hot-spring weight 0.6) is not cold at bush density 0.14: a third of the map is above 0.6 and
agents average a body temperature of 0.64. With weight 0.1: 76% of cells below 0.3, 11% comfortable.

| brain (springs 0.1, temp_rate 0.03) | energy | lifetime | body temperature |
|---|---|---|---|
| cold not sensed, world not cold | 37.8 +-2.7 | 758 | 0.50 |
| cold not sensed, cold world | 35.2 +-3.3 | 521 | 0.37 |
| + cold-gated thermotaxis | 33.2 +-2.8 | 516 | 0.39 |
| + thermotaxis shut by a hunger gate | 34.1 +-2.6 | 524 | 0.39 |

temp_rate 0.005 and springs 0.05 give the same picture (35.0 / 34.0 / 33.8 and 35.9 / 35.3 / 35.5).

- Cold removes 30% of the lifetime but 7% of the energy: the ticks before starvation yield about 0.011 energy each
  (average 0.05). The most a warmth drive can win is about 3 of 38.
- Gradient thermotaxis does not change body temperature: a spring's gradient reaches four cells.
- Nothing in the brain lets a drive switch foraging: steering is direct sensor -> motor reflexes, the ganglion is idle.
- Open (user to decide): cold as sluggishness (failed actions when cold), or drives and affect move to chapter 2.

## v23 (2026-10-03, night): fitness back to the well-fed lifetime; a drives design that works at generation 0

Decided with the user: eating harder when hungry and looking after warmth are too elementary to postpone, and
energy acquired cannot select for either (it pays for every bite and hardly for staying alive). Changes:
fitness = well-fed lifetime from the first meal again (`default_fitness`; `fitness_energy` kept); the taste teacher
is removed from 1.5 (pain teacher only); a skin-temperature sense (`BodyConfig.skin`); chapter 1 is being rerun.

### Was the well-fed lifetime too weak a signal? (`scripts/probes/recovery.py`, damaged 1.0 population, 40 generations)

Both with recombination; intact population 174 energy, 783 ticks; damaged start 155 energy.

| selected on | energy reached [seeds] | lifetime reached [seeds] |
|---|---|---|
| well-fed lifetime from first meal | 176.1 [175.4 178.0 174.8] | 826 [840 800 837] |
| energy acquired | 178.0 [175.8 179.8 178.4] | 790 [803 794 773] |

A tie. The earlier weakness of the lifetime fitness was mostly the lack of recombination and the poor world (v18
numbers: genome share 0.18 vs 0.22; asexual recovery 163.7 vs 160.9).

### What the cold costs under each fitness (1.5 population, generation 0, springs 0.1)

| | energy acquired | well-fed lifetime |
|---|---|---|
| world not cold | 37.8 | 400 |
| cold world, cold not sensed | 35.2 | 274 |
| cost | -7% | -32% |

### Kinesis (`scripts/probes/drives.py`, 1.5 population, generation 0, 4 worlds, springs 0.1, temp_rate 0.03)

User's suggestion: no gradient climbing and no sight of the spring; move while cold, stay when warm (orthokinesis).

| brain | well-fed lifetime | lifetime | energy | body temperature |
|---|---|---|---|---|
| cold not sensed, world not cold | 385 +-30 | 730 | 38.0 | 0.50 |
| cold not sensed (control) | 266 +-19 | 509 | 35.2 | 0.37 |
| old rule: climb the gradient, turn when cooling | 270 +-12 | 520 | 35.0 | 0.40 |
| run when the body is cold | 177 +-5 | 440 | 19.1 | 0.36 |
| run + rest, reading the body, hunger gate 0.67 | 355 +-33 | 694 | 35.8 | 0.37 |
| run + rest (weight 3), reading the skin | 182 +-5 | 485 | 17.2 | 0.38 |
| run + rest (weight 9), reading the skin | 125 +-5 | 515 | 8.0 | 0.50 |
| the same + turn when cooling + hunger gate 0.67 | **438 +-51** | 866 | 35.3 | 0.44 |
| the same with temp_rate 0.1 | 421 +-39 | 835 | 34.9 | 0.43 |
| hunger gate at 0.5 / 0.33 (weight 3, skin / body) | 320 / 264 | 715 / 601 | 32.0 / 29.7 | 0.38 / 0.35 |

- Run-when-cold with rest-when-warm finds and holds warmth (0.50), but only if it reads the temperature of the
  place (skin): the body's own lags by about 30 ticks, so an animal crossing a warm zone never notices. And the
  rest signal must nearly block FORWARD (weight 9); at 3 the foraging reflexes carry the animal out again.
- Without hunger the warmth seeker starves. With an inhibitory `hungry` cell shutting the warmth mode below 2/3
  stomach it does both: +65% over the cold-blind brain, above that brain in a world without cold.
- The hunger gate helps by itself (+33-41% with warmth never reached): a fed animal moves on instead of biting on
  a full stomach and emptying its own bushes.
- Turning when the skin cools made no measurable difference (371 vs 376 at weight 3); it starts out of the circuit.

### Rerun of the chain under the well-fed lifetime: first seed of 1.0 and 1.1

| run (last generation / last 25) | well-fed lifetime | lifetime | energy | alive at the cap | FORWARD | USE | camping |
|---|---|---|---|---|---|---|---|
| 1.0 | 644 | 839 | 162 | 162 of 256 | 0.53 | 0.25 | 4% |
| 1.1 main | 512 | 826 | 63 | 81 | 0.50 | 0.27 | 8% |
| 1.1 control (pain-blind) | 572 | 909 | 35 | 133 | 0.21 | 0.54 | 44% |

Life cap (`scripts/probes/ceiling.py`, the same populations with longer lives):

| population | cap 1000: fitness, alive at cap | cap 2000 | cap 3000 |
|---|---|---|---|
| 1.0 | 619, 62% | 1065, 50% | 1438, 42% |
| 1.1 main | 527, 30% | 652, 14% | 696, 4% |
| 1.1 control | 572, 55% | 722, 14% | 743, 1% |

- **The ceiling is real in 1.0**: most agents reach the cap, and 42% are still alive at 3000 ticks (they can sustain
  themselves indefinitely), so a longer life does not remove it.
- **In 1.1 the pain-blind control is ahead on this seed** (572 vs 512; also at longer caps). It stays put and keeps
  using the bush in front of it (USE costs nothing), so it burns little: a birth reserve of 20 lasts 800 ticks at
  rest. The main brain eats 63-82 energy and absorbs about 12 of it: its grasp programme bites whenever something
  valued is adjacent, full or not, empties its surroundings and pays for the walking.
- One seed; seeds 1 and 2 and stage 1.5 are running. If it holds, the valence brain needs the hunger signal (bite and
  search when hungry) to do well under a survival fitness, which argues for drives directly after 1.1.

### Stage 1.1 under the well-fed lifetime, born full, three seeds (v23)

| | main | control (pain-blind) |
|---|---|---|
| fitness, last 50 generations | 534 +-4 [538 526 538] | 427 +-57 [524 327 431] |
| re-evaluated in shared worlds | 542 +-11 | 506 +-31 |
| head to head | +262 +-47 (+173, +334, +278) | |

Lesions (% of intact, per seed, mean): valence_av 59 60 53 (57); no_feed 62 61 62 (62); ganglion_e 72 28 21 (40);
ganglion_i 75 100 85 (87); valence_app 103 108 99 (104); grasp 104 108 100 (104); no_touch 102 114 104 (107).

- Main wins head to head in every seed; the control of seed 0 was one lineage that found the stay-put strategy.
- Under the survival fitness the ganglion is used again and the aversive side still is; the appetitive side
  (appetitive cells, grasp programme, contact gate) is not: a bite whenever something valued is adjacent, full or
  not, does not help survival. This is where hunger belongs.
- Stage 1.5, seed 0 only, last generation: main 551, control 581 (one noisy generation; not analysed).

## v24 (2026-10-03, late): born a quarter full

`WorldConfig.start_food` = 0.25 in W10 and every world derived from it (user agreed): a full stomach covered
400-800 ticks at rest, most of a 1000-tick life, so sitting still reached the cap. The chain rerun on this footing
(1.0 scratch + 60 generations at 10% mutation, then 1.1 with three seeds) was started and **interrupted by a
shutdown**; nothing of it is analysed. Restart it from stage 1.0 (commands in STATUS).

### The chain rerun born a quarter full (2026-10-07)

Stage 1.0 from scratch, 400 generations (`s1_0_steering/20261007-164901`): well-fed lifetime 151 in the first ten
generations, 630 over the last fifty, 174 of 256 alive at the cap (born full: 644, 162). It evolves with the small
reserve; the ceiling is unchanged. Then 60 generations at 10% mutation (`20261007-172058`).

Stage 1.1, three seeds, 150 generations, evaluated with `python -m life.book export 1.1 --evaluate` (8 shared
worlds):

| | main | control |
|---|---|---|
| well-fed lifetime, last 50 generations | 519 +-18 [551 487 520] | 394 +-94 [440 214 528] |
| re-evaluated in shared worlds | 525 +-15 [550 498 528] | 433 +-105 [501 228 571] |
| poison share of meals | 0.06 [0.06 0.02 0.09] | 0.13 [0.12 0.19 0.09] |
| head to head | +290 +-105 [+283 +476 +112] | |

Lesions (% of intact, per seed, mean): valence_av 38 32 34 (35); no_feed 44 39 39 (41); ganglion_e 12 27 14 (17);
valence_app 94 84 75 (84); grasp 94 83 67 (81); no_touch 106 24 106 (79); ganglion_i 76 99 107 (94).

- The aversive side is used in every seed, as born full. The appetitive side and the grasp programme now cost
  6-33% when silenced (born full: 104, unused); the effect is small in seed 0.
- The ganglion is the main forager (17% without it), as under the well-fed lifetime born full.
- The control of seed 2 reaches main's level (528 vs 520, poison share 0.09 vs 0.09): a pain-blind brain can evolve
  avoidance by look, since the poison looks the same in every generation. Main minus control measures how much
  easier the value cells make that, not what pain is worth.

Stage 1.5, seed 0 only (current definition), last 50 generations: main 434 vs control 455, poison share 0.20 vs
0.17, lifetime 768 vs 803 of 2000. The two further seeds were not started (below).

### Why learning does not pay in 1.5: the rule and the food stock (2026-10-07)

**The rule cannot become a classifier** (`scripts/probes/classify.py`: one aversive cell on the 8 look features of
the six types, bites with probability 1 - aversion, no brain and no world). Probability of biting after 200
encounters, rate 0.3, 200 lives: staple / novel good / novel poison:

| rule | NOVEL_SIM 0.45 | NOVEL_SIM 0.1 |
|---|---|---|
| pain only, dw = eta * pain * look (1.5 as defined) | 0.09 / 0.40 / 0.05 | 0.35 / 0.92 / 0.07 |
| pain minus the safety cell (v19-v20) | 0.88 / 1.00 / 0.10 | 0.94 / 1.00 / 0.07 |
| prediction error, dw = eta * (pain - a) * look | 0.97 / 1.00 / 0.22 | 0.99 / 1.00 / 0.19 |

Every look shares part of the gooseberry look and nothing lowers the aversive weights (no negative teacher, no
decay), so suspicion of the poison ends on the staple. Both corrected rules converge.

**Generation 0** (`gen0.py`, new 1.1 population, 6 shared worlds, well-fed lifetime, learning on / off; good and
poison berries per agent over the life):

| world | rule, rate | fitness on / off | good on / off | poison on / off |
|---|---|---|---|---|
| as defined (poison -2) | 0.05 | 379 / 414 | 18.9 / 22.5 | 3.5 / 5.1 |
| as defined | 0.3 | 276 / 414 | 12.9 / 22.5 | 1.5 / 5.1 |
| NOVEL_SIM 0.1, poison -6 | 0.05 | 318 / 294 | 17.5 / 17.4 | 2.0 / 2.3 |
| NOVEL_SIM 0.1, poison -6 | 0.3 | 281 / 294 | 14.7 / 17.4 | 1.2 / 2.3 |
| poison -3 (both blanket policies earn zero from novel food) | 0.05 | 350 / 374 | 18.6 / 21.5 | 3.2 / 4.4 |
| poison -3 | 0.3 | 255 / 374 | 12.7 / 21.5 | 1.5 / 4.4 |
| poison -3, safety cell | 0.05 | 351 / 374 | 18.8 / 21.5 | 3.2 / 4.4 |
| poison -3, safety cell | 0.3 | 263 / 374 | 13.1 / 21.5 | 1.5 / 4.4 |
| poison -3, NOVEL_SIM 0.1 | 0.05 | 377 / 372 | 19.1 / 20.0 | 2.6 / 3.3 |
| poison -3, NOVEL_SIM 0.1 | 0.3 | 321 / 372 | 15.4 / 20.0 | 1.4 / 3.3 |
| poison -3, NOVEL_SIM 0.1, safety cell | 0.3 | 328 / 372 | 15.7 / 20.0 | 1.4 / 3.3 |

Standard errors 31-46 over worlds (not paired). The safety cell changes nothing in the world although it repairs
the rule on paper; not traced yet (probably: aversion blocks the bite, so a mistrusted good food rarely gets its
safe meal, and lives end first, next point).

**The food is a stock that is raced down.** An empty bush regrows after REGROW_FOOD / hunger_per_tick = 960 ticks
in the 1.1 and 1.5 worlds (480 in 1.0) and every bush starts full. Share of bushes with berries and animals alive
(of 64) in the last-generation recordings:

| tick | 1.0 all types | alive | 1.1 the four types eaten | alive | 1.5 staple | alive |
|---|---|---|---|---|---|---|
| 100 | 52-58% | 63 | 36-67% | 64 | 49% | 62 |
| 300 | 14-19% | 63 | 10-21% | 63 | 17% | 57 |
| 500 | 14-21% | 49 | 1-11% | 61 | 9% | 50 |
| 750 | 42-50% | 36 | 1-5% | 43 | 5% | 35 |
| 999 | 25-31% | 35 | 19-32% | 18 | 29% | 16 |

The mean stomach goes from 5 to 18 of 20 in the first 100 ticks. In the 1.5 recording three types (the ancestral
poison and two novel ones) stay 86-100% full all life while animals starve; 9 are alive at tick 1500, when the
bushes are back at 93%. Holding back cannot be selected: whoever waits finds nothing left (user, 2026-10-07).

Proposed, to be decided by the user: (1) food as a flow: bushes start at random points of their regrowth cycle and
grow berries back one at a time; (2) a full stomach blocks the bite, as a reflex in the 1.1 grasp programme;
(3) 1.5 on the prediction-error rule with a poison cost of 3, checked with classify.py, the assay and generation 0.

## v25 (2026-10-07, evening): food as a flow, a bite cost

Agreed with the user: bushes keep regrowing all at once (so camping still does not pay), but the start is
staggered; no hand-built block on eating when full (the animal should come to that itself once it can sense its
stomach), but a bite costs energy so that restraint can pay.

- `WorldConfig.start_spent` = 0.7 in W10 and every world derived from it: that share of the bushes starts empty, at
  a random point of its regrowth.
- `WorldConfig.eat_cost` = 0.2 food units per bite (a tenth of a berry), also on a full stomach.

Values chosen with `scripts/probes/supply.py` (the born-a-quarter-full 1.1 population, one life, not adapted):

| world | edible bushes full at tick 0 / 300 / 600 / 900 | mean stomach at 300 / 600 / 900 | alive at the end (of 256) |
|---|---|---|---|
| v24 | 100 / 19 / 5 / 2 % | 83 / 54 / 26 % | 89 |
| start_spent 0.5 | 51 / 19 / 12 / 12 % | 78 / 68 / 62 % | 177 |
| start_spent 0.5, eat_cost 0.2 | 51 / 19 / 14 / 15 % | 76 / 60 / 55 % | 160 |
| start_spent 0.7, eat_cost 0.2 | 32 / 19 / 18 / 21 % | 73 / 69 / 67 % | 162 |

Stage 1.0 from scratch (`s1_0_steering/20261007-184358`, 400 generations): well-fed lifetime 40 in the first ten
generations, 533 over the last fifty; after 60 generations at 10% mutation (`20261007-191412`) 608, 183 of 256
alive at the cap. Supply over a life of that population: edible bushes full 33% at birth, then 18-25% all life;
stomach 73-82%; 80% alive at the end; 66% of bites on a stomach over 90% full.

Stage 1.1, three seeds, 150 generations (`python -m life.book export 1.1 --evaluate`, 8 shared worlds):

| | main | control |
|---|---|---|
| well-fed lifetime, last 50 generations | 452 +-15 [422 458 475] | 186 +-13 [189 163 207] |
| re-evaluated in shared worlds | 465 +-17 [431 481 482] | 188 +-27 [172 150 241] |
| poison share of meals | 0.07 [0.09 0.06 0.05] | 0.17 [0.16 0.18 0.17] |
| head to head | +401 +-35 [+342 +464 +398] | |

Lesions (% of intact, per seed, mean): valence_av 44 40 37 (40); no_feed 50 45 40 (45); ganglion_e 44 95 52 (63);
ganglion_i 55 41 96 (64); valence_app 85 93 93 (90); grasp 87 90 91 (89); no_touch 86 99 95 (94).

- Passes: the aversive side is used in every seed and main is far ahead of control in every seed.
- No control found avoidance by look within 150 generations on this footing (poison share 0.16-0.18 at the end;
  born a quarter full with every bush full, two of three did after generation 60).
- Appetitive cells and grasp cost 7-15% when silenced in every seed: used, weakly.
- Supply over a life (seed 0): edible bushes full 32% at birth, then 14-19%; stomach 62-72%; 52% alive at the end;
  half of the bites on a stomach over 90% full.

### Stage 1.5 on the v25 footing: the safety cell works, but a life holds about one use of a lesson (2026-10-07)

Brain-only assay (`assay.py`, new 1.1 population, rate 0.3, two rounds), P(USE) with the bush adjacent, before ->
after -> after four more staple meals:

| bush | pain only | with the safety cell (variant `safety`) |
|---|---|---|
| staple | 0.37 -> 0.08 -> 0.08 | 0.37 -> 0.33 -> 0.79 |
| novel good | 0.37 -> 0.47 -> 0.47 | 0.37 -> 0.66 -> 0.66 |
| novel poison | 0.41 -> 0.06 -> 0.06 | 0.41 -> 0.06 -> 0.14 |

Generation 0 (`gen0.py`, 6 shared worlds; well-fed lifetime; good / poison berries per agent over the life;
learning off is the same population with the rates at zero):

| poison cost (food points; a good berry is +3) | off | rate 0.05 | rate 0.3 | safety, rate 0.3 |
|---|---|---|---|---|
| -3 | 300; 17.9 / 1.8 | 321; 19.0 / 1.8 | 301; 16.0 / 1.3 | 311; 17.0 / 1.4 |
| -9 | 206; 12.6 / 0.9 | 210; 12.8 / 0.9 | 205; 12.0 / 0.9 | 213; 12.4 / 0.8 |
| -18 | 164; 10.3 / 0.7 | 156; 9.6 / 0.6 | 156; 9.5 / 0.6 | 158; 9.6 / 0.6 |

Standard errors 37-60 over worlds: no variant differs from learning off. Unlike in the v24 world, learning no
longer lowers the eating of everything much (that was largely the famine).

Why nothing shows (`scripts/probes/lessons.py`, learning off, 3 worlds, poison -3): lifetime 517 of 2000; **2.1
visits to a good bush per life (5.6 bites each), 1.8 visits to a poison bush (1.0 bite each: the pain reflex ends
the visit), 0.95 later visits to a type that already hurt.** A perfect learner saves about one bite per life.
(The user's point, 2026-10-07: the cost of trying a bush is one berry, the gain six.) A bush holds 12 food units,
480 ticks of resting metabolism, so a life is a handful of bushes.

Smaller bushes with the density raised to match (`BUSH_BERRIES`, new constant, default 6) do not change this at
generation 0: 2 berries at density 0.42: lifetime 460, 3.6 good visits, 2.0 poison visits, 0.87 repeats; 3 berries
at 0.28: 568, 3.7, 2.3, 1.21; 2 berries with poison -9: 256, 1.8, 0.9, 0.13. The population is not adapted to these
worlds and dies early, so this is a weak test.

### Longer lives: the animals freeze and starve next to food (2026-10-07, late)

The user proposed longer lives (fewer generations to match), a lower metabolic rate and bushes of 2-3 berries, so
that a life holds many decisions. Probes with 6000-tick lives, populations not adapted to them.

`lessons.py`, the 1.1 population in the 1.5 world, learning off, poison -3 (density unchanged; regrowth scales
with bush size and metabolic rate, so the food flow per bush follows the metabolic rate):

| world | lifetime of 6000 | good-bush visits (bites each) | poison-bush visits | repeat visits to a type that hurt |
|---|---|---|---|---|
| as defined | 586 | 2.3 (5.7) | 1.4 | 0.66 |
| 2 berries per bush | 441 | 3.1 (2.0) | 1.1 | 0.35 |
| 2 berries, hunger 0.0125 | 1055 | 4.4 (2.0) | 1.7 | 0.70 |
| 3 berries, hunger 0.0125 | 1252 | 4.2 (2.9) | 2.2 | 1.07 |

`supply.py`, the 1.1 population in its own world, 6000 ticks: alive at tick 600 / 1800 / 3000 / 5400 and edible
bushes full at the same ticks:

| world | alive | edible bushes full |
|---|---|---|
| as defined | 65 / 29 / 9 / 0 % | 15 / 50 / 76 / 99 % |
| hunger 0.0125 | 82 / 51 / 38 / 21 % | 7 / 12 / 27 / 65 % |
| 3 berries, hunger 0.0125 | 79 / 61 / 52 / 39 % | 12 / 20 / 28 / 40 % |

The animals die while the bushes fill up: not a shortage. Trace of the as-defined run: of 166 animals that starved
after tick 600, 120 stood in one cell with USE on over 70% of the ticks 200 ticks before death, facing open ground
(81) or an empty bush (36), with 6.7 full bushes within 3 cells on average; median 780 ticks from the last meal
to death, about one stomach at rest. No pairs facing each other. In a normal 1000-tick life "USE at nothing" takes
6-15% of the ticks alive (1.0 and 1.1).

Reading: a life of 1000 ticks is 1.25 stomachs long in the 1.1 world (20 / 0.025 = 800 ticks at rest) and 2.5 in
1.0. An animal that fills up and then freezes still scores well under the well-fed lifetime, and reaches the cap,
so selection cannot remove the trap. This is the "staying put is a good strategy" issue and the ceiling at the
life cap. Lives have to be many stomachs long. The mechanism of the freeze itself is not traced yet.

## v26 trials (2026-10-07, night): long lives, and why the animals still do not learn

All single seed, cheap runs (user: no full reruns until a design looks final; and look inside a finished run,
beyond lesions, before starting the next).

**First trial of 1.5, a cliff:** hunger 0.05, 4000 ticks, 3-berry bushes, poison -6, safety cell, 50 generations.
Mean lifetime 100-300 of 4000 in main and control for all 50 generations. Doubling the metabolic rate and halving
the bush asked four times the visit rate of a population that already struggled. No generation-0 check was made
first; `lessons.py` shows it in three minutes.

**The parent has to be evolved under long lives first.** 1.1 population, own world, 4000 ticks, 30 generations
(`s1_1_valence_long/20261007-213445`): lifetime 1340 -> about 1900, alive at the cap 9 -> about 70 of 256. Supply
probe: 52% die in the first 1200 ticks (edible bushes 13% full), then survival is nearly flat (45% at tick 1600,
31% at 3600; bushes 25-39% full). Still climbing at generation 30.

**The learning world must not replace familiar food.** `lessons.py`, 4000 ticks, learning off, hunger 0.025:

| world | population | lifetime | good visits | poison visits | repeats |
|---|---|---|---|---|---|
| novel half of the supply, 6 berries, poison -3 | 1.1 | 586 | 2.3 | 1.4 | 0.66 |
| novel half, 3 berries, poison -6 | 1.1 | 453 | 2.7 | 0.9 | 0.24 |
| the same, born half full | 1.1 | 677 | 3.9 | 1.6 | 0.55 |
| novel half, 6 berries, poison -3 | 1.1 long | 696 | 3.3 | 2.1 | 1.13 |
| staple weight 3 at density 0.19 (familiar food as in 1.1), 3 berries, poison -6 | 1.1 long | 1310 | 15.8 | 3.5 | 2.05 |
| the same, 6 berries | 1.1 long | 1236 | 11.9 | 3.8 | 2.34 |

(Reference: the 1.1 population in its own world, 4000 ticks: lifetime 1286; the long-life one: 1908.)

**Second trial of 1.5** (`s1_5_association/20261007-214843` and control): W15 = W11 at density 0.19, staple weight
3, 3-berry bushes, poison -6, safety cell, 4000 ticks, 50 generations, from the long-life 1.1 population.
Generations 40-49: main 1470 vs control 1570; lifetime 1970 vs 2070; poison share 0.051 vs 0.049, first / second
half of life 0.057 / 0.042 vs 0.055 / 0.042. Both still climbing (832 -> 1470, 961 -> 1570).
Lesions, main (8 worlds; intact 1625): no_plasticity 1622, us_pain 1613, safety 1633; valence_av 58, no_feed 74,
ganglion_e 257, valence_app 514, grasp 575, ganglion_i 751, no_touch 992. The teacher is unused.

**Inside that run** (`scripts/probes/learned.py`, 64 recorded animals):
- Inherited learning rate, look -> aversive: mean 0.021, median 0.006 (start 0.05, cap 0.5): evolution turned
  learning down.
- Red, a novel poison: 47 animals bit it, 5.2 bites each; drive of its look on the aversive cells -0.37 inherited,
  +0.028 after the first bite; aversive activity facing it 0.05 before, 0.04 after. Inherited drives of the novel
  looks range from -0.38 to +0.58, so a lesson has to be large to matter.
- Pain cell 0.96 after a poison bite, 0.00 after a good one. Safety cell above 0.05 after 0.8% of good bites:
  aversive activity at the moment of a good bite is 0.001. An animal bites only while its aversive cells are
  silent, so "expected bad, was fine" never happens. That is why the safety cell did nothing in the world while
  it works in the assay, where the meals are scripted.

**Paper check with that all-or-none block** (`classify.py`, BLOCK=hard: bites only while aversion < 0.1; share of
lives in which the type is still eaten after 200 encounters: staple / novel good / novel poison):

| rule | rate 0.05 | rate 0.3 |
|---|---|---|
| pain only | 0.70 / 1.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| safety cell | 1.00 / 1.00 / 0.00 | 0.00 / 0.46 / 0.00 |
| prediction error | 1.00 / 1.00 / 0.00 | 0.00 / 0.64 / 0.00 |
| centred: dw = eta * pain * (look - usual look) | 1.00 / 1.00 / 0.00 | 1.00 / 1.00 / 0.00 |

With a hard block every uncentred rule refuses the staple for good at a rate large enough to matter against the
inherited drives; at a small rate it works on paper but is too weak in the brain. Only the centred lesson (the eyes
adapted to the look shared by everything met) is robust to the rate. Not yet implemented in the brain.

## v27-v28 (2026-10-07 night to 2026-10-08): where a lesson lands, and a world that makes learning worth having

All at generation 0 or single seed. Population: `s1_1_valence_long` (the 1.1 animals after 30 generations of
4000-tick lives). Probes added: `learned.py` (rates, weight changes per type, aversive responses, teacher
activity), `record.py` (one recorded generation-0 life on a brain variant), `lessons.py` (visits, repeats, value of
perfect learning), variants `centred`, `ahead`, `pulse`, `safety`; `DENSE=1` and `RUN=` in the environment.

### Three attribution faults, found by looking inside a recorded life

Generation 0 in the v26 trial world (staple half of the bushes), learning on vs off in the same 3 worlds, rate 0.3
unless stated; learning off: well-fed lifetime 746, good berries 52.9, poison berries 4.6.

| variant | well-fed lifetime | good berries | poison berries |
|---|---|---|---|
| as defined in v26 (whole look, all eyes, safety cell) | 456 | 26.5 | 2.1 |
| centred | 661 | 42.9 | 3.3 |
| centred, rate 0.1 | 744 | 50.4 | 4.1 |
| forward eye only (`ahead`) | 680 | 43.6 | 3.3 |
| ahead, rate 0.1 | 791 | 52.4 | 4.4 |
| ahead, centred | 705 | 47.6 | 3.9 |
| ahead, one-tick teacher (`pulse`) | 760 | 49.7 | 3.9 |
| ahead, pulse, centred | 749 | 52.0 | 4.2 |
| ahead, pulse, no safety cell | 759 | 49.9 | 3.8 |
| all eyes, pulse | 521 | 31.7 | 2.5 |
| all eyes, pulse, centred | 700 | 46.1 | 3.6 |
| ahead, pulse, every learned synapse present (`DENSE`) | 552 | 32.7 | 2.5 |
| ahead, pulse, centred, DENSE | 814 | 55.8 | 3.9 |
| all eyes, pulse, centred, DENSE | 687 | 45.7 | 3.2 |

Standard errors over the three worlds are 235-380 (not paired); read the table as directions.

1. **The shared look.** Every look is 0.45 gooseberry, so a lesson about a poison is 45% a lesson about the staple.
   Centring (the synapse learns from the look minus the slow average of each look input, tau 0.99) removes it. The
   forward eye sees something on 73% of ticks alive (53% a full bush, 44% an empty bush, 3% a berry); with that
   real average the spill onto the staple falls from 0.45 to about 0.10 (all ticks) or -0.07 (ticks with something
   in view). The average lives in the input neuron (its activity trace, `BrainConfig.in_trace_tau`);
   `ProjectionSpec.centred` makes a learned projection read it. Transmission is unchanged. The user prefers this
   mechanism, and would move it into the input neuron itself (everything downstream sees the adapted look) as the
   habituation stage.
2. **The side eyes.** All five eye columns had learned look -> aversive synapses. Going into a poison bite the
   forward eye sees the bitten type 80% of the time; each side eye sees something at 48-68% of bites, and 88-96%
   of that is another type, at full strength. `ahead`: only vis+0 learns.
3. **The fading pain.** With pain_decay 0.3 the teacher is 0.96, 0.54, 0.18, 0.05 on ticks +1..+4 after the bite;
   the forward eye is on the bitten type 99% at +1 and 8-11% from +2 on (another bush 56-63%). In a recorded life
   the staple's drive on the aversive cells rose by 0.29, the bitten poison's by 0.30. Fix adopted: pain lasts one
   tick in the 1.5 world (pain_decay 0; user's choice over a threshold on the teacher cell). The reflex is intact:
   1.0 bites per poison visit, lifetime 1264 vs 1310.

Also: only 42% of the forward look -> aversive synapses exist (the learned projection inherits the sparse evolved
wiring); an absent synapse cannot learn. With all present and the lesson centred, one bite raises the poison's
drive by 0.44 and the staple's by 0.11, but the aversive response facing it only goes 0.06 -> 0.13 (the innately
avoided types sit at a drive of 0.8-0.9, response 0.3-0.4): a lesson has to be about twice as large to block the
next bite. `Stage.dense_plastic`, and the learning rate may evolve up to 1.0 in 1.5.

The safety cell never fires in a living animal (above 0.05 after 0.8-4.5% of good bites): removed again.

### The economy made learning nearly worthless (user, 2026-10-08)

In the v26 trial world the animals ate eleven good berries for every poison one (4.5 good visits per poison visit,
three berries against one). Losing a fifth of the good berries to avoid a third of the poison bites costs 21 food
units and gains 5; break-even needs 2 good berries lost per bite avoided, measured 7.7. `lessons.py`: a perfect
learner would save 8% of intake. The world asked for near-perfect learning.

v28 world: only the four novel types, half of them poison each life, no staple and no ancestral poison; 6-berry
bushes; a poison bite costs two berries. Whoever cannot learn pays one bite on every visit to a poison bush for
life, and avoiding everything novel is starving.

`lessons.py` at generation 0 (learning off): the 1.1 animals starve there (lifetime 283-358 of 4000, about one
good bush per life; with the staple as a fifth of the bushes 414): they have no appetite for looks they did not
evolve with. The user: a cliff is acceptable if enough fitness signal is left to recover in a few dozen
generations.

### First one-seed runs in the v28 world: the first bite was a coin flip

Stage 1.5 (all eyes, centred, dense, rate 0.3 up to 1.0), variant 1.5f (forward eye only) and the control, 40
generations of 4000 ticks, born a quarter full. Generations 30-39: well-fed lifetime 145 / 102 / 105, mean
lifetime 341 / 303 / 295, no climb in any of them. Inside: median lifetime 130-160 ticks; first meal at tick 25;
20-25% never eat. Born with 5 food units, a first bite that is poison (one chance in two) costs 4 and the animal
is dead 40 ticks later, whatever its genome. USE is 60-75% of all ticks, moving 20%, facing a full bush 3-4%.
In main the inherited learning rate went up for the first time (0.30 -> mean 0.40, median 0.46); drive of the two
poison types after the first bite +0.35 and +0.43, of the two good types -0.04 and +0.04.
Correction: born half full in the 1.5 world (W15 start_food 0.5).

### v28, second and third one-seed runs: learning is used (2026-10-08)

**Born half full** (nothing else changed), generations 30-39: well-fed lifetime 249 (main), 221 (1.5f), 198
(control); lifetimes about 550 ticks, no climb. `forage.py`: the parent in its own world moves on 66% of ticks and
grasps at nothing on 19%; in the novel-only world main moves on 38% and grasps at nothing on 49%, the control on
8% and 89%; the bushes are 96-100% full from mid-life on.

**Two constants retuned** (generation 0, `lessons.py`, learning off):

| NOVEL_SIM | poison bite | lifetime | good / poison visits | repeat poison visits | perfect learning saves |
|---|---|---|---|---|---|
| 0.45 | -6 | 543 | 1.2 / 1.0 | 0.3 | 11% of intake |
| 0.65 | -6 | 503 | 2.1 / 2.3 | 1.1 | 20% |
| 0.8 | -6 | 409 | 2.5 / 3.4 | 1.9 | 28% |
| 0.8 | -4.5 | 480 | 2.8 / 4.3 | 2.7 | 27% |
| 0.8 | -3 | 591 | 3.4 / 6.0 | 4.3 | 23% |
| 0.65 | -3 | 678 | 2.8 / 3.6 | 2.2 | 15% |

NOVEL_SIM 0.8: a novel bush looks as much like the gooseberry as the six types of the 1.1 world do (at 0.45 the
1.1 animals hardly took it for food). POISON_FOOD -3 (one berry, 2 food units): at -6 a bite on a stomach under a
fifth full killed, a coin flip for a hungry animal when half of the bushes are poison.

**Third runs** (NOVEL_SIM 0.8, POISON_FOOD -3, born half full, 60 generations; `s1_5_association/20261008-001544`,
`_fwd`, `_control`), generations 50-59:

| | main (all eyes) | 1.5f (forward eye) | control |
|---|---|---|---|
| well-fed lifetime | 485 | 462 | 272 |
| lifetime of 4000 | 808 | 804 | 576 |
| poison share of meals | 0.17 | 0.15 | 0.26 |
| poison share, first / second half of life | 0.20 / 0.04 | 0.18 / 0.04 | 0.26 / 0.07 |

Main is ahead from the first generations (0-4: 539 vs 354). No run climbs over the 60 generations.
Lesions of main (8 worlds; well-fed lifetime, poison share): intact 787, 0.13; **no_plasticity 458, 0.22; us_pain
458, 0.22**; valence_av 85; no_feed 90; ganglion_e 389; no_touch 499; ganglion_i 509; grasp 553; valence_app 570.
The first run in which the teacher is used: without it the animals lose 42%.
Inside (`learned.py`): learning rate mean 0.24, median 0.18 (start 0.3, cap 1.0; range 0.01-0.75). A poison type's
drive on the aversive cells rises by 0.27-0.41 after the first bite, a good type's by 0.16; the aversive response
facing the poison goes 0.13 -> 0.22 and 0.22 -> 0.26; a poison type is still bitten 2.5-2.8 times. All eyes and
forward eye only do equally well, so all eyes stays (the user's preference); 1.5f is kept as a variant.

### The freeze is biting at an empty bush, and it is resting (`scripts/probes/freeze.py`)

Episodes of 50+ ticks in one cell with USE and no full bush in front:

| population | share of ticks alive | in front: empty bush / open ground | ends in death | stomach at start / end |
|---|---|---|---|---|
| 1.0 (v25) | 7% | 7% / 93% | 10% | 80% / 55% |
| 1.1 (v25) | 29% | 64% / 28% | 24% | 70% / 54% |
| 1.1 long | 15% | 53% / 42% | 10% | 73% / 56% |
| 1.5 main (third run) | 45% | 91% / 5% | 15% | 61% / 39% |

Mid-freeze in 1.5: appetitive cells 0.51, grasp 0.83, no_touch 0.05, aversive 0.00; median length about 100
ticks; 85% end with the animal moving on. An empty bush carries the colour of the full bush of its type, the value
cells answer to the colour, and the contact-gated grasp programme bites. Grasping at nothing costs no energy
(eat_cost is charged per bite that yields), moving does, so standing at an empty bush is the cheapest thing a fed
animal can do; nothing tells it when to stop. Two planned stages address this without new mechanisms: hunger (rest
when fed, forage when hungry) and habituation (the pull of what yields nothing fades). Hunger probably belongs
before the learning stage; not decided (the user's decision, STATUS).

### Chain for the final runs

1.0 -> 1.1 (v25, unchanged; 1.1 three seeds) -> **1.1l** (new: the 1.1 brain and world under lives of 4000 ticks,
60 generations, no control: an adaptation step) -> 1.5 (40 generations, three seeds, control = the 1.1 brain).

## v29 (2026-10-08): long lives as a stage, stage 1.5 over three seeds, a drives trial

### 1.1l long lives (`s1_1_valence_long/20261008-010357`, 60 generations from the 1.1 lineage run)

The 1.1 brain in the 1.1 world, 4000 ticks. Well-fed lifetime 838 -> 1600, lifetime 1340 -> 2220, alive at the cap
9 -> 97 of 256. `forage.py`: moving on 74% of ticks, grasping at nothing 12%, 110 berries per life; `freeze.py`:
freezes 7% of ticks alive (29% in 1.1), 96% end with the animal moving on. In a familiar world long lives select
the standing still away. No new circuit, no control: an adaptation step, the parent of every long-lived stage.

### 1.5 association, three seeds (40 generations of 4000 ticks; `python -m life.book export 1.5 --evaluate`)

World: four novel types only (new look and meaning per life, two poison), NOVEL_SIM 0.8, 6-berry bushes, poison
bite -3, pain one tick, born half full, density 0.19. Brain: pain teaches the centred look -> aversive synapses,
all eyes, every learned synapse present, rate from 0.3 (cap 1.0). Control: the 1.1 brain.

| | main | control |
|---|---|---|
| well-fed lifetime, last 50 generations window | 427 +-24 [474 395 412] | 320 +-3 [321 314 324] |
| re-evaluated in 8 shared worlds | 483 +-38 [431 558 460] | 308 +-18 [281 300 343] |
| lifetime of 4000 | 770 +-29 | 619 +-3 |
| poison share of meals | 0.19 [0.19 0.19 0.20] | 0.25 [0.25 0.25 0.26] |
| main - control | +107 +-23 (runs), +175 +-42 (re-evaluated), head to head +71 +-50 [-29 +126 +115] | |

Lesions (% of intact, per seed, mean): **no_plasticity 57 42 54 (51); us_pain 57 42 54 (51)**; valence_av 18 12 16
(15); no_feed 18 13 17 (16); ganglion_e 30 41 44 (38); no_touch 41 50 45 (45); ganglion_i 66 63 61 (63);
valence_app 67 73 80 (73); grasp 66 75 81 (74).

- **Passes.** Used: without learning the animals lose half their fitness, in every seed. Not worse: main is ahead
  of control in the runs and in the re-evaluation in every seed; head to head in two of three.
- Neither main nor control changes detectably over the 40 generations. Fitness swings between generations with a
  standard deviation of about 290 (new worlds and looks each generation), so a block of ten generations has a
  standard error of about 90; an apparent decline in seed 0 (652 in generations 0-4, 469 in 30-39) is inside
  that. Main is ahead from generation 0: the rule works as designed, evolution did not have to find it.
- How much of fitness is the genome (`repeat.py`, a genome's fitness, mean of four siblings, between four
  worlds): r = 0.17 in the 1.5 world, 0.18 in the 1.1l world; per single animal r = 0.01 (`evo_check.py`). Born
  full (0.06), density 0.3 (0.20) or both (0.21) do not change it beyond the noise of the measurement.
- Lives are short (770 of 4000) and 45% of the ticks alive are spent biting at an empty bush (above).
- Learned rates at the end, seed 0: mean 0.24 (start 0.3).

### 1.7 drives, trial (one seed, 40 generations; `s1_7_drives/20261008-015359` and control)

Circuit from the generation-0 probe (cold, warm_seek, rest reading the skin with weight 9, hungry at 0.67 shutting
warm_seek and rest) on the 1.5 brain, plus evolvable synapses from `hungry`, starting at zero, onto grasp, no_feed,
valence_app and the ganglion. World: the 1.5 world made cold (ambient 0.25), hot springs at spawn weight 0.1.
Generation 0 on the 1.5 seed-0 population (`drives.py`, 4 worlds): world not cold 287; cold, not sensed 172; the
circuit 209 (body 0.43); the circuit without the hunger gate 81.

Generations 30-39: main 309 vs control 186; lifetime 572 vs 397; body temperature 0.42 vs 0.36; poison share 0.22
vs 0.23. Main rises a little (209 in generations 0-4).
Lesions of main (8 worlds; intact 270): **hungry 74 (27%)**; rest 156 (58%); cold 236 and warm_seek 236 (88%);
no_plasticity 256 and us_pain 256 (95%).
Inside: hungry active on 46% of ticks alive, rest 52%, cold 28%, warm_seek 11%. `hungry -> warm_seek` was scaled
by evolution from the designed 6 to a mean of 10. The evolvable `hungry` -> feeding synapses are at 0.06-0.16
(what mutation alone gives from zero in 40 generations): hunger does not gate feeding yet. `hungry` is an
inhibitory cell, so it can release the bite block but cannot stop a fed animal biting directly; whether that or
the short run is the reason is open. Grasping at nothing is still 43% of ticks; median lifetime 408.
Learning matters little here (95% without it): lives of 400-600 ticks leave few repeats.

## v30 (2026-10-08, early morning): habituation by adapting look inputs; hot springs were a trap

### 1.6h habituation, three seeds (40 generations of 4000 ticks; parent 1.5; control = the 1.5 brain)

The user's preferred form of the centred lesson (2026-10-07), moved into the input neuron: every look input
(`vis*.app*`) passes on its input minus its slow average (`BrainConfig.in_adapt`, `in_trace_tau` 0.98, about 50
ticks). Everything downstream sees the adapted look; the learned synapses are plain again (no `centred`). The
world is the 1.5 world. Generation 0 (`repeat.py`): the 1.5 population drops from 528 to 264 with adapting inputs
(its synapses were shaped on the raw look); between worlds a genome's fitness correlates r = 0.40.

| | main | control |
|---|---|---|
| well-fed lifetime, all 40 generations | 500 +-18 [520 515 465] | 400 +-25 [449 374 376] |
| re-evaluated in 8 shared worlds | 597 +-52 [493 651 647] | 518 +-5 [528 512 513] |
| poison share of meals | 0.16 | 0.20 |
| main - control | +100 +-21 (runs), +79 +-57 (re-evaluated; seed 0 -36) | |

Seed 0 by block of generations: main 374, 436, 562, 512, 602; control 460, 417, 502, 402, 455: main starts behind
and overtakes, the first run in the learning world that improves over the generations.

**The lesion of the adaptation** (`adapt_off.py`: the final populations with the adaptation switched off, same
animals, same 4 worlds): 593 -> 203, 721 -> 224, 638 -> 223: **34, 31, 35% of intact**.
Region lesions (% of intact): no_plasticity 39 41 39 (40); us_pain the same; valence_av 11 9 11 (10); no_feed 12;
ganglion_e 30; ganglion_i 63; valence_app 72; grasp 73; no_touch 85. Learning is used more than in 1.5 (51).

How a life is spent (recordings of the last generation, per seed; main vs control): moving 64% [71 71 49] vs 51%;
standing and biting at nothing 19% [11 11 35] vs 36% [40 25 42]; berries per life 62 vs 39; lifetime 910 vs 851.
(For 1.5's final runs the same measure gives 31% +-6 for main and 46% for its control; the 45% quoted earlier was
from a trial run.) `freeze.py`, seed 0: freezes of 50+ ticks take 7% of the ticks alive.
`learned.py`, seed 0: aversive response facing a poison type 0.06 -> 0.23 and 0.00 -> 0.27 after the first bite;
facing the good types 0.04 and 0.10 throughout.

- **Passes**: used in every seed; main above control in the runs in every seed and in the re-evaluation in two of
  three.
- The head-to-head test is not valid here: both halves run on the new brain, so the control animals get adapting
  inputs their synapses were never tuned to (main half 975, control half 317). It is left out of the book page
  (`life.book.NO_HEAD_TO_HEAD`).
- Seed 2 keeps more standing still (35%). Median lifetime is below the mean (364 vs 906 in seed 0): more animals
  die early and the others do much better.

### Hot springs were a trap

The drives probe gave the habituation population 247 in the drives world with the cold switched off, against 733
in its own world. Separated: the extra body senses change nothing (733); hot springs at spawn weight 0.1 (2% of
objects) give 277, with the animals facing a spring on 33% of their ticks and biting at it on 13%. A hot spring
was the only object that blocks movement (bushes do not): the animals walked into it and stayed. This was in both
earlier drives trials and in the probe of 2026-10-03: most of "the cost of cold" was the trap, and part of what
the circuit won back was `rest` stopping the pushing. Fix: a spring does not block (`berry_world`).

Generation 0 with that fix (`drives.py`, habituation population, 4 worlds): world not cold 571; cold, not sensed
317 (-44%, body 0.34); circuit with the hunger gate 382 (body 0.40); without the gate 63. At ambient 0.35: 401 and
462.

### 1.7 drives, trial on the habituation brain with non-blocking springs (one seed, 40 generations)

`s1_7_drives/20261008-044432` and control (the habituation brain). By block of generations: main 373, 382, 327,
407; control 245, 314, 231, 332; body temperature 0.40 vs 0.35. Lesions of main (intact 358): hungry 94 (26%);
no_plasticity and us_pain 261 (73%); rest 308 (86%); cold and warm_seek 338 (95%). Main stands still on 35% of
ticks (control 12%): `rest` holds it where it is warm.
On one seed: `hungry` and `rest` are used, `cold` and `warm_seek` marginally. Not replicated: how hunger should
act on feeding is open (the evolvable `hungry` -> feeding synapses were not looked at again in this run).
(The two earlier drives trials, on the 1.5 brain and on the habituation brain with blocking springs, gave main 309
vs control 186 and 307 vs 187; superseded.)

