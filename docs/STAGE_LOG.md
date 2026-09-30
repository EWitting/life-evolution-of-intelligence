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
