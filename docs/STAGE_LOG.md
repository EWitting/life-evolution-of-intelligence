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
act on feeding is open. The evolvable `hungry` -> feeding synapses are at a mean size of 0.04-0.17 after 40
generations (grasp 0.17, valence_app 0.11, ganglion 0.07, no_feed 0.04), about what mutation alone gives from zero:
no sign that evolution uses them.
(The two earlier drives trials, on the 1.5 brain and on the habituation brain with blocking springs, gave main 309
vs control 186 and 307 vs 187; superseded.)

## v31 (2026-10-08, morning): the check for the rerun order; the plan

The user agreed to a complete rerun of chapter 1 with habituation before association (`docs/RERUN_PLAN.md`, removed 2026-10-17; in git history).
The one risk checked first: does habituation still pay in the familiar 1.1 world, where long lives alone had
already cut the standing still to 7% of ticks?

**1.2h, trial, one seed** (`s1_2_habituation/20261008-091643` and control): adapting look inputs on the 1.1 brain,
the 1.1 world, lives of 4000 ticks, 40 generations, from the 1.1l population; control the 1.1 brain.
By block of generations (0-4, 5-9, 10-19, 20-29, 30-39): main 984, 1479, 1545, 1518, 1610; control 1546, 1614,
1649, 1591, 1681. Alive at the cap 97 vs 106 of 256. Main recovers from the setback within about ten generations
and ends level with the control, not ahead.
`adapt_off.py` (3 worlds): adaptation on 1593, off 1345: **84% of intact** (31-35% in the novel-food world).
`forage.py`: standing and biting at nothing 8% of ticks vs 21%; moving 75% vs 61%; berries per life 127 vs 122;
alive at the end 48% vs 47%.
Reading: used, weakly; not worse within the noise of one seed. Enough to go ahead with the planned order and
judge it over three seeds; the fallback order (association with the centred rule, then habituation) has passed
with three seeds.

Book (all published): the 1.0 page shows the scratch run followed by the settling run; "Watch a main / control
run" buttons, run directories folded away; light and dark theme with Quarto's switch; buttons and dropdown
readable in dark mode; the zoomed-in world tile repaints when the theme is switched.
`adapt_off.py` now also works for worlds without per-life looks.


## v32 (2026-10-08, afternoon): the rerun starts; a dud bush gives habituation its job

Chain as now defined in `stages.py`: 1.0 (scratch, 1000 ticks; then a settling run at 4000) -> 1.1 (4000 ticks, 50
generations) -> 1.2h habituation -> 1.5 association on the habituation brain (plain learned synapses) -> 1.7
drives. Side paths kept defined: 1.1l, 1.5c (the centred association of the old order), 1.5f, 1.6h. The user
(2026-10-08): the old order is also acceptable if it ends up working much better; the order below is preferred.

### Which world for habituation (one seed each, old lineage `s1_1_valence_long`, 40 generations of 4000 ticks)

The user's idea: novel berry colours already in the habituation stage, to give the adaptation more to do. Tried
with existing world code only (`berry_world(novel_looks=..., duds=...)`).

| world | main, last 4 generations | control | adaptation off, % of intact |
|---|---|---|---|
| the 1.1 world (v31) | 1650 | 1710 | 84 |
| good types 1-3 with a colour drawn per life | 1198 | 1402 | 72 |
| all four good types so | 1177 | 1512 | 78 |
| the 1.1 world + a dud bush type (colour per life, never a berry) | 1369 | 1103 | 34 |
| good types 1-3 novel + the dud | 1070 | 1163 | 51 |

- Novel colours on good food do not help: the adaptation is used a little more and main ends behind its control.
- The 31-35% of the old order (v30) was mostly learning: there the learned synapses relied on the adapted look,
  and no_plasticity alone was 40% of intact.
- The dud (the world of the parked stage 1.4, density raised by 7.5/6.5 so the food stays the same) is the case
  habituation is for. Adopted with the user's agreement: it shows why habituation works, and it makes the step to
  association smaller.
- Association is not disturbed at generation 0 (`lessons.py`, STAGE=1.5, learning off, 3 worlds): the population
  from the dud world lives 668 ticks in the association world and has 5.2 later visits to a poison type that
  already hurt; the one from the plain world 514 and 5.9. The association world is unchanged.

### Seed 0 down the new chain

- **1.0 from scratch** (`s1_0_steering/20261008-131127`, 400 generations of 1000 ticks): well-fed lifetime
  530-555. **Settled** (`20261008-134551`, 60 generations of 4000 ticks, mutation 0.1): about 2100, lifetime
  2600, 45% alive at the cap; moving 62% of ticks, freezes 11% of ticks, 96% of them on open ground with the
  stomach 85% full at the start, 90% end with the animal moving on.
- **1.1** (`s1_1_valence/20261008-140718`, 50 generations of 4000 ticks): main 280 -> 1550-1600; control (pain
  blind) about 170, lifetime median 199, no avoidance by look. Lesions (4 worlds, % of intact): valence_av 9,
  no_feed 18, ganglion_e 29, ganglion_i 31, grasp 89, valence_app 92, no_touch 102. Moving 68%, biting at
  nothing 16%; freezes 12% of ticks, 86% at an empty bush, 97% end with the animal moving on.
- **1.2h habituation, dud world** (`s1_2_habituation/20261008-142723`): by block of 4 generations main 477, 978,
  1020, 1208, 1240, 1354, 1184, 1311, 1344, 1494; control 530, 1037, 819, 1170, 1237, 1450, 1082, 1273, 1304,
  1069. `adapt_off.py` (3 worlds): 1365 -> 513, **38% of intact**. The control swings with the world (last three
  generations 1642, 1458, 469; main 1501, 1472, 1478): when the dud's colour of that life is attractive the
  animals without adaptation stay at it. Recorded last generation: biting at a dud on 26% of ticks (main) and
  52% (control, its 469 generation). `forage.py` and the book's life table counted the dud as a full bush; fixed.
- **1.5 association** (`s1_5_association/20261008-150424`): by block main 892, 696, 453, 994, 543, 512, 571, 363,
  526, 680; control 622, 283, 306, 630, 492, 249, 412, 240, 352, 329. Main ahead from generation 0 in every
  block. Lesions not measured yet.
- Stopped by the machine running low on memory (not finished, to be started again): 1.7 drives seed 0 (at
  generation 20 of 40: main 159, control 93), 1.1 seed 2 (seed 1 is finished), the lesions of 1.5.

### Three seeds of the new chain (2026-10-08, evening; `python -m life.book export <key> --evaluate`, 8 worlds)

Seeds 1 and 2 of every stage start from the seed-0 run of the parent stage. Percentages are fitness with the
region silenced, % of intact, per seed.

**1.1 valence** (seed 0: 50 generations; seeds 1 and 2: 100, because at 50 they were at 768 and 909 and still
climbing while seed 0 had found the look -> aversive wiring by generation 10; the runs record what they continue
in `continues.txt` and the book joins the curves).

| | main | control |
|---|---|---|
| well-fed lifetime, last 50 generations | 1139 +-40 [1216 1124 1079] | 176 +-6 [166 185 176] |
| re-evaluated | 1479 +-105 [1572 1594 1270] | 190 +-12 |
| head to head | +1919 +-155 [2074 2074 1610] | |

Lesions: valence_av 9 9 12; no_feed 17 10 13; ganglion_e 31 28 62; ganglion_i 29 90 53; grasp 90 71 88;
valence_app 90 73 93; no_touch 99 91 73. Moving 67% of ticks, biting at nothing 15% (control 14% and 76%).
**Passes.** No control found avoidance by look in 50-100 generations of long lives.

**1.2h habituation, dud world.**

| | main | control |
|---|---|---|
| well-fed lifetime, all 40 generations | 1139 +-15 [1161 1146 1111] | 1073 +-35 [1097 1119 1003] |
| re-evaluated | 1361 +-56 [1463 1269 1350] | 1272 +-36 [1245 1227 1343] |
| main - control, re-evaluated | +89 +-65 [218 42 8] | |

`adapt_off.py` (4 worlds): 1392 -> 494, 1275 -> 698, 1326 -> 481: **36, 55, 36% of intact**. Lesions: valence_av
9 11 9; no_feed 20 20 14; ganglion_e 16 12 28; ganglion_i 34 61 22; grasp 83 76 89; valence_app 85 78 89;
no_touch 100 100 100. Recorded last generation: moving 57% vs 29%; biting at nothing (duds included) 28% vs 62%;
lifetime 1901 vs 1174. **Passes**: used in every seed; main at or above control in every seed, by a small margin
in two. No head to head for this stage (the change is in the inputs).

**1.5 association** (on the habituation brain, plain learned synapses).

| | main | control | old order (v29) |
|---|---|---|---|
| well-fed lifetime, 40 generations | 592 +-20 [623 599 555] | 354 +-19 [392 331 339] | 427 vs 320 |
| re-evaluated | 721 +-68 [607 715 841] | 360 +-5 | 483 vs 308 |
| main - control, re-evaluated | +361 +-66 | | +175 +-42 |
| head to head | +520 +-120 [290 696 573] | | +71 +-50 |
| poison share of meals | 0.13 0.13 0.15 | 0.23 0.24 0.24 | 0.19 vs 0.25 |

Lesions: **no_plasticity 43 48 43; us_pain the same**; valence_av 13 18 12; no_feed 16 26 15; ganglion_e 40 47
50; ganglion_i 33 81 48; valence_app 90 93 102; grasp 90 95 107; no_touch 92 95 98. Seed 0 inside (`learned.py`):
the inherited learning rate rose from 0.3 to a mean of 0.60 (median 0.70; it fell to 0.24 in the old order);
aversive activity facing a poison type 0.12 -> 0.39 and 0.04 -> 0.35 after the first bite, facing the good types
0.05-0.12. Recorded last generation: moving 69% vs 59%, biting at nothing 13% vs 22%, lifetime 1168 vs 597.
**Passes**, more clearly than in the old order.

**Trim: nothing is removed.** `no_touch` is at 100 in habituation but at 73-99 in 1.1 and 92-98 in association;
the appetitive cells and `grasp` are at 71-93 in 1.1 and 76-89 in habituation. Every cell is used somewhere on
the chain.

**Frozen** (user, 2026-10-08: a partial freeze, so that 1.0, 1.1 and 1.5 are not rerun again): 1.0, 1.1, 1.2h,
1.5. `life/freeze.py`, `frozen/` (definitions and final populations of every seed), `tests/test_frozen.py`.
The chapter exam is dropped as a criterion (user: later populations carry equipment for things earlier worlds do
not contain, so some regression means nothing); a sanity check at most.

**1.7 drives** (on the frozen 1.5; not frozen).

| | main | control |
|---|---|---|
| well-fed lifetime, 40 generations | 386 +-14 [392 359 408] | 309 +-18 [325 329 273] |
| re-evaluated | 401 +-2 [400 404 399] | 341 +-38 [276 409 337] |
| main - control, re-evaluated | +60 +-37 [124 -4 62] | |
| head to head | **-65 +-33** [-88 -108 0]: main half 444 402 432, control half 532 510 432 | |

Lesions: hungry 26 27 18; rest 72 86 71; cold and warm_seek 89 94 86; no_plasticity and us_pain 71 66 76;
valence_av 25 22 25; no_feed 27 24 27; grasp 80 75 69; valence_app 85 79 72; no_touch 89 91 90; ganglion_e 23 43
26; ganglion_i 27 37 30. Recorded last generation: moving 45% of ticks vs 71%; biting at nothing 32% vs 12%;
berries per life 32 vs 43; lifetime 600 vs 590; body temperature 0.39.
`strengths.py`: rest -> out x1.09, x0.99, x0.98 of its designed strength; hungry -> rest x1.43, x1.15, x1.26;
hungry -> warm_seek x1.10, x1.25, x1.64; warm_seek -> out x0.92, x0.56, x0.98.
`density.py 1.7` (new; each population alone in the 8 head-to-head worlds, at full and at half the number of
animals): main 519 -> 873, 401 -> 710, 490 -> 763; control 416 -> 746, 430 -> 666, 352 -> 598. Halving the
animals raises fitness by 55-80%: food is a race in the first part of life.

Reading: used (hungry and rest clearly, the warmth pair by about 10% in every seed) and ahead of the control when
each lives among its own. In a shared world the control half does better than among its own (432-532 vs 352-430)
and the main half slightly worse than among its own (402-444 vs 401-519): animals that rest when fed eat less
(32 vs 43 berries) and leave the bushes to the roamers; what they gain is a warm body that burns less. Within
main populations resting was not selected down. Not shown: whether a roamer could invade a population with
drives. Left for the user: freeze as it is, or tune (a retune reruns drives only).

**Drives frozen as it is** (user, 2026-10-08, evening): the control wins the head-to-head by being greedy in a food
race; the animals with drives are the more efficient ones among their own. `frozen/1.7`, tag `chapter-1-drives`.

## v33 (2026-10-12): affect, before any design: what could a lasting state be worth?

Design frame agreed with the user: affect is built on the frozen drives brain as a second page of the same system
(a need selects a mode; then modes become states that last and that events can set). Names: serotonin for a
"fed, stay" state; dopamine is kept for the learning signal of chapter 2; plain names otherwise (`hungry`, `cold`).
The hunger-driven roam nucleus of the old stage 1.3 is dropped: `hungry` does that job.

New probe `scripts/probes/dwell.py` (from a recording: how visits end, the wait for the next meal, clumping).
`record.py` takes world overrides. No evolution was run.

Last generation of drives, main seeds 0 / 1 / 2 (control seed 0 in brackets):

| | main | control |
|---|---|---|
| bites per visit to a good bush (a bush holds 6) | 5.5 / 5.6 / 5.4 | (5.6) |
| visits that leave berries behind | 12% / 9% / 12% | (5%) |
| stomach after such a visit | 81% / 73% / 96% full | (74%) |
| an emptied bush stays empty (median) | 960 ticks | |
| good bushes with berries within 3 cells of a meal, vs the whole map | 0.038 vs 0.075, 0.048 vs 0.069, 0.046 vs 0.066 | (0.035 vs 0.071) |
| next visit is to poison, after a poison bite / after a good meal | 67 / 59%, 60 / 52%, 62 / 51% | (63 / 49%) |
| wait after a poison bite (median) | 17 / 19 / 13 ticks | (22) |

- **Staying at food is already done**: the grasp programme empties a bush in one visit, one bite per tick.
- **One bush is a full stomach**, so after a meal the animal is not looking for food; `hungry` going silent already
  hands over to the warmth mode. A "fed, stay" state would repeat that.
- **Nothing to search for nearby**: food is spread evenly, the spot just eaten is poorer than average, and the bush
  is empty for 960 ticks.
- **Nothing to flee from**: poison is not clumped either, and the animals already leave a poison bush within about
  20 ticks. (The slightly higher poison share after a poison bite is probably early life, before the lesson.)

So in the drives world neither a "search here" nor a "get away" state can pay, whatever the circuit.

One generation-0 life of the drives population (seed 0, 64 animals, one world each, so the fitness is a rough
figure) with the existing `patches` option, which puts all objects, hot springs included, into discs of radius 6:

| world | well-fed lifetime | good bushes within 3 cells of a meal vs whole map | next bush (median) | warm_seek on |
|---|---|---|---|---|
| drives world | 717 | 0.054 vs 0.062 | 14 cells | 15% |
| 80 patches | 572 | 0.100 vs 0.071 | 8 cells | 11% |
| 30 patches | 298 | 0.197 vs 0.076 | 3 cells | 5% |

Patches make "food here means food nearby" true (up to 2.6 times the map average) but poison is mixed in just as
much (poison around a poison bite 0.213 vs 0.082 at 30 patches), so a meal and a poison bite say the same thing
about the place. Open, for the user: what the affect world is (see STATUS).

### The affect world: one berry type per patch (user, 2026-10-12)

New world option `patch_types` (`life/world.py`): the patches are shared out over the object types by spawn weight
and a cell belongs to the nearest patch centre; heat sources stay spread evenly. Draft stage 1.8 (`s1_8_affect`):
the drives world with 80 patches of radius 6, brain still the drives brain. `tests/test_frozen.py` and
`life.freeze` now read a stored configuration with options added since at their defaults
(`freeze.with_defaults`): a new option that is switched off does not change a frozen stage. The rules digests of
the frozen stages are unchanged (10 tests pass).

No cliff (`calib.py 1.8`, drives population seed 0, 4 worlds): own world 326; 80 patches of radius 6: 292; 40 of
radius 8: 276; 160 of radius 4: 299; 240 of radius 3: 305.

What a bite says about the place (`dwell.py`, one generation-0 life, 80 patches of radius 6): bushes with berries
within 3 cells, good / poison: around a good meal 0.165 / 0.009, around a poison bite 0.013 / 0.201, whole map
0.062 / 0.093. The next visit is to poison after 76% of poison bites and after 10% of good meals.

### Generation 0: two hard-wired lasting states (`scripts/probes/affect.py`)

Drives population (seed 0), unchanged, 4 shared worlds, paired difference in well-fed lifetime to the drives brain.
`alarm<d>`: pain -> modulator `alarm` (cleared by the factor d per tick) -> `flee` cell -> FORWARD.
`stay<d>`: taste -> `fed` cell -> modulator `serotonin` -> receptor on `rest`, which then holds FORWARD back also
where it is cold; `hungry` still shuts `rest`.

| variant | patch world (base 394) | poison share (base 0.110) | world without patches (base 470) |
|---|---|---|---|
| alarm0.9 (about 20 ticks) | +46 +-8 | 0.091 | +9 +-21 |
| alarm0.97 (about 70 ticks) | -30 +-22 | 0.098 | |
| stay0.97 | +73 +-44 | 0.089 | |
| stay0.99 (about 200 ticks) | +200 +-62 | 0.087 | +75 +-44 |
| alarm0.9 + stay0.97 | +117 +-23 | 0.077 | |
| alarm0.9 + stay0.99 | +252 +-29 | 0.069 | |

- Both states pay before any evolution, and together more than apart. Four worlds: sizes are rough, the signs of
  the paired differences are the same in every world for alarm0.9, stay0.99 and the combinations.
- The alarm is about the place (nothing without patches) and must be short: at 70 ticks it runs the animal out of
  the good patches as well.
- About a third of what staying wins is not about the place (+75 without patches: an animal that sits still when
  fed spends less on moving); the rest is.
- Modulator persistence (`ModulatorSpec.decay`: released at once, cleared slowly) gives the fast rise and slow
  fade that the old `raphe` cell lacked. Neither modulator teaches anything.

### Stage 1.8 affect as defined; seed 0 (2026-10-12)

Definition (user, 2026-10-12): pain (`us_pain`) -> modulator `alarm` (decay 0.9) -> `flee` -> FORWARD; taste ->
`fed` -> modulator `serotonin` (decay 0.99) -> `settle` -> `rest`. Control: the drives brain in the patch world.
Evolution may tune everything (user): the strengths are tunable hard-wired projections, and the fade time of a
persistent modulator is now a gene (`Genome.clear`, one value per modulator; the time constant 1 / (1 - decay)
mutates like a tunable strength, within `tune_range` of the designed value). Populations stored before the gene
existed load with the designed decay; 45 tests pass, the drives brain gives the same numbers as before the change.

Generation 0 of the stage as defined: +250 +-35 over the drives brain (4 shared worlds; 644 vs 394).

Seed 0, 40 generations, fitness in blocks of ten generations: main 766, 777, 809, 983; control 451, 486, 512, 659.
Poison share of the berries eaten: main 0.056 -> 0.046, control 0.115 -> 0.077. About 23 s per generation with
two processes.

Inside the run: the fade times stayed close to the design (alarm 0.879, about 8 ticks; serotonin 0.988, about 84
ticks). Strengths: `flee` -> out 3.0 -> 3.5, `settle` -> `rest` 3.0 (unchanged), `rest` -> out 9 -> 7.3,
`warm_seek` -> out 3 -> 2.3. Both populations move little in the last generation (16% and 14% of ticks) and both
have `rest` on for over 60% of ticks: the control found its own way to stay put.

Lesions, seed 0 (4 worlds, % of intact 967): `flee` 84, `fed` and `settle` 64 (one chain), `rest` 40, `hungry` 43,
`cold` 102, plasticity off 105. Both new states are used. In this seed the warmth-seeking pair and the learned
aversion no longer add anything in the patch world: staying in a good patch replaces both. To check over the three
seeds.

### Stage 1.8 affect over three seeds; chapter 1 frozen (2026-10-12)

Seeds 1 and 2 from the same drives population (seed-0 run). 40 generations each, about 20 s per generation.
`book/data/1.8.json` (export with evaluation: 8 shared worlds per seed).

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| main, last 10 generations | 983 | 906 | 941 | |
| control, last 10 generations | 659 | 681 | 538 | |
| main, whole run | 834 | 727 | 747 | 769 +-33 |
| control, whole run | 527 | 519 | 409 | 485 +-38 |
| re-evaluated, main | 956 | 876 | 925 | 919 +-23 |
| re-evaluated, control | 660 | 591 | 624 | 625 +-20 |
| re-evaluated, difference | +297 | +285 | +301 | +294 +-5 |
| head to head, main half - control half | +604 | +258 | +583 | +482 +-112 |

Lesions, % of intact, per seed: `fed` 63 54 66, `settle` 63 54 64, `flee` 81 86 89, `rest` 41 36 37, `hungry`
41 54 41, `us_pain` 79 77 81, `valence_av` 76 68 68, `no_feed` 82 76 81, `valence_app` 92 87 90, `grasp` 92 89 89,
`no_touch` 97 95 95, `ganglion_e` 16 24 17, `ganglion_i` 10 32 52. **Not used in this world: `cold` and
`warm_seek` 101 105 107, plasticity off 99 95 99.**

After a bite (recordings of the last generation, mean of three seeds, main / control): steps in the 20 ticks after
a poison bite 15.3 / 11.2; after the last bite of a good meal 1.5 / 5.1; the next bush after a poison bite is poison
again in 58 / 64%; after a good meal in 11 / 11%.

Reading: passes clearly, in every row and every seed (the first stage since valence where the head to head agrees
with the rest). Serotonin does most of the work; the alarm adds less. Staying in a good patch replaces two things
the lineage had: walking for warmth when fed, and knowing a poison by its look. The cells stay (frozen stages);
the consequence is for the world of chapter 2, where learning needs poison standing between good bushes.

Frozen as `frozen/1.8` (user, 2026-10-12: complete and freeze chapter 1), tag `chapter-1`. 47 tests pass.

Tools changed on the way: `life.book` evaluates in parallel processes (`--jobs`, default 5) and prints a line per
finished task; a single process took about four times as long and showed nothing until the end.

## v34 (2026-10-12, evening): chapter 2 begins; which world gives learning a job again?

New: `WorldConfig.patch_purity` (the chance that an object in a typed patch is the patch's own type; 1 = the 1.8
world, unchanged) and `scripts/probes/world2.py` (a frozen population in a variant of the 1.8 world, intact and with
one circuit silenced). Seed 0, four worlds (eight where stated), well-fed lifetime, % of intact.

The affect population (1.8):

| world | intact | plasticity off | cold | rest | hungry | flee | fed | poison share (intact / plasticity off) |
|---|---|---|---|---|---|---|---|---|
| one type per patch (the 1.8 world) | 993 | 101 | 102 | 39 | 42 | 85 | 61 | 0.034 / 0.042 |
| patches 80% own type | 645 | 105 | 99 | 38 | 43 | 94 | 61 | 0.091 / 0.099 |
| patches 60% own type | 428 | 106 | 107 | 41 | 47 | 106 | 71 | 0.151 / 0.165 |
| patches, types fully mixed | 276 | 97 | 101 | | | 83 | 78 | 0.219 / 0.247 |
| no patches (the drives world), 8 worlds | 263 | 116 | 108 | | | 96 | 93 | 0.233 / 0.237 |
| no patches, twice the food (density 0.3) | 408 | 109 | | | | | | 0.210 / 0.228 |
| patches 60%, density 0.3 | 622 | 104 | | | | | | 0.133 / 0.151 |

The drives population (1.7) in the same worlds:

| world | intact | plasticity off | poison share (intact / plasticity off) | berries (intact / plasticity off) |
|---|---|---|---|---|
| patches 60% own type | 234 | 101 | 0.205 / 0.244 | 19.5 / 24.6 |
| no patches, density 0.3 | 544 | 77 | 0.191 / 0.247 | 47.4 / 46.2 |

Room for learning (`lessons.py`, the affect population with learning off, two worlds): a perfect learner would save
7% of intake in the 80% patches, 11% in the 60% patches, 16% without patches (the association world was designed
at 23%). Per life about 14 bush visits, 5 to 7 of them to poison at 1.1 bites each; an animal is hurt by 1.6 of
the two poison types and comes back to a type that already hurt it 3.5 to 5.5 times.

Reading (one seed, few worlds: a direction, not a result):

- Mixing poison back into the patches does not bring learning back for the affect population, in any world tried,
  also not with twice the food (lives 40% longer). With learning on it eats fewer berries and hardly less poison.
- The drives population still gains from learning where food is plentiful and evenly mixed (77% without it, poison
  share 0.19 against 0.25, the same number of berries). So the affect lineage lost the good use of learning in its
  40 generations in a world where learning was neutral; the rule and the learning rates (0.45) are still there.
- In patches the room is small anyway (7 to 11%): a mistake costs one bite, because the pain reflex ends the visit.
- Life length is not the main limit: the first bite of each poison type is the only one a longer life cannot save
  (1.6 of 6 to 8 poison bites).
- Worlds differ a lot (lifetime 320 to 900 over eight worlds of one setting): compare within the same worlds only.
  A recorded and an unrecorded simulation of the same life give the same numbers (checked).

**Does the lineage pick learning up again? (trial stages `x.w60`, `x.weven`: the affect brain and population, nothing
new, 40 generations, seed 0.)** Before and after in the same eight worlds (`stages.lesion`, seed 123):

| world | | intact | plasticity off | berries (intact / off) | poison share (intact / off) | learning rate |
|---|---|---|---|---|---|---|
| no patches, density 0.3 (`x.weven`) | before | 443 | 119% | 31.3 / 41.2 | 0.212 / 0.226 | 0.45 |
| | after | 690 | 87% | 56.2 / 56.6 | 0.189 / 0.225 | 0.14 |
| patches 60% own type (`x.w60`) | before | 417 | 106% | 29.1 / 33.7 | 0.147 / 0.166 | 0.45 |
| | after | 494 | 102% | 39.4 / 42.6 | 0.150 / 0.166 | 0.24 |

Other lesions after, % of intact: `x.weven` cold 96, flee 97, fed 84; `x.w60` cold 99, flee 90, fed 65.

Reading: where the world pays for learning the lineage has it back within 40 generations (plasticity off from 119%
to 87%, and learning no longer costs berries), and the way back was a lower learning rate: 0.45 to 0.14. The rate
had drifted up in the affect stage (0.33 after drives), where nothing held it; at 0.45 a lesson spread to good
food. Restoring the learned synapses that had gone missing (87% present) changes nothing. In the 60% patches
learning stays neutral: the rate came down too, the room is small (11%). One seed.

**In between (`x.w30`: a third poison in a good patch, density 0.19; 30 generations, lesions in four worlds).**
Before: intact 312, plasticity off 104%, flee 97, fed 88. After: intact 334, plasticity off 108%, flee 90, fed 93;
learning rate 0.45 -> 0.15; poison share 0.216 with learning, 0.211 without. Worse than both earlier candidates:
learning is not picked up and the lasting states matter little. Lives are short (about 570 of 4000 ticks, against
about 1020 in `x.weven`, which had density 0.3). 15 minutes for the whole check.

**The same with more food (`x.w30f`: density 0.3).** Before: intact 462, plasticity off 106%, flee 74, fed 86.
After: intact 498, plasticity off 98%, flee 90, fed 81; learning rate 0.19; poison share 0.209 with learning,
0.212 without. More food and longer lives (about 750 ticks) do not bring learning back in patches either.

Summary of the four candidates after evolution (seed 0; % of intact): `x.weven` plasticity off 87, fed 84, flee
97; `x.w60` 102, 65, 90; `x.w30` 108, 93, 90; `x.w30f` 98, 81, 90. Only the world without patches gives learning a
job; every world with patches gives the lasting states one and leaves learning neutral, at any mix tried. In
patches the lineage turns the learning rate down until a lesson no longer changes what is eaten.

## v35 (2026-10-12, evening): chapter 2, first stage defined: camera eyes and the tectum; generation 0

World of chapter 2 (user: no patches): `W20` = the drives world at density 0.3, `world_ch2` (a novel type's bush,
berry and empty bush share one colour, `berry_world(shared_colour=True)`). Stages `2.0` (the affect brain with
camera eyes, `B20`: the control) and `2.1` (`with_tectum`: a 9-cell map with one winner, `orient_l`, `orient_r`
and `approach` programmes, a `touch` gate, aversion shuts all three). The first design of chapter 2 keeps its
definitions under the keys `old2.1`-`old2.5`.

Generation 0 (`scripts/probes/tectum.py`): the population of trial `x.weven` (the affect population after 40
generations in this world), four worlds, well-fed lifetime:

| variant | well-fed lifetime | berries | poison share | per world |
|---|---|---|---|---|
| old eyes, old colours (as it evolved) | 961 | 78.0 | 0.147 | 572 1399 497 1375 |
| old eyes, shared colour | 532 | 45.4 | 0.204 | 502 290 697 637 |
| camera eyes, shared colour (control of 2.1) | 572 | 43.6 | 0.170 | 566 458 671 593 |
| tectum as defined | 410 | 31.6 | 0.195 | 338 216 558 527 |
| tectum, orienting only | 562 | 43.8 | 0.171 | 502 420 748 576 |
| tectum, approach only | 405 | 31.0 | 0.193 | 325 215 590 491 |
| tectum, programmes at half strength | 539 | 42.2 | 0.176 | 554 321 708 572 |
| tectum, programmes at double strength | 156 | 11.8 | 0.253 | 103 74 248 200 |

Reading: the camera eyes cost nothing. The shared colour costs a great deal in two of four worlds (the looks are
drawn differently, so the worlds are not the same worlds; to be repeated with more). Orienting is neutral; the
approach programme hurts, more the stronger it is.

How the `x.weven` animals spend a life (`forage.py`, `freeze.py`, recorded generation): median lifetime 393 of
4000 while 92% of the bushes are full by mid-life; facing a full bush on 12% of ticks, and then USE 0.48, FORWARD
0.37, turn 0.15; a third of all ticks is spent in stands of 50+ ticks grasping at nothing, which begin with a
stomach at 89% and end at 71% (the fed animal staying where it ate, `settle` and `rest` at 0.63).

## v36 (2026-10-12, night): the tectum is left out; camera eyes with interpolated columns (stage 2.1, seed 0)

Decisions (user): no tectum stage (its innate half answers to moving and looming things, which this world lacks;
value choosing a place is a gate that comes with the basal ganglia; the value cells already have their own synapse
from every eye column, no copies of a circuit per column). The shared colour is dropped: in eight worlds the
adapted population lost 42% with it (1144 -> 661, poison share 0.13 -> 0.18); a full bush and its own empty bush
go from a cosine of 0.22 to 0.55. The new eye columns start as the mean of their neighbours
(`stages.interpolate_eyes`, `Stage.adapt_init`), not silent.

Generation 0 (`scripts/probes/eyes.py`, population of `x.weven`, eight worlds): old eyes 1144; new columns silent
1199 +-168 (poison share 0.10); interpolated 75 (twice the input from the eyes); interpolated with every eye synapse
times 5/9: 1027 +-91, poison share 0.06.

Evolution, 40 generations from the frozen affect population in the world of chapter 2, seed 0, lesions in four
worlds (`runs/logs/x_world_lesion.py`):

| start of the new columns | intact | plasticity off | flee off | fed off | poison share (intact / plasticity off) | learning rate |
|---|---|---|---|---|---|---|
| interpolated, times 5/9 (`2.1`) | 791 | 51% | 93% | 92% | 0.094 / 0.218 | 0.26 |
| silent (trial `x.eyes0`) | 492 | 68% | 90% | 103% | 0.138 / 0.225 | 0.30 |

Mean size of the look weights onto the value cells per column (-60 ... +60) after the run: interpolated,
appetitive 0.11 0.12 0.12 0.14 0.13 0.19 0.16 0.16 0.14; silent, appetitive 0.14 0.13 0.23 0.09 0.21 0.04 0.22 0.05
0.25 (the four new columns still weak after 40 generations).

Reading (one seed): the head start holds after evolution, and learning is used more than at any point in chapter
1 (plasticity off 51%; 43-48% in the association stage, with poison at half of all bushes there as here). Nine
eyes feeding the pain rule halve the poison share. To watch: the fitness within both runs falls over the
generations while the berries eaten rise (interpolated: 858 in generations 0-5, 654 in 20-30, 767 in 30-40), as
in a population that forages well enough to run into the supply.

Housekeeping: `runs/` from 17.5 GB to 1.3 GB (user): superseded runs, archives, probe runs, every side-path run
and all `dashboard.html` files are deleted. Kept: the 36 runs behind the frozen stages and the book (population,
recording, curve, configuration) and the chapter 2 runs. Side-path stages are still defined; their populations are
gone.

## v37 (2026-10-12, late): the world at twice the area; stage 2.1 (camera eyes) frozen with three seeds

**Crowding** (`scripts/probes/crowding.py`, the 128-size run of 2.1, four worlds, well-fed lifetime; same 256
animals and bush density): start population 487 / 676 / 765 and evolved population 788 / 1247 / 1722 at 128 / 181 /
256 cells per side; 14 / 25 / 37% of the evolved animals reach the life cap; 25 / 31-37 / 40-65 s per simulation.
The better population gains more from space, so crowding compresses a stage's gain. The world of chapter 2 is
181 x 181 (user). The falling fitness within a run was not a real decline (every generation has a new world;
in the same worlds evolved beats start).

It is not a shortage of food. Budget: good bushes on 14.6% of cells, 10.8 net food units per bush per 960 ticks;
an animal burns about 0.044 units per tick (0.025 x 1.36 for the cold x 1.30 for moving). Supported at perfect
harvest: 610 / 1220 / 2440 animals at the three sizes; 256 animals use 42 / 21 / 10% of the flow. At birth the
smallest world holds 720 full good bushes (one per 23 cells). What does scale with animals per cell is open
(animals as obstacles and sights, local depletion, empty bushes left by others).

**How the animals die** (recorded generation of seed 0 in the 181 world, 64 animals). An ideal animal would reach
a full bush in 5-10 ticks (nearest full bush about 2 cells away, nearest full good one about 3), last 167 ticks
sprinting on the stomach it is born with, and never die. The real ones find their first good berry at tick 11
(median) and then die of poison: the 26 that die before tick 400 ate 6 good berries and took 4 poison bites
(quartiles 3-8; 8.8 of the 10 food units they are born with), losing 0.069 units per tick alive to poison, more than
to moving and the cold. 11 of 64 never eat anything good. The 10 that reach the cap took 30 poison bites (a perfect
learner: 2). A good bite adds 0.6 units on average instead of 1.8 (bites on a full stomach).

**Stage 2.1, three seeds** (40 generations from the frozen affect population, lesions in four worlds):

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| intact | 1085 | 1109 | 1154 |
| plasticity off | 53% | 44% | 41% |
| flee off | 83% | 96% | 99% |
| fed off | 93% | 100% | 100% |
| poison share, intact / plasticity off | 0.10 / 0.18 | 0.07 / 0.18 | 0.08 / 0.20 |

The starting population scores 676 in the same worlds (seed 0). Frozen as `frozen/2.1` (user, 2026-10-12), an
adaptation step without control; the comparison is the silent-start trial at 128 (v36). The first three-seed run
was stopped by the low-memory reaper when all three wrote their results at once (about 2 GB each at the end of a
run, 0.4 GB before); a run now writes its 150 MB dashboard only with `--dashboard`.

## v38 (2026-10-12, night): what the crowding is; no lottery of inborn taste; the book page of 2.1

**The crowding is good bushes running out near the animals** (`scripts/probes/early.py`, the frozen 2.1 seed 0, the
first 400 ticks, two worlds; user's idea). A good bush is emptied in six bites, a poison bush loses one berry and
stays full. At tick 400, at 128 / 181 / 256 cells per side: share of the bushes carrying berries that is poison
0.75 / 0.66 / 0.59 (0.51 at tick 0); full good bushes within 5 cells of an animal 3.2 / 5.1 / 6.6; alive 49 /
65 / 77%; good bites per animal 21 / 30 / 40; poison bites 5.8 / 6.1 / 6.1. So the sizes differ in good bites,
not in poison bites. The food budget of v37 assumed perfect harvesting; in the first 400 ticks at 128 the animals
eat about half of all good berries there are, twice what they need (bites on a full stomach).

**No lottery of inborn taste** (`scripts/probes/luck.py`, 64 genomes, eight worlds; user's question: does a
generation select the genomes whose inherited look weights happen to dislike that world's poison?). Correlation of
that inborn dislike with fitness within a world: -0.03 with learning, 0.00 without. A genome's fitness in one
world and in another: +0.16 (+0.09 without learning). The best quarter of a world scores +109% above the mean
there and +12% in another world (+101%, +9%). Selection is noisy from the luck of a life, not from a taste that
fits one world. The measure of inborn taste is crude (look weights averaged over the eyes).

**Book**: page `book/camera-eyes.qmd` (first page of chapter 2), data `book/data/2.1.json` (lesions in four worlds)
and `crowding_2.1.json` (the frozen population at three sizes: start 487 / 676 / 765, evolved 567 / 1015 / 1569; the
population evolved at 128 had scored 788 there). `life.book export` now silences only the regions that the stage or
its parent added (`--lesions recent`, the default; `all` for the full sweep): 6 tasks instead of 18, 13 minutes.

Ideas noted, not run (user): born with a fuller stomach against early deaths (measure with `luck.py`); more bushes
or fewer animals instead of more area; the temperature field is convolved over the whole map every tick though
hot springs never move (likely why world size costs time); compacting the arrays to the living animals.

## v39 (2026-10-12, midnight): the simulation no longer costs agents x cells; a fuller stomach changes little

**Speed.** World size cost time because of the loop in which the animals USE the cell they face one after the
other (`world.step_world`): it carried the three maps (object, uses, timer), and every animal's turn copied them,
so a tick cost agents x cells. Profile of the world step alone, 256 animals: 3.1 ms per tick at 128 x 128 and
11.7 ms at 256 x 256 (64 animals: 0.7 and 3.7). The loop now carries the faced cells only (one slot per cell, the
slot of the first animal facing it) and writes them back once: 1.15 and 1.19 ms. Whole simulations
(`crowding.py`, seconds per world of 4000 ticks, before -> after): 128: 18 -> 13; 181: 24 -> 15-18; 256: 37-38 ->
14-16. Every number is unchanged (487 / 567, 676 / 1015, 765 / 1569), 49 tests pass. The first guess, the
temperature field convolved over the whole map every tick, was wrong: computing it only around each animal
(`world.temperature_at`, identical values) changed no timing. It stays (it follows heat sources that change).

**A fuller stomach at birth** (`luck.py 2.1 start_food=...`, the frozen seed 0, eight worlds, learning on):

| born | mean fitness | dead before tick 400 | at the cap | fitness in one world and another | best quarter in another world |
|---|---|---|---|---|---|
| half full (as frozen) | 1027 | about 35% | | +0.16 | +12% |
| three quarters | 1108 | 34% | 21% | +0.17 | +17% |
| full | 1126 | 30% | 21% | +0.15 | +14% |

Little changes: the early deaths are postponed, not prevented (these animals lose food on balance, whatever they
start with), and what selection keeps from one world to the next is the same within the noise.

**Compacting to the living animals** (estimate from the survival curve of the recorded generation: alive 77% at
tick 200, 59% at 400, 42% at 800, 25% at 1600, 16% at the end): brains updated per tick as a share of the
population, with a check every 100 ticks: never 1.00; halving at 1/2, 1/4, 1/8: 0.43; also at 3/4 and 3/8: 0.39;
exact: 0.33. Not built.

**Book**: the page of 2.1 is now "A wider world and sharper eyes" (`book/wider-world.qmd`): preparing the ground
for the chapter; the tectum is not mentioned (user).

To try (user): is there a patch purity at which both the lasting states and learning pay? (v34 tried 1.0, 0.8,
0.6, 0.3 and 0 with the affect population, before the learning rate had come down and before the larger world.)

## v40 (2026-10-13): the dead are dropped from the arrays; three world sizes; patch purity with the camera-eyes population

**Speed** (user: build it first, the runs add up). An unrecorded simulation checks every 100 ticks how many animals
live and, at a half, a quarter and an eighth of the population, goes on with arrays of that size
(`run.make_simulate`, `COMPACT`; `world.take_agents`). A row keeps the index of its animal (`ids`), so every animal
keeps its random draws; the living keep their order (the USE loop goes by index); the dead are seen and heard by
nobody; the statistics of a dropped animal are written to the full arrays at the cut. The frozen 2.1 seed 0 at
181 x 181, one process alone: 18.7 -> 8.4 s per world of 4000 ticks, compiling included in about 10 s. Every
statistic is identical, also in `crowding.py` (676 / 1015, 765 / 1569). Test `test_dropping_the_dead_changes_nothing`
(50 tests). Two small changes of meaning: the food of a dead animal is reported as 0 (it was slightly negative
when the animal died on the last tick simulated), and pain still scheduled for a dead animal
(`sickness_delay`, not used on the chain) is no longer counted. Recorded simulations are unchanged.

**World size** (`crowding.py 2.1 181 256 362`, four worlds, 256 animals, the same density of bushes):

| cells per side | start population | evolved (frozen 2.1, seed 0) | gain | evolved at the life cap | berries | poison share |
|---|---|---|---|---|---|---|
| 181 (as frozen) | 676 | 1015 | +339 | 19% | 170 | 0.098 |
| 256 | 765 | 1569 | +804 | 33% | 284 | 0.081 |
| 362 | 815 | 1924 | +1109 | 45% | 385 | 0.066 |

The start population gains little from space (+13%, +21%), the evolved one a great deal (+55%, +90%): at 181 two
thirds of what evolution added is still hidden by crowding, at 256 about a quarter (taking 362 as the measure,
where it has not levelled off either). The life cap (user): a large share at the cap is no problem as long as the
well-fed lifetime still tells animals apart, and it does (it counts how full the stomach is, not only ticks
alive). Time per world is the same at every size (10-20 s with three processes running).

**Patch purity, generation 0** (`world2.py` with `STAGE=2.1`: the frozen 2.1 seed 0 in its world at 256 x 256 with
320 patches of radius 6, the share of the map that the affect world had; four worlds; % of intact):

| world | intact | plasticity off | fed off | flee off | rest off | poison share (intact / plasticity off) |
|---|---|---|---|---|---|---|
| no patches | 1046 | 48 +-9 | 104 | 84 | 100 | 0.098 / 0.206 |
| patches 30% own type | 561 | 90 +-19 | 94 | 99 | 98 | 0.113 / 0.184 |
| patches 60% | 602 | 86 +-15 | 95 | 95 | 96 | 0.089 / 0.149 |
| patches 80% | 672 | 89 +-17 | 95 | 95 | 90 | 0.062 / 0.107 |
| patches 100% (the affect world's kind) | 831 | 96 +-13 | 90 | 77 | 88 | 0.033 / 0.056 |

Reading (one seed, four worlds): unlike the affect population in v34 (plasticity off 104-106% in every patched
world), this population gains from learning in mixed patches too (10-14%, within the noise of four worlds), and
the lasting states matter a little (5-10%) though it spent 40 generations in a world that had no use for them.
Neither is large at generation 0, and every patched world is much poorer than the open one (561-831 against
1046). Whether the lineage takes both up is a question for evolution: trials `x.c256`, `x.c60`, `x.c80` (the frozen
2.1 population, 40 generations, seed 0, 256 x 256 without patches and with patches 60% and 80% their own type).

**The trials** (40 generations from the frozen 2.1 seed 0, 256 x 256, seed 0; 14 minutes each with four processes
running). Lesions after the run in four worlds of the trial's own world (`runs/logs/x_world_lesion.py`, % of
intact), and the start and evolved populations in 16 other shared worlds (`scripts/probes/before_after.py`):

| trial | intact, four worlds | plasticity off | flee off | fed off | poison share (intact / plasticity off) | after - before, 16 worlds |
|---|---|---|---|---|---|---|
| `x.c256` no patches | 1645 | 55 | 89 | 98 | 0.069 / 0.158 | +19 +-40 (better in 9 of 16) |
| `x.c60` patches 60% | 1514 | 64 | 75 | 94 | 0.066 / 0.122 | +558 +-71 (16 of 16) |
| `x.c70` patches 70% | 1416 | **69** | **78** | **82** | 0.054 / 0.099 | +402 +-57 (16 of 16) |
| `x.c80` patches 80% | 1567 | 100 | 64 | 76 | 0.065 / 0.071 | +641 +-69 (16 of 16) |

The start population in the same four worlds: no patches intact 1604, plasticity off 61, flee 86, fed 97; 60%:
881, 91, 84, 92; 80%: 897, 86, 81, 95.

Reading (one seed, lesions in four worlds: a direction):

- **There is a crossover between 60% and 80%.** At 60% the lineage uses learning and the alarm and hardly the
  stay-after-a-meal state; at 80% it uses both lasting states and has dropped learning altogether (the poison
  share is the same with and without it); at 70% all three are used: plasticity off 69, flee off 78, fed off 82.
- The alarm pays in every world at this size (64-89%); it was 83, 96, 99 at 181 x 181 without patches.
- After 40 generations the patched worlds are as rich as the open one (1362-1492 against 1378 in the 16 worlds):
  the poverty at generation 0 was a population that did not know patches.
- The open world holds its level in 40 more generations (+19 +-40).

**A mistake in the first round of these trials, kept as a lesson.** They were started with plain `stages <key>`,
which mutated every weight in every child (`weight_mutation_prob` 1.0; the frozen stages ran at 0.1 through
`replicate` and `chain`). In the open world the population then lost a quarter in 40 generations (-325 +-100 in 16
worlds, berries 247 -> 139, plasticity off 61 -> 87%), the 60% trial gained nothing (+44 +-56) and no trial used
the fed state (95-101%): mutational load, which hides what a world selects for. A warm-started stage run from the
command line now defaults to 0.1. Two more things it showed: four worlds are too few to compare two populations
(they gave +319 for the 60% trial that 16 worlds put at +44), and the vmapped simulation that evolution uses gives
the same numbers as the plain one (checked in `before_after.py`), so the compaction was not the cause.

**A fourth size** (`crowding.py 2.1 512`): start 876, evolved 2149, 51% at the life cap, 458 berries, poison share
0.057. The evolved population over 181 / 256 / 362 / 512: 1015 / 1569 / 1924 / 2149 (+55%, +23%, +12% per
doubling of the area): levelling off, with half the animals at the cap at 512.

## v41 (2026-10-13, afternoon): patches 70% pure in 16 worlds; lasting states that add up, generation 0

**Seed 0 of `x.c70` with lesions in 16 worlds** (`scripts/probes/trial_lesion.py`, `WORLDS=16`): intact 1162;
plasticity off 62% of intact, rest off 78, fed off 80, flee off 89. The four-world picture holds. Seeds 1 and 2
(`x_ch2_256_p70_seed1`, `_seed2`, each from its own frozen 2.1 population) have evolved; their lesions are not
done (stopped for low memory, see below).

**Lasting states that add up** (user's idea; not tried before). `ModulatorSpec.accumulate`: the modulator is
decay x the previous value + the new value, a leaky sum, instead of the larger of the two (released at once); the
cell that reads it gets a threshold counted in bites. Off by default, the frozen stages are unchanged (50 tests; `freeze.with_defaults` now also fills in a new option of
a modulator).
Generation 0 (`scripts/probes/sum_affect.py x.c70 ...`, the evolved population of `x.c70`, eight worlds, paired
difference to the population as evolved, 1633; one bite is 0.96 on `us_pain` and 1.00 on `fed`; evolved fade
times alarm 6 ticks, serotonin 33):

| variant | difference |
|---|---|
| alarm adds up, no threshold | -5 +-30 |
| alarm above 1.5 poison bites | -147 +-76 |
| the same, fading over about 33 ticks | -29 +-45 |
| stay adds up, no threshold | +62 +-33 |
| stay above 3 good bites | -4 +-26 |
| stay above 9 good bites (a bush has 6) | -147 +-31 |
| the same, fading over about 100 ticks | +1 +-20 |

Reading: nothing is clearly better at generation 0 in a population tuned to release at once. A threshold alone
hurts because the evolved fade times are short (the bites fade before they add up); with a longer fade it is
neutral. The question is what evolution does with it: trials `x.s60`, `x.s70`, `x.s80` (brain `B20S`: both states
as sums, thresholds 1.5 and 9 bites, fades 33 and 100 ticks, thresholds and fades evolvable; the frozen 2.1
seed 0, 256 x 256, patches 60, 70, 80% pure). The test: is the fed state used at 60% and learning at 80%, where
release at once left one of them idle (`x.c60`, `x.c80`)?

**All trials with lesions in 16 worlds** (seed 0 unless stated; % of intact; intact in the same 16 worlds per
purity). A first batch of six JAX processes was stopped for low memory with the laptop in use; three at a time
(0.8 GB each) run fine.

| trial | intact | plasticity off | rest off | fed off | flee off |
|---|---|---|---|---|---|
| 60%, released at once (`x.c60`) | 1410 | 57 | 90 | 92 | 80 |
| 60%, adding up (`x.s60`) | 1223 | 85 | 79 | 92 | 91 |
| 70%, at once, seed 0 (`x.c70`) | 1162 | 62 | 78 | 80 | 89 |
| 70%, at once, seed 1 | 1077 | 46 | 98 | 98 | 78 |
| 70%, at once, seed 2 | 1147 | 95 | 69 | 101 | 82 |
| 70%, adding up (`x.s70`) | 1358 | 86 | 66 | 74 | 92 |
| 80%, at once (`x.c80`) | 1410 | 102 | 49 | 77 | 70 |
| 80%, adding up (`x.s80`) | 1295 | 97 | 64 | 79 | 90 |

Evolved fade times (ticks, alarm / serotonin): at once 5-7 / 33-39; adding up 27-30 / 97-124. The thresholds of
the adding-up trials did not move (flee 1.4-1.5 bites, settle 9.0): a bias mutates in steps of 0.1.

Reading:

- **70% is not a world where all three are reliably used.** Seed 0 uses learning and the stay state; seed 1 only
  learning (and the alarm); seed 2 neither learning nor `fed` (its `rest` serves warmth). A lineage settles on one
  way of avoiding poison: know it by its look, or stay where the food was good. In a patch world the two do the
  same job, so evolution keeps one; which one depends on purity, and near the crossover on the seed.
- **Adding up does not widen the range.** The fed state is used at 70% and 80% as with release at once and not at
  60%; learning is used less at every purity (85, 86, 97 against 57, 62, 102); the alarm is used less (90-92
  against 70-89: two poison bites within the fade are rare). Intact fitness is lower at 60% and 80% and higher at
  70%. At 70% it is the one trial with both moderately used (86 and 74), one seed.
- Next candidate (trial `x.cmix`): patches of bushes with the types mixed evenly inside them. Then a meal says
  "more bushes close by" (the stay state) and nothing about which are poison (learning): two jobs instead of one.

**Patches with the types mixed inside them** (`x.cmix`: rich and poor ground instead of good and bad ground; three
seeds, each from its own frozen 2.1 population, 40 generations, lesions in 16 worlds, % of intact):

| seed | intact | plasticity off | rest off | fed off | flee off |
|---|---|---|---|---|---|
| 0 | 1383 | 52 | 99 | 99 | 90 |
| 1 | 1546 | 33 | 98 | 100 | 97 |
| 2 | 1458 | 41 | 94 | 100 | 93 |

Learning does all the work, more than in any other world tried; the stay state is unused in every seed and the
alarm nearly so (the user's prediction: hunger alone is enough there, and a poison bite says nothing about the
neighbours). Out as a candidate.

**Candidate for the chapter (user): patches 70% their own type with the lasting states adding up** (`x.s70`): in
its one seed every circuit is used (86 / 66 / 74 / 92), and the same holds at 60% and 80% but for learning at 80%
(97). Seeds 1 and 2 are needed before 2.1 is frozen again in it. `trial_lesion.py` now prints standard errors
over the worlds (the fitness, and each lesion's ratio to intact). The first run of the two seeds was stopped for
low memory at the end of generation 40, before the populations were written: a run records its last generation
again at the end (about 2 GB for a moment, v37), and two runs reached that point together. Start runs a few
minutes apart.

**The candidate over three seeds** (`x.s70`: patches 70% their own type, lasting states that add up; each seed from
its own frozen 2.1 population, 40 generations, weight mutation 0.1; lesions in 16 worlds, % of intact +- the
standard error of the ratio over the worlds; the two seeds were started some minutes apart and ran through):

| seed | intact | plasticity off | rest off | fed off | flee off | poison share (intact / plasticity off) |
|---|---|---|---|---|---|---|
| 0 | 1358 +-86 | 86 +-4 | 66 +-2 | 74 +-2 | 92 +-1 | 0.066 / 0.105 |
| 1 | 1359 +-101 | 78 +-7 | 76 +-3 | 77 +-3 | 86 +-1 | 0.068 / 0.119 |
| 2 | 1134 +-83 | 86 +-3 | 69 +-3 | 86 +-2 | 91 +-2 | 0.084 / 0.118 |

Every circuit is used in every seed, each several standard errors below 100: learning 78-86, the stay state
(`fed`) 74-86, `rest` 66-76, the alarm 86-92. With release at once at the same purity the seeds went three
different ways (plasticity off 62, 46, 95; fed off 80, 98, 101). The user's reading holds: states that need
several bites are used next to learning where patches are mixed.

## v42 (2026-10-13, evening): the world and the affect of chapter 2 chosen; 2.0 and 2.1 frozen

Decisions (user): the world of chapter 2 is 256 x 256 with patches 70% their own type, and the lasting states
add up over bites (thresholds as set: `flee` above 1.5 poison bites, `settle` above 9 good bites). One stage in
the book, framed as tuning that prepares for the chapter; the eyes are a small part of it, not a new module.

**Straight from the affect population it does not work** (trial `x.direct`: camera eyes, the patches and the
states that add up at once; three seeds from the frozen 1.8 seed 0, 40 generations, 16 worlds, % of intact):

| seed | intact | plasticity off | rest off | fed off | flee off | learning rate |
|---|---|---|---|---|---|---|
| 0 | 1239 +-65 | 90 +-3 | 64 +-2 | 66 +-2 | 86 +-2 | 0.08 |
| 1 | 1220 +-69 | 84 +-3 | 64 +-3 | 79 +-2 | 82 +-1 | 0.15 |
| 2 | 1208 +-57 | 102 +-2 | 45 +-2 | 66 +-1 | 83 +-2 | 0.00 |

Seed 2 turned its learning rate to 0 and still had it at 0 after 40 more generations (intact 1506, plasticity off
100 +-1, fed off 60, rest off 33; `stages 2.1 --continue`, a new flag: start from a run of the stage itself without
applying the stage's change to the population again). The affect population arrives with a learning rate that
drifted up while learning was idle (0.36; the open-world populations have 0.14-0.25): at that rate a lesson
spreads to good food, and in patches switching learning off is the shorter way. In open country learning is the
only way to tell poison, so there the rate comes down and learning is used; the patches then keep it.

**So the lineage has two steps** (user: option 1), and the trial runs that showed it are the frozen runs:

- **2.0** = the first 2.1 (camera eyes in open country at 181 x 181, frozen 2026-10-12), renamed; the same
  populations (git tag `stage-2.1`).
- **2.1** = the trial `x.s70`: brain `B20S`, world `W21`, each seed from the same seed of 2.0, weight mutation
  0.1, 40 generations. Run directories renamed to `s2_1_tuning*`; the direct-route runs are `x_2_1_direct*`.

Stage 2.1 over three seeds (book export, 16 worlds): re-evaluated 1358 1359 1134; plasticity off 86 78 86, fed
and settle off 74 77 86, rest off 66 76 69, flee off 92 86 91. The seed-0 population against its starting
population (2.0 seed 0 with the new thresholds) in four worlds, at 128 / 181 / 256 / 362 cells per side with the
same share of the map in patches (`crowding.py`, which now scales the number of patches): start 339 / 578 / 765 /
915, evolved 696 / 1093 / 1463 / 1661; at the life cap 11 / 18 / 26 / 32%.

Book: the page is now "Preparing the ground" (`book/wider-world.qmd`): a world with work for every part (table of
the worlds tried, `book/data/worlds_2.1.json`), two steps, moods that build up, a little more to see, results. The
export silences `flee`, `fed`, `settle` and `rest` for this stage (`book.LESION_REGIONS`). 52 tests.

## v43 (2026-10-17): can the affect stage itself be run in the world of chapter 2?

Question (user): replace stage 1.8 by the affect stage in the world of 2.1 (256 x 256, patches 70% their own type,
lasting states that add up), from the drives population, with the eyes of chapter 1, so that 2.0 and 2.1 leave
the chain. Trials `x.affect2` and `x.affect2lr` (the inherited learning rates times 0.4); 40 generations, weight
mutation 0.1, each seed from its own frozen 1.7 run; lesions in 16 worlds (`trial_lesion.py`), % of intact.

| run | intact | plasticity off | fed off | flee off | rest off | learning rate (median of the largest per genome) |
|---|---|---|---|---|---|---|
| `x.affect2`, about generation 0 (one generation run) | 641 | 111 +-6 | 92 | 97 | 70 | 0.41 (as inherited) |
| `x.affect2` seed 0 | 892 | 106 +-6 | 93 | 90 | 54 | 0.16 |
| `x.affect2` seed 0, 40 more generations (`--continue`) | 1313 | 103 +-1 | 83 | 77 | 41 | 0.035 |
| `x.affect2` seed 1 | 865 | 100 +-2 | 76 | 90 | 44 | 0.00 |
| `x.affect2` seed 2 | 1210 | 101 +-1 | 87 | 83 | 39 | 0.00 |
| `x.affect2lr`, about generation 0 | 730 | 97 +-4 | 93 | 93 | | 0.16 |
| `x.affect2lr` seed 0 | 1184 | **84 +-4** | 70 | 88 | 68 | 0.13 |
| `x.affect2lr` seed 1 | 1084 | 98 +-3 | 96 | 88 | 47 | 0.00 |

Reading:

- The lasting states are used in every run; learning is not, except in `x.affect2lr` seed 0, which looks like 2.1
  (plasticity off 78-86 there). `cold` is unused (105).
- At generation 0 learning at the inherited rate (0.41, tuned in the sparse drives world) avoids poison (share
  0.105 against 0.144) but passes up a third of the good food (56 berries against 80): a lesson spreads to
  look-alikes. Times 0.4 it costs far less good food (71 against 80) and is about neutral (97).
- Room for learning (`lessons.py`, the 80-generation population, learning off): 17 poison bites per life, 15 of
  them after the first of their type, 12 visits to a type that already hurt; a perfect learner saves 30 food
  units, 9% of intake, at a lifetime of 2150 of 4000 ticks. The room grows with lifetime (about double at the cap).
- 40 generations is not converged: seed 0 gained another 47% in the same 16 worlds between generation 40 and 80.
  The training curve (new worlds every generation) does not show this; compare populations in shared worlds.
- In patches a lineage can drop learning and rely on the lasting states; a rate at 0 is a flat spot that
  selection does not leave (also `x.direct` seed 2 after 80 generations). The 2.1 lineage reached the patches
  with learning in heavy use and a low rate after 40 generations in open country (2.0).

## v44 (2026-10-17, evening): learning rates that cannot get back; how long a stage needs; what limits learning

**Convergence** (`x.affect2` seed 0, inherited mutation, continued in steps of 40 generations; the same 16 worlds,
% of intact):

| generation | intact | plasticity off | fed off | flee off | rest off | learning rate |
|---|---|---|---|---|---|---|
| 0 | 641 | 111 | 92 | 97 | 70 | 0.41 |
| 40 | 892 | 106 | 93 | 90 | 54 | 0.16 |
| 80 | 1313 | 103 | 83 | 77 | 41 | 0.035 |
| 120 | 1354 | 104 | 79 | 77 | 40 | 0 |
| 160 | 1527 | 102 | 96 | 77 | 42 | 0 |
| 200 | 1536 | 102 | 96 | 75 | 36 | 0 |

Levelled off at about 160 generations, four times the usual stage. The training curve (new worlds every
generation) does not show this; only populations compared in shared worlds do. Learning did not come back with
longer lives (about 1900 ticks); the meal state faded after generation 120.

**Why learning rates go to 0 and stay there.** A rate mutates by `eta_max * N(0, 0.3)` clipped to [0, eta_max];
`eta_max` is 1.0 from the association stage on, so the step has sd 0.3 against rates of 0.1-0.4. From 0.13 a
mutation lands at exactly 0 with a chance of 33%, from 0.05 with 43%; from 0 half the mutations stay at 0 and a
quarter of the upward ones land below 0.1. Where learning is near neutral (in patches the lasting states cover
poison too), the clip piles the population up at 0, and a climb back meets a reset on nearly every second step.
In open country a child at 0 dies out, which is why the 2.0 -> 2.1 route kept learning. New:
`EvolutionConfig.eta_log_std` (off by default): a rate is multiplied by `exp(eta_log_std * N)` from at least
`eta_floor` (0.01), as fade times and strengths already mutate; `Stage.evolution` sets it per stage.

**Multiplicative rates** (`x.affect2log`, `eta_log_std` 0.3, otherwise `x.affect2`):

| seed | generation | intact | plasticity off | fed off | flee off | rest off | poison share (on / off) | rate |
|---|---|---|---|---|---|---|---|---|
| 0 | 40 | 1100 | 89 +-7 | 84 | 90 | 58 | 0.092 / 0.121 | 0.155 |
| 0 | 80 | 1336 | 100 +-3 | 81 | 83 | 54 | 0.093 / 0.102 | 0.195 |
| 1 | 40 | 1061 | 92 +-7 | 82 | 90 | 61 | 0.093 / 0.120 | 0.248 |
| 1 | 80 | 1256 | 91 +-5 | 90 | 94 | 63 | 0.077 / 0.109 | 0.188 |
| 2 | 40 | 1387 | 99 +-1 | 87 | 77 | 40 | 0.099 / 0.100 | 0.008 |
| 2 | 80 | 1426 | 101 +-1 | 97 | 83 | 35 | 0.098 / 0.099 | 0.006 |

The rate is kept in two seeds of three, but its value shrinks as the rest of the brain improves: at generation 80
learning trades good food for less poison about evenly.

**What limits learning** (the three populations of generation 80, 16 worlds, learning on / off):

- A slower adaptation of the look (`scripts/probes/adapt_tau.py`, about 200 and 500 ticks instead of 50, so that in
  patches the average is not the local type): learning cuts a little more poison but costs more good food, and
  intact fitness falls (the brains are tuned to 50). Not the bottleneck.
- Novel types less alike (`scripts/probes/novel_sim.py`, `NOVEL_SIM` 0.5 instead of 0.8; two novel types then
  share about 0.25 of their look instead of 0.64): plasticity off 94 +-4 and 79 +-5 instead of 100 and 91 (seeds 0
  and 1), and learning no longer costs good food (berries 112 / 109 and 105 / 94 against 141 / 150 and 137 / 149).
  Spreading to look-alikes is what limits learning in this world. Intact falls (novel bushes look less like the
  bushes these brains approach).

Trial running: `x.affect2sim` (multiplicative rates, novel similarity 0.5, from drives, 80 generations, three
seeds). The idea (user): make the looks easy to tell apart until the pallium, and give the pallium the look-alikes
back as its job.

**`x.affect2sim`** (multiplicative rates, novel similarity 0.5, 80 generations from drives; 16 worlds, % of intact):

| seed | intact | plasticity off | fed off | flee off | rest off | poison share (on / off) | rate |
|---|---|---|---|---|---|---|---|
| 0 | 721 | 67 +-4 | 100 | 96 | 95 | 0.084 / 0.137 | 0.24 |
| 1 | 1120 | 53 +-6 | 98 | 98 | 88 | 0.070 / 0.155 | 0.32 |
| 2 | 968 | 87 +-5 | 92 | 88 | 68 | 0.119 / 0.142 | 0.08 |

Learning is used in every seed, and strongly in two; there the lasting states go idle. The same either-or as with
patch purity: in patches of one type, learning (by look) and the lasting states (by place) do the same job, which
bushes here are poison, so a lineage keeps one. Next idea (discussed with the user): a world in which they answer
different questions, as in the worm (dwelling: how much food is here; aversive learning: which food is bad).

## v45 (2026-10-17, night): stage 1.8 affect redefined and frozen; chapter 1 final

Stage `1.8` = brain `B18S` (the lasting states adding up over bites), world `W18S` (= the 2.1 world: 256 x 256,
density 0.3, patches 70% their own type), chapter 1 eyes, from the frozen 1.7 seed 0 (all three seeds, as in
chapter 1), 100 generations, weight mutation 0.1, learning rates mutating multiplicatively (`eta_log_std` 0.3).
Control: the drives brain in the same world (`sum_affect_init` now leaves a brain without the new cells alone; the
first control run stopped on it). Run directories `s1_8_affect2*`. The first affect stage is `1.8old`; it, 2.0 and
2.1 stay frozen off the chain. Book export (`--evaluate --jobs 3`, 8 worlds, 19 minutes):

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| last 50 generations, main - control | -11 | +221 | +215 | +142 +-76 |
| re-evaluated, main / control | 1222 / 1421 | 1483 / 1150 | 1501 / 1080 | +185 +-194 |
| head to head, main - control | +32 | +478 | +138 | +216 +-135 |
| flee off, % | 86 | 77 | 88 | |
| fed off / settle off, % | 94 / 86 | 89 / 89 | 86 / 83 | |
| rest off, % | 62 | 39 | 51 | |
| plasticity off, % | 98 | 105 | 100 | |
| cold / warm_seek off, % | 102 | 100 | 102 | |
| learning rate, main / control | 0.23 / 0.40 | 0.11 / 0.24 | 0.13 / 0.13 | |

Every new circuit is used in every seed; main is ahead of control in two seeds and level in seed 0 (behind in its
own worlds, slightly ahead in shared ones). Learning is unused but kept (no rate near 0), as agreed: the pallium
stage is to make it pay. Chapter 1 frozen again (tag `chapter-1-v2`); the book page of affect rewritten, "Preparing
the ground" removed.

## v46 (2026-10-18): before the pallium, what is there to learn and does an expansion separate the looks?

On the frozen 1.8 population (seed 0) in its own world:

- **What there is to learn** (`lessons.py`, learning off, two lives): lifetime 1964 of 4000; 23.9 good visits (7.6
  bites each) and 15.4 poison visits (1.3 bites) per life; 17.2 of 19.0 poison bites come after the first of their
  type: a perfect learner would save 34.5 food units, **9% of intake**.
- **Bites on a full stomach** (`supply.py 1.8`): **71%** of bites land on a stomach over 90% full; edible bushes are
  75-80% full from mid-life on (no shortage of food).
- **Separation, brain-free** (`pallium_sep.py`, looks as the world draws them, one eye column; transfer = the
  aversion a lesson on one novel type leaves on another, relative to itself):

  | input to the learned synapses | look itself | best random pallium (N 32-128, fan 2-4, k-WTA) |
  |---|---|---|
  | raw look, similarity 0.8 | 0.64 | 0.35-0.45 |
  | look minus half the mean of the four types | 0.22 | 0.16-0.18 |
  | look minus the mean (ideal habituation) | -0.33 | 0.05 |

  A sparse code is non-negative, so it cannot say "less like the poison than the average": once the look is
  centred, the plain linear read-out separates better than any expansion.
- **Separation on real lives** (`pallium_real.py`, two recorded lives, learning as evolved; overlap of the look
  inputs, all five eye columns, at a poison bite with those at every later bite of the same animal, relative to
  itself):

  | | same poison | other poison | good | good / same |
  |---|---|---|---|---|
  | adapted look (as the synapses see it) | 0.31 | 0.16 | 0.08 | **0.24** |
  | pallium on the adapted look (4 settings) | 0.39-0.43 | 0.21-0.29 | 0.18-0.24 | 0.46-0.58 |
  | look before adaptation (reconstructed) | 0.64 | 0.51 | 0.40 | 0.63 |
  | pallium on the look before adaptation | 0.62-0.67 | 0.43-0.49 | 0.27-0.41 | 0.43-0.63 |

  Habituation already does what the pallium was planned for, and better: a lesson lands on good food at a quarter
  of its strength on the poison itself. A random expansion on top of it doubles the spreading. What limits a
  lesson here is rather that **the same poison looks different from bite to bite** (overlap 0.31): the adapted look
  depends on what the animal has been looking at (in a patch 70% of one type, the local type fades).
- Open: whether the gain of learning at novel similarity 0.5 (v44) comes from less spreading or from a more
  consistent adapted look; the pallium as planned (ADR-025) has no job in this world as it stands.

**Harder looks** (`pallium_sep.py`, same measure; second number: after one lesson on each of two poison types, the
aversion to a good type relative to that to a poison, 0 = perfect, 1 = none; best pallium of N=64, fan 2-4, k 4-16):

| looks | look, raw | pallium, raw | look, adapted 0.5 | pallium, adapted 0.5 |
|---|---|---|---|---|
| random, similarity 0.8 | 0.77 | 0.56 | 0.35 | 0.27 |
| random, similarity 0.9 | 0.90 | 0.70 | 0.65 | 0.42 |
| random, similarity 0.95 | 0.95 | 0.81 | 0.82 | 0.64 |
| XOR (poison ++ and --, good +- and -+), 0.8 | 1.00 | 0.63 | 1.00 | 0.37 |
| XOR, 0.9 | 1.00 | 0.79 | 1.00 | 0.59 |

The more alike the looks, the more the expansion adds over a partly adapted look (real lives at 0.8 sit near
"adapted 0.5": 0.24 measured against 0.22). With XOR looks no linear read-out discriminates at all, adapted or not
(with an ideal habituation the two lessons cancel); the pallium does, and better on the adapted look than on the raw.

**What makes learning worth it** (`learn_worth.py`, napkin model: visit counts of 1.8; aversion = eta x sum of
lessons x transfer; bite with sigmoid(3 (appetite - aversion)); best appetite and eta by grid, as evolution could
set them; gain over the best non-learner in % of intake; good visits worth their food or a half or a quarter of it):

- Without spread from bite to bite, learning pays up to a transfer to good food of about 0.3 (9%, 18%, 37%), less by
  0.5 and nothing by 0.8: one strong lesson per poison type is enough, then the poison is refused and lessons stop.
- With the spread of the 1.8 world (overlap sd about 1 relative to its mean on the same poison) learning is worth
  at most 0.6% even at transfer 0; at spread 0.5, 2-4% at transfer up to 0.2. This is why 1.8 leaves learning idle.
- With XOR looks the plain look makes a lesson on one poison clear the other (T_o -0.5 to -1): nothing left above a
  transfer of 0.1.

**Real lives, all five eye columns** (`pallium_real.py`; 'summed' = all lessons so far read against later bites,
good / poison, once both poison types have taught): 0.40-0.76 in the 1.8 world, 0.77-1.05 with XOR looks at 0.8,
0.58-0.93 at 0.6, pallium or not. The side columns see neighbouring bushes at a bite and carry most of the spread.

**Real lives, the eye ahead only** (`COLS=vis+0`; summed good / poison, lower is better; adaptation strength
recomputed offline, the animals behave as recorded; one random pallium per setting):

| world | look, strength 0 / 0.5 / 1 | best pallium, strength 0 / 0.5 / 1 |
|---|---|---|
| 1.8 (random looks, 0.8) | 0.60 / 0.46 / 0.33 | 0.06 / 0.21 / 0.38 |
| XOR looks, 0.8 | 1.02 / 0.93 / 0.84 | 0.63 / 0.63 / 0.66 |
| XOR looks, 0.6 | 0.90 / 0.78 / 0.65 | 0.49 / 0.41 / 0.43 |

The lesson must be about what is at the mouth: with the eye ahead the same poison overlaps itself 0.9-1.0 (sd 0.05
at strength 0, 0.5-1.2 at strength 1). On that eye the pallium separates well, best on a look adapted at most half:
full adaptation makes the code of one type depend on its surroundings. For the plain look, adaptation helps (it
centres). XOR at 0.8 is too hard even for a random pallium; at 0.6 it brings 0.65-0.90 down to about 0.45, near the
napkin's limit for a transfer that still pays (0.3-0.4 with the other poison untouched).

New: `BrainConfig.in_adapt_strength` (default 1), `berry_world(novel_xor=...)` / `learning_world(novel_xor=...)`.

**Adaptation strength as a gene** (user, 2026-10-18: so that habituation cannot hold back later stages): `Genome.adapt`
(one per genome), on with `BrainConfig.in_adapt_evolve`, starting at `in_adapt_strength`; mutates like a fade time
(chance `tune_prob`, factor exp(`tune_std` N)), within [0.02, 1]; from either parent; carried over by remapping only
between layouts where it evolves. Stored populations get the designed strength (1), so the frozen stages behave as
before. On from the pallium stage on.

**Magnified pallia** (`pallium_real.py MAG=...`: cells per eye column by distance from the centre, centre / next /
outer; fan 3; k-WTA 1/16 or 1/8 per column; summed good / poison, best of the two k; user's question whether a
fovea-like density would serve later stages):

| cells | 1.8 world, strength 0.5 / 1 | XOR 0.6, strength 0.5 / 1 |
|---|---|---|
| 64 / 0 / 0 (eye ahead) | 0.21 / 0.40 | 0.41 / 0.43 |
| 64 / 8 / 2 | 0.50 / 0.47 | 0.53 / 0.53 |
| 64 / 16 / 4 | 0.48 / 0.36 | 0.59 / 0.59 |
| 64 / 32 / 16 | 0.65 / 0.68 | 0.57 / 0.68 |
| 64 / 64 / 64 | 0.66 / 0.72 | 0.67 / 0.73 |

Even a few side cells undo much of what the eye ahead gains: whatever they see at a bite is taught too. Decided
(user): the pallium reads the eye ahead only. Peripheral identity may come back when a stage chooses between visible
targets (the actor), better as a selection of one location (tectum-like) than as a density gradient.

## v47 (2026-10-18): the pallium stage defined (2.2)

Stage `2.2` (`s2_2_pallium`, from 1.8 seed 0): brain `B22P` = the affect brain + region `pallium` (128 cells, k-WTA
4) reading only the eye ahead (`vis+0.app*`, all 8 features, fixed random weights per genome) + learned
`pallium -> valence_av` (the pain teacher, eligibility 0.5, starting rate `PALLIUM_ETA` 0.1); the adaptation
strength evolves (main and control: `Stage.control_brain`). World: the 1.8 world with XOR looks at similarity 0.6
(`pallium_world`).

- **Size and sparseness** (`pallium_sep.py`, XOR at 0.6, discrimination unadapted / half adapted): 64 cells, fan 3,
  k 8: 0.62 / 0.44; 128, all 8 features, k 4: 0.23 / 0.10; 256, fan 8, k 2-4: 0.10-0.17 / 0.03-0.05. Larger and
  sparser is better; 128 / 4 chosen for cost.
- **Assay** (`assay_xor.py`, 24 agents of 1.8, two rounds of meals of each type; aversive activity / P(bite) after,
  poison ++, --, good +-, -+; before 0.03-0.06 / 0.34-0.57):

  | | starting rate 0.3 | 0.1 | 0.03 |
  |---|---|---|---|
  | pallium learns only | 0.99 0.98 0.33 0.47 / 0.00 0.00 0.27 0.12 | 0.53 0.52 0.10 0.14 / 0.00 0.04 0.36 0.29 | 0.14 0.13 0.04 0.05 / 0.31 0.13 0.45 0.37 |
  | look learns only | 0.15 0.12 0.14 0.12 / 0.27 0.18 0.21 0.36 | | |

  The look synapses cannot tell XOR poison from good (they only make the animal shyer of all four); the pallium
  can. At 0.3 the aversive cells saturate for good food too (4 active cells near 1 carry a big step).
- **What there is to learn** (`lessons.py`, learning off): lifetime 1964 in the 1.8 world, with XOR looks 1650
  (similarity 0.8), 1323 (0.6), 1244 (0.5): less alike novel bushes look less like food to the 1.8 animals. A
  perfect learner would save 10% of intake at each.
- **Generation 0** (`gen0.py OFF=pallium->valence_av`, 8 worlds, paired): learning on 1020 +-121, pallium learning
  off 999 +-122, all learning off 1016 +-110 (good berries 65.4 / 65.4 / 70.0 in the first half, poison 7.3 / 7.6 /
  8.6). Neutral, as a diagnostic: the pallium adds a little, the inherited look learning costs good food as much as
  it saves.

## v48 (2026-10-18): the pallium stage at 0.6 is not used; what learning does in its world

Seed 0 of 2.2 (100 generations, XOR looks at 0.6): last 50 generations main 1182 +-30, control 1289 +-35. Lesions (16
shared worlds): pallium 98 +-2, no_plasticity 98 +-2. Learning rates: pallium 0.10 -> 0.18, look 0.23 -> 0.10 (control
0.08); adaptation strength 0.92 (control 0.88). Seeds 1 and 2 stopped (user: too weak to go on; investigate first).
Brain step optimised meanwhile (fixed weights computed at birth, input columns never multiplied): a generation of
2.2 is 1.7x faster, same lives.

**Paired, the evolved seed-0 population** (`gen0.py RUN=<2.2 run> OFF=...`, 8 worlds; fitness relative to all
learning on):

| world | all on | pallium off | look off | all off |
|---|---|---|---|---|
| as defined (XOR 0.6) | 1150 | -22 +-28 | +3 +-9 | -2 +-19 |
| poison twice as costly (`POISON_FOOD` -6) | 719 | +10 +-22 | +21 +-18 | +6 +-19 |
| similarity 0.5 | 1071 | -63 +-22 | -43 +-15 | -69 +-17 |

At 0.6 no learning is used; a costlier poison makes poison rarer (about 4 berries a life) and learning useless. At
0.5 learning is used, but the gain is not fewer poison bites (4.5 / 1.9 per half with learning, 4.7 / 1.8 without):
it is more good food late in life (25.3 against 22.6) and longer lives. Inside a life at 0.6 (`pallium_life.py`):
at bites the pallium's 4 winners fire at 0.62 for poison and good alike; with nothing ahead about 3 cells stay
weakly active (0.23), the k-WTA baseline.

More paired variants (same population and test): XOR 0.5 with patches 40% their own type: all on 818, pallium off
-26 +-16, look off -42 +-25, all off -53 +-26 (more poison: 6.1 in the first half); XOR 0.4: all on 912, pallium off
-31 +-18, look off -16 +-10, all off -36 +-18. Poison bites hardly change with learning in any of them.

**Is the k-WTA needed?** (brain-free, XOR, 128 cells, all 8 features; discrimination unadapted / half adapted at 0.5):
k-WTA 4 0.20 / 0.10; no k-WTA (rectified tanh, 64 cells active) 0.87 / 0.82; a fixed threshold 1.5 0.25 / 0.09 (6-9
cells), 2.0 0.13 / 0.02 (1-3 cells). Sparseness is what separates; a threshold does it as well when it gives as few
cells, but its sparseness follows the strength of the input. k-WTA stays (a stand-in for global feedback
inhibition, the APL neuron of the mushroom body).

**Why lessons do not cut poison bites** (seed-0 population of 2.2 at 0.6, tested at 0.5; `lesson_curve.py`,
`lesson_code.py`, one recorded life):

- P(bite) at a poison bush hardly falls with lessons of its type (0 / 1 / 2 / 3-4 / 5+: 0.40 0.37 0.44 0.47 0.34,
  all learning off 0.37 0.50 0.49 0.39 0.48); pallium learning alone (look off) changes nothing; look learning lowers
  bites on good bushes as much as on poison (5+ lessons: good 0.47 with it, 0.55-0.57 without). The lasting alarm
  confounds counts: even with learning off, a poison not yet learned is bitten less as lessons of the other type
  grow (0.44 -> 0.11).
- The pallium code separates in real lives: of an encounter's active pallium cells, 62% were active at an earlier
  lesson on the same poison, 1% for the other poison, 9% for a good type.
- Its drive onto the aversive cells rises only for the learned poison (0 -> 5+ lessons: 0.37 -> 0.66; good bushes
  0.31 -> 0.29), but the cells sit far below threshold when a bush is ahead: total input -0.92 (poison, no lesson)
  to -0.04 (5+), bias -0.37. The negative part is inherited: `vis*.hit` / `vis*.near` -> aversive weights (sum
  -0.69 in 1.8, -1.10 in 2.2 main, -1.23 in 2.2 control): 'do not be averse to what is in front of you'. Look
  learning gains more drive than the pallium (+0.15 -> +0.73) but on everything.
- Reading: in XOR worlds look learning spreads to all novel bushes; evolution answers in both populations by turning
  aversion down whenever a bush is ahead (and the look rate down, 0.23 -> 0.10). In main the same offset mutes the
  pallium, whose lessons are specific but start 1.3 below threshold. A race between lesson strength and an
  inherited offset, which evolution tips one way or the other: why a small change of the world decides whether
  learning is used.

**The rule genes.** The learned projections' rule genes evolved: pallium -> aversive A -0.85, B 0.86 (design 0, 1),
look -> aversive A -0.17 (control -0.40); in 1.8 the look rule had stayed at A +0.01. The pain itself drives the
aversive cells at a lesson (post near 1), so (A post + B) pre is about 0: lessons nulled. Not a flaw of the genetic
algorithm (it happened in main and control, and not in 1.8): selection for 'learn only while not already averse'
where lessons cost more than they save. With the rule reset to the design (`RULE_RESET=1`, look learning off), the
pallium's drive onto the aversive cells at encounters rises with lessons for the learned poison 0.53 -> 2.09 (5+)
and for good bushes 0.35 -> 0.93: specific, but with a spread that costs good food (good bushes bitten 0.64 -> 0.47).
User (2026-10-18): do not switch evolution off where it turns a feature down; fix the reason it does (start values
and genuine flaws of the genetic algorithm may be changed). So: evolve at 0.5, where learning paid (paired test),
and find what makes lessons spread. The 0.5 main run (stopped at generation 65 for a change that was reverted) was
restarted; seed 1 runs alongside.

**What makes lessons spread** (`lesson_code.py`, the same recorded life, rule reset, look learning off). The brain
learns faithfully: the learned drive (weights now minus at birth) and an ideal learner that adds the pallium code of
every poison bite spread alike (5+ lessons: good / poison 0.60 / 1.79 learned, 2.66 / 7.80 ideal). The spread is in
the codes: per lesson, a lesson reaches good bushes at 0.21 of its strength on its own poison with k-WTA 4. Codes
recomputed from the recorded input (k=4 reproduces the recorded code exactly) with other sizes: k 1 / 2 / 4 / 8:
0.06 / 0.14 / 0.21 / 0.26. Set: k-WTA 2, starting pallium rate 0.2 (half the drive per lesson). Assay at 0.5
(`assay_xor.py`, pallium alone, two rounds): poisons 0.63 / 0.59 aversion, P(bite) 0.00 / 0.04; good 0.07 / 0.11,
P(bite) 0.38 / 0.33 (before 0.46 / 0.42). The two main runs restarted with it (controls have no pallium and stand).

**2.2 with k-WTA 2, rate 0.2, XOR at 0.5, seed 0** (100 generations): last 50 generations main 1023 +-31, control
1106 +-32 (poison share 0.090 both). Evolved: pallium rate 0.2 -> 0.52, its rule A +0.28 B 1.16 (Hebbian kept);
look rate 0.23 -> 0.06 (control: 0.34); adaptation 0.89 (control 0.98). Lesions (16 worlds, % of intact): pallium
**90 +-2** (poison share 0.092 -> 0.099), no_plasticity 90 +-2, flee 93 +-2, rest 58 +-1. The pallium carries the
learning and is used.

**Making learning pay** (user: reduce patch purity, accepting less use of the lasting states). Paired, the seed-0
population evolved with k-WTA 2 (`gen0.py OFF=pallium->valence_av`, 8 worlds): patches mixed evenly (purity 0.25):
all on 1016, pallium learning off -190 +-29, all off -242 +-32 (-24%); good berries in the second half 23.7 / 15.8 /
14.1; poison bites per half hardly change (9.7 / 9.2 / 9.4), lives are longer. Mixed and half the bushes (density
0.15): all on 546, pallium off -138 +-28, all off -156 +-21: harsher, learning not worth more. Set: `W22` = the 1.8
world with purity 0.25. Rerun with three seeds.

**Is vision -> pallium inhibitory?** No: the weights are mixed (46% positive, 54% negative, mean -0.009); the book's
diagram took the sign of the mean and drew it as inhibitory (fixed: both signs common = mixed). Brain-free, XOR 0.5,
128 cells, k-WTA 2, discrimination unadapted / half adapted: mixed 0.13 / 0.05, excitatory only 0.20 / 0.09, ON/OFF
channels excitatory 0.26 / 0.17. Afferents are excitatory in the real pallium; excitatory-only is an option if
Dale's law should hold here too.

**CPU benchmark** (`bench.py`, 2.2, two other processes running): 256 animals 6.1 s per 1000 ticks, 512 17.8,
1024 24.7, 2048 38.9 (19-35 us per animal-tick). WSL2 is not installed yet (needs the user: admin + reboot).

## v49 (2026-10-18): the GPU

WSL2 + CUDA set up (`docs/SETUP.md`). Benchmark `bench.py` (stage 2.2, animals and world area scaled together, s per
1000 ticks): at first GPU 3.1 / 9.0 / 14.4 for 256 / 1024 / 2048 animals against CPU 5.9 / 14.3 / 21.4, the GPU at
about 10% use. The USE step ran one agent after another (`lax.scan` over agents, ADR-003); now in rounds by rank among
the agents on the same cell (same order within a cell, identical results): GPU **0.61 / 1.38 / 2.09**, CPU unchanged
(5.6 / 14.1 / 20.5): about 10x. Memory: siblings no longer get full copies of every N x N gene (`brain.block_genome`;
states built at birth per genome); eligibility and depression only for the plastic block; patch placement as a running
minimum. All identical bit for bit (generation statistics) and tested. 4096 animals still exceed 8 GB: the per-animal
weight matrix (1 GB) and the fixed weights copied to siblings (0.7 GB) remain.

## v50 (2026-10-18, evening): 2.2 in mixed patches, three seeds on the GPU; how noisy the comparisons are

Stage 2.2 with `W22` (patch purity 0.25), k-WTA 2, pallium rate 0.2, XOR at 0.5; three seeds of main and control
on the GPU (`scripts/gpu_queue.sh`, three at a time: six runs in about 17 minutes). Book export (`--evaluate`,
first 8 worlds):

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| pallium off, % of intact | 92 | 91 | 93 | **92 +-0.6** |
| no_plasticity, % | 67 | 77 | 87 | **77 +-6** |
| flee / fed / settle off, % | 98 / 94 / 94 | 101 / 103 / 89 | 92 / 95 / 97 | |
| last 50 generations, main - control | | | | +94 +-53 |
| head to head | +448 | +261 | -106 | +201 +-163 |
| lab, only the pallium learns: P(bite) poison / good (before 0.23-0.37 / 0.12-0.34) | 0.08 / 0.13 | 0.06 / 0.35 | 0.15 / 0.39 | |
| lab, only the look learns: poison / good | 0.12 / 0.10 | 0.17 / 0.32 | 0.17 / 0.15 | |

The pallium is used in every seed; learning as a whole more (the look learns too). The lasting states are used less
in mixed patches (as accepted). Main ahead of control on average, with large standard errors.

**Noise** (`noise.py`, 16 shared worlds, the final populations):

| seed | training curve, last 50: sd per generation (main / control) | re-evaluated main - control, paired | resolvable (2 s.e. paired) | sd between genomes within a world / between worlds |
|---|---|---|---|---|
| 0 | 132 / 158 | +111 +-42 | 9.8% | 615 / 165 |
| 1 | 262 / 231 | +301 +-61 | 11.7% | 648 / 224 |
| 2 | 168 / 187 | +58 +-42 | 10.9% | 569 / 124 |

With 16 worlds, main is ahead in every seed; the export's 8 worlds had seed 0 at -105. Readings: (1) 8 worlds are
too few to compare two populations of one seed; the paired standard error falls as 1/sqrt(worlds), so 64 worlds
resolve about 5%, cheap on the GPU. (2) One life per genome is the largest noise: genomes in one world spread by
about 600 around a mean of about 850, four times the spread between worlds. Selection works on that: more siblings
per genome (or more genomes) would sharpen evolution more than anything else. (3) The seeds differ more from each
other (main - control +58 to +301) than the standard error of one comparison: three seeds stay necessary.

**A remap bug** (found while checking why `noise.py` and the export disagreed about the controls). `remap_genomes`
counted every projection with a designed start value as hard-wired, so a remap reset evolvable projections with
a design (the drives stage's `hungry -> grasp / no_feed / valence_app / ganglion`, designed 0) to the design, and it
drew fixed random projections (the pallium's input) anew. Affected: the warm starts of 1.8 and 2.2 (main and control
alike began with the hunger synapses at 0); the head to head (the control is remapped onto main's brain, main is not:
**biased towards main** from 1.8 on); every probe that loads with remapping (`load_population(run, exp)`: gen0,
lessons, the assays, `lesson_*`, `pallium_real`, `noise.py`, the book's lab test: evolved pallium populations got a
new random pallium). Not affected: lesions and re-evaluation of the export (loaded directly), evolution within a run,
chapter 1 before 1.8. Fixed (only non-evolvable designed weights take the design; everything else carries over;
remapping onto the same layout is now the identity, tested). The 2.2 runs are rerun with the fixed warm start; the
head to head of 1.8 (frozen) is re-evaluated, its runs are not rerun. The 2.2 numbers above (v48-v50) that came from
probes or the head to head are affected.

## v51 (2026-10-18, night): 2.2 after the remap fix; the head to head checked; a larger pallium

The six 2.2 runs rerun with the fixed warm start (`runs/*/20261018-18*`), evaluated on the GPU (export 16 worlds).

| | seed 0 | seed 1 | seed 2 | mean |
|---|---|---|---|---|
| training, last 50: main / control | 920 / 837 | 899 / 754 | 980 / 889 | +106 +-20 |
| re-evaluated apart (16 worlds each): main / control | 913 / 927 | 964 / 813 | 977 / 894 | +73 +-48 |
| head to head (same worlds, half each) | +184 | +288 | +214 | **+229 +-31** |
| lesion pallium (% of intact) | 90 | 88 | 92 | 90 +-1 |
| no plasticity | 77 | 67 | 87 | 77 +-6 |
| settle / flee / fed | 96 / 101 / 100 | 94 / 99 / 101 | 91 / 97 / 99 | |
| lab, only the pallium learns: bites poison / good (before 0.38 / 0.39) | 0.16 / 0.47 | 0.20 / 0.41 | 0.11 / 0.23 | 0.16 / 0.37 |
| lab, only the look learns | 0.44 / 0.45 | 0.14 / 0.13 | 0.07 / 0.11 | 0.22 / 0.23 |

Within a life the poison share of bites falls from 0.14 (first quarter) to 0.08 (last). The pallium is discriminative
in every seed (the look alone shies from both kinds alike, as XOR predicts). `flee` and `fed` are not used in mixed
patches; `settle` a little.

**Noise** (`noise.py`, 16 paired worlds, separate evaluation): main - control +46 +-55, +191 +-36, +52 +-37;
resolvable 13%, 8%, 8% of main; sd between genomes in one world 570-630 against 100-200 between worlds. Same
picture as v50: one life per genome dominates, and 16 worlds are not enough to separate two populations by 5%.

**The head to head is fair now** (`scripts/probes/versus_check.py`): each control population alone, in its own layout
and in main's layout with the regions it lacks silenced (as `versus` runs it), in the same 8 worlds: equal to within
1 point for 2.2 and 1.8. Main's lead there is its own. That the head to head shows more than separate evaluation is
expected: the halves compete for the same bushes, and the better forager also takes food away from the other half.

**1.8 re-evaluated** (frozen, not rerun): head to head +178 +-66 (+82, +304, +147), before the fix +216 +-135; the
lesions as before (rest 60 39 52, hungry 23 10 21, flee 85 77 87, fed 92 89 86, settle 86 89 82). The bias
of the old head to head was small; the frozen verdict stands.

**A larger pallium** (backup check, `pallium_real.py COLS=vis+0 FAN=8 MAG=N/0/0`, the 2.2 populations, two recorded
lives): transfer of the summed lessons to good bites (good / poison) 0.13 at 128 cells with 2 active (the stage);
256 cells: 0.03-0.12, 512: 0.06-0.21, no trend with size or k. The look itself 0.68. Transfer is far below the
napkin threshold of about 0.3 (v46) already; what is left is the spread of the read-out from one poison bite to the
next (sd about 0.85 of its mean). A larger pallium is not worth its cost.

Also: runs take turns for the recording at the end (`life.run.recording_lock`, a lock file per OS); three runs at
a time on the GPU again.

**Repeatability** (`repeatability.py`, seed 0, 8 worlds): sd between genomes in one world 591 (main) / 574 (control),
of the genome means over 8 worlds 246 / 290. Correlation of the genome scores (4 siblings) between worlds 0.05 /
0.14, one life 0.01 / 0.04 (Spearman-Brown); 32 lives would reach 0.29 / 0.57. In a converged population genetic
differences are small, so most of the spread is luck. `siblings` divides the fixed 256 animals into genomes;
`episodes` (parallel worlds, vmapped) adds lives per genome at about 2.3x the time for 4x (GPU benchmark v49).

## v52 (2026-10-19): 2.2 frozen; noise at generation 0

**2.2 frozen** (user, 2026-10-19), three seeds, git tag `stage-2.2`; `flee` and `fed` kept though unused in mixed
patches. Book: "Watching it work" strips for drives (`warm_rest`) and affect (`flight`, with the alarm level).

**Noise at generation 0** (`scripts/probes/noise_gen0.py`, 32 shared worlds; user: v51 measured a converged
population, where little but luck is left). The starting population of 2.2 (1.8 seed 0 on the 2.2 brain, main and
control as `run_stage` builds them) against the final 2.2 populations of seed 0. Repeatability = correlation of the
same genomes' scores from disjoint worlds, measured directly:

| lives per genome | 1 | 4 (1 world) | 4 (4 worlds) | 16 (4 worlds) | 16 (16 worlds) | 64 (16 worlds) |
|---|---|---|---|---|---|---|
| gen 0 main | 0.03 | 0.10 | 0.13 | 0.34 | 0.44 | 0.68 |
| gen 0 control | 0.03 | 0.11 | 0.13 | 0.35 | 0.34 | 0.65 |
| final main | 0.03 | 0.09 | 0.11 | 0.31 | 0.33 | 0.62 |
| final control | 0.05 | 0.18 | 0.22 | 0.49 | 0.63 | 0.82 |

- Generation 0 is no better than the end of a run: the genetic spread is small at the start of a stage too (it
  starts from a population converged in the parent's world). One life is almost all luck everywhere.
- Lives add up as Spearman-Brown predicts. Selection accuracy (about the square root): now (4 siblings, 1 episode)
  0.32; episodes 4 (16 lives) 0.58-0.66; episodes 16 about 0.8. The response to selection per generation scales
  with it: episodes 4 about 1.9x per generation for about 2.3x the GPU time (v49).
- Lives in separate worlds are worth only a little more than siblings in one world (16 lives: 0.44 vs 0.34 in main,
  equal in control): genome x world interaction is small.

**Main vs control** (paired, same worlds): sd of the difference per world 57 at generation 0 (the two populations
are nearly the same animals), 197 for the final populations. Standard error of the difference, final populations:
re-evaluation 8 worlds 70, 16 worlds 49, 32 worlds 35, 64 worlds 25 (2 s.e. = 14%, 10%, 7%, 5% of main); training
curve over 50 generations (world noise only, if main and control see the same worlds) 28 with 1 episode, 14 with 4.
The spread between seeds seen in 2.2 fits these errors: last 50 generations +83, +145, +91 (sd 34, against 28
expected), re-evaluated in 16 worlds -14, +151, +83 (sd 83, against 49): most of the difference between seeds is
evaluation and world noise, not genuinely different outcomes.

**Small worlds as episodes** (`scripts/probes/small_world.py`, user 2026-10-19): 128 x 128, 64 animals, 80 patches
(every density as in W22), 1 sibling, 128 worlds run 16 at a time. Economy unchanged: generation 0 fitness 610 vs 603
in the large world, lifetime 937 vs 929, meals 83 vs 84; final populations 932 / 868 vs 981 / 890 re-evaluated large.
Repeatability with 16 lives (16 small worlds): 0.35, 0.34, 0.31, 0.52, the same as 4 large worlds x 4 siblings
(0.34, 0.35, 0.31, 0.49). The 16 x 1 figure from large worlds (0.44, 0.63) does not carry over: a small world adds
its own luck. Main vs control paired, final: 128 small worlds s.e. 24, 32 large worlds 35 (same animal-lives), but
the small worlds take 1.5-2x the GPU time (main: 7-10 s per 16 small worlds against 5 s per 4 large), so per GPU
second the two are equal. Small worlds buy nothing: episodes 4 in the stage's own world is the choice.

**One large world, 16 siblings** (`small_world.py FACTOR=0.25 SIBLINGS=16 WORLDS=16 BATCH=1`): 512 x 512, 1024
animals, 1280 patches, 64 genomes x 16 siblings. Repeatability with 16 lives (one world): generation 0 0.33 / 0.32
(as 4 x 4: 0.34 / 0.35), final main 0.13 (4 x 4: 0.31), final control 0.40 (0.49); two worlds (32 lives) 0.54 / 0.54 /
0.23 / 0.56. About 25% faster than 4 x 4 (gen 0 main 4.5 s against 5.9 s per generation). The economy shifts: fitness
709 vs 603 at generation 0 (lifetime 1063 vs 929), final 1057 / 935 vs 981 / 890; the small worlds (610) did not, so
it is not the edge. Main vs control, final, paired: sd per world 188, s.e. with 16 worlds 47 (as 64 worlds of 256).
Not better than 4 x 4: the same at generation 0, worse late in a run, a little cheaper.

## v53 (2026-10-19): budget test stopped; GPU memory per setting

**Budget test** (side stages `x.b1`, `x.b4`, `x.b256`: 200 generations, episodes 4 for 87, 256 genomes in one 512 x 512
world for 114) stopped by the user after about 10 minutes: too large for a quick check. With three runs sharing the
GPU, the per-generation costs differed from the solo measurements: 7.8 s, 22 s (2.8x) and 26 s (3.3x), so the
budgets were not equal anyway. Three runs at a time filled the laptop's RAM (Windows paged; WSL held 4.3 of 5.9 GB)
and the GPU (7.6 of 8.1 GB). No results; the definitions and `budget_eval.py` were removed again (git history, commit b523a28).

**GPU memory of one process** (`scripts/probes/gpu_mem.py`, one process per setting, alone, idle 0.4 GB): peak JAX
memory of one generation 327 MiB (2.2: 256 animals, 1 episode), 711 MiB (episodes 4), 1293 MiB (1024 animals in one
world); the recording of the last generation (64-animal sample, 4000 ticks) about 900 MiB in every setting. nvidia-smi
peak per process, CUDA context and allocator included: 1.6, 2.1 and 2.9 GB. During evolution, before the recording,
about 1.0, 1.4 and 2.0 GB. Runs record one at a time (`recording_lock`), so three episodes-4 runs need about
3 x 1.4 + 0.9 = 5.1 GB of the 8.1 GB, which fits when Windows uses little of the GPU; 256 genomes in one large world
would fit two.

## v54 (2026-10-19): toward the dopamine critic: a snake world, the critic at generation 0, lab tests

Probes only; no stage was run or frozen. Side stage `x.critic` = the 2.2 brain and world (DRAFT), generation 0 of
the frozen 2.2 population (seed 0), `DENSE=1`, 16 paired worlds. Base (pain-gated learning, as 2.2) 1010 +- 58,
learning off 715 +- 35.

**Snake world (for a later memory stage).** `WorldConfig.snakes` etc.: hidden rattlesnakes (OHOL 764, not visible)
hop every 10 ticks; a creature that comes within `rattle_radius` hears one rattle (one tick, silent at the bite), a
bite costs food and pain. Wind gusts share the signed `hear` channel (rattle +s, gust -s, s drawn per world;
`BodyConfig.hear`, not wired to the pallium). Frozen 2.2 (deaf), baseline 1059: 32 snakes, radius 4, 1 food per bite
-114 +- 38; 2 food -219; 64 snakes -253. Usual gap rattle -> bite 3-5 ticks; about a quarter of bites come without
a rattle. Proposed for the memory stage: 32 snakes, radius 4, 2 food per bite (gaps of 2, 4, 6 to test).

**The critic as a formula** (`scripts/probes/variants.py`: value cells read the pallium, one-tick copies, the error
da = taste - pain + k (gamma V - V') computed as a modulator; ADR-025 plans to replace the formula by cells):
- `td` (the error teaches every pain-gated synapse) 148, `tdp` (only those from the pallium) 226 at first. Cause:
  **inherited plasticity genes override `abcd`** (`remap_genomes` copies A-D for a projection that keeps its rule),
  so the sign meant to flip with the error did not: good meals taught aversion. Fixed by teaching those synapses with
  the negated error (modulator `da_neg`) instead.
- **Timing** (`ProjectionSpec.elig_delay`, new): the error of tick t is about the move from t-1 to t, so a trace
  must take the presynaptic activity of t-1 (dopamine acts on activity 0.3-2 s earlier, Yagishita et al. 2014);
  the value cells read the pallium of the same tick (phase 1). `freeze.with_defaults` now fills new options of
  projections and regions too (stored definitions predate `elig_delay`).
- `td2` (one error per valence, the PAM / PPL1 split): 864 without the delay, 935 +- 33 with it; gamma 0.7
  1088 +- 44 (+78 +- 38), valence weight 0.5 1009. Set aside by the user for now: a single signed error is the
  standard mammalian picture and should work.
- Single error after both fixes: gamma 0.9 `td` 722, `tdp` 897; gamma 0.7 770 / 829; scale 3 667 / 490; eta 0.3
  618 / 642; **taste weight 0.3: 981 +- 37 / 987 +- 43** (-29 / -24); gamma 0.5, scale 3, taste 0.3: 960 +- 50 /
  853 +- 59 (`td` eats more good berries than base, 45.6 vs 39.7 in the second half, but also more poison).

**Can taste be predicted at all?** (`scripts/probes/taste_predict.py`, one recorded life, offline fits of the taste
of t+1.) Object ahead + action R^2 0.998; object ahead alone 0.80; the pallium 0.51, with weights >= 0 0.50, its
pattern as a lookup key 0.51. The pallium is not stale (the same tick fits best) and loses nothing it gets, but it
reads only the look of the first object in line of sight, not its distance (`vis+0.near`): a bush at the mouth and
one four cells away are the same state. The TD fixed point also needs values of about 2.3 bites for a good bush at
gamma 0.9, while the value cells reach 1. Agreed: give the value cells more context (distance first; an efference
copy of the action later, if at all).

**Lab tests** (`life/labtests.py` PROTOCOLS + `conditioning`, `scripts/probes/lab_conditioning.py`; scripted cue
looks, cue alone 1 tick, cue + pain 3 ticks, 4 blank; readout the aversive cells' input drive, as their rate
saturates within two trials). Should have come before generation 0 (BRAIN_EVOLUTION principle 4; now in `CLAUDE.md`).
At gamma 0.5, scale 3, taste 0.3, `td`:
- the error moves to the cue (cue tick 0 -> -0.6 from trial 2), the pain becomes 60% predicted (-0.96 -> -0.4 per
  tick); the base teacher is the same on every trial;
- extinction: positive error when the pain is left out (+0.55 per tick on the first omission), critic value 0.54 ->
  0, aversion drive 6.5 -> 4.5 (base stays at 6.9);
- blocking: B learned in compound after A, drive 1.08 +- 0.18 vs 1.74 +- 0.22 after an unrelated pretraining
  (base 1.93 / 1.95);
- generalisation as in base: a look-alike (80%) takes most of the value, the opposite colour direction none.
- Flaw: a relief burst on the tick the cue disappears (+0.7 every trial). The state is only what is in view, so the
  critic cannot know when the pain stops; at gamma 0.9 the pain cannot be predicted away at all (a value of
  pain / (1 - gamma) = 10 per tick would be needed). TD models need a representation of time since the cue for this
  (complete serial compound; Ludvig, Sutton & Kehoe 2008): the general recurrent memory saved for a later stage is
  needed even without a gap between cue and outcome.

Next: cue shift far -> near and second-order conditioning once the value cells read `vis+0.near`; how the relief
burst should be handled; the critic stage itself is still to be agreed with the user.

## v55 (2026-10-19, evening): a per-life pallium; distance for the critic; larger pallia

Probes on `x.critic` at generation 0 (frozen 2.2 population, seed 0, `DENSE=1`, 16 paired worlds; base 1010 +- 58).

**Per-life pallium wiring** (`ProjectionSpec.per_life`, new; user: option 1): the pallium's input synapses were drawn
once per genome and inherited unmutated (selection among the first draws only). Now, where set, they are drawn anew
at every birth for each sibling (`brain.init_state(key=...)`, `run.make_simulate`), as the mushroom body's wiring
differs between individuals (Caron et al. 2013); what reads the pallium starts at 0 and is not evolved (variant
`plife`). The frozen 2.2 keeps its inherited code. Generation 0: +9 +- 21 (no cost).

**Distance for the critic.** The pallium reads only the look of the first object in line of sight, so a bush at the
mouth and one four cells away are one state (v54). Offline (`taste_predict.py`): `near` as a separate input adds
nothing (R^2 0.51 -> 0.52), the pallium gated by "at the mouth" reaches 0.73 (ceiling 0.80). `near` fed into the
pallium splits every look into unrelated codes per distance (R^2 0.20); a new `touch` sense into the pallium spread
lessons to good types (XOR lab good 0.26: at the mouth it pushes the same cells in every look) and was removed again.
The existing `no_touch` cell (inhibitory, about 0 at the mouth, 0.9 otherwise) into the pallium, one phase later so
both describe the same tick (variant `pnt`, value cells one phase after the pallium): per animal R^2 0.50 -> 0.58,
with 512 cells 0.68.

**Pallium size and k** (`scripts/probes/pallium_sweep.py`: taste R^2 per animal; XOR lab with only the pallium
learning, aversion poison / good):

| variant | R^2 | XOR poison / good |
|---|---|---|
| base | 0.49 | 0.36 / 0.09 |
| plife | 0.50 | 0.50 / 0.10 |
| plife + pn512 | 0.51 | 0.60 / 0.02 |
| plife + pnt | 0.58 | 0.47 / 0.09 |
| plife + pnt + pn512 | 0.68 | 0.55 / 0.02 |
| plife + pnt + pn512 + pk4 | 0.58 | 0.93 / 0.06 |
| plife + pnt + pn1024 (32 animals) | 0.66 | 0.53 / 0.02 |
| plife + pnt + pn1024 + pk4 | 0.63 | 0.97 / 0.04 |
| plife + pnt + pn1024 + pk8 | 0.66 | 1.00 / 0.13 |

At 128 cells more winners spread lessons to good types (pk4 0.69, pk8 0.91, with touch); at 512 and 1024 cells k = 4
discriminates almost perfectly. 1024 cells do not fit a full population in WSL's 5.9 GB: the remap runs in host
memory (about 20 arrays of genomes x n x n); the probe uses 32 / 24 animals there.

Generation 0, vs base: plife + pn512 **+46 +- 18**; plife + pnt + pn512 -26 +- 18; plife + pnt + pn512 + pk4 +10 +- 21.
`no_touch` costs about 70 without a critic (a lesson at the bite reaches the bush in sight less well).
`approach_trace.py`: the pallium is not silent while approaching with `no_touch`, aversion by distance unchanged;
moving the pallium a phase later alone changes nothing. Single recorded worlds diverge with any change of the
per-life draws, so their lifetimes are not comparable.

**The critic on these pallia** (TD_TASTE 0.3, gamma 0.9), vs plife + pn512 (1056 +- 66): tdp -70 +- 38, pnt + tdp
-91 +- 32, pnt + td -122 +- 57, pnt + pk4 + tdp -232 +- 41; the critic variants eat fewer good berries in the second
half (29-35 vs 42.5). With distance the critic still does not pay at generation 0 in this world.

## v56 (2026-10-19, night): the reversal world is the critic's driver; cost of a larger pallium

**Cost** (`scripts/probes/cost.py`: one generation as evolution runs it, 4 episodes x 256 animals x 4000 ticks, one
process alone): base (252 neurons) 6.9 s, 0.7 GB GPU; per-life wiring 8.5 s, 1.1 GB (every animal holds its own
fixed weights); pallium 256 (380 neurons) 25.2 s, 2.4 GB; pallium 512 (636 neurons) does not fit (6.0-6.4 GB asked).
The brain is one dense n x n matrix per animal, so time and memory grow with n^2 although the pallium has few
synapses. Decision (user): a larger pallium only if crucial to the critic; the critic is tested at 128.

**Reversal world** (side stage `x.reversal`: x.critic with `berry_world(reverse=True)`: at `switch_tick` the two
poison novel types turn good and the two good ones poison, looks unchanged; `REV_SWITCH` 800, as only 45% of the
incoming population is alive at tick 800 and 11% at 2000). `scripts/probes/reversal_curve.py` (bites per living
creature over the life, P(bite) and aversive cells with each kind at the mouth before / after; plot in runs/logs);
`gen0.py` now counts good and poison per phase.
- Both base and `td` (gamma 0.5, scale 3, taste 0.3) adapt within about 100 ticks: the new poisons hurt, and the
  former poisons are still tried (P(bite) 0.21 at the mouth before the switch). After it, P(bite) on a former
  poison 0.47 (base) / 0.66 (td); alive at tick 2000 11% / 19%.
- Why the base adapts at all: **input adaptation** (habituation of the look, `in_adapt`). The aversive cells facing
  a former poison drop 0.31 -> 0.19 in base; with adaptation off (variant `noadapt`) 0.32 -> 0.30 and P(bite) only
  0.33. The pain-gated synapses do not extinguish, as the lab test said. Lab interference (`life.labtests`
  PROTOCOLS `interference`, `xor interference`): pain lessons on B lower the aversion to A by 18% with random looks,
  not at all with the XOR corners of this world.
- **Generation 0** (16 paired worlds): base 688 +- 20; learning off 725 +- 15 (+36 +- 14: pain-gated learning hurts
  in a reversal); **td 832 +- 19, +144 +- 17**; tdp 739 +- 23, +51 +- 23. td eats 67 good berries per creature
  after the switch against 34 (base) and 52 (learning off), with somewhat more poison. The first world where the
  critic pays: what was learned stops being true, and only a prediction error can unlearn it.

Open: the switch time (and several reversals per life, not supported yet); the critic's settings in this world;
then lesions and the stage proper with the user.

## v57 (2026-10-19, night): stage 2.3, the dopamine critic as a formula; frozen

**Generation-0 sweep** (at most 15 minutes, user): the reversal settings of v56 kept (switch at 800, gamma 0.5,
value scale 3, taste weight 0.3; "if it works don't touch it"). Gamma, scale and taste are not evolvable while the
error is a formula (user); they become genes when it is built from cells.

**Per-slot pallium wiring** (`ProjectionSpec.per_slot`, `run.SLOT_KEY`): the random in -> pallium weights belong to
the population slot, not the genome: the same draw for slot i in every generation, siblings share it, in main and
control alike. Inherited lucky codes are ruled out without the cost of per-life draws (v56: 8.5 s vs 6.9 s per
generation). Dropping the dead keeps each brain state with its creature (test). Projections out of the pallium
start at 0 and are not evolved.

**Stage 2.3** (`B23` = per-slot pallium on the critic, `W23`, EVOLUTION_NEW, 100 generations, three seeds; control:
the 2.2 brain with per-slot wiring in the same world). Last 50 generations main - control +151 +- 20 (+164, +113,
+176); re-evaluated in 64 worlds +127 +- 7 (1369 vs 1242); head to head +177 +- 50 (+213, +240, +79).
- **100 generations are needed.** Gap by generation: +101 at 0, about -25 at 20, +20 at 40, +128 at 60, +165 to
  +171 at 80-95. A 40-generation run would have shown nothing; learning stages keep 100.
- Lesions (64 worlds, % of intact): value_app 84 83 68, no_plasticity 83 83 83, pallium 83 82 80, us_taste 89 93
  80, **value_av 101 100 97 (unused)**. value_app_prev 23 33 20 and value_av_prev 33 57 82 are not a measure of use:
  silencing the copy alone leaves a standing error that unteaches every aversion.
- Evolved learning rates (start): in -> valence_av (0.3) main 0.04-0.17, control 0.04-0.10; pallium -> valence_av
  (0.2) main 0.08-0.10, control 0.03 / 0.22 / 0.04; pallium -> value_app (0.1) 0.06-0.08; -> value_av (0.1)
  0.03-0.04. The control turns pallium learning down in two seeds (pain-gated learning hurts in a reversal, v56).
- In the world (`reversal_life`, last generation, recorded): both populations switch at about the same rate.
  P(bite) at the mouth on a former poison 0.34 -> 0.85 (control 0.38 -> 0.83), on a former food 0.87 -> 0.32 (0.88
  -> 0.31). The control switches through input adaptation (v56). Main eats more throughout (172 vs 135 per life),
  more pain (19.8 vs 16.4), and lives longer (1743 vs 1596 ticks).
- Lab (`conditioning`, 24 creatures per seed, drive of the aversive cells): after 6 painful meals 2.7 (control 1.2);
  after 12 more without pain 1.8 (control 1.2: no extinction without the critic). Blocking: second cue -0.16 vs
  -0.03 in the control arm (control -0.86 / -0.85): the second cue is not learned in three bites in either, so
  nothing to block. Error per tick of a trial (`dopamine_shift`): first meal 0 at the bush, -0.96 at the bites;
  sixth meal -0.31 at the bush, -0.66 at the bites; first meal without pain +0.33 at the bites (the omitted pain).

Frozen (tag `stage-2.3`), `value_av` kept as half of the critic that the next stage builds from cells. Book page
`book/critic.qmd` (extras `reversal_life`, `conditioning_lab`, `dopamine_shift`; new `lines` component).

## v58 (2026-10-20): the reversal crossover is mostly the visit-ending reflex

`scripts/probes/reversal_variants.py`: the book's reversal figure for the frozen 2.3 populations (last generation,
64 creatures, two worlds per seed) in worlds they did not evolve in. New `WorldConfig.switch_every` (phases alternate
every N ticks from switch_tick; 0 = one switch).
- **every=400**: both brains switch within one 100-tick bin at all six swaps, without slowing. P(bite) at the mouth
  0.33 -> 0.85 / 0.84 -> 0.33 (critic), 0.35 -> 0.83 / 0.85 -> 0.34 (control). Alive at the end 29% both.
- **one=3** (one berry per bush worth 3, regrowth as for 3; every use takes the last-use transition, since a bush
  with num_uses 1 would otherwise never empty): no discrimination at all. P(bite) poison-first 0.46 / 0.47, good-first
  0.46 / 0.45 (critic); 0.55 / 0.49, 0.58 / 0.46 (control); alive at the end 20% / 23%.
So the crossover in the 6-berry world is mostly bites per visit (stop at the first painful berry), which reverses by
itself; a learned aversion by look hardly shows in the world although the lab shows one. Open, first next session
(STATUS step 0).

## v59 (2026-10-20): why the 2.3 critic does not judge a bush by its look; one-berry bushes

Probes on the frozen 2.3 populations (last generation); no stage was run.

**First bite on arrival** (`scripts/probes/first_bite.py`, read from the recordings of the last generation, one
world of 64 creatures per seed, no new simulation; a visit = consecutive ticks with the same full bush at the
mouth). Pooled over three seeds, main (control): P(any bite in a visit) before the switch poison 0.46 (0.49), good
0.52 (0.50); after it poison 0.43 (0.41), good 0.47 (0.42). The bite comes on the first tick of the visit or not at
all. A second bite after a painful first: 0.05-0.09 (0.04-0.06); after a good first: 0.92-0.97. P(bite) on a poison
type does not fall with earlier painful bites from that same type (before the switch 0.40 with none, 0.48-0.51
with 1-10+; control 0.42 -> 0.49-0.59). **No brain judges a bush by its look; all of the discrimination is the
second bite.** At arrival the aversive cells are low and alike for poison and good (main, after the switch, 0.16 vs
0.14); main's appetitive cells are about three times the control's (0.35-0.39 vs 0.10-0.12), driven by `value_app`
(0.05-0.11, somewhat higher on good types).

**Where the learning pays** (`gen0.py`, frozen 2.3 seed 0 on its own brain, 16 paired worlds, 2.3 world): intact
1415 +- 35; look -> aversion learning off (`in->valence_av[vis*.app*]`, `pallium->valence_av`) -47 +- 15; critic
value learning off (`pallium->value_app`, `pallium->value_av`) **-254 +- 28**; all learning off -268 +- 24. Without
the value learning the creatures eat fewer good berries (63.6 vs 79.7 after the switch) and fewer poison (12.4 vs
15.0). So the no_plasticity lesion (83%) measures a learned appetite for bushes (value_app -> valence_app), not
an aversion by look: the user's hunch holds for discrimination, but evolution did not switch learning off.

**Why the lab error only partly cancels the pain** (`scripts/probes/critic_ceiling.py`, conditioning, A -> pain,
60 trials, three seeds x 24 creatures). Evolved rates: bites -0.97 (trial 1), -0.67 (6), -0.53 (24), -0.50 (60); cue
-0.40; the tick after the cue is gone +0.40. Value learning x5: -0.55 at 60; x20: -0.71 (overshoot). Not too few
trials: a ceiling. The state is only "which look is ahead", and da = -pain - 1.5 V(t) + 3 V(t-1) (gamma 0.5, scale
3): while the cue stays in view the value drains by (1 - gamma) per tick, so every in-view tick gives a positive
error, and the tick after the cue goes gives the relief burst, both credited to the cue (`elig_delay`). Cue in view
for 1 / 4 / 8 / 16 ticks before the bites (protocol option `lead`, new): bites at trial 60 -0.50 / -0.57 / -0.75 /
-0.87, with +0.44 / +0.30 / +0.15 per in-view tick.
**In the world** (recordings, `mod` da) the error one tick after a painful bite is -0.97 at the first one and
-0.89 to -0.96 after 11 or more: the pain is never predicted. A bush is usually in view for many ticks before
arrival (median 15-18 ticks with something ahead), and a visit holds one painful bite, not three. The pallium code
at arrival is stable per type (cosine 0.59-0.66 within a type, 0.03-0.05 between), so the code is not the problem;
the state without distance or time is (v54, v55: TD needs a state that changes as the outcome nears).

**One-berry bushes** (`berry_world(one_berry=F)`, new, default off; draft stage `x.one` = 2.3 with one berry worth
`ONE_BERRY` berries; generation 0 of frozen 2.3 seed 0, 16 worlds; phases before / after the switch):
| F | intact | look -> aversion off | value learning off | all off | good / poison eaten (after the switch) |
|---|---|---|---|---|---|
| 6 berries (2.3) | 1415 | -47 +- 15 | -254 +- 28 | -268 +- 24 | 79.7 / 15.0 |
| 3 | 1061 | -29 +- 18 | -120 +- 18 | -147 +- 18 | 15.2 / 13.9 |
| 2 | 523 | +22 +- 15 | -73 +- 13 | -92 +- 10 | 7.7 / 7.0 |
| 1 | 72 | -1 | -3 | -4 | (dead by tick 220) |
At F = 3 the berry counts read directly as decisions (good 15.2 vs poison 13.9: no discrimination at all), and
trying every bush costs a third of a good bush instead of a sixth. F <= 2 starves the population. One-berry bushes
alone do not make the look matter at generation 0, since the critic cannot learn an aversive value from a look seen
from afar in either world.

## v60 (2026-10-20): the touch-gated pallium (`pnt`) in the one-berry world: lab, generation 0, one seed

User's choice (after v59): the critic's state should change as a bush nears; the best variant at 128 cells in v55
was `pnt` (the pallium also reads `no_touch`, look x at-the-mouth); no larger pallium (cost, v56). Now
`stages.pallium_touch` (= probe variant `pnt`; on a brain with the critic the value cells move one phase later with
the pallium, so they still read this tick's code; the probe variant was fixed for that too).

**Lab** (`critic_ceiling.py ... approach v=pnt`, frozen 2.3, three seeds x 24, 60 trials of A -> pain, evolved
rates; new protocol option `approach`: all cue ticks but the last seen from afar, `vis+0.near` 0.4):
| cue in view before the bites | plain: bites at trial 24 / 60, aversive value | pnt: bites at 24 / 60, aversive value |
|---|---|---|
| 1 tick (at the mouth) | -0.53 / -0.50, 0.49 | -0.54 / -0.52, 0.49 |
| 8 ticks (7 from afar) | -0.77 / -0.75, 0.40 | -0.65 / -0.58, 0.52 |
| 16 ticks (15 from afar) | -0.88 / -0.87, 0.29 | -0.73 / -0.65, 0.50 |
With `pnt` the far ticks carry almost no error (+0.01 to +0.05 instead of +0.15 to +0.30): the far code no longer
drains the value at the mouth. Learning is slower with a long approach (trial 24 vs 60), but the value at the mouth
reaches what a cue at the mouth alone gives.

**Generation 0** (`gen0.py`, STAGE=x.one, frozen 2.3 seed 0, 16 paired worlds; good / poison eaten after the switch):
| world | plain | pnt vs plain | learning off vs plain | lifetime | pain |
|---|---|---|---|---|---|
| one berry 3 | 1061 (15.2 / 13.9) | -42 +- 14 | -147 +- 18 | 1434 | 22 |
| + born 0.75 full | 1249 (17.6 / 16.2) | -34 +- 16 | -174 +- 24 | 1671 | 25 |
| + born 0.75, hunger 0.015 | 1972 (20.6 / 19.4) | -37 +- 16 | -158 +- 29 | 2523 | 26 |
| one berry 2, born 0.75, hunger 0.0167 | 1169 (15.8 / 14.3) | -62 +- 11 | -157 +- 23 | 1616 | 23 |
`pnt` costs 35-60 on the inherited population (its codes are new; v55: about 70 without a critic); nothing
discriminates by look yet in any world. Chosen: one berry worth 3, born 0.75 full, hunger 0.015 (regrowth 800 ticks,
follows the metabolic rate): lives of about 2500 ticks with about 26 painful bites, the most lessons per life.

**Evolution, seed 0** (draft stage `x.onet`: main `pallium_touch(B23)`, control the 2.3 brain, same world, warm
start from frozen 2.3 seed 0, 100 generations, EVOLUTION_NEW; about 40 s per generation). Generation 0 in the run:
main 2030, control 2050, poison share of bites 0.48 in both.
By generation 70 (block means of 10): main 1959 -> 2200, control 2010 -> 2222; **poison share of bites 0.48-0.49
in main and 0.48 in control throughout**: no look discrimination evolves in either within 70 generations.

**Where lab and world part** (lab on frozen 2.3, three seeds x 24; `labtests` readout options `test_far` /
`test_ticks`, new; `scripts/probes/lab_mixed.py`, new):
- P(bite) on the trained look in the lab: one painful trial takes it from 0.50 to 0.16-0.19, two to 0.02 (plain and
  pnt alike, readout as before: a rested brain, the cue 3 ticks at the mouth). Read on the first tick at the mouth
  after a sudden appearance: no effect (0.49 after two trials, 0.25 after 24); after 8 ticks from afar and one at
  the mouth (as the world decides a bite): 0.18, 0.02 (pnt: 0.34, 0.16, 0.04 after 6). The aversion needs a few
  ticks of the look to build up, which an approach gives.
- Interleaved good meals (1 pain lesson on A, 3 good meals of B, 6 rounds): A 0.05 / B 0.83 (pnt 0.16 / 0.81).
  With the world's XOR looks (pain on X++ and X--, 3 good meals of X+- and X-+ each): poison corners 0.08, good
  0.54 (pnt 0.15 / 0.58). Drive of the aversive cells: -1.53 at start, +0.5 per painful lesson.
- Hunger: in the 2.3 recordings P(bite) on a known poison type (3+ painful bites earlier) is 0.4-0.55 at every
  stomach fullness; not the gap.
- **In the world the learning separates the types, but about 15x more weakly per lesson.** Learned drive onto
  `valence_av` from the mean pallium code of each type at arrival, with the weight snapshots of the 2.3 recording
  (seed 0): critic tick 300 poison 0.15 / good 0.06, tick 700 0.28 / 0.10 (control 0.11 / 0.04); the look synapses
  carry nothing (+-0.02). After about 5-8 painful bites by tick 700 that is 0.03 per lesson against 0.5 in the lab.
  Known parts of the difference: a lab lesson is three ticks of pain, a world bite one (pain_decay 0); the
  in-view ticks before a bite give a positive error that, negated, unteaches the aversion of the same code (the
  2.3 brain teaches its aversion with `da_neg`).
- **One tick of pain per lesson** (lab option `outcome_ticks`, new; XOR protocol as above, plain brain): drive
  +0.2 for the first lesson, then about +0.1 per lesson (three ticks: +0.5); P(bite) poison / good corners after 6
  rounds 0.22 / 0.60 (three ticks: 0.08 / 0.54). A world bite gives one tick of pain (`pain_decay` 0 since W15), so
  each lesson is three to five times weaker than the lab's; the rest of the factor 15 is still open.

**Seed 0 finished** (`runs/x_one_touch/20261020-110332`, `runs/x_one_touch_control/20261020-110338`). Last 50
generations: main 2216 +- 14, control 2305 +- 10 (main - control about -90); poison share 0.493 / 0.477. Recordings
(`first_bite.py <run dirs>`): P(bite) per visit, main, poison / good before the switch 0.42 / 0.43, after it
0.41 / 0.39 (control 0.41 / 0.47, 0.39 / 0.39); on a poison type with 0, 1-2, 3-5, 6+ earlier painful bites
0.29, 0.43, 0.49, 0.64 (rising, as in 2.3). The split by stomach fullness is confounded (a creature that does not
bite stays hungry), so not used. **Evolved learning rates** (median per projection; frozen 2.3 seed 0 -> main /
control): look -> valence_av 0.041 -> 0.018 / 0.018; pallium -> valence_av 0.070 -> 0.043 / 0.108; **pallium ->
value_app 0.069 -> 0.223 / 0.236**; pallium -> value_av 0.024 -> 0.025 / 0.007. Evolution triples the appetitive
value learning in both and turns the aversion learning down in main: in this world too, what pays is a learned
appetite for bushes, not caution by look.

**Lingering pain at generation 0** (x.onet, frozen 2.3 seed 0, 16 worlds; `pain_decay` 0.5 = 1, 0.5, 0.25 ...
after a poison bite): 1754 vs 1935 at 0; after the switch good / poison 16.8 / 15.1 (at 0: 20.1 / 19.1); learning
on - off +106 vs +121. A hint of discrimination at a cost in food; not evolved.

Summary: `pnt` works in the lab (the far view no longer drains the value at the mouth), but in the world neither
brain learns caution by look, at generation 0 or after 100 generations; a world lesson (one tick of pain) is three
to five times weaker than the lab's, and the rest of the gap (about 15x in learned drive per lesson) is open.
Not tried yet: a stronger single lesson (pain value per bite, `poison_pain`), the unteaching by the positive error
during the approach (`da_neg` on the pain-taught synapses), a lab readout from the world's own weight snapshots.

## v61 (2026-10-20): bridging the lab and the world (replays of recorded inputs)

`scripts/probes/bridge.py` (new): `record` lives (64 creatures, one world, the run's last population) with the raw
inputs every brain received (`make_simulate(record_obs=True)`, new); `replay` feeds them back into the same brains,
open loop, with parts of the stream removed, and reads the weights of tick 800 (the switch) out as in the lab (a
rested brain, each novel type's look 8 ticks from afar, one at the mouth); `arrive` recomputes the chance of a bite
on the first tick of every visit (ticks 400-800) from the replayed state, as it was and with one thing changed
(`brain.step(with_logits=True)`, new). The replay reproduces the recorded weights (largest difference 0.006-0.008,
the float16 recording). Frozen 2.3 seed 0 in the 2.3 world unless stated.

**The learning is in the weights.** Lab readout of the world's weights at tick 800 (about 7 painful bites per
creature by then): P(bite) poison / good 0.46 / 0.74, aversive drive -0.90 / -1.13. Stripping the stream changes
little: side eyes blank 0.46 / 0.74; also the eye ahead blank unless a full bush 0.40 / 0.81; also blank outside
visit windows 0.38 / 0.81. Clutter in the stream is not the gap.
**Lab lessons were stronger than world lessons.** The XOR lab (pain on two corners, three good meals of each of
the other two per lesson) with world-like lessons (`outcome_ticks` 1, `linger` 0, new options: one tick of pain,
none after) after 8 lessons: drive -0.97 / -1.40, P(bite) 0.39 / 0.61, the world's values. With three ticks of
pain and 0.3 after (the old lab trial): +1.28 / -0.66, 0.08 / 0.54. So the lab overstated a world lesson.
**Pain intensity cannot make a lesson stronger**: `us_pain` = tanh(2 pain) is 0.96 at pain 1 and 1.00 at pain 3.
World with pain 3 per poison berry (`stages.POISON_PAIN`, new, default 1; the food cost stays `POISON_FOOD`):
readout drive -0.90 / -1.13, the same as at pain 1. Only the learning rates set a lesson's size, and evolution
lowers them.
**What lingers after a bite** (w23 recording): the pain is felt one tick after the bite, with the same bush at the
mouth in 99% of cases (the lesson lands on the right bush); a tick later the creature has turned: good bush or empty
43%, poison bush or empty 30%, the same bush 16%. With mixed patches a lingering pain would mostly teach aversion
to what happens to be in view; the relief error there (+0.1) is small and is credited to the tick before (the
same bush).
**At the bush the learning hardly decides** (`arrive`; P(bite) on the first tick of a visit as in the world /
weights of birth / only the aversive synapses at birth / only the appetitive ones / no input adaptation):
| recording | poison | good |
|---|---|---|
| w23 (2.3 world) | 0.54 / 0.60 / 0.60 / 0.54 / 0.70 | 0.65 / 0.67 / 0.67 / 0.65 / 0.78 |
| easy looks (SIM 0, XOR off: a random look per type) | 0.47 / 0.52 / 0.52 / 0.47 / 0.50 | 0.51 / 0.51 / 0.51 / 0.51 / 0.49 |
| onet1 (x.onet main, its one-berry world) | 0.44 / 0.48 / 0.48 / 0.44 / 0.57 | 0.50 / 0.51 / 0.51 / 0.50 / 0.59 |
Learning lowers the bite on poison by 0.04-0.06 at arrival, all of it through the aversive synapses; the learned
appetite does nothing on that tick (its fitness value, v59, is elsewhere: approach or staying). With easy looks
the readout separates the types well (0.26 / 0.47) while the world does not (0.47 / 0.51): the weights know, the
decision at the bush does not use them. Input adaptation lowers bites on everything (0.70 -> 0.54) but is not the
gap. x.onet's own weights read out at 0.95 / 0.97 (drive -1.39 / -1.42): after 100 generations it learns almost no
aversion. Open: why the same weights move the lab readout (8 ticks from afar, stomach 0.5, nothing else in view)
so much more than the decision at a real arrival.

**Correction and the path to the bite** (same day, later). `bridge.py swap`: at every real arrival the effect of
learning (learned minus birth weights) in five situations from the lab readout to the world. The readout gap of
the easy-looks world (0.26 / 0.47) was mostly innate: with birth weights 0.18 / 0.30; learning adds +0.04 / +0.09.
In w23 (lab readout at the arrival's weights) learning gives poison -0.09, good +0.35: mostly a learned appetite for
the good types. Replaying the real 9 ticks from the real state reproduces the world (0.54 / 0.54). So the learned
aversion is weak in the world, not unused; "the weights know" above was wrong.
`scripts/probes/bite_trace.py` (new; frozen 2.3 seed 0 recording, first tick of visits before tick 800, poison
after 2+ painful bites from that type): when it bites a known poison the aversive cells are silent (0.001; held
back 0.077), and so is `no_feed` (0.003; held back 0.137, which then inhibits the bite outputs by -1.15 and grasp by
-0.41). `hungry` inhibits `no_feed` only slightly (-0.03 to -0.05). Input to `valence_av` on those ticks: learned
looks +0.50 and pallium +0.37 and value_av +0.14 on a known poison (good: -0.43, +0.05, +0.05), so the learned
aversion separates the kinds by about 1.3; but the other eye inputs give -2.35 (hit and near of every eye: vis-60.hit
-0.69, vis+0.near -0.39, vis+60.hit -0.32, vis+30.near -0.30, vis+0.hit -0.22, ...; inherited weights, e.g.
vis+0.near -0.50, vis+0.hit -0.23) and the bias -0.5 to -0.7. **The learned aversion is real but stays about 2 below
the aversive cells' threshold**: an innate "things near me are not alarming" inhibition, evolved in chapter 1, masks
it. Seven one-tick lessons give about +1; breaking through needs about +3 (the old lab trials reached +2.8 in 12
three-tick lessons).

## v62 (2026-10-20): start values that let the learned aversion act; the reversal punishes it

`no_touch` works: with nothing adjacent (17% of ticks in 2.3, 29% in x.onet) it is at 0.96-0.97 and the bite
action falls to 0.30 / 0.17 (0.91 / 0.79 with something adjacent). So the inherited hit / near inhibition of the
aversive cells is not what allows eating up close; it keeps the aversion reflex quiet near anything.
`scripts/probes/rebalance.py` (new): generation 0 of x.onet (frozen 2.3 seed 0, 16 paired worlds) with start
values changed: `eta<k>` (look and pallium -> valence_av learning rates x k), `inh<k>` (inherited hit / near / agent
/ wall / held -> valence_av x k), `bias<d>` (valence_av bias + d). Poison share of the berries eaten after the switch:
| change | vs base | poison share |
|---|---|---|
| base (1935) | | 0.49 |
| eta0 | -12 +- 15 | 0.49 |
| eta4 / eta10 / eta20 | +7 +- 26 / -101 +- 33 / -226 +- 39 | 0.47 / 0.45 / 0.44 |
| inh0.5 / bias1 | -8 +- 21 / +2 +- 24 | 0.49 / 0.48 |
| eta4+inh0.5 / eta4+bias1 | -36 +- 24 / -34 +- 23 | 0.46 / 0.46 |
| inh0.5+bias1 | -179 +- 29 | 0.48 (shy of all bushes) |
| eta4+inh0.5+bias1 | -292 +- 31 | 0.43 |
| eta10+inh0.5 / eta10+bias1 | -218 +- 38 / -210 +- 39 | 0.43 / 0.44 |
Lowering the threshold alone does nothing or makes the creatures shy of everything; the learning rate is what
discriminates. **The reversal is what makes it cost**: with no switch in a life (`switch_tick` 4000) eta4
**+150 +- 33**, eta10 +125 +- 54 (poison share 0.47 -> 0.45 / 0.43), eta10+bias1 +32; with the switch at 2000 eta4
+45 +- 20, eta10 -35 +- 33; at 800 (the stage) eta4 +7, eta10 -101. Every lesson turns wrong at the switch and the
critic cannot unlearn it in time, so evolution lowers the aversion learning rates (v57, v60). The reversal is also
what made the critic pay (v56): the two goals pull apart in this world.
