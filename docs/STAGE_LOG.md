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
