# Brain evolution sequence: architectures, learning rules and experiments

The plan for growing the brain one module at a time, in roughly the order brains evolved. Chapter 1 is described
as built (it ends with the pallium, ADR-027); later chapters are the current plan. Where the project stands: `docs/STATUS.md`; numbers, runs and the
variants that were tried: `docs/STAGE_LOG.md`; the reasons behind design rules: `docs/DECISIONS.md`. Exact wiring:
`life/experiments/` (`chapter1.py`, `chapter2.py`; map in its README). Framework terms (regions, projections, `rule`, `modulated`, warm start) are those of
ADR-007, ADR-012 and ADR-013. Last brought up to date 2026-10-20.

The spine is Max Bennett's *A Brief History of Intelligence* (2023): **steering -> reinforcing -> simulating ->
mentalizing -> speaking**. Around it are finer steps taken from comparative neuroscience. The literature does
not settle many of these points, so every stage says how confident the claim is. "Mya" means million years
ago; all dates are approximate.

Stages are named, not numbered, in the book. A stage gets a key (`life/experiments/`) when it is defined; keys are
historical names, not positions (STATUS has the chain).

---

## 0. Where learning sits in the sequence

Bennett puts **associative learning inside breakthrough 1 (steering)**, in the first bilaterians. What comes with
vertebrates (breakthrough 2) is *reinforcement learning with prediction errors* and *pattern recognition* (the
pallium, which later becomes cortex). This project puts the pallium's first job, a sparse expansion of the look, at
the end of chapter 1: it is still taught by pain, the teacher of chapter 1, and insects have the same design in the
mushroom body. Chapter 2 starts with the prediction error (ADR-027).

The oldest associative learning is not plain Hebbian learning. In *Aplysia*, *C. elegans* and *Drosophila* a
conditioned stimulus (CS) becomes linked to an unconditioned stimulus (US) through a **third factor**, a
neuromodulator released by the US, and the synapse keeps a short trace of its recent activity because the US
comes after the CS:

> dW = eta · US(t) · eligibility(pre), with a raw US (not yet a prediction error).

So the step from "Hebbian" to "RL" is four smaller steps, and the chapters follow them:
(a) unmodulated adaptation (habituation; later clustering),
(b) US-gated learning with eligibility traces (classical conditioning, on the look and on a sparse expansion of
    it; chapter 1),
(c) the teacher becomes a *prediction error* (TD dopamine; chapter 2),
(d) the prediction error trains *action selection* (actor-critic basal ganglia; chapter 2).

---

## 1. Design principles

1. **One module or one rule per stage.** A stage adds regions and projections, or one new learning rule or
   core mechanism, not both at once where that can be avoided.
2. **New cells have a fixed meaning** (ADR-016). A module is a set of named cell types with hard-wired defining
   inputs and outputs and tunable strengths. Without that, evolution routes behaviour through the existing
   ganglion and the new module stays unused. Value cells and modes act on motor programmes, not on the motor
   neurons directly (ADR-020).
3. **Every stage has an ecological driver.** The world is built so that the new module pays, and the previous
   brain measurably lacks it. If the control solves the task, the task is not yet the right one; the control
   must have had the means and still not got there.
4. **Design first, evolution tunes.** A module is designed and checked step by step before any evolution: the
   brain alone in scripted situations (assay; lab experiments for learning stages), then generation 0 with the
   mechanism on and off in the same worlds, then a short evolution (about 40 generations) that tunes the module
   and the rest of the brain to each other (ADR-022, ADR-025). A change after which something must be learned
   again from scratch gets a longer run.
5. **A stage succeeds when its circuit is used**: three seeds, lesions per region below intact, main not clearly
   worse than control. Controls are warm-started from the same population (ADR-012). Older circuits are
   checked with lesions after every stage: a part that is idle for a stage drifts.
6. **What is learned must not be inheritable** (ADR-017). The looks and meanings a learner is about are drawn per
   life; ancestral things (a staple food, hot springs, walls, other animals) keep a fixed look so that the
   innate compartment has something to be innate about. From the pallium on, the weights of a learned region
   are drawn at each birth; the genome holds their statistics and rule genes.
7. **New world mechanics come from OHOL where possible**; a mechanism of our own only when OHOL has nothing
   that fits. Each behind a config switch that defaults off (ADR-010).
8. **Old circuits stay, keep learning, and compete with the new** (user, 2026-10-21). A new module does not
   replace, rewire or freeze what came before: brains add on top, and the innate and Pavlovian systems
   (hypothalamus, amygdala, brainstem) still drive approach and avoidance next to the basal ganglia that learn
   actions. So the valence cells keep their fixed outputs onto the eating and turning programmes, and the newer
   circuits act on the same programmes. Where the paths should differ is designed, not left to evolution: every
   new learned projection has a biological counterpart and a reason, and evolution tunes only strengths and
   learning rates (open all-to-all wiring lets evolution route everything through the oldest path). When a new
   teacher arrives, it teaches both sides that a mirrored pair of cells has (good and bad), unless a difference is
   meant and written down; the wiring review by hand (CLAUDE.md) checks this. Worlds and lab tests include cases
   where the old path is wrong (an omission schedule: food only when *not* approaching), so lesions show which
   path is used where. A path that ends up unused is investigated, and leaves the chain only at freezing.

---

## 2. Core mechanisms

| id | mechanism | status | used by |
|---|---|---|---|
| M1 | **Eligibility traces** for modulated rules: `dW = eta · mod · trace` | built | association; dopamine and actor |
| M2 | **Named modulators** written by designated cells, read by projections (plasticity) and receptors (state) | built | association (`us_pain`), affect (`alarm`, `serotonin`), dopamine |
| M3 | **Gain and receptor modulation**: a modulator changes a cell's bias or input gain | built (receptors, `gain` projections) | affect |
| M4 | **Adaptation**: an input passes on itself minus its slow average | built | habituation |
| M4b | **Short-term synaptic depression** (Tsodyks-Markram resource per synapse) | built, not on the chain | |
| M5 | **k-winners-take-all** inside a region | built | pallium |
| M6 | **Delta rule** with a teacher region: `dW = eta · pre · (teacher - post)` | built | cerebellum, cortex |
| M7 | **Efference copy**: the last action as input | built | hippocampus, forward models |
| M8 | **Offline mode**: motor output gated off, predicted input replaces real input (thalamic gate) | to build | chapter 3 |
| M9 | **Observation extensions**: aftertaste, appearance noise and occlusion, other agents' facing, held item and last action, a sound vector | partly built (noise) | per stage |
| M10 | **World mechanics**: temperature and hot springs, patches (with a purity), dud bushes, mid-life rule switch, delayed sickness (built); moving objects (OHOL `move`), seasons (to build) | partly built | per stage |

Delayed sickness is built but not used: it teaches whatever is in view when the sickness starts. A stage that
uses it must show that the blame lands on what was eaten (for example a trace of what was in the mouth).

### 2.1 Where STDP fits

Spike-timing-dependent plasticity is not a stage of its own: it is a synaptic mechanism found wherever it has
been looked for (*Xenopus* tectum, fish, insect mushroom body, mammalian cortex and hippocampus) and most likely
present from the first centralised nervous systems on. In this rate-coded framework (ADR-002) it is the `trace`
rule. One tick is about a second, so it behaves like behavioural-timescale plasticity (BTSP; Bittner et al.
2017) rather than millisecond STDP; millisecond STDP needs `steps_per_tick > 1` and is kept for specific tests.

| role | where | how |
|---|---|---|
| eligibility trace for three-factor learning (Izhikevich 2007; Gerstner et al. 2018) | association, dopamine, actor | only CS-before-US pairings get credit |
| sequence learning and anticipation | hippocampus sequences, replay | asymmetric plasticity on recurrent place-cell synapses: each place cell excites the next one along a travelled path (Mehta et al. 2000) |
| predictive learning in cortex | chapter 3 | the plainer alternative to the delta rule, worth comparing |

---

## 3. Overview

| chapter | stage | animals (approx.) | brain added | rule or mechanism | world that makes it pay | status |
|---|---|---|---|---|---|---|
| 1 | steering | ~560 Mya, first bilaterians | ganglion (exc + inh), sensor -> motor reflexes | none (evolved) | several berry types | built |
| 1 | valence | early bilaterians | appetitive and aversive value cells on a grasp programme | none | inherited poison types | built |
| 1 | habituation | all animals | adapting look inputs | M4 | a dud bush type with a new look every life | built |
| 1 | association | early bilaterians | pain as the teacher of look -> aversive synapses | US-gated learning, M1, M2 | novel foods with a new look every life | built |
| 1 | drives | early bilaterians | `cold`, `hungry`, warmth by kinesis | need cells select modes | cold, few hot springs | built |
| 1 | affect | early bilaterians | lasting states after pain (`alarm`) and a meal (`serotonin`) that add up over bites | modulators with receptors | berry types in patches 70% their own kind | built |
| 1 | pallium | bilaterians (insect mushroom body), vertebrates (pallium) | sparse expansion of the look, taught by pain | fixed random projection per genome, M5 | combinations of features (XOR looks) | built |
| 2 | dopamine | early vertebrates | critic: opponent value cells, dopamine cell, habenula | TD error as teacher | poison and good kinds that swap within a life, small meals | formula built; cells planned |
| 2 | hands | early vertebrates | pick, hold, eat | none | food can be held | planned |
| 2 | basal ganglia | early vertebrates | striatum D1/D2, GPi, actor | dopamine-gated, signed by pathway | which action works is drawn per life | planned |
| 2 | curiosity | early vertebrates | novelty signal onto dopamine | familiarity memory | rewards hidden behind objects | planned |
| 2 | hippocampus | early vertebrates | a map of places: place and heading codes (medial pallium) | path integration from efference copy (M7), place -> value under `da` | food at places fixed per life, out of sight | planned |
| 2 | parked | vertebrates | cerebellum, noradrenaline and acetylcholine, pallium clustering | | | after the five above |
| 3 | | ~200 Mya, early mammals | neocortex as a generative model; simulation; sequences and replay on the place map; goals; episodes | M6, M8 | latent learning, detours, devaluation | planned |
| 4 | | ~60-30 Mya, primates | model of own mind, of others; imitation; future needs | | groups, crafting chains, seasons | planned |
| 5 | | hominins | signals, learned meaning, shared attention, culture | | signalling games, deep crafting trees | far off |

---

## 4. The stages

Each planned stage has the same fields. **Adds** = new regions and projections. **Rule** = learning rule or
mechanism. **World** = what makes it pay; what is drawn per life. **Control**, **Metric**, **Expected**.
**Confidence** = how sure the evolutionary claim is.

### Chapter 0: before bilaterians (context only)

Cnidarians (about 650-600 Mya): a diffuse nerve net with reflexes and pacemaker neurons, no brain. Not a stage;
there is evidence of simple associative learning even here (Botton-Amiot et al. 2023), so the boundary with
chapter 1 is blurry.

### Chapter 1: steering (first bilaterians, about 600-550 Mya)

A bilateral body that moves forward. Steering is one decision, over and over: turn toward good things, away from
bad ones. Model animals: *C. elegans*, flatworms, the presumed worm-like common ancestor. Order as built: steering
-> valence -> habituation -> association -> drives -> affect -> pallium. Results per stage: STATUS and the book.

The world of chapter 1: OHOL berry bushes and colour look-alikes on a 128 x 128 map with 256 animals; five
coarse eye columns (range 5); lives of 4000 ticks; fitness is the well-fed lifetime from the first meal (ADR-024).
The affect stage moves to 256 x 256; the pallium stage keeps that map and mixes the patches.

#### Steering
- **Adds.** `ganglion_e` (16 excitatory) and `ganglion_i` (8 inhibitory) between the senses and the motor
  neurons, plus direct sensor -> motor reflexes. All evolved. The animal sees only the outside world; movement
  costs energy; it eats what it grasps.
- **Confidence.** High for a centralised sensorimotor path in early bilaterians.

#### Valence
- **Biology.** Bilaterian sensory neurons fall mostly into appetitive and aversive classes that converge on a few
  interneurons (in *C. elegans* AIY/AIZ-type cells bias runs against turns; aversive cues stop pharyngeal
  pumping). Bennett: sorting the world into good and bad is the original job of a brain.
- **Adds.** Value cells `valence_app` and `valence_av` (taste and pain enter the brain only here) acting on a
  contact-gated `grasp` programme; aversion turns away and, through `no_feed`, blocks the bite. Input from object
  identity (the look) is kept separate from generic features, because only identity may later be learned about.
- **World.** Six berry types, two always poison (inheritable).
- **Confidence.** Medium: the valence framing is Bennett's synthesis; the ancestor's anatomy is inferred.

#### Habituation
- **Biology.** A weaker response to a repeated harmless stimulus is found in essentially every animal.
- **Adds.** No new cells: every look input passes on its input minus its slow average (about 50 ticks), so what
  stays in view fades and what is unusual stands out (M4).
- **World.** The valence world plus a dud bush type that never bears fruit, with a new look every life.
- **Confidence.** High.

#### Association (classical conditioning)
- **Biology.** Worms associate an odour with starvation or food, *Aplysia* a touch with a shock, flies an odour
  with sugar or shock. The synapse from the CS cell is strengthened when it was recently active *and* the US
  modulator arrives; in insects at the mushroom-body output synapses, with dopamine cells carrying the US.
- **Adds.** `us_pain` (hard-wired from the pain sense) as the teacher of the adapted look -> aversive synapses,
  with a short eligibility trace: `dW = eta · us_pain · trace(pre)`. The learning rate evolves.
- **World.** A world of only novel foods, half of them poison, with a new look every life.
- **Learned from pain only** (ADR-024). A taste teacher on the look -> appetitive synapses went unused in six
  seeds, and learned appetite spread to look-alike poison: learning what is good waits for dopamine, which has a
  predicted reward to compare with.
- **Confidence.** High that modulated associative learning exists across bilaterians; medium whether the common
  ancestor had it.

#### Drives
- **Biology.** Neurosecretory cells that sense internal state and broadcast it by neuropeptides are deeply
  conserved (the vertebrate hypothalamus and the insect pars intercerebralis probably share this ancestor).
  A drive changes what matters: food draws only when hungry.
- **Adds.** `cold` -> `warm_seek` -> forward (keep moving while the skin is cold); `rest` (stay where it is
  warm); `hungry` shuts both, so a hungry animal forages and a fed one looks after its warmth: kinesis, no
  gradient climbed. `hungry` also gets evolvable synapses onto feeding (not hard-wired).
- **World.** The association world made cold (OHOL-style temperature, cold multiplies hunger), with a few hot
  springs that do not block.
- **Confidence.** High that homeostatic neuropeptide control is ancient; medium that it looked like a
  hypothalamus in the first bilaterians.

#### Affect
- **Biology.** In *C. elegans*, serotonin released on food produces dwelling and lasts minutes after the food is
  gone (Flavell et al. 2013); pain and alarm leave a state too. Bennett's affect is such a global, lasting
  state that biases every reflex at once. Area-restricted search is found in worms, flies, fish, birds and
  mammals.
- **Adds.** Two states that outlast their trigger, each a modulator that adds up over events and fades slowly:
  pain -> `alarm` -> `flee` -> forward (above about 1.5 poison bites); taste -> `fed` -> `serotonin` -> `settle`
  -> `rest` (above about 9 good bites, a bush has 6). Fade times and thresholds evolve. `hungry` does the job of
  a roaming state.
- **World.** 256 x 256 (four times the area of the earlier worlds; at 128 crowding hides most of what a better
  forager gains), more food, berry types in patches that are 70% their own kind: mixed enough that a bush must
  be known by its look, pure enough that a meal or a poison bite says something about the neighbours.
- **Why this world.** In patches of one type learning has nothing to do (the first bite tells the animal about
  every bush around it); in open country or evenly mixed patches the lasting states have nothing to do. At 70%
  both have some work, but they answer the same question (which bushes here are poison), one by look and one by
  place, so a lineage leans on one of them. In this stage it leans on the lasting states; learning stays weak but
  its rate is not lost, and the pallium is what should make it pay again (STAGE_LOG v34, v40-v45).
- **Evolution settings this stage introduced** (standard from here on): 100 generations; learning rates mutate
  multiplicatively. With the additive step an idle learning rate is clipped to 0 and stays there.
- **The first affect stage** (`1.8old`: states released in full by one bite, one berry type per patch) and the two
  tuning steps after it (`2.0` camera eyes in open country, `2.1` the chapter 2 world) are frozen as a side path.
- **Confidence.** High for the phenomenon; medium for the exact neuromodulator mapping in the ancestor.

#### Pallium: sparse expansion of the look
- **Biology.** The oldest pallium is largely olfactory: piriform-like cortex with sparse, random, expansive
  input. Insects have the same design in the mushroom body (Kenyon cells, held sparse by one inhibitory cell),
  with dopamine-taught synapses at its output; annelid mushroom bodies and the vertebrate pallium may share an
  origin (Tomer et al. 2010). Random sparse expansion separates inputs that overlap (Babadi & Sompolinsky 2014;
  Litwin-Kumar et al. 2017).
- **Why in chapter 1** (ADR-027). The networks that prepare the input come before the circuits that learn from
  them, so that learning works on separated codes and not on the dense look. The pallium is still taught by pain
  alone, as in the association stage, so it closes chapter 1 and chapter 2 starts with the prediction error. The
  name stays `pallium` (the region of every later stage); the stage key stays `2.2`.
- **The job it got** (STAGE_LOG v46-v48). Habituation already centres the look, and a centred look separates
  random look-alikes as well as a random expansion does (an expansion's code cannot go below zero). What no weighted
  sum of the look can do is a combination: in the stage's world the four novel types are the corners of two colour
  directions drawn per life (poison ++ and --, good +- and -+, `berry_world(novel_xor=True)`), so a lesson on one
  poison by look clears the other. The pallium separates them.
- **Adds.** Region `pallium`, 128 cells, k-WTA 2, reading only the eye ahead (`vis+0.app*`, all 8 features; at a
  bite the side eyes see the neighbouring bushes, and even a few side cells undo the gain), fixed random weights
  of mixed sign drawn per genome (inherited, never mutated; not yet per birth as ADR-017 planned). The pain rule of
  the association stage also teaches `pallium -> valence_av` (starting rate 0.2). From this stage on the adaptation
  strength of the look is a gene (`Genome.adapt`), in main and control.
- **World.** The affect world with XOR looks at novel similarity 0.5 and patches mixed evenly (purity 0.25). At
  similarity 0.6 no learning paid; at purity 0.7 the place said which bushes were poison, the alarm carried the
  animal off, and a lesson by look saved only the first bite at the next bush of a kind (pallium used in one seed
  of three). Paired on an evolved population, all learning off costs 24% at purity 0.25, 6% at 0.7.
- **What evolution does with it.** Where lessons cost good food it turns them down through the rule genes (A
  drifted to -0.85, which with the pain driving the aversive cells nulls every lesson; or C negative, which erases
  earlier lessons), and it damps aversion whenever a bush is ahead (`vis*.hit/near -> valence_av`). The rule genes
  stay evolvable (user, 2026-10-18): the remedy is a world and a code in which the lessons pay.
- **Probes.** `pallium_sep.py` (brain-free separation), `pallium_real.py` (real lives, eye columns, adaptation
  strengths, magnified pallia), `learn_worth.py` (napkin: what transfer still pays), `assay_xor.py` and
  `life.labtests.xor_meals` (the lab test of the book), `lesson_curve.py` and `lesson_code.py` (acquisition and
  the pallium code in real lives), `gen0.py OFF=` (paired, learning of chosen projections off).
- **Control.** The affect brain, with the adaptation gene.
- **Metric.** Lesions of the pallium and of learning; the lab test (bites on the poison and the good bushes after
  meals, pallium and look learning on and off).
- **Result** (v51, after the remap fix). Pallium lesion 90%, all learning off 77%; main ahead of control in every
  seed (training +106 +-20, head to head +229 +-31). In the lab the pallium alone cuts bites at the poison bushes
  from 0.38 to 0.16 and leaves the good ones (0.37); the look alone lowers both alike. A larger pallium (256, 512
  cells) does not lower the transfer to good food further (0.13 at 128).
- **Confidence.** High that random sparse expansion with learning at its output is old (insect mushroom body,
  vertebrate piriform cortex) and that the vertebrate pallium began as sensory association cortex; that the
  first bilaterians had such a region, and that it served vision, are inferences of the model.

#### Left out of chapter 1
- **Extinction and reversal**: learned weights relaxing back toward the inherited ones, in a world whose
  meanings swap mid-life. Learning was unused there and the relaxing weights lost head to head in every seed.
  Reversal is left to error-driven learning (dopamine).
- **Sensory-specific satiety** (OHOL's "yum" bonus) and contact hazards: no stage; habituation got its job from
  the dud bush.

### Chapter 2: reinforcing (early vertebrates, about 520-450 Mya)

Ecological driver: Cambrian predation and active hunting in open water. The vertebrate brain plan is already
complete in lampreys: forebrain (pallium, basal ganglia, hypothalamus), midbrain (tectum, dopamine cells),
hindbrain (pattern generators, a small cerebellum, clear only in jawed vertebrates). Grillner's lamprey work shows
the basal ganglia wired as in mammals, the main reason for putting reinforcement learning here.

Order: **dopamine -> hands -> basal ganglia -> curiosity -> hippocampus.** The pallium, planned first here, closes
chapter 1 (ADR-027). Parked until these five work: cerebellum, uncertainty modulators, pallium clustering. The
world is the one of the critic stage (2.3h, small meals) until a stage needs more; the eyes stay those of chapter 1.

How the learning stages are checked: **lab experiments** take the place of the assay. One animal in scripted
situations, as in an animal lab (acquisition curves, omission and extinction, blocking, discrimination of
look-alikes, later devaluation), grown out of `scripts/probes/assay.py`, with results stored in the book's data
files and shown next to the world results.

Region names from here on: `pallium`, `striatum`, `gpi`, `snc` (dopamine), `habenula`, later `hippocampus`,
`cerebellum`.

#### Dopamine: the critic
- **Biology.** Midbrain dopamine cells report `δ = r + γV(s') − V(s)` (Schultz, Dayan & Montague 1997): the
  response moves from the reward to the earliest cue that predicts it, and a fully predicted reward no longer
  produces one. This explains blocking and second-order conditioning, which US-gated learning does not. With it
  the animal can learn what is good, which chapter 1 could not.
- **Built complete.** Both teachers (taste and pain), the prediction error, opponent value cells, as in the
  vertebrate circuit. Parts are removed only when experiments on the finished circuit show them unused.
- **The error is computed by cells, not written as a formula:**
  - values are an opponent pair (rates cannot be negative): `value_app` and `value_av` in the ventral striatum,
    learned from the pallium by TD(lambda) (`dW = eta · da · trace`);
  - a dopamine cell `snc` with a **tonic rate**; the modulator is its rate minus that baseline
    (`ModulatorSpec.baseline`): a burst is a positive error, a pause a negative one, capped at the tonic rate as
    in real dopamine cells (Bayer & Glimcher 2005);
  - **the subtraction is inhibition, the delay a slow pathway.** Value reaches the dopamine cell fast and
    exciting (disinhibition through the pallidum) and slow and inhibiting (striosomes directly onto the cell);
    fast minus slow is the change of value, the `γV(t) − V(t−1)` part (Houk, Adams & Barto 1995; Kim et al.
    2020). The slow path is a leaky inhibitory cell (`alpha` < 1); with `alpha` = 1 it is the exact one-tick
    TD, one end of the range evolution tunes over;
  - **bad news through the habenula**: expected-bad value and pain excite `habenula`, which inhibits the
    dopamine cell (described in the lamprey: Stephenson-Jones et al. 2013). Taste excites the dopamine cell
    directly;
  - **dopamine is short-lived** (taken up within about a second): the modulator does not persist; the memory of
    what was active is the eligibility trace in the synapse (a few seconds: Yagishita et al. 2014);
  - the value cells feed valence (incentive salience), and every `us`-gated projection switches to `da`.
- **Rule.** TD(lambda) as a three-factor rule with eligibility traces. No separate Rescorla-Wagner stage: in
  continuous time a prediction active before the outcome unlearns itself, and TD is the continuous-time form.
- **World: cues that predict food.** A cue some ticks before food, its look drawn per life, taken from OHOL
  objects and transitions where possible.
- **Lab experiments.** Acquisition, the response moving from reward to cue, a pause at an omitted reward,
  blocking, discrimination of look-alikes, reversal.
- **Control.** The pallium brain with US-gated learning.
- **Expected.** The TD animal approaches the first cue of a chain and learns what is good; the US-gated animal
  learns only cues close to the outcome.
- **Confidence.** High for mammals, good evidence for fish; the RPE in lampreys is plausible but less directly
  shown.

#### Hands
- **Adds.** Pick up, hold, eat (`stages.with_hands`: a value cell per sign for the thing in hand and an `ingest`
  programme). With eat-on-grasp there is no action to choose; the actor needs one.
- **How.** A short adaptation step from the dopamine population into a pick-hold-eat world, no controlled
  experiment (in the trial `x.hands` the lineage adapted in about ten generations), with a check that it has not
  lost much against eat-on-grasp.

#### Basal ganglia with the actor
- **Biology.** Pattern generators produce whole programmes (swim, orient, bite, escape); the GPi/SNr inhibit all
  of them tonically, and the striatum disinhibits one at a time (Redgrave, Prescott & Gurney 1999). The same
  `δ` trains the pallium -> striatum synapses: on D1 (Go) cells a positive `δ` strengthens, on D2 (NoGo) cells a
  negative one (Frank 2005; Collins & Frank 2014). This is operant learning: which *action* to take, not only
  the value of a thing. One stage: the fixed selection circuit alone has little to gain, since `hungry` already
  settles the conflict between drives.
- **Adds.** `striatum_d1`, `striatum_d2`, `gpi` (tonic), `gpi -> programmes` inhibitory; pallium -> striatum
  plastic under `da`, signed by pathway. Striatum cells have a fixed meaning and act on programmes.
- **Rule.** `dW = eta · (+δ) · e` on D1, `dW = eta · (−δ) · e` on D2.
- **World: an action -> outcome contingency drawn per life.** For example: one food source yields only when used
  while holding a particular object, which object works is drawn per life (from OHOL transitions).
- **Control.** The hands brain (critic only).
- **Metric.** Yields per life, second half against first half; lab experiments as in a lever-pressing rat.
- **Confidence.** High for mammals; the D1/D2 split is conserved at least to lampreys.

#### Curiosity
- **Biology.** Fish and all later vertebrates explore new objects, and Bennett puts curiosity in this
  breakthrough. Dopamine cells respond to novel stimuli, and the response fades as the stimulus becomes familiar
  (Ljungberg, Apicella & Schultz 1992; Horvitz 2000). It is the classic fix for exploration in RL.
- **What is new and what is not.** This is curiosity about what is *unfamiliar*, which needs only a memory of
  what has been seen. Curiosity about what cannot yet be *predicted* (surprise, learning progress) needs a model
  of the senses and belongs to chapter 3.
- **Where novelty comes from** (decided when the stage is designed):
  - lasting familiarity on the pallium: each thing has its own sparse code, and the synapses from that code onto
    a `novelty` cell weaken with every sighting and stay weak (repetition suppression; anti-Hebbian familiarity
    models of the perirhinal cortex, Bogacz & Brown 2003). Stimulus-specific and lifelong;
  - the adapted look of the habituation stage: already built, but it remembers only about 50 ticks ("unusual in
    the last minute", not "never seen") and spreads across look-alikes;
  - a fast route through the tectum, whose responses adapt to repeated stimuli and which drives short-latency
    dopamine responses to salient events (Dommett et al. 2005; Redgrave & Gurney 2006).
  `novelty -> snc` then adds a bonus to `δ`.
- **World: hidden rewards.** Food found by interacting with objects (using some non-food object sometimes yields
  food; which one is drawn per life), among many inert objects.
- **Control.** The basal ganglia brain.
- **Metric.** Share of object types tried; food per life.
- **Confidence.** Medium: novelty responses of dopamine cells are well shown in mammals; in fish the evidence
  for curiosity is mostly behavioural.

#### Hippocampus: a map of places
- **Biology.** The hippocampus counterpart of fish is part of the pallium (medial in most vertebrates, lateral in
  the everted pallium of teleosts). Goldfish with it lesioned lose place learning but keep cue learning
  (Rodríguez et al. 2002), and Bennett puts spatial maps in this breakthrough.
- **Why here, after the actor** (ADR-027). A place code gives the critic something new to value: places, not only
  looks. A value map is useless unless the creature can follow it, and the brain so far only steers toward what it
  sees; an actor that learns "turn here, at this place" can. Following a map by playing routes in the head is
  chapter 3.
- **Adds** (new mechanisms, agreed with the user when the stage is designed). A sense of the creature's own
  movement and heading (path integration from efference copy, M7), landmarks, place cells; place -> value and
  place -> actor synapses under `da`.
- **World.** Food at places fixed per life, out of sight (the eyes reach 5 cells on a 256 x 256 map): the harsh
  world's scarce bushes, the hot springs. First check, at generation 0: would a perfect memory of places pay at
  all, or does a better random search do as well?
- **Lab.** Place against cue, as in the goldfish experiments (does the creature return to where the food was or
  to what marked it?), and a water-maze-like test: a hidden place found from different starts.
- **Control.** The curiosity brain.
- **Confidence.** High for a spatial role of the fish pallium; medium for place cells as in mammals.
- **Left for chapter 3.** Sequences (asymmetric plasticity on recurrent place-cell synapses, a route fixed per
  life), the substrate of replay and of simulating paths.

#### Parked until the five stages above work
- **Cerebellum.** Granule expansion, Purkinje cells learning from a climbing-fibre error (delta rule, M6).
  World: a warning object that stings after a delay drawn per life. A grid world gives it little to do; its role
  here is to bring in the delta rule chapter 3 depends on.
- **Noradrenaline and acetylcholine.** A cell reading `|δ|` over time scales the learning rates and the action
  noise. World: lives whose meanings are stable or switch often, drawn per life.
- **Pallium clustering.** Competitive Hebb / Oja and recurrent pattern completion on the pallium, when a world
  with noisy, occluded looks needs categories formed within a life.
- **Delayed sickness**, with blame that lands on what was eaten (section 2).

#### Not a stage
- **Tectum.** Its innate half answers to moving and looming things, which this world lacks; its other half,
  value choosing a place, comes with the basal ganglia; the value cells already have their own synapse from
  every eye column. A target map with one winner was neutral to harmful at generation 0 (STAGE_LOG v35, v36).
- **Camera eyes.** Nine columns instead of five cost compute and change little in a world this simple
  (STAGE_LOG v35-v37). Revisit when a stage needs finer vision.

### Chapter 3: simulating (early mammals, about 220-150 Mya)

Ecological driver (Bennett): small nocturnal mammals among dinosaurs gained by planning moves before making them.
The new structure is the six-layered **neocortex** (sensory, motor, agranular prefrontal) in a loop with the
thalamus and basal ganglia. Birds reached similar abilities with a differently organised pallium, so
"simulation" is a function, not uniquely a layered cortex. Planned stages: a cortex that predicts, trying actions
in the head, replay, goals held over time, single events. The tests are the classic ones from the rat lab:
latent learning, detours, outcome devaluation.

#### Neocortex as a generative model
- **Biology.** Cortex learns to predict its own input; the error between prediction and input drives learning
  (Rao & Ballard 1999; Friston; Hawkins).
- **Adds.** `cortex_l4` (input), `cortex_l23` (state, with efference copy so it predicts the consequences of
  actions), `cortex_pred` (predicted next input), `error` (input − prediction). `cortex_l23` joins the pallium
  as input to the striatum and the value cells.
- **Rule.** Delta rule (M6) with `error` as teacher; Oja on `l4 -> l23`.
- **World.** Partial observability with predictable structure: food that regrows on a schedule at positions
  fixed per life, objects often occluded.
- **Also here.** Curiosity about what cannot be predicted: the prediction error, or how fast it falls, as a
  bonus to dopamine.
- **Confidence.** Medium; predictive processing is a strong framework, not established fact.

#### Vicarious trial and error
- **Biology.** Rats pause at choice points while the hippocampus sweeps ahead along each option (Johnson & Redish
  2007). Latent learning, detours and outcome devaluation show behaviour that model-free RL cannot produce.
- **Adds.** Offline mode (M8): a `deliberate` cell (driven by conflict or uncertainty in the striatum) switches
  the thalamic gate; for some internal steps motor output is suppressed, the cortex's prediction replaces real
  input, a candidate action is fed as efference copy, and the value cells evaluate the imagined state.
- **World.** Latent learning (a maze-like area without reward, then food at one place), devaluation (sated on one
  food: does the animal stop working for it at once?), detour (a known path blocked mid-life).
- **Control.** The same world model used only as a representation.
- **Confidence.** High for the behaviour; the circuit is still debated.

#### Replay and sleep
- **Biology.** At rest the hippocampus replays sequences forward and backward (Foster & Wilson 2006); replay
  drives consolidation and value updates. Dyna (Sutton 1990) is the RL counterpart.
- **Adds.** First **sequences** on the place map of chapter 2: asymmetric plasticity on recurrent place-cell
  synapses, so each place cell excites the next one along a travelled path (Mehta et al. 2000). Then a rest state
  triggers offline mode with no sensory input; the normal `da`-gated rules learn from what is generated. World: little real experience (short lives or few rewards), safe resting.
- **Confidence.** High for replay; medium for its necessity.

#### Agranular prefrontal cortex: goals over time
- **Biology.** Medial PFC holds rules and goals and gates which options get simulated; PFC-BG-thalamus loops
  update working memory in a gated way (O'Reilly & Frank 2006).
- **Adds.** `pfc` with recurrent self-excitation; a basal ganglia loop that learns when to update it, under `da`.
- **World.** Multi-step goals with distractors (fetch a tool shown at the start of a life, carry it past food to
  a target).
- **Confidence.** Medium.

#### Episodic memory and counterfactuals
- **Biology.** One-shot storage of what, where and when in the hippocampus; regret-like signals after a missed
  better option (Steiner & Redish 2014).
- **Adds.** Fast one-shot storage on a subset of hippocampal synapses; a counterfactual `δ` from a simulated
  alternative. World: a rich cache that appears once per life; near misses.
- **Confidence.** Medium.

### Chapter 4: mentalizing (primates, about 65-5 Mya)

Ecological driver (Bennett, after Dunbar and Humphrey): large groups where status and alliances decide access to
food. The new structure is **granular prefrontal cortex**, present in primates only. The world needs other
animals that matter, and the life cycle has to become continuous (STATUS, future goals).

- **A model of one's own mind.** `gpfc` predicts the animal's own next action and its chance of success (delta
  rule); an opt-out programme. World: noisy looks, costly poison, a safe small alternative.
  Confidence medium.
- **Theory of mind.** Observation extensions (M9: others' facing, held object, last action); a `tom` region, a
  copy of the self-model fed with what others do. World: following an informed animal; food competition with a
  dominant animal that takes what it sees. Confidence medium.
- **Imitation.** The inferred action of a successful animal as a teacher of one's own policy. World: a crafting
  chain too long to find alone (a larger OHOL slice), which chain works drawn per life. Confidence medium-high
  for great apes.
- **Anticipating future needs.** In offline mode, a simulated future drive state replaces the current one. World:
  seasons with a winter in which nothing regrows; storing food beforehand. Confidence low-medium.

### Chapter 5: speaking (hominins, about 2-0.1 Mya)

Ecological driver: cooperation in hunting and child care, and knowledge that accumulates across generations.
This chapter is the least certain, both biologically and in whether a grid world can carry it.

- **Innate signals** (possible from chapter 2 on, as a side branch): a sound vector, affect -> VOCALIZE and
  sound -> affect; moving predators and related animals so that calling pays. Confidence high.
- **Learned signal meaning**: plastic sound -> pallium and value under `da`; a signalling game where conventions
  form within a life. Confidence medium.
- **Shared attention and naming**: association between a heard sound and an object boosted when two animals look
  at the same thing. Confidence low-medium.
- **Language as shared simulation and culture** (Bennett): one brain drives another's simulator. Test: a crafting
  tree too deep for one life; does the depth reached keep rising across generations with communication on?
  Speculative as a mechanism, but a clean target.

---

## 5. Region -> function -> stage

| structure | real function (simplified) | stage | framework form |
|---|---|---|---|
| brainstem pattern generators | complete motor programmes (turn, bite, run); reflexes | steering, valence | evolved fixed weights; programmes with a fixed meaning |
| hypothalamus | internal state, drives; which goal matters | drives | need cells selecting modes |
| neuromodulatory nuclei (raphe 5-HT, VTA/SNc DA, LC NE, basal forebrain ACh) | lasting states, reward prediction, uncertainty | affect (`alarm`, `serotonin`), dopamine, parked (NE, ACh) | modulator cells with receptors (M2, M3) |
| valence / amygdala | good and bad; fear and appetitive conditioning | valence, association | value cells; US-gated plastic synapses |
| pallium (olfactory cortex; insect mushroom body) | pattern separation, later completion and categories | pallium (end of chapter 1); parked (clustering) | expansion + k-WTA; later Oja / recurrent Hebb |
| basal ganglia (striatum, GPi/SNr, habenula, SNc) | action selection by disinhibition; critic and actor | dopamine, basal ganglia | value cells, dopamine cell, D1/D2 under `da` |
| hippocampus (medial pallium) | space, sequences, episodes, replay | hippocampus (end of chapter 2); sequences, replay, episodes in chapter 3 | place codes, recurrent plasticity, one-shot storage |
| cerebellum | timing, forward models | parked | expansion + delta rule |
| optic tectum | orienting, looming escape | not a stage (see chapter 2) | |
| thalamus | relay and gating; perception against imagination | chapter 3 | the gate of offline mode (M8) |
| neocortex | generative model of the senses | chapter 3 | predictive layers, delta rule |
| agranular / granular PFC | goals; models of own and others' minds | chapters 3, 4 | attractor with gated update; self-model |

---

## 6. Convergent side branches (optional)

These evolved independently and make good "does the same function need the same structure?" experiments.

- **Insect mushroom body:** expansion (Kenyon cells) plus dopamine-gated output synapses, the association stage
  and the pallium without a vertebrate pallium; the best-understood modulated learning circuit (full
  *Drosophila* connectomes exist), a realistic template.
- **Insect central complex:** head-direction ring and path integration (Seelig & Jayaraman 2015), the cleanest
  model for the hippocampus stage.
- **Octopus:** complex learning with a very different, distributed brain.
- **Birds:** mammal-level planning with a non-layered pallium; corvids cache food.

---

## 7. Main sources

- Bennett, M. (2023). *A Brief History of Intelligence.*
- Cisek, P. (2019). Resynthesizing behavior through phylogenetic refinement. *Attention, Perception &
  Psychophysics.*
- Grillner, S. & Robertson, B. (2016). The basal ganglia over 500 million years. *Current Biology.*
- Striedter, G. & Northcutt, R. G. (2020). *Brains Through Time.*
- Schultz, Dayan & Montague (1997). A neural substrate of prediction and reward. *Science.*
- Ljungberg, Apicella & Schultz (1992). Responses of monkey dopamine neurons during learning of behavioral
  reactions. *J. Neurophysiol.*
- Bogacz & Brown (2003). Comparison of computational models of familiarity discrimination in the perirhinal
  cortex. *Hippocampus.*
- Redgrave & Gurney (2006). The short-latency dopamine signal: a role in discovering novel actions? *Nat. Rev.
  Neurosci.*
- Frank, M. (2005). Dynamic dopamine modulation in the basal ganglia. *J. Cogn. Neurosci.*
- Redgrave, Prescott & Gurney (1999). The basal ganglia: a vertebrate solution to the selection problem?
- Flavell et al. (2013). Serotonin and PDF neuropeptide control of roaming and dwelling in *C. elegans*. *Cell.*
- Aso et al. (2014). The neuronal architecture of the mushroom body. *eLife.*
- Litwin-Kumar et al. (2017). Optimal degrees of synaptic connectivity. *Neuron.*
- Rodríguez et al. (2002). Conservation of spatial memory function in the pallial forebrain of reptiles and
  ray-finned fishes. *J. Neurosci.*
- Rao & Ballard (1999). Predictive coding in the visual cortex. *Nature Neuroscience.*
- Johnson & Redish (2007). Neural ensembles in CA3 transiently encode paths forward of the animal.
- O'Reilly & Frank (2006). Making working memory work (PBWM). *Neural Computation.*
- Passingham & Wise (2012). *The Neurobiology of the Prefrontal Cortex.*
- Tomasello, M. (2008). *Origins of Human Communication.*
