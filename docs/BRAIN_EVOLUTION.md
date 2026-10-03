# Brain evolution sequence: architectures, learning rules and experiments

A plan for growing the brain one module at a time, in roughly the order brains evolved. Status and findings
per stage: `docs/STATUS.md`; details: `docs/STAGE_LOG.md`. The implemented stages sometimes deviate from this
plan (for example fixed-meaning cell types, ADR-016); STATUS/STAGE_LOG describe what was actually built. Framework terms (regions, projections, `rule`, `modulated`, warm start) are those of
ADR-007, ADR-012 and ADR-013.

The spine is Max Bennett's *A Brief History of Intelligence* (2023): **steering → reinforcing → simulating →
mentalizing → speaking**. Around it are finer steps taken from comparative neuroscience. The literature does
not settle many of these points, so every stage says how confident the claim is. "Mya" means million years
ago; all dates are approximate.

---

## 0. Why exp02 got stuck, and what that says about the order

One correction to the summary of the book first. Bennett puts **associative learning inside breakthrough 1
(steering)**, in the first bilaterians (nematode-like worms). What comes with vertebrates (breakthrough 2) is
*reinforcement learning with prediction errors* **and** *pattern recognition* (the pallium, which later
becomes cortex). So the book's order is:

1. Steering: valence, affect, and associative learning (classical conditioning) of stimuli onto good/bad.
2. Reinforcing: basal ganglia with temporal-difference dopamine, curiosity, pattern recognition in the
   pallium, spatial maps, timing.
3. Simulating: neocortex as a generative model, vicarious trial and error, episodic memory, counterfactuals.
4. Mentalizing: granular prefrontal cortex, models of one's own mind and of others' minds, imitation,
   planning for future needs.
5. Speaking: language, shared attention, culture that accumulates.

Why this matters for exp02: **the oldest associative learning is not pure Hebbian learning either.** In
*Aplysia*, *C. elegans* and *Drosophila*, a conditioned stimulus (CS) becomes linked to an unconditioned
stimulus (US) through a **third factor**, a neuromodulator released by the US: serotonin in *Aplysia*,
dopamine and octopamine in insects. The synapse also has to "remember" that it was recently active, because
the US comes after the CS. So the smallest real step up from evolved reflexes is:

> dW = eta · US(t) · eligibility(pre, post), where eligibility is a decaying trace of recent co-activity
> and US is a *raw* valence signal (not yet a prediction error).

exp02's rule was `dW = eta · (A·pre·post + ...)`. When the poison's pain arrived, the berry's appearance had
already left the input (it had been eaten). Nothing in the rule linked "pain now" to "that appearance a tick
ago", and unmodulated Hebb strengthens every co-activation regardless of consequence. The fix is not
reinforcement learning. It is the ingredients of stage 1.5 below: a valence neuron that acts as a
modulator, an eligibility trace, and pain short enough that a purely reactive policy cannot do the job.

In short, the step from "Hebbian" to "RL" is really four smaller steps:
(a) unmodulated Hebb (habituation, clustering),
(b) US-gated Hebb with eligibility traces (classical conditioning),
(c) the modulator becomes a *prediction error* (Rescorla-Wagner, then TD),
(d) the prediction error trains *action selection* (actor-critic basal ganglia).

---

## 1. Design principles for the sequence

1. **One module or one rule per stage.** A stage adds regions and projections, or one new learning rule or
   core mechanism, never both at once if that can be avoided.
2. **Every stage has an ecological driver.** In biology, each innovation paid off because of a change in the
   environment (Cambrian predation, vertebrate open-water hunting, nocturnal mammal life, arboreal primate
   social groups). In the framework that driver is an *environment feature* that makes the new module pay,
   and the experiment should be built so that the previous architecture measurably fails at it.
3. **Controls are warm-started from the same population.** Run A continues evolving the old architecture
   for the same number of generations. Run B adds the new module. Both start from the previous stage's
   `population.npz` (ADR-012). The comparison is B against A, not B against generation 0.
4. **Growth by duplication, starting silent.** Most new brain regions evolved by duplicating and then
   specialising existing structures (gene and genome duplications in early vertebrates are the classic
   case). In practice, a new region either (i) starts with zero outgoing weights, so inherited behaviour is
   unchanged until evolution connects it, or (ii) starts as a noisy copy of an existing region. Either way
   the warm start does not fall off a cliff.
5. **Randomise per life whatever must be learned** (ADR-012). Keep whatever was learned at the previous
   stage *profitable*, so that the new skill is an addition on top of it.
6. **Keep a reactive or old-architecture baseline in every experiment.** exp02 showed that the environment
   often provides a shortcut. If the control solves the task, the task is not yet the right one.

---

## 2. Core mechanisms this sequence will need

This collects everything the stages below ask of the core. **Status 2026-09-30:** M1-M7 and M9 are implemented
(ADR-015/016; M3 as receptors for broadcast modulators plus `gain` projections); M10 partly (mid-life rule switch,
temperature, patches, variety bonus, delayed sickness; moving objects not yet); M8 (offline mode) is still to do.

| id | mechanism | used from stage | notes |
|---|---|---|---|
| M1 | **Eligibility traces for modulated rules**: per synapse `e ← λ e + pre·post`, `dW = eta·mod·e` | 1.5 | `trace_tau` already exists per neuron; a per-synapse trace is `[N,N]` more state, the same size as `w`. |
| M2 | **Internal, named modulators** (**ADR**, extends ADR-013): a vector `mod[k]` for k in {`da`, `5ht`, `ne`, `ach`, `us`}, each written by a designated region (for example the mean activity of neurons in region `vta`). Each projection chooses *which* modulator gates its plasticity. | 1.5 (`us`), 2.5 (`da`) | Replaces today's world-provided scalar `reward`. Keep `reward` as a special case of the `us` modulator, computed by a hard-wired taste region. |
| M3 | **Gain modulation**: a modulator multiplies the input gain of a target region (`h_j ← g(mod)·h_j`) instead of adding to it. | 1.2 | This is how hypothalamic and neuromodulatory state switches behaviour modes. Additive-only networks can approximate it, but evolution finds that much harder. |
| M4 | **Short-term synaptic depression** (a resource variable per synapse that is used up by pre-synaptic activity and recovers with a time constant) | 1.4 | Habituation. Rate-based Tsodyks-Markram model, so it satisfies ADR-002. |
| M5 | **Lateral inhibition / k-winners-take-all** inside a region | 2.1 | Can be done today with fixed negative intra-region projections; a `kwta` flag on the region would be cheaper and more stable. |
| M6 | **Error-driven (delta) rule** with a teacher region: `dW = eta·pre·(teacher − post)` | 2.9 (cerebellum), 3.1 (cortex) | The climbing fibre in the cerebellum; prediction error in cortex. |
| M7 | **Efference copy input**: the last action as a one-hot in the body channel | 2.8 | Widening the body channel is allowed by ADR-005. Needed for path integration and for forward models. |
| M8 | **Offline mode** (**ADR**): for some brain steps the motor output is gated off and sensory input is replaced by a predicted-input region (thalamic gate). Triggered by a region's activity (for example "deliberate" at choice points) or during rest/sleep. | 3.2 | This is "simulation" in the sense of the book. It costs extra `steps_per_tick`. |
| M9 | **Observation extensions** (ADR-005 widening): aftertaste (appearance of the last eaten object, decaying), appearance noise/occlusion, second body meter (e.g. water), other agents' facing/held item/last action, sound vector | per stage | A smell channel would be a *new* channel and needs an **ADR**; the stages below avoid it by using short-range vision as "smell". |
| M10 | **World mechanics**: moving objects (OHOL `move` transitions, already a known gap), contact hazards, seasons/regrowth schedules, mid-life rule changes | 1.4, 2.9, 3.x, 4.4 | Each behind a config switch that defaults off (ADR-010). |

### 2.1 Where STDP fits

Spike-timing-dependent plasticity (STDP) is not a stage of its own because it is not a lineage innovation.
It is a synaptic mechanism found wherever it has been looked for: *Xenopus* tectum (Zhang et al. 1998, one of
the first in vivo demonstrations), fish, insects (locust mushroom body, Cassenaer & Laurent 2007), and
mammalian cortex and hippocampus (Markram et al. 1997; Bi & Poo 1998). It was most likely present from the
first centralised nervous systems onward. What it adds is **causal order**: a synapse strengthens when pre
fires *before* post and weakens when it fires after.

In this framework (rate coded, ADR-002), STDP is the existing `trace` rule
(`A·pre_trace·post − C·pre·post_trace`). Two points about timescale:
- One tick is a behavioural-timescale step (about a second), not milliseconds. Our "STDP" therefore behaves
  less like classic millisecond STDP and more like **behavioural-timescale plasticity** (BTSP; Bittner et
  al. 2017, hippocampus, windows of seconds), which is arguably the better match anyway.
- Classic millisecond STDP can only be modelled with `steps_per_tick > 1`. That makes the brain slower, so
  keep it for specific tests.

STDP plays four distinct roles in the plan:

| role | stage | how |
|---|---|---|
| **Eligibility trace for three-factor learning** ("neo-Hebbian": STDP tags a synapse, and a later modulator converts the tag into a weight change; Izhikevich 2007, Gerstner et al. 2018) | 1.5, 2.5, 2.6 | M1 is built on the `trace` rule instead of `pre·post`, so only CS-*before*-US pairings get credit. Test: a variant of 1.5 where the look-alike berry is sometimes eaten *just after* the pain (backward pairing) must not be learned as poison. |
| **Map refinement / self-organisation** | 2.1 (tectum), 2.3 (pallium) | Unmodulated STDP plus lateral inhibition sharpens topographic maps and feature detectors during life. Optional "development" variant: the tectum starts coarse and self-organises in the first ticks of life instead of being fully genetically specified (genome encodes the rule, not the map; a smaller genome). |
| **Sequence learning and anticipation** | 2.8 (hippocampus), 3.3 (replay) | Asymmetric STDP on recurrent place-cell synapses makes each place cell excite the *next* one along a travelled path. Place fields shift backwards with experience (Mehta et al. 2000), activity then predicts where the animal goes next, and sequences can be replayed offline. This was the old exp03 idea ("STDP sequence learning") and belongs here as **stage 2.8b**: food appears at the end of a route that is fixed per life. The STDP agent anticipates (turns early and cuts corners); the symmetric-Hebb agent of 2.8 does not. |
| **Predictive learning in cortex** | 3.1 | Temporally asymmetric learning is one of the local mechanisms proposed for learning to predict the next input. The delta rule (M6) is the main tool; STDP is the biologically plainer alternative worth comparing. |

---

## 3. Overview table

| stage | era, animals (approx.) | brain structure added | new learning rule / mechanism | experiment where it beats the previous stage |
|---|---|---|---|---|
| 0.1 | ~650–600 Mya, cnidarians | diffuse nerve net: sensor→motor reflex | none (evolved weights) | baseline only |
| 1.0 | ~560 Mya, first bilaterians | centralised sensorimotor path (ganglion, exc + inh) | none | several food types |
| 1.1 | early bilaterians | **valence cell types** with innate motor meaning | none | inheritable poison look-alikes |
| 1.2 | early bilaterians | **hypothalamic drives** (hunger, cold) + innate thermotaxis | broadcast modulator + receptors | cold world with hot springs |
| 1.3 | early bilaterians | **affective states**: serotonin dwell / PDF roam | slow nuclei + receptors | patchy food: area-restricted search |
| 1.4 | cnidarians onward | habituation (sensory-specific satiety) | short-term depression (M4) | OHOL yum variety bonus |
| 1.5 | early bilaterians | plastic sensory→valence synapses (proto-amygdala / mushroom-body output) | **US-gated Hebb + eligibility trace** (M1, M2 `us`) | exp02 redesigned: poison identity per life |
| 1.6 | bilaterians | extinction and reversal | decay of learned weights toward inherited ones | poison identity flips mid-life |
| 2.1 | ~520 Mya, early vertebrates | **optic tectum**: topographic map, target competition | lateral inhibition (M5) | several targets in view |
| 2.2 | early vertebrates | **pallium as expansion layer** (sparse random coding) | fixed sparse projection + k-WTA | nonlinear (XOR-like) poison rule |
| 2.3 | early vertebrates | **pallium pattern completion and clustering** | competitive Hebb / Oja + recurrent Hebb | noisy, occluded appearance, few samples |
| 2.4 | early vertebrates | **basal ganglia** gating of brainstem motor programs | fixed disinhibition circuit | conflicting drives: less dithering |
| 2.5 | early vertebrates | **dopamine as reward prediction error** (critic, opponent value populations) | TD error as modulator; dopamine teaches valence | look-alike discrimination without over-generalising; chains |
| 2.6 | early vertebrates | **actor**: striatal Go/NoGo (D1/D2) learning of actions | RPE-gated three-factor on cortex→striatum, signed by pathway | per-life action→outcome contingency |
| 2.7 | early vertebrates | **curiosity**: novelty as intrinsic reward | familiarity from habituation (1.4) feeds dopamine | sparse, hidden rewards |
| 2.8 | early vertebrates | **medial pallium (hippocampus homologue)**: place and head-direction codes | efference copy (M7), Hebbian place-reward links | food location fixed per life but out of view |
| 2.8b | early vertebrates | recurrent place-cell sequences | asymmetric STDP (`trace` rule) on recurrent synapses | anticipating a route fixed per life |
| 2.9 | jawed vertebrates | **cerebellum**: timing and forward prediction | delta rule with climbing-fibre teacher (M6) | eyeblink-style timed avoidance |
| 2.10 | vertebrates | **uncertainty modulators** (NE, ACh) | modulator-scaled learning rate | stable vs volatile worlds |
| 3.1 | ~200 Mya, early mammals | **neocortex** as predictive, generative model | self-supervised prediction error (M6) | partial observability, predictable regrowth |
| 3.2 | early mammals | **vicarious trial and error**: simulation at choice points | offline mode (M8) | latent learning, detours, outcome devaluation |
| 3.3 | early mammals | **replay and sleep** (hippocampus→cortex→striatum) | offline replay trains model-free system (Dyna) | same experience, fewer samples |
| 3.4 | early mammals | **agranular prefrontal cortex**: goals held over time | persistent attractor + gated update | multi-step goals with distractors |
| 3.5 | mammals | **episodic memory and counterfactuals** | one-shot hippocampal storage, "what if" RPE | one-shot events, near misses |
| 4.1 | ~60–30 Mya, primates | **granular PFC: model of own mind** | confidence readout from own predictive model | opt-out under uncertainty |
| 4.2 | primates | **theory of mind**: self-model applied to others | inverse planning over observed agents | infer others' goals; follow or avoid |
| 4.3 | primates | **imitation / observational learning** | others' actions as teacher signal | crafting chain too hard to discover alone |
| 4.4 | primates, great apes | **anticipating future needs** | simulation under a drive not currently felt | seasons: store food before winter |
| 5.1 | vertebrates onward (side branch) | innate signals (alarm and food calls) | evolved sound→affect links | predator alarm, food call |
| 5.2 | primates, birds | learned signal meaning | RL over signals (signalling games) | per-life random signal conventions |
| 5.3 | hominins, <2 Mya | shared attention, naming, proto-grammar | joint-attention-gated association | instruction in cooperative crafting |
| 5.4 | Homo sapiens, <0.3 Mya | language as shared simulation and cumulative culture | teaching across generations | knowledge that no single life can rediscover |

---

## 4. The stages

Each stage has the same fields. **Adds** = new regions and projections. **Rule** = learning rule or
mechanism. **Experiment** = world setup; what is randomised per life; control; metric; expected outcome.
**Confidence** = how sure the evolutionary claim is.

### Chapter 0: before bilaterians (context, optional)

#### 0.1 Nerve net reflexes
- **When, who.** Cnidarians (jellyfish, sea anemones), about 650–600 Mya. Sponges have no neurons, though
  they do have many of the synaptic genes.
- **Biology.** A diffuse net with no brain. Reflexes such as withdrawal and feeding tentacle movements. Some
  pacemaker neurons drive rhythmic swimming, the ancestor of central pattern generators.
- **Framework.** `in→out` fixed projection only, no hidden region. Useful as the "how far do pure reflexes
  go" baseline for exp01.
- **Experiment.** None needed; exp01 with the hidden region removed is the baseline.
- **Confidence.** High for the anatomy. Recent work (for example Botton-Amiot et al. 2023 on *Nematostella*)
  suggests even cnidarians may show simple associative learning, so the boundary between chapters 0 and 1
  is blurry.

### Chapter 1: steering (first bilaterians, about 600–550 Mya)

Ecological driver: a bilateral body that moves forward. Steering means one decision, over and over: turn
toward good things, away from bad ones. The earliest brain is a cluster of neurons at the front that turns
sensory input into "approach" or "avoid". Model animals: *C. elegans*, flatworms, and the presumed
worm-like common ancestor. (*C. elegans* is highly derived, so it is only a proxy.)

Implementation note (2026-09-30): every new module below is a set of **cell types with a fixed meaning**
(hard-wired defining inputs and outputs; ADR-016). Without that, evolution simply routes behaviour through the
existing ganglion and the new module stays unused (stage 1.1 v1/v2 in `docs/STAGE_LOG.md`). Exact wiring:
`life/experiments/stages.py`; results: `docs/STATUS.md`.

#### 1.0 Centralised sensorimotor steering
- **Adds.** `ganglion_e` (16 excitatory) and `ganglion_i` (8 inhibitory) interneurons between the senses and the
  motor neurons, plus direct sensor->motor reflex arcs. All evolved.
- **Senses.** 5 coarse vision columns (range 5), body state, sound: early bilaterians had simple photoreceptors
  and chemosensing; camera eyes come with vertebrates (2.1).
- **World.** Four kinds of berry bush (the OHOL gooseberry and colour look-alikes, *similar* appearances) and
  wild onions. Several food types are needed: with one, the lineage becomes a specialist that ignores new food.

#### 1.1 Valence neurons
- **Biology.** Bilaterian sensory neurons mostly fall into appetitive and aversive classes that converge onto a
  few interneurons (in *C. elegans*, AIY/AIZ-type interneurons bias forward runs vs turns; aversive cues suppress
  pharyngeal pumping). Bennett argues that valence, classifying the world into good and bad, is the original
  function of a brain.
- **Adds (v18, ADR-020).** Value cells `valence_app` and `valence_av` (two cells each for what is seen, one each
  for what is in hand; taste and pain enter the brain only here) acting on four motor programmes, each permitted
  only in its context: `approach` (FORWARD; shut on contact), `grasp` (USE; something adjacent, empty hand),
  `ingest` (EAT; something in hand), `reject` (USE; something bad in hand). Aversion also turns the animal away
  and, through `no_feed`, blocks eating. Evolved input from object identity (appearance) and, separately, from
  generic features (something there, near, wall), because only identity may later become a conditioned stimulus.
- **World.** 6 berry types, 2 of them always poison (inheritable); more bushes so the good food stays constant.
- **Result.** Clearly better than the control; valence lesions destroy foraging (STATUS).
- **Confidence.** Medium. The valence/steering framing is Bennett's synthesis; the ancestor's anatomy is inferred.

#### 1.2 Drives: hypothalamus-like homeostatic neurons
- **Biology.** Neurosecretory cells that sense internal state (feeding state, osmolarity, temperature) and
  broadcast it by neuropeptides are deeply conserved; the vertebrate hypothalamus and the insect
  pars intercerebralis probably share this ancestor (Tessmar-Raible, Arendt et al.). Drives change the
  *valence* of stimuli: food is attractive only when hungry. Thermotaxis in *C. elegans* works on temperature
  *changes* sensed by AFD, with run-and-tumble steering.
- **Adds.** `hunger` (fires when the food meter is low; broadcast as a modulator, receptors on `valence_app`
  raise appetite with need), `cold` (fires below a comfortable temperature) gating two innate thermotaxis
  interneurons: cold AND warming -> keep going forward, cold AND cooling -> turn. New senses: body temperature and
  its change.
- **World.** OHOL-style temperature: body temperature drifts toward the ambient temperature plus heat from hot
  springs (OHOL heatValue); cold multiplies hunger (as in OHOL). More bushes compensate so it is not a cliff.
- **Replaced.** The two-nutrient forager of the first plan (OHOL has no second nutrient; temperature is the
  OHOL-faithful second drive).
- **Confidence.** High that homeostatic neuropeptide control is ancient. Medium that it looked like a
  hypothalamus in the first bilaterians.

#### 1.3 Affect: persistent behavioural modes (dwelling, roaming)
- **Biology.** In *C. elegans*, serotonin released on food encounter produces **dwelling** (slow, many turns,
  local search) and neuropeptide PDF produces **roaming** (long straight runs) (Flavell et al. 2013). The
  mode lasts minutes, long after the stimulus is gone. Bennett's "affect" is exactly this: a global,
  persistent state along valence and arousal that biases every reflex at once. (Escape and stress modes are
  left for later.)
- **Adds.** `raphe` (serotonin, slow, driven by taste) broadcasting `5ht` onto receptors of `dwell` (-> turns);
  `pdf` (slow, driven by hunger) broadcasting `pdf` onto receptors of `roam` (-> forward). The two nuclei
  inhibit each other through hard-wired inhibitory receptors (v5); evolved input from valence.
- **World.** Food in 10 dense patches (calibrated so foraging still pays).
- **Confidence.** High for the phenomenon across bilaterians (area-restricted search is found in worms,
  flies, fish, birds and mammals). Medium for the exact neuromodulator mapping in the ancestor.

#### 1.4 Habituation
- **Biology.** Habituation (a weaker response to a repeated harmless stimulus) is found in essentially every
  animal, and even in single cells. Sensory-specific satiety (a food eaten repeatedly loses appeal) is one form.
  Sensitisation (stronger reflexes after a noxious event, serotonergic in *Aplysia*) is left for later.
- **Adds.** Short-term synaptic depression (`u <- u + (1-u)/tau_rec - U u pre`, effective weight `w u`) on the
  identity -> appetitive synapses.
- **World.** OHOL's "yum" mechanic: a food not eaten recently is worth more, a repeated one less, so switching
  between food types pays. (Replaces the contact-hazard design of the first plan: no OHOL counterpart.)
- **Caveat.** Weakest payoff of chapter 1; it is needed later (2.7 reuses habituation as familiarity).
- **Confidence.** High.

#### 1.5 Associative learning: classical conditioning with a US-gated rule (exp02 done right)
- **Biology.** Worms learn to associate an odour or salt concentration with starvation or food; *Aplysia*
  learns to associate a touch with a shock; bees and flies learn odour->sugar or odour->shock. The mechanism
  is activity-dependent, modulator-gated plasticity: the synapse from the CS neuron is strengthened when
  it was recently active *and* the US modulator arrives. In insects this happens at the mushroom body
  output synapses, with dopamine neurons carrying the US (Aso, Rubin et al.).
- **Adds (ADR-021).** US neurons `us_taste` and `us_pain` (hard-wired from the senses), each its own teacher:
  taste (`us_app`) teaches the identity -> appetitive synapses about what was sensed just before (short eligibility
  trace), sickness (`us_av`) teaches the identity -> aversive synapses about what was sensed up to several ticks
  earlier (long trace): `dW = eta * teacher * trace(pre)`. One signed teacher with one long trace erased its own
  lessons. Learning rates evolve per projection.
- **World (v5, ADR-017).** Two novel berry types get a new look every life and one of them is poison (on top
  of two ancestral good and two ancestral poison types); poison costs a whole berry; lives last 2000 ticks;
  sickness arrives 12 ticks after eating and fades quickly, so no reactive policy can use it.
- **Metric.** Poison fraction in the second vs first half of life; the decisive test is the lesion
  "no_plasticity" (same population, learning off).
- **Finding so far.** Learning acts (pain drops) but over-generalises across look-alikes, because non-error-driven
  Hebbian conditioning accumulates the features the look-alikes share. Error-driven learning (TD, 2.5) should fix
  this; see STATUS/STAGE_LOG.
- **Confidence.** High that modulated associative learning exists across bilaterians. Medium on whether the
  *common ancestor* had it or whether it evolved several times.

#### 1.6 Extinction and reversal
- **Biology.** Extinction is not forgetting: after a CS stops predicting the US, the response fades, but it
  returns spontaneously later (Pavlov). Neurally this is new inhibitory learning layered over the old
  association, plus fast and slow memory components.
- **Adds.** The learned weights relax back toward their inherited values (`decay`), a fast forgetting component.
- **World.** As 1.5, but the poison swaps to the other look-alike halfway through life.
- **Confidence.** High for the phenomena in vertebrates and many invertebrates; the fast-component
  implementation is a modelling choice.

### Chapter 2: reinforcing (early vertebrates, about 520–450 Mya)

Ecological driver: Cambrian predation and active hunting in open water, followed by vision-guided
navigation. The vertebrate brain plan is already complete in lampreys: forebrain (pallium, subpallium incl.
**basal ganglia**, **hypothalamus**), midbrain (**tectum**, dopamine neurons), hindbrain (reticular
formation, motor pattern generators, and a small **cerebellum**, which appears clearly only in jawed
vertebrates). Grillner's lamprey work shows the basal ganglia circuit (striatum, GPi/SNr, STN, SNc
dopamine) already present and wired as in mammals. That is the main anatomical reason for putting RL here.

Suggested region names from here on: `tectum`, `pallium`, `striatum`, `gpi`, `snc` (dopamine),
`hippocampus` (medial pallium), `cerebellum`, `brainstem` (motor programs; the old `hidden` can be renamed
to this).

#### 2.1 Optic tectum: topographic target selection
- **Biology.** The tectum (superior colliculus in mammals) is a retinotopic map. It picks *one* target by
  lateral inhibition and drives orienting and approach toward it (or escape from a looming object).
- **Adds.** Region `tectum` with one neuron per vision column (or two: approach and avoid). Topographic
  `in(vision col c)→tectum(c)` projections. Fixed negative projections `tectum→tectum` (lateral
  inhibition, M5). `tectum→out(turn left/right/forward)` topographic.
- **Rule.** None new.
- **Experiment.** Dense food with several targets in view simultaneously, and a few hazards next to food.
- **Control.** 1.x architecture (distributed hidden region).
- **Metric.** Ticks from "food in view" to "reaching it"; oscillation (left-right dithering count).
- **Expected.** Fewer Buridan's-ass failures (averaging between two targets and walking between them).
  Evolution *can* find this without the module, but the topographic prior makes it immediate.
- **Confidence.** High (the tectum is a vertebrate hallmark; homologous map-like structures exist in some
  invertebrates).

#### 2.2 Pallium as an expansion layer (sparse random coding)
- **Biology.** The oldest pallium is largely olfactory: piriform-like cortex with sparse, random, expansive
  input from the olfactory bulb. Insects converged on the same design (mushroom body Kenyon cells). Random
  sparse expansion turns an entangled input into one where almost any category is linearly separable
  (Babadi & Sompolinsky 2014; Litwin-Kumar et al. 2017).
- **Adds.** Region `pallium` (large: 5–10× the appearance features), fixed sparse random `in→pallium`,
  k-WTA (M5). The plastic, US-gated projection of 1.5 now runs `pallium→valence` instead of
  `in→valence`.
- **Rule.** None new; the 1.5 rule on a different substrate.
- **Experiment: nonlinear poison rule.** Four berry types built from two appearance features (colour A/B ×
  shape A/B). Poison = an XOR of the two (e.g. A-A and B-B are poison), randomised per life by which pair
  is poison.
- **Control.** 1.6 (plastic `in→valence`).
- **Metric.** Within-life drop in poison fraction.
- **Expected.** The control cannot represent XOR with a single plastic layer onto valence; the expansion
  agent can.
- **Confidence.** High that the vertebrate pallium originated as olfactory/sensory association cortex; the
  exact "expansion" function in early vertebrates is a model inference.

#### 2.3 Pallium: pattern completion and clustering (unsupervised Hebbian)
- **Biology.** Piriform cortex and the hippocampus have strong recurrent excitatory connections that store
  patterns and complete them from partial input (Marr; Hopfield; Haberly). Competitive Hebbian learning
  with inhibition clusters similar inputs into categories. This is the "associative learning for pattern
  completion and clustering" in the user's summary. Note that it is *unsupervised*: no US needed.
- **Adds.** `pallium→pallium` recurrent excitation (plastic `hebb`, unmodulated, with weight
  normalisation), and `in→pallium` becomes plastic `oja`/competitive instead of fixed random.
- **Rule.** Oja or competitive Hebb (exists: `rule="oja"`); recurrent Hebb with normalisation (normalisation). This is where plain Hebbian learning finally does the job it is suited for.
- **Experiment: noisy, occluded objects.** Appearance vectors get per-sighting noise, and some vision
  features are randomly dropped (M9). New object types are introduced per life (random appearances), so
  categories must be formed *within* the life. Poison is one of the new categories.
- **Control.** 2.2 (fixed random expansion).
- **Metric.** Within-life poison-fraction drop at high noise; number of samples needed.
- **Expected.** Clustering gives noise-robust categories, so the valence learning on top needs fewer
  samples. At zero noise the two should tie. Run a noise sweep to show the crossover.
- **Confidence.** Medium–high (recurrent autoassociation in olfactory and hippocampal pallium is widely
  conserved in amniotes; less certain for the earliest vertebrates).

#### 2.4 Basal ganglia: gated selection of brainstem motor programs (fixed)
- **Biology.** Brainstem and spinal pattern generators produce complete action programs (swim, orient,
  bite, escape). The GPi/SNr output nuclei inhibit all of them tonically. The striatum, driven by
  pallium, thalamus and tectum, disinhibits one channel at a time (Redgrave, Prescott & Gurney 1999). This
  is a selection architecture; learning comes in 2.5–2.6.
- **Adds.** `striatum` (one channel per action, or per "program"), `gpi` (one per action, high bias, i.e.
  tonically active), `gpi→out` fixed inhibitory, `striatum→gpi` fixed inhibitory. Inputs to striatum:
  `pallium`, `tectum`, `valence`, `hypothalamus`. Optional `stn` (diffuse excitation of gpi) for a
  "hold everything" brake when there is conflict.
- **Rule.** None new (fixed, evolved).
- **Experiment: conflicting drives.** Food next to a thorn hazard; hunger vs pain; two meters low at once
  (from 1.2).
- **Control.** 2.3 without the BG loop.
- **Metric.** Dithering (switching between two actions over consecutive ticks without progress); decisive
  completion of action sequences.
- **Expected.** Small gains on its own. This stage mostly sets up the structure 2.5/2.6 learn in. It is
  acceptable to merge it with 2.6.
- **Confidence.** High (the lamprey BG is well characterised).

#### Note: why there is no separate Rescorla-Wagner stage
The 1.5 probes showed that plain US-gated conditioning over-generalises across look-alikes; error-driven
(Rescorla-Wagner) learning is the textbook fix. In continuous time, however, a prediction that is active *before*
the outcome arrives counts as an error and unlearns itself; the temporal-difference error
`r + gamma V(t) - V(t-1)` is the continuous-time form of the same idea. So 2.5 introduces TD directly and lets
dopamine replace the raw US as the teacher of the valence synapses.

#### 2.5 Dopamine as reward prediction error (the critic)
- **Biology.** Midbrain dopamine neurons report `δ = r + γV(s′) − V(s)` (Schultz, Dayan & Montague 1997). The
  response moves from the reward to the earliest predicting cue, and a fully predicted reward no longer
  produces a response. This explains blocking and second-order conditioning, which pure US-gated learning
  (1.5) does not.
- **Adds (as implemented).** Rates cannot be negative, so values are an opponent pair: `value_app` (expected
  good) and `value_av` (expected bad) in the ventral striatum, plus copies delayed by one step. The modulator
  `da = (us_taste - us_pain) + gamma (V_app - V_av) - (V_app_prev - V_av_prev)` (weighted region means) is the TD
  error. Identity features and the pallium -> value synapses learn by TD(lambda) (`dW = eta * da * trace`); the
  value populations feed valence (incentive salience); every `us`-gated projection switches to `da`.
- **Rule.** TD(lambda) as a three-factor rule with eligibility traces.
- **Experiment: chained cues.** A reward requires a short causal chain: e.g. a berry is only available at
  bushes next to a marker object (a stone), or a food source must be USEd twice with a delay. The marker
  identity is randomised per life. Reward comes 5–20 ticks after the first cue.
- **Control.** 2.4 with US-gated (1.5-style) learning.
- **Metric.** Approach latency to the *first* cue in the chain, later in life vs earlier; blocking test (a
  second cue added after the first is learned should not be learned).
- **Expected.** The TD agent learns to approach the first cue in the chain; the US-gated agent learns only
  cues close to the reward (or needs very long traces, which then overgeneralise).
- **Confidence.** High for mammals and good evidence for fish; the dopamine RPE in lampreys is plausible but
  less directly shown.

#### 2.6 Actor: learned action selection in the striatum (Go/NoGo)
- **Biology.** The same `δ` trains cortex→striatum synapses: on D1 (direct, "Go") cells a positive `δ`
  strengthens, on D2 (indirect, "NoGo") cells a negative `δ` strengthens (Frank 2005; Collins & Frank 2014).
  This is actor-critic RL, and operant (instrumental) conditioning. It differs from Pavlovian learning in
  1.5: it learns which *action* to take, not just the value of a stimulus.
- **Adds.** Split `striatum` into `striatum_d1` and `striatum_d2`; `pallium→striatum_*` plastic,
  `modulated="da"`, with opposite sign conventions; `d1→gpi` inhibitory (Go), `d2→(gpe)→gpi` net excitatory
  (NoGo).
- **Rule.** `dW = eta·(+δ)·e` on D1, `dW = eta·(−δ)·e` on D2; eligibility trace per synapse (M1).
- **Experiment: per-life action→outcome contingency.** In OHOL terms: one food source yields only if the
  agent USEs it while holding a particular tool (e.g. a stone), and which tool works is randomised per life.
  Or: a bush gives food on USE, a look-alike gives food only on repeated USE after a wait. The agent must
  learn *what to do*, not only what to approach.
- **Control.** 2.5 (critic only; actions still from evolved fixed weights plus value-driven approach).
- **Metric.** Rate of successful yields per life, second half vs first half.
- **Expected.** Only the actor-critic agent learns the per-life action.
- **Confidence.** High for mammals; the D1/D2 split is conserved at least to lampreys (Grillner lab).

#### 2.7 Curiosity: novelty as intrinsic reward
- **Biology.** Dopamine neurons also respond to novel, unexpected stimuli; fish and all later vertebrates
  explore new objects, and Bennett puts curiosity here. It is the classic fix for exploration in RL.
- **Adds.** A `novelty` region driven by pallium activity *not* explained by habituated (familiar)
  synapses from 1.4; `novelty→snc` adds a bonus to `δ`.
- **Rule.** No new rule; reuses M4 depression as familiarity.
- **Experiment: sparse hidden rewards.** Food is rare and found only by interacting with objects
  (USE on some non-food object occasionally yields food, identity randomised per life), in a world full of
  inert objects.
- **Control.** 2.6.
- **Metric.** Fraction of object types the agent has interacted with; food per life.
- **Expected.** The curious agent samples more object types and finds the productive one; the control
  settles on the first thing that pays.
- **Confidence.** Medium (novelty responses of dopamine neurons are well shown in mammals; evidence for
  "curiosity" in fish is mostly behavioural).

#### 2.8 Hippocampus homologue: spatial maps
- **Biology.** Lesions of the goldfish lateral pallium (the homologue of the hippocampus) impair spatial but
  not cue learning (Rodríguez, Salas et al.). Head-direction cells (in flies and in the vertebrate
  brainstem/thalamus) do path integration from self-motion. Place cells bind locations to what is there.
- **Adds.**
  - Body input: efference copy of the last action (M7).
  - A `head_direction` ring attractor (fixed, evolved or hand-set) updated by turn efference copies.
  - A `hippocampus` region receiving `head_direction`, efference copy of FORWARD and pallial object
    features; recurrent (place-like) Hebbian connections; `hippocampus→value` plastic under `da`.
- **Rule.** Recurrent Hebb (2.3) plus TD on place features (2.5).
- **Experiment: remembered location.** In each life, a single rich food patch sits at a random position,
  out of view from most of the map, and regrows; landmarks (distinct rocks) are scattered. The agent must
  return to it after eating elsewhere or after the patch has regrown.
- **Control.** 2.7.
- **Metric.** Ticks from leaving the patch to returning to it, later in life.
- **Expected.** The map agent returns directly; the control has to rediscover it by search.
- **Confidence.** Medium–high for the teleost hippocampus homologue; lower for lampreys.

#### 2.8b Sequences: asymmetric STDP on place cells
- **Biology.** Recurrent hippocampal synapses with asymmetric (pre-before-post) plasticity learn travelled
  paths; place fields shift backwards with repetition, so activity anticipates the next location (Mehta et
  al. 2000). This is the substrate of replay (3.3) and of simulating paths (3.2).
- **Adds.** No new region. `hippocampus→hippocampus` switches from symmetric `hebb` to `trace` (STDP-like).
- **Experiment: anticipating a route.** A reliable food source sits at the end of a winding corridor or
  route (blocking objects), fixed per life, visited repeatedly.
- **Control.** 2.8 with symmetric Hebb.
- **Metric.** Ticks per traversal later in life; how early the agent turns before each bend.
- **Confidence.** High for the phenomenon in rodents; medium for early vertebrates.

#### 2.9 Cerebellum: timing and forward prediction
- **Biology.** The cerebellum appears clearly in jawed vertebrates (lampreys have a small version; hagfish
  have none). Granule cells are another expansion layer; Purkinje cells learn from a climbing-fibre error
  signal (Marr-Albus-Ito). Classic function: precisely timed conditioned responses (eyeblink) and forward
  models that predict the sensory consequences of one's own actions.
- **Adds.** `granule` (expansion of `in` + efference copy + a bank of slow time-cells), `purkinje`,
  `climbing` (teacher: pain or prediction error), `purkinje→out` inhibitory (via deep nuclei).
- **Rule.** New: delta rule with a teacher region (M6), `dW = −eta·pre·climbing` (LTD on error).
- **Experiment: timed hazard.** A warning object appears in view (e.g. a flame starts flickering) and after
  exactly `k` ticks it stings anything facing it; stepping away at the right time avoids it, but stepping
  away too early loses the food next to it. `k` is randomised per life (5–30 ticks).
- **Control.** 2.8.
- **Metric.** Pain per exposure and food lost, later in life.
- **Expected.** The cerebellar agent times the avoidance; the BG agent can learn "avoid when the warning is
  on" but pays for leaving early.
- **Honest caveat.** A discrete grid world gives the cerebellum little to do. Its bigger role here is
  introducing the **delta rule**, which chapter 3 depends on. This can be a short stage.
- **Confidence.** High for the anatomy and eyeblink function; the forward-model role is well supported in
  mammals.

#### 2.10 Uncertainty modulators: noradrenaline and acetylcholine
- **Biology.** The locus coeruleus (noradrenaline, NE) and basal forebrain (acetylcholine, ACh) exist in all
  vertebrates. Theory (Yu & Dayan 2005; Behrens et al. 2007) says NE signals unexpected uncertainty
  (the world changed: learn faster, explore) and ACh signals expected uncertainty (the cues are unreliable:
  rely on priors).
- **Adds.** An `lc` region reading `|δ|` over time and writing `ne` (M2). `ne` scales `eta` of plastic
  projections and the action temperature.
- **Rule.** Learning rate scaled by a modulator (meta-learning as mechanism).
- **Experiment: stable vs volatile lives.** Per life, the poison/reward identity either never changes
  (stable) or reverses every 100–200 ticks (volatile), randomised.
- **Control.** 2.9 (fixed `eta`).
- **Metric.** Reward per life, separately for stable and volatile lives.
- **Expected.** A fixed `eta` is a compromise that is too slow for volatile lives or too noisy for stable
  ones; the NE agent adapts.
- **Confidence.** Medium (the neuromodulators are conserved; the computational roles are theory with
  growing but indirect evidence).

### Chapter 3: simulating (early mammals, about 220–150 Mya)

Ecological driver (Bennett's version): small nocturnal mammals living among dinosaurs benefited from
planning moves before making them. The new structure is the six-layered **neocortex** (sensory, motor,
agranular prefrontal), in a loop with the thalamus and basal ganglia. Birds converged on similar abilities
with a differently organised pallium (DVR), so "simulation" is a function, not uniquely a layered cortex.

#### 3.1 Neocortex as a generative model (self-supervised prediction)
- **Biology.** The predictive processing view (Rao & Ballard 1999; Friston; Hawkins) treats cortex as learning
  to predict its own input, with the error between prediction and input driving learning. Feedforward
  input arrives in layer 4, predictions come top-down into layer 1/apical dendrites; the thalamus relays
  and gates.
- **Adds.** `cortex_l4` (input), `cortex_l23` (state), `cortex_pred` (predicted next input), `error`
  (input − prediction, fixed wiring). `cortex_l23` also receives efference copy (M7) so that it predicts
  the consequences of actions. `cortex_l23→striatum` and `→value` join the pallium inputs.
- **Rule.** Delta rule (M6) with `error` as the teacher: `dW(l23→pred) = eta·l23·error`. Unsupervised
  Hebb/Oja on `l4→l23`.
- **Experiment: partial observability with predictable structure.** Narrow field of view (fewer columns),
  with food that regrows on a fixed schedule at fixed positions per life, and objects that are often
  occluded. The good state representation is "what is where and when", which has to be inferred.
- **Control.** 2.10.
- **Metric.** Food per tick; also a probe: prediction error on held-out ticks (the dashboard can show
  `error` activity dropping over a life).
- **Expected.** A modest gain in reward; the main visible effect is a learned world model. It becomes
  decisive in 3.2.
- **Confidence.** Medium; predictive processing is a strong framework but not established fact.

#### 3.2 Vicarious trial and error: simulating actions before taking them
- **Biology.** Rats pause at choice points and "look" both ways; the hippocampus then sweeps ahead along each
  option (Johnson & Redish 2007). Tolman's latent learning, detours and outcome devaluation (Adams &
  Dickinson) show behaviour that model-free RL cannot produce.
- **Adds.** M8 offline mode. A `deliberate` neuron (driven by high uncertainty / conflict in striatum, i.e.
  from 2.10's `ne`) switches the thalamic gate: for K internal steps, motor output is suppressed, the
  cortex's predicted next input replaces real input, a candidate action from striatum is fed as efference
  copy, and `value` evaluates the imagined state. The best imagined action is then executed.
- **Rule.** No new plasticity; a new *mode* of using the model.
- **Experiments.**
  1. *Latent learning.* The first half of life has no reward in a walled maze-like area (OHOL walls or
     blocking objects); then food appears at one location. The model-based agent goes there directly.
  2. *Outcome devaluation.* Two foods, two meters (1.2). After the agent is sated on one, does it
     immediately stop working for it (goal-directed) or keep going (habit)?
  3. *Detour.* A known path gets blocked mid-life (M10).
- **Control.** 3.1 without offline mode (same world model, used only as a representation).
- **Metric.** Ticks to reach the reward after it appears; responses to the devalued food.
- **Expected.** Clear wins in all three for the simulating agent. This is the stage where "more than RL"
  (the thing exp02 was reaching for) actually appears.
- **Confidence.** High for the behaviour in mammals; the specific circuit is still debated.

#### 3.3 Replay and sleep: learning from simulated experience
- **Biology.** During rest and sleep, the hippocampus replays sequences, forwards and backwards (Foster &
  Wilson 2006); replay seems to drive consolidation into cortex and value updates in striatum. Dyna (Sutton
  1990) is the RL counterpart.
- **Adds.** A rest state (NOOP for a stretch or a NOOP-heavy "sleep" when sated) triggers offline mode with
  no sensory input: the cortex/hippocampus generate sequences, and the normal `da`-gated rules learn from
  them.
- **Experiment.** Same as 2.8 or 3.2, but with experience limited (short lives or few rewards); make
  resting safe (no predators yet).
- **Control.** 3.2 with offline mode only at choice points.
- **Metric.** Reward per life vs amount of real experience.
- **Expected.** Better sample efficiency. Resting evolves as a behaviour in its own right.
- **Confidence.** High for replay; medium for its necessity in learning.

#### 3.4 Agranular prefrontal cortex: holding goals over time
- **Biology.** Rodent medial PFC holds task rules and goals and gates which options get simulated;
  PFC-BG-thalamus loops update working memory in a gated way (O'Reilly & Frank 2006, PBWM).
- **Adds.** `pfc` with strong recurrent self-excitation (attractors, slow leak); a separate BG loop
  (`striatum_pfc`) that learns *when to update* `pfc` content, under `da`; `pfc→cortex_l23` biases
  simulation, `pfc→striatum` biases action.
- **Rule.** Existing `da`-gated rules, now also on the gating loop.
- **Experiment: multi-step goals with distractors.** A reward needs a sequence such as "pick up tool X at
  place A, carry it past food to target B, USE". Which tool and which target are shown by a cue at the
  start of each life (or each trial). Distracting food is on the way.
- **Control.** 3.3.
- **Metric.** Completed sequences per life.
- **Expected.** The PFC agent keeps its goal; the control drops the tool at the first food.
- **Confidence.** Medium (PFC homologies across mammals are debated, but working-memory gating is well
  supported).

#### 3.5 Episodic memory and counterfactual learning
- **Biology.** One-shot storage of "what, where, when" (hippocampus). Rats show regret-like signals after a
  missed better option (Steiner & Redish 2014). Bennett puts both in the simulation breakthrough.
- **Adds.** Fast one-shot Hebbian storage in `hippocampus` (very high `eta` on a subset of synapses, reset
  or decaying after use); a counterfactual `δ` computed from a simulated alternative action.
- **Experiment.** One-shot events: a rare, rich food cache that appears once per life at a random place and
  must be remembered after one visit; near-miss situations where the unchosen option would have paid.
- **Control.** 3.4.
- **Expected.** Better use of single experiences.
- **Confidence.** Medium.

### Chapter 4: mentalizing (primates, about 65–5 Mya)

Ecological driver (Bennett, following the social brain hypothesis of Dunbar and Humphrey): large, complex
social groups where status and alliances decide access to food and mates. The new structure is
**granular prefrontal cortex** (with a layer 4), present in primates only (Preuss; Passingham & Wise 2012).
Here the framework needs other agents that matter: the multi-agent world already exists (ADR-008), and
continuous lifecycles (the continuous life cycle, STATUS.md) become necessary.

#### 4.1 A model of one's own mind (metacognition)
- **Biology.** Granular PFC models the agent's *own* intentions and knowledge; monkeys can report confidence
  and opt out of hard choices (Hampton 2001; Kiani & Shadlen 2009).
- **Adds.** `gpfc` region that receives `pfc`, `cortex_l23` and `error`, and predicts the agent's own next
  action and its success probability. It is trained by delta rule (M6) on the actual action and outcome.
  `gpfc→striatum` can pick an "opt out" program.
- **Experiment: opt-out under uncertainty.** A choice between two berries whose appearance is noisy (2.3
  noise), and a third safe but small food source. Poison costs a lot.
- **Control.** Chapter 3 final architecture.
- **Metric.** Reward per life; whether opt-outs track difficulty.
- **Expected.** Better choices in hard trials. The main purpose is to build the self-model that 4.2 reuses.
- **Confidence.** Medium (metacognition is also claimed in rats and birds, and the granular-PFC link is
  Bennett's synthesis).

#### 4.2 Theory of mind: the self-model applied to others
- **Biology.** Primates infer others' goals and what they have seen (Hare, Call & Tomasello 2001 on
  chimpanzee food competition). The "simulation theory" of mind reading: run one's own model with the
  other's inputs.
- **Adds.** Observation extensions (M9): other agents' facing, held object and last action in the vision
  columns that hit an agent. A `tom` region: a copy of `gpfc`/`pfc` (growth by duplication, principle 4)
  whose inputs come from observed agents instead of oneself; it predicts their next action and inferred
  goal.
- **Rule.** Delta rule on observed actions (predict what they do next; the error comes when they do it).
- **Experiments.**
  1. *Following an informed agent.* Some agents (e.g. older, or born with a genome that sees further) know
     where food is; others can infer it from where they head.
  2. *Food competition.* A dominant agent takes food it sees; subordinates do better picking food the
     dominant one *cannot see* (requires modelling its view).
- **Control.** 4.1.
- **Metric.** Food per life for followers/subordinates.
- **Confidence.** Medium; ToM in non-human primates is real but more limited than in humans, and whether it
  needs granular PFC specifically is debated.

#### 4.3 Imitation and observational learning
- **Biology.** Primates (and some birds) learn tool use by watching (e.g. chimpanzee nut cracking, learned
  over years by watching the mother). This uses the ToM model's inferred action as a teacher signal.
- **Adds.** `tom`'s inferred action for an observed agent feeds `striatum` as an imitation target: a `da`
  bonus when the agent's own action matches an observed successful action (or a delta rule from observed
  action to own policy).
- **Experiment: crafting chain too hard to discover.** Use a larger OHOL slice (bigger OHOL slices, STATUS.md, e.g. sharp stone
  → skewer → ... → cooked food) worth a lot of food. A few "demonstrator" agents are warm-started with the
  skill hard-wired or pre-learned; others must acquire it within life. Randomise per life which of two
  chains works.
- **Control.** 4.2 (can see others but has no imitation pathway).
- **Metric.** Fraction of naive agents that complete the chain.
- **Confidence.** Medium–high for imitation in great apes; the mechanism is a model choice.

#### 4.4 Anticipating future needs
- **Biology.** The Bischof-Köhler hypothesis: only humans (and perhaps great apes and corvids; Mulcahy &
  Call 2006) plan for a drive they do not feel right now. Bennett puts this with mentalizing, as modelling
  one's *future self* as another mind.
- **Adds.** In offline mode (M8), the `hypothalamus` input to simulated value can be replaced by a
  *simulated* future drive state (e.g. "hungry, in winter").
- **Experiment: seasons.** A long life with a winter period in which nothing regrows (M10). Carrying/storing
  food beforehand (drop items in a spot, come back) is the only way to survive winter. The winter start is
  signalled by a cue (e.g. day length via a body input).
- **Control.** 4.3 with offline mode that only uses the current drive.
- **Metric.** Winter survival.
- **Confidence.** Low–medium; this is contested for all non-human animals.

### Chapter 5: speaking (hominins, about 2–0.1 Mya)

Ecological driver: cooperation in hunting and child care, and knowledge that accumulates across
generations. Language probably relied on pre-existing capacities (shared attention, imitation, vocal
learning, which birds evolved independently) plus a hominin-specific drive to share attention and teach
(Tomasello). Honest scope note: this chapter is aspirational for a rate-coded grid world, but the first two
steps are feasible and connect to the future goals in STATUS.md.

#### 5.1 Innate signals (side branch, possible from chapter 1–2 onward)
- **Biology.** Vocal or chemical alarm and food calls exist in fish, birds and monkeys (vervet alarm calls:
  Seyfarth & Cheney). They are innate and tied to affect.
- **Adds.** Sound channel widened to a small vector (ADR-005 allows this); `affect→out(VOCALIZE)` and
  `sound→affect`.
- **Experiment.** Moving predators (M10); related agents (lineages, the continuous life cycle, STATUS.md) so that calling pays
  through kin selection. The control is agents without sound input.
- **Confidence.** High.

#### 5.2 Learned signal meaning
- **Adds.** Plastic `sound→pallium/value` under `da`; vocalization content selectable by BG.
- **Experiment.** A signalling game: one agent sees where food is and cannot reach it, another can reach it
  but cannot see it; both are rewarded. The mapping from signal to meaning is not given; conventions must
  form within a life. Different groups form different conventions.
- **Confidence.** Medium (learned signalling in some primates and many birds).

#### 5.3 Shared attention, naming, proto-grammar
- **Adds.** A joint-attention signal: when two agents look at the same object, association between the
  heard sound and that object is boosted (a modulator gated by `tom` detecting shared gaze).
- **Experiment.** Cooperative crafting where the recipe is known to one agent and the materials to
  another; instruction-following.
- **Confidence.** Low–medium as a claim about evolution (Tomasello's view is influential but debated).

#### 5.4 Language as shared simulation and cumulative culture
- **Idea (Bennett).** Language lets one brain drive another's simulator: "the berries past the big rock are
  poison" installs a model-based lesson without the listener tasting them. Across generations, knowledge
  ratchets up.
- **Experiment.** The continuous lifecycle (the continuous life cycle, STATUS.md) with a crafting tree deep enough that no single
  life can rediscover it; measure whether the depth of crafts reached keeps rising across generations with
  communication on and plateaus with it off.
- **Confidence.** Speculative as a mechanism, but a clean measurable target.

---

## 5. Region → function → first stage

This is a reference for "what does the brainstem do vs the hypothalamus" in this project.

| structure | real function (simplified) | where in this plan | framework form |
|---|---|---|---|
| spinal cord / brainstem pattern generators, reticular formation | complete motor programs (swim, turn, bite); posture; basic reflexes; arousal | 1.0 (`hidden` → rename `brainstem`) | evolved fixed weights; later gated by BG |
| hypothalamus | internal-state sensing (hunger, thirst, temperature), drives, hormones; selects which goal matters | 1.2 | drive neurons from body inputs; gain modulation (M3) |
| neuromodulatory nuclei (raphe 5-HT, VTA/SNc DA, LC NE, basal forebrain ACh) | broadcast state: satiety/patience, reward prediction, surprise, attention | 1.3 (5-HT-like), 2.5 (DA), 2.10 (NE, ACh) | named modulator regions (M2) |
| valence / amygdala | good/bad classification of stimuli; fear and appetitive conditioning | 1.1, 1.5 (in vertebrates the pallial amygdala takes this role) | US-gated plastic projection onto valence |
| optic tectum / superior colliculus | retinotopic target selection, orienting, looming escape | 2.1 | topographic region with lateral inhibition |
| pallium (olfactory cortex) | pattern separation, completion, categories | 2.2, 2.3 | expansion + k-WTA + Oja/recurrent Hebb |
| basal ganglia (striatum, GPi/SNr, STN) | action selection by disinhibition; RL actor and critic | 2.4–2.6 | striatum channels, tonic GPi, `da`-gated rules |
| hippocampus (medial pallium) | space, sequences, episodic memory, replay | 2.8, 3.3, 3.5 | head-direction ring, place-like recurrent Hebb, one-shot storage |
| cerebellum | timing, forward models, error-driven motor learning | 2.9 | expansion + delta rule with climbing-fibre teacher |
| thalamus | relay and gating of input to cortex; switches between perception and imagination | 3.1–3.2 | the gate of offline mode (M8) |
| neocortex (sensory) | generative model of the senses | 3.1 | predictive-coding layers, delta rule |
| agranular PFC (rodents and up) | goals, working memory, which simulations to run | 3.4 | attractor + gated update loop |
| granular PFC (primates) | model of own mind, others' minds | 4.1–4.2 | self-model; duplicated for others |
| language areas | shared simulation | 5.x | aspirational |

---

## 6. Convergent side branches (optional)

These evolved independently and make good "does the same function need the same structure?" experiments.

- **Insect mushroom body:** expansion (Kenyon cells) + dopamine-gated output synapses, i.e. stages 1.5 + 2.2
  without a vertebrate pallium. It is the best-understood modulated associative learning circuit (the full
  larval and adult *Drosophila* connectomes exist) and is a realistic, well-documented template for
  implementing 1.5/2.2.
- **Insect central complex:** head-direction ring attractor and path integration (Seelig & Jayaraman 2015),
  the cleanest model for 2.8.
- **Octopus:** complex learning with a very different, distributed brain.
- **Birds:** mammal-level simulation and planning with a nuclear (non-layered) pallium; corvids cache food
  (4.4 analogue).

---

## 7. Order of work

Current status, and what runs next: `docs/STATUS.md`.

---

## 8. Main sources (for further reading)

- Bennett, M. (2023). *A Brief History of Intelligence.*
- Cisek, P. (2019). Resynthesizing behavior through phylogenetic refinement. *Attention, Perception &
  Psychophysics.* A stepwise evolutionary account of the brain, very close in spirit to this document.
- Grillner, S. & Robertson, B. (2016). The basal ganglia over 500 million years. *Current Biology.*
- Striedter, G. & Northcutt, R. G. (2020). *Brains Through Time.*
- Schultz, Dayan & Montague (1997). A neural substrate of prediction and reward. *Science.*
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
