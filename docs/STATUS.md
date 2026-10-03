# Status

One screen on where the project is. Plan: `docs/BRAIN_EVOLUTION.md`. Details, numbers and run directories:
`docs/STAGE_LOG.md`. Newest decisions: `docs/DECISIONS.md` (ADR-017 to ADR-022). Last updated 2026-10-03.

## How a stage is judged

A stage succeeds when its circuit is **used**, so that later stages can build on it, and main is not clearly worse
than control (user, 2026-10-01). One life is mostly luck, so a design is checked in this order (ADR-022,
`scripts/probes/README.md`):

1. **Mechanism, brain only**: `scripts/probes/assay.py` (scripted experience, then actions in fixed situations).
2. **Generation 0, paired**: `scripts/probes/gen0.py` (parent population, mechanism on vs off, same worlds).
3. **Evolution**: main vs control, 64 genomes x 4 siblings, 150 generations, three seeds for any claim:

       stages replicate <key>      two more seeds of main and control
       stages summary <key>        last 50 generations and re-evaluation in 8 shared worlds, +- s.e. over seeds
       stages versus <key>         main and control agents in the same worlds
       stages lesions <key>        fitness with each region silenced, % of intact, per seed

**Fitness = well-fed lifetime from the first meal** (ADR-018). Activity passes through the layers within a tick
(ADR-019).

## Stages (v18, 2026-10-03)

Status: **done**, **defined** (in `life/experiments/stages.py`, not run on the current definitions), **planned**.
Everything below 1.1 has changed underneath stages 1.2-1.6 (programme cells, ordered propagation, siblings), so
their last results (STAGE_LOG, v9 and v13) are from older definitions.

| stage | what it adds | status | finding (one seed) |
|---|---|---|---|
| 1.0 steering | ganglion (exc + inh, lagging normalisation), evolved reflexes; sees the outside world only; movement costs energy | done (v18) | Foraging evolves from random brains: lifetime 692 of 1000, 27 meals per life. |
| 1.1 valence | value cells (seen / held) acting on four context-gated motor programmes: approach, grasp, ingest, reject; taste and pain enter only here | done (v18) | Used: silencing the appetitive cells leaves 37% of fitness, the aversive cells 61% (poison eaten x4), grasp 37%. Head to head +54. Clean behaviour at the bush. Approach, ingest and reject programmes not used in this seed. Agents spend ~45% of their time pressing USE at empty bushes. |
| x.learn (side) | conditioning directly on the 1.1 brain; immediate pain, short traces; novel foods per life | generation-0 tests only | On the v18 stage-1.1 population, learning on vs off: fitness 460 vs 469 (a tie), poison share of berries eaten 0.16 vs 0.19, a third less pain, but a quarter fewer berries eaten. Selective and no longer harmful; not yet paying. |
| 1.2 drives | `hunger`, `cold` gating innate thermotaxis; food level and temperature enter only here | defined | v9/v13: not used (lesions 100-102%). |
| 1.3 affect | serotonin (dwell) and PDF (roam), mutually inhibiting | defined | v9: PDF/roam used, serotonin weakly. |
| 1.4 habituation | short-term depression on identity -> appetitive synapses; dud bushes with a per-life look | defined | v13: 629 vs 594. v9 (variety bonus): not used. |
| 1.5 association | two teachers (taste, sickness) on identity -> value synapses; novel foods per life | defined | v13: flat, learning unused; see x.learn. |
| 1.6 reversal | learned weights relax toward inherited values; poison swaps mid-life | defined | |
| 2.1-2.5, x.td | tectum, pallium, basal ganglia, dopamine TD | defined (older design) | To revisit after chapter 1: per-life pallium weights (ADR-017), phases, programme cells as the basal ganglia's targets. |
| 2.6-2.10, 3.x-5.x | actor, curiosity, hippocampus, cerebellum; simulating, mentalizing, speaking | planned | |

## Decisions waiting for the user (2026-10-03)

1. **Fitness = energy acquired?** Food eaten without the stomach cap (surplus becomes offspring). Measured 15-30%
   more selection signal than the current fitness, and it has no ceiling: the current one stops rewarding an agent
   whose stomach is full, which makes resting at empty bushes rational.
2. **Evolution machinery.** A normal mutation is invisible to selection (0.04 of the spread of one life).
   Candidates: recombination (reproduction is asexual now), fewer and larger mutations, more siblings; to be compared
   on a recovery benchmark (damage a population, measure how fast each setting repairs it).
3. **Chapter-1 feeding: eat on grasp?** Pick, hold, then eat is OHOL's formulation; holding is a late invention in
   animals and causes much of the valence stage's wiring (hand gates, reject). Holding could arrive as its own stage.
4. **Critical path.** For chapter 2 we need 1.0, 1.1 and learning. Park 1.2-1.4 (drives, affect, habituation) until
   learning works?
5. **Learning rule.** Latent inhibition (familiar food is protected from blame) and blocking in chapter 1, or wait
   for the prediction error of 2.5? Latent inhibition would also give habituation a job.

## Known issues

- Agents press USE at empty bushes for almost half their lives (v17, v18).
- USE and EAT cost no energy (FORWARD +50%, turning +25%).
- Stages 1.2-1.6 and all of chapter 2 are defined on top of the new 1.1 but not run; their worlds need the no-cliff
  calibration again (`scripts/probes/calib.py`).
- The metabolic-rate calibration lowers how often an agent must eat; the learning stage needs many meals.

Practical notes: with 256 agents one process runs non-plastic stages at about 4 s and learning stages (2000 ticks)
at about 18-35 s per generation; at most two JAX processes at a time. A sleeping laptop pauses runs, and background
commands are stopped after two hours, so run long batches in pieces. `scripts/probes/prune.py` keeps only the
newest finished run per stage directory. The dashboard clips displayed weights at `w_max`.

## Core and tools (done)

Brain (ADR-015, 016, 019, 020): rectified rates, Dale's law, hard-wiring with tunable strength, input-feature
selection, source and target sub-ranges, topographic projections, named modulators with receptors and persistence,
eligibility traces, gain, depression, decay, delta rule, k-WTA, lagging divisive normalisation, ordered propagation
within a tick (phases), plasticity computed on the plastic block only, genome remapping across layouts.
World: look-alike and per-life appearances, dud bushes, OHOL temperature and hot springs, skin-temperature
gradient sense, delayed sickness, taste, patches, variety bonus, mid-life rule switch, appearance noise, movement
and (optional) neural metabolic cost, regrowth tied to the metabolic rate, multi-clone OHOL slicing.
Evolution: one life per individual, siblings, elitism + tournament, per-projection plasticity and strength genes.
Tools: stage runner and chain, replicate, summary, lesion(s), versus, respond, compare, dashboard (architecture
view with hard-wired projections drawn dark; a 64-agent sample of large populations), probes (`scripts/probes/`).
28 tests.

## Future goals (beyond the current chapters)

- **Continuous life cycle** (for chapters 4-5, mentalizing and speaking): birth near a parent, aging, lineages,
  overlapping generations, instead of generational resets. The `parent` field already exists in the world state.
- Bigger OHOL slices: stones, kindling, fire, cooking (needs `_LA` last-use-actor transitions and leftovers).
- Moving objects/animals (OHOL `move` transitions) for predators, prey and interception (cerebellum, alarm calls).
- Communication: VOCALIZE with a small vector (sound channel widening); a GIVE action for altruism.
- Scale: GPU runs in WSL2 (`docs/SETUP.md`), larger populations (selection is noisy with 64 agents and one life).
- Several seeds per stage comparison.

## Known simplifications to revisit

- `_LA` (last use of actor) transitions are skipped; tools never wear out.
- `move` transitions (animals) are skipped; nothing moves except agents.
- Eating drops any leftover object (bowl, plate) instead of keeping it in hand.
- Cell interactions are resolved sequentially by agent index (deterministic but biased toward low indices).
- Sound uses an N x N distance matrix; fine below ~5k agents.
- World generation ignores biomes: `map_chance` is used as a flat spawn weight; temperature has one ambient value.
- Object appearance is a hash (ADR-006) except for designed look-alikes; no sprite-derived appearance yet.
- Rates are updated synchronously once per tick; one tick is behavioural time (~1 s), so "STDP" is BTSP-like.
