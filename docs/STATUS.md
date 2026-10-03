# Status

One screen on where the project is. Plan: `docs/BRAIN_EVOLUTION.md`. Details, numbers and run directories:
`docs/STAGE_LOG.md`. Newest decisions: `docs/DECISIONS.md` (ADR-017 to ADR-023). Last updated 2026-10-03, evening.

## How a stage is judged

A stage succeeds when its circuit is **used**, so that later stages can build on it, and main is not clearly worse
than control (user, 2026-10-01). One life is mostly luck, so a design is checked in this order (ADR-022,
`scripts/probes/README.md`):

1. **Mechanism, brain only**: `scripts/probes/assay.py` (scripted experience, then actions in fixed situations).
2. **Generation 0, paired**: `scripts/probes/gen0.py` (parent population, mechanism on vs off, same worlds). A
   diagnostic, not a gate: a neutral or mildly negative result can still go on to evolution.
3. **Evolution**: main vs control, 64 genomes x 4 siblings, recombination, 150 generations, three seeds:

       stages replicate <key>      two more seeds of main and control
       stages summary <key>        last 50 generations (+ re-evaluation in 8 shared worlds), +- s.e. over seeds
       stages versus <key>         main and control agents in the same worlds
       stages lesions <key>        fitness with each region silenced, % of intact, per seed

**Fitness = energy acquired** (ADR-023). Cells that lesions show unused in every seed are removed.

## Stages (v21, 2026-10-03)

Critical path: 1.0 -> 1.1 -> 1.5 -> 1.6 -> chapter 2. Three seeds unless stated; lesion = fitness with the region
silenced, % of intact, mean over seeds.

| stage | what it adds | status | finding |
|---|---|---|---|
| 1.0 steering | ganglion (exc + inh, lagging normalisation), evolved sensor -> motor reflexes; sees the outside world only; movement costs energy; the animal eats what it grasps | done (1 seed) | 172 energy per life, lifetime 801 of 1000, 88 meals. |
| 1.1 valence | appetitive and aversive value cells (taste and pain enter the brain only here) acting on a contact-gated grasp programme; aversion turns away and blocks the bite | done | Main 60.1 vs control 29.5, head to head +81 +-3. Lesions: appetitive 46, aversive 52, no_feed 55, grasp 46, gate 59. The ganglion is idle in this stage (99-102). |
| 1.5 association | taste teaches the identity -> appetitive synapses, pain the identity -> aversive ones (short traces); novel foods per life | done | Main 43.6 vs control 40.9 (+2.7 +-0.8), head to head +9.7 +-2.8. The pain teacher is used (92 when silenced, every seed); the taste teacher is not (104); plasticity as a whole 97. | |
| 1.6 reversal | learned weights relax toward inherited values; the novel types swap meaning mid-life | defined, not run | |
| x.hands (side) | pick, hold, then eat: held-item value cells and an ingest programme on the 1.1 brain | done (2 runs) | Adapts in ~10 generations with the extension (energy 65-67, as with eat on grasp); the plain brain adapted in one of two runs. Lesions: ingest 0, gate 55. | |
| 1.2 drives, 1.3 affect, 1.4 habituation | hunger and cold; serotonin and PDF; depression with dud bushes | parked side branch on 1.1 | Not used in their last runs (v9, v13; STAGE_LOG). |
| 2.1-2.5, x.td | tectum, pallium, basal ganglia, dopamine TD | defined on the new path, older design | To revisit: per-life pallium weights (ADR-017), holding before 2.6, delayed sickness with the prediction error. |
| 2.6-2.10, 3.x-5.x | actor, curiosity, hippocampus, cerebellum; simulating, mentalizing, speaking | planned | |

## Next steps: finalizing chapter 1 (plan agreed 2026-10-03)

Per stage, three seeds: **used** (silencing the new cells costs energy in every seed), **not worse** (main at or
above control within one standard error, summary and head to head), **no regression** (the population does as well
in the previous stage's world as the previous population).

1. Stage 1.6 (reversal): three seeds running. Seed 0: plasticity 100, taste teacher 104, pain teacher 98. If the
   other seeds agree, learning is frozen at 1.5 and the taste teacher is removed (appetitive learning returns with
   the reward prediction error in chapter 2).
2. Unpark drives, affect and habituation, **appended after the learning stages**, each rebuilt on today's
   architecture with one design pass (generation-0 measurement of what the mechanic costs, then three seeds). A
   module that stays unused is left out of the frozen brain, unless later stages need the brain to have developed
   with it (the need state for reward valuation, the modulator systems); such a module is kept and worked on.
3. Final trim: the idle ganglion (decide after drives), designed strengths moved to where evolution put them.
4. One clean rerun of the whole chain at the frozen settings, three seeds, plus a chapter exam (every stage's
   population in every earlier world).
5. Freeze: a configuration snapshot test per stage, stored final populations, a git tag, one reference table, a list
   of known side paths. Then the review of chapter 2.

Learning at its present strength is accepted as final if it is significant over seeds (user, 2026-10-03).

## Known issues and open points

- The ganglion (24 neurons) is unused in stage 1.1 in every seed.
- The pain-blind control of 1.1 does not evolve avoidance by look within 150 generations, though it could.
- Learned appetite for the staple spreads to the innately avoided poison type, which looks 80% like it (assay).
- Fitness is supply-limited once a population forages well (60% of bushes empty); accepted.
- USE and EAT cost no energy. Chapter 2 definitions are stale. The dashboard clips displayed weights at `w_max`.

Practical notes: one process runs a non-plastic stage at about 4-5 s per generation and a learning stage (2000
ticks) at 11-20 s; up to about five JAX processes in parallel (16 threads, ~300 MB each). A sleeping laptop pauses
runs, and background commands are stopped after two hours, so run long batches in pieces.
`scripts/probes/prune.py` keeps only the newest finished run per stage directory.

## Core and tools (done)

Brain (ADR-015, 016, 019, 020): rectified rates, Dale's law, hard-wiring with tunable strength, input-feature
selection, source and target sub-ranges, topographic projections, named modulators with receptors and persistence,
eligibility traces, gain, depression, decay, delta rule, k-WTA, lagging divisive normalisation, ordered propagation
within a tick (phases), plasticity computed on the plastic block only, genome remapping across layouts.
World: look-alike and per-life appearances, dud bushes, OHOL temperature and hot springs, skin-temperature
gradient sense, delayed sickness, taste, patches, variety bonus, mid-life rule switch, appearance noise, movement
and (optional) neural metabolic cost, regrowth tied to the metabolic rate, multi-clone OHOL slicing.
Evolution: one life per individual, siblings, elitism + tournament, recombination, per-projection plasticity and
strength genes.
Tools: stage runner and chain, replicate, summary, lesion(s), versus, respond, compare, dashboard (architecture
view with hard-wired projections drawn dark; a 64-agent sample of large populations), probes (`scripts/probes/`).
32 tests.

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
