# Status

One screen on where the project is. Plan: `docs/BRAIN_EVOLUTION.md`. Details, numbers and run directories:
`docs/STAGE_LOG.md`. Newest decisions: `docs/DECISIONS.md` (ADR-017 to ADR-024). Last updated 2026-10-07.

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

4. **Book**: when the three seeds of a stage are finished, update its page in the interactive book (`book/`,
   workflow in `book/README.md`) and commit the data file:

       python -m life.book export <key> --evaluate     book/data/<key>.json: curves, summary, lesions, head to head
       python -m life.book dashboards <key>            light dashboards of seed 0 (not committed)

   `--evaluate` runs the lesions, the head to head and the re-evaluation itself (one JAX process, about 20
   minutes for 1.1), so `stages summary/versus/lesions` need not be run separately for the book. Tables, graphs
   and diagrams on the page follow the data file; the text of a page is written by hand when the stage is frozen.

**Fitness = well-fed lifetime from the first meal** (ADR-024; v19-v22 used energy acquired). Cells that lesions
show unused in every seed are removed.

## Stages (v24, 2026-10-07)

Chain: 1.0 -> 1.1 -> 1.5, then drives -> affect -> habituation -> chapter 2. 1.0 and 1.1 are rerun born a quarter
full (v24); **1.5 is on hold** (see next steps). Findings marked v21/v22 are from the energy-acquired fitness. Three seeds unless stated;
lesion = fitness with the region silenced, % of intact, mean over seeds.

| stage | what it adds | status | finding |
|---|---|---|---|
| 1.0 steering | ganglion (exc + inh, lagging normalisation), evolved sensor -> motor reflexes; sees the outside world only; movement costs energy; the animal eats what it grasps | rerun v24 (1 seed) | Evolves from scratch born a quarter full: well-fed lifetime 630, 174 of 256 alive at the cap (ceiling, as born full). |
| 1.1 valence | appetitive and aversive value cells (taste and pain enter the brain only here) acting on a contact-gated grasp programme; aversion turns away and blocks the bite | **passes** (v24, three seeds) | Main 519 +-18 vs control 394 +-94, head to head +290 +-105. Lesions: aversive 35, no_feed 41, ganglion_e 17, appetitive 84, grasp 81, no_touch 79. The control of one seed evolved avoidance by look and equals main. |
| 1.5 association | pain teaches the identity -> aversive synapses about what was just bitten (short trace); novel foods per life | **on hold**: the rule fails on paper | v24, one seed: main 434 vs control 455. The pain-only rule cannot separate good from poison (suspicion spreads to the staple and is never taken back), and the edible food is gone by mid-life (STAGE_LOG, 2026-10-07). v21, two teachers: head to head +9.7 +-2.8. |
| drives (old 1.2) | `cold` and `hungry` need cells; warmth by kinesis: `warm_seek` (run while the skin is cold), `rest` (stay where it is warm); `hungry` shuts the warmth mode: forage when hungry, look after warmth when fed. Senses: body and skin temperature. World: cold, few hot springs | designed, generation-0 probe done | Generation 0: well-fed lifetime 438 vs 266 for the cold-blind brain (+65%), body temperature 0.44 vs 0.37 (STAGE_LOG v23). To do: stage definition, three seeds, lesions. |
| affect (old 1.3) | serotonin (dwell) and PDF (roam) states | to redesign on drives | Unused in its last run (v13, old architecture). Kept if at all possible: later stages need the modulator systems. |
| habituation (old 1.4) | depression on the identity -> appetitive synapses; dud bushes | one design pass | Unused in its last run (v13). |
| 1.6 reversal | learned weights relax toward inherited values; the novel types swap meaning mid-life | done, **left out** | v22: main 42.8 vs control 41.5, head to head -6.5 +-1.6. Plasticity 101, taste teacher 105, pain teacher 97. |
| x.hands (side) | pick, hold, then eat: held-item value cells and an ingest programme on the 1.1 brain | done (2 runs) | v21: adapts in ~10 generations with the extension (energy 65-67, as with eat on grasp); the plain brain adapted in one of two runs. Lesions: ingest 0, gate 55. |
| 2.1-2.5, x.td | tectum, pallium, basal ganglia, dopamine TD | defined, older design | To revisit: per-life pallium weights (ADR-017), holding before 2.6, delayed sickness and appetitive learning with the prediction error. |
| 2.6-2.10, 3.x-5.x | actor, curiosity, hippocampus, cerebellum; simulating, mentalizing, speaking | planned | |

## Next steps: finalizing chapter 1 (plan agreed 2026-10-03)

Per stage, three seeds: **used** (silencing the new cells costs fitness in every seed), **not worse** (main at or
above control within one standard error, summary and head to head), **no regression** (the population does as well
in the previous stage's world as the previous population).

1. **Resume here: three proposals wait for the user's decision** (STAGE_LOG, 2026-10-07):
   (a) food as a flow: bushes start at random points of their regrowth cycle and grow berries back one at a time
   (now every bush starts full and regrows after 960 ticks, so the stock is raced down in about 500 ticks and most
   animals starve); (b) a full stomach blocks the bite, as a reflex in the 1.1 grasp programme; (c) 1.5 on the
   prediction-error rule (dw = eta * (pain - aversive activity) * look) with a poison cost of 3. (a) and (b) change
   the footing of every stage, so 1.0 and 1.1 are rerun after them. Check a rule first with
   `scripts/probes/classify.py`, then the assay, then generation 0.
2. Stage 1.1 passes born a quarter full, and its appetitive side is now used a little (84). Decide whether drives
   come directly after 1.1.
3. Drives: stage definition from the probe circuit, three seeds, lesions of `cold`, `hungry`, `warm_seek`, `rest`.
4. Affect, redesigned on the drives stage; then one design pass for habituation. An unused module stays out of the
   frozen brain unless later stages need the brain to have developed with it.
5. Final trim: the idle ganglion, designed strengths moved to where evolution put them.
6. Chapter exam (every stage's population in every earlier world), then freeze: a configuration snapshot test per
   stage, stored final populations, a git tag, one reference table, a list of known side paths. Then chapter 2.

## Known issues and open points

- The pain-blind control of 1.1 does not evolve avoidance by look within 150 generations, though it could.
- Learned appetite for the staple spreads to the innately avoided poison type, which looks 80% like it (assay).
- Fitness is supply-limited once a population forages well (60% of bushes empty); accepted.
- The food is a stock, not a flow: all bushes start full and regrow after 960 ticks (480 in 1.0); it is eaten down
  in about 500 ticks, then most animals starve. Read every lifetime and late-life result with this in mind.
- Most meals are taken on a nearly full stomach (59-84%, STAGE_LOG v22); nothing before the drives stage can
  sense hunger.
- The well-fed lifetime has a ceiling at the life cap; in 1.0 most agents reach it at any cap (STAGE_LOG v23).
- The birth reserve lasts 800 ticks at rest, so staying put is a good strategy under a survival fitness.
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
