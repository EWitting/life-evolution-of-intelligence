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

## Stages (v25, 2026-10-07)

Chain: 1.0 -> 1.1 -> 1.1l (long lives) -> 1.5 -> 1.6h (habituation) -> 1.7 drives (trial) -> affect -> chapter 2.
Footing v25 (born a quarter full, food as a flow, a bite cost) for 1.0 and 1.1; from 1.1l on lives are 4000
ticks. Three seeds unless stated; lesion = fitness with the region silenced, % of intact, mean over seeds. Findings
marked v21/v22 are from the energy-acquired fitness.

| stage | what it adds | status | finding |
|---|---|---|---|
| 1.0 steering | ganglion (exc + inh, lagging normalisation), evolved sensor -> motor reflexes; sees the outside world only; movement costs energy; the animal eats what it grasps | rerun v25 (1 seed) | Evolves from scratch: well-fed lifetime 608, 183 of 256 alive at the cap (ceiling). Supply steady: 18-25% of bushes full all life. |
| 1.1 valence | appetitive and aversive value cells (taste and pain enter the brain only here) acting on a contact-gated grasp programme; aversion turns away and blocks the bite | **passes** (v25, three seeds) | Main 452 +-15 vs control 186 +-13, head to head +401 +-35. Lesions: aversive 40, no_feed 45, ganglion_e 63, ganglion_i 64, appetitive 90, grasp 89, no_touch 94. v24 (every bush full at birth): main 519 vs control 394; two of three controls evolved avoidance by look. |
| 1.1l long lives | nothing new in the brain: the 1.1 animals under lives of 4000 ticks, an adaptation step | done (1 lineage) | Lifetime 1340 -> 2220 of 4000, alive at the cap 9 -> 97 of 256, freezes 29% -> 7% of ticks. In a life of 1000 ticks (1.25 stomachs) an animal that ate once and stood still reached the cap. |
| 1.5 association | pain teaches the look -> aversive synapses what sets the bitten food apart (the look minus each look input's slow average); a world of only novel foods, half of them poison | **passes** (v29, three seeds) | Main 427 +-24 vs control 320 +-3; re-evaluated 483 vs 308 (+175 +-42); head to head +71 +-50. Lesions: no_plasticity 51, us_pain 51 (57 42 54), aversive 15, no_feed 16. Poison share 0.19 vs 0.25. No change over 40 generations in main or control. Lives are short (770 of 4000). |
| 1.6h habituation | no new cells: every look input passes on its input minus its slow average (about 50 ticks), so what stays in view fades and what is unusual stands out; the learned synapses of 1.5 are plain again | **passes** (v30, three seeds) | Main 500 +-18 vs control 400 +-25; re-evaluated 597 vs 518 (+79 +-57). Adaptation switched off: 34, 31, 35% of intact. Learning: no_plasticity 40. Standing and biting at nothing 19% of ticks vs 36% in the control. Head to head not valid for this stage. |
| 1.7 drives (old 1.2) | `cold` and `hungry` need cells; warmth by kinesis: `warm_seek` (run while the skin is cold), `rest` (stay where it is warm); `hungry` shuts the warmth mode; evolvable `hungry` -> feeding synapses. World: the 1.5 world made cold, hot springs (which no longer block) | **trial, one seed** | Main 407 vs control 332 (generations 30-39), body 0.40 vs 0.35. Lesions: hungry 26, plasticity 73, rest 86, cold 95, warm_seek 95. Cold costs 44% at generation 0. To do: decide how hunger acts on feeding, then three seeds. |
| affect (old 1.3) | serotonin (dwell) and PDF (roam) states | to redesign on drives | Unused in its last run (v13, old architecture). Kept if at all possible: later stages need the modulator systems. |
| 1.6 reversal | learned weights relax toward inherited values; the novel types swap meaning mid-life | done, **left out** | v22: main 42.8 vs control 41.5, head to head -6.5 +-1.6. Plasticity 101, taste teacher 105, pain teacher 97. |
| x.hands (side) | pick, hold, then eat: held-item value cells and an ingest programme on the 1.1 brain | done (2 runs) | v21: adapts in ~10 generations with the extension (energy 65-67, as with eat on grasp); the plain brain adapted in one of two runs. Lesions: ingest 0, gate 55. |
| 2.1-2.5, x.td | tectum, pallium, basal ganglia, dopamine TD | defined, older design | To revisit: per-life pallium weights (ADR-017), holding before 2.6, delayed sickness and appetitive learning with the prediction error. |
| 2.6-2.10, 3.x-5.x | actor, curiosity, hippocampus, cerebellum; simulating, mentalizing, speaking | planned | |

## Next steps: finalizing chapter 1 (plan agreed 2026-10-03)

Per stage, three seeds: **used** (silencing the new cells costs fitness in every seed), **not worse** (main at or
above control within one standard error, summary and head to head), **no regression** (the population does as well
in the previous stage's world as the previous population).

1. **Resume here: a complete rerun of chapter 1 is agreed** (user, 2026-10-08), all stages, three seeds, as the
   milestone that closes the learning work, provided the findings carry over to a new lineage. Decided:
   - **Order: habituation before association is allowed**, so 1.0 -> 1.1 -> habituation (adapting look inputs;
     control the 1.1 brain) -> association (plain learned synapses on adapted inputs; control the habituation
     brain) -> drives. A one-seed check of habituation on the 1.1 brain in the 1.1 world was started
     (`stages chain 1.2h 1.2h`, runs/s1_2_habituation): read it first. The risk it tests: in the familiar world
     long lives alone already cut the standing still to 7% of ticks, so habituation may have little left to win
     there and could fail "used" before the novel-food world.
   - **1.1l is not meant to stay a stage.** It exists because 1.0 and 1.1 were evolved with lives of 1000 ticks.
     In the rerun, long lives start earlier: 1.0 from scratch at 1000 ticks (nearly everything dies young at
     first), its settling run at 4000 ticks, and 4000 from then on. Check with `freeze.py` and `supply.py` that
     the settled 1.0 population uses a long life before going on.
   - **Hunger** stays one inhibitory cell with the evolvable path onto feeding; no second pathway (user: with the
     path present, low hunger is close enough to "stop biting when fed", and the results are good).
   - **Drives over three seeds** is part of the rerun.
   - **Short lives** (about 900 of 4000 ticks) are accepted as long as there is time to learn and fitness tells
     animals apart. No tuning now. If it is tuned later: first find out why they starve; born full is acceptable
     when a stomach is a quarter of a life or less (user: a full stomach was only a problem when it lasted a
     lifetime).
   - Before the rerun, list what could stop working with a new lineage and check the cheap ones at generation 0;
     run the chain one seed first, then the other seeds, then the evaluations and the book.
2. Rules of work (user, 2026-10-07): cheap trials first (one stage, one seed, generation 0 where possible), full
   chain and three seeds only when a design looks final; look inside a finished run before starting the next
   (`scripts/probes/README.md`, step 4); no new mechanisms without the user.
3. Affect, redesigned on the drives stage. An unused module stays out of the frozen brain unless later stages need
   the brain to have developed with it.
4. Whether hunger should come before the learning stage (it would let the 1.5 animals use a long life) is open.
5. Final trim: the idle ganglion, designed strengths moved to where evolution put them.
6. Chapter exam (every stage's population in every earlier world), then freeze: a configuration snapshot test per
   stage, stored final populations, a git tag, one reference table, a list of known side paths. Then chapter 2.

## Known issues and open points

- The pain-blind control of 1.1 does not evolve avoidance by look within 150 generations, though it could.
- Learned appetite for the staple spreads to the innately avoided poison type, which looks 80% like it (assay).
- Fitness is supply-limited once a population forages well (60% of bushes empty); accepted.
- Until v24 the food was a stock (every bush full at birth, regrowth as long as a life): eaten down in about 500
  ticks, then famine. Results before v25 about lifetime or late life carry that. Check `scripts/probes/supply.py`
  when a world changes.
- Half to two-thirds of the bites are on a stomach over 90% full; nothing before the drives stage can sense the
  stomach. In drives, give `hungry` an evolvable connection onto the grasp programme (not hard-wired).
- Delayed sickness (planned for chapter 2) blames the wrong bush: the animal bites poison, turns to a good bush,
  the sickness starts, and whatever it is looking at then is taught as bad (seen in v11-v13; user, 2026-10-07).
  A longer trace does not solve it. Any design with delay must show in `classify.py` and the assay that the blame
  lands on what was eaten (for example a trace of what was in the mouth, not of what is in view).
- The well-fed lifetime has a ceiling at the life cap; in 1.0 most agents reach it at any cap (STAGE_LOG v23).
- A stomach lasts 800 ticks at rest: stages 1.0 and 1.1 still have lives of 1000 ticks, in which standing still
  after a meal reaches the cap; 1.1l removes that before any later stage.
- The learning world is new every generation, so fitness swings between generations (sd about 290 in 1.5): read
  trends over blocks of generations with their standard error, not from single blocks.
- Chapter 2 definitions are stale. The dashboard clips displayed weights at `w_max`.

Practical notes: one process runs a non-plastic stage at about 4-5 s per generation of 1000 ticks and a stage with
4000-tick lives at 15-40 s (an unrecorded simulation stops when every animal is dead); up to about five JAX processes in parallel (16 threads, ~300 MB each). A sleeping laptop pauses
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
