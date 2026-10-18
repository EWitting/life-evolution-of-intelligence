# Status

One screen on where the project is. Plan: `docs/BRAIN_EVOLUTION.md`. **Frozen stages: `frozen/README.md`.** Details, numbers and run directories:
`docs/STAGE_LOG.md`. Newest decisions: `docs/DECISIONS.md` (ADR-017 to ADR-025). Last updated 2026-10-18 (pallium stage 2.2 run with three seeds, provisional; brain step 1.7x faster; before that 2026-10-17: affect redefined in the world of chapter 2 and frozen; 2.0 and 2.1 off the chain; learning rates mutate multiplicatively; 100 generations per stage).

## How a stage is judged

A stage succeeds when its circuit is **used**, so that later stages can build on it, and main is not clearly worse
than control (user, 2026-10-01). One life is mostly luck, so a design is checked in this order (ADR-022,
`scripts/probes/README.md`):

1. **Mechanism, brain only**: `scripts/probes/assay.py` (scripted experience, then actions in fixed situations).
2. **Generation 0, paired**: `scripts/probes/gen0.py` (parent population, mechanism on vs off, same worlds). A
   diagnostic, not a gate: a neutral or mildly negative result can still go on to evolution.
3. **Evolution**: main vs control, 64 genomes x 4 siblings, recombination, **100 generations** (user, 2026-10-17:
   long enough to tell whether a circuit is used, short of full convergence at about 160, STAGE_LOG v44), weight
   mutation 0.1, learning rates mutating multiplicatively (`eta_log_std` 0.3, `Stage.evolution`), three seeds:

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

## Stages (rerun of 2026-10-08, affect 2026-10-17; STAGE_LOG v32, v33, v43-v45)

Chain: 1.0 -> 1.1 -> 1.2h habituation -> 1.5 association -> 1.7 drives -> 1.8 affect -> chapter 2. The keys are
historical names, not positions; the book shows the stages by name only (no stage numbers, user 2026-10-08). Lives are 4000
ticks from the settling run of 1.0 on. Footing v25 (born a quarter full, food as a flow, a bite cost). Every stage
starts from the seed-0 run of its parent. Three seeds unless stated; lesion = fitness with the region silenced, %
of intact, per seed.

**Chapter 1 is frozen: 1.0, 1.1, 1.2h, 1.5, 1.7 (2026-10-08, tag `chapter-1-drives`) and 1.8 (2026-10-17, tag `chapter-1-v2`).** The first affect route (1.8old, 2.0, 2.1) stays frozen off the chain. Definitions and final populations are in
`frozen/` (`frozen/README.md`, `life/freeze.py`); `tests/test_frozen.py` fails when a frozen definition changes.
**Do not rerun these stages or their lineage** (user, 2026-10-08). A change that has to touch one is made by
remapping the stored populations and freezing again.

| stage | what it adds | status | finding |
|---|---|---|---|
| 1.0 steering | ganglion (exc + inh, lagging normalisation), evolved sensor -> motor reflexes; sees the outside world only; movement costs energy; the animal eats what it grasps | **frozen** (1 lineage) | From scratch at 1000 ticks (530-555), then settled at 4000 ticks: well-fed lifetime about 2100, 45% alive at the cap, moving 62% of ticks. |
| 1.1 valence | appetitive and aversive value cells (taste and pain enter the brain only here) acting on a contact-gated grasp programme; aversion turns away and blocks the bite | **frozen**, passes | Main 1139 +-40 vs control 176 +-6; re-evaluated 1479 vs 190; head to head +1919 +-155. Lesions: aversive 9 9 12, no_feed 17 10 13, grasp 90 71 88, appetitive 90 73 93, no_touch 99 91 73. Seeds 1 and 2 needed 100 generations (seed 0: 50). No control found avoidance by look. |
| 1.2h habituation | no new cells: every look input passes on its input minus its slow average (about 50 ticks), so what stays in view fades and what is unusual stands out. World: the 1.1 world plus a dud bush type (colour per life, never a berry) | **frozen**, passes | Main 1139 +-15 vs control 1073 +-35; re-evaluated 1361 vs 1272 (+89 +-65: 218, 42, 8). Adaptation switched off: 36, 55, 36% of intact. Biting at nothing 28% of ticks vs 62%. Without the dud the adaptation was used weakly (84%). No head to head (the change is in the inputs). |
| 1.5 association | `us_pain` teaches the look -> aversive synapses (plain: the adapted look already carries what sets a food apart); a world of only novel foods, half of them poison | **frozen**, passes | Main 592 +-20 vs control 354 +-19; re-evaluated 721 vs 360 (+361 +-66); head to head +520 +-120. Lesions: no_plasticity and us_pain 43 48 43. Poison share 0.13-0.15 vs 0.23-0.24. Learning rate evolved up (0.3 -> 0.60 in seed 0). Clearer than the old order (+175, head to head +71). Lives are short (about 1170 of 4000 in the recorded generation). |
| 1.7 drives | `cold` and `hungry` need cells; warmth by kinesis: `warm_seek` (run while the skin is cold), `rest` (stay where it is warm); `hungry` shuts the warmth mode; evolvable `hungry` -> feeding synapses. World: the 1.5 world made cold, hot springs (which do not block) | **frozen**, passes (user's decision on the head to head) | Main 386 +-14 vs control 309 +-18; re-evaluated 401 vs 341 (+60 +-37: 124, -4, 62). **Head to head -65 +-33** (-88, -108, 0): the control half does better next to main animals (432-532) than among its own (276-409). Lesions: hungry 26 27 18, rest 72 86 71, cold and warm_seek 89 94 86, plasticity 71 66 76. Main moves 45% of ticks vs 71%, bites at nothing 32% vs 12%. Either population alone at half density does 55-80% better (`density.py`): food is a race, resting animals leave food to roamers and gain by burning less. |
| 1.8 affect | two states that outlast their trigger and add up over bites (a leaky sum): pain -> `alarm` -> `flee` -> forward above about 1.5 poison bites; taste -> `fed` -> `serotonin` -> `settle` -> `rest` above about 9 good bites. Thresholds and fade times evolve. World: 256 x 256 (four times the area), density 0.3, berry types in patches 70% their own kind (320 patches of radius 6); chapter 1 eyes | **frozen**, passes | Re-evaluated (8 worlds) main 1222 1483 1501 vs control 1421 1150 1080 (+185 +-194); **head to head +216 +-135** (+32, +478, +138). Lesions: flee 86 77 88, fed 94 89 86, settle 86 89 83, rest 62 39 51, hungry 24 10 21. Not used in this world: plasticity 98 105 100, cold and warm_seek 102 100 102. Learning rates kept (0.23, 0.11, 0.13; control 0.40, 0.24, 0.13). 100 generations, multiplicative learning-rate mutation. |
| side paths | 1.1l, 1.5c (centred association before habituation), 1.5f, 1.6h, 1.6 reversal, x.hands, x.td, old 1.2-1.4 | defined, not on the path | Listed with reasons in `frozen/README.md`. |
| first affect route | 1.8old (states released at once, one berry type per patch; learning unused 99 95 99) -> 2.0 (camera eyes in open country: learning back) -> 2.1 (the chapter 2 world, states that add up) | frozen, off the chain (2026-10-17) | The new 1.8 does this in one stage. Straight from drives with the additive mutation of learning rates, lineages set the rate to 0 and keep it there (`x.affect2`, 200 generations); multiplicative steps keep it (v43, v44). |
| 2.2 pallium | `pallium` (128 cells, k-WTA 2) reads the eye ahead (`vis+0.app*`, fixed random weights per genome); pain teaches `pallium -> valence_av`; adaptation strength evolves (`Genome.adapt`). World: the 1.8 world with XOR looks at similarity 0.5 (poison ++ and --, good +- and -+) | **provisional**, not frozen | Head to head +322 +-106 (+220, +211, +535); re-evaluated +79 +-66. Lesions (8 worlds): pallium 91 101 99, no_plasticity 94 102 100, flee 94 89 98, settle 83 85 88. Lab (`xor_meals`, pallium only): poison refused more than good in seeds 0 and 1, not in 2; look alone shies from all four. Evolution turns pallium lessons down through the rule genes (A, C) where they cost good food (STAGE_LOG v46-v48). |
| chapter 2, next | pallium -> dopamine critic -> hands -> basal ganglia with actor -> curiosity | planned (ADR-025) | The definitions `old2.1`-`old2.5` and `x.td` in `stages.py` are the first design on the reversal brain; each stage is defined again when its turn comes. |
| later, 3.x-5.x | pallium clustering, hippocampus, cerebellum, uncertainty modulators; simulating, mentalizing, speaking | planned | |

## Next steps

1. **Resume here: the pallium stage (2.2) is provisional** (STAGE_LOG v46-v48, book page published). The pallium
   separates XOR looks in real lives (an encounter's active cells were active at a lesson of the same poison 62%,
   other poison 1%, good 9%), and main beats control in shared worlds in every seed, but the pallium is used in the
   world in one seed of three. Why lessons pay little: pain and the alarm already end a poison visit after a bite or
   two, so a lesson saves the first bite at the next bush of that kind; and evolution damps the aversive cells when
   a bush is ahead (`vis*.hit/near -> valence_av` about -1.1, from -0.7 in 1.8), and turns the lessons down through
   the rule genes where they cost good food (A -0.85 at similarity 0.6; C -0.27 in seed 2 at 0.5). User
   (2026-10-18): never switch evolution off; fix what makes evolution turn a feature down (start values and real
   flaws of the genetic algorithm may change). Probes: `lesson_curve.py`, `lesson_code.py`, `assay_xor.py`,
   `pallium_real.py`, `learn_worth.py`, `gen0.py OFF=`. Next: make a learned lesson worth more in this world (what
   a known poison costs beyond its first bite), then three seeds again, then freeze. After that: WSL2 and the GPU.
   Rules (STAGE_LOG v40-v44): weight mutation 0.1 for warm-started runs; learning rates mutate multiplicatively
   (with the additive step an idle rate is trapped at 0); 100 generations; compare populations and lesions in 16
   shared worlds with standard errors (`trial_lesion.py`), never by the training curve (new worlds every
   generation); at most three JAX processes with the laptop in use, started a few minutes apart; check older
   circuits with lesions after every stage.
2. Cheaper evaluations (user, 2026-10-12): the export runs in parallel (`--jobs`), prints a line per finished
   task and silences only the regions of the stage and its parent (`--lesions recent`, default); four worlds for a
   routine check (`--worlds 4`), seed 0 first. A stage run writes its 150 MB dashboard only with `--dashboard`.
3. Optional, out of curiosity: the frozen populations in earlier worlds (a sanity check, not a criterion).
4. Rules of work (user): cheap trials first (one stage, one seed, generation 0 where possible); look inside a
   finished run before starting the next (`scripts/probes/README.md`, step 4); no new mechanisms without the
   user; **no reruns of frozen stages**. The chapter exam (every population in every earlier world) is dropped as
   a criterion: later populations carry equipment for things earlier worlds lack. A sanity check at most.
5. Trim (2026-10-08): nothing is removed; every cell is used somewhere on the chain.
6. Book to-do (user, 2026-10-17): a "Watching it work" strip for the drives page (a fed animal resting at a hot
   spring) and the affect page (an animal fleeing after a poison bite; wait until the current work on affect is
   done). Each needs an `@extra` in `life/book.py` that picks the moment with `life_strip`.

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
- The learning world is new every generation, so fitness swings between generations (sd about 290 in 1.5): read
  trends over blocks of generations with their standard error, not from single blocks.
- The first chapter 2 design in `stages.py` (`old2.x`) is stale (ADR-025). The dashboard clips displayed weights at `w_max`.
- `runs/` was cleaned on 2026-10-12: side-path runs and all `dashboard.html` files are gone (light dashboards come from the recordings). Again on 2026-10-18: all `x_*` trials except `x_affect_ch2world_sim05*`, superseded runs inside chain stage directories, and the runs of 1.8old, 2.0 and 2.1 (their populations are in `frozen/`; `x.direct` and `x.eyes0` would need a run restored from there). Left: the runs of the chain 1.0-1.8 and the `sim05` trial.

Practical notes: one process runs a non-plastic stage at about 4-5 s per generation of 1000 ticks and a stage with
4000-tick lives at 15-40 s (an unrecorded simulation stops when every animal is dead); how many JAX processes fit in parallel depends on free memory, so on whether the laptop is in use for other things (five were killed for low memory on 2026-10-08 with VS Code and Chrome open; six or seven ran with them closed): check free memory and the size of one running process before starting a batch. A sleeping laptop pauses
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
strength genes, an adaptation-strength gene (`Genome.adapt`, from the pallium stage on).
Tools: stage runner and chain, replicate, summary, lesion(s), versus, respond, compare, dashboard (architecture
view with hard-wired projections drawn dark; a 64-agent sample of large populations), probes (`scripts/probes/`).
52 tests.

## Future goals (beyond the current chapters)

- **Continuous life cycle** (for chapters 4-5, mentalizing and speaking): birth near a parent, aging, lineages,
  overlapping generations, instead of generational resets. The `parent` field already exists in the world state.
- Bigger OHOL slices: stones, kindling, fire, cooking (needs `_LA` last-use-actor transitions and leftovers).
- Moving objects/animals (OHOL `move` transitions) for predators, prey and interception (cerebellum, alarm calls).
- Communication: VOCALIZE with a small vector (sound channel widening); a GIVE action for altruism.
- Scale: GPU runs in WSL2 (`docs/SETUP.md`), larger populations (selection is noisy with 64 agents and one life).
- Several seeds per stage comparison.
- **Self-adaptive mutation** (user, 2026-10-17): step sizes as genes (one for weights, one for learning rates per
  genome, mutating multiplicatively, as in evolution strategies), so that mutation shrinks once a population has
  settled. Only with much larger populations and longer runs: with 64 genomes, one life each and recombination
  the signal on a step-size gene is weak, and noisy selection favours small steps too early. Test first on the
  recovery benchmark (STAGE_LOG v20).

## Known simplifications to revisit

- `_LA` (last use of actor) transitions are skipped; tools never wear out.
- `move` transitions (animals) are skipped; nothing moves except agents.
- Eating drops any leftover object (bowl, plate) instead of keeping it in hand.
- Cell interactions are resolved sequentially by agent index (deterministic but biased toward low indices).
- Sound uses an N x N distance matrix; fine below ~5k agents.
- World generation ignores biomes: `map_chance` is used as a flat spawn weight; temperature has one ambient value.
- Object appearance is a hash (ADR-006) except for designed look-alikes; no sprite-derived appearance yet.
- A berry type's "colour" is not one shared look: `lookalike_appearance` and the per-life looks add the same
  coefficients in a different basis for the bush, the berry and the empty bush, so a full bush and its own empty
  bush are no more alike than chance (cosine -0.6 to +0.8 between types, found 2026-10-08 while proofreading the
  book). Part of the frozen worlds, so it stays for chapter 1; the animals bite at empty bushes because empty
  bushes as a class look enough like food. Worth fixing in a chapter 2 world if a stage needs "same colour".
- Rates are updated synchronously once per tick; one tick is behavioural time (~1 s), so "STDP" is BTSP-like.
