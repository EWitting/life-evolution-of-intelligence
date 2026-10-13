# Status

One screen on where the project is. Plan: `docs/BRAIN_EVOLUTION.md`. **Frozen stages: `frozen/README.md`.** Details, numbers and run directories:
`docs/STAGE_LOG.md`. Newest decisions: `docs/DECISIONS.md` (ADR-017 to ADR-025). Last updated 2026-10-13 (dead animals dropped from the arrays; world size and patch purity tried).

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

## Stages (rerun of 2026-10-08, affect 2026-10-12; STAGE_LOG v32, v33)

Chain: 1.0 -> 1.1 -> 1.2h habituation -> 1.5 association -> 1.7 drives -> 1.8 affect -> chapter 2. The keys are
historical names, not positions; the book shows the stages by name only (no stage numbers, user 2026-10-08). Lives are 4000
ticks from the settling run of 1.0 on. Footing v25 (born a quarter full, food as a flow, a bite cost). Every stage
starts from the seed-0 run of its parent. Three seeds unless stated; lesion = fitness with the region silenced, %
of intact, per seed.

**Stage 2.1 is frozen (2026-10-12).** **Chapter 1 is frozen: 1.0, 1.1, 1.2h, 1.5, 1.7 (2026-10-08, tag `chapter-1-drives`) and 1.8 (2026-10-12, tag `chapter-1`).** Definitions and final populations are in
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
| 1.8 affect | two states that outlast their trigger, each a modulator released at once and cleared slowly: pain -> `alarm` -> `flee` -> forward; taste -> `fed` -> `serotonin` -> `settle` -> `rest`. Fade times are genes (`Genome.clear`). World: the drives world with every berry type in its own patches (`patch_types`) | **frozen**, passes | Main 769 +-33 vs control 485 +-38 (last 50 generations of 40: the whole run); re-evaluated 919 vs 625 (+294 +-5: 297, 285, 301); head to head +482 +-112 (604, 258, 583). Lesions: fed 63 54 66, settle 63 54 64, flee 81 86 89, rest 41 36 37, hungry 41 54 41. **No longer used in this world: cold and warm_seek 101 105 107, plasticity 99 95 99.** Fade times stayed near the design (seed 0: alarm 8 ticks, serotonin 84). Both populations were still rising at generation 40. |
| side paths | 1.1l, 1.5c (centred association before habituation), 1.5f, 1.6h, 1.6 reversal, x.hands, x.td, old 1.2-1.4 | defined, not on the path | Listed with reasons in `frozen/README.md`. |
| chapter 2 world | the drives world without patches, more food (density 0.3), twice the area (181 x 181); looks as in chapter 1 | chosen (user, 2026-10-12) | Of five worlds tried only this one gives learning a job again (STAGE_LOG v34). Twice the area because crowding compressed gains (evolved 788 at 128, 1247 at 181, 1722 at 256); it is not a shortage of food (v37). A shared colour for a type's bush and empty bush cost 42% and is dropped. |
| 2.1 camera eyes | 9 eye columns, range 6; no new cells; the new columns start as the mean of their neighbours, all eye synapses times 5/9 | **frozen** (three seeds, no control) | Re-evaluated 1085 1109 1154 (start population 676); plasticity off 53 44 41%, flee 83 96 99, fed 93 100 100; poison share 0.07-0.10 against 0.18-0.20 without learning. The tectum has no stage (user): STAGE_LOG v35, v36. |
| chapter 2, next | pallium -> dopamine critic -> hands -> basal ganglia with actor -> curiosity | planned (ADR-025) | The definitions `old2.1`-`old2.5` and `x.td` in `stages.py` are the first design on the reversal brain; each stage is defined again when its turn comes. |
| later, 3.x-5.x | pallium clustering, hippocampus, cerebellum, uncertainty modulators; simulating, mentalizing, speaking | planned | |

## Next steps

1. **Resume here: the pallium** (ADR-025). Stage 2.1 (camera eyes) is frozen and has its book page. First probe
   for the pallium: the assay of the association stage with the learned synapses reading the pallium. What it has
   to fix is measured (STAGE_LOG v37, v38): the animals find food at once and die of repeated poison bites (30 in a
   full life; a perfect learner takes 2), and they strip the good bushes near them by biting on a full stomach.
   Before it, two things were tried (STAGE_LOG v40, 2026-10-13), both waiting for the user's decision:
   (a) **World size.** Frozen 2.1 seed 0 at 181 / 256 / 362 per side: start population 676 / 765 / 815, evolved
   1015 / 1569 / 1924, at the life cap 19 / 33 / 45% (no problem as long as the well-fed lifetime tells animals
   apart, user). Crowding still hides two thirds of the stage's gain at 181 and about a quarter at 256. Size costs
   no time. If a larger world is taken, 2.1 is run and frozen again in it (the user's decision; about 15 minutes
   for three seeds in parallel) and its book page exported again. **Open before that:** the trial `x.c256` (the
   frozen 2.1 population, 40 more generations in the open world at 256) ended worse than it began (-325 +-100 in
   16 shared worlds); find out why first (a second seed; the proper 2.1 from the affect population at 256
   compared with the frozen one in the same worlds, `scripts/probes/before_after.py`).
   (b) **Patch purity.** No purity gives both learning and the stay-after-a-meal state a job: `fed` off is 92-101%
   of intact from 0 to 80% purity (generation 0 and after 40 generations, trials `x.c60`, `x.c80`), and pays only
   in pure patches, where learning does not. Learning (70-87%) and the alarm (`flee` off about 80% at 256) pay in
   every mixed world. Not tried: a world with both kinds of ground (some pure patches in open, mixed country).
   Done: the simulation no longer costs agents x cells, and an unrecorded simulation drops the dead from its
   arrays (about half the time, identical numbers). Measured and left: a fuller stomach at birth (little change).
   Compare two populations in 16 shared worlds, not four.
2. Cheaper evaluations (user, 2026-10-12): the export runs in parallel (`--jobs`), prints a line per finished
   task and silences only the regions of the stage and its parent (`--lesions recent`, default); four worlds for a
   routine check (`--worlds 4`), seed 0 first. A stage run writes its 150 MB dashboard only with `--dashboard`.
3. Optional, out of curiosity: the frozen populations in earlier worlds (a sanity check, not a criterion).
4. Rules of work (user): cheap trials first (one stage, one seed, generation 0 where possible); look inside a
   finished run before starting the next (`scripts/probes/README.md`, step 4); no new mechanisms without the
   user; **no reruns of frozen stages**. The chapter exam (every population in every earlier world) is dropped as
   a criterion: later populations carry equipment for things earlier worlds lack. A sanity check at most.
5. Trim (2026-10-08): nothing is removed; every cell is used somewhere on the chain. `docs/RERUN_PLAN.md` is
   carried out and kept for reference.

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
- `runs/` was cleaned on 2026-10-12: side-path runs and all `dashboard.html` files are gone (light dashboards come from the recordings).

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
strength genes.
Tools: stage runner and chain, replicate, summary, lesion(s), versus, respond, compare, dashboard (architecture
view with hard-wired projections drawn dark; a 64-agent sample of large populations), probes (`scripts/probes/`).
50 tests.

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
- A berry type's "colour" is not one shared look: `lookalike_appearance` and the per-life looks add the same
  coefficients in a different basis for the bush, the berry and the empty bush, so a full bush and its own empty
  bush are no more alike than chance (cosine -0.6 to +0.8 between types, found 2026-10-08 while proofreading the
  book). Part of the frozen worlds, so it stays for chapter 1; the animals bite at empty bushes because empty
  bushes as a class look enough like food. Worth fixing in a chapter 2 world if a stage needs "same colour".
- Rates are updated synchronously once per tick; one tick is behavioural time (~1 s), so "STDP" is BTSP-like.
