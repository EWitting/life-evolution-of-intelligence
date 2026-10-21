# Status

One screen on where the project is. Plan: `docs/BRAIN_EVOLUTION.md`. **Frozen stages: `frozen/README.md`.** Details, numbers and run directories:
`docs/STAGE_LOG.md`. Newest decisions: `docs/DECISIONS.md` (ADR-017 to ADR-027). Last updated 2026-10-20 (the pallium closes chapter 1 and chapter 2 starts with the critic; the hippocampus, map only, closes chapter 2: ADR-027). Before that 2026-10-18 night (pallium stage 2.2 rerun after a remap fix, passes in mixed patches, three seeds, not yet frozen; runs on the GPU in WSL2, about 10x faster; before that 2026-10-17: affect redefined in the world of chapter 2 and frozen; 2.0 and 2.1 off the chain; learning rates mutate multiplicatively; 100 generations per stage).

## How a stage is judged

A stage succeeds when its circuit is **used**, so that later stages can build on it, and main is not clearly worse
than control (user, 2026-10-01). One life is mostly luck, so a design is checked in this order (ADR-022,
`scripts/probes/README.md`):

1. **Mechanism, brain only**: `scripts/probes/assay.py` (scripted experience, then actions in fixed situations).
   For learning stages: **lab tests**, textbook experiments from the literature (`life/labtests.py`,
   `scripts/probes/lab_conditioning.py`: acquisition, extinction, blocking), done before generation 0 and
   evolution, with results in the book (BRAIN_EVOLUTION principle 4).
2. **Generation 0, paired**: `scripts/probes/gen0.py` (parent population, mechanism on vs off, same worlds). A
   diagnostic, not a gate: a neutral or mildly negative result can still go on to evolution.
3. **Evolution**: main vs control, 64 genomes x 4 siblings, recombination, at most **100 generations** (user,
   2026-10-17: long enough to tell whether a circuit is used, short of full convergence at about 160, STAGE_LOG v44),
   weight mutation 0.1, learning rates mutating multiplicatively (`eta_log_std` 0.3, `Stage.evolution`), **2 episodes
   per generation** for new stages (`EVOLUTION_2EP`, user 2026-10-20), three seeds.
   **Smart early stopping is the default** (user, 2026-10-20): runs write `checkpoint.npz` every 10 generations; the
   agent running them reads the training curves and checkpoints (learning rates, recordings, quick lesions) and stops
   a run once the trends are clear and the main - control outcome is unlikely to change. Convergence is not required.
   Exception: a stage whose world changes over the generations (`Stage.schedule`, e.g. 2.3h: no reversal for the
   first 50) runs at least into the changed world far enough to judge it.

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

Chain: 1.0 -> 1.1 -> 1.2h habituation -> 1.5 association -> 1.7 drives -> 1.8 affect -> 2.2 pallium | chapter 2: 2.3c
critic (from cells) -> ... The keys are
historical names, not positions; the book shows the stages by name only (no stage numbers, user 2026-10-08). Lives are 4000
ticks from the settling run of 1.0 on. Footing v25 (born a quarter full, food as a flow, a bite cost). Every stage
starts from the seed-0 run of its parent. Three seeds unless stated; lesion = fitness with the region silenced, %
of intact, per seed.

**Chapter 1 is frozen: 1.0, 1.1, 1.2h, 1.5, 1.7 (2026-10-08, tag `chapter-1-drives`), 1.8 (2026-10-17, tag `chapter-1-v2`) and 2.2 pallium (2026-10-19, tag `stage-2.2`; moved into chapter 1 on 2026-10-20, ADR-027, key unchanged).** The first affect route (1.8old, 2.0, 2.1) stays frozen off the chain. Definitions and final populations are in
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
| 1.8 affect | two states that outlast their trigger and add up over bites (a leaky sum): pain -> `alarm` -> `flee` -> forward above about 1.5 poison bites; taste -> `fed` -> `serotonin` -> `settle` -> `rest` above about 9 good bites. Thresholds and fade times evolve. World: 256 x 256 (four times the area), density 0.3, berry types in patches 70% their own kind (320 patches of radius 6); chapter 1 eyes | **frozen**, passes | Re-evaluated (8 worlds) main 1246 1471 1508 vs control 1434 1148 1082 (+187 +-190); **head to head +178 +-66** (+82, +304, +147; re-evaluated after the remap fix, v51; before +216 +-135). Lesions: flee 86 77 88, fed 94 89 86, settle 86 89 83, rest 62 39 51, hungry 24 10 21. Not used in this world: plasticity 98 105 100, cold and warm_seek 102 100 102. Learning rates kept (0.23, 0.11, 0.13; control 0.40, 0.24, 0.13). 100 generations, multiplicative learning-rate mutation. |
| side paths | 1.1l, 1.5c (centred association before habituation), 1.5f, 1.6h, 1.6 reversal, x.hands, x.td, old 1.2-1.4 | defined, not on the path | Listed with reasons in `frozen/README.md`. |
| first affect route | 1.8old (states released at once, one berry type per patch; learning unused 99 95 99) -> 2.0 (camera eyes in open country: learning back) -> 2.1 (the chapter 2 world, states that add up) | frozen, off the chain (2026-10-17) | The new 1.8 does this in one stage. Straight from drives with the additive mutation of learning rates, lineages set the rate to 0 and keep it there (`x.affect2`, 200 generations); multiplicative steps keep it (v43, v44). |
| 2.2 pallium (chapter 1, last stage) | `pallium` (128 cells, k-WTA 2) reads the eye ahead (`vis+0.app*`, fixed random weights of mixed sign per genome); pain teaches `pallium -> valence_av` (start 0.2); adaptation strength evolves (`Genome.adapt`, main and control). World `W22`: the 1.8 world with XOR looks at similarity 0.5 (poison ++ and --, good +- and -+) and patches mixed evenly (purity 0.25) | **frozen** 2026-10-19, passes (`flee` and `fed` kept though unused here) | Main - control: last 50 generations +106 +-20 (+83, +145, +91); **head to head +229 +-31** (+184, +288, +214); re-evaluated apart, 16 worlds: +73 +-48 (-14, +151, +83). Lesions (16 worlds): pallium 90 88 92, no_plasticity 77 67 87, settle 96 94 91; flee and fed not used (97-101). Lab (`xor_meals`), bites poison / good after lessons, only the pallium learning 0.16 / 0.37 (before 0.38 / 0.39); only the look 0.22 / 0.23. Poison share of bites 0.14 early in a life, 0.08 late. A larger pallium (256, 512) does not lower the transfer to good bites (0.13 at 128). STAGE_LOG v46-v51. |
| 2.3 critic | dopamine critic as a formula: value cells `value_app` / `value_av` (3 + 3) on the pallium, one-tick copies, `us_taste`; one signed TD error `da` = taste x 0.3 - pain + 3 (0.5 V' - V) teaches the value cells and, negated (`da_neg`), every synapse pain taught (look and pallium -> `valence_av`), one tick late (`elig_delay`). Pallium wiring per population slot (`per_slot`, main and control). World `W23`: W22 with a reversal at tick 800 (poison and good XOR types swap) | frozen 2026-10-19, **off the chain since 2026-10-20** (replaced by 2.3h) (`value_av` kept though unused) | Main - control: last 50 generations +151 +-20; re-evaluated (64 worlds) +127 +-7 (+123, +119, +140); **head to head +177 +-50** (+213, +240, +79). Lesions (64 worlds): value_app 84 83 68, no_plasticity 83 83 83, pallium 83 82 80, us_taste 89 93 80, value_av 101 100 97 (the `_prev` lesions, 20-82, leave an error that never cancels: not a measure of use). Both populations switch at the reversal about equally (P(bite) on a former poison 0.85 vs 0.83 after it); main eats more throughout (172 vs 135) and lives longer. Lab: the error moves from the bites to the cue and turns positive when the pain is omitted; extinction 2.7 -> 1.8 (control 1.2 -> 1.2); no blocking (the second cue is not learned in either). The gap appears only after generation 40. STAGE_LOG v54-v57. |
| 2.3h critic | the 2.3 brain (formula critic) in a harsh world: one berry per bush worth one berry, a poison berry as costly as a good one is good, a quarter of the metabolic rate, born 0.75 full; the poison and good kinds swap at tick 800 from generation 50 on (`calm_first(50)`); aversion learning rates start x10; 2 episodes per generation | frozen 2026-10-20, **off the chain since 2026-10-21** (replaced by 2.3c) | Re-evaluated (16 worlds) main 1065 vs control 802, **+264 +-87** (+238, +128, +426); **head to head +359 +-76**. Lesions: no_plasticity 40 54 52, pallium 87 99 85, us_taste 87 93 90, **value_av 94 87 91 (used)**, value_app 105 101 109 (not used). After the switch the critic's first bite still favours the kinds good now in all three seeds (poison / good 0.06 / 0.14, 0.06 / 0.09, 0.08 / 0.11), the control's in one of three. Lab: aversion after 6 painful meals 12.0 (control 9.7), after 12 without pain 2.1 (control 9.7); a small blocking effect (second cue 4.8 vs 5.5; control 2.6 vs 2.8). STAGE_LOG v58-v65. |
| 2.3c critic | the critic built from cells: a dopamine cell with a resting rate (rate minus rest = the error), a tonic habenula carrying pain, inhibitory ventral striatal value cells (`value_app` D1-type rule, `value_av` D2-type) reaching dopamine fast through tonic pallidal cells (`ventral_pallidum`, `gp_habenular`) and slowly through slow synapses (`ProjectionSpec.lag`): TD, gamma about 0.9; 2.3h's harsh world and calm start | **frozen** 2026-10-21, passes; on the critic page | Re-evaluated (16 worlds) main 1086 vs control (pain teacher) 802, **+284 +-100** (+91, +337, +424); **head to head +798 +-65**. Lesions: no_plasticity 46 52 55, pallium 83 92 89, value_av 97 91 87 (used), value_app 101 106 105 (not used); dopamine 46 43 38, habenula 48 48 52, ventral_pallidum 51 50 53, gp_habenular 50 59 54 (tonic cells: a standing error, not a pure measure of use). Lab: extinction 10.0 -> -0.45 (control 9.7 -> 9.7; seed 2 overshoots to -2.9); error on the sixth painful meal: bush -0.38, bites -0.82. After the switch the critic's creatures take up the kinds that turned good (bites per creature 0.6 -> 1.0-1.2), the control's barely. Seed 2 resumed from its generation-79 checkpoint after a memory kill and run to 100. STAGE_LOG v66-v71. |
| chapter 2, next | dopamine critic (from cells) -> hands -> basal ganglia with actor -> curiosity -> hippocampus (map of places only) | planned (ADR-025, ADR-027) | The definitions `old2.1`-`old2.5` and `x.td` in `stages.py` are the first design on the reversal brain; each stage is defined again when its turn comes. |
| later, 3.x-5.x | pallium clustering, cerebellum, uncertainty modulators (parked in chapter 2); sequences and replay on the place map, simulating, mentalizing, speaking | planned | |

## Next steps

0. **Done (2026-10-20): the critic stage is 2.3h** (frozen, tag `stage-2.3h`, on the critic page; 2.3 off the chain).
   The road there (STAGE_LOG v58-v65): in the 2.3 world no creature judged a bush by its look (a first bite on
   every bush, the visit ended by the pain); the learned aversion was real but held below the aversive cells'
   threshold by inherited hit / near inhibition, a world lesson being one tick of pain; the reversal punished strong
   aversion learning, so evolution lowered it. A harsh world of small meals (poison as costly as food is good)
   makes learning indispensable (off: half the fitness lost), and fifty calm generations before the reversal let
   caution by look evolve first. Tools from this work: `scripts/probes/bridge.py` (record raw inputs, replay them,
   counterfactuals at real arrivals), `first_bite.py`, `bite_trace.py`, `rebalance.py`, `lab_mixed.py`,
   `lab_critic.py`; `Stage.schedule`; run checkpoints; poison per phase in dashboards.
1. **Done (2026-10-21): the critic is built from cells (2.3c)**, frozen, tag `stage-2.3c`, on the critic page (2.3h
   off the chain). The road (STAGE_LOG v66-v70): the simplest circuit (Rescorla-Wagner: value cells inhibiting the
   dopamine cell directly) reproduced the teacher but its predictions read a known poison in view as relief, so TD
   was needed already here; slow synapses (`ProjectionSpec.lag`) give the slow path; the outputs of tonic cells are
   not tuned (they set the dopamine cell's rest); path strengths per value cell (a 3x scale error found by regressing
   the error on the value cells in a recording; `cell_trace.py`). Probe at the arrival with recorded activations, not
   only the output signal (user). **Next: hands, then the basal ganglia with the actor** (BRAIN_EVOLUTION), reusing
   the dopamine cell, the D1/D2 rules and the pallidal relays. Open: `value_app` unused in this world; blocking only
   weakly; a sense of time since a cue; the look pathway generalises "bushes hurt" (about 80% of good bushes passed
   by, v68); an asymmetry between burst and dip learning (a gene, for the user to agree).
   Earlier (2.2 frozen, tag `stage-2.2`, `flee` and `fed` kept). Old chapter 2 definitions (`old2.x`, `x.td` in
   `stages.py`) are stale.
   **Noise settings for every new stage** (STAGE_LOG v52, user 2026-10-19): `Stage.evolution = EVOLUTION_NEW`
   (episodes 4: 4 siblings x 4 worlds = 16 lives per genome; learning rates multiplicative); frozen stages keep
   episodes 1. Measured at generation 0 of 2.2 (`noise_gen0.py`): one life is as much luck there as at the end of a
   run (repeatability with 4 lives 0.10, 16 lives 0.34, 64 lives 0.68); 16 lives about double the selection
   accuracy for about 2.3x the GPU time (about 20 min per run). One 512 x 512 world x 16 siblings or 16 worlds of
   128 x 128 x 1 sibling repeat no better (`small_world.py`). Comparisons: the spread between the 2.2 seeds is
   mostly evaluation noise; re-evaluate in 64 worlds (s.e. of main - control about 25, 5% at 2 s.e.; 16 worlds: 49)
   and pass `--worlds 64` to `life.book export`. Measure the GPU memory of the first episodes-4 run (how many fit
   side by side; 1.3-1.6 GB is used at idle by Windows).
   **Runs: use the GPU** (`docs/SETUP.md`): `wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/.../Life && bash
   scripts/gpu_queue.sh <job file> 3'` (job file lines: `<log name>|<stage arguments>`), logs in `runs/logs/`;
   book export and probes the same way with `XLA_PYTHON_CLIENT_PREALLOCATE=false ~/.venvs/life/bin/python ...`.
   At most two GPU runs at a time (2.0-2.5 GB of RAM each inside WSL, capped at 6 GB by `.wslconfig`, which drops
   its file cache when idle, `autoMemoryReclaim=dropCache`; measured 2026-10-20); they take turns for the memory-heavy recording at the end (`life.run.recording_lock`). One book export with `--evaluate --jobs 2` at a time (not covered by the lock), and nothing else heavy on Windows meanwhile; 4096 animals
   do not fit in 8 GB yet (the per-animal weight matrix `BrainState.w` could hold only the plastic block).
   Rules (STAGE_LOG v40-v44): weight mutation 0.1 for warm-started runs; learning rates mutate multiplicatively
   (with the additive step an idle rate is trapped at 0); 100 generations; compare populations and lesions in shared
   worlds with standard errors, never by the training curve (new worlds every generation); check older circuits
   with lesions after every stage. In patches of one type, learning and the lasting states do the same job and a
   lineage leans on one of them: a world gives two mechanisms each a job only if they answer different questions.
2. Cheaper evaluations (user, 2026-10-12): the export runs in parallel (`--jobs`), prints a line per finished
   task and silences only the regions of the stage and its parent (`--lesions recent`, default); four worlds for a
   routine check (`--worlds 4`), seed 0 first. A stage run writes its 150 MB dashboard only with `--dashboard`.
3. Optional, out of curiosity: the frozen populations in earlier worlds (a sanity check, not a criterion).
4. Rules of work (user): cheap trials first (one stage, one seed, generation 0 where possible); look inside a
   finished run before starting the next (`scripts/probes/README.md`, step 4); no new mechanisms without the
   user; **no reruns of frozen stages**. The chapter exam (every population in every earlier world) is dropped as
   a criterion: later populations carry equipment for things earlier worlds lack. A sanity check at most.
5. Trim (2026-10-08): nothing is removed; every cell is used somewhere on the chain.
6. Book: "Watching it work" strips for drives (`warm_rest`) and affect (`flight`) done (2026-10-19). Rewritten as a
   textbook the same day: background page, takeaway box, `verdict` and `bars` components, full results folded away,
   brain diagrams with new-circuit / all-cells / everything views, OHOL sprites with a skull on poison kinds in strips and dashboards (`book/README.md`).
   The new stage pages follow that structure.

## Known issues and open points

- **To do: reorganise and document the code (user, 2026-10-19).** Bugs have been getting more frequent, and the code
  is hard to follow for the user and for agents.
  - `life/experiments/stages.py` is about 1600 lines with every stage in one file: split it per stage or chapter
    (world builders, brain definitions, stage registry).
  - Ambiguous names: `life/lab.py` (stepping a world by hand) vs `life/labtests.py` (scripted lab experiments).
  - `life/brain.py` has long blocks of JAX math without comments; for example the plasticity coefficients
    `Genome.A`-`D` are defined without saying what they are (for the hebb rule: dw = eta * mod * (A pre post + B pre +
    C post + D); the other rules use A and C differently, see `plasticity_rule`).
  - (Done 2026-10-20) Recordings store the life's food and pain values (`save_recording(rules=...)`), and the
    dashboard and the book's strips mark poison per phase of a life (`dashboard.poison_phases`; older recordings:
    inferred per phase from the bites).
  - An example of a bug this caused: an `abcd` set on a projection is silently replaced by the inherited genes
    (`remap_genomes` carries A-D over for a projection that keeps its rule), which flipped the sign of the `td`
    critic variant (STAGE_LOG v54).

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

Practical notes: on the GPU (WSL2) a 100-generation run of 256 animals with 4000-tick lives takes about 8 minutes, three at a time (STAGE_LOG v49-v50). On the CPU: one process runs a non-plastic stage at about 4-5 s per generation of 1000 ticks and a stage with
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
- Scale: GPU runs in WSL2 are set up (`docs/SETUP.md`, about 10x the CPU); larger populations next (selection is noisy with 64 genomes and one life each, v50).
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
