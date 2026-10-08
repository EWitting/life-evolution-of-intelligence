# Plan: the complete rerun of chapter 1

Written 2026-10-08 at the end of session 5, for a fresh session. Read `docs/STATUS.md` first (one screen), then
this file. Numbers and dead ends behind every statement here are in `docs/STAGE_LOG.md`, sections v24 to v31.

## What was agreed with the user

After session 5 the learning stage works for the first time. The user wants a **complete rerun of chapter 1, all
stages, three seeds, from stage 1.0**, as the milestone that closes this work, on the condition that the findings
carry over to a new lineage (user, 2026-10-08). Decisions that go with it:

- **Order: habituation before association** is allowed and is the plan: 1.0 -> 1.1 -> habituation -> association
  -> drives.
- **Stage 1.1l (long lives) does not stay.** Long lives start earlier (below).
- **Hunger stays one inhibitory cell** with its evolvable synapses onto feeding. No second (satiety) pathway.
- **Drives is part of the rerun**, three seeds.
- **Short lives are accepted** (about 900 of 4000 ticks in the learning world) as long as there is time to learn
  and fitness tells animals apart. Do not tune this now. If it is ever tuned: first find out why they starve.
  Born full is acceptable when a stomach is a quarter of a life or less.
- **Rules of work**: cheap trials first (one stage, one seed, generation 0 where possible); the whole chain and
  three seeds only when a design looks final; look inside a finished run before starting the next; minor tuning
  of constants is fine, **no new mechanisms and nothing that feels like a hack without the user**; document
  choices and findings as you go. The book may be edited and published without asking.

## What the chain looks like now, and what the rerun changes

Now (as run in session 5; every stage below exists in `life/experiments/stages.py`):

| key | directory | what it is | state |
|---|---|---|---|
| 1.0 | s1_0_steering | reflex steering | v25 footing, one lineage, lives of 1000 ticks |
| 1.1 | s1_1_valence | value cells, grasp programme | passes, 3 seeds, lives of 1000 ticks |
| 1.1l | s1_1_valence_long | the 1.1 brain under lives of 4000 ticks | adaptation step, one lineage |
| 1.5 | s1_5_association | pain teaches the *centred* look -> aversive synapses | passes, 3 seeds |
| 1.6h | s1_6_habituation | look inputs adapt to their slow average; learned synapses plain | passes, 3 seeds |
| 1.7 | s1_7_drives | cold, hungry, warm_seek, rest on the 1.6h brain | trial, 1 seed |
| 1.2h | s1_2_habituation | adapting look inputs on the 1.1 brain, 1.1 world | trial, 1 seed (the check for the new order) |
| 1.5f | s1_5_association_fwd | 1.5 with only the forward eye learning | variant, not needed |

The rerun:

| step | brain | world, life | control | generations (suggested) |
|---|---|---|---|---|
| 1.0 from scratch | B10 | W10, 1000 ticks | none | 400 |
| 1.0 settle | B10 | W10, **4000 ticks** | none | 60 at 10% mutation |
| 1.1 valence | B11 | W11, **4000 ticks** | B10 | about 50 |
| habituation | B11 + adapting look inputs | W11, 4000 ticks | B11 | 40 |
| association | habituation brain + `us_pain` + plain learned look -> aversive synapses | W15, 4000 ticks | the habituation brain | 40 |
| drives | association brain + cold, warm_seek, rest, hungry | W17, 4000 ticks | the association brain | 40 |

Why long lives move into 1.0: a stomach lasts 400 ticks at rest in 1.0 and 800 in 1.1, so in a life of 1000 ticks
an animal that ate once and stood still reached the cap. Stage 1.0 from scratch stays at 1000 ticks (nearly
everything dies young at first, longer lives would only cost compute); its settling run at 4000 ticks is where
the standing still is selected away, and everything after has 4000-tick lives. `stages <key> --ticks 4000
--init-from <run>` exists for the settling run.

## Code changes needed before running

All in `life/experiments/stages.py` unless stated. None is a new mechanism; every piece exists.

1. **Stage 1.1**: `ticks=4000`, fewer generations (150 at 1000 ticks was the old setting; about 50).
2. **Habituation as a stage of the chain**: take the trial `1.2h`, set its parent to `"1.1"`, remove "TRIAL" from
   its notes. Brain: `replace(B11, in_adapt=CS_VIS, in_trace_tau=ADAPT_TAU)`.
3. **Association on the habituation brain**: the brain that `B16H` is now (`_B15` with `in_adapt` and plain
   `_split_cs`, no `centred`), with the habituation stage as parent, world `W15`, `**_S15`. Its control is then
   the habituation brain automatically (control = parent's brain).
4. **Drives**: `B17` extends that association brain; parent the association stage.
5. Retire `1.1l`, the centred `1.5`, `1.5f` and `1.6h` from the chain. Keep them defined or note them as side
   paths in STATUS, as the user prefers a list of known side paths at the freeze.
6. `life/book.py`: `NO_HEAD_TO_HEAD` must name the habituation stage's key (head to head runs both halves on the
   new brain, which is unfair when the change is in the inputs); the `life_spent` and `adaptation_off` extras are
   registered per key, update the keys; the `first_lesson` extras belong to the association key.
7. Keys and directories: reusing `1.5` / `s1_5_association` and `1.7` / `s1_7_drives` is fine (the tools take the
   newest finished run per directory and the book flags runs that differ from the stage definition), but old runs
   of other definitions sit in the same directories. `scripts/probes/prune.py` deletes superseded runs; look at
   what it will delete before running it, and ask the user before deleting anything.
8. Tests: `tests/test_brain_v2.py::test_pain_teaches_only_the_aversive_synapses` builds stage "1.5"; keep it
   passing after the redefinition. Run `python -m pytest -q` (35 tests, about 2 minutes).

## Order of work

One seed down the whole chain first, looking inside each run; then seeds 1 and 2; then evaluations and the book.

1. `stages 1.0` (scratch, about 35 minutes), then `stages 1.0 --init-from <that run> --mutation-prob 0.1
   --generations 60 --ticks 4000`. Check: `forage.py`, `freeze.py`, `supply.py s1_0...` on the settled run: is the
   population using the long life (freezes few, survival not collapsing)?
2. `stages chain 1.1 1.1 --mutation-prob 0.1`. Check lesions of the value cells on this seed
   (`stages lesion <run>`), `forage.py`, `freeze.py`.
3. Habituation, `chain <key> <key>`. Check `adapt_off.py <key>` (the lesion of the adaptation: there is no
   region to silence), `forage.py` main vs control.
4. Association. Check `learned.py` (rates, where the lesson lands, teacher activity), `lessons.py` at generation
   0 if anything about the world changed, lesions (`no_plasticity`, `us_pain`).
5. Drives. Check lesions of `cold`, `warm_seek`, `rest`, `hungry`; the `hungry` -> feeding weights (STAGE_LOG
   v30 shows how); `drives.py` at generation 0 with `PARENT=<association key>` before evolving.
6. `stages replicate <key> --seeds 1,2 --mutation-prob 0.1` per stage (it starts every seed from the parent's
   lineage run, so only seed 0 has to exist for the parent).
7. `python -m life.book export <key> --evaluate` per stage (40-75 minutes each for 4000-tick stages; one at a
   time), `adapt_off.py` for habituation (it writes `book/data/adapt_off_<key>.json`),
   `python -m life.book dashboards <key>`.
8. Book text, STATUS, STAGE_LOG, publish (below). Then the user's freeze steps in STATUS (chapter exam, snapshot
   tests, tag).

Rough compute: about 6 to 8 hours in total with four processes in parallel. Up to about five JAX processes at
once; background commands may be stopped after two hours, so run in pieces.

## What could stop working with a new lineage, and what is known

- **Association: expected to hold.** Its three fixes are by design: the lesson lands on what sets a food apart
  (now through the adapting inputs), the teacher fires for one tick (pain_decay 0 in W15), every learned synapse
  exists (`Stage.dense_plastic`). Main was ahead of control from generation 0 in every run.
- **The world settings: expected to hold.** NOVEL_SIM 0.8, POISON_FOOD -3, born half full in W15, only novel
  types, 6-berry bushes, food as a flow, bite cost, non-blocking hot springs. Each was chosen with a generation-0
  probe that takes minutes to repeat (`lessons.py`, `supply.py`, `drives.py`).
- **Habituation before association: the weak point, checked on one seed (2026-10-08, `s1_2_habituation`).**
  In the familiar 1.1 world long lives alone already cut the standing still, so there is less to win:
  - main recovers from the expected setback (984 in generations 0-4) to 1610 in generations 30-39; control 1681.
    About level, main not ahead.
  - adaptation switched off in the final population: 1593 -> 1345, **84% of intact** (used, weakly; in the
    novel-food world it was 31-35%).
  - standing and biting at nothing 8% of ticks vs 21% in the control; moving 75% vs 61%.
  So in this order habituation should pass "used" by a modest margin and "not worse" within noise. Judge it over
  three seeds. If it fails there, the fallback is the order that already passed with three seeds: association
  with the centred rule (stage 1.5 as it is), then habituation (1.6h). Tell the user before switching.
- **Association's control changes.** In the new order its control is the habituation brain without learning,
  which forages better than the old control. The gap may be smaller than the +107 / +175 measured in session 5.
  The lesions (`no_plasticity`, `us_pain`) are what decide "used".
- **Fitness is noisy between generations in the learning world** (new looks every generation; sd about 290 in
  1.5). Read trends over blocks of ten generations with their standard error (about 90), never from one block.
- **1.1 with 4000-tick lives is new.** Its pain-blind control needed 60-100 generations of 1000 ticks to find
  avoidance by look in some seeds; with 50 generations of long lives the control may or may not get there. That
  is fine for the criterion (used, not worse); say on the book page what the comparison does and does not show.

## The book

Sources in `book/`, workflow in `book/README.md`, published at
https://ewitting.github.io/life-evolution-of-intelligence/ (GitHub Pages serves the `gh-pages` branch).

    python -m life.book export <key> [--evaluate]        book/data/<key>.json
    python -m life.book dashboards <key>                 light dashboards of seed 0 (not committed)
    "C:\\Program Files\\Quarto\\bin\\quarto.exe" publish gh-pages book --no-prompt --no-browser

Text to rewrite when the order changes (numbers never go into the text; they come from the data files):

- `book/habituation.qmd` is written as the stage *after* learning ("the animals of the previous stage learn which
  berries are bad"). In the new order it comes before: the motivation is the standing at empty bushes that stage
  1.1 already shows, and learning is not mentioned yet.
- `book/1-5-association.qmd`, section "What exactly gets blamed": the first of the three points (the slow average
  of each look input) is no longer part of this stage; it is what the eyes already do since habituation. The
  callout "What still goes wrong: standing at an empty bush" belongs to the old order.
- `book/_quarto.yml` chapter order; `book/drives.qmd` is a stub and needs its page; `book/1-0-steering.qmd` is
  mostly a stub; `book/index.qmd` should say that lives are long from the first stage on.
- The pages of 1.0 and 1.1 describe 1000-tick lives implicitly; check them after the rerun.

Recent book features: the 1.0 page shows the whole history (scratch and settling), "Watch a main / control run"
buttons with the run directories folded away, a light and dark theme, a generic `table` component for extras.

## Practical notes for this machine

- Python is `.venv/Scripts/python.exe` (there is no `python` on PATH). Run from the project root.
- In this shell, a here-document containing apostrophes or `\\` escapes can fail or be altered: write a small
  script file with the Write tool and run it. A bare `/` or `+0.3`-style argument can be rewritten by Git Bash;
  the probes use `+` as their group separator for that reason.
- `fitness.csv` is written when a run ends; follow progress in `runs/logs/<name>.log`.
- An unrecorded simulation stops when every animal is dead, so early generations of a hard world are quick.
- Every probe and its purpose: `scripts/probes/README.md`. Environment variables: `STAGE`, `RUN`, `TICKS`,
  `WORLDS`, `DENSE`, `PARENT`.
