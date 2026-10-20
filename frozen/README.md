# Frozen stages

A frozen stage is finished: its definition in `life/experiments/stages.py` does not change and its lineage is not
rerun. Each directory holds the stage's definition as data and the final population of every seed. Tooling and
file layout: `life/freeze.py`. Guard: `tests/test_frozen.py` fails when a frozen stage's definition changes or a
stored population no longer loads.

    python -m life.freeze <key>       freeze a stage (or freeze it again after an agreed change)
    python -m life.freeze --table     the table below
    life.freeze.load_frozen(key, seed, exp=None)     a stored population, remapped onto exp's layout when given

A later change that has to touch a frozen stage (removing a cell that lesions show unused, renaming a region) is
made by remapping the stored populations onto the new layout and freezing again, with the reason in
`docs/STAGE_LOG.md`. Results and criteria per stage: `docs/STATUS.md` and the book.

## Chapter 1 (frozen 2026-10-08, git tag `chapter-1-drives`; 1.0 to 1.5 also as `chapter-1-partial`; affect redefined and frozen 2026-10-17, tag `chapter-1-v2`)

**Order of the chain: 1.0 -> 1.1 -> 1.2h -> 1.5 -> 1.7 -> 1.8.** The keys are historical names, not positions (1.5 and
1.7 keep the numbers they had in an earlier order); they stay because run directories, data files and the stage
log use them. The book shows the stages by name only, in this order: Steering, Valence, Habituation, Association,
Drives, Affect.

The lineage is one chain: every stage starts from the seed-0 run of its parent. Seeds 1 and 2 of a stage start
from that same parent run.

| stage | what it is | parent | life | seeds |
|---|---|---|---|---|
| 1.0 | reflex steering (ganglion, evolved sensor -> motor reflexes) | - | 1000 ticks from scratch, then settled at 4000 | 1 |
| 1.1 | valence: appetitive and aversive value cells, contact-gated grasp programme | 1.0 | 4000 | 3 |
| 1.2h | habituation: the look inputs adapt to their slow average; world with a dud bush type | 1.1 | 4000 | 3 |
| 1.5 | association: pain teaches the look -> aversive synapses; only novel foods, half poison | 1.2h | 4000 | 3 |
| 1.7 | drives: `cold`, `warm_seek`, `rest`, `hungry`; the 1.5 world made cold, with hot springs | 1.5 | 4000 | 3 |
| 1.8 | affect: `alarm` -> `flee`, `serotonin` (`fed`) -> `settle` -> `rest`, adding up over bites; 256 x 256, berry types in patches 70% their own kind; learning rates mutate multiplicatively; 100 generations | 1.7 | 4000 | 3 |

Chapter 1 is complete. A stored population written before a gene existed (the modulator fade times, `Genome.clear`,
2026-10-12) loads with the designed value, and a stored configuration is read with options added since at their
defaults (`freeze.with_defaults`).

## Chapter 2 (frozen from 2026-10-19)

| stage | what it is | parent | life | seeds |
|---|---|---|---|---|
| 2.2 | pallium: 128 cells (k-WTA 2) read the eye ahead through fixed random weights of mixed sign; pain teaches `pallium -> valence_av`; the adaptation strength evolves; world `W22` (the 1.8 world with XOR looks, patches mixed evenly) | 1.8 | 4000 | 3 |
| 2.3h | dopamine critic as a formula (the 2.3 brain): value cells on the pallium and one signed TD error that teaches them and, negated, every synapse pain taught; pallium wiring per population slot. Harsh world: one berry per bush worth one berry, a poison berry costing as much (POISON_FOOD -3), a quarter of the metabolic rate, born 0.75 full; the poison and good kinds swap at tick 800 from generation 50 on (`Stage.schedule` = `calm_first(50)`); aversion learning rates start x10 (`av_eta_init`); 2 episodes per generation | 2.2 | 4000 | 3 |

2.2: `flee` and `fed` are unused in this world (lesions 97-101%) and kept: frozen chapter 1 circuits that later worlds
may need (user, 2026-10-19).

2.3h (frozen 2026-10-20, replacing 2.3 on the chain, user): the calm start and the x10 start values are functions in
`stages.py` (`calm_first`, `av_eta_init`) and not part of the stored rules digest, which is taken at generation 0 with
the reversal on. STAGE_LOG v59-v65.

## The first critic (frozen 2026-10-19, off the chain since 2026-10-20)

| stage | what it is | parent | life | seeds |
|---|---|---|---|---|
| 2.3 | the same brain in `W23` (the pallium world, six berries to a bush, a reversal at tick 800 from the first generation) | 2.2 | 4000 | 3 |

In that world no creature judged a bush by its look before biting (a first bite on every bush, the visit ended by
the pain): the critic paid by a learned appetite for bushes, and its learned aversion stayed below the aversive
cells' threshold (STAGE_LOG v58-v61).

## The first affect route (frozen, off the chain since 2026-10-17)

| stage | what it is | parent | life | seeds |
|---|---|---|---|---|
| 1.8old | the first affect stage: states released at once by one bite; the 1.7 world with one berry type per patch (frozen 2026-10-12 as 1.8, tag `chapter-1`) | 1.7 | 4000 | 3 |
| 2.0 | camera eyes (9 eye columns, range 6, no new cells) in open country: no patches, density 0.3, 181 x 181 | 1.8old | 4000 | 3 |
| 2.1 | tuning for chapter 2: 256 x 256, patches 70% their own type; the lasting states add up over bites (`B20S`) | 2.0 | 4000 | 3 |

In 1.8old learning went unused and its rate drifted up; 2.0 brought learning back in open country, 2.1 moved the
lineage into the world of chapter 2. The new 1.8 does the same in one stage: what kept learning from surviving the
direct route was the additive mutation of learning rates, which traps an idle rate at 0 (STAGE_LOG v43-v45). 2.0
(frozen 2026-10-12 as 2.1, renamed 2026-10-13; git tag `stage-2.1`) starts from the frozen 1.8old seed 0 with the
new eye columns interpolated (`stages.interpolate_eyes`); 2.1 starts each seed from the same seed of 2.0, with
`stages.sum_affect_init`.

## Side paths (defined in `stages.py`, not part of the lineage)

The runs of the side paths were deleted on 2026-10-12 (disk space; findings are in `docs/STAGE_LOG.md`). A stage
that starts from one needs that parent rerun first.

| key | what it was | why it is not on the path |
|---|---|---|
| 1.1l | the 1.1 brain given long lives as a separate step | 1.1 itself now has long lives |
| 1.5c | association before habituation, the learned synapses centring the look themselves | passed over three seeds (session 5); the order with habituation first passes more clearly |
| 1.5f | 1.5c with only the forward eye learning | no better than all eyes |
| 1.6h | habituation after 1.5c | the old order |
| 1.6 | reversal: learned weights relax, the novel types swap meaning mid-life | not used (v22) |
| 1.2, 1.3, 1.4 | older drives, affect and habituation designs | parked 2026-10-03; drives is now 1.7 |
| x.hands | pick, hold, then eat | side test: the lineage adapts in about ten generations |
| x.td | TD critic on the old reversal brain | side test for chapter 2 |

Habituation worlds that were tried and left out (one seed each, `docs/STAGE_LOG.md` v32): the plain 1.1 world
(adaptation used weakly), novel colours for three or all four good berry types (main behind control), novel
colours for three good types plus the dud (works, less clearly than the dud alone).
