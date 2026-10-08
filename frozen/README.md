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

## Chapter 1 (frozen 2026-10-08, git tag `chapter-1-partial`)

The lineage is one chain: every stage starts from the seed-0 run of its parent. Seeds 1 and 2 of a stage start
from that same parent run.

| stage | what it is | parent | life | seeds |
|---|---|---|---|---|
| 1.0 | reflex steering (ganglion, evolved sensor -> motor reflexes) | - | 1000 ticks from scratch, then settled at 4000 | 1 |
| 1.1 | valence: appetitive and aversive value cells, contact-gated grasp programme | 1.0 | 4000 | 3 |
| 1.2h | habituation: the look inputs adapt to their slow average; world with a dud bush type | 1.1 | 4000 | 3 |
| 1.5 | association: pain teaches the look -> aversive synapses; only novel foods, half poison | 1.2h | 4000 | 3 |

Not frozen yet: 1.7 drives (run over three seeds on the frozen 1.5; see `docs/STATUS.md`) and affect (to be
designed on drives). Both sit at the end of the chain, so finishing them cannot change anything above.

## Side paths (defined in `stages.py`, not part of the lineage)

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
