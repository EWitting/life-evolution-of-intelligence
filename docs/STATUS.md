# Status

One screen on where the project is. Plan: `docs/BRAIN_EVOLUTION.md`. Details, numbers and run directories:
`docs/STAGE_LOG.md`. Newest decisions: `docs/DECISIONS.md` (ADR-017, ADR-018). Last updated 2026-10-01.

## How a stage is judged

Main = the stage's new brain; control = the parent's brain in the same new world, both warm-started from the
parent's final population, 200 generations, one life per genome, **three seeds** (`stages replicate <key>`).
`stages summary <key>` gives the mean of the last 50 generations and the final populations re-evaluated in 8
shared worlds, with the standard error over seeds. Then a **lesion study** (`stages lesion <run>`: silence each
new region, and switch plasticity off). **Fitness = well-fed lifetime from the first meal** (ADR-018).

## Stages (v6, 2026-10-01)

Status: **done** (run and analysed), **defined** (in `life/experiments/stages.py`, not yet run), **planned**.
Numbers are the v6 run, which still used the well-fed lifetime counted from birth and the `cold` bug (below).

| stage | what it adds | status | short finding (v6, 3 seeds, main vs control) |
|---|---|---|---|
| 0.9 bootstrap | the 1.0 brain from random weights, free movement | done | Foraging evolves: fitness 587, 28 meals per life. |
| 1.0 steering | ganglion (exc + inh, lagging normalisation, alpha 1); movement now costs energy | done | 522, 32 meals per life. From scratch with movement cost, evolution stood still (fitness artefact, now fixed but not re-tested). |
| 1.1 valence | appetitive / aversive cell types with *fixed meaning* | done | Tie: 440 vs 452 (re-evaluated +9 +-4). Less poison and pain, no fitness gain. |
| 1.2 drives | `hunger` (broadcast), `cold` gating innate thermotaxis | done, to redo | 309 vs 325. Ran with the `cold` bug. |
| 1.3 affect | serotonin (dwell) and PDF (roam), mutually inhibiting | done, to redo | 276 vs 285. |
| 1.4 habituation | short-term depression on identity -> appetitive synapses; "yum" bonus | done, to redo | 282 vs 273. |
| 1.5 association | US neurons, `us` modulator, identity -> valence plasticity with eligibility traces | defined, not run | New world (ADR-017): two novel foods per life, costlier poison, 2000 ticks. |
| 1.6 reversal | learned weights relax toward inherited values; poison swaps mid-life | defined, not run | |
| 2.1 tectum | retinotopic map + inhibitory pool; 9-column eyes | defined | |
| 2.2 pallium expansion | 48 sparse k-WTA neurons; US-gated learning from pallium; XOR poison world | defined (v4 design) | To do first: input weights drawn per life (ADR-017), per-life looks. |
| 2.3 pallium clustering | Oja input + recurrent Hebb; noisy appearance | defined | |
| 2.4 basal ganglia | striatum -| tonic GPi -| motor (disinhibition), fixed | defined | |
| x.td (side test) | 1.6 brain + TD critic only | defined, not re-run | |
| 2.5 dopamine TD | opponent value populations, TD-error `da` | defined | |
| 2.6-2.10 | actor (D1/D2), curiosity, hippocampal map, cerebellum, NE/ACh | planned | |
| 3.x simulating | neocortex as predictive model, offline mode, PFC, episodic memory | planned | Needs core mechanism M8 (offline mode). |
| 4.x mentalizing, 5.x speaking | self-model, theory of mind, imitation; signals to language | planned | Needs the continuous life cycle (below). |

## Next steps (where we stopped, 2026-10-01)

1. **Probe the first-meal fitness from scratch with movement cost** (`stages 1.0 --init-from none`, two seeds; a
   few minutes). It was started and then stopped when the machine ran low on memory. If foraging evolves, drop
   stage 0.9.
2. **Re-run 0.9/1.0-1.4 with three seeds** under the first-meal fitness and the fixed `cold` neuron (about an hour).
3. **Calibrate and run 1.5** (one seed plus the plasticity-off lesion, about 15 minutes at the new speed), then
   1.6 and x.td.
4. **Open design questions.** No chapter-1 module beats its control. Candidates: poison is cheap (2/3 food unit per
   berry in 1.1-1.4); the 1.2 world is a cliff (fitness 440 -> 320); the innate valence wiring drives FORWARD,
   USE and EAT from the same cells although the actions exclude each other; serotonin/dwell was unused in v5.
5. Metabolic cost of neural activity is implemented but off (`WorldConfig.brain_cost`).

Practical notes: one process runs non-plastic stages at about 0.9 s and learning stages (2000 ticks) at about
2.3 s per generation; at most two JAX processes at a time. The lineage run of a stage is the newest run in its
un-suffixed directory; `_seedN`, `_control`, `probe*` directories are variants. The dashboard clips displayed
weights at `w_max`.

## Core and tools (done)

Brain v2 (ADR-015/016): rectified rates, Dale's law, hard-wiring, input-feature selection, topographic
projections, named modulators with receptors, eligibility traces, gain, depression, decay, delta rule, k-WTA, divisive normalisation,
genome remapping across layouts. World: look-alike appearances, OHOL temperature and hot springs, delayed
sickness, taste, temperature-change sense, patches, variety bonus, mid-life rule switch, appearance noise,
multi-clone OHOL slicing (incl. category tools). Tools: stage runner and chain, lesion, respond, compare,
dashboard architecture view (groups, modulator nodes). 24 tests.

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
