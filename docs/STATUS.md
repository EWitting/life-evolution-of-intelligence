# Status

One screen on where the project is. Plan: `docs/BRAIN_EVOLUTION.md`. Details, numbers and run directories:
`docs/STAGE_LOG.md`. Newest decisions: `docs/DECISIONS.md` (ADR-015, ADR-016). Last updated 2026-09-30.

## How a stage is judged

Main = the stage's new brain; control = the parent's brain in the same new world, both warm-started from the
parent's final population, 200 generations, one life per genome. Compared on the last 25 generations. Then a
**lesion study** (`stages lesion <run>`: silence each new region, and switch plasticity off) checks that the gain
really comes from the new module. One seed per run so far, so differences of a few fitness points are noise.

## Stages

Status: **done** (run and analysed), **running**, **defined** (in `life/experiments/stages.py`, not yet run),
**planned** (only in the plan).

| stage | what it adds | status | short finding |
|---|---|---|---|
| 1.0 steering | ganglion (exc + inh interneurons), evolved reflexes | done | Foraging evolves from scratch: food eaten 10 -> 47 in 400 generations. Needs several food types, or the lineage becomes a specialist that ignores new food. |
| 1.1 valence | appetitive / aversive cell types with *fixed meaning* (taste/pain in; feeding/turning out; `no_feed` suppression) | done | Main 53 vs control 40 fitness, less poison (0.12 vs 0.19). Lesions: valence now essential (49 -> 7). Without fixed meaning (v1, v2) evolution ignored the module: extra neurons just get absorbed into the ganglion. |
| 1.2 drives | hypothalamus: `hunger` (broadcast, receptors on appetite), `cold` gating innate thermotaxis (run-and-tumble on temperature change) | running (v4) | v3: small edge within noise; `cold`/`warm_run` became essential but were co-opted as a general "keep moving" drive rather than for warming. World rebalanced (v4) because the cold was a cliff. |
| 1.3 affect | serotonin (`raphe` -> dwell) and PDF (-> roam), slow, broadcast through receptors | queued (v4) | v3: small edge; lesions: serotonin/dwell used (-5 points), roam a little (-2). Revealed mutational load (fitness fell under selection); now 10% of weights mutate per child. |
| 1.4 habituation | short-term depression on sensory -> appetitive synapses; OHOL "yum" variety bonus | queued (v4) | v3: small edge (+1.6), within noise. |
| 1.5 association | US neurons, `us` modulator, identity -> valence plasticity with eligibility traces; poison identity per life, delayed sickness | queued (v4) | v3: main > control but learning neutral (plasticity off = same score). Causes found: (a) no innate valence->motor path (fixed), (b) conditioning generic "something is there" features (fixed), (c) plain Hebbian conditioning over-generalises across look-alikes (-> error-driven learning, stage 2.5), (d) world too harsh to learn in, ~5 berries per life (rebalanced in v4). |
| 1.6 reversal | learned weights relax toward inherited values; poison swaps mid-life | queued (v4) | v3: main 16.7 vs 13.9, not yet checked with lesions. |
| 2.1 tectum | retinotopic map (topographic projection) + inhibitory pool; 9-column eyes | defined | |
| 2.2 pallium expansion | 48 sparse k-WTA neurons with fixed random input; US-gated learning from pallium; XOR poison world | defined | |
| 2.3 pallium clustering | Oja input + recurrent Hebb; noisy appearance | defined | |
| 2.4 basal ganglia | striatum -| tonic GPi -| motor (disinhibition), fixed | defined | |
| 2.5 dopamine TD | opponent value populations, TD-error `da` from weighted region terms; dopamine replaces the raw US as teacher | defined | Expected to fix the over-generalisation of 1.5. |
| 2.6-2.10 | actor (D1/D2), curiosity, hippocampal map (+ STDP sequences), cerebellum, NE/ACh | planned | 2.6 candidate world: OHOL stone -> sharp stone -> dig wild carrot / burdock chain (verified in the data). |
| 3.x simulating | neocortex as predictive model, offline mode (simulation, replay), PFC, episodic memory | planned | Needs core mechanism M8 (offline mode). |
| 4.x mentalizing, 5.x speaking | self-model, theory of mind, imitation, future needs; signals to language | planned | Needs the continuous life cycle (below). |

## Core and tools (done)

Brain v2 (ADR-015/016): rectified rates, Dale's law, hard-wiring, input-feature selection, topographic
projections, named modulators with receptors, eligibility traces, gain, depression, decay, delta rule, k-WTA,
genome remapping across layouts. World: look-alike appearances, OHOL temperature and hot springs, delayed
sickness, taste, temperature-change sense, patches, variety bonus, mid-life rule switch, appearance noise,
multi-clone OHOL slicing (incl. category tools). Tools: stage runner and chain, lesion, respond, compare,
dashboard architecture view (groups, modulator nodes). 21 tests.

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
