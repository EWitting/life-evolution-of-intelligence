# Status

One screen on where the project is. Plan: `docs/BRAIN_EVOLUTION.md`. Details, numbers and run directories:
`docs/STAGE_LOG.md`. Newest decisions: `docs/DECISIONS.md` (ADR-017, ADR-018). Last updated 2026-10-01.

## How a stage is judged

Main = the stage's new brain; control = the parent's brain in the same new world, both warm-started from the
parent's final population, 200 generations, one life per genome. Compared on the last 25 generations. Then a
**lesion study** (`stages lesion <run>`: silence each new region, and switch plasticity off) checks that the gain
really comes from the new module. **Fitness = well-fed lifetime** (sum over ticks alive of food level / full
stomach; ADR-018). Seed spread is about +-60 fitness, so a single-seed gap below ~100 means nothing: use
`stages replicate <key> --seeds 1,2`.

## Stages (v5, 2026-10-01)

Status: **done** (run and analysed), **defined** (in `life/experiments/stages.py`, not yet run), **planned**.
v4 results (old fitness: gross food - pain + 0.01 x ticks) are in STAGE_LOG; they are not comparable.

| stage | what it adds | status | short finding (v5) |
|---|---|---|---|
| 1.0 steering | ganglion (exc + inh interneurons, lagging divisive normalisation), evolved reflexes | done (1 seed + probes) | Fitness 408, lifetime 649 of 1000. No saturated neurons any more (was 47-84% of the time). |
| 1.1 valence | appetitive / aversive cell types with *fixed meaning* | done (3 seeds) | **Tie**: main 386 vs control 387. Lesions: valence cells carry the behaviour. The v4 win came from the old fitness (pain term, gorging). |
| 1.2 drives | `hunger` (broadcast, receptors on appetite), `cold` gating innate thermotaxis | done (1 seed) | Main 232 vs control 281: within seed noise. |
| 1.3 affect | serotonin (`raphe` -> dwell) and PDF (-> roam), now mutually inhibiting | done (1 seed) | Tie (284 vs 288). PDF/roam essential (lesion 280 -> 175); serotonin/dwell unused. |
| 1.4 habituation | short-term depression on identity -> appetitive synapses; OHOL "yum" bonus | done (1 seed) | 320 vs 307: within noise. |
| 1.5 association | US neurons, `us` modulator, identity -> valence plasticity with eligibility traces | defined (new world), not run | World redefined (ADR-017): two novel foods per life, costlier poison, 2000 ticks. Blocked by the food economy (below). |
| 1.6 reversal | learned weights relax toward inherited values; poison swaps mid-life | defined (new world), not run | |
| 2.1 tectum | retinotopic map + inhibitory pool; 9-column eyes | defined | |
| 2.2 pallium expansion | 48 sparse k-WTA neurons; US-gated learning from pallium; XOR poison world | defined (v4 design) | To do first: input weights drawn per life (ADR-017), per-life looks. |
| 2.3 pallium clustering | Oja input + recurrent Hebb; noisy appearance | defined | |
| 2.4 basal ganglia | striatum -| tonic GPi -| motor (disinhibition), fixed | defined | |
| x.td (side test) | 1.6 brain + TD critic only | defined, not re-run in v5 | v4: learning neutral instead of harmful. |
| 2.5 dopamine TD | opponent value populations, TD-error `da` | defined | |
| 2.6-2.10 | actor (D1/D2), curiosity, hippocampal map, cerebellum, NE/ACh | planned | |
| 3.x simulating | neocortex as predictive model, offline mode, PFC, episodic memory | planned | Needs core mechanism M8 (offline mode). |
| 4.x mentalizing, 5.x speaking | self-model, theory of mind, imitation; signals to language | planned | Needs the continuous life cycle (below). |

## Open questions for the next session (where we stopped, 2026-10-01)

1. **Food economy (blocks the learning stages).** Agents eat 5-7 berries in a whole life and most die near the
   never-eating baseline; the 1.0 world holds less food (about 2600 units) than 64 agents need for 1000 ticks
   (3200), and one berry is 30% of a stomach. Within-life learning needs many small meals. Options: smaller
   berries with more bushes, faster regrowth, fewer agents per world. Then re-run 1.0-1.4 and run 1.5/1.6.
2. **Do the chapter-1 modules pay at all?** Under the honest fitness, valence ties over three seeds and the rest
   are single seeds inside the noise. The modules are used (lesions) but give no measurable advantage in their
   current worlds. Replicate 1.2-1.4 (three seeds) after the economy is fixed.
3. **Not yet tested:** movement costs no energy (hunger is flat per tick; walking = standing still); `alpha` 0.5
   delays every path through an interneuron by a tick at half strength (candidates: alpha 1 on the ganglion, two
   brain steps per tick, alpha as an evolved gene); the innate valence wiring drives FORWARD, USE and EAT from
   the same cells although the three actions exclude each other.
4. Metabolic cost of neural activity is implemented but off (`WorldConfig.brain_cost`; ADR-018).

Practical notes: non-plastic stages run at about 1-2 s per generation, plastic ones at about 9 s (1000 ticks)
with two processes on the CPU; at most two JAX processes at a time. The lineage run of a stage is the newest run
in its un-suffixed directory; `_seedN`, `_control`, `probe*` directories are variants.

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
