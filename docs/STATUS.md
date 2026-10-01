# Status

One screen on where the project is. Plan: `docs/BRAIN_EVOLUTION.md`. Details, numbers and run directories:
`docs/STAGE_LOG.md`. Newest decisions: `docs/DECISIONS.md` (ADR-017, ADR-018). Last updated 2026-10-01.

## How a stage is judged

A stage succeeds when its circuit is **used**, so that later stages can build on it: silencing the new regions
clearly lowers fitness, and main is not clearly worse than control (user, 2026-10-01; main >> control is not
required). Main = the stage's new brain; control = the parent's brain in the same new world; both warm-started
from the parent's final population, 200 generations, one life per genome, **three seeds**.

    stages replicate <key>      two more seeds of main and control
    stages summary <key>        main vs control: last 50 generations, and re-evaluated in 8 shared worlds, +- s.e.
    stages lesions <key>        fitness with each region silenced, % of intact, per seed

**Fitness = well-fed lifetime from the first meal** (ADR-018).

## Stages (v9, 2026-10-01)

Status: **done** (run and analysed), **defined** (in `life/experiments/stages.py`, not yet run), **planned**.
"Lesion" = fitness with the region silenced, % of intact, mean of three seeds.

| stage | what it adds | status | main vs control | is the circuit used? |
|---|---|---|---|---|
| 1.0 steering | ganglion (exc + inh, lagging normalisation, alpha 1), evolved reflexes; movement costs energy | done | n/a | Foraging evolves from random brains: lifetime about 690 of 1000, 33 meals per life. |
| 1.1 valence | appetitive / aversive cell types with *fixed meaning* | done | 539 vs 537 | **Yes.** valence_av 73, no_feed 78, valence_app 87. |
| 1.2 drives | `hunger` (broadcast), `cold` gating innate thermotaxis | done | 609 vs 634 | **No.** hunger, cold, warm_run, warm_turn all 100-102. |
| 1.3 affect | serotonin (dwell) and PDF (roam), mutually inhibiting | done | 539 vs 567 | **Partly.** pdf 85, roam 85; raphe 87, dwell 91. |
| 1.4 habituation | short-term depression on identity -> appetitive synapses; "yum" bonus | done | 557 vs 567 | **No.** Depression off = 100. |
| 1.5 association | US neurons, `us` modulator, identity -> valence plasticity with eligibility traces | done (1 seed) | 735 vs 883 | **No.** Plasticity off = 97.5; main below control. An agent eats only about 19 berries per life. |
| 1.6 reversal | learned weights relax toward inherited values; poison swaps mid-life | defined, not run | | |
| 2.1 tectum | retinotopic map + inhibitory pool; 9-column eyes | defined | | |
| 2.2 pallium expansion | 48 sparse k-WTA neurons; US-gated learning from pallium; XOR poison world | defined (v4 design) | | To do first: input weights drawn per life (ADR-017), per-life looks. |
| 2.3 pallium clustering | Oja input + recurrent Hebb; noisy appearance | defined | | |
| 2.4 basal ganglia | striatum -| tonic GPi -| motor (disinhibition), fixed | defined | | |
| x.td (side test) | 1.6 brain + TD critic only | defined, not re-run | | |
| 2.5 dopamine TD | opponent value populations, TD-error `da` | defined | | |
| 2.6-2.10 | actor (D1/D2), curiosity, hippocampal map, cerebellum, NE/ACh | planned | | |
| 3.x simulating | neocortex as predictive model, offline mode, PFC, episodic memory | planned | | Needs core mechanism M8 (offline mode). |
| 4.x mentalizing, 5.x speaking | self-model, theory of mind, imitation; signals to language | planned | | Needs the continuous life cycle (below). |

## Open questions (where we stopped, 2026-10-01)

1. **The drives (1.2) are not used.** The thermotaxis reflex works when it fires but fires on 4% of ticks and does
   not change body temperature; hunger only scales appetite. Ideas, none tested: make the reflex persist for a few
   ticks or let `cold` suppress feeding (a real competition between drives); put food near the springs so that
   warmth is a cue to good habitat (why worms do thermotaxis); give satiety a point (eating costs something when
   full). The calibrated worlds are also easy (lifetime 900 of 1000), which weakens every pressure.
2. **Habituation (1.4) is not used.** The variety bonus (+-30%) may be too small, or the depression too slow.
3. **Lesions vary a lot between seeds** (valence_av 50-84% of intact in 1.2). The lineage follows seed 0; picking
   the seed in which the circuits are most used as the lineage would be a deliberate choice.
4. **Calibration level.** Parent lifetime is held at about 850-900 of 1000. A harder setting (say 600) would raise
   every selection pressure but shortens the lives that learning needs.
5. Camping is up to 18-36% of 100-tick windows in 1.2-1.4.
6. Not started: 1.6, x.td, ADR-017's per-life pallium weights for chapter 2. Metabolic cost of neural activity is
   implemented but off.

Practical notes: one process runs non-plastic stages at about 0.9 s and learning stages (2000 ticks) at 2-3 s
per generation; at most two JAX processes at a time. A sleeping laptop pauses runs, and background commands are
stopped after two hours of wall-clock time, so run long batches in pieces. The lineage run of a stage is the
newest run in its un-suffixed directory; `_seedN`, `_control`, `probe*` directories are variants. The dashboard
clips displayed weights at `w_max`.

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
