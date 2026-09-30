# Roadmap and experiment status

Status values: planned / running / done / blocked. Keep this table current (ADR-011).

| # | experiment | what it needs from the core | status |
|---|---|---|---|
| 01 | Evolved forager: find gooseberry bushes, pick, eat | world, vision, fixed-weight brain, evolution | done (see EXPERIMENT_LOG.md) |
| 02 | Hebbian association: two look-alike berries, poison identity flips each generation, must learn within life | per-synapse ABCD plasticity, per-episode rule swap, warm start | runs; plasticity not yet required by the task, see EXPERIMENT_LOG.md and step 0 below |
| 03 | STDP-like sequence learning (trace rule on rates) | `rule="trace"` projections exist (ADR-013); needs an experiment that rewards order | mechanism done, experiment planned |
| 04 | Basal ganglia action selection with dopamine as the three-factor modulator | `modulated=True` projections and `modulator="reward"` exist (ADR-013); a BG region computing its own prediction error is still to be written | half done |
| 05 | Joint evolution of architecture + reflexes + lifetime RL | 04 plus larger populations on GPU | planned |
| 06 | Bigger OHOL slice: stones, kindling, fire, cooking | `expand_hops` slicing; `_LA` last-use-actor transitions; leftover objects after eating | planned |
| 07 | Continuous lifecycle: birth near parent, aging, lineages | replace generational reset with birth/death in `step_world`; uses `parent` field | planned |
| 08 | Communication: VOCALIZE gains a small vector; predator objects that move | sound channel width > 1 (ADR-005 allows widening); `move` transitions | planned |
| 09 | Empathy / altruism: give food to a hungry neighbour | a GIVE action (append to enum) or transition on agents | planned |
| 10 | Cross-generational learning by observation | 07 + 08 + recording of who was watching whom | planned |

## Next steps for exp02 (in order of expected payoff)
0. Make the task need memory. The v3 control solved it reactively through the lingering `pain` input (EXPERIMENT_LOG.md). Set `WorldConfig.pain_decay` to about 0.3 and lower `spawn_density` so agents rarely get repeated berries from the same bush; report the control's poison fraction next to the plastic one every run.
1. Scale: 32 agents x 4 episodes on CPU is tiny for evolving plasticity. On the GPU (docs/SETUP.md) run 256+ agents, 8+ episodes, 300+ generations. The code needs no change.
2. Aftertaste: pain arrives one tick after EAT, when the held-appearance input is already zero. Add a body feature that keeps the last eaten object's appearance for a few ticks (widening the body channel is allowed by ADR-005). This gives the Hebbian rule a direct coincidence between appearance and pain.
3. Eligibility traces: replace `pre` and `post` in the ABCD rule by low-pass traces (the STDP-like rule planned for exp03). Same benefit as 2, more general.
4. Weight normalization or decay on plastic synapses so Hebbian growth cannot saturate at `w_max`.

## Known simplifications to revisit
- `_LA` (last use of actor) transitions are skipped; tools never wear out.
- `move` transitions (animals) are skipped; nothing moves except agents.
- Eating drops any leftover object (bowl, plate) instead of keeping it in hand.
- Cell interactions are resolved sequentially by agent index (deterministic but biased toward low indices); shuffle the order per tick if this matters.
- Sound uses an N x N distance matrix; fine below ~5k agents, replace with a grid convolution beyond.
- World generation ignores biomes: `map_chance` is used as a flat spawn weight.
