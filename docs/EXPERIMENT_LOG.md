# Experiment log

Newest at the bottom. Each entry: what was run, the numbers that matter, what was concluded, what changed as a result.
Run directories are under `runs/<experiment>/<timestamp>/` (git-ignored); logs of the runs quoted here are `runs/*.log`.

## 2026-09-18 exp01 evolved forager, first attempt (1 episode, 400 ticks)

- 40 generations, 32 agents, 24x24 world, no plasticity. About 15 s total on CPU.
- Result: no progress. Mean fitness flat around 8, max jumping between 10 and 112 depending on who spawned next to a bush. Nobody died (400 ticks x 0.05 hunger = exactly max food).
- Conclusion: fitness was spawn luck. Change: ADR-012 episode averaging (4 worlds per generation) and 800-tick lives.

## 2026-09-18 exp01 evolved forager, with episode averaging (4 episodes, 800 ticks)

- 60 generations, same population size. About 70 s on CPU.

| generation | mean fitness | mean net food eaten | expected survivors of 32 |
|---|---|---|---|
| 0 | 5.4 | 1.2 | 0.25 |
| 24 | 11.9 | 7.6 | 1.75 |
| 42 | 28.9 | 24.0 | 5.25 |
| 54 | 40.0 | 34.8 | 6.0 |
| 58 | 41.6 | 36.4 | 7.5 |

- Conclusion: foraging evolves reliably. Generation-to-generation noise is still large (gen 59 dipped to 22); more episodes or a bigger population would smooth it once on GPU.
- Viewer check: `runs/exp01_evolved_forager/20260918-170524/frame300.png` and `replay.gif` render correctly.

## 2026-09-18 exp02 Hebbian association, from random population

- 80 generations, plastic and non-plastic control, poison = -2 food and -2 fitness per pain.
- Result: both collapsed to "never eat" (net food about 0, fitness = survival bonus only). Poison cost 6, good berry gained 6, so random eating had zero expected value and agents never tasted poison often enough for plasticity to matter.
- Changes: poison softened to -1 food and -1 fitness per pain (random eating now pays +1.5 per berry), and warm start from the exp01 population (ADR-012, `--init-from latest`).

## 2026-09-18 exp02 warm-started, poison per generation (logs: exp02_plastic.log, exp02_plastic_v2.log, exp02_control.log)

- 80 generations each. Poison fraction = poison berries eaten / all berries eaten (0.5 = no discrimination).
- Plastic v1 (dense eta mutation): collapsed within 20 generations to net food < 0. Cause: every mutation added noise to every synapse's learning rate, so about half of all synapses became plastic at once and runaway Hebbian growth destroyed the inherited foraging circuit. Change: `eta_mutation_prob`, sparse on/off toggling of plasticity per synapse.
- Plastic v2 (sparse eta): survived but weak. Last 20 generations: net food 1.1, poison fraction 0.44.
- Control (no plasticity): last 20 generations net food 4.2, poison fraction 0.30, but fitness swung between 2 and 22 from one generation to the next.
- Conclusion: with one poison assignment per generation, evolution simply chases the current assignment (the control's 0.30 is an inherited preference, not learning), and the swing drowns any plasticity signal. Change: poison assigned per episode within a generation (half the worlds each way), implemented by letting `rules_for_generation` return rulesets stacked along the episode axis.

## 2026-09-18 exp02 v3: poison per episode, 100 generations (logs: exp02_plastic_v3.log, exp02_control_v3.log)

| variant | net food (last 25 gens) | poison fraction (last 25 gens) |
|---|---|---|
| control, no plasticity | 8.7 | 0.23 |
| plastic (sparse eta, ABCD) | 2.4 | 0.34 |

- The control discriminates (0.23 vs 0.5 chance) *without* synaptic plasticity. Explanation: `pain` is a body input that decays by 0.9 per tick (about 20 ticks visible), and a bush gives six berries, so "when in pain, stop eating or leave this bush" is a purely reactive policy that avoids most poison. The environment provides the memory.
- The plastic variant is worse: mutating A, B, C, D and toggling eta perturbs the inherited circuit more than it helps, and there is nothing plasticity can do that the reactive policy does not already do.
- Conclusion: exp02 does not yet *require* lifetime learning. Status stays "running" until the task forces memory across time: make pain brief (`pain_decay` about 0.3, so it is gone within 2-3 ticks) and mix bush types so that a single bush no longer gives repeated samples; then a reactive policy cannot tell the berries apart and only an association learned earlier in life can. See docs/ROADMAP.md "Next steps for exp02". Run the control alongside every time: it is the test of whether plasticity is really needed.
- Framework status: every mechanism used here worked as designed (warm start, per-episode rules, sparse eta mutation, recording, viewer). What remains is experiment design, which is cheap to iterate: a 100-generation pair of runs takes under 4 minutes on the CPU.

## 2026-09-18 direction change: tooling over tuning

- The user decided: one life per genome (episodes averaging off, ADR-012 amended), and simulation tuning is theirs to do later. All numbers above that used 4 episodes are historical; expect noisier curves now.
- Added since: region/projection brain architecture with per-projection rules and a reward modulator (ADR-013), weight snapshots in recordings, the interactive `dashboard.html` with OHOL sprites, and `Lab` for stepping worlds by hand (ADR-014). Experiment files were kept but not re-tuned.
