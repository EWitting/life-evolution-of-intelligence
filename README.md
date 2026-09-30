# Life

A hobby framework for evolving and learning rate-coded brain models inside a 2D world built from
One Hour One Life (OHOL) objects and recipes. Goal and motivation: `Intention.md`.

**Start here, in this order:** `docs/DECISIONS.md` (the rules of the project), `docs/SETUP.md`,
`docs/ROADMAP.md`, `docs/EXPERIMENT_LOG.md`, `docs/OHOL_FORMAT.md`. Then `life/__init__.py` for the module map.

## Quick start (Windows, CPU)

```
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m life.experiments.exp01_evolved_forager --quick
.venv\Scripts\python.exe -m life.experiments.exp01_evolved_forager
start runs\exp01_evolved_forager\<timestamp>\dashboard.html
.venv\Scripts\python.exe -m life.experiments.exp02_hebbian_association --init-from latest
.venv\Scripts\python.exe -m life.experiments.exp02_hebbian_association --init-from latest --no-plastic
```

Every run directory contains `dashboard.html`: open it in any browser to scrub through the final generation,
click agents, and inspect their retina, activations and weights. Regenerate it with
`python -m life.dashboard <run dir>` after changing `life/dashboard.html`.

## Poking at things by hand (Python or Jupyter)

```python
from life.lab import Lab
from life.experiments import exp01_evolved_forager as e
from life.run import load_population, latest_run
exp = e.make_config(); rs = e.build_ruleset(exp.world)
lab = Lab(exp, rs, population=load_population(latest_run("exp01_evolved_forager")))
lab.step(100)                      # advance
lab.agent(3), lab.obs(3), lab.brain(3), lab.world_summary()
lab.step(10, override={3: 4})      # force agent 3 to USE for 10 ticks
lab.render(focus=3)                # matplotlib figure
lab.export("runs/lab_session")     # dashboard.html of everything stepped so far
```

## How the pieces fit

```
Ruleset (OHOL slice)  ->  RuleArrays  --+
                                        |-> step_world(cfg, rules, state, actions) -> state, events
WorldState (grid + agents)  ------------+        ^                                      |
        |                                        |                        modulator (reward = food - pain)
        v                                        |                                      v
observe_all -> obs {vision, body, sound} -> brain.step (vmap over agents) -> actions   mod
                                                  ^
                     Genome (w0, mask, b, eta, A, B, C, D) per agent + Layout (regions, projections, rules)
                                                  ^
                                   evolution.next_generation(fitness)
```

One generation = `lax.scan` over T ticks of (observe, brain step, world step), jitted as a whole.
Experiments only pick a ruleset slice, configs and a fitness function (ADR-010).

## Defining a brain architecture (ADR-013)

```python
from life.config import BrainConfig, RegionSpec, ProjectionSpec
BrainConfig(
    regions=(RegionSpec("cortex", 64, alpha=0.5), RegionSpec("striatum", 16, alpha=0.8, trace_tau=0.6)),
    projections=(ProjectionSpec("in", "cortex", density=0.3, rule="oja", eta_init=0.01),
                 ProjectionSpec("cortex", "cortex", density=0.2, rule="trace"),
                 ProjectionSpec("cortex", "striatum", density=0.5, rule="hebb", modulated=True),
                 ProjectionSpec("striatum", "out", density=1.0)),
    modulator="reward")
```
Rules: `fixed`, `hebb`, `oja`, `trace`; `modulated=True` gates the update by the modulator (three-factor).
Regions `in` and `out` exist automatically. Evolution decides per synapse whether plasticity is on
(`EvolutionConfig.plastic=True`), starting from `eta_init`.

## Adding things (for a future maintainer)

- **A new experiment:** copy `life/experiments/exp01_evolved_forager.py`, change the slice, config and fitness. Add a row to `docs/ROADMAP.md` and an entry to `docs/EXPERIMENT_LOG.md` when run.
- **A bigger world:** use `ohol.slice_ruleset(data, ids, expand_hops=n)`; inspect ids with `scripts/ohol_inspect.py`.
- **A new mechanic:** add a config field with a default that keeps old behaviour, implement in `world.py` or `sensors.py`, add a test in `tests/`.
- **A new learning rule:** add a name to `brain.RULES` and a branch in `brain.plasticity`; add a test like `test_rules_change_weights_only_where_plastic`.
- **A new brain module (e.g. basal ganglia with its own reward prediction error):** declare it as a region, compute its output inside `brain.step` after the matrix update behind a config flag, and feed it into `mod`.
- **A new action:** append to `life/actions.py` and handle it in `step_world`. Never reorder.
- **A dashboard panel:** edit `life/dashboard.html` (plain JS; data fields are documented in `life/dashboard.py: build_data`), then `python -m life.dashboard <run dir>`.
- **Never** change a **Fixed** decision without a new ADR entry.
