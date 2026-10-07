# Life

A hobby framework for evolving and learning rate-coded brain models inside a 2D world built from
One Hour One Life (OHOL) objects and recipes. Goal and motivation: `Intention.md`.

**Start here, in this order:**
1. `docs/STATUS.md`: where the project is: every stage of the brain-evolution sequence, its status and findings,
   future goals, known simplifications.
2. `docs/BRAIN_EVOLUTION.md`: the plan: the sequence of brain architectures and experiments.
3. `docs/STAGE_LOG.md`: the detailed lab notebook behind STATUS (runs, numbers, what went wrong and why).
4. `docs/DECISIONS.md`: design decisions (ADRs). Guidance from the initial build; newer ADRs supersede older ones.
5. `docs/SETUP.md`, `docs/OHOL_FORMAT.md`, then `life/__init__.py` for the module map.

Results are published as an interactive book, one page per stage:
**https://ewitting.github.io/life-evolution-of-intelligence/** (sources in `book/`; how to update it after a run:
`book/README.md`).

## Quick start (Windows, CPU)

```
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m life.experiments.stages list
.venv\Scripts\python.exe -m life.experiments.stages 1.0 --generations 200          # root of the lineage
.venv\Scripts\python.exe -m life.experiments.stages 1.1 [--control]                 # warm-starts from newest 1.0 run
.venv\Scripts\python.exe -m life.experiments.stages chain 1.1 1.1 --mutation-prob 0.1   # main + control of a stage
start runs\s1_1_valence\<timestamp>\dashboard.html
```

Inspection tools (all take a run directory):

```
python -m life.compare s1_1_valence s1_1_valence_control        # first/last 25-generation averages of fitness.csv
python -m life.experiments.stages lesion  <run>                  # silence each region (and plasticity), measure
python -m life.experiments.stages respond <run>                  # region activity / action probabilities per object
python -m life.dashboard <run>                                   # regenerate dashboard.html
python -m life.dashboard <run> --light                           # dashboard_light.html, a few MB (for the book)
python -m life.book export 1.1 --evaluate                        # the stage's data file for the book (book/README.md)
```

Every run directory contains `dashboard.html`: the first panel shows the architecture (regions as blocks grouped
into modules, projections as arrows, neuromodulators as diamonds); click a block or arrow for its activations and
weights. Then the world, the focused agent's senses, activations over time, the full weight matrix and fitness.

## Poking at things by hand (Python or Jupyter)

```python
from life.lab import Lab
from life.experiments import stages as S
from life.run import load_population, latest_run
s = S.STAGES["1.1"]; exp = S.make_exp(s, s.brain, "lab", 1, 0)
rs, rules_fn = s.build(exp)
lab = Lab(exp, rs, population=load_population(latest_run("s1_1_valence"), exp), rules=rules_fn(0, None))
lab.step(100)                      # advance
lab.agent(3), lab.obs(3), lab.brain(3), lab.world_summary()
lab.step(10, override={3: 4})      # force agent 3 to USE for 10 ticks
lab.export("runs/lab_session")     # dashboard.html of everything stepped so far
```

## How the pieces fit

```
Ruleset (OHOL slice)  ->  RuleArrays  --+
                                        |-> step_world(cfg, rules, state, actions) -> state, events
WorldState (grid + agents)  ------------+        ^                                      |
        |                                        |                         world signals (food, pain)
        v                                        |                                      v
observe_all -> obs (named input features) -> brain.step (vmap over agents) -> actions  (optional)
                                                  ^
             Genome (w0, mask, b, eta, A, B, C, D) per agent + Layout (regions, projections, modulators)
                                                  ^
                                   evolution.next_generation(fitness)
```

One generation = `lax.scan` over T ticks of (observe, brain step, world step), jitted as a whole.

## Defining a brain (ADR-013, ADR-015, ADR-016)

```python
from life.config import BrainConfig, RegionSpec as R, ProjectionSpec as P, ModulatorSpec as Mod
BrainConfig(
    regions=(R("valence_av", 3, sign="exc", group="valence"),
             R("raphe", 2, sign="exc", alpha=0.03, group="affect"),
             R("dwell", 2, sign="exc", receptors=(("5ht", "bias", 2.0),), group="affect")),
    projections=(P("in", "valence_av", src_select=("pain",), density=1.0, w_init=3.0, evolve=False),  # hard-wired
                 P("in", "valence_av", src_select=("vis*.app*",), rule="hebb", modulator="us",
                   eta_init=0.05, elig_tau=0.8, abcd=(0, -1, 0, 0)),                                     # learned
                 P("valence_av", "out", dst_range=(2, 4), density=1.0, w_init=3.0, evolve=False)),    # -> turns
    modulators=(Mod("5ht", pos="raphe"), Mod("us", pos="us_taste", neg="us_pain")))
```

Regions: size, leak `alpha`, Dale's-law `sign`, `kwta`, fixed `bias`, `receptors` for broadcast modulators, visual
`group`. Projections: rule (`fixed`, `hebb`, `oja`, `trace`, `delta`), `modulator`, eligibility trace, `decay`
toward w0, short-term `depression`, `kind` (`add`/`gain`), topology (`full`, `one_to_one`, `topographic`),
`src_select` (input features by name), `dst_range`, hard-wiring (`w_init`, `evolve=False`).
Modulators: named broadcast signals from region activity (`pos`, `neg`, weighted `terms`).

## Adding things (for a future maintainer)

- **A new stage:** add it to `life/experiments/stages.py` (world, brain via `extend(parent_brain, ...)`, fitness,
  metrics), run main and control, write findings in `docs/STAGE_LOG.md` and the summary row in `docs/STATUS.md`.
- **A bigger world:** use `ohol.slice_ruleset(data, ids, expand_hops=n, clone_sets=...)`.
- **A new mechanic:** add a config field with a default that keeps old behaviour, implement in `world.py` or
  `sensors.py`, add a test in `tests/`.
- **A new learning rule:** add a name to `brain.RULES` and a branch in `brain.plasticity_rule`; add a test.
- **A new action:** append to `life/actions.py` and handle it in `step_world`. Never reorder.
- **A dashboard panel:** edit `life/dashboard.html` (plain JS; data fields in `life/dashboard.py: build_data`).
- The old standalone experiments `life/experiments/exp01_*.py` and `exp02_*.py` are superseded by the stages.
