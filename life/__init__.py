"""Life: evolving, learning rate-coded brains in an OHOL-style 2D world.

Read docs/DECISIONS.md before changing anything. Module map:
- actions.py    the fixed action enum (ADR-005)
- config.py     all tunable knobs as frozen dataclasses with documented defaults
- ruleset.py    the object/transition table in compact local ids (ADR-004)
- ohol.py       parser for One Hour One Life data and slicing into a Ruleset
- world.py      world state and the jitted world step (ADR-003)
- sensors.py    egocentric vision, body and sound observations (ADR-005/006)
- brain.py      regions + projections compiled onto one masked rate network with per-projection rules (ADR-013)
- evolution.py  selection and mutation over batched genomes (ADR-008)
- run.py        generation loop, logging and recording (ADR-009, ADR-014)
- dashboard.py  self-contained interactive dashboard.html for a run (ADR-014); page is dashboard.html
- lab.py        step a world by hand, inspect agents, export to the dashboard (ADR-014)
- sprites.py    composite OHOL sprites for the dashboard (viewer only)
- viewer.py     simple matplotlib replay / gif export
- experiments/  one file per experiment: ruleset slice + config + fitness (ADR-010)
"""
