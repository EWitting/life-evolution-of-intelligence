# Probes

Small analysis scripts for understanding *why* a stage works or does not. Run from the project root, for example
`uv run python scripts/probes/precision.py s1_1_valence`. Most take experiment names (directories under `runs/`) or
stage keys. They are diagnostic tools, not part of the stage pipeline, and have no tests.

Protocol for a new circuit or learning rule (2026-10-03): evolution runs are slow and their fitness is mostly luck,
so check a design in this order:

1. **Mechanism, brain only** (`assay.py`): scripted experience, then action probabilities and cell activities in
   fixed situations. Does the circuit compute what it is meant to?
2. **Generation 0, paired** (`gen0.py`): the parent population with the new mechanism on and off in the same
   worlds. Does it help, or at least not hurt, before any evolution?
3. **Evolution** (`stages <key>`, `replicate`, `summary`, `lesions`, `versus`): tuning, and a check that the circuit
   is still used.

| script | question it answers |
|---|---|
| `assay.py` | After scripted meals, what does an agent do when a bush of each type is ahead or its berry is in hand? |
| `gen0.py` | Does switching learning on help the incoming population at generation 0? (`variants.py`: brain variants) |
| `recovery.py` | Which evolution settings repair a damaged population fastest (fitness, recombination, mutation, siblings)? |
| `events.py` | In a recorded life, how do picking, eating and valence for a berry type change after the first lesson? |
| `precision.py` | What do recorded agents do in standard situations, split by good / poison / dud types? |
| `strengths.py` | How far has evolution scaled each tunable hard-wired projection from its designed value? |
| `fitness_signal.py` | Which fitness definition separates better from worse genomes most clearly (signal vs luck)? |
| `evo_check.py` | Are generations and mutation settings sensible: curve, repeatability of one life, mutational load? |
| `calib.py` | Does the parent population keep its lifetime when it enters a stage's world (no cliff)? |
| `eco.py` | Economy and behaviour of the last generation: meals, actions, camping, saturation, seconds per generation. |
| `curve.py` | A column of `fitness.csv` in ten blocks of generations. |
| `prune.py` | Delete superseded runs: keep the newest finished run of every stage directory. |
