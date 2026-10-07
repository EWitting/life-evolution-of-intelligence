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
0. **On paper** (`classify.py`): can the learning rule separate good from poison at all, with the looks the stage
   uses and the way the animals bite?
3. **Evolution** (`stages <key>`, `replicate`, `summary`, `lesions`, `versus`): tuning, and a check that the circuit
   is still used.
4. **Inside the finished run, before the next one** (2026-10-07): lesions and fitness are not enough. `learned.py`
   (rates, what changed in the weights, teacher activity), `forage.py` and `freeze.py` (how a life is spent),
   `supply.py` (food over a life), `lessons.py` (how often a lesson can be used, what perfect learning is worth).

Try a change on the smallest informative run first: one stage, one seed, generation 0 where possible. Whole chain
and three seeds only when a design looks final.

| script | question it answers |
|---|---|
| `assay.py` | After scripted meals, what does an agent do when a bush of each type is ahead or its berry is in hand? |
| `gen0.py` | Does switching learning on help the incoming population at generation 0? (`variants.py`: brain variants) |
| `drives.py` | What does the cold cost the incoming population, and which hard-wired warmth circuit (kinesis parts, hunger gate) wins it back at generation 0? |
| `ceiling.py` | How many agents reach the life cap, and what is the fitness at longer caps? |
| `recovery.py` | Which evolution settings repair a damaged population fastest (fitness, recombination, mutation, siblings)? |
| `events.py` | In a recorded life, how do picking, eating and valence for a berry type change after the first lesson? |
| `precision.py` | What do recorded agents do in standard situations, split by good / poison / dud types? |
| `strengths.py` | How far has evolution scaled each tunable hard-wired projection from its designed value? |
| `fitness_signal.py` | Which fitness definition separates better from worse genomes most clearly (signal vs luck)? |
| `evo_check.py` | Are generations and mutation settings sensible: curve, repeatability of one life, mutational load? |
| `calib.py` | Does the parent population keep its lifetime when it enters a stage's world (no cliff)? |
| `eco.py` | Economy and behaviour of the last generation: meals, actions, camping, saturation, seconds per generation. |
| `curve.py` | A column of `fitness.csv` in ten blocks of generations. |
| `classify.py` | No brain, no world: does a learning rule turn bites into a good / poison classifier? (`BLOCK=hard`: bites only while aversion is low) |
| `learned.py` | What was learned within a life: inherited rates, change of each type's drive on the aversive cells after the first bite, teacher activity. |
| `lessons.py` | Per life: bush visits, repeat visits to a poison type that already hurt, and what a perfect learner would save. Also the cliff check for a learning world. |
| `supply.py` | Edible bushes carrying berries, animals alive and mean stomach over a life (stock or flow?). |
| `forage.py` | How a life is spent: moving, facing a bush, grasping at nothing, berries per life. |
| `freeze.py` | Episodes of standing and grasping at nothing: what is in front, what started them, what the cells do. |
| `record.py` | Record one generation-0 life of a population on a brain variant, readable by `learned.py` and the dashboard. |
| `prune.py` | Delete superseded runs: keep the newest finished run of every stage directory. |

Common environment variables: `STAGE=<key>`, `RUN=<run directory>` (another population than the parent's newest),
`TICKS=<life length>`, `WORLDS=<n>`, `DENSE=1` (every learned synapse present). `variants.py` lists brain variants.
