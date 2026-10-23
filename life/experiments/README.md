# Stages: where things are

A stage is a world, a brain built on the parent stage's brain, and a fitness (`base.Stage`). The plan is in
`docs/BRAIN_EVOLUTION.md`, results in `docs/STAGE_LOG.md`, what is frozen in `frozen/README.md`.

| file | what is in it |
|---|---|
| `stages.py` | the command line (`python -m life.experiments.stages ...`) and a re-export of everything below, so `from life.experiments import stages as S; S.<name>` works for every name |
| `base.py` | `Stage`, the registry `STAGES` and `stage()`, fitness functions, `extend` and `fixed` (building a brain on another), action ranges on the motor neurons (`FWD`, `TURNS`, `USE_`, `EAT_`), evolution settings of new stages (`EVOLUTION_2EP`) |
| `worlds.py` | the berry world (`berry_world`: OHOL bush types, looks fixed or drawn per life, XOR looks, poison, one-berry bushes, reversals), `poison_metrics`, `calm_first` |
| `chapter1.py` | the chain of chapter 1, in order: 1.0 steering, 1.1 valence, 1.2h habituation, 1.5 association, 1.7 drives, 1.8 affect, 2.2 pallium (all frozen). Each section: why, the brain, the world, the stage |
| `chapter2.py` | chapter 2: the critic and the drafts and frozen stages on the way to it (2.3, 2.3h, 2.3c, 2.4, 2.3s on the chain), then hands (`with_hands`), the start of the actor stage |
| `side_paths.py` | everything not on the chain: parked branches (1.2-1.4), side paths (1.1l, 1.5c, 1.5f, 1.6h, 1.6, x.hands), the first affect route (1.8old, 2.0, 2.1), the first chapter 2 design (old2.x, x.td) and trials (x.*). Names the chain also uses carry a suffix here (`W12_PARKED`, `W22_OLD`, `B23_OLD`, `B24_OLD`) |
| `evaluate.py` | `make_exp`, `run_stage`, `resume_stage`, `lesion` / `lesions`, `versus`, `respond`, `seed_runs` |

Frozen stages must not change: `tests/test_frozen.py` compares each with its stored definition.

## Adding a stage

In the chapter's file, in this order: a comment block (biology, what is new and why, what generation 0 and the lab
tests showed), the brain as a function of the parent's (`extend(...)` or a transform like `with_hands(brain)`), the
world (`replace(<parent world>, ...)` plus a builder from `worlds.py`), then `stage(Stage(key, run_name, parent, brain,
world, vision, body, build, ..., control_brain=..., evolution=EVOLUTION_2EP))`. New cells need a hover text in
`book/data/glossary.json`; check the wiring by mirror first (`python -m life.mirror_check <key>`, CLAUDE.md).

A few traps:
- `remap_genomes` carries the plasticity genes A-D over for a projection that keeps its rule, so a changed `abcd`
  on such a projection is ignored (`brain.remap_genomes`).
- Hard-wired outputs of tonically active cells are not tuned (`fixed(..., tune=False)`): a tuned one moves the rest
  of the cell it drives.
- The stage's world is built for generation 0 by lesions, the book and probes; a world that changes over the
  generations goes in `Stage.schedule`, not in the builder.
