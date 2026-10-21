# Life: working rules for agents

Brains evolved stage by stage in an OHOL-like world (JAX). Start with `docs/STATUS.md` (where the project is, how a
stage is judged, next steps). Plan and principles: `docs/BRAIN_EVOLUTION.md`; decisions: `docs/DECISIONS.md`;
results: `docs/STAGE_LOG.md`.

## Checking a design: always in this order, before any evolution run

One life is mostly luck and an evolution run mixes many confounders, so fitness numbers alone say little
(ADR-022, BRAIN_EVOLUTION principle 4, `scripts/probes/README.md`).

0. **Wiring review, by hand.** A text agent does not see the diagram, so read the brain as the user reads it: for
   every new or changed cell, list its inputs and outputs and compare them with its counterpart on the other side
   (good / bad: `_app` / `_av`, taste / pain, `da` / `da_neg`, dopamine / habenula). Every difference must be meant
   and written down; check the same for the inputs that one side takes through a relay cell and the other directly,
   and for signals or cells that nothing reads any more. Then run `python -m life.mirror_check <key>` (its side by
   side table and flags; `tests/test_mirror_check.py` fails on an unexplained flag) as a backstop, not a replacement.
1. **Lab tests, brain only.** One creature in scripted situations, as in an animal lab. For learning stages, use
   textbook experiments from the literature (acquisition, extinction, blocking, second-order conditioning,
   discrimination of look-alikes, devaluation, ...): their predictions are known, so the results are easy to
   read. Code: `life/labtests.py` (`xor_meals`, `PROTOCOLS` + `conditioning`), drivers
   `scripts/probes/lab_conditioning.py`, `assay.py`, `assay_xor.py`. The results also go into the book, next to
   the world results.
2. **Generation 0, paired** (`scripts/probes/gen0.py`): the parent population with the mechanism on and off in
   the same worlds. A diagnostic, not a gate; balances and scales may need adjusting at generation 0 when a
   module is added.
3. **Evolution**: one seed first, three seeds for a claim; the book export evaluates in 8 worlds (16 only to
   compare with a stage measured in 16) and lesions only what the stage's entry in `life/book.py` `LESIONS` names,
   each with the question it answers (a motivated choice, not every new cell); then look inside the finished run (lesions, learned
   weights, traces) before starting the next one. Stop runs early once trends are clear and the main - control
   outcome is unlikely to change (read `checkpoint.npz`, written every 10 generations); convergence is not required.

## Standing rules

- No new mechanisms (new connections, cell types, inputs) without agreeing them with the user; balance and scale
  changes at generation 0 are fine.
- Frozen stages (`frozen/`) are not rerun; changes go by remapping, and a lineage rerun or refreeze needs the user.
- Never switch evolution off for a gene; if evolution turns a feature down, find out why it does not pay.
- Cheap tests first: one stage, one seed, generation 0 where possible.
- GPU runs in WSL2 (`docs/SETUP.md`), at most two JAX processes at once (2-2.5 GB each), started minutes apart. After a kill,
  restart (from the last checkpoint where possible) only when a second try is likely to get through: the cause was
  fixed, or the memory pressure was a one-off (fewer runs at once, laptop less busy); otherwise report it.
- The book is about brains; simulated individuals are "creatures" there ("animal" only for real animals).
