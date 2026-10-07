# Brief: the interactive book

**Status 2026-10-07: built.** The pipeline, the 1.1 pilot page and the light dashboards exist. How to keep the
book up to date after a run, and how to add a figure for one stage: `book/README.md`. This brief stays as the
record of what the user asked for and why. Still open: publishing (GitHub Pages, `gh-pages` branch), the text of
the other stage pages (written when each stage is frozen), evaluation data for 1.0 and 1.5.

Written 2026-10-07 for a separate Claude Code session. The experiment session keeps running chapter 1 in the same
working tree while you work on this; see "Working next to the experiment session" at the end.

## What the user wants

The project evolves brains stage by stage (`docs/STATUS.md`, `docs/BRAIN_EVOLUTION.md`). Its results now live in a
lab notebook (`docs/STAGE_LOG.md`) and in one self-contained dashboard per run. The user wants them packaged as an
interactive **book** with more educational value: one page per stage, each a small scientific report that also
introduces the biology and builds intuition. Per stage:

- a short narrative explanation of the mechanism and of the world it lives in, cohesive from stage to stage, with
  the biological mechanism it models;
- the results: a table, graphs, a diagram of the brain;
- links to the dashboards of the related runs;
- where possible a **zoomed-in concrete example**: one animal over a few dozen ticks, for example eating a bad
  berry and its valence cells and synapses changing, the way the simulation itself was debugged.

## Decisions already taken with the user

- **Format: a Quarto book published to GitHub Pages.** Chapters are Markdown (`.qmd`); interactive parts are plain
  JavaScript on static data files, with no server. Quarto is not installed yet, and the repository has no git
  remote yet. Ask the user before installing software, creating a remote or publishing anything.
- **Prose and numbers are separate.** Each stage page reads data files written by an export command, so rerunning
  a stage updates tables, graphs and diagrams without editing the page. Only the narrative is written by hand.
- **Standard export first, tailoring in small amounts.** The user's ideal is that figures and summaries are chosen
  per stage in detail: one stage is about evolution, another about learning within a life, and the events and
  values worth showing differ (valence over time, body temperature, weights). That is too much for now. Build a
  standard export that every stage gets, custom text per stage, and a limited number of tailored zoomed-in
  examples, probes or metrics. Design the export so a stage can add its own extras later without changing the
  standard part (a per-stage hook, extra named series in the same file).
- **Timing.** Build the pipeline and one pilot page now. Write each stage's narrative when that stage is frozen
  (STATUS, next steps, point 6). Do not write pages for drives, affect or habituation yet: their wiring will change.
- **Pilot stage: 1.1 valence.** It has the most settled mechanism. Its numbers are being rerun (v24, born a quarter
  full), so treat every number as provisional and never copy numbers into prose.

## What to build

1. **Skeleton**: a `book/` directory with the Quarto project, an introduction page and one page per chapter-1
   stage in the chain order of STATUS (stubs except the pilot).
2. **Standard export**, for example `python -m life.book export <stage key>`, writing `book/data/<stage>.json`
   (small, committed; `runs/` is not in git). Content for every stage:
   - main and control fitness curves per seed, and the last-50-generation summary with standard errors
     (`stages summary`);
   - head to head (`stages versus`) and lesions per region and seed (`stages lesions`);
   - the layout and projections for the brain diagram, with the hard-wired projections marked;
   - the world and stage settings worth showing, and the run directories the data came from;
   - a slot for per-stage extras.
   `summary`, `versus` and `lesions` in `life/experiments/stages.py` already return these values as well as
   printing them; `seed_runs` finds the runs.
3. **Page components** in plain JavaScript, shared by all pages: results table, fitness curves with seeds, lesion
   chart, brain diagram. `life/dashboard.html` already draws the architecture (`renderArch`, `autoLayout`) and the
   world (`drawWorld`); reuse that code where it fits.
4. **One zoomed-in example for 1.1**: a strip the reader scrubs through, with a small view of the world around one
   animal, the activity of the few cells that matter (appetitive, aversive, `no_feed`, `grasp`) and the action
   taken. A few kilobytes of JSON. `scripts/probes/events.py` follows individuals through a recorded life and
   `scripts/probes/assay.py` runs scripted experience on the brain alone; both are starting points. In 1.1 nothing
   is learned within a life, so the example is the innate reaction to a bad berry; the learning version (weights
   stepping up after pain) belongs to 1.5.
5. **Light dashboard export.** The dashboards are 33 MB (stage 1.0) to 71 MB (stage 1.5) each; GitHub Pages allows
   100 MB per file and about 1 GB per site. Add an option to `life/dashboard.py` aiming under 10 MB, and link one
   reference run per stage. Keep all 64 agents and the full life in the world view (user, 2026-10-07) and cut the
   three arrays that make up almost all of the size (measured on a stage 1.5 run, 70 MB page):
   - activations `x`, 19.5 MB, and weight snapshots `w_snap`, 22.2 MB: store them for a few agents only (the best
     one and a few chosen ones); the page must then show which agents have a brain view;
   - the world `grid`, 21.8 MB: it is stored in full every tick although few cells change per tick; store the
     first tick and the changes.
   Everything else (positions, actions, food, pain, modulators) is about 4 MB together and can stay for all agents. Publish built output to a separate branch so
   it stays out of the history of `main`.

## Style

- The user reads everything: plain language, short paragraphs, terms explained when first used. He prefers
  realism and cells with a fixed meaning he can refer to by name.
- A stage is judged by whether its circuit is **used** (lesions), with main not worse than control; it does not
  have to beat control (STATUS, "How a stage is judged"). Present results that way. Three seeds and standard
  errors; a single run is not evidence.
- Show what did not work where it teaches something (the unused appetitive side of 1.1 born full, the reversal
  stage left out).

## Working next to the experiment session

- The experiment session runs up to five JAX processes and edits `life/experiments/stages.py`, `docs/STATUS.md`
  and `docs/STAGE_LOG.md`. Keep to your own files: `book/`, a new `life/book.py`, the dashboard option. If you
  need a change in `stages.py`, keep it small and say so in the commit message.
- Run at most one JAX process at a time, and prefer reading existing runs over simulating.
- Runs older than 2026-10-03 21:40 are born full (before v24) and are for comparison only. The export should take
  whatever `seed_runs` returns at the time and record the run directories, so a later re-export picks up the
  reruns.
- You may commit without asking (the user gave that permission for this project). Commit only your own files.
