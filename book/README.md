# The book

A Quarto book with one page per stage, published at https://ewitting.github.io/life-evolution-of-intelligence/.

## How the pages are written

- One page per stage on the chain, in chain order, by name only (no stage numbers). A page is a small report,
  with the core first and detail below it: a short introduction that names what the stage adds and just enough
  of its world to follow, the brain diagram, the circuit, where it helps one animal followed over a few dozen
  ticks (`strip`), the biology the stage models, the world it lives in (with its settings), the results
  (curves, table, lesions) and the runs with links to the dashboards. Text above the world section must not
  depend on it; a term from the biology section that the circuit needs is explained there in a few words.
- Plain language, short paragraphs, terms explained when first used; cells are named by their fixed meaning.
- A stage is judged by whether its circuit is used (lesions), with main not clearly worse than control. Three
  seeds and standard errors; a single run is not evidence.
- The page is about the brain: world tuning and tried variants in a few sentences, few raw numbers. Show what did
  not work where it teaches something about a circuit.
- The text is written when a stage is frozen.

Prose and numbers are separate. A page is Markdown (`.qmd`) with placeholders such as
`<div data-life="curves" data-stage="1.1"></div>`. `js/life-book.js` fills them from `data/<stage>.json`, which
an export command writes from the runs. Rerunning a stage and exporting again updates every table, graph and
diagram; only the text is written by hand. Never copy a number into the text.

## After a stage has been (re)run

    python -m life.book export 1.1 --evaluate    # curves, summary, brain + lesions, head to head (simulates)
    python -m life.book export 1.1               # the same without simulating; keeps the earlier evaluation
    python -m life.book dashboards 1.1           # light dashboards of seed 0 (main, control) -> book/dashboards/,
                                                 # and the data file again, so that the page links them

Commit `data/*.json`. `dashboards/` and the rendered `_book/` are not committed to `main`.

The export takes the newest finished run of every seed (`stages.seed_runs`) and records the run directories. The
page then warns by itself when the runs differ from the current stage definition, when there are fewer than three
seeds, or when the lesions were measured on older runs than the curves. Run at most one `--evaluate` at a time
next to other experiments: it is a JAX process.

After a rerun, also read the page's text. Most of it describes the mechanism and stays true, but a paragraph or
callout that describes a *result* (for example "What did not work" on the 1.1 page) may no longer match the
figures and has to be rewritten by hand.

## Brain diagram layout

The diagrams place the regions from `data/brain_layout.json`: one grid position per region, shared by all stages
(empty rows and columns of a stage are left out), so that a region stays where the reader saw it in earlier
chapters. The senses and motor neurons are blocks with a socket per sense and per group of actions. The file is
made with `tools/brain_layout` (`python tools/brain_layout/serve.py`, then http://localhost:8765): it optimises the
layout over the chain of stages with tunable weights and writes the file on Save. A stage that the file does not
have (a new stage) gets the old layout in columns by depth until the tool is run again; use its forward mode, or
pin the existing regions, so that the earlier diagrams do not change. Dashboards exported after the file exists
offer the same layout (link "book layout" under the architecture diagram).

## Preview and publish

    quarto preview book                          # local preview in the browser, reloads on save
    quarto publish gh-pages book                 # renders and pushes the site to the gh-pages branch

GitHub Pages serves the `gh-pages` branch (repository Settings, Pages). On a free GitHub account Pages needs a
public repository.

## Components (`data-life="..."`)

| component | shows | needs |
|---|---|---|
| `status` | a note when the runs are provisional, fewer than three seeds, or not evaluated | |
| `settings` | world, senses and evolution settings, in a box that is closed until clicked (`data-open` opens it) | |
| `brain` | regions and projections; hard-wired dark, new in this stage outlined | |
| `curves` | main and control over the generations, every seed and the mean | |
| `summary` | results table: last generations, re-evaluation, head to head | `--evaluate` for the last two |
| `lesions` | fitness with each region silenced, per seed | `--evaluate` |
| `runs` | run directories and dashboard links | `dashboards` for the links |
| `strip` | one animal over a few dozen ticks (`data-example="<name>"`) | a per-stage extra |

## Adding something for one stage

Register an extra in `life/book.py`; it lands under `extras` in the stage's data file and leaves the standard
part alone:

    @extra("1.5")
    def first_lesson(stage, main_runs, control_runs):
        return "first_lesson", life_strip(main_runs[0], agent, t0, t1, regions=(...), inputs=("pain",))

`life_strip` cuts a window out of a run's recording for the `strip` component. A new kind of figure needs a new
component in `js/life-book.js` (a function `(root, data)` added to `COMPONENTS`).
