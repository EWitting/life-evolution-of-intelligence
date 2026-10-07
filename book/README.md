# The book

A Quarto book with one page per stage. Design and decisions: `docs/BOOK_BRIEF.md`.

Prose and numbers are separate. A page is Markdown (`.qmd`) with placeholders such as
`<div data-life="curves" data-stage="1.1"></div>`. `js/life-book.js` fills them from `data/<stage>.json`, which
an export command writes from the runs. Rerunning a stage and exporting again updates every table, graph and
diagram; only the text is written by hand. Never copy a number into the text.

## After a stage has been (re)run

    python -m life.book export 1.1 --evaluate    # curves, summary, brain + lesions, head to head (simulates)
    python -m life.book export 1.1               # the same without simulating; keeps the earlier evaluation
    python -m life.book dashboards 1.1           # light dashboards of seed 0 (main, control) -> book/dashboards/

Commit `data/*.json`. `dashboards/` and the rendered `_book/` are not committed to `main`.

The export takes the newest finished run of every seed (`stages.seed_runs`) and records the run directories. The
page then warns by itself when the runs differ from the current stage definition, when there are fewer than three
seeds, or when the lesions were measured on older runs than the curves. Run at most one `--evaluate` at a time
next to other experiments: it is a JAX process.

After a rerun, also read the page's text. Most of it describes the mechanism and stays true, but a paragraph or
callout that describes a *result* (for example "What did not work" on the 1.1 page) may no longer match the
figures and has to be rewritten by hand.

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
