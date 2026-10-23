# The book

A Quarto book with one page per stage, published at https://ewitting.github.io/life-evolution-of-intelligence/.

## How the pages are written

- One page per stage on the chain, in chain order, by name only (no stage numbers). The book teaches before it
  documents: a reader who skims the bold sentences and the figures should get the point. A page has a short title
  with a subtitle line (`[...]{.lb-subtitle}`), a takeaway box (`::: {.lb-takeaway}`: what is new, what it did),
  a short introduction, the brain diagram, the circuit, where it helps one creature followed over a few dozen ticks
  (`strip`), the biology, the world (short, with its settings), **Did it work?** (the `verdict` component, two or
  three sentences that say which bar to look at, and at most one more figure that makes the point, often `bars`),
  **Full results** (collapsed callouts: curves, the full table, all lesions, life tables, what did not work) and
  the runs. Text above the world section must not depend on it.
- Background that every page assumes (neurons, inhibition, diagrams, evolution, glossary) is on `background.qmd`;
  a part of a real brain is explained on the page of the stage that models it.
- Titles and headings are one to three words, so that the sidebar and contents fit on one line.
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
    python -m life.book export 1.1 --no-extras   # quick: brain and curves only (e.g. for the layout tool); keeps earlier extras
    python -m life.book dashboards 1.1           # light dashboards of seed 0 (main, control) -> book/dashboards/,
                                                 # and the data file again, so that the page links them

Commit `data/*.json`. `dashboards/` and the rendered `_book/` are not committed to `main`.

### The front page

Two visuals on `index.qmd` show the latest stage of the chain (the brain layout's `chain`):

- **The growing brain** (`growth` below) reads `data/growth.json`, which every export writes again (or
  `python -m life.book growth`): it follows by itself. Stage names are the titles of the pages whose `status` line
  shows the stage. It draws every part of every stage in the brain layout's `union` (all stages' regions and
  sockets together); what a stage removes or rewires turns red and fades away. Save the layout tool once after a
  stage that adds or removes regions, else the animation falls back to the latest stage's layout and leaves out
  regions that only earlier stages have.
- **The clip** (`media/front-light.mp4` and `front-dark.mp4`, posters `front-*.jpg`; the page plays the one of its
  theme) is recorded from the latest stage's published light dashboard in the dashboard's hero mode (`?hero`: world
  and brain only, on the book's background, connections glowing with their activity). It does **not** follow by
  itself. After a new stage joins the chain and its dashboards are published (`python -m life.book dashboards <key>`):

      python scripts/front_clip.py --pick      # good scenes of the new stage: creature and first tick (bites, pains)
      # write the one you like to media/front_clip.json: {"stage": "<key>", "agent": <i>, "t0": <tick>, "ticks": 200}
      python scripts/front_clip.py             # records light and dark (about a minute each)

  Look at `media/front-light.jpg` and `front-dark.jpg` (or the page), then commit `media/`. A scene saved for an
  older stage is ignored (the script then takes the best `--pick` itself), so the clip never shows an old brain by
  accident, but a scene picked by eye is usually better. Needs `uv pip install -e .[media]` (playwright, which uses the
  installed Edge, and ffmpeg from imageio-ffmpeg). Options: `--agent`, `--t0`, `--ticks`, `--theme`.

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
chapters. The senses and motor neurons are drawn in one of three forms, chosen in the tool: a block with a socket
per sense or group of actions, a tall block without sockets, or one small box. Cells and group boxes take the colour
of their top-level group (hues in the same file, also used by the dashboards). The file is
made with `tools/brain_layout` (`python tools/brain_layout/serve.py`, then http://localhost:8765): it optimises the
layout over the chain of stages with tunable weights and writes the file on Save. The chain is the stages the
pages show (`data-stage="..."` in `book/*.qmd`); data files of stages taken off the chain are ignored. A stage that the file does not
have (a new stage) gets the old layout in columns by depth until the tool is run again (the tool reads the brains
from `data/*.json`, so a quick `export <key> --no-extras` after the first finished run is enough to lay out a new
stage before its page is written); use its forward mode, or
pin the existing regions, so that the earlier diagrams do not change. Dashboards exported after the file exists
offer the same layout (link "book layout" under the architecture diagram).

## Hover texts of the diagrams

Hovering a cell, a group box or a modulator line in a brain diagram shows a short text from `data/glossary.json`
(kept by hand, not written by the export): `cells` by region name, `groups` by their **full path**
(`valence/gates`, `forebrain/basal_ganglia`: two groups may share a last name), `modulators` by name. One or two
plain sentences on what it does, in the book's words ("creature", no numbers that a run could change).

**A stage that adds a cell, a group or a modulator adds its entry in the same change.**
`tests/test_book.py::test_glossary_covers_every_diagram` fails for every region, group path or modulator of an
exported stage that has no entry. When a cell's role changes in a later stage, update its text: there is one text
per name for all pages.

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
| `brain` | regions and projections; hard-wired dark, new in this stage outlined; hover texts from `data/glossary.json`, the legend behind a "Legend" link under the diagram | |
| `curves` | main and control over the generations, every seed and the mean | |
| `summary` | results table: last generations, re-evaluation, head to head | `--evaluate` for the last two |
| `lesions` | fitness with each region silenced, per seed | `--evaluate` |
| `runs` | run directories and dashboard links | `dashboards` for the links |
| `strip` | one creature over a few dozen ticks (`data-example="<name>"`), OHOL sprites, ☠ on poison kinds | a per-stage extra; `python -m life.book sprites` |
| `verdict` | is the circuit used (`data-used="<lesion>,<table extra>:<label>"`), new against old brain (apart, sharing a world) | `--evaluate` |
| `bars` | rows of a table extra as bars with the seeds (`data-rows`, `data-col0`, `data-scale`, `data-unit`) | a per-stage extra |
| `growth` | front page, no `data-stage`: the brains of the chain one after the other, in the layout of all stages together (`union`), without names; what a stage removes fades away | `data/growth.json` (every export) |

The brain diagram has three views when the stage adds something: *new circuit* (the default: what the stage added
and what it connects to directly; a group none of that touches is an empty outline with its name in the middle),
*all cells* (the same with every other cell faded) and *everything*. Regions never move between the views.

`data/sprites.json` holds the sprites of every object in a strip (recoloured like the dashboards) and the ground
texture: `python -m life.book sprites` after exporting a stage with a new strip.

## Adding something for one stage

Register an extra in `life/book.py`; it lands under `extras` in the stage's data file and leaves the standard
part alone:

    @extra("1.5")
    def first_lesson(stage, main_runs, control_runs):
        return "first_lesson", life_strip(main_runs[0], agent, t0, t1, regions=(...), inputs=("pain",))

`life_strip` cuts a window out of a run's recording for the `strip` component. A new kind of figure needs a new
component in `js/life-book.js` (a function `(root, data)` added to `COMPONENTS`).
