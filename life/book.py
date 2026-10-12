"""Data files for the interactive book (docs/BOOK_BRIEF.md): prose and numbers are separate.

    python -m life.book export 1.1               # book/data/1.1.json from the runs `stages.seed_runs` finds now
    python -m life.book export 1.1 --evaluate    # also re-evaluation, head to head and lesions (simulates: one
                                                 # JAX process, minutes per seed)
    python -m life.book dashboards 1.1           # light dashboards of the stage's lineage runs (main and control)

Every stage gets the same standard export: fitness curves per seed, the last-generations summary, head to head,
lesions, the brain (regions, projections, what is hard-wired, what is new in this stage), the settings and the run
directories the numbers came from. A stage adds its own extras with `@extra(<key>)` below; they land under
"extras" in the same file and never change the standard part.

Without --evaluate nothing is simulated: the evaluation of an earlier export is kept, and marked stale when the
runs have changed since.
"""
from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable

import numpy as np

from .run import RUNS_DIR

BOOK_DIR = Path(__file__).resolve().parent.parent / "book"
DATA_DIR = BOOK_DIR / "data"
DASH_DIR = BOOK_DIR / "dashboards"
WINDOW = 50                                  # generations averaged for the summary (as `stages summary`)
CURVE_COLUMNS = ("fit_mean", "fit_max", "alive_ticks", "eaten", "pain", "survivors", "temp_mean", "poison_frac")
WORLD_KEYS = ("height", "width", "num_agents", "spawn_density", "max_food", "start_food", "hunger_per_tick",
              "food_scale", "move_cost", "turn_cost", "eat_on_pick", "pain_decay", "temperature", "ambient_temp",
              "patches", "switch_tick", "sickness_delay")
EVOLUTION_KEYS = ("generations", "ticks_per_generation", "siblings", "crossover", "weight_mutation_prob",
                  "mutation_std", "elite_frac", "tournament", "plastic")

EXTRAS: dict[str, list[Callable]] = {}
# Stages whose change is in how the inputs work, not in a region: in the head-to-head test both halves run on the
# new brain, so the control animals would get adapting inputs their synapses were never tuned to. Not shown.
NO_HEAD_TO_HEAD = ("1.2h", "1.6h")


def extra(key: str):
    """Register a per-stage extra: fn(stage, main_runs, control_runs) -> (name, json-able value)."""
    def deco(fn):
        EXTRAS.setdefault(key, []).append(fn)
        return fn
    return deco


# ------------------------------------------------------------------ helpers

def _round(v, digits: int = 4):
    """Floats to `digits` significant digits (the files are committed: keep them small and diffable)."""
    if isinstance(v, dict):
        return {k: _round(x, digits) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_round(x, digits) for x in v]
    if isinstance(v, (float, np.floating)):
        v = float(v)
        if not np.isfinite(v):
            return None
        return float(f"{v:.{digits}g}")
    if isinstance(v, np.integer):
        return int(v)
    return v


def _decimals(v, places: int = 3):
    """Floats to `places` decimals (activities and levels: a decaying 1e-9 is 0)."""
    if isinstance(v, dict):
        return {k: _decimals(x, places) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_decimals(x, places) for x in v]
    if isinstance(v, (float, np.floating)):
        r = round(float(v), places)
        return int(r) if r == int(r) else r
    return v


def _se(v) -> float:
    return float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")


def _stat(v) -> dict:
    return {"seeds": list(v), "mean": float(np.mean(v)), "se": _se(v)}


def _rel(run: Path) -> str:
    return f"{run.parent.name}/{run.name}"


def _seed(run: Path) -> int:
    name = run.parent.name
    return int(name.rsplit("_seed", 1)[1]) if "_seed" in name else 0


def _config(run: Path) -> dict:
    return json.loads((run / "config.json").read_text())


def _rows(run: Path) -> list[dict]:
    with open(run / "fitness.csv") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def run_info(run: Path) -> dict:
    cfg = _config(run)
    start = cfg["world"].get("start_food", 1.0)
    return {"dir": _rel(run), "seed": _seed(run), "generations": cfg["evolution"]["generations"],
            "finished": datetime.fromtimestamp((run / "population.npz").stat().st_mtime).isoformat(timespec="minutes"),
            "start_food": start}


# ------------------------------------------------------------------ the standard parts

def _history(run: Path) -> list[Path]:
    """The run preceded by the runs of the same directory it continues (`continues.txt`, written by the stage runner
    when a run starts from an earlier run of its own experiment), in time order: a stage evolved from scratch and
    then settled, or a seed given more generations, shows as one curve."""
    out = [run]
    while (out[0] / "continues.txt").exists():
        out.insert(0, out[0].parent / (out[0] / "continues.txt").read_text().strip())
    return out


def curves(runs: list[Path]) -> list[dict]:
    """Per seed: one list per fitness.csv column worth plotting, one value per generation."""
    out = []
    for run in runs:
        rows = [row for r in _history(run) for row in _rows(r)]
        out.append({"seed": _seed(run), **{c: [r[c] for r in rows] for c in CURVE_COLUMNS if c in rows[0]}})
    return out


def summary(main: list[Path], control: list[Path], window: int = WINDOW) -> dict:
    """Mean over the last `window` generations per seed, then mean and standard error over the seeds."""
    out = {"window": window}
    for label, runs in (("main", main), ("control", control)):
        if not runs:
            continue
        last = [_rows(r)[-window:] for r in runs]
        out[label] = {c: _stat([float(np.mean([row[c] for row in rows])) for rows in last])
                      for c in CURVE_COLUMNS if c in last[0][0]}
    if "main" in out and "control" in out and len(main) == len(control):
        d = np.array(out["main"]["fit_mean"]["seeds"]) - np.array(out["control"]["fit_mean"]["seeds"])
        out["difference"] = _stat(d.tolist())
    return out


def _indices(layout: dict, p: dict) -> tuple[list[int], list[int]]:
    """Rows (pre) and columns (post) of the weight matrix that projection p covers."""
    off = dict(zip(layout["names"], zip(layout["offsets"], layout["sizes"])))
    o, n = off[p["src"]]
    if p["src"] == "in" and p.get("src_select"):
        rows = [i for i, f in enumerate(layout["in_names"]) if any(fnmatch(f, s) for s in p["src_select"])]
    else:
        lo, hi = p["src_range"] if p.get("src_range") else (0, n)
        rows = list(range(o + lo, o + hi))
    o, n = off[p["dst"]]
    lo, hi = p["dst_range"] if p.get("dst_range") else (0, n)
    return rows, list(range(o + lo, o + hi))


def brain_view(s, run: Path) -> dict:
    """Regions and projections of the run's brain for the diagram: what is hard-wired, what this stage added, and
    how strong each projection is in the final population (mean over genomes; no simulation)."""
    from .experiments.stages import STAGES
    cfg, layout = _config(run), json.loads((run / "layout.json").read_text())
    brain = cfg["brain"]
    parent = STAGES[s.parent].brain if s.parent else None
    old_regions = {r.name for r in parent.regions} if parent else set()
    old_proj = {(p.src, p.dst, tuple(p.src_select), tuple(p.dst_range), tuple(p.src_range))
                for p in parent.projections} if parent else set()
    spec = {r["name"]: r for r in brain["regions"]}
    regions = []
    for name, size, sign, group in zip(layout["names"], layout["sizes"], layout["sign"], layout["groups"]):
        r = spec.get(name, {})
        regions.append({"name": name, "size": size, "sign": int(sign), "group": group,
                        "new": bool(parent) and name not in old_regions and name not in ("in", "out"),
                        "receptors": r.get("receptors", [])})
    with np.load(run / "population.npz") as f:
        w0, mask = f["w0"].astype(np.float32), f["mask"].astype(np.float32)
    signs = dict(zip(layout["names"], layout["sign"]))
    projections = []
    for p in brain["projections"]:
        rows, cols = _indices(layout, p)
        w = (w0 * mask)[:, rows][:, :, cols]                       # [genomes, pre, post]
        present = mask[:, rows][:, :, cols] > 0
        n = float(present.sum(axis=(1, 2)).mean())
        mean_abs = float(np.abs(w).sum() / max(1.0, present.sum()))
        # Dale rows hold magnitudes: the sign is the source region's. Sensory rows are signed.
        sign = signs[p["src"]] or float(np.sign(w.sum()))
        key = (p["src"], p["dst"], tuple(p.get("src_select", ())), tuple(p.get("dst_range", ())), tuple(p.get("src_range", ())))
        projections.append({
            "src": p["src"], "dst": p["dst"], "src_select": list(p.get("src_select", ())),
            "src_range": list(p.get("src_range", ())), "dst_range": list(p.get("dst_range", ())),
            "hard": not p.get("evolve", True), "rule": p.get("rule", "fixed"), "kind": p.get("kind", "add"),
            "modulator": p.get("modulator", ""), "designed": p.get("w_init"), "new": bool(parent) and key not in old_proj,
            "sign": int(sign), "synapses": n, "mean_abs": mean_abs,
            "drive": float(np.abs(w).sum(axis=1).mean())})        # summed |w| arriving at one target neuron
    from . import actions as A
    # modulators: which are new, and for a lasting one how long it stays (ticks), as designed and as evolved
    old_mods = {m.name for m in parent.modulators} if parent else set()
    with np.load(run / "population.npz") as f:
        clear = f["clear"].astype(np.float64) if "clear" in f.files else None
    modulators = []
    for i, m in enumerate(brain.get("modulators", [])):
        m = dict(m, new=bool(parent) and m["name"] not in old_mods)
        if m.get("decay", 0.0) > 0:
            m["lasts_designed"] = 1.0 / (1.0 - m["decay"])
            m["lasts"] = float((1.0 / (1.0 - clear[:, i])).mean()) if clear is not None else m["lasts_designed"]
        modulators.append(m)
    return {"run": _rel(run), "neurons": layout["n"], "regions": regions, "projections": projections,
            "modulators": modulators, "actions": list(A.NAMES)}


def settings(s, run: Path) -> dict:
    cfg = _config(run)
    rs = json.loads((run / "ruleset.json").read_text())
    fit = s.fitness.__name__ if s.fitness else "default_fitness"
    return {"world": {k: cfg["world"].get(k, 1.0 if k == "start_food" else None) for k in WORLD_KEYS},
            "vision": cfg["vision"], "body": cfg["body"],
            "evolution": {k: cfg["evolution"].get(k) for k in EVOLUTION_KEYS},
            "fitness": fit, "objects": [n for n in rs["names"][1:]]}


def provisional(s, runs: list[Path]) -> list[str]:
    """Settings of the runs that differ from the stage as it is defined now: their numbers will change on a rerun."""
    from dataclasses import asdict
    now, out = asdict(s.world), []
    for k in WORLD_KEYS:
        was = {_config(r)["world"].get(k, 1.0 if k == "start_food" else None) for r in runs}
        if was != {now.get(k)}:
            out.append(f"{k}: runs {sorted(was, key=str)}, stage definition {now.get(k)}")
    return out


def _lesion_task(task) -> dict:
    from .experiments import stages as S
    _, _, run, worlds, only = task
    return S.lesion(run, worlds=worlds, only=only)


def _versus_task(key: str, worlds: int) -> list:
    from .experiments import stages as S
    return [float(d) for d in S.versus(key, worlds)]


def evaluate(key: str, worlds: int = 8, jobs: int = 5, lesions: str = "recent") -> dict:
    """The three measurements that need simulation, over every seed: the final populations re-evaluated in
    `worlds` shared worlds, with each region silenced (lesions), and main against control in the same worlds.
    lesions 'recent': only the regions that this stage or its parent added (and plasticity); 'all': every region,
    which for a late stage is most of the cost (user, 2026-10-12: the full sweep only when it is needed)."""
    from .experiments import stages as S
    s = S.STAGES[key]
    older = set()
    if lesions == "recent" and s.parent and S.STAGES[s.parent].parent:
        older = {r.name for r in S.STAGES[S.STAGES[s.parent].parent].brain.regions}
    main, control = S.seed_runs(s), S.seed_runs(s, control=True)
    out = {"worlds": worlds, "runs": [_rel(r) for r in main + control],
           "when": datetime.now().isoformat(timespec="minutes")}
    # every simulation is independent and one JAX process uses few cores, so the work is cut into tasks (a few
    # lesions of one seed each) for `jobs` processes. The worlds depend only on the seed of `lesion`, so the cut
    # does not change a number.
    from concurrent.futures import ProcessPoolExecutor
    tasks = []
    for i, r in enumerate(main):
        names = json.loads((r / "layout.json").read_text())["names"]
        labels = ["intact", "no_plasticity", "no_depression"] + [n for n in names if n not in ("in", "out") and n not in older]
        tasks += [("main", i, str(r), worlds, tuple(labels[k:k + 3])) for k in range(0, len(labels), 3)]
    tasks += [("control", i, str(r), worlds, ("intact",)) for i, r in enumerate(control)]
    import time
    from concurrent.futures import as_completed
    t0 = time.time()
    print(f"evaluate {key}: {len(tasks) + bool(control)} tasks in {jobs} processes, {worlds} worlds each", flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_lesion_task, t): f"{t[0]} seed {t[1]}: {', '.join(t[4])}" for t in tasks}
        versus = pool.submit(_versus_task, key, worlds) if control else None
        if versus:
            futures[versus] = "head to head"
        for n, f in enumerate(as_completed(futures), 1):   # one line per finished task: the progress of the export
            print(f"  {n}/{len(futures)} after {(time.time() - t0) / 60:.0f} min: {futures[f]}", flush=True)
        done = [f.result() for f in list(futures)[:len(tasks)]]
        versus = versus.result() if versus else None
    les = [{} for _ in main]
    ctrl_fit = [None] * len(control)
    for (kind, i, *_), res in zip(tasks, done):
        if kind == "main":
            les[i].update(res)
        else:
            ctrl_fit[i] = res["intact"]["fitness"]
    order = [l for l in ["intact", "no_plasticity", "no_depression"] + [n for n in names if n not in ("in", "out")] if l in les[0]]
    les = [{l: d[l] for l in order} for d in les]
    out["reevaluated"] = {"main": _stat([l["intact"]["fitness"] for l in les])}
    out["lesions"] = {label: {"percent": _stat([100.0 * l[label]["fitness"] / l["intact"]["fitness"] for l in les]),
                              "fitness": [l[label]["fitness"] for l in les]}
                      for label in les[0] if label != "intact"}
    out["intact"] = {k: [l["intact"][k] for l in les] for k in les[0]["intact"]}
    if control:
        ctrl = ctrl_fit
        out["reevaluated"]["control"] = _stat(ctrl)
        if len(ctrl) == len(les):
            out["reevaluated"]["difference"] = _stat([l["intact"]["fitness"] - c for l, c in zip(les, ctrl)])
        out["versus"] = _stat([float(d) for d in versus])
    return out


# ------------------------------------------------------------------ zoomed-in examples

DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])   # up, right, down, left (world.py)
# berry-bush look-alikes are named after colours (stages.COLOURS); the plain gooseberry bush is green
NAMED_COLOURS = {"plain": [0.33, 0.6, 0.3], "Red": [0.8, 0.26, 0.22], "Blue": [0.22, 0.47, 0.82], "Yellow": [0.88, 0.7, 0.05],
                 "Purple": [0.56, 0.32, 0.72], "White": [0.93, 0.93, 0.9], "Black": [0.15, 0.15, 0.15],
                 "Pink": [0.93, 0.55, 0.7], "Orange": [0.93, 0.5, 0.15], "Onion": [0.72, 0.6, 0.42]}


def life_strip(run: Path, agent: int, t0: int, t1: int, regions: tuple, inputs: tuple = (), radius: int = 4) -> dict:
    """A few dozen ticks of one recorded animal for the book's scrubbing strip: the world around it, what it did
    and the activity of a few named cells. Step t shows the world *before* the action of tick t (the recorded
    state of t - 1), the cells and the action of tick t, and the food level and pain after it."""
    rec = np.load(run / "recording.npz")
    rs = json.loads((run / "ruleset.json").read_text())
    layout = json.loads((run / "layout.json").read_text())
    cfg = _config(run)
    from . import actions as A
    grid, pos, dr, act, ate, food, pain, alive, x = (rec[k] for k in ("grid", "pos", "dir", "action", "ate", "food", "pain", "alive", "x"))
    T, N = act.shape
    H, W = grid.shape[1:]
    t0, t1 = max(1, t0), min(T, t1)
    off = dict(zip(layout["names"], zip(layout["offsets"], layout["sizes"])))
    k = cfg["vision"]["appearance_dim"]
    app = rec["appearance"] if "appearance" in rec else np.zeros((len(rs["names"]), k))
    # what eating each object did in this recording: the stage's poison is set per world, not in ruleset.json
    effect = {}                                  # by the pain it gave, not by the stomach: a bite when full gains nothing
    for o in np.unique(ate[ate > 0]):
        effect[int(o)] = "poison" if np.mean(pain[ate == o] > 0.5) > 0.5 else "food"
    # a bush is judged by its berry (the next object id, as in scripts/probes/events.py)
    names = rs["names"]
    kind = lambda n: "empty bush" if "Empty" in n else "bush" if n.endswith("Bush") else "other"
    # drawn in the colour its name says (the animal itself sees the appearance vector, not this colour)
    def colour(i, n):
        if not i:
            return None
        named = NAMED_COLOURS.get(n.split()[0], NAMED_COLOURS["Onion" if "Onion" in n else "plain"] if "Bush" in n or "Onion" in n or "berry" in n else None)
        return named or [round(0.5 + 0.45 * float(c), 3) for c in app[i][:3]]
    objects = [{"name": n, "colour": colour(i, n),
                "kind": kind(n), "effect": effect.get(i + 1 if kind(n) == "bush" else i)} for i, n in enumerate(names)]
    size = 2 * radius + 1
    steps = []
    for t in range(t0, t1):
        py, px = (int(v) for v in pos[t - 1, agent])
        patch = np.full((size, size), -1, np.int16)                # -1 = beyond the edge of the world
        ys, xs = np.arange(py - radius, py + radius + 1), np.arange(px - radius, px + radius + 1)
        oky, okx = (ys >= 0) & (ys < H), (xs >= 0) & (xs < W)
        patch[np.ix_(oky, okx)] = grid[t - 1][np.ix_(ys[oky], xs[okx])]
        others = []
        for j in range(N):
            if j != agent and alive[t - 1, j]:
                dy, dx = int(pos[t - 1, j, 0]) - py, int(pos[t - 1, j, 1]) - px
                if abs(dy) <= radius and abs(dx) <= radius:
                    others.append([dy, dx, int(dr[t - 1, j])])
        steps.append({
            "tick": t, "patch": patch.ravel().tolist(), "dir": int(dr[t - 1, agent]), "others": others,
            "action": int(act[t, agent]), "ate": int(ate[t, agent]),
            "food": float(food[t, agent]), "pain": float(pain[t, agent]),
            "inputs": [float(x[t, agent, layout["in_names"].index(n)]) for n in inputs],
            "cells": [[float(v) for v in x[t, agent, off[r][0]:off[r][0] + off[r][1]]] for r in regions],
            "motor": [float(v) for v in x[t, agent, off["out"][0]:off["out"][0] + off["out"][1]]]})
    return _decimals({"run": _rel(run), "agent": agent, "radius": radius, "objects": objects, "actions": list(A.NAMES),
                      "regions": list(regions), "inputs": list(inputs), "max_food": cfg["world"]["max_food"],
                      "steps": steps})


@extra("1.1")
def bad_berry(s, main: list[Path], control: list[Path]):
    """The innate reaction to a bad berry: one animal of the lineage run that bites a poison berry and, within
    the next 20 ticks, eats a good one. Chosen automatically, so a rerun yields a new example of the same kind."""
    run = main[0]
    rec = np.load(run / "recording.npz")
    ate, food, alive, act, pain = rec["ate"], rec["food"], rec["alive"], rec["action"], rec["pain"]
    T, N = ate.shape
    hurts = np.zeros(len(json.loads((run / "ruleset.json").read_text())["names"]), bool)
    for o in np.unique(ate[ate > 0]):           # the poison is set per world, not in ruleset.json
        hurts[o] = np.mean(pain[ate == o] > 0.5) > 0.5
    bad, good = (ate > 0) & hurts[ate], (ate > 0) & ~hurts[ate]   # not by the stomach: a bite when full gains nothing
    before, after = 8, 22
    best = None
    for i in range(N):
        for t in np.nonzero(bad[:, i])[0]:
            if t - before < 1 or t + after > T or not alive[t + after - 1, i]:
                continue
            meals = np.nonzero(good[t + 1:t + after, i])[0]
            # prefer: the animal turns away at once, a good meal follows soon, no second bad bite in the window
            score = (2.0 * (act[t + 1, i] in (2, 3)) + (2.0 - 0.05 * meals[0] if len(meals) else 0.0)
                     - 2.0 * (bad[t - before:t + after, i].sum() - 1) + 0.5 * good[t - before:t, i].any())
            if best is None or score > best[0]:
                best = (score, i, int(t))
    if best is None:
        return "bad_berry", None
    _, agent, t = best
    strip = life_strip(run, agent, t - before, t + after, regions=("valence_app", "valence_av", "no_feed", "no_touch", "grasp"),
                       inputs=("taste", "pain"))
    strip["event_tick"] = t
    return "bad_berry", strip


def _front(rec) -> np.ndarray:
    """[T, N] the object directly in front of every recorded animal (-1 beyond the edge of the world)."""
    grid, pos, dr = rec["grid"], rec["pos"], rec["dir"]
    T = grid.shape[0]; H, W = grid.shape[1:]
    front = pos.astype(int) + DIRS[dr.astype(int) % 4]
    inb = (front >= 0).all(-1) & (front[..., 0] < H) & (front[..., 1] < W)
    fc = np.clip(front, 0, [H - 1, W - 1])
    return np.where(inb, grid[np.arange(T)[:, None], fc[..., 0], fc[..., 1]], -1)


@extra("1.0")
def to_a_bush(s, main: list[Path], control: list[Path]):
    """Steering by reflex: one animal of the lineage run that walks up to a bush and eats, after a stretch without
    food. Chosen automatically."""
    run = main[0]
    rec = np.load(run / "recording.npz")
    ate, alive, act, pos = rec["ate"], rec["alive"], rec["action"], rec["pos"].astype(int)
    T, N = ate.shape
    before, after = 16, 8
    best = None
    for i in range(N):
        meals = np.nonzero(ate[:, i] > 0)[0]
        for t in meals:
            if t - before < 1 or t + after > T or not alive[t + after - 1, i] or (ate[t - before:t, i] > 0).any():
                continue
            walked = int((np.abs(np.diff(pos[t - before:t, i], axis=0)).sum(-1) > 0).sum())
            turned = int(np.isin(act[t - before:t, i], (2, 3)).sum())
            score = walked + 2.0 * min(turned, 3) + 2.0 * (ate[t + 1:t + after, i] > 0).sum()
            if best is None or score > best[0]:
                best = (score, i, int(t))
    if best is None:
        return "to_a_bush", None
    _, agent, t = best
    strip = life_strip(run, agent, t - before, t + after, regions=("ganglion_e", "ganglion_i"))
    strip["event_tick"] = t
    return "to_a_bush", strip


@extra("1.2h")
def dud_visit(s, main: list[Path], control: list[Path]):
    """Habituation: one animal of the lineage run that walks up to a dud, bites at it while the look fades from
    its eyes, and leaves. Chosen automatically: a visit of 12 to 35 ticks with the largest fall in appetite."""
    run = main[0]
    rec = np.load(run / "recording.npz")
    names = json.loads((run / "ruleset.json").read_text())["names"]
    layout = json.loads((run / "layout.json").read_text())
    if "Black Wild Gooseberry Bush" not in names:
        return "dud_visit", None
    dud = names.index("Black Wild Gooseberry Bush")
    alive, x = rec["alive"].astype(bool), rec["x"]
    T, N = alive.shape
    off = dict(zip(layout["names"], zip(layout["offsets"], layout["sizes"])))
    app = x[..., off["valence_app"][0]:off["valence_app"][0] + off["valence_app"][1]].astype(np.float32).mean(-1)
    facing = (_front(rec) == dud) & alive
    before, after = 6, 6
    best = None
    for i in range(N):
        t = before + 1
        while t < T - after:
            if facing[t, i] and not facing[t - 1, i]:
                e = t
                while e < T - after and facing[e, i]:
                    e += 1
                if 12 <= e - t <= 35 and alive[e + after - 1, i]:
                    score = float(app[t + 1:t + 4, i].mean() - app[e - 3:e, i].mean())
                    if best is None or score > best[0]:
                        best = (score, i, t, e)
                t = e + 1
            else:
                t += 1
    if best is None:
        return "dud_visit", None
    _, agent, t, e = best
    look = [n for n in layout["in_names"] if n.startswith("vis+0.app")]
    idx = [layout["in_names"].index(n) for n in look]
    strongest = [look[j] for j in np.argsort(-np.abs(x[t + 1, agent, idx].astype(np.float32)))[:2]]
    strip = life_strip(run, agent, t - before, e + after, regions=("valence_app", "valence_av", "no_touch", "grasp"),
                       inputs=tuple(strongest))
    strip["event_tick"] = t
    return "dud_visit", strip


_LESSON: dict = {}


def _lesson(main: list[Path]):
    """Learning within a life, two moments of one animal of the lineage run: its first bite of a poison type that
    is new to it, and a later meeting with a bush of the same type that it leaves alone. Chosen automatically."""
    run = main[0]
    if run in _LESSON:
        return _LESSON[run]
    rec = np.load(run / "recording.npz")
    rs = json.loads((run / "ruleset.json").read_text()); layout = json.loads((run / "layout.json").read_text())
    names = rs["names"]
    grid, pos, dr, act, ate, pain, alive, x = (rec[k] for k in ("grid", "pos", "dir", "action", "ate", "pain", "alive", "x"))
    T, N = act.shape; H, W = grid.shape[1:]
    off = dict(zip(layout["names"], zip(layout["offsets"], layout["sizes"])))
    av = x[..., off["valence_av"][0]:off["valence_av"][0] + off["valence_av"][1]].astype(np.float32).mean(-1)
    dirs = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
    front = pos.astype(int) + dirs[dr.astype(int) % 4]
    inb = (front >= 0).all(-1) & (front[..., 0] < H) & (front[..., 1] < W)
    fc = np.clip(front, 0, [H - 1, W - 1])
    obj = np.where(inb, grid[np.arange(T)[:, None], fc[..., 0], fc[..., 1]], -1)
    bush = [i for i, n in enumerate(names) if n.endswith("Bush") and "Empty" not in n]
    poison = [b for b in bush if (ate == b + 1).any() and np.mean(pain[ate == b + 1] > 0.5) > 0.5]
    before, after = 8, 14
    best = None
    for i in range(N):
        for b in poison:
            bites = np.nonzero(ate[:, i] == b + 1)[0]
            if not len(bites) or bites[0] < before + 1:
                continue
            t1 = int(bites[0])
            # later: the same type right in front for two ticks, alive long after, and no bite of it then
            for t2 in np.nonzero((obj[:-1, i] == b) & (obj[1:, i] == b))[0]:
                t2 = int(t2)
                if t2 < t1 + 30 or t2 + after >= T or not alive[t2 + after, i] or t2 - before < 1:
                    continue
                if (ate[t2 - 2:t2 + after, i] == b + 1).any():
                    continue
                score = float(av[t2:t2 + 2, i].mean() - av[t1 - 1, i]) + 0.3 * bool((ate[t2:t2 + after, i] > 0).any())
                if best is None or score > best[0]:
                    best = (score, i, t1, t2)
                break
    if best is None:
        _LESSON[run] = (None, None)
        return _LESSON[run]
    _, agent, t1, t2 = best
    regions = ("valence_app", "valence_av", "no_feed", "no_touch", "grasp", "us_pain")
    bite = life_strip(run, agent, t1 - before, t1 + after, regions=regions, inputs=("taste", "pain"))
    again = life_strip(run, agent, t2 - before, t2 + after, regions=regions, inputs=("taste", "pain"))
    bite["event_tick"], again["event_tick"] = t1, t2
    _LESSON[run] = (bite, again)
    return _LESSON[run]


@extra("1.5")
def first_lesson(s, main: list[Path], control: list[Path]):
    return "first_lesson", _lesson(main)[0]


@extra("1.5")
def lesson_again(s, main: list[Path], control: list[Path]):
    return "lesson_again", _lesson(main)[1]


def _spent(run: Path) -> dict:
    """How the animals of a run's recording spend their life (as scripts/probes/forage.py): share of the ticks alive
    spent moving, and standing and biting with no full bush in front; berries eaten per life; mean lifetime."""
    rec = np.load(run / "recording.npz")
    names = json.loads((run / "ruleset.json").read_text())["names"]
    grid, pos, dr, act, alive, ate = (rec[k] for k in ("grid", "pos", "dir", "action", "alive", "ate"))
    alive = alive.astype(bool)
    T, N = act.shape
    H, W = grid.shape[1:]
    front = pos.astype(int) + np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])[dr.astype(int) % 4]
    inb = (front >= 0).all(-1) & (front[..., 0] < H) & (front[..., 1] < W)
    fc = np.clip(front, 0, [H - 1, W - 1])
    obj = np.where(inb, grid[np.arange(T)[:, None], fc[..., 0], fc[..., 1]], -1)
    # the seventh bush type (Black) exists only in worlds where it is the dud, which never carries a berry
    full = [i for i, n in enumerate(names) if n.endswith("Bush") and "Empty" not in n and n != "Black Wild Gooseberry Bush"]
    al = alive[:-1]
    idle = (act[1:] == 4) & al & ~np.isin(obj[:-1], full)
    moved = np.abs(np.diff(pos.astype(int), axis=0)).sum(-1) > 0
    return {"moving": 100.0 * float(moved[al].mean()), "idle": 100.0 * float(idle.sum() / max(1, al.sum())),
            "berries": float((ate > 0).sum() / N), "lifetime": float(alive.sum() / N)}


def life_spent(s, main: list[Path], control: list[Path]):
    """A small table for the page: how a life is spent in main and in control, mean over the seeds."""
    m, c = [_spent(r) for r in main], [_spent(r) for r in control]
    rows = [("Moving, % of the ticks alive", "moving"), ("Standing and biting at nothing, % of the ticks alive", "idle"),
            ("Berries eaten per life", "berries"), ("Lifetime, ticks", "lifetime")]
    return "life_spent", {"columns": ["In the recorded last generation", "main"] + (["control"] if c else []),
                          "rows": [[label, _stat([x[k] for x in m])] + ([_stat([x[k] for x in c])] if c else []) for label, k in rows],
                          "caption": "From the recording of the last generation of every seed (a sample of 64 animals each)."}


for _key in ("1.1", "1.2h", "1.5", "1.6h", "1.7", "1.8", "2.1"):
    extra(_key)(life_spent)


@extra("2.1")
def crowding(s, main: list[Path], control: list[Path]):
    """The stage's starting and evolved populations in worlds of several sizes, measured by
    scripts/probes/crowding.py (it simulates, so it is run separately and leaves its result next to the data files)."""
    f = BOOK_DIR / "data" / f"crowding_{s.key}.json"
    return "crowding", json.loads(f.read_text()) if f.exists() else None


@extra("1.2h")
@extra("1.6h")
def adaptation_off(s, main: list[Path], control: list[Path]):
    """The lesion of the adaptation, measured by scripts/probes/adapt_off.py (it simulates, so it is run separately
    and leaves its result next to the data files)."""
    f = BOOK_DIR / "data" / f"adapt_off_{s.key}.json"
    return "adaptation_off", json.loads(f.read_text()) if f.exists() else None


# ------------------------------------------------------------------ export

def export(key: str, do_evaluate: bool = False, worlds: int = 8, jobs: int = 5, lesions: str = "recent") -> Path:
    from .experiments import stages as S
    s = S.STAGES[key]
    main, control = S.seed_runs(s), S.seed_runs(s, control=True)
    if not main:
        raise FileNotFoundError(f"no finished run of stage {key} under {RUNS_DIR}")
    out_path = DATA_DIR / f"{key}.json"
    old = json.loads(out_path.read_text()) if out_path.exists() else {}
    data = {
        "stage": {"key": s.key, "name": s.name, "parent": s.parent, "notes": s.notes, "plastic": s.plastic},
        "exported": datetime.now().isoformat(timespec="minutes"),
        "runs": {"main": [run_info(r) for r in main], "control": [run_info(r) for r in control]},
        "provisional": provisional(s, main + control),
        "curves": {"main": curves(main), "control": curves(control)},
        "summary": summary(main, control),
        "brain": brain_view(s, main[0]),
        "settings": settings(s, main[0]),
    }
    runs_now = [_rel(r) for r in main + control]
    if do_evaluate:
        data["evaluation"] = evaluate(key, worlds, jobs, lesions)
    elif old.get("evaluation"):
        data["evaluation"] = old["evaluation"]
    if data.get("evaluation"):
        data["evaluation"]["stale"] = data["evaluation"]["runs"] != runs_now
        if key in NO_HEAD_TO_HEAD:
            data["evaluation"]["versus"] = None
    for label in ("main", "control"):                       # links to the light dashboards that exist
        for info in data["runs"][label]:
            name = info["dir"].split("/")[0] + ".html"
            info["dashboard"] = f"dashboards/{name}" if (DASH_DIR / name).exists() else None
    data["extras"] = {}
    for fn in EXTRAS.get(key, []):
        name, value = fn(s, main, control)
        data["extras"][name] = value
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_round(data), separators=(",", ":")), encoding="utf-8")
    return out_path


def dashboards(key: str) -> list[Path]:
    """Light dashboards (life.dashboard, light=True) of the stage's reference runs: seed 0 of main and control."""
    from .experiments import stages as S
    from .dashboard import export_dashboard
    s = S.STAGES[key]
    DASH_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for runs in (S.seed_runs(s), S.seed_runs(s, control=True)):
        for run in runs[:1]:
            out.append(export_dashboard(run, light=True, out=DASH_DIR / f"{run.parent.name}.html"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m life.book")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help="write book/data/<stage>.json")
    e.add_argument("keys", nargs="+")
    e.add_argument("--evaluate", action="store_true", help="also simulate: re-evaluation, head to head, lesions")
    e.add_argument("--worlds", type=int, default=8)
    e.add_argument("--jobs", type=int, default=5, help="parallel processes for --evaluate (about 1 GB each)")
    e.add_argument("--lesions", choices=("recent", "all"), default="recent",
                   help="regions to silence: those this stage or its parent added (default), or every region")
    d = sub.add_parser("dashboards", help="write light dashboards to book/dashboards/")
    d.add_argument("keys", nargs="+")
    a = ap.parse_args(argv)
    for key in a.keys:
        if a.cmd == "export":
            p = export(key, a.evaluate, a.worlds, a.jobs, a.lesions)
            print(f"wrote {p} ({p.stat().st_size / 1024:.0f} kB)")
        else:
            for p in dashboards(key):
                print(f"wrote {p} ({p.stat().st_size / 1e6:.1f} MB)")



def _after_bite(run: Path) -> dict:
    """What follows a bite, from a run's recording (as scripts/probes/dwell.py). A visit is a run of bites at one
    bush. Poison types are drawn per life: a type is poison when its bites hurt."""
    rec = np.load(run / "recording.npz")
    pos, dr, alive, ate, pain = (rec[k] for k in ("pos", "dir", "alive", "ate", "pain"))
    T, N = ate.shape
    bad = {int(o) for o in np.unique(ate[ate > 0]) if pain[ate == o].mean() > 0.5}
    front = pos.astype(int) + np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])[dr.astype(int) % 4]
    moved = np.abs(np.diff(pos.astype(int), axis=0)).sum(-1) > 0
    nxt = {True: [], False: []}; walk = {True: [], False: []}
    for i in range(N):
        ts = np.nonzero(ate[:, i] > 0)[0]
        visits = []                                   # [type, last tick, bush cell]
        for t in ts:
            if visits and visits[-1][0] == int(ate[t, i]) and (front[t, i] == visits[-1][2]).all():
                visits[-1][1] = t
            else:
                visits.append([int(ate[t, i]), t, front[t, i].copy()])
        for (o, t1, _), (o2, _, _) in zip(visits, visits[1:]):
            nxt[o in bad].append(o2 in bad)
        for o, t1, _ in visits:
            if t1 + 20 < T and alive[t1 + 20, i]:
                walk[o in bad].append(moved[t1:t1 + 20, i].sum())
    mean = lambda v: float(np.mean(v)) if len(v) else float("nan")
    return {"poison_after_poison": 100.0 * mean(nxt[True]), "poison_after_good": 100.0 * mean(nxt[False]),
            "steps_after_poison": mean(walk[True]), "steps_after_good": mean(walk[False])}


@extra("1.8")
def after_a_bite(s, main: list[Path], control: list[Path]):
    """A table for the page: what the animals do after a poison bite and after a good meal, main and control."""
    m, c = [_after_bite(r) for r in main], [_after_bite(r) for r in control]
    rows = [("Steps taken in the 20 ticks after a poison bite", "steps_after_poison"),
            ("Steps taken in the 20 ticks after the last bite of a good meal", "steps_after_good"),
            ("After a poison bite, the next bush bitten is poison again, %", "poison_after_poison"),
            ("After a good meal, the next bush bitten is poison, %", "poison_after_good")]
    return "after_a_bite", {"columns": ["In the recorded last generation", "main"] + (["control"] if c else []),
                            "rows": [[label, _stat([x[k] for x in m])] + ([_stat([x[k] for x in c])] if c else []) for label, k in rows],
                            "caption": "From the recording of the last generation of every seed (a sample of 64 animals each)."}

if __name__ == "__main__":
    main()
