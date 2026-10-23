"""The brain-evolution sequence of docs/BRAIN_EVOLUTION.md as runnable stages.

Every stage = a world (OHOL slice + world mechanics), a brain (regions/projections added on top of the previous
stage) and a fitness. Stages form one lineage: by default a stage warm-starts from the newest run of its parent
stage, remapped onto the new layout (new parts start silent, brain.remap_genomes). The optional control runs the
*parent's* brain in the *new* world from the same starting population, so the new module is the only difference.

    python -m life.experiments.stages list
    python -m life.experiments.stages 1.0 --generations 150
    python -m life.experiments.stages 1.1 --generations 100 [--control] [--init-from <run dir> | --init-from none]
    python -m life.experiments.stages replicate 1.1                     # two more seeds of main and control
    python -m life.experiments.stages summary 1.1                       # main vs control over the seeds
    python -m life.experiments.stages versus 1.1                        # main and control agents in the same worlds
    python -m life.experiments.stages lesions 1.1                       # is the circuit used? lesions over the seeds
    python -m life.experiments.stages lesion runs/s1_1_valence/<ts>     # silence each region, measure behaviour
    python -m life.experiments.stages respond runs/s1_1_valence/<ts>    # region activity/actions per object seen or held

Results and interpretation: docs/STAGE_LOG.md. Where the definitions are (life/experiments/README.md):
base.py (Stage, registry, helpers), worlds.py, chapter1.py, chapter2.py, side_paths.py, evaluate.py (running and
measuring). This file is the command line and re-exports all of them.
"""
from __future__ import annotations
import argparse

import numpy as np

# Every stage definition and tool, under its old name: `from life.experiments import stages as S` and `S.<name>` keep
# working. Importing the modules registers the stages: the chain first (chapter 1, chapter 2), then the side paths.
from life.experiments.base import *  # noqa: F401,F403
from life.experiments.worlds import *  # noqa: F401,F403
from life.experiments.side_paths import *  # noqa: F401,F403
from life.experiments.chapter1 import *  # noqa: F401,F403
from life.experiments.chapter2 import *  # noqa: F401,F403
from life.experiments.evaluate import *  # noqa: F401,F403
from life.experiments.evaluate import EVOLUTION_OVERRIDES
from life.experiments.chapter1 import _B15, _S15, _GENE, _split_cs  # noqa: F401
from life.experiments.side_paths import _S2  # noqa: F401
from life.run import latest_run, RUNS_DIR


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "chain":
        # chain <first> <last> [--generations N] [--no-control]: each stage (main and control in parallel,
        # both from the parent's newest main run), then the next. Logs in runs/logs/<name>.log.
        import subprocess
        from pathlib import Path
        q = argparse.ArgumentParser(prog="stages chain")
        q.add_argument("first")
        q.add_argument("last")
        q.add_argument("--generations", type=int, default=None)
        q.add_argument("--no-control", action="store_true")
        q.add_argument("--mutation-prob", default=None)
        b = q.parse_args(argv[1:])
        keys = list(STAGES)
        todo = keys[keys.index(b.first): keys.index(b.last) + 1]
        logs = RUNS_DIR / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        for k in todo:
            s = STAGES[k]
            parent = str(latest_run(STAGES[s.parent].name)) if s.parent else "none"
            base = [sys.executable, "-m", "life.experiments.stages", k, "--init-from", parent]
            if b.generations:
                base += ["--generations", str(b.generations)]
            if b.mutation_prob:
                base += ["--mutation-prob", str(b.mutation_prob)]
            jobs = [(s.name, base)] + ([] if b.no_control or not s.parent else [(s.name + "_control", base + ["--control"])])
            procs = [subprocess.Popen(cmd, stdout=open(logs / f"{n}.log", "w"), stderr=subprocess.STDOUT)
                     for n, cmd in jobs]
            codes = [p.wait() for p in procs]
            print(f"stage {k}: exit codes {codes}", flush=True)
            if any(codes):
                return
        return
    if argv and argv[0] == "replicate":
        # replicate <key> --seeds 1,2 [--generations N] [--mutation-prob p]: main + control per seed, in parallel,
        # from the parent's newest lineage run; experiment names get the suffix _seed<S>
        import subprocess
        q = argparse.ArgumentParser(prog="stages replicate")
        q.add_argument("key")
        q.add_argument("--seeds", default="1,2")
        q.add_argument("--generations", type=int, default=None)
        q.add_argument("--mutation-prob", default="0.1")
        b = q.parse_args(argv[1:])
        s = STAGES[b.key]
        parent = str(latest_run(STAGES[s.parent].name))
        logs = RUNS_DIR / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        for seed in b.seeds.split(","):
            base = [sys.executable, "-m", "life.experiments.stages", b.key, "--init-from", parent, "--seed", seed,
                    "--suffix", f"_seed{seed}", "--mutation-prob", b.mutation_prob]
            if b.generations:
                base += ["--generations", str(b.generations)]
            jobs = [(f"{s.name}_seed{seed}", base), (f"{s.name}_control_seed{seed}", base + ["--control"])]
            procs = [subprocess.Popen(cmd, stdout=open(logs / f"{n}.log", "w"), stderr=subprocess.STDOUT) for n, cmd in jobs]
            print(f"stage {b.key} seed {seed}: exit codes {[p.wait() for p in procs]}", flush=True)
        return
    if argv and argv[0] == "summary":
        # summary <key> [--window 50] [--worlds 8]: main and control over every seed (run name variants).
        # Two cheap ways to cut measurement noise: the mean over the last `window` generations, and the final
        # population re-evaluated in `worlds` fresh worlds that are the same for every run (0 = skip).
        # What remains, the spread between seeds, is how far lineages drift apart; only more seeds average that.
        from life.compare import load, window_means
        q = argparse.ArgumentParser(prog="stages summary")
        q.add_argument("key")
        q.add_argument("--window", type=int, default=50)
        q.add_argument("--worlds", type=int, default=8)
        b = q.parse_args(argv[1:])
        s = STAGES[b.key]
        se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        res = {}
        for label, base in (("main", s.name), ("control", s.name + "_control")):
            dirs = []
            for d in sorted(RUNS_DIR.iterdir()):
                rest = d.name[len(base):]
                if d.name.startswith(base) and (rest == "" or rest.startswith("_seed")):
                    runs = sorted(r for r in d.iterdir() if (r / "population.npz").exists())
                    if runs:
                        dirs.append(runs[-1])
            if not dirs:
                continue
            lasts = [window_means(load(r), b.window)[1] for r in dirs]
            keys = [k for k in ("fit_mean", "alive_ticks", "eaten", "pain", "temp_mean", "poison_frac") if k in lasts[0]]
            vals = {k: [x[k] for x in lasts] for k in keys}
            if b.worlds > 0:
                vals["re-evaluated"] = [lesion(str(r), worlds=b.worlds, only=("intact",))["intact"]["fitness"] for r in dirs]
            res[label] = vals
            print(f"{label:8s} n={len(dirs)}  " + "  ".join(
                f"{k} {np.mean(v):.3g} +-{se(v):.2g} [{' '.join(f'{x:.3g}' for x in v)}]" for k, v in vals.items()))
        if len(res) == 2:
            for k in ("fit_mean", "re-evaluated"):
                if k in res["main"] and len(res["main"][k]) == len(res["control"][k]):
                    diff = np.array(res["main"][k]) - np.array(res["control"][k])
                    print(f"main - control, {k}: {diff.mean():+.3g} +-{se(diff):.2g} (standard error over {len(diff)} seeds)")
        return res
    if argv and argv[0] == "respond":
        q = argparse.ArgumentParser(prog="stages respond")
        q.add_argument("run_dir")
        q.add_argument("--regions", default="")
        b = q.parse_args(argv[1:])
        rows = respond(b.run_dir, [r for r in b.regions.split(",") if r] or None)
        cols = [c for c in rows[0] if c not in ("object", "view")]
        print("object".ljust(34) + "view ".ljust(7) + "".join(c[:11].rjust(12) for c in cols))
        for r in rows:
            print(r["object"][:33].ljust(34) + r["view"].ljust(7) + "".join(f"{r[c]:12.3f}" for c in cols))
        return rows
    if argv and argv[0] == "versus":
        q = argparse.ArgumentParser(prog="stages versus")
        q.add_argument("key")
        q.add_argument("--worlds", type=int, default=8)
        b = q.parse_args(argv[1:])
        return versus(b.key, b.worlds)
    if argv and argv[0] == "lesions":
        q = argparse.ArgumentParser(prog="stages lesions")
        q.add_argument("key")
        q.add_argument("--worlds", type=int, default=8)
        b = q.parse_args(argv[1:])
        return lesions(b.key, b.worlds)
    if argv and argv[0] == "lesion":
        q = argparse.ArgumentParser(prog="stages lesion")
        q.add_argument("run_dir")
        q.add_argument("--regions", default="")
        q.add_argument("--worlds", type=int, default=8)
        b = q.parse_args(argv[1:])
        res = lesion(b.run_dir, [r for r in b.regions.split(",") if r] or None, b.worlds)
        cols = list(next(iter(res.values())))
        print("lesioned region".ljust(18) + "".join(c[:14].rjust(15) for c in cols))
        for k, row in res.items():
            print(k.ljust(18) + "".join(f"{row[c]:15.3f}" for c in cols))
        return res
    p = argparse.ArgumentParser()
    p.add_argument("stage", help="stage key, e.g. 1.0, 'list', or 'lesion <run dir>'")
    p.add_argument("--generations", type=int, default=None)
    p.add_argument("--control", action="store_true", help="parent's brain in this stage's world, same start")
    p.add_argument("--init-from", default="auto", help="'auto' (newest parent run), 'none', or a run dir")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mutation-prob", type=float, default=None, help="per-synapse chance that a weight mutates")
    p.add_argument("--suffix", default="", help="appended to the run's experiment name (variants)")
    p.add_argument("--ticks", type=int, default=None, help="ticks per life, instead of the stage's own (trials)")
    p.add_argument("--continue", dest="cont", action="store_true",
                   help="--init-from is a run of this stage: do not apply the stage's change to the start population again")
    p.add_argument("--dashboard", action="store_true",
                   help="also write the run's full dashboard.html (about 150 MB and a peak in memory at the end of the "
                        "run; the book's light dashboards are made from the recording: `python -m life.book dashboards`)")
    p.add_argument("--resume", default=None, help="a run of this stage stopped early: go on from its checkpoint.npz")
    a = p.parse_args(argv)
    if a.stage == "list":
        for s in STAGES.values():
            print(f"{s.key:5s} {s.name:32s} parent={s.parent}  {s.notes}")
        return
    if a.mutation_prob is None and a.init_from != "none" and STAGES[a.stage].parent:
        # a warm-started population is near its optimum: with every weight mutating in every child it loses ground
        # (mutational load; the trial x.c256 of 2026-10-13 lost a quarter in 40 generations at 1.0, STAGE_LOG v40)
        a.mutation_prob = 0.1
    if a.mutation_prob is not None:
        EVOLUTION_OVERRIDES["weight_mutation_prob"] = a.mutation_prob
    return run_stage(a.stage, a.generations, a.control, a.init_from, a.seed, suffix=a.suffix, ticks=a.ticks,
                     dashboard=a.dashboard, adapt=not a.cont, resume=a.resume)


if __name__ == "__main__":
    main()
