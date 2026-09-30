"""Compare runs from their fitness.csv: averages of every column over the first and last windows of generations.

    python -m life.compare runs/s1_1_valence/<ts> runs/s1_1_valence_control/<ts> [--window 25]
    python -m life.compare s1_1_valence s1_1_valence_control          # experiment names: newest run of each
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path

from .run import RUNS_DIR

SKIP = {"gen", "seconds"}


def resolve(arg: str) -> Path:
    p = Path(arg)
    if (p / "fitness.csv").exists():
        return p
    runs = sorted(d for d in (RUNS_DIR / arg).iterdir() if (d / "fitness.csv").exists())
    return runs[-1]


def load(p: Path) -> list[dict]:
    with open(p / "fitness.csv") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def window_means(rows: list[dict], w: int) -> tuple[dict, dict]:
    cols = [k for k in rows[0] if k not in SKIP]
    avg = lambda rs: {k: sum(r[k] for r in rs) / len(rs) for k in cols}
    return avg(rows[:w]), avg(rows[-w:])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--window", type=int, default=25)
    a = ap.parse_args(argv)
    for arg in a.runs:
        p = resolve(arg)
        rows = load(p)
        first, last = window_means(rows, a.window)
        print(f"{p.parent.name}/{p.name}  ({len(rows)} generations, windows of {a.window})")
        for k in first:
            print(f"    {k:18s} first {first[k]:9.3f}   last {last[k]:9.3f}")


if __name__ == "__main__":
    main()
