"""The critic's cells tick by tick in the lab (labtests.conditioning, extinction protocol): mean rate of each cell on
the first and sixth painful meal and the first meal without pain, and the error (da), for a run's population.
    python scripts/probes/lab_cells.py <run dir> [<run dir> ...] [world]
world: world-like trials (one tick of pain, nothing after) instead of the lab's (three ticks, 0.3 after)."""
import sys
from pathlib import Path
import numpy as np
from life.config import ExperimentConfig
from life.experiments import stages as S
from life.labtests import PROTOCOLS, conditioning
from life.run import load_population

CELLS = ("value_av", "gp_habenular", "habenula", "value_app", "ventral_pallidum", "dopamine")
TICKS = ("bush", "bite 1", "bite 2", "bite 3", "after", "gone", "", "", "")
extra = dict(outcome_ticks=1, linger=0.0) if "world" in sys.argv[1:] else {}
for run in [Path(a) for a in sys.argv[1:] if a != "world"]:
    exp = ExperimentConfig.from_json((run / "config.json").read_text())
    pop = load_population(run, exp)
    r = conditioning(exp, pop, {**PROTOCOLS["extinction"], **extra}, S.PALLIUM_SIM, cells=CELLS)
    print(f"\n{run}")
    for trial, label in ((0, "first painful meal"), (5, "sixth painful meal"), (6, "first meal without pain")):
        print(f"  {label}")
        n = r["signal"].shape[-1]
        print("    " + f"{'tick':18s}" + "".join(f"{t:>8s}" for t in TICKS[:n]))
        for c in CELLS:
            print("    " + f"{c:18s}" + "".join(f"{v:8.2f}" for v in r["cells"][c][trial].mean(0)))
        print("    " + f"{'error (da)':18s}" + "".join(f"{v:+8.2f}" for v in r["signal"][trial].mean(0)))
