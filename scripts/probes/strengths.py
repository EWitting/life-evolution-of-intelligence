"""Evolved strength of the tunable hard-wired projections of a run, as a multiple of the designed value."""
import sys, numpy as np
from life.config import ExperimentConfig
from life.run import load_population, latest_run, make_layout
for name in sys.argv[1:]:
    run = latest_run(name); exp = ExperimentConfig.from_json((run / "config.json").read_text()); L = make_layout(exp)
    pop = load_population(run); w0 = np.asarray(pop.w0); design = np.nan_to_num(np.asarray(L.w_init)); pid = np.asarray(L.proj_id)
    print(f"== {name} ({run.name}); tuning range x{1 / exp.evolution.tune_range:.2f} .. x{exp.evolution.tune_range:.0f}")
    for i, n in enumerate(L.proj_names):
        m = (pid == i) & (np.asarray(L.tune) > 0)
        if m.any():
            r = np.abs(w0[:, m]).mean(1) / abs(design[m].mean())
            print(f"  {n:40s} designed {design[m].mean():+5.1f}   x{r.mean():.2f}  (genomes: min x{r.min():.2f}, max x{r.max():.2f})")
