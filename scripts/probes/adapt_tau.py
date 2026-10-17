"""Does a slower adaptation of the look inputs make learning worth more? A run's population, intact and with
plasticity off, with the adaptation time constant as evolved (BrainConfig.in_trace_tau) and with others.

    uv run python scripts/probes/adapt_tau.py <run directory> [tau ...]     WORLDS=<n> (default 16)

In patches the slow average becomes the local patch type's look, so the adapted look of that type is about 0
inside its own patch and a lesson is learned relative to the patch (STAGE_LOG v44)."""
import json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np
from life.experiments import stages as S

run = Path(sys.argv[1])
taus = [float(t) for t in sys.argv[2:]] or [0.998]
cfg = json.loads((run / "config.json").read_text())
for tau in [cfg["brain"]["in_trace_tau"]] + taus:
    tmp = Path(tempfile.mkdtemp()) / run.parent.name / "x"
    tmp.mkdir(parents=True)
    c = json.loads(json.dumps(cfg)); c["brain"]["in_trace_tau"] = tau
    (tmp / "config.json").write_text(json.dumps(c))
    shutil.copy(run / "population.npz", tmp / "population.npz")
    res = S.lesion(str(tmp), worlds=int(os.environ.get("WORLDS", 16)), only=("intact", "no_plasticity"), per_world=True)
    a, b = np.array(res["intact"]["fitness_worlds"]), np.array(res["no_plasticity"]["fitness_worlds"])
    r = 100 * b / a
    print(f"[{run.parent.name}] tau {tau:.4f} (about {1 / (1 - tau):.0f} ticks): intact {a.mean():5.0f}  plasticity off "
          f"{100 * b.mean() / a.mean():3.0f}% +-{r.std(ddof=1) / np.sqrt(len(r)):.0f}  poison share "
          f"{res['intact']['poison_frac']:.3f} / {res['no_plasticity']['poison_frac']:.3f}  berries "
          f"{res['intact']['berries']:.0f} / {res['no_plasticity']['berries']:.0f}", flush=True)
