"""Is spreading to look-alikes what limits learning? A run's population, intact and with plasticity off, in its own
world with the novel types drawn as alike as designed (NOVEL_SIM, 0.8) and less alike.

    uv run python scripts/probes/novel_sim.py <run directory> [sim ...]     WORLDS=<n> (default 16)

Two novel types share about NOVEL_SIM^2 of their look (0.64 at 0.8, 0.25 at 0.5). STAGE_LOG v44."""
import os, sys
import numpy as np
from life.experiments import stages as S

run = sys.argv[1]
for sim in [S.NOVEL_SIM] + [float(x) for x in sys.argv[2:] or ["0.5"]]:
    S.NOVEL_SIM = sim   # read by learning_world when the worlds are built
    res = S.lesion(run, worlds=int(os.environ.get("WORLDS", 16)), only=("intact", "no_plasticity"), per_world=True)
    a, b = np.array(res["intact"]["fitness_worlds"]), np.array(res["no_plasticity"]["fitness_worlds"])
    r = 100 * b / a
    print(f"[{run}] novel similarity {sim}: intact {a.mean():5.0f}  plasticity off {100 * b.mean() / a.mean():3.0f}% "
          f"+-{r.std(ddof=1) / np.sqrt(len(r)):.0f}  poison share {res['intact']['poison_frac']:.3f} / "
          f"{res['no_plasticity']['poison_frac']:.3f}  berries {res['intact']['berries']:.0f} / "
          f"{res['no_plasticity']['berries']:.0f}", flush=True)
