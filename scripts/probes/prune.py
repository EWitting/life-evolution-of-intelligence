"""Keep only the newest finished run (and anything newer, in progress) in each stage directory under runs/."""
import shutil
from pathlib import Path
for d in sorted(Path("runs").iterdir()):
    if not d.is_dir() or d.name == "logs" or d.name.startswith(("_archive", "exp01")):
        continue
    runs = sorted(r for r in d.iterdir() if r.is_dir())
    done = [r for r in runs if (r / "population.npz").exists()]
    for r in runs:
        if done and r.name < done[-1].name:
            shutil.rmtree(r)
