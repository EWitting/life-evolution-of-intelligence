"""Lesions of a trial stage: `<key>` the newest run, `<key> before` the frozen parent population (seed 0), `<key> <run
directory name>` another seed's run (e.g. x_ch2_256_p70_seed1).

    uv run python scripts/probes/trial_lesion.py <key> [before | <name>]     WORLDS=<n> (default 4), ONLY=a,b,..."""
import os, sys, shutil, tempfile
from pathlib import Path
from life.experiments import stages as S
from life.run import latest_run
s = S.STAGES[sys.argv[1]]
before = sys.argv[2:] == ["before"]
if before:   # the frozen parent population as a run directory of this stage
    run = Path(tempfile.mkdtemp()) / s.name / "gen0"
    run.mkdir(parents=True)
    (run / "config.json").write_text(S.make_exp(s, s.brain, s.name, 1, 0).to_json())
    shutil.copy(Path("frozen") / s.parent / "seed0.npz", run / "population.npz")
elif sys.argv[2:]:   # a seed's run directory name, e.g. s2_1_eyes_seed1
    run = latest_run(sys.argv[2])
else:
    run = latest_run(s.name)
res = S.lesion(str(run), worlds=int(os.environ.get("WORLDS", 4)),
               only=["intact"] + os.environ.get("ONLY", "no_plasticity,fed,flee").split(","))
for k, r in res.items():
    print(f"[{s.key} {run.parent.name} {'before' if before else 'after '}] {k:14s} fitness {r['fitness']:5.0f} ({100 * r['fitness'] / res['intact']['fitness']:3.0f}% of intact)  lifetime {r['alive']:5.0f}  berries {r['berries']:5.1f}  poison share {r['poison_frac']:.3f}", flush=True)
