"""Freezing a stage: its definition and its final populations are stored under `frozen/<key>/`, in the repository,
and `tests/test_frozen.py` fails when the stage's definition in `life/experiments/stages.py` no longer matches.

    python -m life.freeze <key> [...]      store (or overwrite) the snapshot of each stage
    python -m life.freeze --table          print the reference table of everything frozen

`frozen/<key>/definition.json`   what the stage is: brain, world, senses, body, evolution settings, a digest of the
                                 rules its world is built from, its parent and the run directories it was frozen from
`frozen/<key>/seed<N>.npz`       the final population of every seed (main), loadable with `load_frozen`
`frozen/<key>/seed<N>.json`      the configuration that population ran under (its layout is rebuilt from it)

A frozen stage is not rerun. A later change that has to touch it (a trim, a renamed region) is made by remapping
the stored populations onto the new layout and freezing again, with the reason written in docs/STAGE_LOG.md."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

FROZEN = Path(__file__).resolve().parents[1] / "frozen"
# settings of a run that are not part of what a stage is
RUN_ONLY = ("name", "seed", "generations")


def rules_digest(s, exp) -> str:
    """A short digest of the world's rules (what yields what, food and pain values, spawn weights, appearances) for
    generation 0 under a fixed key: `Stage.build` is a function, so this is how a change to it is noticed."""
    import jax
    rs, rules_fn = s.build(exp)
    rules = rules_fn(0, jax.random.PRNGKey(0))
    h = hashlib.sha256()
    for name, a in sorted(rules._asdict().items()):
        h.update(name.encode())
        h.update(np.round(np.asarray(a, np.float64), 4).tobytes())
    return h.hexdigest()[:16]


def definition(key: str) -> dict:
    """The stage as defined now, as plain data."""
    from .experiments import stages as S
    s = S.STAGES[key]
    exp = S.make_exp(s, s.brain, s.name, s.generations, 0)
    d = json.loads(exp.to_json())
    for k in RUN_ONLY:
        d.pop(k, None)
        d["evolution"].pop(k, None)
    return {"key": key, "parent": s.parent, "plastic": s.plastic, "dense_plastic": s.dense_plastic,
            "rules": rules_digest(s, exp), "config": d}


def freeze(key: str) -> Path:
    from .experiments import stages as S
    s = S.STAGES[key]
    out = FROZEN / key
    out.mkdir(parents=True, exist_ok=True)
    d = definition(key)
    runs = S.seed_runs(s)
    d["runs"] = []
    for i, run in enumerate(runs):
        cfg = json.loads((run / "config.json").read_text())
        seed = cfg["evolution"]["seed"]
        # the population must have lived under the stage as defined (a run may differ in life length: 1.0 settles)
        for part in ("brain", "world", "vision", "body"):
            assert cfg[part] == d["config"][part], f"{run}: {part} differs from the definition of {key}"
        shutil.copyfile(run / "population.npz", out / f"seed{seed}.npz")
        shutil.copyfile(run / "config.json", out / f"seed{seed}.json")
        d["runs"].append({"seed": seed, "dir": str(run.relative_to(S.RUNS_DIR.parent)).replace("\\", "/"),
                          "ticks": cfg["evolution"]["ticks_per_generation"],
                          "generations": cfg["evolution"]["generations"]})
    (out / "definition.json").write_text(json.dumps(d, indent=1))
    return out


def load_frozen(key: str, seed: int = 0, exp=None):
    """The frozen population of a stage (remapped onto `exp`'s layout when given, as load_population does)."""
    import jax
    import jax.numpy as jnp
    from . import brain
    from .config import ExperimentConfig
    from .run import make_layout
    with np.load(FROZEN / key / f"seed{seed}.npz") as f:
        pop = brain.Genome(**{k: jnp.asarray(f[k]) for k in brain.Genome._fields})
    if exp is None:
        return pop
    old = make_layout(ExperimentConfig.from_json((FROZEN / key / f"seed{seed}.json").read_text()))
    return brain.remap_genomes(pop, old, make_layout(exp), jax.random.PRNGKey(seed))


def frozen_keys() -> list[str]:
    return sorted(p.parent.name for p in FROZEN.glob("*/definition.json")) if FROZEN.exists() else []


def table() -> str:
    """The reference table: one row per frozen stage."""
    rows = ["| stage | parent | regions | world | life | seeds | frozen from |", "|---|---|---|---|---|---|---|"]
    for key in frozen_keys():
        d = json.loads((FROZEN / key / "definition.json").read_text())
        c = d["config"]
        regions = ", ".join(f"{r['name']} {r['size']}" for r in c["brain"]["regions"])
        w = c["world"]
        world = f"{w['height']}x{w['width']}, density {w['spawn_density']}, born {w['start_food']:.2g} full" \
                + (", cold" if w.get("temperature") else "")
        runs = "<br>".join(f"`{r['dir']}`" for r in d["runs"])
        rows.append(f"| {key} | {d['parent'] or '-'} | {regions} | {world} | {d['runs'][0]['ticks']} | {len(d['runs'])} | {runs} |")
    return "\n".join(rows)


if __name__ == "__main__":
    if sys.argv[1:] == ["--table"]:
        print(table())
    else:
        for k in sys.argv[1:]:
            print("frozen", freeze(k))
