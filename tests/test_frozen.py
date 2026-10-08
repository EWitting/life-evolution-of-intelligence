"""Frozen stages (life/freeze.py): the definition of a frozen stage must not change, and its stored populations must
still load. Run: .venv\\Scripts\\python.exe -m pytest -q tests/test_frozen.py

If this fails after an intended change to a frozen stage: remap the stored populations, freeze the stage again
(`python -m life.freeze <key>`) and write the reason in docs/STAGE_LOG.md. Do not rerun the lineage."""
import json

import pytest

from life import freeze

KEYS = freeze.frozen_keys()


@pytest.mark.parametrize("key", KEYS)
def test_definition_unchanged(key):
    stored = json.loads((freeze.FROZEN / key / "definition.json").read_text())
    now = freeze.definition(key)
    for part in ("parent", "plastic", "dense_plastic", "rules"):
        assert now[part] == stored[part], f"stage {key}: {part} changed since it was frozen"
    for part, value in stored["config"].items():
        assert now["config"][part] == value, f"stage {key}: {part} changed since it was frozen"


@pytest.mark.parametrize("key", KEYS)
def test_populations_load(key):
    from life.experiments import stages as S
    from life.run import make_layout
    s = S.STAGES[key]
    exp = S.make_exp(s, s.brain, s.name, 1, 0)
    n = len(make_layout(exp).names)
    stored = json.loads((freeze.FROZEN / key / "definition.json").read_text())
    assert stored["runs"], f"stage {key}: no population stored"
    for r in stored["runs"]:
        pop = freeze.load_frozen(key, r["seed"])
        assert pop.w0.shape[0] == exp.world.num_agents // exp.evolution.siblings
        assert pop.b.shape[1] == pop.w0.shape[1] == pop.w0.shape[2]
        assert freeze.load_frozen(key, r["seed"], exp).w0.shape == pop.w0.shape, f"stage {key}: layout changed"
    assert n > 0
