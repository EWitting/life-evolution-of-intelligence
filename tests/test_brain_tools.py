"""Brain architecture, evolution, lab and dashboard tests. Run: .venv\\Scripts\\python.exe -m pytest -q"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from life import actions as A
from life import brain
from life.config import (WorldConfig, VisionConfig, BrainConfig, EvolutionConfig, ExperimentConfig,
                         RegionSpec, ProjectionSpec)
from life.evolution import next_generation
from life.lab import Lab
from test_core import toy_ruleset


def two_region_cfg():
    return BrainConfig(
        regions=(RegionSpec("cortex", 6, alpha=0.5), RegionSpec("striatum", 4, alpha=0.8, trace_tau=0.5)),
        projections=(ProjectionSpec("in", "cortex", density=1.0, rule="fixed"),
                     ProjectionSpec("cortex", "cortex", density=0.5, rule="oja"),
                     ProjectionSpec("cortex", "striatum", density=1.0, rule="trace", modulated=True, eta_init=0.01),
                     ProjectionSpec("striatum", "out", density=1.0, rule="hebb", eta_init=0.01)),
        modulator="reward")


def test_layout_regions_and_rules():
    L = brain.build_layout(two_region_cfg(), n_in=5, n_out=A.NUM_ACTIONS)
    assert L.names == ("in", "cortex", "striatum", "out") and L.n == 5 + 6 + 4 + A.NUM_ACTIONS
    assert L.region("striatum") == slice(11, 15)
    assert float(L.allowed[:, :5].sum()) == 0                                 # nothing targets inputs
    assert int(L.rule[L.region("cortex"), L.region("striatum")][0, 0]) == brain.RULES["trace"]
    assert float(L.modulated[L.region("cortex"), L.region("striatum")].mean()) == 1.0
    assert float(L.allowed[L.region("in"), L.region("out")].sum()) == 0     # no in->out projection here
    assert float(L.alpha[L.region("striatum")][0]) == pytest.approx(0.8)


def test_rules_change_weights_only_where_plastic():
    cfg = two_region_cfg()
    L = brain.build_layout(cfg, n_in=5, n_out=A.NUM_ACTIONS)
    g = brain.init_genome(jax.random.PRNGKey(0), L)
    st = brain.init_state(g)
    obs = jnp.ones(5)
    for _ in range(3):
        st, act = brain.step(cfg, L, g, st, obs, jnp.float32(1.0), jax.random.PRNGKey(1))
    dw = np.abs(np.asarray(st.w - g.w0 * g.mask))
    assert dw[L.region("in"), :].sum() == 0                                   # fixed projection unchanged
    assert dw[L.region("striatum"), L.region("out")].sum() > 0               # hebb with eta_init > 0
    assert dw[L.region("cortex"), L.region("striatum")].sum() > 0            # trace rule, modulated by 1.0
    st0 = brain.init_state(g)
    st0, _ = brain.step(cfg, L, g, st0, obs, jnp.float32(0.0), jax.random.PRNGKey(1))
    assert np.abs(np.asarray(st0.w - g.w0 * g.mask))[L.region("cortex"), L.region("striatum")].sum() == 0  # mod 0


def test_evolution_respects_layout():
    L = brain.build_layout(two_region_cfg(), n_in=5, n_out=A.NUM_ACTIONS)
    pop = jax.vmap(lambda k: brain.init_genome(k, L))(jax.random.split(jax.random.PRNGKey(0), 8))
    pop2 = next_generation(jax.random.PRNGKey(2), pop, jnp.arange(8.0), EvolutionConfig(plastic=True), L)
    assert float((pop2.mask * (1 - L.allowed)).sum()) == 0                    # never outside projections
    assert float((pop2.eta * (L.rule == 0)).sum()) == 0                       # never plastic on fixed rule


def test_lab_step_inspect_export(tmp_path):
    rs = toy_ruleset()
    exp = ExperimentConfig(world=WorldConfig(height=10, width=10, num_agents=4), vision=VisionConfig(columns=3, range=3),
                           brain=BrainConfig(regions=(RegionSpec("hidden", 4),)),
                           evolution=EvolutionConfig(record_weights_every=5))
    lab = Lab(exp, rs)
    lab.step(7, override={0: A.TURN_LEFT})
    assert lab.tick == 7 and len(lab.history) == 7 and len(lab.w_snaps) == 1
    o = lab.obs(0)
    assert o["vision"].shape == (3, 4 + exp.vision.appearance_dim) and o["body"].shape[0] == 4 + exp.vision.appearance_dim
    b = lab.brain(0)
    assert set(b["activations"]) == {"in", "hidden", "out"} and b["w"].shape == (lab.layout.n, lab.layout.n)
    assert "objects" in lab.world_summary() and lab.agent(0)["alive"]
    out = lab.export(tmp_path / "lab")
    assert (out / "dashboard.html").exists() and (out / "recording.npz").exists()
    html = (out / "dashboard.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/null" not in html and '"names"' in html


def test_quick_run_with_dashboard(tmp_path):
    from life.run import run_evolution
    rs = toy_ruleset()
    exp = ExperimentConfig(name="t", world=WorldConfig(height=10, width=10, num_agents=4), vision=VisionConfig(columns=3, range=3),
                           brain=BrainConfig(regions=(RegionSpec("hidden", 4),)),
                           evolution=EvolutionConfig(generations=2, ticks_per_generation=10, record_weights_every=5))
    res = run_evolution(exp, rs, lambda s: s["eaten"], out_dir=tmp_path / "run", verbose=False)
    for f in ("config.json", "ruleset.json", "layout.json", "fitness.csv", "population.npz", "recording.npz", "dashboard.html"):
        assert (tmp_path / "run" / f).exists(), f
    rec = np.load(tmp_path / "run" / "recording.npz")
    assert rec["w_snap"].shape[0] == 2 and rec["x"].shape == (10, 4, res["pop"].b.shape[1])
