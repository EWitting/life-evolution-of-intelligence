"""Brain v2 features: Dale's law, hard-wiring, gain, eligibility, depression, delta, k-WTA, region modulators,
remapping genomes across layouts, delayed sickness, named inputs. Run: .venv\\Scripts\\python.exe -m pytest -q"""
import jax
import jax.numpy as jnp
import numpy as np

from life import actions as A
from life import brain
from life.config import (BrainConfig, RegionSpec as R, ProjectionSpec as P, ModulatorSpec, EvolutionConfig,
                         VisionConfig, BodyConfig, ExperimentConfig, WorldConfig)
from life.evolution import next_generation
from life.sensors import input_names

NO = jnp.zeros(3)


def run(cfg, obs, n=3, sig=NO, n_in=3, g=None):
    L = brain.build_layout(cfg, n_in=n_in, n_out=A.NUM_ACTIONS)
    g = g if g is not None else brain.init_genome(jax.random.PRNGKey(0), L)
    st = brain.init_state(g, L)
    for _ in range(n):
        st, _ = brain.step(cfg, L, g, st, jnp.asarray(obs, jnp.float32), sig, jax.random.PRNGKey(1))
    return L, g, st


def test_dale_sign_and_hardwired_one_to_one():
    cfg = BrainConfig(regions=(R("a", 3, sign="exc", alpha=1.0, bias=0.0), R("i", 3, sign="inh", alpha=1.0)),
                      projections=(P("in", "a", topology="one_to_one", w_init=2.0, evolve=False),
                                   P("a", "i", density=1.0), P("i", "out", density=1.0)))
    L, g, st = run(cfg, [1, 0, 1])
    blk = np.asarray(g.w0[L.region("in"), L.region("a")])
    assert np.allclose(blk, 2.0 * np.eye(3))
    assert (np.asarray(g.w0[L.region("a")]) >= 0).all() and (np.asarray(g.w0[L.region("i")]) >= 0).all()
    x = np.asarray(st.x)
    assert x[L.region("a")][1] == 0 and x[L.region("a")][0] > 0.9          # one-to-one copy, rectified
    eff = np.asarray(brain.effective(L, g.w0))
    assert (eff[L.region("i")] <= 0).all()                                  # inhibitory neurons only inhibit
    pop = jax.vmap(lambda k: brain.init_genome(k, L))(jax.random.split(jax.random.PRNGKey(0), 6))
    pop2 = next_generation(jax.random.PRNGKey(3), pop, jnp.arange(6.0), EvolutionConfig(mutation_std=0.5), L)
    assert np.allclose(np.asarray(pop2.w0[:, :3, 3:6]), 2.0 * np.eye(3))  # hard-wired survives mutation
    assert (np.asarray(pop2.w0[:, 3:9]) >= 0).all()                         # Dale magnitudes stay >= 0


def test_gain_projection_multiplies():
    # gain comes from a separate (modulatory) region: one matrix holds one synapse per neuron pair
    base = (P("in", "a", topology="one_to_one", w_init=0.5, evolve=False),)
    gain = base + (P("g", "a", kind="gain", topology="one_to_one", w_init=1.0, evolve=False),)
    rs = (R("g", 3, alpha=1.0, bias=5.0), R("a", 3, alpha=1.0, bias=0.0))
    L, _, s1 = run(BrainConfig(regions=rs, projections=base), [1, 1, 1], n=3)
    _, _, s2 = run(BrainConfig(regions=rs, projections=gain), [1, 1, 1], n=3)
    a = L.region("a")
    assert np.allclose(np.asarray(s1.x[a]), np.tanh(0.5), atol=1e-5)
    assert np.allclose(np.asarray(s2.x[a]), np.tanh(0.5 * np.exp(np.tanh(5.0))), atol=1e-4)


def test_region_modulator_and_eligibility_bridge_a_delay():
    # plastic in->a is gated by modulator 'us' read from region 'u'; the eligibility trace remembers
    # the pre activity so a modulator arriving later still changes the weight
    cfg = BrainConfig(regions=(R("a", 3, alpha=1.0, bias=0.5), R("u", 1, alpha=1.0, bias=-5.0)),
                      projections=(P("in", "a", density=1.0, rule="hebb", modulator="us", eta_init=0.1, elig_tau=0.9, w_init=0.0),
                                   P("in", "u", density=1.0, w_init=0.0)),
                      modulators=(ModulatorSpec("us", pos="u", scale=1.0),))
    L = brain.build_layout(cfg, n_in=3, n_out=A.NUM_ACTIONS)
    g = brain.init_genome(jax.random.PRNGKey(0), L)
    st = brain.init_state(g, L)

    def step(st, obs, gg=g):
        return brain.step(cfg, L, gg, st, jnp.asarray(obs, jnp.float32), NO, jax.random.PRNGKey(1))[0]

    st = step(st, [1, 0, 0])                         # CS on input 0, no US: builds eligibility only
    st = step(st, [0, 0, 0])
    assert np.allclose(np.asarray(st.w), np.asarray(g.w0 * g.mask))
    g2 = g._replace(b=g.b.at[L.region("u")].set(5.0))  # now the US neuron fires -> modulator > 0
    st = step(st, [0, 0, 0], g2)
    st = step(st, [0, 0, 0], g2)
    dw = np.asarray(st.w - g.w0 * g.mask)[L.region("in"), L.region("a")]
    assert dw[0].min() > 0 and np.abs(dw[1:]).max() == 0   # only the synapses of the earlier CS changed


def test_depression_habituates():
    cfg = BrainConfig(regions=(R("a", 1, alpha=1.0, bias=0.0),),
                      projections=(P("in", "a", density=1.0, w_init=1.0, evolve=False, depression=(0.3, 50.0)),))
    L, g, st = run(cfg, [1, 1, 1], n=1)
    first = float(st.x[3])
    _, _, st = run(cfg, [1, 1, 1], n=20)
    assert float(st.x[3]) < 0.6 * first


def test_delta_rule_learns_teacher():
    cfg = BrainConfig(regions=(R("t", 2, alpha=1.0, bias=0.8), R("p", 2, alpha=1.0, bias=0.0)),
                      projections=(P("in", "p", density=1.0, rule="delta", teacher="t", eta_init=0.2, w_init=0.0),))
    L, g, st = run(cfg, [1, 1, 1], n=60)
    assert np.allclose(np.asarray(st.x[L.region("p")]), np.tanh(0.8), atol=0.05)


def test_kwta_keeps_k():
    cfg = BrainConfig(regions=(R("a", 6, alpha=1.0, kwta=2),), projections=(P("in", "a", density=1.0),))
    L, g, st = run(cfg, [1, 0.5, 1], n=2)
    assert int((np.asarray(st.x[L.region("a")]) > 0).sum()) <= 2


def test_remap_keeps_behaviour_and_new_region_is_silent():
    old = BrainConfig(regions=(R("h", 4),), projections=(P("in", "h", density=1.0), P("h", "out", density=1.0)))
    new = BrainConfig(regions=(R("v", 3), R("h", 4)),
                      projections=(P("in", "h", density=1.0), P("h", "out", density=1.0), P("in", "v", density=1.0),
                                   P("v", "out", density=1.0)))
    Lo = brain.build_layout(old, 3, A.NUM_ACTIONS, ("f0", "f1", "f2"))
    Ln = brain.build_layout(new, 4, A.NUM_ACTIONS, ("f0", "fX", "f1", "f2"))
    pop = jax.vmap(lambda k: brain.init_genome(k, Lo))(jax.random.split(jax.random.PRNGKey(0), 2))
    pop2 = brain.remap_genomes(pop, Lo, Ln, jax.random.PRNGKey(1))
    obs_o, obs_n = jnp.array([0.3, 1.0, 0.5]), jnp.array([0.3, 0.9, 1.0, 0.5])
    for i in range(2):
        go = jax.tree_util.tree_map(lambda a: a[i], pop)
        gn = jax.tree_util.tree_map(lambda a: a[i], pop2)
        so, sn = brain.init_state(go, Lo), brain.init_state(gn, Ln)
        for _ in range(3):
            so, _ = brain.step(old, Lo, go, so, obs_o, NO, jax.random.PRNGKey(0))
            sn, _ = brain.step(new, Ln, gn, sn, obs_n, NO, jax.random.PRNGKey(0))
        assert np.allclose(np.asarray(so.x[Lo.region("out")]), np.asarray(sn.x[Ln.region("out")]), atol=1e-5)


def test_named_inputs_match_across_resolutions_and_config_roundtrip():
    n5 = input_names(VisionConfig(columns=5, fov_degrees=120), BodyConfig(efference=True, taste=True))
    n9 = input_names(VisionConfig(columns=9, fov_degrees=120))
    assert "vis-30.hit" in n5 and "vis-30.hit" in n9 and "efc_EAT" in n5 and "taste" in n5
    exp = ExperimentConfig(body=BodyConfig(taste=True), brain=BrainConfig(
        regions=(R("a", 2, sign="inh", receptors=(("da", "gain", 1.0),), group="x/y"),), projections=(P("in", "a", depression=(0.1, 5.0), abcd=(1, -0.2, 0, 0)),),
        modulators=(ModulatorSpec("da", pos="a", baseline=0.1),)))
    back = ExperimentConfig.from_json(exp.to_json())
    assert back == exp


def test_delayed_sickness():
    from life.world import init_world, step_world
    from test_core import toy_ruleset
    rs = toy_ruleset()
    pv = np.zeros(rs.size, np.float32)
    edible = int(np.nonzero(rs.food_value > 0)[0][0])
    pv[edible] = 1.0
    rules = rs.to_arrays(4, pain_value=pv)
    cfg = WorldConfig(height=6, width=6, num_agents=1, sickness_delay=3, pain_decay=0.0, spawn_density=0.0)
    st = init_world(cfg, rules, jax.random.PRNGKey(0))._replace(held=jnp.array([edible]))
    pains = []
    for a in [A.EAT, A.NOOP, A.NOOP, A.NOOP, A.NOOP]:
        st, ev = step_world(cfg, rules, st, jnp.array([a]), jax.random.PRNGKey(0))
        pains.append(float(ev["pain"][0]))
    assert pains == [0.0, 0.0, 0.0, 1.0, 0.0] and float(st.taste[0]) == 0.0


def test_receptors_broadcast_gain_and_bias():
    # modulator 'm' = activity of region 's' (tonically ~tanh(3)); target 'a' has a gain receptor, 'c' a bias one
    rs = (R("s", 1, alpha=1.0, bias=3.0), R("a", 2, alpha=1.0, bias=0.0, receptors=(("m", "gain", 1.0),)),
          R("c", 2, alpha=1.0, bias=0.0, receptors=(("m", "bias", 0.5),)))
    cfg = BrainConfig(regions=rs, projections=(P("in", "a", density=1.0, w_init=0.1, evolve=False),),
                      modulators=(ModulatorSpec("m", pos="s"),))
    L, g, st = run(cfg, [1, 1, 1], n=3)
    m = np.tanh(3.0)
    assert np.allclose(np.asarray(st.x[L.region("a")]), np.tanh(0.3 * np.exp(m)), atol=1e-4)
    assert np.allclose(np.asarray(st.x[L.region("c")]), np.tanh(0.5 * m), atol=1e-4)
    assert L.groups == ("", "", "", "", "")
