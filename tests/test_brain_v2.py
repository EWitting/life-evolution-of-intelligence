"""Brain v2 features: Dale's law, hard-wiring, gain, eligibility, depression, delta, k-WTA, region modulators,
remapping genomes across layouts, delayed sickness, named inputs, divisive normalisation, metabolic cost, per-life looks. Run: .venv\\Scripts\\python.exe -m pytest -q"""
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


def test_divisive_normalisation_instant_and_lagged():
    # 4 neurons driven with h = 4 each: the pool (mean of max(h, 0)) is 4, so with norm 2 the input becomes 4 / 9
    mk = lambda lag: BrainConfig(regions=(R("a", 4, sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, norm=2.0, norm_lag=lag),),
                                 projections=(P("in", "a", topology="one_to_one", w_init=4.0, evolve=False),))
    L, g, st = run(mk(False), [1, 1, 1, 1], n=1, n_in=4)
    assert np.allclose(np.asarray(st.x[L.region("a")]), np.tanh(4 / 9), atol=1e-5)
    L, g, st1 = run(mk(True), [1, 1, 1, 1], n=1, n_in=4)       # lagging inhibition: the onset passes at full strength
    assert np.allclose(np.asarray(st1.x[L.region("a")]), np.tanh(4.0), atol=1e-5)
    L, g, st2 = run(mk(True), [1, 1, 1, 1], n=2, n_in=4)
    assert np.allclose(np.asarray(st2.x[L.region("a")]), np.tanh(4 / 9), atol=1e-5)
    L, g, st = run(mk(False), [1, 0, 0, 0], n=1, n_in=4)        # a lone active neuron is barely divided (pool = 1)
    assert np.allclose(float(st.x[L.region("a")][0]), np.tanh(4 / 3), atol=1e-5)


def test_brain_cost_raises_hunger():
    from life.world import init_world, step_world
    from test_core import toy_ruleset
    rules = toy_ruleset().to_arrays(4)
    cfg = WorldConfig(height=6, width=6, num_agents=2, spawn_density=0.0, brain_cost=1.0)
    st = init_world(cfg, rules, jax.random.PRNGKey(0))
    st, _ = step_world(cfg, rules, st, jnp.array([A.NOOP, A.NOOP]), jax.random.PRNGKey(0), jnp.array([0.0, 0.5]))
    lost = cfg.max_food - np.asarray(st.food)
    assert np.allclose(lost, [cfg.hunger_per_tick, 1.5 * cfg.hunger_per_tick], atol=1e-6)
    cfg = WorldConfig(height=6, width=6, num_agents=3, spawn_density=0.0, move_cost=0.5, turn_cost=0.25)
    st = init_world(cfg, rules, jax.random.PRNGKey(0))
    st, _ = step_world(cfg, rules, st, jnp.array([A.NOOP, A.FORWARD, A.TURN_LEFT]), jax.random.PRNGKey(0))
    lost = (cfg.max_food - np.asarray(st.food)) / cfg.hunger_per_tick
    assert np.allclose(lost, [1.0, 1.5, 1.25], atol=1e-5)                 # walking and turning cost energy


def test_novel_looks_are_drawn_per_life_and_fed_stat():
    from life.experiments import stages as S
    from life.run import make_simulate
    s = S.STAGES["1.6"]
    exp = S.make_exp(s, s.brain, "t", 1, 0)
    rs, fn = S.learning_world(exp, reverse=True)
    a, b = fn(0, jax.random.PRNGKey(1)), fn(0, jax.random.PRNGKey(2))
    app_a, app_b = np.asarray(a.appearance), np.asarray(b.appearance)     # [E, 2 phases, M, K]
    berry = lambda v: rs.local(S.variant(v)[S.BERRY])
    assert np.allclose(app_a[0, 0], app_a[0, 1])                          # the look does not change mid-life
    assert not np.allclose(app_a[0, 0, berry(2)], app_b[0, 0, berry(2)])  # but it does between lives
    assert np.allclose(app_a[0, 0, berry(4)], app_b[0, 0, berry(4)])      # ancestral types keep their look
    assert np.allclose(app_a[0, 0, berry(2)] @ app_a[0, 0, berry(0)], S.NOVEL_SIM, atol=1e-5)
    fv = np.asarray(a.food_value)[0]
    novel = np.array([fv[:, berry(v)] for v in S.NOVEL])                  # [4 types, 2 phases]
    assert sorted(novel[:, 0]) == [S.POISON_FOOD, S.POISON_FOOD, 3.0, 3.0]  # two of the four are poison
    assert (novel[:, 0] != novel[:, 1]).all()                             # and good and poison swap mid-life
    assert fv[0, berry(0)] == 3.0 and fv[0, berry(4)] == S.POISON_FOOD   # the ancestral types keep their meaning
    # fed = sum over ticks alive of food / max_food: an agent that never eats and starves after 400 ticks gets ~200
    exp2 = ExperimentConfig(world=WorldConfig(height=8, width=8, num_agents=2, spawn_density=0.0),
                            evolution=EvolutionConfig(ticks_per_generation=500, record_weights_every=100))
    from life.run import make_layout
    L = make_layout(exp2)
    pop = jax.vmap(lambda k: brain.init_genome(k, L))(jax.random.split(jax.random.PRNGKey(0), 2))
    from test_core import toy_ruleset
    stats, _ = make_simulate(exp2, record=False)(toy_ruleset().to_arrays(exp2.vision.appearance_dim), pop, jax.random.PRNGKey(0))
    assert np.allclose(np.asarray(stats["fed"]), 200.0, atol=2.0) and (np.abs(np.asarray(stats["alive_ticks"]) - 400) <= 1).all()


def test_phases_propagate_within_one_tick():
    # in -> a -> b: synchronously b sees a one tick late; with b in a later phase it responds in the same tick
    mk = lambda ph: BrainConfig(regions=(R("a", 3, sign="exc", alpha=1.0, bias=0.0, evolve_bias=False),
                                         R("b", 3, sign="exc", alpha=1.0, bias=0.0, evolve_bias=False, phase=ph)),
                                projections=(P("in", "a", topology="one_to_one", w_init=2.0, evolve=False),
                                             P("a", "b", topology="one_to_one", w_init=2.0, evolve=False)))
    L, g, st = run(mk(0), [1, 0, 1], n=1)
    assert np.allclose(np.asarray(st.x[L.region("b")]), 0.0)
    L, g, st = run(mk(1), [1, 0, 1], n=1)
    a = np.tanh(2.0)
    assert np.allclose(np.asarray(st.x[L.region("a")]), [a, 0, a], atol=1e-5)
    assert np.allclose(np.asarray(st.x[L.region("b")]), [np.tanh(2 * a), 0, np.tanh(2 * a)], atol=1e-5)
    L2, g2, st2 = run(mk(0), [1, 0, 1], n=2)                         # the synchronous brain gets there a tick later
    assert np.allclose(np.asarray(st2.x[L2.region("b")]), np.asarray(st.x[L.region("b")]), atol=1e-5)


def test_tunable_hardwired_strength_scales_whole_projection():
    from life.evolution import mutate
    cfg = BrainConfig(regions=(R("a", 3, sign="exc", alpha=1.0), R("i", 2, sign="inh", alpha=1.0)),
                      projections=(P("in", "a", density=1.0, w_init=-2.0, evolve=False, tune=True),
                                   P("a", "i", density=1.0, w_init=1.5, evolve=False, tune=True),
                                   P("i", "out", density=1.0, w_init=1.0, evolve=False)))
    L = brain.build_layout(cfg, n_in=3, n_out=A.NUM_ACTIONS)
    g = brain.init_genome(jax.random.PRNGKey(0), L)
    ecfg = EvolutionConfig(tune_prob=1.0, tune_std=0.5, tune_range=3.0)
    changed = 0
    for seed in range(5):
        m = mutate(jax.random.PRNGKey(seed), g, ecfg, L)
        ia = np.asarray(m.w0[L.region("in"), L.region("a")])
        ai = np.asarray(m.w0[L.region("a"), L.region("i")])
        io = np.asarray(m.w0[L.region("i"), L.region("out")])
        assert np.allclose(ia, ia[0, 0]) and -6.0 - 1e-5 <= ia[0, 0] <= -2.0 / 3 + 1e-5   # one factor, sign kept, in range
        assert np.allclose(ai, ai[0, 0]) and 0.5 - 1e-5 <= ai[0, 0] <= 4.5 + 1e-5
        assert np.allclose(io, 1.0)                                                   # hard-wired without tune: exact
        changed += int(abs(ia[0, 0] + 2.0) > 1e-3)
    assert changed >= 3


def test_valence_programmes_fire_only_in_their_context():
    # stage 1.1 brain with every evolved weight removed: only the hard-wired value -> programme -> motor circuit acts
    from life.experiments import stages as S
    from life.run import make_layout
    s = S.STAGES["1.1"]
    exp = S.make_exp(s, s.brain, "t", 1, 0)
    L = make_layout(exp)
    names = list(L.in_names)
    g = brain.init_genome(jax.random.PRNGKey(0), L)
    w0 = np.where(np.asarray(L.evolve_w) > 0, 0.0, np.asarray(g.w0))
    mask = np.asarray(g.mask).copy()
    b = np.where(np.asarray(L.evolve_b) > 0, 0.0, np.asarray(g.b))
    app, av = L.region("valence_app"), L.region("valence_av")

    def syn(feature, post, w):
        i = names.index(feature)
        w0[i, post], mask[i, post] = w, 1.0
    syn("vis+0.app0", app.start, 3.0); syn("vis+0.app0", app.start + 1, 3.0)   # look 0, seen: valued
    syn("held_app0", app.start + 2, 3.0)                                        # look 0, in hand: valued
    syn("held_app1", av.start + 2, 3.0)                                         # look 1, in hand: bad
    g = g._replace(w0=jnp.asarray(w0), mask=jnp.asarray(mask), b=jnp.asarray(b))

    def react(near=0.0, seen=None, held=None):
        o = np.zeros(L.n_in, np.float32)
        if near:
            o[names.index("vis+0.hit")], o[names.index("vis+0.near")] = 1.0, near
        if seen is not None:
            o[names.index(f"vis+0.app{seen}")] = 1.0
        if held is not None:
            o[names.index("held")], o[names.index(f"held_app{held}")] = 1.0, 1.0
        st, act = brain.step(exp.brain, L, g, brain.init_state(g, L), jnp.asarray(o), NO, jax.random.PRNGKey(0))
        return [float(st.x[L.region(r)][0]) for r in ("approach", "grasp", "ingest", "reject")], int(act)

    on, off = 0.8, 0.05                                   # one tick is enough: the layers update in order (phases)
    p, act = react(near=0.4, seen=0)                      # something valued at a distance: approach
    assert p[0] > on and max(p[1:]) < off and act == A.FORWARD
    p, act = react(near=0.8, seen=0)                      # the same thing adjacent, empty hand: grasp, no approach
    assert p[1] > on and max(p[0], p[2], p[3]) < off and act == A.USE
    p, _ = react(near=0.8)                                # something adjacent that is not valued: nothing
    assert max(p) < off
    p, act = react(held=0)                                # something valued in hand: ingest
    assert p[2] > on and max(p[0], p[1], p[3]) < off and act == A.EAT
    p, act = react(held=1)                                # something bad in hand: reject (USE puts it down), no eating
    assert p[3] > on and max(p[:3]) < off and act == A.USE
    p, act = react(near=0.8, seen=0, held=0)              # full hand next to a valued bush: eat, do not grasp
    assert p[2] > on and p[1] < off and act == A.EAT
