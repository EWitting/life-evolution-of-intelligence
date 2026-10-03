"""Core behaviour tests. Run: .venv\\Scripts\\python.exe -m pytest -q"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from life import actions as A
from life import brain, ohol
from life.config import WorldConfig, VisionConfig, BrainConfig, EvolutionConfig, ExperimentConfig, RegionSpec
from life.ruleset import RulesetBuilder
from life.sensors import ray_offsets, observe_all, flatten_obs, obs_size
from life.world import init_world, step_world
from life.run import make_simulate, make_layout

K = 4


def toy_ruleset():
    """bush(10) gives berry(11) twice, then becomes stump(12) which regrows into bush after 5 ticks."""
    b = RulesetBuilder()
    b.add_object(10, "bush", permanent=True, holdable=False, num_uses=2, map_chance=1.0)
    b.add_object(11, "berry", food_value=1.0)
    b.add_object(12, "stump", permanent=True, holdable=False)
    b.add_object(13, "rock", permanent=True, holdable=False, blocks=True)
    b.add_transition(0, 10, 11, 10)
    b.add_transition(0, 10, 11, 12, last_use=True)
    b.add_decay(12, 10, 5)
    return b.build()


def empty_world(cfg, rules):
    st = init_world(cfg, rules, jax.random.PRNGKey(0))
    return st._replace(grid_obj=jnp.zeros_like(st.grid_obj), grid_uses=jnp.ones_like(st.grid_uses),
                       grid_timer=-jnp.ones_like(st.grid_timer))


def place(st, y, x, obj, rules):
    return st._replace(grid_obj=st.grid_obj.at[y, x].set(obj), grid_uses=st.grid_uses.at[y, x].set(rules.num_uses[obj]),
                       grid_timer=st.grid_timer.at[y, x].set(rules.decay_ticks[obj]))


def test_pick_eat_last_use_and_regrow():
    rs = toy_ruleset()
    rules = rs.to_arrays(K)
    cfg = WorldConfig(height=8, width=8, num_agents=1, hunger_per_tick=0.0)
    st = empty_world(cfg, rules)
    bush, berry, stump = rs.local(10), rs.local(11), rs.local(12)
    st = place(st, 2, 3, bush, rules)
    st = st._replace(pos=jnp.array([[3, 3]]), dir=jnp.array([0]), food=jnp.array([5.0]))
    key = jax.random.PRNGKey(1)
    st, _ = step_world(cfg, rules, st, jnp.array([A.USE]), key)
    assert int(st.held[0]) == berry and int(st.grid_obj[2, 3]) == bush and int(st.grid_uses[2, 3]) == 1
    st, ev = step_world(cfg, rules, st, jnp.array([A.EAT]), key)
    assert int(st.held[0]) == 0 and float(st.food[0]) == pytest.approx(5.0 + cfg.food_scale)
    st, _ = step_world(cfg, rules, st, jnp.array([A.USE]), key)     # last use -> stump
    assert int(st.held[0]) == berry and int(st.grid_obj[2, 3]) == stump and int(st.grid_timer[2, 3]) == 5
    st, _ = step_world(cfg, rules, st, jnp.array([A.USE]), key)     # drop not possible on stump; nothing changes
    assert int(st.held[0]) == berry
    for _ in range(5):
        st, _ = step_world(cfg, rules, st, jnp.array([A.NOOP]), key)
    assert int(st.grid_obj[2, 3]) == bush and int(st.grid_uses[2, 3]) == 2


def test_eat_on_pick_eats_what_is_grasped():
    rs = toy_ruleset()
    rules = rs.to_arrays(K)
    cfg = WorldConfig(height=8, width=8, num_agents=1, hunger_per_tick=0.0, eat_on_pick=True)
    st = empty_world(cfg, rules)
    bush, berry = rs.local(10), rs.local(11)
    st = place(st, 2, 3, bush, rules)
    st = st._replace(pos=jnp.array([[3, 3]]), dir=jnp.array([0]), food=jnp.array([5.0]))
    st, ev = step_world(cfg, rules, st, jnp.array([A.USE]), jax.random.PRNGKey(1))
    assert int(st.held[0]) == 0 and int(ev["ate"][0]) == berry              # nothing is carried: eaten in the same tick
    assert float(st.food[0]) == pytest.approx(5.0 + cfg.food_scale) and int(st.grid_uses[2, 3]) == 1
    assert float(st.taste[0]) == pytest.approx(1.0)


def test_move_turn_block_and_drop():
    rs = toy_ruleset()
    rules = rs.to_arrays(K)
    cfg = WorldConfig(height=8, width=8, num_agents=1, hunger_per_tick=0.0)
    st = empty_world(cfg, rules)
    st = place(st, 3, 5, rs.local(13), rules)
    st = st._replace(pos=jnp.array([[3, 3]]), dir=jnp.array([1]), held=jnp.array([rs.local(11)]))
    key = jax.random.PRNGKey(0)
    st, _ = step_world(cfg, rules, st, jnp.array([A.FORWARD]), key)
    assert st.pos[0].tolist() == [3, 4]
    st, _ = step_world(cfg, rules, st, jnp.array([A.FORWARD]), key)   # blocked by rock
    assert st.pos[0].tolist() == [3, 4]
    st, _ = step_world(cfg, rules, st, jnp.array([A.TURN_LEFT]), key)
    assert int(st.dir[0]) == 0
    st, _ = step_world(cfg, rules, st, jnp.array([A.USE]), key)       # drop berry on empty cell (2,4)
    assert int(st.held[0]) == 0 and int(st.grid_obj[2, 4]) == rs.local(11)
    st, _ = step_world(cfg, rules, st, jnp.array([A.USE]), key)       # pick it up again
    assert int(st.held[0]) == rs.local(11) and int(st.grid_obj[2, 4]) == 0


def test_starvation():
    rs = toy_ruleset()
    rules = rs.to_arrays(K)
    cfg = WorldConfig(height=4, width=4, num_agents=2, hunger_per_tick=1.0, max_food=2.0)
    st = empty_world(cfg, rules)
    for _ in range(3):
        st, _ = step_world(cfg, rules, st, jnp.array([A.NOOP, A.NOOP]), jax.random.PRNGKey(0))
    assert not bool(st.alive.any()) and int(st.alive_ticks[0]) == 2


def test_vision_sees_object_wall_and_agent():
    rs = toy_ruleset()
    rules = rs.to_arrays(K)
    v = VisionConfig(columns=3, fov_degrees=90.0, range=4, appearance_dim=K)
    cfg = WorldConfig(height=8, width=8, num_agents=2)
    st = empty_world(cfg, rules)
    st = place(st, 1, 4, rs.local(10), rules)
    st = st._replace(pos=jnp.array([[4, 4], [2, 1]]), dir=jnp.array([0, 0]))
    obs = observe_all(v, cfg, rules, st, jnp.asarray(ray_offsets(v)))
    vis = np.asarray(obs["vision"])
    centre = vis[0, 1]
    assert centre[0] == 1 and centre[2] == 0 and centre[3] == 0            # hit, not agent, not wall
    assert np.allclose(centre[4:], np.asarray(rules.appearance[rs.local(10)]))
    assert centre[1] == pytest.approx(1 - 3 / 4)                            # distance 3
    assert vis[1, 1, 3] == 1                                                # agent 1 looks up into the wall
    assert flatten_obs(obs).shape == (2, obs_size(v))


def test_simulate_generation_runs():
    rs = toy_ruleset()
    exp = ExperimentConfig(world=WorldConfig(height=10, width=10, num_agents=4), vision=VisionConfig(columns=3, range=3),
                           brain=BrainConfig(regions=(RegionSpec("hidden", 4),)),
                           evolution=EvolutionConfig(ticks_per_generation=5, record_weights_every=5))
    layout = make_layout(exp)
    pop = jax.vmap(lambda k: brain.init_genome(k, layout))(jax.random.split(jax.random.PRNGKey(0), 4))
    stats, recs = make_simulate(exp, record=True)(rs.to_arrays(exp.vision.appearance_dim), pop, jax.random.PRNGKey(3))
    assert stats["alive_ticks"].shape == (4,) and recs["grid"].shape == (5, 10, 10) and recs["w_snap"].shape[0] == 1


@pytest.mark.skipif(not (ohol.DEFAULT_DATA_DIR / "objects").exists(), reason="OHOL data not downloaded")
def test_ohol_gooseberry_slice():
    data = ohol.load()
    assert data.objects[30].name.startswith("Wild Gooseberry Bush")
    rs = ohol.slice_ruleset(data, [30, 31, 279])
    bush, berry, empty_bush = rs.local(30), rs.local(31), rs.local(279)
    assert rs.food_value[berry] > 0 and rs.holdable[berry] and not rs.holdable[bush]
    t = rs.use_table[0, bush]
    assert t >= 0 and rs.trans_new_actor[t] == berry and rs.trans_new_target[t] == bush
    lt = rs.last_use_table[0, bush]
    assert lt >= 0 and rs.trans_new_target[lt] == empty_bush
    assert rs.decay_new[empty_bush] == -1          # in OHOL the empty bush only regrows after watering
    rs2 = ohol.slice_ruleset(data, [30, 31, 279], clones={30: 100030, 31: 100031, 279: 100279},
                             extra_decays={279: (30, 50)})
    cb = rs2.local(100030)
    assert rs2.trans_new_actor[rs2.use_table[0, cb]] == rs2.local(100031)
    assert rs2.decay_new[rs2.local(279)] == rs2.local(30) and rs2.decay_ticks[rs2.local(279)] == 50
    assert rs2.decay_new[rs2.local(100279)] == cb


def test_siblings_share_a_genome(tmp_path):
    from life.run import run_evolution, make_simulate, make_layout
    from life import brain
    exp = ExperimentConfig(world=WorldConfig(height=8, width=8, num_agents=8, spawn_density=0.1),
                           evolution=EvolutionConfig(generations=2, ticks_per_generation=50, record_weights_every=50,
                                                     siblings=4))
    rs = toy_ruleset()
    out = run_evolution(exp, rs, lambda st: st["alive_ticks"].astype(jnp.float32), out_dir=tmp_path, verbose=False,
                        dashboard=False)
    assert out["pop"].b.shape[0] == 2                                  # 8 agents = 2 genomes x 4 siblings
    with np.load(tmp_path / "population.npz") as f:
        assert f["w0"].shape[0] == 2
    stats, _ = make_simulate(exp, record=False)(rs.to_arrays(exp.vision.appearance_dim), out["pop"], jax.random.PRNGKey(0))
    assert stats["alive_ticks"].shape == (8,)                          # statistics stay per individual
