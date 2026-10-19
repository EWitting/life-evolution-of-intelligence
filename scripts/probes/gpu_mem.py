"""GPU memory of one evolution process of a stage: one generation as `run_evolution` simulates it (all episodes at
once), then the recording of the last generation (the dashboard sample). Peak bytes JAX had in use after each; the
process also holds the CUDA context (a few hundred MB, visible only in nvidia-smi). Run one process per stage.

    XLA_PYTHON_CLIENT_PREALLOCATE=false python scripts/probes/gpu_mem.py <stage key>"""
import sys, time
import jax
from life.experiments import stages as S
from life.run import make_simulate, load_population, latest_run

s = S.STAGES[sys.argv[1]]
exp = S.make_exp(s, s.brain, "mem", 1, 0)
E = exp.evolution.episodes
pop = load_population(latest_run(S.STAGES[s.parent].name), exp)
rs, fn = s.build(exp)
rules = fn(0, jax.random.PRNGKey(1))
if rules.food_value.ndim == 1:
    rules = jax.tree_util.tree_map(lambda a: a[None].repeat(E, 0), rules)
dev = jax.devices()[0]
sim = jax.jit(jax.vmap(make_simulate(exp, record=False), in_axes=(0, None, 0)))
t0 = time.time()
jax.block_until_ready(sim(rules, pop, jax.random.split(jax.random.PRNGKey(2), E)))
peak = dev.memory_stats()["peak_bytes_in_use"] / 2**20
print(f"{s.key}: {exp.world.num_agents} animals x {E} episodes ({exp.world.height}^2): generation peak {peak:.0f} MiB "
      f"({time.time() - t0:.0f} s with compilation)", flush=True)
from dataclasses import replace
ra = exp.evolution.record_agents
f = min(ra, exp.world.num_agents // exp.evolution.siblings) / exp.world.num_agents
exp_rec = replace(exp, world=replace(exp.world, num_agents=round(exp.world.num_agents * f), height=max(8, round(exp.world.height * f ** 0.5)),
                                     width=max(8, round(exp.world.width * f ** 0.5)), patches=round(exp.world.patches * f)))
rec = make_simulate(exp_rec, record=True)
r0 = jax.tree_util.tree_map(lambda a: a[0], rules)
jax.block_until_ready(rec(r0, jax.tree_util.tree_map(lambda a: a[:exp_rec.world.num_agents // exp.evolution.siblings], pop), jax.random.PRNGKey(3)))
print(f"{s.key}: after the recording peak {dev.memory_stats()['peak_bytes_in_use'] / 2**20:.0f} MiB", flush=True)
