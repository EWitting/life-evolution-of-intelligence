"""What a brain variant costs to evolve: one generation as run_evolution simulates it (all episodes at once, the
stage's evolution settings), timed after compilation, with the peak GPU memory JAX used and the peak host memory of
the process (the remap of the population runs on the host). One variant per process, so the peaks are its own:
    STAGE=x.critic python scripts/probes/cost.py <variant>        variants: see variants.py"""
import os, sys, time, resource
import jax, numpy as np
from life.experiments import stages as S
from life.run import make_simulate, load_population, latest_run, make_layout
from variants import variant

s = S.STAGES[os.environ.get("STAGE", "x.critic")]
v = sys.argv[1] if len(sys.argv) > 1 else "base"
exp = S.make_exp(s, variant(s.brain, v), "cost", 1, 0)
E = exp.evolution.episodes
pop = load_population(latest_run(S.STAGES[s.parent].name), exp)
host_remap = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
rs, fn = s.build(exp)
rules = fn(0, jax.random.PRNGKey(1))
if rules.food_value.ndim == 1 + (exp.world.switch_tick > 0):
    rules = jax.tree_util.tree_map(lambda a: a[None].repeat(E, 0), rules)
sim = jax.jit(jax.vmap(make_simulate(exp, record=False), in_axes=(0, None, 0)))
keys = jax.random.split(jax.random.PRNGKey(2), E)
t0 = time.time(); out = sim(rules, pop, keys); jax.block_until_ready(out); first = time.time() - t0
t0 = time.time(); out = sim(rules, pop, keys); jax.block_until_ready(out); gen = time.time() - t0
stats = jax.devices()[0].memory_stats() or {}
L = make_layout(exp)
print(f"{v:28s} neurons {L.n:5d}  {E} episodes x {exp.world.num_agents} animals x {exp.evolution.ticks_per_generation} ticks: "
      f"{gen:6.1f} s per generation (first {first:.0f} s with compiling)  GPU peak {stats.get('peak_bytes_in_use', 0) / 2 ** 20:6.0f} MiB  "
      f"host peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:6.0f} MiB (after the remap {host_remap:.0f})", flush=True)
