"""Speed of one unrecorded generation at several scales, on whatever device JAX uses (CPU here, the GPU in WSL2).
The stage's brain and world (default 2.2), with the number of animals and the world's area scaled together, so the
density stays the same; the parent's population tiled to the size. Lives are cut to TICKS (default 1000) so that a
run is short; the time is given per 1000 ticks.

    [STAGE=2.2] [TICKS=1000] uv run python scripts/probes/bench.py [agents ...]      default 256 512 1024 2048"""
import os, sys, time, math
from dataclasses import replace
import jax
from life.experiments import stages as S
from life.run import make_simulate, load_population, latest_run

s = S.STAGES[os.environ.get("STAGE", "2.2")]
TICKS = int(os.environ.get("TICKS", 1000))
sizes = [int(a) for a in sys.argv[1:]] or [256, 512, 1024, 2048]
print(f"device {jax.devices()[0]}, stage {s.key}, {TICKS} ticks", flush=True)
for n in sizes:
    side = int(round(s.world.height * math.sqrt(n / s.world.num_agents)))
    st = replace(s, ticks=TICKS, world=replace(s.world, num_agents=n, height=side, width=side))
    exp = S.make_exp(st, st.brain, "bench", 1, 0)
    pop = load_population(latest_run(S.STAGES[s.parent].name), exp)
    rs, fn = st.build(exp)
    rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, jax.random.PRNGKey(1)))
    sim = make_simulate(exp, record=False)
    t0 = time.time(); out, _ = sim(rules, pop, jax.random.PRNGKey(2)); jax.block_until_ready(out); tc = time.time() - t0
    t0 = time.time(); out, _ = sim(rules, pop, jax.random.PRNGKey(3)); jax.block_until_ready(out); t = time.time() - t0
    print(f"  {n:5d} animals, world {side} x {side}: {1000 * t / TICKS:6.2f} s per 1000 ticks "
          f"({1e6 * t / TICKS / n:6.1f} us per animal-tick; first call with compilation {tc:.0f} s)", flush=True)
