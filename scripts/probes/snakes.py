"""The hidden snakes of the dopamine-critic world (WorldConfig.snakes) met by a population that cannot hear yet: how
often an animal hears a rattle, is bitten, how long after a rattle a bite comes, and what the snakes cost. The
stage's parent population on its brain (`hear` is in the input but nothing reads it), in the stage's world with
snakes added; the same worlds without snakes for the cost.
    [STAGE=2.2] [WORLDS=4] python scripts/probes/snakes.py snakes=64 rattle_radius=4 snake_food=4 [...]
lower-case field=value: a WorldConfig field (several settings: separate runs)."""
import os, sys
from dataclasses import replace
import jax, jax.numpy as jnp, numpy as np
from life import brain
from life.experiments import stages as S
from life.run import load_population, latest_run, make_layout, make_tick, make_simulate

s = S.STAGES[os.environ.get("STAGE", "2.2")]
WORLDS = int(os.environ.get("WORLDS", 4))
GAPS = 12             # histogram of ticks since the last rattle at a bite, 1 .. GAPS, and 'none within GAPS'
over = {k: v for k, v in (a.split("=") for a in sys.argv[1:])}
w = replace(s.world, **{k: type(getattr(s.world, k))(float(v)) for k, v in over.items()})
s = replace(s, world=w, body=replace(s.body, hear=True))
print("settings:", over, flush=True)
exp = S.make_exp(s, s.brain, "snakes", 1, 0)
pop = load_population(os.environ.get("RUN") or S.seed_runs(S.STAGES["2.2"])[0], exp, seed=0)   # the frozen 2.2, seed 0
layout = make_layout(exp)
tick = make_tick(exp, layout)
N, T = exp.world.num_agents, s.ticks
rs, fn = s.build(exp)


@jax.jit
def life(rules, pop, key):
    """Per tick: rattle heard, bitten, alive. The prelude of run.make_simulate (siblings, birth)."""
    P = pop.b.shape[0]
    b0 = jax.vmap(lambda g: brain.init_state(g, layout))(pop)
    pop = brain.block_genome(layout, pop)
    sib = lambda x: jnp.repeat(x, N // P, axis=0)
    pop, b0 = jax.tree_util.tree_map(sib, pop), jax.tree_util.tree_map(sib, b0)
    kw, ks = jax.random.split(key)
    from life.world import init_world
    world = init_world(exp.world, rules, kw)

    def one(c, k):
        world, b, sig = c
        world, b, sig, acts, obs, ev = tick(rules, pop, world, b, sig, k, -jnp.ones(N, jnp.int32))
        return (world, b, sig), (world.hear * world.rattle_sign > 0, world.bites, world.alive)

    (world, _, _), (rat, bites, alive) = jax.lax.scan(one, (world, b0, jnp.zeros((N, 3))), jax.random.split(ks, T))
    bit = jnp.diff(bites, axis=0, prepend=jnp.zeros((1, N))) > 0
    return rat, bit, alive


def since_rattle(rat, bit):
    """[GAPS + 1] counts of bites by ticks since the last rattle (last bin: none within GAPS)."""
    rat, bit = np.asarray(rat), np.asarray(bit)
    last = np.full(rat.shape[1], -10 ** 9)
    h = np.zeros(GAPS + 1)
    for t in range(rat.shape[0]):
        g = t - last[bit[t]]
        h += np.bincount(np.where((g >= 1) & (g <= GAPS), g - 1, GAPS), minlength=GAPS + 1)
        last = np.where(rat[t], t, last)
    return h


sim_on, sim_off = make_simulate(exp, record=False), make_simulate(S.make_exp(replace(s, world=replace(w, snakes=0)), s.brain, "snakes", 1, 0), record=False)
acc, hist = [], np.zeros(GAPS + 1)
for i, k in enumerate(jax.random.split(jax.random.PRNGKey(41), WORLDS)):
    kr, ks = jax.random.split(k)
    rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
    rat, bit, alive = life(rules, pop, ks)
    hist += since_rattle(rat, bit)
    f_on = float((s.fitness or S.default_fitness)(sim_on(rules, pop, ks)[0]).mean())
    f_off = float((s.fitness or S.default_fitness)(sim_off(rules, pop, ks)[0]).mean())
    acc.append([f_on, f_off, float(np.asarray(rat).sum(0).mean()), float(np.asarray(bit).sum(0).mean()),
                float(np.asarray(alive).sum(0).mean())])
    print(f"  world {i + 1}/{WORLDS}", flush=True)
a = np.array(acc); m = a.mean(0); d = a[:, 0] - a[:, 1]
print(f"fitness with snakes {m[0]:.0f}, without {m[1]:.0f}, paired {d.mean():+.0f} +-{d.std(ddof=1) / np.sqrt(WORLDS):.0f}; "
      f"lifetime {m[4]:.0f}; per life: rattles {m[2]:.1f}, bites {m[3]:.1f}")
print("bites by ticks since the last rattle:", " ".join(f"{g + 1}:{hist[g] / hist.sum():.2f}" for g in range(GAPS)),
      f"none:{hist[GAPS] / hist.sum():.2f}")
