"""The first ticks of a life, by world size: does what is left on the bushes turn poisonous? A good bush is emptied
in six bites, a poison bush loses one berry and stays 'full', so where animals are dense the bushes that still
carry berries are more and more the poison ones. A frozen population in its stage's world at several sizes (same
animals, same bush density), recorded for the first TICKS ticks.

    uv run python scripts/probes/early.py <frozen stage key> [size ...]     WORLDS=<n> (default 2), TICKS (default 400)
'near' = within 5 cells of a living animal (what the animals have around them), averaged over the animals."""
import os, sys
from dataclasses import replace
import jax, numpy as np
from life.experiments import stages as S
from life.freeze import load_frozen
from life.run import make_simulate

s0 = S.STAGES[sys.argv[1]]
WORLDS, TICKS, R = int(os.environ.get("WORLDS", 2)), int(os.environ.get("TICKS", 400)), 5
AT = [t for t in (0, 25, 50, 100, 200, 300, TICKS - 1) if t < TICKS]
for size in [int(a) for a in sys.argv[2:]] or [s0.world.height]:
    s = replace(s0, world=replace(s0.world, height=size, width=size), ticks=TICKS)
    exp = S.make_exp(s, s.brain, "early_probe", 1, 0)
    pop = load_frozen(s0.key, 0, exp)
    rs, fn = s.build(exp)
    sim = make_simulate(exp, record=True)
    bush = np.array([rs.local(S.variant(v)[S.BUSH]) for v in S.NOVEL])
    berry = np.array([rs.local(S.variant(v)[S.BERRY]) for v in S.NOVEL])
    acc = []
    for k in jax.random.split(jax.random.PRNGKey(123), WORLDS):
        kr, ks = jax.random.split(k)
        rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
        bad = np.asarray(rules.pain_value)[berry] > 0
        _, rec = sim(rules, pop, ks)
        grid, pos, alive, ate = (np.asarray(rec[x]) for x in ("grid", "pos", "alive", "ate"))
        alive = alive.astype(bool); N = alive.shape[1]
        gb, pb = np.isin(ate, berry[~bad]), np.isin(ate, berry[bad])
        rows = []
        for t in AT:
            g = grid[t]
            good, poison = np.isin(g, bush[~bad]), np.isin(g, bush[bad])
            ng = npn = 0.0
            who = np.nonzero(alive[t])[0]
            for i in who:
                y, x = pos[t, i]
                sl = (slice(max(0, y - R), y + R + 1), slice(max(0, x - R), x + R + 1))
                ng += good[sl].sum(); npn += poison[sl].sum()
            rows.append([poison.sum() / max(1, good.sum() + poison.sum()), npn / max(1.0, ng + npn), ng / max(1, len(who)),
                         alive[t].mean(), gb[:t + 1].sum() / N, pb[:t + 1].sum() / N])
        acc.append(rows)
    m = np.mean(acc, 0)
    print(f"[{s0.key} {size}x{size}] bushes carrying berries: share that is poison, everywhere and near the animals; full good "
          f"bushes near an animal; animals alive; good and poison bites per animal so far", flush=True)
    for t, r in zip(AT, m):
        print(f"   tick {t:4d}: poison share {r[0]:.2f}, near animals {r[1]:.2f}; good bushes near {r[2]:4.1f}; alive {r[3]:4.0%}; "
              f"bites good {r[4]:5.1f} poison {r[5]:4.1f}", flush=True)
