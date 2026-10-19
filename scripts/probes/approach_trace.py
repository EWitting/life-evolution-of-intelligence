"""What the pallium and the aversive cells do while a creature approaches a bush, by distance (one recorded life of
64 animals, as da_trace.py, generation 0 of the parent population):
    STAGE=x.critic DENSE=1 python scripts/probes/approach_trace.py plife plife+pnt
Per variant, for the object seen straight ahead (vis+0) at distance 1 (at the mouth) to 5, split into poison bushes
and good bushes, second half of life (after lessons): share of those ticks, pallium activity (sum), aversive cells,
and how often the creature then turns away (TURN_LEFT / TURN_RIGHT)."""
import os, sys
from dataclasses import replace
import jax, numpy as np
from life import actions as A
from life.experiments import stages as S
from life.sensors import ray_offsets
from life.run import load_population, latest_run, make_simulate, make_layout
from life.world import DIRS
from variants import variant

s = S.STAGES[os.environ.get("STAGE", "x.critic")]
NA = 64
f = NA / s.world.num_agents
world = replace(s.world, num_agents=NA, height=round(s.world.height * f ** 0.5), width=round(s.world.width * f ** 0.5),
                patches=round(s.world.patches * f))
for v in sys.argv[1:]:
    exp = S.make_exp(replace(s, world=world), variant(s.brain, v), "approach", 1, 0)
    exp = replace(exp, evolution=replace(exp.evolution, siblings=1, record_weights_every=exp.evolution.ticks_per_generation))
    pop = load_population(os.environ.get("RUN") or latest_run(S.STAGES[s.parent].name), exp, seed=0)
    L = make_layout(exp)
    if os.environ.get("DENSE"):
        pl = (np.asarray(L.rule) > 0) & (np.asarray(L.allowed) > 0)
        pop = pop._replace(mask=jax.numpy.where(pl[None], 1.0, pop.mask))
    rs, fn = s.build(exp)
    kr, ks = jax.random.split(jax.random.PRNGKey(31))
    rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
    _, rec = make_simulate(exp, record=True)(rules, pop, ks)
    rec = {k: np.asarray(rec[k]) for k in ("x", "alive", "action", "grid", "pos", "dir")}
    x = rec["x"].astype(np.float32)
    # the object straight ahead along the forward eye's line (grid, not the input: the looks are adapted), as the
    # brain of tick t saw it: the world after the step of t-1 (other creatures in the line are ignored)
    off = ray_offsets(exp.vision)[:, exp.vision.columns // 2]                       # [4, R, 2]
    grid = np.r_[rec["grid"][:1] * 0, rec["grid"][:-1]]
    pos, dr = np.r_[rec["pos"][:1], rec["pos"][:-1]].astype(int), np.r_[rec["dir"][:1], rec["dir"][:-1]].astype(int)
    cells = pos[:, :, None, :] + off[dr]                                             # [T, N, R, 2]
    H, W = grid.shape[1:]
    inb = (cells[..., 0] >= 0) & (cells[..., 0] < H) & (cells[..., 1] >= 0) & (cells[..., 1] < W)
    cy, cx = np.clip(cells[..., 0], 0, H - 1), np.clip(cells[..., 1], 0, W - 1)
    obj = grid[np.arange(grid.shape[0])[:, None, None], cy, cx] * inb
    hit = (obj != 0) | ~inb
    first = hit.argmax(-1)
    dist = np.where(hit.any(-1), first + 1, 0)                                       # 1 = at the mouth .. R
    ahead = np.take_along_axis(obj, first[..., None], -1)[..., 0]
    pv, fv = np.asarray(rules.pain_value), np.asarray(rules.food_value)
    cls = np.zeros(pv.shape[0], np.int8)                                             # 1 good bush, 2 poison bush
    for t in range(6):
        b, berry = rs.local(S.variant(t)[S.BUSH]), rs.local(S.variant(t)[S.BERRY])
        cls[b] = 2 if pv[berry] > 0 else (1 if fv[berry] > 0 else 0)
    seen = cls[ahead]
    T = x.shape[0]
    late = (np.arange(T) >= T // 2)[:, None] & rec["alive"].astype(bool)
    pal = x[..., L.region("pallium")].sum(-1)
    av = x[..., L.region("valence_av")].mean(-1)
    turn = np.isin(np.r_[rec["action"][1:], rec["action"][-1:]], [A.TURN_LEFT, A.TURN_RIGHT])   # the next action
    print(f"{v}: lifetime {rec['alive'].sum(0).mean():.0f}; second half, object straight ahead by distance")
    for k, label in ((2, "poison bush"), (1, "good bush")):
        row = []
        for d in range(1, exp.vision.range + 1):
            m = late & (seen == k) & (dist == d)
            row.append(f"d{d}: {m.mean():5.1%} pal {pal[m].mean():.2f} av {av[m].mean():.2f} turn {turn[m].mean():.2f}")
        print(f"  {label:12s} " + " | ".join(row))
