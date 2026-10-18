"""Inside the pallium in real lives: the newest population of the pallium stage (or RUN) in its world, one recorded
life. At poison bites, good bites and ticks with nothing ahead: the pallium's activity (mean rate of its active
cells, number of cells above 0.05), its drive onto the aversive cells (pallium rates . learned weights) and the
aversive cells' activity; and how the learned pallium -> aversive weights grow over the life.

    STAGE=2.2 [RUN=<run dir>] uv run python scripts/probes/pallium_life.py"""
import os, fnmatch
import jax, numpy as np
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate, make_layout

s = S.STAGES[os.environ.get("STAGE", "2.2")]
exp = S.make_exp(s, s.brain, "pallium_life", 1, 0)
L = make_layout(exp)
pop = load_population(os.environ.get("RUN") or latest_run(s.name), exp, seed=0)
rs, fn = s.build(exp)
sim = make_simulate(exp, record=True)
kr, ks = jax.random.split(jax.random.PRNGKey(7))
rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
st, rec = sim(rules, pop, ks)
pain = np.asarray(rules.pain_value)
x = np.asarray(rec["x"], np.float32); ate = np.asarray(rec["ate"]); alive = np.asarray(rec["alive"])
T, N = ate.shape
pal, av = L.region("pallium"), L.region("valence_av")
hit = list(L.in_names).index("vis+0.hit")
look = [list(L.in_names).index(f"vis+0.app{k}") for k in range(exp.vision.appearance_dim)]
xp = x[:, :, pal]; xav = x[:, :, av].mean(-1)
# learned weights from the recorded snapshots (every record_weights_every ticks)
snaps = rec.get("w") if isinstance(rec, dict) and "w" in rec else None
bites_bad = (ate > 0) & (pain[ate] > 0); bites_good = (ate > 0) & (pain[ate] == 0)
nothing = (x[:, :, hit] == 0) & alive.astype(bool)
print(f"stage {s.key}: {bites_bad.sum()} poison bites, {bites_good.sum()} good bites in one life of {N} animals")
print(f"  {'':18s}{'look |x|':>10s}{'pallium: mean of top 4':>24s}{'cells > 0.05':>14s}{'aversive cells':>16s}")
for nm, m in (("poison bite", bites_bad), ("good bite", bites_good), ("nothing ahead", nothing)):
    top = np.sort(xp[m], -1)[:, -4:].mean() if m.any() else np.nan
    print(f"  {nm:18s}{np.abs(x[:, :, look][m]).sum(-1).mean():10.2f}{top:24.3f}{(xp[m] > 0.05).sum(-1).mean():14.1f}"
          f"{xav[m].mean():16.3f}")
print("  aversive activity in the tick before a bite, poison / good, by quarter of the life:")
q = np.arange(T) * 4 // T
for k in range(4):
    pb = bites_bad[1:] & (q[1:, None] == k); gb = bites_good[1:] & (q[1:, None] == k)
    print(f"    quarter {k + 1}: {xav[:-1][pb].mean():.3f} / {xav[:-1][gb].mean():.3f}  (bites {pb.sum()} / {gb.sum()})")
