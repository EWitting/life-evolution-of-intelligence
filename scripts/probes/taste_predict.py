"""Could the critic's value cells predict taste at all? One recorded life (64 animals, the stage's world scaled down,
as da_trace.py) of a variant; then, offline, how well the pallium of tick t predicts the taste of tick t+1 (what the
value cells must do for the error at a bite to vanish), against what better readers could do:
    STAGE=x.critic DENSE=1 python scripts/probes/taste_predict.py [variant] [NAME=value ...]
    linear         least squares on the pallium (any sign, with bias)
    dale           least squares on the pallium with weights >= 0 and no bias (the pallium is excitatory and the
                   value cells have bias 0)
    ahead          the mean taste for each object straight ahead (the best any reader of what is ahead can do)
    ahead+action   the same per object ahead and action taken (the pallium does not see the action)
    TD             the TD(0) fixed point on the pallium (LSTD, gamma TD_GAMMA): the value range it needs and the
                   error left at a bite, against taste alone (no critic)
R^2 is on the second half of the animals, fitted on the first half."""
import os, sys
from dataclasses import replace
import jax, numpy as np
from scipy.optimize import nnls
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate, make_layout
from life.world import DIRS
from variants import variant

args = [a for a in sys.argv[1:] if "=" not in a]
V = args[0] if args else "base"
for a in sys.argv[1:]:
    if "=" in a:
        k_, v_ = a.split("=")
        setattr(S, k_, float(v_))
s = S.STAGES[os.environ.get("STAGE", "x.critic")]
NA = 64
f = NA / s.world.num_agents
world = replace(s.world, num_agents=NA, height=round(s.world.height * f ** 0.5), width=round(s.world.width * f ** 0.5),
                patches=round(s.world.patches * f))
exp = S.make_exp(replace(s, world=world), variant(s.brain, V), "taste_predict", 1, 0)
exp = replace(exp, evolution=replace(exp.evolution, siblings=1))
pop = load_population(os.environ.get("RUN") or latest_run(S.STAGES[s.parent].name), exp, seed=0)
L = make_layout(exp)
if os.environ.get("DENSE"):
    pl = (np.asarray(L.rule) > 0) & (np.asarray(L.allowed) > 0)
    pop = pop._replace(mask=jax.numpy.where(pl[None], 1.0, pop.mask))
pop = jax.tree_util.tree_map(lambda a: a[:NA], pop)
rs, fn = s.build(exp)
kr, ks = jax.random.split(jax.random.PRNGKey(31))
rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
st, rec = make_simulate(exp, record=True)(rules, pop, ks)
rec = {k: np.asarray(v) for k, v in rec.items() if k != "w_snap"}

x = rec["x"].astype(np.float32)                                   # [T, NA, n]
i = L.names.index("pallium")
P = x[..., L.offsets[i]:L.offsets[i] + L.sizes[i]]
taste = x[..., list(L.in_names).index("taste")]
alive = rec["alive"].astype(bool)
T = x.shape[0]
front = rec["pos"].astype(int) + np.asarray(DIRS)[rec["dir"].astype(int)]
fy = np.clip(front[..., 0], 0, world.height - 1); fx = np.clip(front[..., 1], 0, world.width - 1)
ahead = rec["grid"][np.arange(T)[:, None], fy, fx].astype(int)
# the recording holds the world after each tick's step, the brain of tick t saw it before: what was ahead when it acted
ahead = np.r_[ahead[:1] * 0, ahead[:-1]]
act = rec["action"].astype(int)

# pairs (t, t+1) with the animal alive at both
ok = alive[:-1] & alive[1:]
half = np.arange(NA) < NA // 2
X_tr, X_te = P[:-1][ok & half[None]], P[:-1][ok & ~half[None]]
y = taste[1:]
y_tr, y_te = y[ok & half[None]], y[ok & ~half[None]]
r2 = lambda pred: 1 - ((y_te - pred) ** 2).mean() / y_te.var()
print(f"{V}: {ok.sum()} animal-ticks, taste > 0 on {np.mean(y_te > 0):.1%}, mean taste {y_te.mean():.3f}, "
      f"pallium active cells per tick {(P[alive] > 0.01).sum(-1).mean():.1f} of {P.shape[-1]}")

A = np.c_[X_tr, np.ones(len(X_tr))]
w = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ y_tr)
print(f"  linear        R^2 {r2(np.c_[X_te, np.ones(len(X_te))] @ w):.3f}")
wd, _ = nnls(X_tr, y_tr)
print(f"  dale          R^2 {r2(X_te @ wd):.3f}")
for name, key in (("ahead", ahead), ("ahead+action", ahead * 16 + act)):
    k_tr, k_te = key[:-1][ok & half[None]], key[:-1][ok & ~half[None]]
    m = {k: y_tr[k_tr == k].mean() for k in np.unique(k_tr)}
    print(f"  {name:13s} R^2 {r2(np.array([m.get(k, y_tr.mean()) for k in k_te])):.3f}")

# TD(0) fixed point on the pallium: V(P_t) = w.P_t (+ b), with taste of t+1 as the reward of the step t -> t+1
g = S.TD_GAMMA if hasattr(S, "TD_GAMMA") else 0.9
ok2 = alive[:-1] & alive[1:]
X0 = np.c_[P[:-1][ok2 & half[None]], np.ones((ok2 & half[None]).sum())]
X1 = np.c_[P[1:][ok2 & half[None]], np.ones((ok2 & half[None]).sum())]
wt = np.linalg.solve(X0.T @ (X0 - g * X1) + 1e-3 * np.eye(X0.shape[1]), X0.T @ y_tr)
Z0 = np.c_[P[:-1][ok2 & ~half[None]], np.ones((ok2 & ~half[None]).sum())]
Z1 = np.c_[P[1:][ok2 & ~half[None]], np.ones((ok2 & ~half[None]).sum())]
v0, v1 = Z0 @ wt, Z1 @ wt
delta = y_te + g * v1 - v0
bite = y_te > 0
print(f"  TD gamma {g}: value 5/50/95/max percentile {np.percentile(v0, 5):.2f} / {np.median(v0):.2f} / "
      f"{np.percentile(v0, 95):.2f} / {v0.max():.2f}   (value cells: k * mean rate <= TD_SCALE = {S.TD_SCALE if hasattr(S, 'TD_SCALE') else 1})")
print(f"    error at a bite {delta[bite].mean():+.3f} (taste alone {y_te[bite].mean():+.3f}),  "
      f"elsewhere mean {delta[~bite].mean():+.3f} sd {delta[~bite].std():.3f}")
# per object ahead at t: mean taste next tick and TD value
k_te = ahead[:-1][ok2 & ~half[None]]
top = np.argsort(-np.bincount(k_te))[:8]
print("  object ahead (id): share, taste next tick, TD value")
for k in top:
    m = k_te == k
    print(f"    {k:5d}  {m.mean():6.1%}  taste {y_te[m].mean():.3f}  value {v0[m].mean():+.2f}")

# where the error of the (object ahead, action) reader is left: per group, its share of the squared error
key = ahead * 16 + act
k_tr, k_te = key[:-1][ok & half[None]], key[:-1][ok & ~half[None]]
m = {k: y_tr[k_tr == k].mean() for k in np.unique(k_tr)}
res = y_te - np.array([m.get(k, y_tr.mean()) for k in k_te])
print(f"  taste values at a bite: {np.unique(np.round(y_te[y_te > 0], 2), return_counts=True)}")
print("  ahead+action groups by share of the squared error: object, action, share of ticks, taste mean / sd, error share")
tot = (res ** 2).sum()
for k in sorted(np.unique(k_te), key=lambda k: -(res[k_te == k] ** 2).sum())[:8]:
    g_ = k_te == k
    print(f"    {k // 16:5d} {k % 16:3d}  {g_.mean():6.1%}  {y_te[g_].mean():.2f} / {y_te[g_].std():.2f}  {(res[g_] ** 2).sum() / tot:6.1%}")
# the same with the action of the tick before (in case taste follows the action by two ticks)

# which tick of the pallium carries the look the action was taken on: fit the taste of t+1 from the pallium of t+lag
print("  pallium lag (taste of t+1 from the pallium of t+lag): R^2 linear / dale")
for lag in (-1, 0, 1, 2):
    a, b = max(0, -lag), T - 1 - max(0, lag)        # t runs over [a, b)
    okl = alive[a:b] & alive[a + 1:b + 1] & alive[a + lag:b + lag]
    Xl, yl = P[a + lag:b + lag], taste[a + 1:b + 1]
    tr_, te_ = okl & half[None], okl & ~half[None]
    A_ = np.c_[Xl[tr_], np.ones(tr_.sum())]
    w_ = np.linalg.solve(A_.T @ A_ + 1e-3 * np.eye(A_.shape[1]), A_.T @ yl[tr_])
    wd_, _ = nnls(Xl[tr_], yl[tr_])
    rr = lambda pred: 1 - ((yl[te_] - pred) ** 2).mean() / yl[te_].var()
    print(f"    {lag:+d}  {rr(np.c_[Xl[te_], np.ones(te_.sum())] @ w_):.3f} / {rr(Xl[te_] @ wd_):.3f}")

# does the pallium tell the objects apart? its pattern (the set of active cells) as a key, and the objects behind it
pat = (P > 0.01)
pk = np.array([hash(r.tobytes()) for r in pat.reshape(-1, pat.shape[-1])]).reshape(pat.shape[:2])
k_tr, k_te = pk[:-1][ok & half[None]], pk[:-1][ok & ~half[None]]
m = {k: y_tr[k_tr == k].mean() for k in np.unique(k_tr)}
print(f"  pallium pattern as key R^2 {r2(np.array([m.get(k, y_tr.mean()) for k in k_te])):.3f}")
names = rs.names
app = np.asarray(rules.appearance)
for obj in (17, 8):
    g_ = (ahead == obj) & alive
    pats, cnt = np.unique(pk[g_], return_counts=True)
    top_ = pats[np.argmax(cnt)]
    share = np.mean(pk[g_] == top_)
    same = (pk == top_) & alive
    objs, oc = np.unique(ahead[same], return_counts=True)
    print(f"  object {obj} ({names[obj]}): its commonest pattern on {share:.0%} of its ticks; ticks with that pattern by object: "
          + ", ".join(f"{o} {names[o]} {c / oc.sum():.0%}" for o, c in sorted(zip(objs, oc), key=lambda z: -z[1])[:4]))
for o in (0, 17, 19, 8, 10):
    print(f"    appearance {o:3d} {names[o][:28]:28s} {np.round(app[o], 2)}")
