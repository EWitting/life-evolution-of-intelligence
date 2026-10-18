"""Why do pallium lessons build up so little in real lives? One recorded life of the newest population of the pallium
stage (or RUN), learning on. For every teaching moment (the pain teacher us_av above 0.1): what is ahead, and the
pallium cells active then (rate > 0.05). For every later encounter (a full novel bush newly ahead, as in
lesson_curve.py): the share of its active pallium cells that were active at an earlier lesson of the same animal,
by what the bush is (the poison type of that lesson, the other poison type, a good type); and the drive of the
learned pallium -> aversive weights onto the aversive cells at encounters, early and late in the life.

    STAGE=2.2 [RUN=<run dir>] [RULE_RESET=1] [LOOK_OFF=1] uv run python scripts/probes/lesson_code.py"""
import os
import jax, numpy as np
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate, make_layout

s = S.STAGES[os.environ.get("STAGE", "2.2")]
exp = S.make_exp(s, s.brain, "lesson_code", 1, 0)
L = make_layout(exp)
pop = load_population(os.environ.get("RUN") or latest_run(s.name), exp, seed=0)
if os.environ.get("RULE_RESET"):   # the learned projections' rule genes back to the design (A, B, C, D = 0, 1, 0, 0)
    import jax.numpy as jnp
    _pl = jnp.asarray((np.asarray(L.rule) > 0) & (np.asarray(L.proj_id) >= 0))[None]
    pop = pop._replace(A=jnp.where(_pl, 0.0, pop.A), B=jnp.where(_pl, 1.0, pop.B), C=jnp.where(_pl, 0.0, pop.C),
                       D=jnp.where(_pl, 0.0, pop.D))
if os.environ.get("LOOK_OFF"):   # no learning on the look synapses
    import jax.numpy as jnp
    _lk = np.asarray(L.proj_id) == list(L.proj_names).index("in->valence_av[vis*.app*]")
    pop = pop._replace(eta=jnp.where(jnp.asarray(_lk)[None], 0.0, pop.eta))
rs, fn = s.build(exp)
sim = make_simulate(exp, record=True)
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
bush = {v: rs.local(S.variant(v)[S.BUSH]) for v in S.NOVEL}
berry = {v: rs.local(S.variant(v)[S.BERRY]) for v in S.NOVEL}
kr, ks = jax.random.split(jax.random.PRNGKey(11))
rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
pain = np.asarray(rules.pain_value)
bad = [v for v in S.NOVEL if pain[berry[v]] > 0]
st, rec = sim(rules, pop, ks)
grid = np.asarray(rec["grid"]); pos = np.asarray(rec["pos"]).astype(int); dr = np.asarray(rec["dir"]).astype(int)
ate = np.asarray(rec["ate"]); alive = np.asarray(rec["alive"]).astype(bool)
pal = L.region("pallium"); av = L.region("valence_av")
xp = np.asarray(rec["x"][:, :, pal], np.float32)
us = np.asarray(rec["mod"], np.float32)[:, :, L.mod_names.index("us_av")]
T, N = ate.shape; H, W = grid.shape[1:]
fr = np.clip(pos + DIRS[dr], 0, [H - 1, W - 1])
ahead = grid[np.arange(T)[:, None], fr[..., 0], fr[..., 1]]
typ = np.full((T, N), -1)
for v, b in bush.items():
    typ[ahead == b] = v
# teaching moments
teach = (us > 0.1) & alive
print(f"stage {s.key}: {teach.sum()} teaching ticks (us_av > 0.1), us_av there {us[teach].mean():.2f}; "
      f"poison bites {sum(np.isin(ate, [berry[v]]).sum() for v in bad)}")
what = typ[teach]
print("  ahead at a teaching tick: " + ", ".join(f"{n} {m:.0%}" for n, m in (
    ("a poison bush", np.isin(what, bad).mean()), ("a good bush", (np.isin(what, [v for v in S.NOVEL if v not in bad])).mean()),
    ("no novel bush", (what < 0).mean()))))
# the type taught: the poison eaten on the tick before
taught = np.full((T, N), -1)
for v in bad:
    taught[1:][np.isin(ate[:-1], [berry[v]])] = v
on = xp > 0.05
new = (typ >= 0) & np.vstack([np.ones((1, N), bool), typ[1:] != typ[:-1]]) & alive
res = {"same poison": [], "other poison": [], "good": []}
for i in range(N):
    lessons = []                       # (tick, type taught, active cells)
    tt = np.nonzero(teach[:, i])[0]
    enc = np.nonzero(new[:-1, i])[0]
    k = 0
    for t in enc:
        while k < len(tt) and tt[k] < t:
            v = taught[tt[k], i] if taught[tt[k], i] >= 0 else (taught[tt[k] - 1, i] if tt[k] > 0 else -1)
            lessons.append((v, on[tt[k], i] | on[tt[k] - 1, i]))
            k += 1
        cells = on[t + 1, i]
        if not lessons or not cells.any():
            continue
        v = typ[t, i]
        for lv, lc in lessons:
            if lv < 0:
                continue
            key = "same poison" if v == lv else "other poison" if v in bad else "good"
            res[key].append((cells & lc).sum() / cells.sum())
print("  share of an encounter's active pallium cells that were active at an earlier lesson (mean over lesson-"
      "encounter pairs): " + ", ".join(f"{k} {np.mean(r):.2f} ({len(r)})" for k, r in res.items()))
print(f"  active pallium cells: at teaching ticks {on[teach].sum(-1).mean():.1f}, at encounters {on[1:][new[:-1]].sum(-1).mean():.1f}")
# learned weights
ws = np.asarray(rec["w_snap"], np.float32)        # [snapshots, N, n, n]
blk = ws[:, :, pal][:, :, :, av]                   # [S, N, pallium, aversive]
print("  learned pallium -> aversive weights (sum over the block, mean over animals alive) by snapshot: " + " ".join(
    f"{blk[j][alive[min(T - 1, (j + 1) * T // len(ws) - 1)]].sum((1, 2)).mean():.2f}" for j in range(0, len(ws), max(1, len(ws) // 8))))
print("  largest single weight at the end: " + f"{blk[-1].max():.2f}; synapses above 0.1: {(blk[-1] > 0.1).sum((1, 2)).mean():.1f} per animal")
# drive of the pallium onto the aversive cells at encounters, by lessons of that type (poison) or of both (good)
snap_of = lambda t: min(len(ws) - 1, t * len(ws) // T)
les = {v: np.cumsum(np.isin(ate, [berry[v]]), 0) for v in bad}
rows = {"poison": [], "good": []}
xin = np.asarray(rec["x"], np.float32)
b_av = np.asarray(pop.b)[:, av]
for t, i in zip(*np.nonzero(new[:-1])):
    v = typ[t, i]
    d_pal = xp[t + 1, i] @ blk[snap_of(t), i]                       # [2] input from the pallium
    d_all = xin[t + 1, i] @ ws[snap_of(t), i][:, av]               # [2] input from everything
    n = les[v][t, i] if v in bad else sum(les[u][t, i] for u in bad)
    rows["poison" if v in bad else "good"].append((n, d_pal.mean(), d_all.mean()))
print(f"  aversive cells' bias {b_av.mean():.2f} (genomes {np.percentile(b_av.mean(1), 10):.2f} to {np.percentile(b_av.mean(1), 90):.2f}); "
      "input to them at encounters, from the pallium / from everything (before the bias):")
for kind, r in rows.items():
    r = np.array(r, float)
    print(f"    {kind:6s}" + "  ".join(f"{a}{'+' if b == 99 else '' if a == b else '-' + str(b)}: {r[m, 1].mean():.2f} / {r[m, 2].mean():.2f}"
                                   for a, b in [(0, 0), (1, 1), (2, 2), (3, 4), (5, 99)] for m in [(r[:, 0] >= a) & (r[:, 0] <= b)] if m.any()))
# where the input to the aversive cells comes from at encounters with a poison bush (no lesson yet / 5+ lessons)
names = list(L.names); offs = list(L.offsets); sizes = list(L.sizes)
inn = list(L.in_names)
groups = {"look (vis*.app*)": [j for j, n_ in enumerate(inn) if ".app" in n_], "other inputs": [j for j, n_ in enumerate(inn) if ".app" not in n_]}
for r_, o_, z_ in zip(names[1:], offs[1:], sizes[1:]):
    groups[r_] = list(range(o_, o_ + z_))
print("  input to the aversive cells by source at encounters with a poison bush, no lesson of its type / 5+:")
cnt = {k: [[], []] for k in groups}
for t, i in zip(*np.nonzero(new[:-1])):
    v = typ[t, i]
    if v not in bad:
        continue
    n = les[v][t, i]
    if n not in (0,) and n < 5:
        continue
    contrib = xin[t + 1, i][:, None] * ws[snap_of(t), i][:, av]
    for k, idx_ in groups.items():
        cnt[k][0 if n == 0 else 1].append(contrib[idx_].sum(0).mean())
print("    " + "  ".join(f"{k} {np.mean(a):+.2f}/{np.mean(b):+.2f}" for k, (a, b) in cnt.items() if abs(np.mean(a)) > 0.02 or abs(np.mean(b)) > 0.02))
oth = groups["other inputs"]; acc_ = []
for t, i in zip(*np.nonzero(new[:-1])):
    if typ[t, i] in bad:
        acc_.append((xin[t + 1, i][oth][:, None] * ws[snap_of(t), i][oth][:, av]).mean(-1))
acc_ = np.mean(acc_, 0); order = np.argsort(acc_)
print("    largest non-look inputs: " + ", ".join(f"{inn[oth[j]]} {acc_[j]:+.2f}" for j in list(order[:6]) + list(order[-3:])))
w0 = np.asarray(pop.w0)[:, :, av].mean((0, 2)); print("    their inherited weights: " + ", ".join(f"{inn[oth[j]]} {w0[oth[j]]:+.2f}" for j in order[:6]))
# the learned part only: (weights now - weights at birth) . code, at encounters, by lessons; and the same with an ideal
# learner that adds the pallium code of the bite tick at every poison bite (no trace, no clipping)
w_birth = (np.asarray(pop.w0) * np.asarray(pop.mask))[:, pal][:, :, av]                # [genomes, pallium, aversive]
sib = N // w_birth.shape[0]
rows = {"poison": [], "good": []}
ideal = np.zeros((N, len(range(*pal.indices(10 ** 6))) if isinstance(pal, slice) else xp.shape[-1]))
bites = {i: [t for t in np.nonzero(np.isin(ate[:, i], [berry[v] for v in bad]))[0]] for i in range(N)}
for t, i in zip(*np.nonzero(new[:-1])):
    v = typ[t, i]
    code = xp[t + 1, i]
    dl = code @ (blk[snap_of(t), i] - w_birth[i // sib])
    past = [b for b in bites[i] if b < t]
    di = code @ xp[past, i].sum(0) if past else 0.0
    n = les[v][t, i] if v in bad else sum(les[u][t, i] for u in bad)
    rows["poison" if v in bad else "good"].append((n, dl.mean(), di))
print("  learned drive (weights now - at birth) / ideal summed overlap with the codes at bite ticks, at encounters:")
for kind, r in rows.items():
    r = np.array(r, float)
    print(f"    {kind:6s}" + "  ".join(f"{a}{'+' if b == 99 else '' if a == b else '-' + str(b)}: {r[m, 1].mean():.2f} / {r[m, 2].mean():.2f}"
                                   for a, b in [(0, 0), (1, 1), (2, 2), (3, 4), (5, 99)] for m in [(r[:, 0] >= a) & (r[:, 0] <= b)] if m.any()))
# the same ideal learner on pallium codes recomputed from the recorded input with other k-WTA sizes (KS=2,4,8)
if os.environ.get("KS"):
    w_in = (np.asarray(pop.w0) * np.asarray(pop.mask))[:, :L.n_in][:, :, pal]               # [genomes, n_in, pallium]
    xi = xin[:, :, :L.n_in]
    for kk in [int(a) for a in os.environ["KS"].split(",")]:
        res_k = {"poison": [], "good": []}
        for i in range(N):
            h = xi[:, i] @ w_in[i // sib]                                                 # the pallium reads this tick's input
            c = np.zeros_like(h); top = np.argsort(-h, -1)[:, :kk]
            np.put_along_axis(c, top, np.maximum(np.tanh(np.take_along_axis(h, top, -1)), 0), -1)
            if kk == 4 and i == 0:
                print(f"  check: recomputed k=4 code against the recorded one, max |difference| {np.abs(c[alive[:, 0]] - xp[alive[:, 0], 0]).max():.3f}")
            past_b = np.array(bites[i], int)
            for t in np.nonzero(new[:-1, i])[0]:
                v = typ[t, i]; pb = past_b[past_b < t]
                if not len(pb):
                    continue
                own = [b for b in pb if v in bad and np.isin(ate[b, i], [berry[v]])]
                n_ = len(own) if v in bad else len(pb)
                res_k["poison" if v in bad else "good"].append((n_, c[t + 1] @ c[pb].sum(0)))
        per = {k_: np.array(r_, float) for k_, r_ in res_k.items()}
        pl = per["poison"]; gd = per["good"]
        pp = (pl[pl[:, 0] > 0, 1] / pl[pl[:, 0] > 0, 0]).mean(); gg = (gd[:, 1] / gd[:, 0]).mean()
        print(f"  k-WTA {kk}: ideal drive per lesson, poison of the lesson's type {pp:.3f}, good {gg:.3f}: good / poison {gg / pp:.2f}")
