"""Acquisition in real lives: what does an animal do in front of a novel bush, as its lessons add up? The newest
population of a stage (or RUN) in its own world, one recorded life per condition (same world and keys), learning
on / with some projections' learning off (OFF, as in gen0.py) / all off.

An encounter: the cell ahead holds a full novel bush and did not on the tick before. For each, the number of
lessons the animal has had so far (poison bites, each followed by pain): of the same poison type, of the other
poison type; the aversive cells' activity and the grasp programme's on the next tick (the brain has then seen the
bush); and whether it bites that bush type within 6 ticks. Rows by lessons of the same type (for a poison bush) or
of both types together (for a good bush).

    STAGE=2.2 [RUN=<run dir>] [PAL_ETA=0.3] [RULE_RESET=1] [OFF='pallium->valence_av;in->valence_av[vis*.app*]'] uv run python scripts/probes/lesson_curve.py"""
import os
import jax, jax.numpy as jnp, numpy as np
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate, make_layout

s = S.STAGES[os.environ.get("STAGE", "2.2")]
exp = S.make_exp(s, s.brain, "lesson_curve", 1, 0)
L = make_layout(exp)
pop = load_population(os.environ.get("RUN") or latest_run(s.name), exp, seed=0)
rs, fn = s.build(exp)
sim = make_simulate(exp, record=True)
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
bush = {v: rs.local(S.variant(v)[S.BUSH]) for v in S.NOVEL}
berry = {v: rs.local(S.variant(v)[S.BERRY]) for v in S.NOVEL}
pid = np.asarray(L.proj_id)
if os.environ.get("PAL_ETA"):   # the pallium synapses' learning rate set to this value in every genome
    _pal = pid == list(L.proj_names).index("pallium->valence_av")
    pop = pop._replace(eta=jnp.where(jnp.asarray(_pal)[None], float(os.environ["PAL_ETA"]), pop.eta))
if os.environ.get("RULE_RESET"):   # the learned projections' rule genes back to the design (A, B, C, D = 0, 1, 0, 0)
    _pl = jnp.asarray((np.asarray(L.rule) > 0) & (pid >= 0))[None]
    pop = pop._replace(A=jnp.where(_pl, 0.0, pop.A), B=jnp.where(_pl, 1.0, pop.B), C=jnp.where(_pl, 0.0, pop.C),
                       D=jnp.where(_pl, 0.0, pop.D))
conds = [("learning on", pop)]
for grp in [g for g in os.environ.get("OFF", "pallium->valence_av").split(";") if g]:
    off = np.isin(pid, [list(L.proj_names).index(n) for n in grp.split(",")])
    conds.append((f"{grp} off", pop._replace(eta=jnp.where(jnp.asarray(off)[None], 0.0, pop.eta))))
conds.append(("learning off", pop._replace(eta=jnp.zeros_like(pop.eta))))
BINS = [(0, 0), (1, 1), (2, 2), (3, 4), (5, 99)]
kr, ks = jax.random.split(jax.random.PRNGKey(int(os.environ.get("KEY", "11"))))
rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
pain = np.asarray(rules.pain_value)
bad = [v for v in S.NOVEL if pain[berry[v]] > 0]; good = [v for v in S.NOVEL if pain[berry[v]] == 0]
print(f"stage {s.key}: poison types {bad}, good types {good}; aversive activity / grasp activity / P(bite within 6 "
      f"ticks) (n encounters)")
for label, g in conds:
    st, rec = sim(rules, g, ks)
    grid = np.asarray(rec["grid"]); pos = np.asarray(rec["pos"]).astype(int); dr = np.asarray(rec["dir"]).astype(int)
    ate = np.asarray(rec["ate"]); alive = np.asarray(rec["alive"]).astype(bool)
    xav = np.asarray(rec["x"][:, :, L.region("valence_av")], np.float32).mean(-1)
    xgr = np.asarray(rec["x"][:, :, L.region("grasp")], np.float32).mean(-1)
    T, N = ate.shape; H, W = grid.shape[1:]
    fr = np.clip(pos + DIRS[dr], 0, [H - 1, W - 1])
    ahead = grid[np.arange(T)[:, None], fr[..., 0], fr[..., 1]]                 # [T, N]
    typ = np.full((T, N), -1)
    for v, b in bush.items():
        typ[ahead == b] = v
    new = (typ >= 0) & np.vstack([np.ones((1, N), bool), typ[1:] != typ[:-1]]) & alive
    les = {v: np.cumsum(np.isin(ate, [berry[v]]), 0) for v in bad}             # poison bites so far, per type
    rows = {k: [] for k in ("poison", "good")}
    for t, i in zip(*np.nonzero(new[:-7])):
        v = typ[t, i]
        bit = np.isin(ate[t + 1:t + 7, i], [berry[v]]).any()
        if v in bad:
            o = [u for u in bad if u != v][0]
            rows["poison"].append((les[v][t, i], les[o][t, i], xav[t + 1, i], xgr[t + 1, i], bit))
        else:
            rows["good"].append((sum(les[u][t, i] for u in bad), 0, xav[t + 1, i], xgr[t + 1, i], bit))
    print(f"\n{label}: fitness {float((s.fitness or S.default_fitness)(st).mean()):.0f}, lifetime {float(st['alive_ticks'].mean()):.0f}")
    for kind, head in (("poison", "lessons of this type"), ("good", "lessons of both types")):
        r = np.array(rows[kind], float)
        line = f"  {kind:6s} bush, by {head:22s}"
        for a, b in BINS:
            m = (r[:, 0] >= a) & (r[:, 0] <= b) if len(r) else np.zeros(0, bool)
            line += f"  {a}{'+' if b == 99 else '' if a == b else '-' + str(b)}: " + (
                f"{r[m, 2].mean():.2f}/{r[m, 3].mean():.2f}/{r[m, 4].mean():.2f} ({m.sum()})" if m.sum() else "-")
        print(line, flush=True)
    r = np.array(rows["poison"], float)
    if len(r):   # the other poison type: lessons of the other type only
        m0 = r[:, 0] == 0
        print("  poison bush with no lesson of its own type, by lessons of the other type: " + "  ".join(
            f"{a}{'+' if b == 99 else '' if a == b else '-' + str(b)}: {r[m, 4].mean():.2f} ({m.sum()})"
            for a, b in BINS for m in [m0 & (r[:, 1] >= a) & (r[:, 1] <= b)] if m.sum()), flush=True)
