"""Pallium designs side by side (variants.py: pslot, pnt, pk<k>, pn<n>), generation 0 of the parent population:
    STAGE=x.critic DENSE=1 python scripts/probes/pallium_sweep.py pslot+pnt pslot+pnt+pk4 ...
Per variant:
  taste R^2   one recorded life (64 animals, as da_trace.py): how well the pallium of tick t predicts the taste of
              t+1 (ridge, any sign / weights >= 0), per animal: fitted on the first half of its life, tested on the
              second; what a critic reading the pallium could learn in one life
  XOR lab     life.labtests.xor_meals, 2 rounds, with only the pallium synapses learning (the look synapses off):
              aversive activity for the poison and the good types after (does a lesson on one poison reach the
              other poison, and stay off the good ones?), and P(bite)
"""
import os, sys
from dataclasses import replace
import jax, numpy as np
from scipy.optimize import nnls
from life.experiments import stages as S
from life.labtests import xor_meals, XOR_TYPES
from life.run import load_population, latest_run, make_simulate, make_layout
from variants import variant

s = S.STAGES[os.environ.get("STAGE", "x.critic")]
NA = int(os.environ.get("NA", 64))   # animals of the recorded life (fewer for large pallia: the remap of the
                                     # population runs in host memory, about 20 arrays of genomes x n x n)
f = NA / s.world.num_agents
world = replace(s.world, num_agents=NA, height=round(s.world.height * f ** 0.5), width=round(s.world.width * f ** 0.5),
                patches=round(s.world.patches * f))
LOOK, PAL = "in->valence_av[vis*.app*]", "pallium->valence_av"
print(f"{'variant':28s} {'taste R^2 lin / dale':>21s} {'lifetime':>9s}   XOR lab, pallium only: aversion poison / good, P(bite) poison / good")
for v in sys.argv[1:]:
    brain = variant(s.brain, v)
    exp = S.make_exp(replace(s, world=world), brain, "pallium_sweep", 1, 0)
    exp = replace(exp, evolution=replace(exp.evolution, siblings=1,   # one weight snapshot: they grow with the brain squared
                                         record_weights_every=exp.evolution.ticks_per_generation))
    pop = load_population(os.environ.get("RUN") or latest_run(S.STAGES[s.parent].name), exp, seed=0)
    L = make_layout(exp)
    if os.environ.get("DENSE"):
        pl = (np.asarray(L.rule) > 0) & (np.asarray(L.allowed) > 0)
        pop = pop._replace(mask=jax.numpy.where(pl[None], 1.0, pop.mask))
    pop = jax.tree_util.tree_map(lambda a: a[:NA], pop)
    rs, fn = s.build(exp)
    kr, ks = jax.random.split(jax.random.PRNGKey(31))
    rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
    _, rec = make_simulate(exp, record=True)(rules, pop, ks)
    x = np.asarray(rec["x"]).astype(np.float32)
    alive = np.asarray(rec["alive"]).astype(bool)
    i = L.names.index("pallium")
    P = x[..., L.offsets[i]:L.offsets[i] + L.sizes[i]]
    taste = x[..., list(L.in_names).index("taste")]
    # per animal (with per-life wiring every animal has its own code): fitted on the first half of its life, tested
    # on the second half; ridge, so 128+ weights from one life do not overfit
    pred, real, predd = [], [], []
    for a in range(NA):
        ok = np.where(alive[:-1, a] & alive[1:, a])[0]
        if len(ok) < 200:
            continue
        tr, te = ok[:len(ok) // 2], ok[len(ok) // 2:]
        Xtr, Xte, ytr = P[tr, a], P[te, a], taste[tr + 1, a]
        A_ = np.c_[Xtr, np.ones(len(Xtr))]
        w = np.linalg.solve(A_.T @ A_ + 1.0 * np.eye(A_.shape[1]), A_.T @ ytr)
        wd, _ = nnls(np.r_[Xtr, np.eye(Xtr.shape[1])], np.r_[ytr, np.zeros(Xtr.shape[1])])   # ridge-like, weights >= 0
        pred.append(np.c_[Xte, np.ones(len(Xte))] @ w); predd.append(Xte @ wd); real.append(taste[te + 1, a])
    real = np.concatenate(real)
    r2 = lambda pr: 1 - ((real - np.concatenate(pr)) ** 2).mean() / real.var()
    lin, dale = r2(pred), r2(predd)

    del rec, pop   # the recording and the population hold GPU memory that the lab test needs at large pallia
    lab_exp = S.make_exp(replace(s, world=replace(s.world, num_agents=24)), brain, "pallium_sweep", 1, 0)
    lab_exp = replace(lab_exp, evolution=replace(lab_exp.evolution, siblings=1))   # the 24 animals the lab test uses
    lab_pop = load_population(os.environ.get("RUN") or latest_run(S.STAGES[s.parent].name), lab_exp, seed=0)
    res = xor_meals(lab_exp, lab_pop, S.PALLIUM_SIM, {"pallium only": [LOOK]}, rounds=2)
    m = {t: np.mean(res[("pallium only", "after", t)], 0) for t in XOR_TYPES}
    pois, good = [m[t] for t in XOR_TYPES if t.startswith("poison")], [m[t] for t in XOR_TYPES if t.startswith("good")]
    print(f"{v:28s} {lin:10.3f} / {dale:.3f} {alive.sum(0).mean():9.0f}   "
          f"{np.mean([a[0] for a in pois]):.2f} / {np.mean([a[0] for a in good]):.2f}   "
          f"{np.mean([a[1] for a in pois]):.2f} / {np.mean([a[1] for a in good]):.2f}", flush=True)
