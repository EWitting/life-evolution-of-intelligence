"""The reversal over a life: bites per living creature in bins of ticks, for the berry types that turn from poison to
good, from good to poison, and the staple (good throughout), in recorded lives (64 animals per world, the stage's
world scaled down as da_trace.py; WORLDS worlds, default 4), generation 0 of the parent population.
    STAGE=x.reversal DENSE=1 python scripts/probes/reversal_curve.py base td ... [switch_tick=600] [NAME=value]
Also printed: alive creatures per bin, and P(bite) when a bush of each kind is at the mouth before and after the
switch (does anyone try a former poison again?). The curves go to runs/logs/reversal_<variants>.png."""
import os, sys
from dataclasses import replace
import jax, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from life import actions as A
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate, make_layout
from life.world import DIRS
from variants import variant

s = S.STAGES[os.environ.get("STAGE", "x.reversal")]
vs = [a for a in sys.argv[1:] if "=" not in a] or ["base"]
for a in sys.argv[1:]:
    if "=" in a:
        k_, v_ = a.split("=")
        if k_[0].isupper():
            setattr(S, k_, float(v_))
        else:
            s = replace(s, world=replace(s.world, **{k_: type(getattr(s.world, k_))(float(v_))}))
WORLDS, NA, BIN = int(os.environ.get("WORLDS", 4)), 64, 100
f = NA / s.world.num_agents
world = replace(s.world, num_agents=NA, height=round(s.world.height * f ** 0.5), width=round(s.world.width * f ** 0.5),
                patches=round(s.world.patches * f))
SW = world.switch_tick
KINDS = ("poison -> good", "good -> poison", "staple")   # this world has no staple (its line stays 0)
fig, axes = plt.subplots(1, len(vs), figsize=(5 * len(vs), 3.6), sharey=True, squeeze=False)
print(f"switch at tick {SW}, {WORLDS} worlds of {NA}, bites per living creature per {BIN} ticks", flush=True)
for vi, v in enumerate(vs):
    exp = S.make_exp(replace(s, world=world), variant(s.brain, v), "reversal", 1, 0)
    exp = replace(exp, evolution=replace(exp.evolution, siblings=1, record_weights_every=exp.evolution.ticks_per_generation))
    pop = load_population(os.environ.get("RUN") or latest_run(S.STAGES[s.parent].name), exp, seed=0)
    L = make_layout(exp)
    if os.environ.get("DENSE"):
        pl = (np.asarray(L.rule) > 0) & (np.asarray(L.allowed) > 0)
        pop = pop._replace(mask=jax.numpy.where(pl[None], 1.0, pop.mask))
    rs, fn = s.build(exp)
    sim = make_simulate(exp, record=True)
    T = exp.evolution.ticks_per_generation
    nb = T // BIN
    bites, alive_n = np.zeros((3, nb)), np.zeros(nb)
    mouth = {k: [[0, 0, 0.0, 0.0], [0, 0, 0.0, 0.0]] for k in KINDS}   # [before, after] x [ticks, bites, aversion, food]
    for w in range(WORLDS):
        kr, ks = jax.random.split(jax.random.PRNGKey(100 + w))
        rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))                  # [2 phases, ...]
        _, rec = sim(rules, pop, ks)
        rec = {k: np.asarray(rec[k]) for k in ("ate", "alive", "action", "grid", "pos", "dir", "x", "food")}
        avx = rec["x"][..., L.region("valence_av")].astype(np.float32).mean(-1)
        pv, fv = np.asarray(rules.pain_value), np.asarray(rules.food_value)
        kind = -np.ones(pv.shape[1], int)                                           # per object id: index in KINDS
        bush_kind = -np.ones(pv.shape[1], int)
        for t in range(6):
            berry, bush = rs.local(S.variant(t)[S.BERRY]), rs.local(S.variant(t)[S.BUSH])
            k = 0 if pv[0, berry] > 0 and pv[1, berry] == 0 else 1 if pv[0, berry] == 0 and pv[1, berry] > 0 else \
                2 if pv[0, berry] == 0 and pv[1, berry] == 0 and fv[0, berry] > 0 else -1
            kind[berry], bush_kind[bush] = k, k
        alive = rec["alive"].astype(bool)
        ate = rec["ate"].astype(int)
        kk = np.where(ate > 0, kind[ate], -1)
        for b in range(nb):
            sl = slice(b * BIN, (b + 1) * BIN)
            alive_n[b] += alive[sl].sum() / BIN
            for k in range(3):
                bites[k, b] += ((kk[sl] == k) & alive[sl]).sum() / BIN
        # what is at the mouth when the brain acts (the world after the previous step), and whether it then bites
        grid = np.r_[rec["grid"][:1] * 0, rec["grid"][:-1]]
        pos = np.r_[rec["pos"][:1], rec["pos"][:-1]].astype(int); dr = np.r_[rec["dir"][:1], rec["dir"][:-1]].astype(int)
        fr = pos + np.asarray(DIRS)[dr]
        fy, fx = np.clip(fr[..., 0], 0, world.height - 1), np.clip(fr[..., 1], 0, world.width - 1)
        bk = bush_kind[grid[np.arange(T)[:, None], fy, fx]]
        bite = np.isin(rec["action"], [A.USE, A.EAT])
        for k, name in enumerate(KINDS):
            for ph, sel in enumerate((np.arange(T) < SW, np.arange(T) >= SW)):
                m = (bk == k) & alive & sel[:, None]
                mouth[name][ph][0] += m.sum(); mouth[name][ph][1] += (m & bite).sum()
                mouth[name][ph][2] += avx[m].sum(); mouth[name][ph][3] += rec["food"][m].sum()
    rate = bites / np.maximum(alive_n, 1e-9)
    ax = axes[0, vi]
    tt = (np.arange(nb) + 0.5) * BIN
    for k, (name, col) in enumerate(zip(KINDS, ("tab:green", "tab:red", "tab:gray"))):
        ax.plot(tt, rate[k] * BIN, color=col, label=name)
    ax.axvline(SW, color="k", ls=":", lw=1)
    ax2 = ax.twinx(); ax2.plot(tt, alive_n / (WORLDS * NA), color="tab:blue", lw=0.8, ls="--"); ax2.set_ylim(0, 1)
    if vi == len(vs) - 1:
        ax2.set_ylabel("alive (dashed)")
    ax.set_title(v); ax.set_xlabel("tick")
    if vi == 0:
        ax.set_ylabel(f"bites per creature per {BIN} ticks"); ax.legend(fontsize=8)
    print(f"\n{v}")
    for b in range(0, nb, 2):
        if alive_n[b] < 1:
            break
        print(f"  t{b * BIN:5d}  alive {alive_n[b] / (WORLDS * NA):4.0%}  " +
              "  ".join(f"{name} {rate[k, b] * BIN:5.2f}" for k, name in enumerate(KINDS)))
    print("  P(bite) with the bush at the mouth, before / after the switch: " + "  ".join(
        f"{name} {mouth[name][0][1] / max(mouth[name][0][0], 1):.2f} / {mouth[name][1][1] / max(mouth[name][1][0], 1):.2f}"
        for name in KINDS), flush=True)
    print("  aversive cells / food with the bush at the mouth, before / after: " + "  ".join(
        f"{name} {mouth[name][0][2] / max(mouth[name][0][0], 1):.2f} / {mouth[name][1][2] / max(mouth[name][1][0], 1):.2f}"
        f" (food {mouth[name][0][3] / max(mouth[name][0][0], 1):.1f} / {mouth[name][1][3] / max(mouth[name][1][0], 1):.1f})"
        for name in KINDS[:2]), flush=True)
out = f"runs/logs/reversal_{'_'.join(v.replace('+', '-') for v in vs)}.png"
fig.tight_layout(); fig.savefig(out, dpi=110)
print("\n" + out)
