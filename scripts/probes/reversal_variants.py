"""The reversal-over-a-life figure of the 2.3 book page (life.book reversal_life) for the frozen 2.3 populations, main
and control, in variants of the 2.3 world that they did not evolve in:
    python scripts/probes/reversal_variants.py [every=400] [one=3] [base]
every=N    the poison and good kinds swap at tick N and then every N ticks (WorldConfig.switch_every)
one=F      every bush carries one berry worth F berries of food (default 3), regrowing in the time F berries take
           (the food per bush and tick as in the 2.3 world); a poison berry is still one bite and one pain. A bite
           is a whole visit, so stopping a visit at the first painful berry no longer helps: only learning does
base       the 2.3 world itself
Plots to runs/logs/reversal_variants_<name>.png, numbers printed."""
import json, sys
from dataclasses import replace
from pathlib import Path
import jax, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from life import actions as A
from life import book
from life.experiments import stages as S
from life.run import load_population, make_simulate
from life.world import DIRS

s = S.STAGES["2.3"]
D = json.load(open("book/data/2.3.json"))
MAIN = [Path("runs") / r["dir"] for r in D["runs"]["main"]]
CTRL = [Path("runs") / r["dir"] for r in D["runs"]["control"]]


def figure(name: str, sv, bin_: int):
    m = [book._reversal_life(sv, r, bin_=bin_) for r in MAIN]
    c = [book._reversal_life(sv, r, bin_=bin_) for r in CTRL]
    nb = len(m[0]["rate"][0])
    x = (np.arange(nb) + 0.5) * bin_
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    W = sv.world
    sw = [W.switch_tick + i * W.switch_every for i in range(100)] if W.switch_every else [W.switch_tick]
    print(f"\n{name}  (bites per 100 creatures alive per tick, bins of {bin_})")
    for ax, title, xs in ((axes[0], "Critic (main)", m), (axes[1], "Pain teacher (control)", c)):
        r = np.mean([q["rate"] for q in xs], 0)
        alive = np.mean([q["alive"] for q in xs], 0)
        ax.plot(x, r[0], color="tab:green", label="poison first")
        ax.plot(x, r[1], color="tab:red", label="good first")
        for t in sw:
            if t < x[-1] + bin_ / 2:
                ax.axvline(t, color="k", ls=":", lw=1)
        ax2 = ax.twinx(); ax2.plot(x, alive * 100, color="gray", ls="--", lw=0.8); ax2.set_ylim(0, 100)
        ax.set_title(title); ax.set_xlabel("tick of life")
        pb = np.array([q["p_bite"] for q in xs])                          # seeds x kind x phase
        print(f"  {title:24s} P(bite) at the mouth, phase 1 / phase 2: poison first "
              f"{pb[:, 0, 0].mean():.2f} / {pb[:, 0, 1].mean():.2f}, good first {pb[:, 1, 0].mean():.2f} / "
              f"{pb[:, 1, 1].mean():.2f};  alive at the end {alive[-1]:.0%}")
        print("    " + " ".join(f"{a:4.1f}/{b:4.1f}" for a, b in zip(r[0], r[1])))
    axes[0].set_ylabel("bites per 100 creatures alive, per tick"); axes[0].legend(fontsize=8)
    fig.suptitle(name); fig.tight_layout()
    out = f"runs/logs/reversal_variants_{name.split()[0]}.png"
    fig.savefig(out, dpi=110); print("  " + out, flush=True)


for arg in sys.argv[1:] or ["base"]:
    if arg == "base":
        figure("base (2.3 world)", s, 200)
    elif arg.startswith("every="):
        n = int(arg.split("=")[1])
        figure(f"every{n} (swap at {n} and every {n} ticks)", replace(s, world=replace(s.world, switch_tick=n, switch_every=n)), 100)
    elif arg.startswith("one"):
        F = float(arg.split("=")[1]) if "=" in arg else 3.0
        regrow = S.regrow_ticks

        def one_berry(exp, F=F):
            S.regrow_ticks = lambda exp, berries=None: regrow(exp, F)   # read by berry_world while it builds
            try:
                rs, fn = S.learning_world(   # pallium_world(exp, reverse=True) with one berry
                    exp, springs=S.SPRINGS_17, berries=1, novel_xor=True, reverse=True, novel_sim=S.PALLIUM_SIM)
            finally:
                S.regrow_ticks = regrow
            # a bush with num_uses 1 never takes its last-use transition (OHOL: uses count only above 1), so it
            # would yield forever: every use of a bush takes the last-use transition (berry in the mouth, bush empty)
            good = lambda r: r._replace(food_value=jax.numpy.where(r.food_value > 0, r.food_value * F, r.food_value),
                                        use_table=jax.numpy.where(r.last_use_table >= 0, r.last_use_table, r.use_table))
            return rs, (lambda gen, key: good(fn(gen, key)))
        figure(f"one{F:g} (one berry per bush worth {F:g})", replace(s, build=one_berry), 200)
