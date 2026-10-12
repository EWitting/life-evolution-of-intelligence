"""Is a generation's selection a lottery of inborn taste? In a learning world the looks and the poison are drawn anew
every life. A genome whose inherited look weights happen to dislike this world's poison types does well without
learning anything, and the next world draws again. A frozen population (64 genomes x 4 siblings) in several worlds:

  luck       per genome and world: the inborn drive of the poison bushes' looks onto the aversive cells minus that of
             the good bushes' looks, minus the same for the appetitive cells (positive = born avoiding this world's
             poison). From the inherited weights of all eye columns; no learning, no simulation.
  fitness    the genome's well-fed lifetime in that world (mean of its four siblings).

Reported: how much of a world's fitness ranking the luck explains (correlation over genomes, per world), and how
much of the ranking carries over from one world to the next (correlation of genome fitness between worlds): what
selection can keep.

    uv run python scripts/probes/luck.py <frozen stage key> [field=value ...]     WorldConfig fields, e.g. start_food=1
    WORLDS=<n> (default 8), SEED=<frozen seed>, LEARN=on (skip the half with learning switched off)"""
import os, sys
from dataclasses import replace
import jax, jax.numpy as jnp, numpy as np
from life.experiments import stages as S
from life.freeze import load_frozen
from life.run import make_layout, make_simulate

s = S.STAGES[sys.argv[1]]
for a in sys.argv[2:]:
    k_, v_ = a.split("=")
    s = replace(s, world=replace(s.world, **{k_: type(getattr(s.world, k_))(float(v_))}))
WORLDS, SEED = int(os.environ.get("WORLDS", 8)), int(os.environ.get("SEED", 0))
exp = S.make_exp(s, s.brain, "luck_probe", 1, 0)
lay = make_layout(exp)
pop = load_frozen(s.key, SEED, exp)
rs, fn = s.build(exp)
sim = make_simulate(exp, record=False)
P, sib = pop.b.shape[0], exp.evolution.siblings
names, off = list(lay.in_names), lay.offsets[lay.names.index("in")]
K = exp.vision.appearance_dim
cols = sorted({n.split(".")[0] for n in names if n.startswith("vis")})
rows = np.array([[off + names.index(f"{c}.app{k}") for k in range(K)] for c in cols])        # [columns, K]
w = np.asarray(pop.w0) * np.asarray(pop.mask)


def cells(region):
    r = lay.region(region)
    return np.arange(r.start, r.stop) if isinstance(r, slice) else np.asarray(r)


look_w = {v: w[:, rows][:, :, :, cells(v)].mean(1).sum(-1) for v in S.VAL}                    # [P, K]: mean over columns
bush = np.array([rs.local(S.variant(v)[S.BUSH]) for v in S.NOVEL])
berry = np.array([rs.local(S.variant(v)[S.BERRY]) for v in S.NOVEL])
for label, g in (("learning on", pop), ("learning off", pop._replace(eta=jnp.zeros_like(pop.eta))))[:1 if os.environ.get("LEARN") == "on" else 2]:
    fits, lucks, bites, lifes = [], [], [], []
    for k in jax.random.split(jax.random.PRNGKey(123), WORLDS):
        kr, ks = jax.random.split(k)
        rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
        bad = np.asarray(rules.pain_value)[berry] > 0
        app = np.asarray(rules.appearance)[bush]                                               # [types, K]
        drive = {v: look_w[v] @ app.T for v in S.VAL}                                          # [P, types]
        net = drive["valence_av"] - drive["valence_app"]
        lucks.append(net[:, bad].mean(1) - net[:, ~bad].mean(1))
        st, _ = sim(rules, g, ks)
        fits.append(np.asarray(st["fed_meal"]).reshape(P, sib).mean(1))
        lifes.append(np.asarray(st["alive_ticks"]))
        bites.append(np.asarray(st["eats"])[:, :, 0].sum(1).reshape(P, sib).mean(1))
    fits, lucks, bites = np.array(fits), np.array(lucks), np.array(bites)
    r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    r_luck = [r(lucks[i], fits[i]) for i in range(WORLDS)]
    r_bite = [r(lucks[i], bites[i]) for i in range(WORLDS)]
    between = [r(fits[i], fits[j]) for i in range(WORLDS) for j in range(i + 1, WORLDS)]
    r_luck_between = [r(lucks[i], lucks[j]) for i in range(WORLDS) for j in range(i + 1, WORLDS)]
    # what a round of selection in one world is worth in the next: the best quarter of world i, scored in world i+1
    gain = [(fits[j][np.argsort(-fits[i])[:P // 4]].mean() - fits[j].mean()) / fits[j].mean()
            for i in range(WORLDS) for j in range(WORLDS) if i != j]
    own = [(fits[i][np.argsort(-fits[i])[:P // 4]].mean() - fits[i].mean()) / fits[i].mean() for i in range(WORLDS)]
    life = np.array(lifes)
    print(f"[{s.key} seed {SEED}, {' '.join(sys.argv[2:]) or 'as frozen'}, {label}] lifetime median {np.median(life):.0f}, dead before tick 400 {(life < 400).mean():.0%}, at the cap {(life >= exp.evolution.ticks_per_generation).mean():.0%}; mean fitness {fits.mean():.0f}; per world: " + " ".join(f"{x:.0f}" for x in fits.mean(1)))
    print(f"   luck and fitness, correlation over the {P} genomes within a world: {np.mean(r_luck):+.2f} (per world " + " ".join(f"{x:+.2f}" for x in r_luck) + ")")
    print(f"   luck and poison bites: {np.mean(r_bite):+.2f};  a genome's luck in one world and in another: {np.mean(r_luck_between):+.2f}")
    print(f"   a genome's fitness in one world and in another: {np.mean(between):+.2f} (range {min(between):+.2f} to {max(between):+.2f})")
    print(f"   the best quarter of a world scores {np.mean(own):+.0%} above the mean in that world and {np.mean(gain):+.0%} in another world", flush=True)
