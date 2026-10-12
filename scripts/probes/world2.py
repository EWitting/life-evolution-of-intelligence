"""Which world for chapter 2? The frozen affect population (stage 1.8), unchanged, in a variant of its world, intact
and with one circuit silenced at a time: a world for the next stages should have a job for every circuit of
chapter 1 (learning, warmth, the lasting states), not only for the last one.

    uv run python scripts/probes/world2.py [field=value ...]     WorldConfig fields, e.g. patch_purity=0.8
    WORLDS=<n> (default 4), SEED=<frozen seed> (default 0), LESIONS=a,b,... (default: the list below),
    POP=<frozen stage> (default 1.8; e.g. 1.7: the drives population on its own brain, in the same worlds),
    EYES=ch2 (9 columns, range 6 instead of 5 columns, range 5; the four new columns start silent)"""
import os, sys
from dataclasses import replace
import jax, jax.numpy as jnp, numpy as np
from life.experiments import stages as S
from life.freeze import load_frozen
from life.run import make_layout, make_simulate

s = S.STAGES["1.8"]
for a in sys.argv[1:]:
    k_, v_ = a.split("=")
    s = replace(s, world=replace(s.world, **{k_: type(getattr(s.world, k_))(float(v_))}))
WORLDS, SEED, POP = int(os.environ.get("WORLDS", 4)), int(os.environ.get("SEED", 0)), os.environ.get("POP", "1.8")
s = replace(s, brain=S.STAGES[POP].brain)
LESIONS = os.environ.get("LESIONS", "no_plasticity,cold,rest,hungry,flee,fed").split(",")
tag = f"{POP} in " + (" ".join(sys.argv[1:]) or "the 1.8 world")
if os.environ.get("EYES") == "ch2":   # the camera eyes of chapter 2: the five old columns keep their inherited weights
    s = replace(s, vision=S.VISION_CH2)
    tag += " eyes=ch2"

exp = S.make_exp(s, s.brain, "world2_probe", 1, 0)
layout = make_layout(exp)
pop = load_frozen(POP, SEED, exp)
LESIONS = [l for l in LESIONS if l == "no_plasticity" or l in layout.names]
rs, fn = s.build(exp)
sim = jax.jit(make_simulate(exp, record=False))
extra = S.poison_metrics(rs)
base = None
for label in ["intact"] + LESIONS:
    g = pop
    if label == "no_plasticity":
        g = pop._replace(eta=jnp.zeros_like(pop.eta))
    elif label != "intact":
        rows = layout.region(label)
        g = pop._replace(w0=pop.w0.at[:, rows, :].set(0.0), mask=pop.mask.at[:, rows, :].set(0.0),
                         b=pop.b.at[:, rows].set(-10.0))
    acc = []
    for k in jax.random.split(jax.random.PRNGKey(71), WORLDS):
        kr, ks = jax.random.split(k)
        rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
        st, _ = sim(rules, g, ks)
        e = extra(st, None)
        acc.append([float(st["fed_meal"].mean()), float(st["alive_ticks"].mean()), float(e["berries"]),
                    float(e["poison_frac"]), float(st["temp_mean"].mean())])
    acc = np.array(acc); m = acc.mean(0)
    if base is None:
        base = acc[:, 0]
    rel = 100 * acc[:, 0] / base
    print(f"[{tag}] {label:14s} well-fed lifetime {m[0]:4.0f} ({rel.mean():3.0f}% of intact +-{rel.std(ddof=1) / np.sqrt(WORLDS):.0f})  "
          f"lifetime {m[1]:5.0f}  berries {m[2]:5.1f}  poison share {m[3]:.3f}  body temperature {m[4]:.2f}   lifetime per world: " + " ".join(f"{x:.0f}" for x in acc[:, 1]), flush=True)
