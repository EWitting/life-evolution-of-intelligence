"""The assay (assay.py) for the pallium stage, brain only: XOR looks (life.labtests.xor_meals). The parent's final
population on this stage's brain (or RUN, a run of this stage), meals of four novel bushes, poison ++ and --, good
+- and -+; mean aversive activity and P(bite) per type before and after, with the learning of the look synapses
and of the pallium synapses on or off.

    STAGE=2.2 [RUN=<run dir>] [PAL_ETA=0.03] uv run python scripts/probes/assay_xor.py [rounds] [SIM]"""
import os, sys
import jax.numpy as jnp, numpy as np
from life.experiments import stages as S
from life.labtests import xor_meals, XOR_TYPES
from life.run import load_population, latest_run, make_layout

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 2
s = S.STAGES[os.environ.get("STAGE", "2.2")]
SIM = float(sys.argv[2]) if len(sys.argv) > 2 else S.PALLIUM_SIM
exp = S.make_exp(s, s.brain, "assay", 1, 0)
L = make_layout(exp); pid = np.asarray(L.proj_id)
pop = load_population(os.environ.get("RUN") or latest_run(S.STAGES[s.parent].name), exp, seed=0)
PAL, LOOK = "pallium->valence_av", "in->valence_av[vis*.app*]"
if os.environ.get("PAL_ETA"):   # a starting learning rate of the pallium synapses other than the stage's
    pop = pop._replace(eta=jnp.where(jnp.asarray(pid == list(L.proj_names).index(PAL))[None], float(os.environ["PAL_ETA"]), pop.eta))
CONDS = {"both learn": [], "look only": [PAL], "pallium only": [LOOK], "neither": [LOOK, PAL]}
res = xor_meals(exp, pop, SIM, CONDS, rounds=ROUNDS)
eta = np.asarray(pop.eta)
rate = lambda n: eta[:, pid == list(L.proj_names).index(n)].mean()
print(f"stage {s.key}, XOR looks at similarity {SIM}, {ROUNDS} rounds, {len(res[('both learn', 'before', XOR_TYPES[0])])} agents; "
      f"learning rates: look {rate(LOOK):.2f}, pallium {rate(PAL):.2f}")
print(f"  {'':14s}{'':8s}" + "".join(f"{t:>20s}" for t in XOR_TYPES) + "      (aversive activity / P(bite))")
for c in CONDS:
    for w in ("before", "after"):
        print(f"  {c:14s}{w:8s}" + "".join(f"{np.mean(res[(c, w, t)], 0)[0]:12.2f} / {np.mean(res[(c, w, t)], 0)[1]:.2f}"
                                          for t in XOR_TYPES))
