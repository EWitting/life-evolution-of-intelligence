"""Record one life of a population on a brain variant, at generation 0 (no evolution), as a run directory that
learned.py, events.py and the dashboard can read: runs/_probe_<variant>/<stage>/.

    STAGE=1.5 RUN=<population run dir> uv run python scripts/probes/record.py <variant> [agents]
variant: see variants.py. agents: size of the recorded world's population (default 64, the world shrinks to match)."""
import os, sys
from dataclasses import replace
import jax, numpy as np
from life.experiments import stages as S
from life.run import load_population, latest_run, make_simulate, make_layout, save_recording, RUNS_DIR
from variants import variant

V = sys.argv[1]; NA = int(sys.argv[2]) if len(sys.argv) > 2 else 64
s = S.STAGES[os.environ.get("STAGE", "1.5")]
f = NA / s.world.num_agents
world = replace(s.world, num_agents=NA, height=round(s.world.height * f ** 0.5), width=round(s.world.width * f ** 0.5))
exp = S.make_exp(replace(s, world=world), variant(s.brain, V), "probe", 1, 0)
exp = replace(exp, evolution=replace(exp.evolution, siblings=1))
pop = load_population(os.environ.get("RUN") or latest_run(S.STAGES[s.parent].name), exp, seed=0)
if os.environ.get("DENSE"):   # every learned synapse exists (inherited ones keep their weight, new ones start at 0)
    from life.run import make_layout as _ml
    _L = _ml(exp); _pl = (np.asarray(_L.rule) > 0) & (np.asarray(_L.allowed) > 0)
    pop = pop._replace(mask=jax.numpy.where(_pl[None], 1.0, pop.mask))
pop = jax.tree_util.tree_map(lambda a: a[:NA], pop)
rs, fn = s.build(exp)
kr, ks = jax.random.split(jax.random.PRNGKey(31))
rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
st, recs = make_simulate(exp, record=True)(rules, pop, ks)
out = RUNS_DIR / (f"_probe_{V}" + ("_dense" if os.environ.get("DENSE") else "")) / s.key
out.mkdir(parents=True, exist_ok=True)
app = rules.appearance if rules.appearance.ndim == 2 else rules.appearance[0]
save_recording(out, exp, rs, make_layout(exp), pop, recs, 0, [], dashboard=False, appearance=app)
np.savez_compressed(out / "population.npz", **{k: np.asarray(v) for k, v in pop._asdict().items()})
print(f"wrote {out}: well-fed lifetime {float(st['fed_meal'].mean()):.0f}, lifetime {float(st['alive_ticks'].mean()):.0f}", flush=True)
import json
(out / "ruleset.json").write_text(json.dumps(rs.to_json_dict()))
(out / "config.json").write_text(exp.to_json())
