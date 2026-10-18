"""Is the head to head fair? The control population alone, in its own layout and in the main layout with the regions
it lacks silenced (as `stages.versus` runs it), in the same worlds. Equal fitness means the head to head hands the
control its own brain; a gap is a bias of the test.

    [STAGE=2.2] [WORLDS=8] python scripts/probes/versus_check.py"""
import os
import jax, jax.numpy as jnp, numpy as np
from life.config import ExperimentConfig
from life.experiments import stages as S
from life.run import make_layout, make_simulate, load_population

s = S.STAGES[os.environ.get("STAGE", "2.2")]
WORLDS = int(os.environ.get("WORLDS", 8))


def fitness(exp, pop):
    rs, fn = s.build(exp)
    sim = jax.jit(make_simulate(exp, record=False))
    out = []
    for k in jax.random.split(jax.random.PRNGKey(321), WORLDS):
        kr, ks = jax.random.split(k)
        rules = fn(0, kr)
        if rules.food_value.ndim > 1 + (exp.world.switch_tick > 0):
            rules = jax.tree_util.tree_map(lambda a: a[0], rules)
        out.append(float((s.fitness or S.default_fitness)(sim(rules, pop, ks)[0]).mean()))
    return np.array(out)


for main_run, ctrl_run in zip(S.seed_runs(s), S.seed_runs(s, control=True)):
    exp = ExperimentConfig.from_json((main_run / "config.json").read_text())
    own_exp = ExperimentConfig.from_json((ctrl_run / "config.json").read_text())
    layout, old = make_layout(exp), make_layout(own_exp)
    own = load_population(ctrl_run)
    ctrl = load_population(ctrl_run, exp, seed=0)
    for r in [n for n in layout.names if n not in old.names]:
        rows = layout.region(r)
        ctrl = ctrl._replace(w0=ctrl.w0.at[:, rows, :].set(0.0), mask=ctrl.mask.at[:, rows, :].set(0.0),
                             b=ctrl.b.at[:, rows].set(-10.0))
    if float(own.eta.max()) == 0:
        ctrl = ctrl._replace(eta=jnp.zeros_like(ctrl.eta))
    a, b = fitness(own_exp, own), fitness(exp, ctrl)
    d = b - a
    print(f"{ctrl_run.parent.name}: own layout {a.mean():.0f}, main layout silenced {b.mean():.0f}, "
          f"difference {d.mean():+.1f} +-{d.std(ddof=1) / np.sqrt(WORLDS):.1f} (paired, {WORLDS} worlds)", flush=True)
