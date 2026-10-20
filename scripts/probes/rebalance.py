"""Generation 0 with start values of the starting population changed (balance, no new mechanism; STAGE_LOG v61-v62):
the learned aversion is real but held about 2 below the aversive cells' threshold by the inherited inhibition from
the eyes' hit / near inputs and by their bias.
    STAGE=x.onet python scripts/probes/rebalance.py [world_field=value ...] base eta4 inh0.5 bias1 eta4+inh0.5 ...
    eta<k>    learning rates of the look and pallium synapses onto valence_av x k
    inh<k>    inherited weights of in -> valence_av from hit, near, agent, wall, held x k
    bias<d>   bias of the valence_av cells + d
    critic0   the critic's value learning off (pallium -> value_app, value_av)
    cut:<projection>   that projection removed (e.g. cut:value_av->valence_av)
    scale:<projection>:<k>   that projection's strength x k
BRAIN=control runs the stage's control brain; RUN=<run dir> another start population.
Every condition in the same worlds as `base` (the stage's start population: its parent's newest run), paired
differences. Per condition: fitness, lifetime, good / poison berries per creature before / after the switch, pain,
and the share of poison among the berries eaten after the switch. A lower-case field=value sets a WorldConfig field
(switch_tick=4000: no reversal within a life; then "before" is the whole life); an upper-case NAME=value a constant
of stages (POISON_FOOD=-9: the food a poison berry costs)."""
import os, sys
import jax, jax.numpy as jnp, numpy as np
from dataclasses import replace
from life.experiments import stages as S
from life.run import latest_run, load_population, make_layout, make_simulate

AV_LEARN = ("in->valence_av[vis*.app*]", "pallium->valence_av")
AV_INH = "in->valence_av[vis*.hit,vis*.near,vis*.agent,vis*.wall,held]"


def rebalance(pop, L, spec: str):
    pid, names = np.asarray(L.proj_id), list(L.proj_names)
    sel = lambda ps: jnp.asarray(np.isin(pid, [names.index(p) for p in ps if p in names]))[None]
    av = L.region("valence_av")
    for part in spec.split("+"):
        if part == "base":
            continue
        if part.startswith("eta"):
            pop = pop._replace(eta=jnp.where(sel(AV_LEARN), pop.eta * float(part[3:]), pop.eta))
        elif part.startswith("inh"):
            pop = pop._replace(w0=jnp.where(sel([AV_INH]), pop.w0 * float(part[3:]), pop.w0))
        elif part == "critic0":   # the critic's value cells learn nothing (its error is then taste - pain)
            pop = pop._replace(eta=jnp.where(sel(("pallium->value_app", "pallium->value_av")), 0.0, pop.eta))
        elif part.startswith("scale:"):   # a projection's strength x k, e.g. scale:ventral_pallidum->dopamine:1.5
            name, k = part[6:].rsplit(":", 1)
            pop = pop._replace(w0=jnp.where(sel([name]), pop.w0 * float(k), pop.w0))
        elif part.startswith("cut:"):   # a projection removed (weights and mask 0), e.g. cut:value_app->valence_app
            m = sel([part[4:]])
            pop = pop._replace(w0=jnp.where(m, 0.0, pop.w0), mask=jnp.where(m, 0.0, pop.mask))
        elif part.startswith("bias"):
            pop = pop._replace(b=pop.b.at[:, av].add(float(part[4:])))
        else:
            raise ValueError(part)
    return pop


if __name__ == "__main__":
    s = S.STAGES[os.environ.get("STAGE", "x.onet")]
    W = int(os.environ.get("WORLDS", 16))
    for a in sys.argv[1:]:
        if "=" in a and a[0].isupper() and a.split("=")[0] != "WORLDS":   # a constant of stages, e.g. POISON_FOOD=-9
            setattr(S, a.split("=")[0], float(a.split("=")[1]))
        if "=" in a and a[0].islower():
            k_, v_ = a.split("=")
            s = replace(s, world=replace(s.world, **{k_: type(getattr(s.world, k_))(float(v_))}))
    print("settings:", [a for a in sys.argv[1:] if "=" in a], flush=True)
    specs = [a for a in sys.argv[1:] if "=" not in a] or ["base"]
    brain = s.control_brain if os.environ.get("BRAIN") == "control" else s.brain   # BRAIN=control: the control's
    exp = S.make_exp(s, brain, "gen0", 1, 0)
    pop0 = load_population(os.environ.get("RUN") or latest_run(S.STAGES[s.parent].name), exp, seed=0)
    L = make_layout(exp)
    rs, fn = s.build(exp)
    sim = jax.jit(make_simulate(exp, record=False))
    berry = [rs.local(S.variant(t)[S.BERRY]) for t in range(6)]
    first = None
    for spec in specs:
        pop = rebalance(pop0, L, spec)
        acc = []
        for k in jax.random.split(jax.random.PRNGKey(31), W):
            kr, ks = jax.random.split(k)
            rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
            pv = np.asarray(rules.pain_value)
            st, _ = sim(rules, pop, ks)
            e = np.asarray(st["eats"]); n = e.shape[0]
            tot = lambda h, bad: sum(e[:, h, berry[t]].sum() for t in range(6) if (pv[h, berry[t]] > 0) == bad) / n
            acc.append([float((s.fitness or S.default_fitness)(st).mean()), float(st["alive_ticks"].mean()),
                        tot(0, False), tot(0, True), tot(1, False), tot(1, True), float(st["pain"].mean())])
        acc = np.array(acc); m = acc.mean(0); se = acc.std(0, ddof=1) / np.sqrt(W)
        first = acc if first is None else first
        dd = acc[:, 0] - first[:, 0]
        paired = f" vs base {dd.mean():+5.0f} +-{dd.std(ddof=1) / np.sqrt(W):3.0f}" if acc is not first else " " * 16
        print(f"{spec:18s} fitness {m[0]:5.0f} +-{se[0]:3.0f}{paired}  lifetime {m[1]:5.0f}  good / poison before "
              f"{m[2]:5.1f} / {m[3]:4.1f}, after {m[4]:5.1f} / {m[5]:4.1f}  poison share before {m[3] / max(m[2] + m[3], 1e-9):.2f}, after {m[5] / max(m[4] + m[5], 1e-9):.2f}"
              f"  pain {m[6]:4.1f}", flush=True)
