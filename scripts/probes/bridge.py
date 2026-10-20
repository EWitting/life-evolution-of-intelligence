"""Bridge between the lab and the world: record lives with the raw inputs every brain received, then replay those
inputs into the same brains (open loop: what a brain learns depends on its inputs, not on its own actions, given
them) with parts of the stream removed, and read the learned weights out as in the lab.
    python scripts/probes/bridge.py record <name> [RUN=<run dir>] [STAGE=2.3] [SIM=0.5] [XOR=1] [PAIN=1] [ONE=0]
                                           [world_field=value ...]
    python scripts/probes/bridge.py replay <name> [conditions ...] [AT=800]
    python scripts/probes/bridge.py arrive <name> [AT=800]
record: the run's last population (default frozen 2.3 seed 0), 64 creatures in the stage's world scaled to them
(as life.book reversal_life), one world, to runs/bridge/<name>.npz. SIM / XOR: the novel looks' similarity to the
original bush and whether they are XOR corners (XOR=0: a random direction per type, so every type is easy to tell
apart by a linear readout of the look). PAIN: the pain of a poison berry (stages.POISON_PAIN); ONE: one berry per
bush worth that many (berry_world one_berry; 0 = six berries).
replay conditions (each also removes what the ones before it in this list remove, except `full`):
    full       the stream as recorded (checks that the replay reproduces the recorded weights)
    noside     the side eyes (vis-60, -30, +30, +60) blank
    bushonly   ... and the eye ahead blank unless the first object in line of sight is a full berry bush
    visits     ... and the eye ahead blank except from 8 ticks before a full bush reaches the mouth to the visit's end
Readout (AT, default the tick of the switch): every creature alive then, a rested brain with the weights of tick AT,
learning off, each novel bush type's look 8 ticks from afar and one at the mouth (labtests test_far / test_ticks);
P(bite) and the input drive of the aversive cells for the types poison before AT and the good ones. The world line
gives what the creatures did: P(bite) per visit on each kind, ticks AT/2 to AT.
arrive: the full replay; on the first tick of every visit to a full bush (ticks AT/2 to AT) the chance of a bite
as the brain had it (it reproduces the world), and again from the same state with one thing changed: `unlearned`
the weights of birth (what learning does to this decision, free of who chose to bite before; `no aversive` /
`no appetitive`: only the synapses onto valence_av and value_av / onto value_app back at birth), `noadapt` the input
adaptation's averages cleared (the look not high-passed by what was in view before). By kind and by painful bites
from that type before (that split selects creatures: those that bite a type collect its lessons)."""
import json, os, sys
from dataclasses import replace
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
from life import actions as A
from life import brain as B
from life.experiments import stages as S
from life.run import SLOT_KEY, load_population, make_layout, make_simulate
from life.world import DIRS

OUT = Path("runs/bridge")
BITE = (A.USE, A.EAT)
SIDE = ("vis-60.", "vis-30.", "vis+30.", "vis+60.")


def setup(meta):
    s = S.STAGES[meta["stage"]]
    sim_, xor, one = float(meta["sim"]), bool(int(meta["xor"])), float(meta.get("one", 0))
    S.POISON_PAIN = float(meta.get("pain", 1.0))
    s = replace(s, build=lambda exp: S.pallium_world(exp, reverse=True, xor=xor, novel_sim=sim_, one_berry=one))
    NA = 64
    f = NA / s.world.num_agents
    world = replace(s.world, num_agents=NA, height=round(s.world.height * f ** 0.5),
                    width=round(s.world.width * f ** 0.5), patches=round(s.world.patches * f),
                    **{k: type(getattr(s.world, k))(float(v)) for k, v in meta["world"].items()})
    run = Path(meta["run"])
    cfg = json.loads((run / "config.json").read_text())
    brain_cfg = s.control_brain if "_control" in run.parent.name else s.brain
    exp = S.make_exp(replace(s, world=world), brain_cfg, "bridge", 1, 0)
    exp = replace(exp, evolution=replace(exp.evolution, siblings=1, record_weights_every=100))
    return s, exp, load_population(run, exp)


def record(name, args):
    kv = dict(a.split("=", 1) for a in args)
    meta = dict(run=kv.pop("RUN", "runs/s2_3_critic/20261019-203352"), stage=kv.pop("STAGE", "2.3"),
                pain=kv.pop("PAIN", "1.0"), one=kv.pop("ONE", "0"),
                sim=kv.pop("SIM", str(S.PALLIUM_SIM)), xor=kv.pop("XOR", "1"), world=kv)
    s, exp, pop = setup(meta)
    rs, fn = s.build(exp)
    kr, ks = jax.random.split(jax.random.PRNGKey(500))
    rules = jax.tree_util.tree_map(lambda a: a[0], fn(0, kr))
    _, rec = make_simulate(exp, record=True, record_obs=True)(rules, pop, ks)
    rec = {k: np.asarray(v) for k, v in rec.items()}
    T, H, W = rec["grid"].shape
    # the object at the mouth when the brain acts at t: the world after step t-1
    g = np.r_[rec["grid"][:1] * 0, rec["grid"][:-1]]
    ps = np.r_[rec["pos"][:1], rec["pos"][:-1]].astype(int)
    ds = np.r_[rec["dir"][:1], rec["dir"][:-1]].astype(int)
    fr = ps + np.asarray(DIRS)[ds]
    front = g[np.arange(T)[:, None], np.clip(fr[..., 0], 0, H - 1), np.clip(fr[..., 1], 0, W - 1)]
    pv = np.asarray(rules.pain_value)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", obs=rec["obs"], sig=rec["sig_in"], ate=rec["ate"], pain=rec["pain"],
                        alive=rec["alive"], action=rec["action"], front=front.astype(np.int16), w_snap=rec["w_snap"],
                        appearance=np.asarray(rules.appearance), pain_value=pv, food_value=np.asarray(rules.food_value))
    (OUT / f"{name}.json").write_text(json.dumps(meta))
    print(f"recorded {OUT / name}.npz: alive at 800 {rec['alive'][799].mean():.0%}, painful bites per creature by 800 "
          f"{((rec['ate'][:800] > 0) & (rec['pain'][:800] > 0.5)).sum() / 64:.1f}")


def kinds(rs, pv, phase):
    """{full bush local id: True (poison) / False (good)} for the novel types in a phase."""
    out = {}
    for t in S.NOVEL:
        bush, berry = rs.local(S.variant(t)[S.BUSH]), rs.local(S.variant(t)[S.BERRY])
        out[bush] = bool(pv[phase, berry] > 0)
    return out


def replay(name, conds, at):
    meta = json.loads((OUT / f"{name}.json").read_text())
    d = dict(np.load(OUT / f"{name}.npz"))
    s, exp, pop = setup(meta)
    rs, _ = s.build(exp)
    L = make_layout(exp)
    names = list(L.in_names)
    K = exp.vision.appearance_dim
    obs0, sig = d["obs"][:at], d["sig"][:at]
    T, N, _ = obs0.shape
    kind = kinds(rs, d["pain_value"], 0)
    bushes = np.array(sorted(kind))
    app = d["appearance"].reshape(-1, *d["appearance"].shape[-2:])[0]   # [M, K] (the same in both phases)
    full_bush = np.array([n.endswith("Wild Gooseberry Bush") and "Empty" not in n for n in rs.names])
    idx = lambda pre: [i for i, n in enumerate(names) if n.startswith(pre)]
    side = [i for p in SIDE for i in idx(p)]
    ahead = idx("vis+0.")
    a_app = [names.index(f"vis+0.app{k}") for k in range(K)]
    a_hit = names.index("vis+0.hit")
    # the first object in line of sight ahead, by its look: the nearest object appearance
    look = obs0[..., a_app]
    dist = ((look[..., None, :] - app[None, None]) ** 2).sum(-1)
    seen = np.where(obs0[..., a_hit] > 0.5, dist.argmin(-1), -1)
    is_bush_ahead = (seen >= 0) & full_bush[np.maximum(seen, 0)]
    # visit windows: 8 ticks before a full bush reaches the mouth to the visit's end
    front = d["front"][:at]
    at_bush = full_bush[front]
    win = np.zeros_like(at_bush)
    for c in range(N):
        on = np.where(at_bush[:, c])[0]
        for t in on:
            win[max(0, t - 8):t + 1, c] = True

    # what the creatures did (ticks at/2 .. at): P(bite) per visit start on each kind
    act, alive = d["action"], d["alive"]
    st = at_bush & np.r_[np.ones((1, N), bool), front[1:] != front[:-1]] & alive[:at]
    st[:at // 2] = False
    bit = np.isin(act[:at], BITE)
    pw = {k: bit[st & np.isin(front, [b for b in bushes if kind[b] == k])].mean() for k in (True, False)}
    print(f"\n{name} ({meta}), readout at tick {at}; creatures alive {alive[at - 1].mean():.0%}")
    print(f"  world: P(bite) on arriving at a bush, ticks {at // 2}-{at}: poison {pw[True]:.2f}, good {pw[False]:.2f}")

    step = jax.jit(jax.vmap(lambda g, st_, o, sg: B.step(exp.brain, L, g, st_, o, sg, jax.random.PRNGKey(0))[0]))
    init = jax.vmap(lambda g, k: B.init_state(g, L, key=k))
    keys = jax.random.split(SLOT_KEY, N)
    gb = B.block_genome(L, pop)
    frozen = pop._replace(eta=jnp.zeros_like(pop.eta))

    def readout(w):
        def o(near, lk):
            x = np.zeros((N, L.n_in), np.float32)
            x[:, names.index("food")] = 0.5
            if "temperature" in names:
                x[:, names.index("temperature")] = 0.5
            if lk is not None:
                x[:, a_hit], x[:, names.index("vis+0.near")] = 1.0, near
                x[:, a_app] = lk
            return jnp.asarray(x)
        res = {}
        z = jnp.zeros((N, 3), jnp.float32)
        for b in bushes:
            s2 = init(frozen, keys)._replace(w=w)
            for _ in range(40):
                s2 = step(frozen, s2, o(0, None), z)
            for _ in range(8):
                s2 = step(frozen, s2, o(0.4, app[b]), z)
            s2 = step(frozen, s2, o(0.8, app[b]), z)
            hall = jax.vmap(lambda x, w_, b_: x @ B.effective(L, w_) + b_)(s2.x, s2.w, frozen.b)
            p = np.asarray(jax.nn.softmax(hall[:, -L.n_out:] * exp.brain.logit_gain / exp.brain.action_temperature, -1))
            av = L.region("valence_av")
            res[b] = (p[:, 4] + p[:, 5], np.asarray(hall)[:, av].mean(-1))
        return res

    living = alive[at - 1]
    for cond in conds:
        o = obs0.copy()
        if cond != "full":
            o[..., side] = 0.0
        if cond in ("bushonly", "visits"):
            o[..., ahead] = np.where(is_bush_ahead[..., None], o[..., ahead], 0.0)
        if cond == "visits":
            o[..., ahead] = np.where(win[..., None], o[..., ahead], 0.0)
        stt = init(pop, keys)
        oj, sj = jnp.asarray(o), jnp.asarray(sig)
        stt, _ = jax.lax.scan(lambda c, t: (step(gb, c, oj[t], sj[t]), None), stt, jnp.arange(T))
        w = stt.w
        if cond == "full":
            rec_w = d["w_snap"][at // 100 - 1].astype(np.float32)
            dif = np.abs(np.asarray(w) - rec_w)[living].max()
            print(f"  replay check: max |w replayed - w recorded| at tick {at} {dif:.4f} (float16 recording)")
        r = readout(w)
        P = {k: np.mean([r[b][0][living] for b in bushes if kind[b] == k]) for k in (True, False)}
        Dv = {k: np.mean([r[b][1][living] for b in bushes if kind[b] == k]) for k in (True, False)}
        print(f"  {cond:9s} lab readout: P(bite) poison {P[True]:.2f} / good {P[False]:.2f};  aversive drive "
              f"{Dv[True]:+.2f} / {Dv[False]:+.2f}", flush=True)


def arrive(name, at):
    meta = json.loads((OUT / f"{name}.json").read_text())
    d = dict(np.load(OUT / f"{name}.npz"))
    s, exp, pop = setup(meta)
    rs, _ = s.build(exp)
    L = make_layout(exp)
    n_in = L.n_in
    obs, sig = jnp.asarray(d["obs"][:at]), jnp.asarray(d["sig"][:at])
    T, N, _ = obs.shape
    kind = kinds(rs, d["pain_value"], 0)
    full_bush = np.array([n.endswith("Wild Gooseberry Bush") and "Empty" not in n for n in rs.names])
    st1 = jax.vmap(lambda g, st_, o, sg: B.step(exp.brain, L, g, st_, o, sg, jax.random.PRNGKey(0), with_logits=True))
    keys = jax.random.split(SLOT_KEY, N)
    gb = B.block_genome(L, pop)
    pb = lambda lg: jax.nn.softmax(lg, -1)[:, 4] + jax.nn.softmax(lg, -1)[:, 5]
    born = jax.vmap(lambda g, k: B.init_state(g, L, key=k))(pop, keys)
    cols = lambda *ns: np.concatenate([np.arange(L.region(n).start, L.region(n).stop) for n in ns if n in L.names])
    av_c, app_c = cols("valence_av", "value_av"), cols("value_app")
    cf = {"unlearned": lambda q: q._replace(w=born.w),
          "no aversive": lambda q: q._replace(w=q.w.at[:, :, av_c].set(born.w[:, :, av_c])),
          "no appetitive": lambda q: q._replace(w=q.w.at[:, :, app_c].set(born.w[:, :, app_c])),
          "noadapt": lambda q: q._replace(tr=q.tr.at[:, :n_in].set(0.0))}

    def body(c, t):
        new, _, lg = st1(gb, c, obs[t], sig[t])
        out = [pb(lg)] + [pb(st1(gb, f(c), obs[t], sig[t])[2]) for f in cf.values()]
        return new, jnp.stack(out)
    _, P = jax.lax.scan(body, born, jnp.arange(T))
    P = np.asarray(P)                                       # T x conditions x N
    front, alive, ate, pain = d["front"][:at], d["alive"][:at], d["ate"][:at].astype(int), d["pain"][:at]
    berry_of = {rs.local(S.variant(t)[S.BUSH]): rs.local(S.variant(t)[S.BERRY]) for t in S.NOVEL}
    labels = ["as in the world"] + list(cf)
    rows = {}
    for c in range(N):
        prior = {}
        for t in range(1, at):
            b = int(front[t, c])
            if ate[t, c] > 0 and pain[t, c] > 0.5:
                prior[int(ate[t, c])] = prior.get(int(ate[t, c]), 0) + 1
            if not (alive[t, c] and full_bush[b] and b in kind and front[t - 1, c] != b) or t < at // 2:
                continue
            k = (kind[b], min(prior.get(berry_of[b], 0), 3))
            rows.setdefault(k, []).append(P[t, :, c])
    print(f"\n{name}: P(bite) on the first tick of a visit, ticks {at // 2}-{at} [visits]")
    print(f"  {'':22s} " + "  ".join(f"{l:>15s}" for l in labels))
    for k in sorted(rows):
        v = np.array(rows[k])
        lab = f"{'poison' if k[0] else 'good'}, {k[1]}{'+' if k[1] == 3 else ''} painful before"
        print(f"  {lab:24s} " + "  ".join(f"{m:15.2f}" for m in v.mean(0)) + f"  [{len(v)}]")
    for kk in (True, False):
        v = np.concatenate([np.array(rows[k]) for k in rows if k[0] == kk])
        print(f"  {'all ' + ('poison' if kk else 'good'):24s} " + "  ".join(f"{m:15.2f}" for m in v.mean(0)) + f"  [{len(v)}]")


if __name__ == "__main__":
    what, name, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    if what == "record":
        record(name, rest)
    elif what == "arrive":
        arrive(name, int(next((a[3:] for a in rest if a.startswith("AT=")), 800)))
    else:
        at = int(next((a[3:] for a in rest if a.startswith("AT=")), 800))
        replay(name, [a for a in rest if "=" not in a] or ["full", "noside", "bushonly", "visits"], at)
