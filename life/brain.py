"""Brain substrate (ADR-007, ADR-013): named regions and projections compiled onto one masked weight matrix.

Neuron order in the state vector x[N]: [in (n_in) | user regions in config order | out (n_out)].
W[i, j] is the synapse from neuron i (pre) to neuron j (post). Columns of input neurons are always 0.
Every projection has its own connectivity, learning rule, modulator, eligibility trace, short-term depression
and kind (additive or gain); these are compiled into per-synapse constant arrays in `Layout` (not evolved).
The genome holds the evolvable parts. Rates are non-negative (max(0, tanh)); regions may obey Dale's law
(`sign`), in which case w stores the magnitude and the effective weight is sign * w.
"""
from __future__ import annotations
import fnmatch
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp

from .config import BrainConfig, RegionSpec, ModulatorSpec

RULES = {"fixed": 0, "hebb": 1, "oja": 2, "trace": 3, "delta": 4}
SIGNS = {"mixed": 0.0, "exc": 1.0, "inh": -1.0}
WORLD_SIGNALS = ("reward", "pain", "food")   # order of the world signal vector passed to step()
GAIN_CLIP = 2.0


class Genome(NamedTuple):
    w0: jnp.ndarray    # [N, N] initial weights (magnitudes for Dale-signed rows)
    mask: jnp.ndarray  # [N, N] float 0/1 synapse presence
    b: jnp.ndarray     # [N] bias
    eta: jnp.ndarray   # [N, N] learning rate, constant within a projection (0 = fixed)
    A: jnp.ndarray     # [N, N] rule coefficient A (constant within a projection)
    B: jnp.ndarray
    C: jnp.ndarray
    D: jnp.ndarray


class BrainState(NamedTuple):
    x: jnp.ndarray     # [N] activations (rates, >= 0 except inputs)
    w: jnp.ndarray     # [N, N] current (plastic) weights
    tr: jnp.ndarray    # [N] low-pass activity traces (for the 'trace' rule)
    e: jnp.ndarray     # [N, N] eligibility traces (projections with elig_tau > 0)
    u: jnp.ndarray     # [N, N] short-term depression resource in [0, 1] (1 = fully recovered)
    mod: jnp.ndarray   # [M] modulator values computed at the end of the previous step
    g: jnp.ndarray     # [N] normalisation pool drive of the previous step (regions with norm_lag)


class Layout(NamedTuple):
    """Compiled architecture. Static per experiment; arrays are constants inside jit."""
    n: int
    n_in: int
    n_out: int
    names: tuple            # region names in neuron order, starting with 'in', ending with 'out'
    offsets: tuple          # start index of each region
    sizes: tuple
    in_names: tuple         # name of every input feature (sensors.input_names)
    proj_names: tuple       # "src->dst" per projection, index = proj_id
    mod_names: tuple        # modulator names, index = modulator id
    mod_specs: tuple        # static: (kind, terms, unused, baseline, scale, decay) per modulator
    kwta: tuple             # static: (offset, size, k, phase) per region with k-WTA
    norm: tuple             # static: (offset, size, strength, lag, phase) per region with divisive normalisation
    phases: tuple           # static: per update phase, the indices of its (non-input) neurons; one entry = synchronous
    pl_rows: tuple          # static: pre neurons with any plastic, decaying or depressing synapse
    pl_cols: tuple          # static: their post neurons; plasticity is computed on the block [pl_rows, pl_cols] only
    groups: tuple           # visual group path per region (same order as names), '' = none
    rec_gain: jnp.ndarray   # [M, N] receptor sensitivity: gain effect of modulator m on neuron j
    rec_bias: jnp.ndarray   # [M, N] receptor sensitivity: additive effect
    alpha: jnp.ndarray      # [N]
    trace_tau: jnp.ndarray  # [N]
    sign: jnp.ndarray       # [N] +1 exc, -1 inh, 0 mixed (per presynaptic neuron)
    bias_init: jnp.ndarray  # [N] nan = random
    evolve_b: jnp.ndarray   # [N] 1 = bias mutates
    allowed: jnp.ndarray    # [N, N] 1 where a projection exists
    density: jnp.ndarray    # [N, N] init density of the projection covering the synapse
    one_to_one: jnp.ndarray # [N, N] 1 where the projection is one-to-one and this is the diagonal
    proj_id: jnp.ndarray    # [N, N] int32 projection index, -1 = none
    rule: jnp.ndarray       # [N, N] int32 rule id (RULES)
    mod_idx: jnp.ndarray    # [N, N] int32 modulator index gating plasticity, -1 = none
    elig: jnp.ndarray       # [N, N] eligibility decay (0 = none)
    gain: jnp.ndarray       # [N, N] 1 for gain (multiplicative) synapses
    evolve_w: jnp.ndarray   # [N, N] 1 where w0 and presence may mutate
    tune: jnp.ndarray       # [N, N] 1 on hard-wired synapses whose projection strength may be scaled by evolution
    w_init: jnp.ndarray     # [N, N] fixed initial weight, nan = random
    dep_U: jnp.ndarray      # [N, N] depression use fraction (0 = no depression)
    dep_rec: jnp.ndarray    # [N, N] recovery rate 1/tau_rec
    decay: jnp.ndarray      # [N, N] relaxation of w toward w0 per step
    teacher: jnp.ndarray    # [N] int32 teacher neuron of each post neuron for 'delta', -1 = none
    eta_init: jnp.ndarray   # [N, N]
    abcd_init: jnp.ndarray  # [4, N, N]
    has_gain: bool
    has_elig: bool
    has_dep: bool
    has_decay: bool
    has_receptors: bool

    def region(self, name: str) -> slice:
        i = self.names.index(name)
        return slice(self.offsets[i], self.offsets[i] + self.sizes[i])

    def to_json_dict(self) -> dict:
        return {"names": list(self.names), "offsets": list(self.offsets), "sizes": list(self.sizes),
                "n": self.n, "n_in": self.n_in, "n_out": self.n_out, "in_names": list(self.in_names),
                "proj_names": list(self.proj_names), "mod_names": list(self.mod_names),
                "sign": [float(s) for s in np.asarray(self.sign)[list(self.offsets)]], "groups": list(self.groups)}


def modulator_specs(cfg: BrainConfig) -> tuple:
    mods = tuple(cfg.modulators)
    if cfg.modulator == "reward" and not any(m.name == "reward" for m in mods):
        mods = (ModulatorSpec("reward", "world:reward"),) + mods
    return mods


def build_layout(cfg: BrainConfig, n_in: int, n_out: int, in_names: tuple | None = None) -> Layout:
    regions = [RegionSpec("in", n_in, alpha=1.0)] + list(cfg.regions) \
        + [RegionSpec("out", n_out, alpha=cfg.out_alpha, phase=cfg.out_phase)]
    names = tuple(r.name for r in regions)
    assert len(set(names)) == len(names), f"duplicate region names in {names}"
    sizes = tuple(r.size for r in regions)
    offsets = tuple(int(o) for o in np.cumsum((0,) + sizes[:-1]))
    n = sum(sizes)
    sl = {r.name: slice(o, o + s) for r, o, s in zip(regions, offsets, sizes)}
    per = lambda f, dt=np.float32: np.concatenate([np.full(r.size, f(r), dt) for r in regions])
    alpha, tau = per(lambda r: r.alpha), per(lambda r: r.trace_tau)
    for r in regions:
        assert r.sign in SIGNS, f"region {r.name}: sign must be one of {list(SIGNS)}"
    sign = per(lambda r: SIGNS[r.sign])
    bias_init = per(lambda r: np.nan if r.bias is None else r.bias)
    evolve_b = per(lambda r: float(r.evolve_bias))
    mods = modulator_specs(cfg)
    mod_names = tuple(m.name for m in mods)
    z = lambda dt=np.float32, v=0: np.full((n, n), v, dt)
    allowed, density, o2o, gain, evolve_w, tune = z(), z(), z(), z(), z(), z()
    proj_id, rule, mod_idx = z(np.int32, -1), z(np.int32), z(np.int32, -1)
    elig, dep_U, dep_rec, eta_init, decay = z(), z(), z(), z(), z()
    w_init = z(v=np.nan)
    abcd = np.zeros((4, n, n), np.float32)
    teacher = -np.ones(n, np.int32)
    proj_names = []
    if in_names is None:
        in_names = tuple(f"in{i}" for i in range(n_in))
    assert len(in_names) == n_in
    for pi, p in enumerate(cfg.projections):
        assert p.src in sl and p.dst in sl, f"projection {p.src}->{p.dst} names an unknown region"
        assert p.dst != "in", "nothing may project onto 'in' (inputs are clamped)"
        assert p.rule in RULES, f"unknown rule {p.rule!r}; choose from {list(RULES)}"
        assert p.kind in ("add", "gain") and p.topology in ("full", "one_to_one", "topographic")
        s, d = sl[p.src], sl[p.dst]
        key = f"{p.src}->{p.dst}" + ("(gain)" if p.kind == "gain" else "")
        if p.dst_range:
            r0, r1 = p.dst_range
            assert 0 <= r0 < r1 <= d.stop - d.start, f"projection {key}: dst_range {p.dst_range} outside {p.dst}"
            d = slice(d.start + r0, d.start + r1)
            key += f"<{r0}:{r1}>"
        if p.src_range:
            assert p.src != "in", "src_range does not apply to 'in' (use src_select)"
            r0, r1 = p.src_range
            assert 0 <= r0 < r1 <= s.stop - s.start, f"projection {key}: src_range {p.src_range} outside {p.src}"
            s = slice(s.start + r0, s.start + r1)
            key = key.replace("->", f"<{r0}:{r1}>->", 1)
        if p.src_select:
            assert p.src == "in", "src_select only applies to projections from 'in'"
            rows = [i for i, f in enumerate(in_names) if any(fnmatch.fnmatchcase(f, pat) for pat in p.src_select)]
            assert rows, f"projection {key}: src_select {p.src_select} matches no input feature"
            s = np.array(rows)
            key += "[" + ",".join(p.src_select) + "]"
        assert key not in proj_names, f"duplicate projection {key}"
        assert allowed[s, d].sum() == 0, f"projection {key} overlaps another projection on the same synapses"
        proj_names.append(key)
        if p.topology == "topographic":   # src rows and dst neurons split into p.groups contiguous groups
            ns = len(s) if isinstance(s, np.ndarray) else s.stop - s.start
            nd = d.stop - d.start
            assert p.groups > 0 and ns % p.groups == 0 and nd % p.groups == 0,                 f"topographic projection {key}: sizes {ns}->{nd} must both divide into {p.groups} groups"
            tb = (np.arange(ns)[:, None] * p.groups // ns == np.arange(nd)[None, :] * p.groups // nd).astype(np.float32)
            allowed[s, d] = tb
            density[s, d] = p.density * tb
        else:
            allowed[s, d] = 1.0
            density[s, d] = p.density if p.topology == "full" else 0.0
        if p.topology == "one_to_one":
            ns = len(s) if isinstance(s, np.ndarray) else s.stop - s.start
            assert ns == d.stop - d.start, f"one_to_one projection {key} needs equal sizes"
            o2o[s, d] = np.eye(ns, dtype=np.float32)
        proj_id[s, d] = pi
        rule[s, d] = RULES[p.rule]
        mname = p.modulator or (mod_names[0] if p.modulated and mod_names else "")
        if p.modulated and not mname:
            raise ValueError(f"projection {key} is modulated but BrainConfig has no modulators")
        if mname:
            assert mname in mod_names, f"projection {key}: unknown modulator {mname!r}; have {mod_names}"
            mod_idx[s, d] = mod_names.index(mname)
        elig[s, d] = p.elig_tau
        gain[s, d] = float(p.kind == "gain")
        evolve_w[s, d] = float(p.evolve)
        assert not (p.tune and (p.evolve or p.w_init is None)), f"projection {key}: tune needs evolve=False and w_init"
        tune[s, d] = float(p.tune)
        if p.w_init is not None:
            w_init[s, d] = p.w_init
        if p.depression:
            dep_U[s, d], dep_rec[s, d] = p.depression[0], 1.0 / max(1.0, p.depression[1])
        eta_init[s, d] = p.eta_init
        decay[s, d] = p.decay
        abcd[:, s, d] = np.asarray(p.abcd, np.float32)[:, None, None]
        if p.rule == "delta":
            assert p.teacher in sl and sizes[names.index(p.teacher)] == d.stop - d.start, \
                f"delta projection {key} needs a teacher region of the same size as {p.dst}"
            teacher[d] = np.arange(sl[p.teacher].start, sl[p.teacher].stop)
    none = allowed == 0   # blocks of topographic projections are only partly allowed
    proj_id[none], rule[none], mod_idx[none] = -1, 0, -1
    eta_init *= allowed
    mod_specs = []
    for m in mods:
        if m.source == "region":
            terms = ([(m.pos, 1.0)] if m.pos else []) + ([(m.neg, -1.0)] if m.neg else []) + [tuple(t) for t in m.terms]
            assert terms, f"modulator {m.name}: needs pos, neg or terms"
            for r, _ in terms:
                assert r in sl, f"modulator {m.name}: unknown region {r!r}"
            mod_specs.append(("region", tuple((sl[r].start, sl[r].stop, float(wt)) for r, wt in terms), (0, 0),
                              m.baseline, m.scale, m.decay))
        else:
            kind = m.source.split(":", 1)[1]
            assert m.source.startswith("world:") and kind in WORLD_SIGNALS, f"modulator {m.name}: bad source {m.source}"
            mod_specs.append((kind, (0, 0), (0, 0), m.baseline, m.scale, m.decay))
    kwta = tuple((sl[r.name].start, r.size, r.kwta, r.phase) for r in regions if r.kwta > 0 and r.name != "in")
    phase_ids = sorted({r.phase for r in regions if r.name != "in"})
    phases = tuple(tuple(i for r in regions if r.name != "in" and r.phase == ph
                         for i in range(sl[r.name].start, sl[r.name].stop)) for ph in phase_ids)
    kwta = tuple((o_, s_, k_, phase_ids.index(ph)) for o_, s_, k_, ph in kwta)
    changing = (rule > 0) | (dep_U > 0) | (decay > 0)
    pl_rows = tuple(int(i) for i in np.where(changing.any(axis=1))[0])
    pl_cols = tuple(int(i) for i in np.where(changing.any(axis=0))[0])
    norm = tuple((sl[r.name].start, r.size, float(r.norm), bool(r.norm_lag), phase_ids.index(r.phase)) for r in regions
                 if r.norm > 0 and r.name != "in")
    rec_gain = np.zeros((max(1, len(mod_names)), n), np.float32)
    rec_bias = np.zeros_like(rec_gain)
    for r in regions:
        for mname, effect, sens in r.receptors:
            assert mname in mod_names, f"region {r.name}: receptor for unknown modulator {mname!r}; have {mod_names}"
            assert effect in ("gain", "bias"), f"region {r.name}: receptor effect must be 'gain' or 'bias'"
            (rec_gain if effect == "gain" else rec_bias)[mod_names.index(mname), sl[r.name]] += sens
    j = jnp.asarray
    return Layout(n=n, n_in=n_in, n_out=n_out, names=names, offsets=offsets, sizes=sizes, in_names=tuple(in_names),
                  proj_names=tuple(proj_names), mod_names=mod_names, mod_specs=tuple(mod_specs), kwta=kwta, norm=norm, phases=phases,
                  pl_rows=pl_rows, pl_cols=pl_cols,
                  alpha=j(alpha), trace_tau=j(tau), sign=j(sign), bias_init=j(bias_init), evolve_b=j(evolve_b),
                  allowed=j(allowed), density=j(density), one_to_one=j(o2o), proj_id=j(proj_id), rule=j(rule),
                  mod_idx=j(mod_idx), elig=j(elig), gain=j(gain), evolve_w=j(evolve_w), tune=j(tune * allowed), w_init=j(w_init),
                  dep_U=j(dep_U), dep_rec=j(dep_rec), decay=j(decay), teacher=j(teacher), eta_init=j(eta_init), abcd_init=j(abcd),
                  has_gain=bool(gain.any()), has_elig=bool((elig > 0).any()), has_dep=bool((dep_U > 0).any()),
                  has_decay=bool((decay > 0).any()), groups=tuple(r.group for r in regions),
                  rec_gain=j(rec_gain), rec_bias=j(rec_bias), has_receptors=bool(rec_gain.any() or rec_bias.any()))


def constrain(layout: Layout, w: jnp.ndarray, w_max: float) -> jnp.ndarray:
    """Dale's law: rows of signed (exc/inh) neurons hold non-negative magnitudes."""
    signed = (layout.sign != 0)[:, None]
    return jnp.where(signed, jnp.clip(w, 0.0, w_max), jnp.clip(w, -w_max, w_max))


def effective(layout: Layout, w: jnp.ndarray) -> jnp.ndarray:
    s = layout.sign[:, None]
    return jnp.where(s != 0, s * w, w)


def init_genome(key: jax.Array, layout: Layout) -> Genome:
    n = layout.n
    k1, k2, k3 = jax.random.split(key, 3)
    rand_mask = (jax.random.uniform(k1, (n, n)) < layout.density).astype(jnp.float32)
    mask = jnp.maximum(rand_mask, layout.one_to_one) * layout.allowed   # one_to_one blocks have density 0
    fan_in = jnp.maximum(mask.sum(axis=0, keepdims=True), 1.0)
    w_rand = jax.random.normal(k2, (n, n)) / jnp.sqrt(fan_in)
    w_rand = jnp.where((layout.sign != 0)[:, None], jnp.abs(w_rand), w_rand)
    w0 = jnp.where(jnp.isnan(layout.w_init), w_rand, jnp.nan_to_num(layout.w_init)) * mask
    b = jnp.where(jnp.isnan(layout.bias_init), 0.1 * jax.random.normal(k3, (n,)), jnp.nan_to_num(layout.bias_init))
    return Genome(w0=w0, mask=mask, b=b, eta=layout.eta_init * layout.allowed, A=layout.abcd_init[0],
                  B=layout.abcd_init[1], C=layout.abcd_init[2], D=layout.abcd_init[3])


def init_state(genome: Genome, layout: Layout | None = None, w_max: float = 4.0) -> BrainState:
    n = genome.b.shape[0]
    m = len(layout.mod_names) if layout is not None else 0
    return BrainState(x=jnp.zeros(n, jnp.float32), w=genome.w0 * genome.mask, tr=jnp.zeros(n, jnp.float32),
                      e=jnp.zeros((n, n), jnp.float32), u=jnp.ones((n, n), jnp.float32),
                      mod=jnp.zeros(max(m, 1), jnp.float32), g=jnp.zeros(n, jnp.float32))


def plasticity_rule(rule, A, B, C, D, w, x_pre, x_post, tr_pre, tr_post, target):
    """Rule value per synapse of a block [pre, post] (before learning rate and modulator), selected by `rule`
    (a constant array). Only the rules that occur in the block are computed. `target`: teacher activity per
    post neuron (rule 'delta')."""
    pre, post = x_pre[:, None], x_post[None, :]
    present = set(np.unique(rule).tolist())
    out = jnp.zeros_like(w)
    if 1 in present:
        out = jnp.where(rule == 1, A * pre * post + B * pre + C * post + D, out)
    if 2 in present:
        out = jnp.where(rule == 2, A * post * (pre - post * w), out)
    if 3 in present:
        out = jnp.where(rule == 3, A * tr_pre[:, None] * post - C * pre * tr_post[None, :], out)
    if 4 in present:
        out = jnp.where(rule == 4, A * pre * (target - x_post)[None, :], out)
    return out


def compute_modulators(layout: Layout, x: jnp.ndarray, world_sig: jnp.ndarray, prev: jnp.ndarray) -> jnp.ndarray:
    out = []
    for i, (kind, terms, _, base, scale, decay) in enumerate(layout.mod_specs):
        if kind == "region":
            v = sum(wt * x[a:b].mean() for a, b, wt in terms)
        else:
            v = world_sig[WORLD_SIGNALS.index(kind)]
        v = scale * (v - base)
        out.append(jnp.maximum(v, decay * prev[i]) if decay > 0 else v)
    return jnp.stack(out) if out else jnp.zeros(1, jnp.float32)


def _kwta(layout: Layout, f: jnp.ndarray, phase: int | None = None) -> jnp.ndarray:
    for off, size, k, ph in layout.kwta:
        if phase is not None and ph != phase:
            continue
        seg = f[off:off + size]
        thr = jnp.sort(seg)[size - min(k, size)]
        f = f.at[off:off + size].set(jnp.where(seg >= thr, seg, 0.0))
    return f


def _normalise(layout: Layout, h: jnp.ndarray, g: jnp.ndarray, phase: int | None = None):
    """Divisive normalisation per region. Returns (h, g) with g the pool drive of this step."""
    for off, size, k, lag, ph in layout.norm:
        if phase is not None and ph != phase:
            continue
        seg = h[off:off + size]
        pool = jnp.maximum(seg, 0.0).mean()
        div = g[off] if lag else pool
        h = h.at[off:off + size].set(seg / (1.0 + k * div))
        g = g.at[off:off + size].set(pool)
    return h, g


def step(cfg: BrainConfig, layout: Layout, genome: Genome, state: BrainState, obs: jnp.ndarray,
         world_sig: jnp.ndarray, key: jax.Array):
    """One world tick of brain activity. `obs` is the flattened observation [n_in]; `world_sig` the vector of
    world signals (WORLD_SIGNALS) of the last tick, used only by 'world:*' modulators. Returns (state, action)."""
    n_in, n_out = layout.n_in, layout.n_out
    x = state.x.at[:n_in].set(obs)
    w, tr, e, u, mod, g = state.w, state.tr, state.e, state.u, state.mod, state.g
    plastic = bool(layout.pl_rows)
    if plastic:   # everything that changes within a life sits in the block [R, C] (usually a small part of w)
        R, C = np.asarray(layout.pl_rows), np.asarray(layout.pl_cols)
        ix = np.ix_(R, C)
        b_rule, b_elig, b_decay, b_mod = (np.asarray(a)[ix] for a in (layout.rule, layout.elig, layout.decay, layout.mod_idx))
        b_depU, b_deprec = np.asarray(layout.dep_U)[ix], np.asarray(layout.dep_rec)[ix]
        b_gate = jnp.where(b_mod >= 0, mod[np.maximum(b_mod, 0)], 1.0)
        b_teacher = np.asarray(layout.teacher)[C]
        b_signed = (np.asarray(layout.sign)[R] != 0)[:, None]
        b_eta, b_mask, b_w0 = genome.eta[ix], genome.mask[ix], genome.w0[ix]
        b_A, b_B, b_C, b_D = genome.A[ix], genome.B[ix], genome.C[ix], genome.D[ix]
    h = None
    for _ in range(cfg.steps_per_tick):
        w_eff = effective(layout, w) * (u if layout.has_dep else 1.0)
        if len(layout.phases) <= 1:   # synchronous: every neuron reads the previous tick
            h = x @ (w_eff * (1.0 - layout.gain) if layout.has_gain else w_eff) + genome.b
            log_gain = 0.0
            if layout.has_gain:
                log_gain = x @ (w_eff * layout.gain)
            if layout.has_receptors:   # broadcast neuromodulation via receptors of the target neurons
                h = h + mod @ layout.rec_bias
                log_gain = log_gain + mod @ layout.rec_gain
            if layout.has_gain or layout.has_receptors:
                h = h * jnp.exp(jnp.clip(log_gain, -GAIN_CLIP, GAIN_CLIP))
            if layout.norm:
                h, g = _normalise(layout, h, g)
            f = _kwta(layout, jnp.maximum(jnp.tanh(h), 0.0))
            x_new = ((1.0 - layout.alpha) * x + layout.alpha * f).at[:n_in].set(obs)
        else:   # ordered: a phase reads this tick's activity of the phases before it
            w_add = w_eff * (1.0 - layout.gain) if layout.has_gain else w_eff
            xc, h = x, jnp.zeros_like(x)
            for pi, cols in enumerate(layout.phases):
                cols = np.asarray(cols)
                hk = xc @ w_add[:, cols] + genome.b[cols]
                lg = 0.0
                if layout.has_gain:
                    lg = xc @ (w_eff * layout.gain)[:, cols]
                if layout.has_receptors:
                    hk = hk + mod @ layout.rec_bias[:, cols]
                    lg = lg + mod @ layout.rec_gain[:, cols]
                if layout.has_gain or layout.has_receptors:
                    hk = hk * jnp.exp(jnp.clip(lg, -GAIN_CLIP, GAIN_CLIP))
                h = h.at[cols].set(hk)
                if layout.norm:
                    h, g = _normalise(layout, h, g, pi)
                f = _kwta(layout, jnp.maximum(jnp.tanh(h), 0.0), pi)
                xc = xc.at[cols].set((1.0 - layout.alpha[cols]) * x[cols] + layout.alpha[cols] * f[cols])
            x_new = xc.at[:n_in].set(obs)
        if plastic:
            wb = w[ix]
            target = jnp.where(b_teacher >= 0, x_new[np.maximum(b_teacher, 0)], x_new[C])
            base = plasticity_rule(b_rule, b_A, b_B, b_C, b_D, wb, x[R], x_new[C], tr[R], tr[C], target)
            if layout.has_elig:
                eb = jnp.where(b_elig > 0, b_elig * e[ix] + base, 0.0)
                e = e.at[ix].set(eb)
                base = jnp.where(b_elig > 0, eb, base)
            wb = wb + b_eta * b_gate * base * b_mask
            if layout.has_decay:
                wb = wb + b_decay * (b_w0 * b_mask - wb)
            wb = jnp.where(b_signed, jnp.clip(wb, 0.0, cfg.w_max), jnp.clip(wb, -cfg.w_max, cfg.w_max))
            w = w.at[ix].set(wb)
            if layout.has_dep:
                ub = u[ix]
                u = u.at[ix].set(jnp.clip(ub + b_deprec * (1.0 - ub) - b_depU * ub * jnp.maximum(x[R], 0.0)[:, None], 0.0, 1.0))
        tr = layout.trace_tau * tr + (1.0 - layout.trace_tau) * x_new
        x = x_new
    mod = compute_modulators(layout, x, world_sig, mod)
    logits = h[-n_out:] * cfg.logit_gain
    if cfg.action_temperature > 0:
        action = jax.random.categorical(key, logits / cfg.action_temperature)
    else:
        action = jnp.argmax(logits)
    return BrainState(x=x, w=w, tr=tr, e=e, u=u, mod=mod, g=g), action.astype(jnp.int32)


# ---------------------------------------------------------------- warm starts across layouts

def remap_genomes(pop: Genome, old: Layout, new: Layout, key: jax.Array) -> Genome:
    """Carry a population to a new layout (ADR-012 'build on top'). Neurons are matched by region name (and
    input features by name); new neurons and new projections come from a fresh init, but every *new* evolvable
    synapse starts at w0 = 0 so inherited behaviour is unchanged until evolution uses it ('start silent').
    Plasticity genes are copied for projections that exist in both layouts with the same rule."""
    idx_old, idx_new = [], []
    for name, off, size in zip(new.names, new.offsets, new.sizes):
        if name == "in":
            pos = {f: i for i, f in enumerate(old.in_names)}
            for i, f in enumerate(new.in_names):
                if f in pos:
                    idx_new.append(i); idx_old.append(pos[f])
        elif name in old.names:
            k = old.names.index(name)
            m = min(size, old.sizes[k])
            idx_new += list(range(off, off + m)); idx_old += list(range(old.offsets[k], old.offsets[k] + m))
    io, inn = np.array(idx_old, np.int64), np.array(idx_new, np.int64)
    P = pop.b.shape[0]
    fresh = jax.vmap(lambda k: init_genome(k, new))(jax.random.split(key, P))
    old_allowed, new_allowed = np.asarray(old.allowed), np.asarray(new.allowed)
    same_rule = np.zeros((new.n, new.n), bool)
    same_rule[np.ix_(inn, inn)] = (np.asarray(old.rule)[np.ix_(io, io)] == np.asarray(new.rule)[np.ix_(inn, inn)]) \
        & (old_allowed[np.ix_(io, io)] > 0)
    carried = np.zeros((new.n, new.n), bool)
    carried[np.ix_(inn, inn)] = old_allowed[np.ix_(io, io)] > 0
    carried &= new_allowed > 0
    hard = ~np.isnan(np.asarray(new.w_init)) | (np.asarray(new.evolve_w) == 0)

    def mat(fresh_a, old_a, where):
        out = np.array(fresh_a)
        sub = np.asarray(old_a)[:, io][:, :, io]
        blk = out[:, inn][:, :, inn]
        wmask = where[np.ix_(inn, inn)][None]
        blk = np.where(wmask, sub, blk)
        out[np.ix_(np.arange(P), inn, inn)] = blk
        return out

    carry_w = carried & ~hard
    w0 = mat(fresh.w0, pop.w0, carry_w)
    mask = mat(fresh.mask, pop.mask, carry_w)
    silent = (new_allowed > 0) & ~carried & ~hard
    w0 = np.where(silent[None], 0.0, w0)
    tuned = np.zeros((new.n, new.n), bool)   # hard-wired strengths tuned by evolution carry over
    tuned[np.ix_(inn, inn)] = (np.asarray(old.tune)[np.ix_(io, io)] > 0) & (np.asarray(new.tune)[np.ix_(inn, inn)] > 0)
    w0 = mat(w0, pop.w0, tuned & carried)
    w0 = np.asarray(constrain(new, jnp.asarray(w0), 1e9))
    genes = {k: mat(getattr(fresh, k), getattr(pop, k), same_rule & carried) for k in ("eta", "A", "B", "C", "D")}
    b = np.array(fresh.b)
    keep_b = np.asarray(new.evolve_b)[inn] > 0
    b[:, inn[keep_b]] = np.asarray(pop.b)[:, io[keep_b]]
    return Genome(w0=jnp.asarray(w0 * mask), mask=jnp.asarray(mask), b=jnp.asarray(b), **{k: jnp.asarray(v) for k, v in genes.items()})
