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
    A: jnp.ndarray     # [N, N] rule coefficients A-D (constant within a projection; ProjectionSpec.abcd at birth of
    B: jnp.ndarray     # a lineage, then inherited and mutated). What they weigh depends on the rule, see
    C: jnp.ndarray     # plasticity_rule; for 'hebb': A pre*post + B pre + C post + D, so (0, 1, 0, 0) is
    D: jnp.ndarray     # "dW ~ pre", the three-factor rule of every modulated projection (dW = eta * mod * trace(pre))
    clear: jnp.ndarray  # [M] per modulator: the fraction of it left after a tick (ModulatorSpec.decay; evolvable when > 0)
    adapt: jnp.ndarray  # [1] adaptation strength of the adapting inputs (BrainConfig.in_adapt_strength; evolvable with
                        # BrainConfig.in_adapt_evolve)


class BrainState(NamedTuple):
    x: jnp.ndarray     # [N] activations (rates, >= 0 except inputs)
    w: jnp.ndarray     # [N, N] current (plastic) weights
    tr: jnp.ndarray    # [N] low-pass activity traces (for the 'trace' rule)
    e: jnp.ndarray     # [pl_rows, pl_cols] eligibility traces of the plastic block (projections with elig_tau > 0)
    u: jnp.ndarray     # [pl_rows, pl_cols] short-term depression resource in [0, 1] (1 = fully recovered); only the
                       # plastic block can depress, so the rest stays at 1 and is not stored
    mod: jnp.ndarray   # [M] modulator values computed at the end of the previous step
    g: jnp.ndarray     # [N] normalisation pool drive of the previous step (regions with norm_lag)
    ws: tuple = ()     # per update phase [N, phase neurons]: the weights that never change within a life (everything
                       # outside the plastic block), signed and without the gain part; computed once at birth
    wg: tuple = ()     # the same for the gain part (projections with gain)
    xd: tuple | jnp.ndarray = ()   # [N] activity of the step before x (only with ProjectionSpec.elig_delay)
    wl: tuple = ()     # per update phase: the fixed weights of slow synapses (ProjectionSpec.lag), applied to the
                       # activity of the previous tick


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
    mod_decay: tuple        # static: the designed decay per modulator (the genome carries the evolved one, Genome.clear)
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
    centred: jnp.ndarray    # [N, N] 1 where the rule's presynaptic term is activity minus the neuron's slow average
    in_adapt: jnp.ndarray   # [n_in] 1 for adapting inputs (BrainConfig.in_adapt), else 0; the strength is Genome.adapt
    has_adapt: bool
    adapt_design: float     # static: BrainConfig.in_adapt_strength
    adapt_evolve: bool      # static: BrainConfig.in_adapt_evolve
    has_gain: bool
    has_elig: bool
    has_dep: bool
    has_decay: bool
    has_receptors: bool
    mod_accum: tuple = ()   # static: ModulatorSpec.accumulate per modulator (empty = none accumulates)
    elig_delay: jnp.ndarray | None = None   # [N, N] 1 where the eligibility trace takes the pre activity of a tick earlier
    has_delay: bool = False
    per_slot: jnp.ndarray | None = None     # [N, N] 1 where synapses belong to the population slot (ProjectionSpec.per_slot)
    has_per_slot: bool = False
    lag: jnp.ndarray | None = None          # [N, N] 1 for slow synapses (ProjectionSpec.lag)
    has_lag: bool = False

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
    regions = [RegionSpec("in", n_in, alpha=1.0, trace_tau=cfg.in_trace_tau)] + list(cfg.regions) \
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
    centred, delay, per_slot, lag = z(), z(), z(), z()
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
        centred[s, d] = float(p.centred)
        assert not p.elig_delay or p.elig_tau > 0, f"projection {key}: elig_delay needs an eligibility trace"
        delay[s, d] = float(p.elig_delay)
        assert not p.per_slot or (not p.evolve and p.w_init is None), f"projection {key}: per_slot needs evolve=False, no w_init"
        per_slot[s, d] = float(p.per_slot)
        assert not p.lag or p.rule == "fixed", f"projection {key}: a slow synapse (lag) cannot be plastic"
        lag[s, d] = float(p.lag)
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
                  proj_names=tuple(proj_names), mod_names=mod_names, mod_specs=tuple(mod_specs),
                  mod_decay=tuple(float(m.decay) for m in mods) or (0.0,), kwta=kwta, norm=norm, phases=phases,
                  pl_rows=pl_rows, pl_cols=pl_cols,
                  alpha=j(alpha), trace_tau=j(tau), sign=j(sign), bias_init=j(bias_init), evolve_b=j(evolve_b),
                  allowed=j(allowed), density=j(density), one_to_one=j(o2o), proj_id=j(proj_id), rule=j(rule),
                  mod_idx=j(mod_idx), elig=j(elig), gain=j(gain), evolve_w=j(evolve_w), tune=j(tune * allowed), w_init=j(w_init),
                  dep_U=j(dep_U), dep_rec=j(dep_rec), decay=j(decay), teacher=j(teacher), eta_init=j(eta_init), abcd_init=j(abcd),
                  centred=j(centred),
                  in_adapt=j(np.array([float(any(fnmatch.fnmatch(nm, pat) for pat in cfg.in_adapt)) for nm in in_names], np.float32)),
                  has_adapt=bool(cfg.in_adapt), adapt_design=float(cfg.in_adapt_strength),
                  adapt_evolve=bool(cfg.in_adapt and cfg.in_adapt_evolve),
                  has_gain=bool(gain.any()), has_elig=bool((elig > 0).any()), has_dep=bool((dep_U > 0).any()),
                  has_decay=bool((decay > 0).any()), groups=tuple(r.group for r in regions),
                  rec_gain=j(rec_gain), rec_bias=j(rec_bias), has_receptors=bool(rec_gain.any() or rec_bias.any()),
                  mod_accum=tuple(float(m.accumulate) for m in mods) if any(m.accumulate for m in mods) else (),
                  elig_delay=j(delay), has_delay=bool(delay.any()),
                  per_slot=j(per_slot * allowed), has_per_slot=bool((per_slot * allowed).any()),
                  lag=j(lag * allowed), has_lag=bool((lag * allowed).any()))


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
                  B=layout.abcd_init[1], C=layout.abcd_init[2], D=layout.abcd_init[3],
                  clear=jnp.asarray(layout.mod_decay, jnp.float32), adapt=jnp.full((1,), layout.adapt_design, jnp.float32))


def genome_from_npz(f, layout: Layout) -> Genome:
    """A stored population. Files written before Genome.clear existed get the designed decay of every modulator, those
    before Genome.adapt the designed adaptation strength."""
    g = {k: jnp.asarray(f[k]) for k in Genome._fields if k in f}
    if "clear" not in g:
        g["clear"] = jnp.tile(jnp.asarray(layout.mod_decay, jnp.float32), (g["b"].shape[0], 1))
    if "adapt" not in g:
        g["adapt"] = jnp.full((g["b"].shape[0], 1), layout.adapt_design, jnp.float32)
    return Genome(**g)


def init_state(genome: Genome, layout: Layout, w_max: float = 4.0, key: jax.Array | None = None) -> BrainState:
    """The brain at birth. With `key` (the population slot's, run.SLOT_KEY), the per-slot synapses (Layout.per_slot)
    are drawn from it, as init_genome draws random ones; without it they are the genome's."""
    n = genome.b.shape[0]
    m = len(layout.mod_names)
    w = genome.w0 * genome.mask
    if layout.has_per_slot and key is not None:
        k1, k2 = jax.random.split(key)
        pl = layout.per_slot > 0
        mask = jnp.where(pl, (jax.random.uniform(k1, (n, n)) < layout.density).astype(jnp.float32) * layout.allowed,
                         genome.mask)
        fan_in = jnp.maximum(mask.sum(axis=0, keepdims=True), 1.0)
        w_rand = jax.random.normal(k2, (n, n)) / jnp.sqrt(fan_in)
        w_rand = jnp.where((layout.sign != 0)[:, None], jnp.abs(w_rand), w_rand)
        w = jnp.where(pl, w_rand * mask, w)
    ws, wg, wl = _fixed_weights(layout, w)
    return BrainState(x=jnp.zeros(n, jnp.float32), w=w, tr=jnp.zeros(n, jnp.float32),
                      e=jnp.zeros((len(layout.pl_rows), len(layout.pl_cols)), jnp.float32),
                      u=jnp.ones((len(layout.pl_rows), len(layout.pl_cols)), jnp.float32),
                      mod=jnp.zeros(max(m, 1), jnp.float32), g=jnp.zeros(n, jnp.float32), ws=ws, wg=wg,
                      xd=jnp.zeros(n, jnp.float32) if layout.has_delay else (), wl=wl)


def block_genome(layout: Layout, genome: Genome) -> Genome:
    """The genome as step() reads it during a life: the N x N genes cut to the plastic block. The full weights are
    only needed at birth (init_state); copies of a genome for its siblings are then small."""
    if not layout.pl_rows:
        return genome
    ix = np.ix_(np.asarray(layout.pl_rows), np.asarray(layout.pl_cols))
    cut = lambda a: a[(..., *ix)] if a.ndim == 3 else a[ix]
    return genome._replace(**{k: cut(getattr(genome, k)) for k in ("w0", "mask", "eta", "A", "B", "C", "D")})


def _fixed_weights(layout: Layout, w: jnp.ndarray) -> tuple:
    """The part of the weights that stays as it is for a life: everything outside the plastic block (the block is
    read from the state every tick). Signed, split into the additive and the gain part, and cut into the columns
    of each update phase, so that the input neurons' columns (no synapse ends on an input) are never multiplied."""
    assert not np.asarray(layout.allowed)[:, :layout.n_in].any(), "a projection ends on the input neurons"
    if layout.pl_rows:
        w = w.at[np.ix_(np.asarray(layout.pl_rows), np.asarray(layout.pl_cols))].set(0.0)
    eff = effective(layout, w)
    add = eff * (1.0 - layout.gain) if layout.has_gain else eff
    wl = ()
    if layout.has_lag:   # slow synapses act on the previous tick's activity: kept apart from the others
        wl = tuple((add * layout.lag)[:, np.asarray(cols)] for cols in layout.phases)
        add = add * (1.0 - layout.lag)
    ws = tuple(add[:, np.asarray(cols)] for cols in layout.phases)
    wg = tuple((eff * layout.gain)[:, np.asarray(cols)] for cols in layout.phases) if layout.has_gain else ()
    return ws, wg, wl


def _block_in_phases(layout: Layout) -> tuple:
    """Static: per update phase, (positions in pl_cols, positions in the phase) of the plastic block's columns."""
    out = []
    for cols in layout.phases:
        where = {c: i for i, c in enumerate(cols)}
        sel = [j for j, c in enumerate(layout.pl_cols) if c in where]
        out.append((np.asarray(sel, np.int32), np.asarray([where[layout.pl_cols[j]] for j in sel], np.int32)))
    return tuple(out)


def plasticity_rule(rule, A, B, C, D, w, x_pre, x_post, tr_pre, tr_post, target, centred=None):
    """Rule value per synapse of a block [pre, post] (before learning rate and modulator), selected by `rule`
    (a constant array). Only the rules that occur in the block are computed. `target`: teacher activity per
    post neuron (rule 'delta'). `centred`: where 1, 'hebb' uses the pre activity minus its slow average tr_pre.

        hebb   A pre post + B pre + C post + D     (general Hebbian polynomial; B alone = presynaptic trace)
        oja    A post (pre - post w)               (Hebb with Oja's normalisation)
        trace  A tr_pre post - C pre tr_post       (timing: pre before post strengthens, post before pre weakens)
        delta  A pre (target - post)               (error-correcting, teacher activity per post neuron)

    The weight change is then eta * mod * this (or * its eligibility trace), see step()."""
    pre, post = x_pre[:, None], x_post[None, :]
    present = set(np.unique(rule).tolist())
    out = jnp.zeros_like(w)
    if 1 in present:
        dev = pre if centred is None else pre - centred * tr_pre[:, None]
        out = jnp.where(rule == 1, A * dev * post + B * dev + C * post + D, out)
    if 2 in present:
        out = jnp.where(rule == 2, A * post * (pre - post * w), out)
    if 3 in present:
        out = jnp.where(rule == 3, A * tr_pre[:, None] * post - C * pre * tr_post[None, :], out)
    if 4 in present:
        out = jnp.where(rule == 4, A * pre * (target - x_post)[None, :], out)
    return out


def compute_modulators(layout: Layout, x: jnp.ndarray, world_sig: jnp.ndarray, prev: jnp.ndarray,
                       clear: jnp.ndarray) -> jnp.ndarray:
    out = []
    for i, (kind, terms, _, base, scale, decay) in enumerate(layout.mod_specs):
        if kind == "region":
            v = sum(wt * x[a:b].mean() for a, b, wt in terms)
        else:
            v = world_sig[WORLD_SIGNALS.index(kind)]
        v = scale * (v - base)
        if decay > 0 and layout.mod_accum and layout.mod_accum[i] > 0:
            out.append(clear[i] * prev[i] + layout.mod_accum[i] * v)
        else:
            out.append(jnp.maximum(v, clear[i] * prev[i]) if decay > 0 else v)
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
         world_sig: jnp.ndarray, key: jax.Array, with_logits: bool = False):
    """One world tick of brain activity. `obs` is the flattened observation [n_in]; `world_sig` the vector of
    world signals (WORLD_SIGNALS) of the last tick, used only by 'world:*' modulators. Returns (state, action), and
    the action logits as well with `with_logits` (probes)."""
    n_in, n_out = layout.n_in, layout.n_out
    w, tr, e, u, mod, g, xd = state.w, state.tr, state.e, state.u, state.mod, state.g, state.xd
    raw = obs
    if layout.has_adapt:   # adapting inputs pass on the input minus its slow average (tr holds the average of the raw input)
        obs = obs - layout.in_adapt * genome.adapt[0] * tr[:n_in]
    x = state.x.at[:n_in].set(obs)
    plastic = bool(layout.pl_rows)
    if plastic:   # everything that changes within a life sits in the block [R, C] (usually a small part of w)
        R, C = np.asarray(layout.pl_rows), np.asarray(layout.pl_cols)
        ix = np.ix_(R, C)
        b_rule, b_elig, b_decay, b_mod = (np.asarray(a)[ix] for a in (layout.rule, layout.elig, layout.decay, layout.mod_idx))
        b_depU, b_deprec = np.asarray(layout.dep_U)[ix], np.asarray(layout.dep_rec)[ix]
        b_gate = jnp.where(b_mod >= 0, mod[np.maximum(b_mod, 0)], 1.0)
        b_teacher = np.asarray(layout.teacher)[C]
        b_signed = (np.asarray(layout.sign)[R] != 0)[:, None]
        blk = (lambda a: a) if genome.eta.shape == (len(R), len(C)) and len(R) * len(C) != layout.n ** 2             else (lambda a: a[ix])   # a genome cut to the block (block_genome) or a full one
        b_eta, b_mask, b_w0 = blk(genome.eta), blk(genome.mask), blk(genome.w0)
        b_A, b_B, b_C, b_D = blk(genome.A), blk(genome.B), blk(genome.C), blk(genome.D)
    h = None
    blocks = _block_in_phases(layout) if plastic else ()
    for _ in range(cfg.steps_per_tick):
        if plastic:   # the plastic block as it is now (the rest of the weights is in state.ws / state.wg)
            sR = np.asarray(layout.sign)[R][:, None]
            eb = jnp.where(sR != 0, sR * w[ix], w[ix]) * (u if layout.has_dep else 1.0)
            gb = np.asarray(layout.gain)[ix] if layout.has_gain else None
            add_b = eb * (1.0 - gb) if layout.has_gain else eb

        def drive(xin, pi):
            """Input to the neurons of phase pi: additive part and (with gain) the log gain."""
            hk = xin @ state.ws[pi]
            if layout.has_lag:   # slow synapses: the activity at the start of the tick (the previous tick's)
                hk = hk + state.x @ state.wl[pi]
            lg = xin @ state.wg[pi] if layout.has_gain else 0.0
            if plastic and len(blocks[pi][0]):
                sel, pos = blocks[pi]
                hk = hk.at[pos].add(xin[R] @ add_b[:, sel])
                if layout.has_gain:
                    lg = lg.at[pos].add(xin[R] @ (eb * gb)[:, sel])
            return hk, lg

        if len(layout.phases) <= 1:   # synchronous: every neuron reads the previous tick
            cols = np.asarray(layout.phases[0])
            hk, lg = drive(x, 0)
            h = genome.b.at[cols].add(hk)   # the input neurons get only their bias, as no synapse ends on them
            log_gain = jnp.zeros_like(h).at[cols].set(lg) if layout.has_gain else 0.0
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
            xc, h = x, jnp.zeros_like(x)
            for pi, cols in enumerate(layout.phases):
                cols = np.asarray(cols)
                hk, lg = drive(xc, pi)
                hk = hk + genome.b[cols]
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
        if plastic:   # learning, on the plastic block only:
            # base = the rule's value for this step (plasticity_rule); with an eligibility trace e = elig * e + base and
            # base = e (the synapse remembers recent pairings for about 1 / (1 - elig) steps); then
            # dW = eta * mod * base * mask, mod = the projection's modulator of the previous step (1 if none: plain
            # Hebb); then relaxation toward w0 (decay), clipping (Dale's law: magnitudes stay >= 0) and short-term
            # depression of the resource u
            wb = w[ix]
            target = jnp.where(b_teacher >= 0, x_new[np.maximum(b_teacher, 0)], x_new[C])
            cen = np.asarray(layout.centred)[ix] if np.asarray(layout.centred)[ix].any() else None
            base = plasticity_rule(b_rule, b_A, b_B, b_C, b_D, wb, x[R], x_new[C], tr[R], tr[C], target, cen)
            if layout.has_delay:   # these synapses' traces take the pre activity of the step before
                late = plasticity_rule(b_rule, b_A, b_B, b_C, b_D, wb, xd[R], x_new[C], tr[R], tr[C], target, cen)
                base = jnp.where(np.asarray(layout.elig_delay)[ix] > 0, late, base)
            if layout.has_elig:
                eb = jnp.where(b_elig > 0, b_elig * e + base, 0.0)
                e = eb
                base = jnp.where(b_elig > 0, eb, base)
            wb = wb + b_eta * b_gate * base * b_mask
            if layout.has_decay:
                wb = wb + b_decay * (b_w0 * b_mask - wb)
            wb = jnp.where(b_signed, jnp.clip(wb, 0.0, cfg.w_max), jnp.clip(wb, -cfg.w_max, cfg.w_max))
            w = w.at[ix].set(wb)
            if layout.has_dep:
                u = jnp.clip(u + b_deprec * (1.0 - u) - b_depU * u * jnp.maximum(x[R], 0.0)[:, None], 0.0, 1.0)
        tr = layout.trace_tau * tr + (1.0 - layout.trace_tau) * (x_new.at[:n_in].set(raw) if layout.has_adapt else x_new)
        if layout.has_delay:
            xd = x
        x = x_new
    mod = compute_modulators(layout, x, world_sig, mod, genome.clear)
    logits = h[-n_out:] * cfg.logit_gain
    if cfg.action_temperature > 0:
        action = jax.random.categorical(key, logits / cfg.action_temperature)
    else:
        action = jnp.argmax(logits)
    new = BrainState(x=x, w=w, tr=tr, e=e, u=u, mod=mod, g=g, ws=state.ws, wg=state.wg, xd=xd, wl=state.wl)
    if with_logits:
        return new, action.astype(jnp.int32), logits / cfg.action_temperature if cfg.action_temperature > 0 else logits
    return new, action.astype(jnp.int32)


# ---------------------------------------------------------------- warm starts across layouts

def remap_genomes(pop: Genome, old: Layout, new: Layout, key: jax.Array) -> Genome:
    """Carry a population to a new layout (ADR-012 'build on top'). Neurons are matched by region name (and
    input features by name); new neurons and new projections come from a fresh init, but every *new* evolvable
    synapse starts at w0 = 0 so inherited behaviour is unchanged until evolution uses it ('start silent').
    Plasticity genes are copied for projections that exist in both layouts with the same rule.
    Mind: that includes A-D. A projection whose `abcd` is changed in a new stage but keeps its rule inherits the old
    coefficients and the new design is silently ignored (a sign flip of the td critic was lost this way, STAGE_LOG
    v54). Change the modulator instead (as `critic()` does with `da_neg`) or give the projection a new rule."""
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
    # hard-wired = not evolvable. An evolvable projection with a designed start (w_init) is not hard-wired: where it
    # existed before its evolved weights carry over; only where it is new it starts at the design value. (Until
    # 2026-10-18 it counted as hard-wired, and every remap reset e.g. the evolved hungry -> feeding synapses.)
    # A fixed random projection (not evolvable, no design value, e.g. the input of the pallium) belongs to the genome
    # too: it carries over, or the evolved weights that read it would read a new random code.
    hard = np.asarray(new.evolve_w) == 0
    designed = ~np.isnan(np.asarray(new.w_init))

    def mat(fresh_a, old_a, where):
        out = np.array(fresh_a)
        sub = np.asarray(old_a)[:, io][:, :, io]
        blk = out[:, inn][:, :, inn]
        wmask = where[np.ix_(inn, inn)][None]
        blk = np.where(wmask, sub, blk)
        out[np.ix_(np.arange(P), inn, inn)] = blk
        return out

    carry_w = carried & ~(hard & designed)
    w0 = mat(fresh.w0, pop.w0, carry_w)
    mask = mat(fresh.mask, pop.mask, carry_w)
    silent = (new_allowed > 0) & ~carried & ~hard & ~designed
    w0 = np.where(silent[None], 0.0, w0)
    tuned = np.zeros((new.n, new.n), bool)   # hard-wired strengths tuned by evolution carry over
    tuned[np.ix_(inn, inn)] = (np.asarray(old.tune)[np.ix_(io, io)] > 0) & (np.asarray(new.tune)[np.ix_(inn, inn)] > 0)
    w0 = mat(w0, pop.w0, tuned & carried)
    w0 = np.asarray(constrain(new, jnp.asarray(w0), 1e9))
    genes = {k: mat(getattr(fresh, k), getattr(pop, k), same_rule & carried) for k in ("eta", "A", "B", "C", "D")}
    b = np.array(fresh.b)
    keep_b = np.asarray(new.evolve_b)[inn] > 0
    b[:, inn[keep_b]] = np.asarray(pop.b)[:, io[keep_b]]
    clear = np.array(fresh.clear)   # an evolved decay carries over for a modulator that persists in both layouts
    for i, m in enumerate(new.mod_names):
        if m in old.mod_names and new.mod_decay[i] > 0 and old.mod_decay[old.mod_names.index(m)] > 0:
            clear[:, i] = np.asarray(pop.clear)[:, old.mod_names.index(m)]
    adapt = np.asarray(pop.adapt) if (old.adapt_evolve and new.adapt_evolve) else np.asarray(fresh.adapt)   # an evolved strength carries over
    return Genome(w0=jnp.asarray(w0 * mask), mask=jnp.asarray(mask), b=jnp.asarray(b), clear=jnp.asarray(clear),
                  adapt=jnp.asarray(adapt), **{k: jnp.asarray(v) for k, v in genes.items()})
