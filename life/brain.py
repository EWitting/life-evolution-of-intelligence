"""Brain substrate (ADR-007, ADR-013): named regions and projections compiled onto one masked weight matrix.

Neuron order in the state vector x[N]: [in (n_in) | user regions in config order | out (n_out)].
W[i, j] is the synapse from neuron i (pre) to neuron j (post). Columns of input neurons are always 0.
Every projection has its own connectivity density, learning rule and modulation flag; these are compiled
into per-synapse constant arrays in `Layout` (not evolved). The genome holds the evolvable parts.
"""
from __future__ import annotations
from typing import NamedTuple
import numpy as np
import jax
import jax.numpy as jnp

from .config import BrainConfig, RegionSpec

RULES = {"fixed": 0, "hebb": 1, "oja": 2, "trace": 3}


class Genome(NamedTuple):
    w0: jnp.ndarray    # [N, N] initial weights
    mask: jnp.ndarray  # [N, N] float 0/1 synapse presence
    b: jnp.ndarray     # [N] bias
    eta: jnp.ndarray   # [N, N] learning rate (0 = fixed synapse)
    A: jnp.ndarray     # [N, N] rule coefficient A
    B: jnp.ndarray     # [N, N] rule coefficient B
    C: jnp.ndarray     # [N, N] rule coefficient C
    D: jnp.ndarray     # [N, N] rule coefficient D


class BrainState(NamedTuple):
    x: jnp.ndarray     # [N] activations
    w: jnp.ndarray     # [N, N] current (plastic) weights
    tr: jnp.ndarray    # [N] low-pass activity traces (for the 'trace' rule)


class Layout(NamedTuple):
    """Compiled architecture. Static per experiment; arrays are constants inside jit."""
    n: int
    n_in: int
    n_out: int
    names: tuple            # region names in neuron order, starting with 'in', ending with 'out'
    offsets: tuple          # start index of each region
    sizes: tuple
    alpha: jnp.ndarray      # [N]
    trace_tau: jnp.ndarray  # [N]
    allowed: jnp.ndarray    # [N, N] 1 where a projection exists
    density: jnp.ndarray    # [N, N] init density of the projection covering the synapse
    rule: jnp.ndarray       # [N, N] int32 rule id (RULES)
    modulated: jnp.ndarray  # [N, N] 1 if the update is multiplied by the modulator
    eta_init: jnp.ndarray   # [N, N]

    def region(self, name: str) -> slice:
        i = self.names.index(name)
        return slice(self.offsets[i], self.offsets[i] + self.sizes[i])

    def to_json_dict(self) -> dict:
        return {"names": list(self.names), "offsets": list(self.offsets), "sizes": list(self.sizes),
                "n": self.n, "n_in": self.n_in, "n_out": self.n_out}


def build_layout(cfg: BrainConfig, n_in: int, n_out: int) -> Layout:
    regions = [RegionSpec("in", n_in, alpha=1.0)] + list(cfg.regions) + [RegionSpec("out", n_out, alpha=cfg.out_alpha)]
    names = tuple(r.name for r in regions)
    assert len(set(names)) == len(names), f"duplicate region names in {names}"
    sizes = tuple(r.size for r in regions)
    offsets = tuple(int(o) for o in np.cumsum((0,) + sizes[:-1]))
    n = sum(sizes)
    alpha = np.concatenate([np.full(r.size, r.alpha, np.float32) for r in regions])
    tau = np.concatenate([np.full(r.size, r.trace_tau, np.float32) for r in regions])
    allowed = np.zeros((n, n), np.float32)
    density = np.zeros((n, n), np.float32)
    rule = np.zeros((n, n), np.int32)
    modulated = np.zeros((n, n), np.float32)
    eta_init = np.zeros((n, n), np.float32)
    sl = {r.name: slice(o, o + s) for r, o, s in zip(regions, offsets, sizes)}
    for p in cfg.projections:
        assert p.src in sl and p.dst in sl, f"projection {p.src}->{p.dst} names an unknown region"
        assert p.dst != "in", "nothing may project onto 'in' (inputs are clamped)"
        assert p.rule in RULES, f"unknown rule {p.rule!r}; choose from {list(RULES)}"
        allowed[sl[p.src], sl[p.dst]] = 1.0
        density[sl[p.src], sl[p.dst]] = p.density
        rule[sl[p.src], sl[p.dst]] = RULES[p.rule]
        modulated[sl[p.src], sl[p.dst]] = float(p.modulated)
        eta_init[sl[p.src], sl[p.dst]] = p.eta_init
    j = jnp.asarray
    return Layout(n=n, n_in=n_in, n_out=n_out, names=names, offsets=offsets, sizes=sizes, alpha=j(alpha),
                  trace_tau=j(tau), allowed=j(allowed), density=j(density), rule=j(rule), modulated=j(modulated),
                  eta_init=j(eta_init))


def init_genome(key: jax.Array, layout: Layout) -> Genome:
    n = layout.n
    k1, k2, k3 = jax.random.split(key, 3)
    mask = (jax.random.uniform(k1, (n, n)) < layout.density).astype(jnp.float32) * layout.allowed
    fan_in = jnp.maximum(mask.sum(axis=0, keepdims=True), 1.0)
    w0 = jax.random.normal(k2, (n, n)) / jnp.sqrt(fan_in) * mask
    b = 0.1 * jax.random.normal(k3, (n,))
    zeros = jnp.zeros((n, n), jnp.float32)
    return Genome(w0=w0, mask=mask, b=b, eta=layout.eta_init * mask, A=jnp.ones((n, n), jnp.float32),
                  B=zeros, C=zeros, D=zeros)


def init_state(genome: Genome) -> BrainState:
    n = genome.b.shape[0]
    return BrainState(x=jnp.zeros(n, jnp.float32), w=genome.w0 * genome.mask, tr=jnp.zeros(n, jnp.float32))


def plasticity(layout: Layout, genome: Genome, w, x_pre, x_post, tr_pre, mod):
    """Per-synapse weight change for all rules at once, selected by layout.rule (ADR-013)."""
    pre, post = x_pre[:, None], x_post[None, :]
    hebb = genome.A * pre * post + genome.B * pre + genome.C * post + genome.D
    oja = genome.A * post * (pre - post * w)
    trace = genome.A * tr_pre[:, None] * post - genome.C * pre * tr_pre[None, :]
    base = jnp.where(layout.rule == 1, hebb, jnp.where(layout.rule == 2, oja, jnp.where(layout.rule == 3, trace, 0.0)))
    gate = jnp.where(layout.modulated > 0, mod, 1.0)
    return genome.eta * gate * base * genome.mask


def step(cfg: BrainConfig, layout: Layout, genome: Genome, state: BrainState, obs: jnp.ndarray,
         mod: jnp.ndarray, key: jax.Array):
    """One world tick of brain activity. `obs` is the flattened observation [n_in]; `mod` the scalar modulator
    for three-factor learning. Returns (new_state, action)."""
    n_in, n_out = layout.n_in, layout.n_out
    x = state.x.at[:n_in].set(obs)
    w, tr = state.w, state.tr
    h = None
    for _ in range(cfg.steps_per_tick):
        h = x @ w + genome.b
        x_new = (1.0 - layout.alpha) * x + layout.alpha * 0.5*(jnp.tanh(h)+1)
        x_new = x_new.at[:n_in].set(obs)
        w = jnp.clip(w + plasticity(layout, genome, w, x, x_new, tr, mod), -cfg.w_max, cfg.w_max)
        tr = layout.trace_tau * tr + (1.0 - layout.trace_tau) * x_new
        x = x_new
    logits = h[-n_out:] * cfg.logit_gain
    if cfg.action_temperature > 0:
        action = jax.random.categorical(key, logits / cfg.action_temperature)
    else:
        action = jnp.argmax(logits)
    return BrainState(x=x, w=w, tr=tr), action.astype(jnp.int32)
