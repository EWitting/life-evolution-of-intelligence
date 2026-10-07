"""Generational evolution over batched genomes (ADR-008): elitism + tournament selection, optional recombination,
mutation."""
from __future__ import annotations
import jax
import jax.numpy as jnp

from .brain import Genome, Layout, constrain
from .config import EvolutionConfig


def mutate(key: jax.Array, g: Genome, cfg: EvolutionConfig, layout: Layout) -> Genome:
    """Weights, presence and biases mutate per synapse/neuron, except where the layout hard-wires them.
    Plasticity genes (eta, A, B, C, D) mutate per projection: one value per cell-type pair (ADR-013)."""
    k = jax.random.split(key, 8)
    n = g.b.shape[0]
    flippable = (layout.evolve_w > 0) & (layout.density > 0)
    flip = (jax.random.uniform(k[0], (n, n)) < cfg.mask_flip_prob) & flippable
    mask = jnp.where(flip, 1.0 - g.mask, g.mask)
    born = (mask > 0) & (g.mask == 0)
    k1a, k1b = jax.random.split(k[1])
    hitw = jax.random.uniform(k1b, (n, n)) < cfg.weight_mutation_prob
    w0 = jnp.where((layout.evolve_w > 0) & hitw, g.w0 + cfg.mutation_std * jax.random.normal(k1a, (n, n)), g.w0)
    w0 = jnp.where(born, 0.5 * jax.random.normal(k[2], (n, n)), w0)
    w0 = constrain(layout, w0, 1e9) * mask
    # hard-wired projections flagged 'tune': the whole projection is multiplied by one factor (sign never flips)
    P = max(1, len(layout.proj_names))
    pid = jnp.where(layout.proj_id >= 0, layout.proj_id, 0)
    kt1, kt2 = jax.random.split(k[7])
    hit_t = jax.random.uniform(kt1, (P,)) < cfg.tune_prob
    factor = jnp.exp(hit_t * cfg.tune_std * jax.random.normal(kt2, (P,)))[pid]
    design = jnp.nan_to_num(layout.w_init)
    mag = jnp.clip(jnp.abs(g.w0) * factor, jnp.abs(design) / cfg.tune_range, jnp.abs(design) * cfg.tune_range)
    w0 = jnp.where(layout.tune > 0, jnp.sign(design) * mag * mask, w0)
    k3a, k3b = jax.random.split(k[3])
    hitb = jax.random.uniform(k3b, (n,)) < cfg.weight_mutation_prob
    b = jnp.where((layout.evolve_b > 0) & hitb, g.b + cfg.mutation_std * jax.random.normal(k3a, (n,)), g.b)
    if not cfg.plastic:
        return g._replace(w0=w0, mask=mask, b=b)
    plastic = (layout.rule > 0) & (layout.proj_id >= 0)
    hit = jax.random.uniform(k[4], (P,)) < cfg.eta_mutation_prob
    per = lambda kk, std: jnp.where(plastic, (hit * std * jax.random.normal(kk, (P,)))[pid], 0.0)
    eta = jnp.clip(g.eta + per(k[5], 0.3 * cfg.eta_max), 0.0, cfg.eta_max) * plastic
    ka, kb, kc, kd = jax.random.split(k[6], 4)
    return Genome(w0=w0, mask=mask, b=b, eta=eta, A=g.A + per(ka, cfg.mutation_std), B=g.B + per(kb, cfg.mutation_std),
                  C=g.C + per(kc, cfg.mutation_std), D=g.D + per(kd, 0.1 * cfg.mutation_std))


def crossover(key: jax.Array, a: Genome, b: Genome, layout: Layout, blend: bool = False) -> Genome:
    """Child of two parents. The unit of inheritance is the neuron: its incoming weights, their presence and its bias
    come together from one parent, so a neuron keeps the input pattern that made it useful. Genes that belong to a
    projection as a whole (learning rule genes, the strength of a tunable hard-wired projection) come from one
    parent per projection. Parents are assumed to be aligned (a common ancestor a few generations back)."""
    if blend:   # mid-parent values; a synapse present in one parent only is inherited at half strength
        half = lambda x, y: 0.5 * (x + y)
        return Genome(w0=half(a.w0 * a.mask, b.w0 * b.mask), mask=jnp.maximum(a.mask, b.mask), b=half(a.b, b.b),
                      eta=half(a.eta, b.eta), A=half(a.A, b.A), B=half(a.B, b.B), C=half(a.C, b.C), D=half(a.D, b.D))
    n = a.b.shape[0]
    k1, k2 = jax.random.split(key)
    from_a = jax.random.bernoulli(k1, 0.5, (n,))
    P = max(1, len(layout.proj_names))
    pid = jnp.where(layout.proj_id >= 0, layout.proj_id, 0)
    proj_a = jax.random.bernoulli(k2, 0.5, (P,))[pid]
    col = from_a[None, :]
    per_proj = lambda x, y: jnp.where(proj_a, x, y)
    return Genome(w0=jnp.where(layout.tune > 0, per_proj(a.w0, b.w0), jnp.where(col, a.w0, b.w0)),
                  mask=jnp.where(col, a.mask, b.mask), b=jnp.where(from_a, a.b, b.b), eta=per_proj(a.eta, b.eta),
                  A=per_proj(a.A, b.A), B=per_proj(a.B, b.B), C=per_proj(a.C, b.C), D=per_proj(a.D, b.D))


def next_generation(key: jax.Array, pop: Genome, fitness: jnp.ndarray, cfg: EvolutionConfig, layout: Layout) -> Genome:
    """pop is a Genome whose leaves have a leading population axis. Deterministic given key."""
    n_pop = fitness.shape[0]
    n_elite = max(1, int(round(cfg.elite_frac * n_pop)))
    order = jnp.argsort(-fitness)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    cand = jax.random.randint(k1, (n_pop, cfg.tournament), 0, n_pop)
    winners = cand[jnp.arange(n_pop), jnp.argmax(fitness[cand], axis=1)]
    parents = jnp.where(jnp.arange(n_pop) < n_elite, order[jnp.arange(n_pop)], winners)
    children = jax.tree_util.tree_map(lambda leaf: leaf[parents], pop)
    if cfg.crossover > 0:   # a second parent, also by tournament
        cand2 = jax.random.randint(k3, (n_pop, cfg.tournament), 0, n_pop)
        second = cand2[jnp.arange(n_pop), jnp.argmax(fitness[cand2], axis=1)]
        other = jax.tree_util.tree_map(lambda leaf: leaf[second], pop)
        blend = cfg.crossover_mode == "blend"
        crossed = jax.vmap(lambda k, x, y: crossover(k, x, y, layout, blend))(jax.random.split(k4, n_pop), children, other)
        sexual = jax.random.uniform(k5, (n_pop,)) < cfg.crossover
        children_x = jax.tree_util.tree_map(
            lambda x, c: jnp.where(sexual.reshape((-1,) + (1,) * (c.ndim - 1)), x, c), crossed, children)
    else:
        children_x = children
    keys = jax.random.split(k2, n_pop)
    mutated = jax.vmap(lambda k, g: mutate(k, g, cfg, layout))(keys, children_x)
    is_elite = jnp.arange(n_pop) < n_elite
    return jax.tree_util.tree_map(
        lambda c, m: jnp.where(is_elite.reshape((-1,) + (1,) * (c.ndim - 1)), c, m), children, mutated)
