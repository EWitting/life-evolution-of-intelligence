"""Generational evolution over batched genomes (ADR-008): elitism + tournament selection + mutation."""
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
    k3a, k3b = jax.random.split(k[3])
    hitb = jax.random.uniform(k3b, (n,)) < cfg.weight_mutation_prob
    b = jnp.where((layout.evolve_b > 0) & hitb, g.b + cfg.mutation_std * jax.random.normal(k3a, (n,)), g.b)
    if not cfg.plastic:
        return g._replace(w0=w0, mask=mask, b=b)
    P = max(1, len(layout.proj_names))
    pid = jnp.where(layout.proj_id >= 0, layout.proj_id, 0)
    plastic = (layout.rule > 0) & (layout.proj_id >= 0)
    hit = jax.random.uniform(k[4], (P,)) < cfg.eta_mutation_prob
    per = lambda kk, std: jnp.where(plastic, (hit * std * jax.random.normal(kk, (P,)))[pid], 0.0)
    eta = jnp.clip(g.eta + per(k[5], 0.3 * cfg.eta_max), 0.0, cfg.eta_max) * plastic
    ka, kb, kc, kd = jax.random.split(k[6], 4)
    return Genome(w0=w0, mask=mask, b=b, eta=eta, A=g.A + per(ka, cfg.mutation_std), B=g.B + per(kb, cfg.mutation_std),
                  C=g.C + per(kc, cfg.mutation_std), D=g.D + per(kd, 0.1 * cfg.mutation_std))


def next_generation(key: jax.Array, pop: Genome, fitness: jnp.ndarray, cfg: EvolutionConfig, layout: Layout) -> Genome:
    """pop is a Genome whose leaves have a leading population axis. Deterministic given key."""
    n_pop = fitness.shape[0]
    n_elite = max(1, int(round(cfg.elite_frac * n_pop)))
    order = jnp.argsort(-fitness)
    k1, k2 = jax.random.split(key)
    cand = jax.random.randint(k1, (n_pop, cfg.tournament), 0, n_pop)
    winners = cand[jnp.arange(n_pop), jnp.argmax(fitness[cand], axis=1)]
    parents = jnp.where(jnp.arange(n_pop) < n_elite, order[jnp.arange(n_pop)], winners)
    children = jax.tree_util.tree_map(lambda leaf: leaf[parents], pop)
    keys = jax.random.split(k2, n_pop)
    mutated = jax.vmap(lambda k, g: mutate(k, g, cfg, layout))(keys, children)
    is_elite = jnp.arange(n_pop) < n_elite
    return jax.tree_util.tree_map(
        lambda c, m: jnp.where(is_elite.reshape((-1,) + (1,) * (c.ndim - 1)), c, m), children, mutated)
