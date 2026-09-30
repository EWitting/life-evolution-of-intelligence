"""Generational evolution over batched genomes (ADR-008): elitism + tournament selection + mutation."""
from __future__ import annotations
import jax
import jax.numpy as jnp

from .brain import Genome, Layout
from .config import EvolutionConfig


def mutate(key: jax.Array, g: Genome, cfg: EvolutionConfig, layout: Layout) -> Genome:
    k = jax.random.split(key, 8)
    n = g.b.shape[0]
    flip = (jax.random.uniform(k[0], (n, n)) < cfg.mask_flip_prob) & (layout.allowed > 0)
    mask = jnp.where(flip, 1.0 - g.mask, g.mask)
    born = (mask > 0) & (g.mask == 0)
    w0 = g.w0 + cfg.mutation_std * jax.random.normal(k[1], (n, n))
    w0 = jnp.where(born, 0.5 * jax.random.normal(k[2], (n, n)), w0) * mask
    b = g.b + cfg.mutation_std * jax.random.normal(k[3], (n,))
    if not cfg.plastic:
        return g._replace(w0=w0, mask=mask, b=b, eta=g.eta * mask)
    noise = lambda i: cfg.mutation_std * jax.random.normal(k[i], (n, n))
    k4a, k4b = jax.random.split(k[4])
    toggle = (jax.random.uniform(k4a, (n, n)) < cfg.eta_mutation_prob) & (layout.rule > 0)
    new_eta = jnp.where(g.eta > 0, 0.0, jax.random.uniform(k4b, (n, n), minval=0.0, maxval=cfg.eta_max))
    eta = jnp.where(toggle, new_eta, g.eta) * mask
    return Genome(w0=w0, mask=mask, b=b, eta=eta, A=g.A + noise(5), B=g.B + noise(6), C=g.C + noise(7),
                  D=g.D + 0.1 * noise(1))


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
