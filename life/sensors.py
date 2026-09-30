"""Observations (ADR-005, ADR-006): egocentric ray vision, body state, and sound.

obs = {"vision": [N, W, 4+K], "body": [N, 4+K], "sound": [N, 1]}
vision column features: [hit, 1 - dist/range, is_agent, is_wall, appearance(K)]
body features:          [food/max_food, age/max_age, pain, held_present, held_appearance(K)]
"""
from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp

from .config import VisionConfig, WorldConfig
from .ruleset import RuleArrays
from .world import WorldState

VISION_FIXED = 4
BODY_FIXED = 4
SOUND_DIM = 1


def obs_size(v: VisionConfig) -> int:
    return v.columns * (VISION_FIXED + v.appearance_dim) + (BODY_FIXED + v.appearance_dim) + SOUND_DIM


def ray_offsets(v: VisionConfig) -> np.ndarray:
    """[4, W, R, 2] integer (dy, dx) offsets of ray cells per facing direction and column.
    Angles are measured clockwise from 'up'. Rounding never yields (0, 0) because max(|cos|,|sin|) >= 0.707."""
    out = np.zeros((4, v.columns, v.range, 2), np.int32)
    spread = np.linspace(-0.5, 0.5, v.columns) * v.fov_degrees if v.columns > 1 else np.zeros(1)
    for d in range(4):
        for c, a in enumerate(spread):
            theta = np.deg2rad(d * 90.0 + a)
            uy, ux = -np.cos(theta), np.sin(theta)
            for r in range(1, v.range + 1):
                out[d, c, r - 1] = (int(np.rint(r * uy)), int(np.rint(r * ux)))
    return out


def observe_all(v: VisionConfig, w: WorldConfig, rules: RuleArrays, state: WorldState, offsets: jnp.ndarray) -> dict:
    H, W = w.height, w.width
    bounds = jnp.array([H - 1, W - 1], jnp.int32)
    occ = jnp.zeros((H, W), jnp.int32).at[state.pos[:, 0], state.pos[:, 1]].add(state.alive.astype(jnp.int32))
    cols = jnp.arange(v.columns)

    def one(pos, d, held, food, age, pain):
        cells = pos[None, None, :] + offsets[d]                       # [W, R, 2]
        inb = ((cells >= 0) & (cells <= bounds)).all(axis=-1)        # [W, R]
        cc = jnp.clip(cells, 0, bounds)
        obj = state.grid_obj[cc[..., 0], cc[..., 1]]
        oc = occ[cc[..., 0], cc[..., 1]]
        hit = (obj != 0) | (oc > 0) | ~inb
        any_hit = hit.any(axis=1)
        first = jnp.argmax(hit, axis=1)
        obj_f, oc_f, inb_f = obj[cols, first], oc[cols, first], inb[cols, first]
        is_wall = any_hit & ~inb_f
        is_agent = any_hit & inb_f & (oc_f > 0)
        app = jnp.where(is_agent[:, None], rules.agent_appearance[None, :], rules.appearance[obj_f])
        app = jnp.where((any_hit & ~is_wall)[:, None], app, 0.0)
        dist = jnp.where(any_hit, 1.0 - (first + 1) / v.range, 0.0)
        vision = jnp.concatenate([any_hit[:, None], dist[:, None], is_agent[:, None], is_wall[:, None], app],
                                 axis=1).astype(jnp.float32)
        body = jnp.concatenate([jnp.array([food / w.max_food, age / w.max_age, pain, held > 0], jnp.float32),
                                rules.appearance[held]])
        return vision, body

    vision, body = jax.vmap(one)(state.pos, state.dir, state.held, state.food, state.age, state.pain)
    delta = jnp.abs(state.pos[:, None, :] - state.pos[None, :, :]).max(axis=-1)   # [N, N] Chebyshev
    n = state.pos.shape[0]
    within = (delta <= w.hear_radius) & ~jnp.eye(n, dtype=bool)
    heard = (within * (state.sound * state.alive)[None, :]).sum(axis=1)[:, None]
    return {"vision": vision, "body": body, "sound": heard.astype(jnp.float32)}


def flatten_obs(obs: dict) -> jnp.ndarray:
    """[N, obs_size] in the fixed order vision, body, sound."""
    n = obs["vision"].shape[0]
    return jnp.concatenate([obs["vision"].reshape(n, -1), obs["body"], obs["sound"]], axis=1)
