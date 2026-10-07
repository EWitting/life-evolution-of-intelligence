"""World state and the jitted world step (ADR-003).

Coordinates are (y, x). Facing directions: 0 = up (-y), 1 = right (+x), 2 = down (+y), 3 = left (-x).
"""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.scipy.signal
import jax.numpy as jnp
from jax import lax

from . import actions as A
from .config import WorldConfig
from .ruleset import RuleArrays, EMPTY

DIRS = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], jnp.int32)


class WorldState(NamedTuple):
    grid_obj: jnp.ndarray     # [H, W] int32 local object id
    grid_uses: jnp.ndarray    # [H, W] int32 uses left
    grid_timer: jnp.ndarray   # [H, W] int32 ticks until decay, -1 = none
    pos: jnp.ndarray          # [N, 2] int32 (y, x)
    dir: jnp.ndarray          # [N] int32 0..3
    held: jnp.ndarray         # [N] int32 local object id (0 = empty hand)
    food: jnp.ndarray         # [N] float32
    age: jnp.ndarray          # [N] int32 ticks
    alive: jnp.ndarray        # [N] bool
    pain: jnp.ndarray         # [N] float32 decaying pain signal
    sound: jnp.ndarray        # [N] float32 vocalization emitted this tick
    parent: jnp.ndarray       # [N] int32 index of parent agent, -1 = none (reserved, ADR-008)
    tick: jnp.ndarray         # [] int32
    eaten: jnp.ndarray        # [N] float32 cumulative net food gained by eating
    alive_ticks: jnp.ndarray  # [N] int32
    pain_total: jnp.ndarray   # [N] float32 cumulative pain received
    taste: jnp.ndarray        # [N] float32 sweetness of what was eaten last tick (food gained / food_scale, >= 0)
    temp: jnp.ndarray         # [N] float32 body temperature, 0.5 = comfortable (WorldConfig.temperature)
    temp_delta: jnp.ndarray   # [N] float32 change of body temperature in the last tick
    skin: jnp.ndarray         # [N] float32 temperature of the cell the agent stands on
    skin_delta: jnp.ndarray   # [N] float32 its change in the last tick (the gradient along the agent's path)
    last_action: jnp.ndarray  # [N] int32 action taken last tick (for the efference copy)
    sick: jnp.ndarray         # [N, D] float32 pain scheduled for the coming ticks (WorldConfig.sickness_delay)
    recent: jnp.ndarray       # [N, M] float32 fading count of what was eaten recently (WorldConfig.variety_bonus)


def init_world(cfg: WorldConfig, rules: RuleArrays, key: jax.Array) -> WorldState:
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    H, W, N = cfg.height, cfg.width, cfg.num_agents
    w = rules.spawn_weight.at[EMPTY].set(0.0)
    total = w.sum()
    p = jnp.where(total > 0, w / jnp.maximum(total, 1e-9), jnp.ones_like(w) / w.shape[0])
    density = cfg.spawn_density
    if cfg.patches > 0:   # objects only inside a few discs; density is raised so the total stays the same
        kc = jax.random.split(k1)
        k1 = kc[0]
        cen = jax.random.randint(kc[1], (cfg.patches, 2), 0, jnp.array([H, W]))
        yy, xx = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
        d2 = ((yy[None] - cen[:, 0, None, None]) ** 2 + (xx[None] - cen[:, 1, None, None]) ** 2).min(axis=0)
        inside = d2 <= cfg.patch_radius ** 2
        density = jnp.where(inside, cfg.spawn_density * H * W / jnp.maximum(inside.sum(), 1), 0.0)
    place = (jax.random.uniform(k1, (H, W)) < density) & (total > 0)
    obj = jax.random.choice(k2, w.shape[0], (H, W), p=p)
    grid = jnp.where(place, obj, EMPTY).astype(jnp.int32)
    pos = jnp.stack([jax.random.randint(k3, (N,), 0, H), jax.random.randint(k4, (N,), 0, W)], axis=1).astype(jnp.int32)
    return WorldState(
        grid_obj=grid,
        grid_uses=rules.num_uses[grid],
        grid_timer=rules.decay_ticks[grid],
        pos=pos,
        dir=jax.random.randint(k5, (N,), 0, 4).astype(jnp.int32),
        held=jnp.zeros(N, jnp.int32),
        food=jnp.full(N, cfg.max_food * cfg.start_food, jnp.float32),
        age=jnp.zeros(N, jnp.int32),
        alive=jnp.ones(N, bool),
        pain=jnp.zeros(N, jnp.float32),
        sound=jnp.zeros(N, jnp.float32),
        parent=-jnp.ones(N, jnp.int32),
        tick=jnp.int32(0),
        eaten=jnp.zeros(N, jnp.float32),
        alive_ticks=jnp.zeros(N, jnp.int32),
        pain_total=jnp.zeros(N, jnp.float32),
        taste=jnp.zeros(N, jnp.float32),
        temp=jnp.full(N, 0.5, jnp.float32),
        temp_delta=jnp.zeros(N, jnp.float32),
        skin=jnp.full(N, cfg.ambient_temp, jnp.float32),
        skin_delta=jnp.zeros(N, jnp.float32),
        last_action=jnp.zeros(N, jnp.int32),
        sick=jnp.zeros((N, max(1, cfg.sickness_delay)), jnp.float32),
        recent=jnp.zeros((N, rules.food_value.shape[0]), jnp.float32),
    )


def local_temperature(cfg: WorldConfig, rules: RuleArrays, grid: jnp.ndarray) -> jnp.ndarray:
    """[H, W] temperature of every cell: ambient plus heat from nearby objects (linear fall-off), clipped to [0, 1]."""
    r = cfg.heat_radius
    d = jnp.abs(jnp.arange(-r, r + 1))
    kern = jnp.clip(1.0 - jnp.maximum(d[:, None], d[None, :]) / (r + 1.0), 0.0, 1.0)
    heat = rules.heat_value[grid] * cfg.heat_scale
    field = jax.scipy.signal.convolve2d(heat, kern, mode="same")
    return jnp.clip(cfg.ambient_temp + field, 0.0, 1.0)


def step_world(cfg: WorldConfig, rules: RuleArrays, state: WorldState, actions: jnp.ndarray, key: jax.Array,
               effort: jnp.ndarray | None = None):
    """One tick. Returns (new_state, events) with events = {"gained": [N] food units gained by eating,
    "pain": [N] pain received, "ate": [N] local id of the object eaten (0 = none)}.
    `effort` [N]: mean firing rate of each agent's brain this tick (WorldConfig.brain_cost)."""
    H, W = cfg.height, cfg.width
    N = actions.shape[0]
    alive = state.alive
    actions = jnp.where(alive, actions, A.NOOP)
    bounds = jnp.array([H - 1, W - 1], jnp.int32)

    # --- time decay of objects (first, so a timer set this tick counts full ticks) ---
    timer = jnp.where(state.grid_timer > 0, state.grid_timer - 1, state.grid_timer)
    fire = (timer == 0) & (rules.decay_new[state.grid_obj] >= 0)
    grid0 = jnp.where(fire, rules.decay_new[state.grid_obj], state.grid_obj)
    uses0 = jnp.where(fire, rules.num_uses[grid0], state.grid_uses)
    timer = jnp.where(fire, rules.decay_ticks[grid0], timer)

    # --- turn and move (parallel) ---
    d = state.dir
    d = jnp.where(actions == A.TURN_LEFT, (d + 3) % 4, d)
    d = jnp.where(actions == A.TURN_RIGHT, (d + 1) % 4, d)
    front = state.pos + DIRS[d]
    inb = ((front >= 0) & (front <= bounds)).all(axis=1)
    fc = jnp.clip(front, 0, bounds)
    blocked = rules.blocks[grid0[fc[:, 0], fc[:, 1]]]
    move = (actions == A.FORWARD) & inb & ~blocked
    pos = jnp.where(move[:, None], fc, state.pos)
    front = pos + DIRS[d]
    inb = ((front >= 0) & (front <= bounds)).all(axis=1)
    fc = jnp.clip(front, 0, bounds)

    # --- USE on the faced cell (sequential over agents, ADR-003) ---
    def use_one(carry, i):
        grid, uses, timer, held = carry
        y, x = fc[i, 0], fc[i, 1]
        do = (actions[i] == A.USE) & inb[i]
        h, t, u = held[i], grid[y, x], uses[y, x]
        ti = rules.use_table[h, t]
        li = rules.last_use_table[h, t]
        has = ti >= 0
        na = rules.trans_new_actor[jnp.maximum(ti, 0)]
        nt = rules.trans_new_target[jnp.maximum(ti, 0)]
        consumes = has & (nt == t) & (rules.num_uses[t] > 1)
        u_after = jnp.maximum(u - consumes.astype(jnp.int32), 0)
        is_last = consumes & (u_after <= 0) & (li >= 0)
        na = jnp.where(is_last, rules.trans_new_actor[jnp.maximum(li, 0)], na)
        nt = jnp.where(is_last, rules.trans_new_target[jnp.maximum(li, 0)], nt)
        pickup = ~has & (h == EMPTY) & (t != EMPTY) & rules.holdable[t]
        drop = ~has & (h != EMPTY) & (t == EMPTY)
        na = jnp.where(has, na, jnp.where(pickup, t, jnp.where(drop, EMPTY, h)))
        nt = jnp.where(has, nt, jnp.where(pickup, EMPTY, jnp.where(drop, h, t)))
        changed = nt != t
        new_uses = jnp.where(changed, rules.num_uses[nt], u_after)
        new_timer = jnp.where(changed, rules.decay_ticks[nt], timer[y, x])
        na = jnp.where(do, na, h)
        nt = jnp.where(do, nt, t)
        new_uses = jnp.where(do, new_uses, u)
        new_timer = jnp.where(do, new_timer, timer[y, x])
        return (grid.at[y, x].set(nt), uses.at[y, x].set(new_uses), timer.at[y, x].set(new_timer),
                held.at[i].set(na)), None

    (grid, uses, timer, held), _ = lax.scan(use_one, (grid0, uses0, timer, state.held), jnp.arange(N))

    # --- EAT ---
    eat = (actions == A.EAT) & rules.edible[held]
    if cfg.eat_on_pick:   # what a USE put into an empty hand is eaten at once
        eat = eat | ((actions == A.USE) & (state.held == EMPTY) & (held != EMPTY) & rules.edible[held])
    ate = jnp.where(eat, held, EMPTY)
    gained = jnp.where(eat, rules.food_value[held] * cfg.food_scale, 0.0)
    recent = state.recent
    if cfg.variety_bonus > 0:   # OHOL yum: fresh foods are worth more, repeated ones less (positive food only)
        rep = jnp.minimum(recent[jnp.arange(N), held], 1.0)
        mult = 1.0 + cfg.variety_bonus * (1.0 - 2.0 * rep)
        gained = jnp.where(gained > 0, gained * mult, gained)
        recent = recent * (1.0 - 1.0 / cfg.variety_tau) + jax.nn.one_hot(jnp.where(eat, held, 0), recent.shape[1]) * eat[:, None]
    pain_eat = jnp.where(eat, rules.pain_value[held], 0.0)
    held = jnp.where(eat, EMPTY, held)
    food = jnp.clip(state.food + gained, 0.0, cfg.max_food)
    # sickness: pain from what was eaten arrives sickness_delay ticks later (0 = at once)
    if cfg.sickness_delay > 0:
        pain_in = state.sick[:, 0]
        sick = jnp.concatenate([state.sick[:, 1:], jnp.zeros((N, 1), jnp.float32)], axis=1)
        sick = sick.at[:, cfg.sickness_delay - 1].add(pain_eat)
    else:
        pain_in, sick = pain_eat, state.sick
    pain = state.pain * cfg.pain_decay + pain_in

    # --- temperature and metabolism ---
    temp = state.temp
    skin = state.skin
    hunger = cfg.hunger_per_tick
    if cfg.temperature:
        local = local_temperature(cfg, rules, grid)[pos[:, 0], pos[:, 1]]
        temp = temp + cfg.temp_rate * (local - temp)
        skin = local
        hunger = hunger * (1.0 + cfg.temp_hunger * 2.0 * jnp.abs(temp - 0.5))
    if cfg.move_cost > 0 or cfg.turn_cost > 0:   # locomotion costs energy
        turning = (actions == A.TURN_LEFT) | (actions == A.TURN_RIGHT)
        hunger = hunger * (1.0 + cfg.move_cost * (actions == A.FORWARD) + cfg.turn_cost * turning)
    if cfg.brain_cost > 0 and effort is not None:
        hunger = hunger * (1.0 + cfg.brain_cost * effort)
    food = food - hunger * alive
    age = state.age + alive.astype(jnp.int32)
    alive_new = alive & (food > 0) & (age < cfg.max_age)

    new_state = state._replace(
        grid_obj=grid, grid_uses=uses, grid_timer=timer,
        pos=pos, dir=d, held=held, food=food, age=age, alive=alive_new, pain=pain,
        sound=(actions == A.VOCALIZE).astype(jnp.float32),
        tick=state.tick + 1,
        eaten=state.eaten + gained,
        alive_ticks=state.alive_ticks + alive.astype(jnp.int32),
        pain_total=state.pain_total + pain_in,
        taste=jnp.maximum(gained, 0.0) / cfg.food_scale,
        last_action=actions.astype(jnp.int32),
        sick=sick,
        temp=temp,
        temp_delta=temp - state.temp,
        skin=skin,
        skin_delta=skin - state.skin,
        recent=recent,
    )
    return new_state, {"gained": gained, "pain": pain_in, "ate": ate}
