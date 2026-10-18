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
    k6, k7 = jax.random.split(jax.random.fold_in(key, 6))
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
    if cfg.patches > 0 and cfg.patch_types:
        # one type per patch: the patches are shared out over the types by spawn weight (in order, so every world
        # has the same mix) and a cell belongs to the nearest patch centre. Heat sources are placed apart, evenly
        k8, k9 = jax.random.split(jax.random.fold_in(key, 8))
        wc = jnp.where(rules.heat_value > 0, 0.0, w)
        we = w - wc
        cdf = jnp.cumsum(wc) / jnp.maximum(wc.sum(), 1e-9)
        patch_type = jnp.searchsorted(cdf, (jnp.arange(cfg.patches) + 0.5) / cfg.patches)
        nearest = ((yy[None] - cen[:, 0, None, None]) ** 2 + (xx[None] - cen[:, 1, None, None]) ** 2).argmin(axis=0)
        own = patch_type[nearest]
        if cfg.patch_purity < 1.0:   # mostly one kind: the rest of a patch is drawn from all types
            k10, k11 = jax.random.split(jax.random.fold_in(key, 10))
            other = jax.random.choice(k11, w.shape[0], (H, W), p=wc / jnp.maximum(wc.sum(), 1e-9))
            own = jnp.where(jax.random.uniform(k10, (H, W)) < cfg.patch_purity, own, other)
        grid = jnp.where(place & (jax.random.uniform(k8, (H, W)) < wc.sum() / jnp.maximum(total, 1e-9)),
                         own, EMPTY).astype(jnp.int32)
        even = jax.random.uniform(k9, (H, W)) < cfg.spawn_density * we.sum() / jnp.maximum(total, 1e-9)
        heat_obj = jax.random.choice(jax.random.fold_in(key, 9), w.shape[0], (H, W),
                                     p=jnp.where(we.sum() > 0, we / jnp.maximum(we.sum(), 1e-9), p))
        grid = jnp.where(even & (we.sum() > 0), heat_obj, grid).astype(jnp.int32)
    timer = rules.decay_ticks[grid]
    if cfg.start_spent > 0:   # staggered regrowth: some objects start as the spent form that decays into them
        M = rules.decay_new.shape[0]
        spent_of = jnp.full(M, -1, jnp.int32).at[jnp.where(rules.decay_new >= 0, rules.decay_new, M)].set(
            jnp.arange(M, dtype=jnp.int32), mode="drop")
        spent = (spent_of[grid] >= 0) & (jax.random.uniform(k6, (H, W)) < cfg.start_spent)
        grid = jnp.where(spent, spent_of[grid], grid)
        full_time = jnp.maximum(rules.decay_ticks[grid], 1)
        timer = jnp.where(spent, 1 + jnp.floor(jax.random.uniform(k7, (H, W)) * full_time).astype(jnp.int32), timer)
    pos = jnp.stack([jax.random.randint(k3, (N,), 0, H), jax.random.randint(k4, (N,), 0, W)], axis=1).astype(jnp.int32)
    return WorldState(
        grid_obj=grid,
        grid_uses=rules.num_uses[grid],
        grid_timer=timer,
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


def take_agents(state: WorldState, sel: jnp.ndarray) -> WorldState:
    """The same world with only the agents `sel` [n] (indices), in that order; the maps are untouched."""
    maps = ("grid_obj", "grid_uses", "grid_timer", "tick")
    return state._replace(**{f: getattr(state, f)[sel] for f in state._fields if f not in maps})


def local_temperature(cfg: WorldConfig, rules: RuleArrays, grid: jnp.ndarray) -> jnp.ndarray:
    """[H, W] temperature of every cell: ambient plus heat from nearby objects (linear fall-off), clipped to [0, 1]."""
    r = cfg.heat_radius
    d = jnp.abs(jnp.arange(-r, r + 1))
    kern = jnp.clip(1.0 - jnp.maximum(d[:, None], d[None, :]) / (r + 1.0), 0.0, 1.0)
    heat = rules.heat_value[grid] * cfg.heat_scale
    field = jax.scipy.signal.convolve2d(heat, kern, mode="same")
    return jnp.clip(cfg.ambient_temp + field, 0.0, 1.0)


def temperature_at(cfg: WorldConfig, rules: RuleArrays, grid: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
    """[N] temperature of the cells at pos [N, 2]: the values of `local_temperature` there, computed from the
    (2r+1) x (2r+1) cells around each position only. The whole field costs a kernel per cell of the map on every
    tick, which at 256 x 256 came close to the cost of all the brains."""
    r = cfg.heat_radius
    off = jnp.arange(-r, r + 1)
    d = jnp.abs(off)
    kern = jnp.clip(1.0 - jnp.maximum(d[:, None], d[None, :]) / (r + 1.0), 0.0, 1.0)
    ys, xs = pos[:, 0, None] + off[None, :], pos[:, 1, None] + off[None, :]                    # [N, k]
    inside = ((ys >= 0) & (ys < cfg.height))[:, :, None] & ((xs >= 0) & (xs < cfg.width))[:, None, :]
    obj = grid[jnp.clip(ys, 0, cfg.height - 1)[:, :, None], jnp.clip(xs, 0, cfg.width - 1)[:, None, :]]
    heat = jnp.where(inside, rules.heat_value[obj] * cfg.heat_scale, 0.0)
    return jnp.clip(cfg.ambient_temp + (heat * kern[None]).sum(axis=(1, 2)), 0.0, 1.0)


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

    # --- USE on the faced cell (in order of agent index, ADR-003) ---
    # The agents act in order of their index, and one that faces the same cell as an earlier one sees what that
    # one left. Only agents on the same cell interact, so the agents act in rounds: in round r every agent that is
    # the r-th to use its cell, all cells at once (usually one to three rounds, not one step per agent; on a GPU the
    # step per agent was most of the time). The loop carries the faced cells only (one slot per cell: the slot of
    # the first agent facing it), not the map.
    cell = fc[:, 0] * W + fc[:, 1]
    slot = jnp.argmax(cell[None, :] == cell[:, None], axis=1)
    fy, fx = fc[:, 0], fc[:, 1]

    def use_round(carry, r):
        """Every agent of rank r (the r-th agent, by index, that uses a cell) acts at once; there is at most one per
        cell, so the order within a cell is the sequential one and cells do not interact."""
        cobj, cuses, ctimer, held = carry
        s = slot
        do = use & (rank == r)
        h, t, u = held, cobj[s], cuses[s]
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
        new_timer = jnp.where(changed, rules.decay_ticks[nt], ctimer[s])
        na = jnp.where(do, na, h)
        dst = jnp.where(do, s, N)           # agents not acting this round write to a slot past the end (dropped)
        return (cobj.at[dst].set(nt, mode="drop"), cuses.at[dst].set(new_uses, mode="drop"),
                ctimer.at[dst].set(new_timer, mode="drop"), na)

    use = (actions == A.USE) & inb
    # rank of an acting agent among the acting agents that face the same cell and come before it
    same = (cell[None, :] == cell[:, None]) & use[None, :] & use[:, None]
    rank = jnp.sum(jnp.tril(same, k=-1), axis=1)
    carry = (grid0[fy, fx], uses0[fy, fx], timer[fy, fx], state.held)
    _, (cobj, cuses, ctimer, held) = lax.while_loop(
        lambda c: c[0] <= jnp.max(jnp.where(use, rank, -1)),
        lambda c: (c[0] + 1, use_round(c[1], c[0])), (jnp.int32(0), carry))
    # every agent writes its cell's final state back (agents sharing a cell write the same values)
    grid = grid0.at[fy, fx].set(cobj[slot])
    uses = uses0.at[fy, fx].set(cuses[slot])
    timer = timer.at[fy, fx].set(ctimer[slot])

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
    if cfg.eat_cost > 0:   # a bite costs energy, so one on a full stomach is a loss
        food = food - cfg.eat_cost * eat
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
        local = temperature_at(cfg, rules, grid, pos)
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
