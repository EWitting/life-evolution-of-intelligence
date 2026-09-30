"""Ruleset: objects and transitions in compact local ids (ADR-004).

Local id 0 is always EMPTY (empty hand / empty ground). Build a Ruleset with RulesetBuilder
(hand-made, for tests) or with life.ohol.slice_ruleset (from OHOL data). `to_arrays` converts
it to the jnp arrays the jitted world step consumes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import NamedTuple
import numpy as np
import jax.numpy as jnp

EMPTY = 0
AGENT_APPEARANCE_SEED = -7   # reserved seed for the agent appearance vector
SYNTHETIC_ID_BASE = 100000   # ohol ids >= this are objects we invented


def appearance_for(seed_id: int, k: int) -> np.ndarray:
    """Deterministic unit vector per object id (ADR-006). Id 0 (empty) is all zeros."""
    if seed_id == EMPTY:
        return np.zeros(k, np.float32)
    rng = np.random.default_rng(np.uint64(seed_id + 2**31))
    v = rng.normal(size=k).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


class RuleArrays(NamedTuple):
    """Device arrays consumed by world/sensors. Indexed by local id. T = number of transitions (>= 1)."""
    food_value: jnp.ndarray      # [M] float; >0 feeds, <0 drains (poison)
    pain_value: jnp.ndarray      # [M] float; pain added to body signal when eaten
    edible: jnp.ndarray          # [M] bool
    permanent: jnp.ndarray       # [M] bool
    holdable: jnp.ndarray        # [M] bool
    blocks: jnp.ndarray          # [M] bool
    num_uses: jnp.ndarray        # [M] int32 (>= 1)
    map_chance: jnp.ndarray      # [M] float; natural spawn weight
    spawn_weight: jnp.ndarray    # [M] float; weight used by init_world (default = map_chance)
    use_table: jnp.ndarray       # [M, M] int32 transition index or -1, indexed [actor, target]
    last_use_table: jnp.ndarray  # [M, M] int32 transition index or -1, applied on the target's last use
    trans_new_actor: jnp.ndarray   # [T] int32
    trans_new_target: jnp.ndarray  # [T] int32
    decay_new: jnp.ndarray       # [M] int32 new object after decay, or -1
    decay_ticks: jnp.ndarray     # [M] int32 ticks until decay (>= 1), or -1
    appearance: jnp.ndarray      # [M, K] float
    agent_appearance: jnp.ndarray  # [K] float


@dataclass
class Ruleset:
    ohol_id: np.ndarray          # [M] int; 0 for EMPTY, >= SYNTHETIC_ID_BASE for invented objects
    names: list[str]
    food_value: np.ndarray
    pain_value: np.ndarray
    permanent: np.ndarray
    holdable: np.ndarray
    blocks: np.ndarray
    num_uses: np.ndarray
    map_chance: np.ndarray
    use_table: np.ndarray
    last_use_table: np.ndarray
    trans_new_actor: np.ndarray
    trans_new_target: np.ndarray
    decay_new: np.ndarray
    decay_ticks: np.ndarray

    @property
    def size(self) -> int:
        return len(self.names)

    def local(self, ohol_id: int) -> int:
        idx = np.nonzero(self.ohol_id == ohol_id)[0]
        if len(idx) == 0:
            raise KeyError(f"ohol id {ohol_id} not in ruleset")
        return int(idx[0])

    def local_by_name(self, name: str) -> int:
        return self.names.index(name)

    def to_arrays(self, appearance_dim: int, spawn_weight: np.ndarray | None = None,
                  food_value: np.ndarray | None = None, pain_value: np.ndarray | None = None) -> RuleArrays:
        """Convert to device arrays. Optional overrides let experiments change edibility or spawning
        without touching the ruleset (used for the per-generation poison swap in exp02)."""
        fv = self.food_value if food_value is None else np.asarray(food_value, np.float32)
        pv = self.pain_value if pain_value is None else np.asarray(pain_value, np.float32)
        sw = self.map_chance if spawn_weight is None else np.asarray(spawn_weight, np.float32)
        app = np.stack([appearance_for(int(i), appearance_dim) for i in self.ohol_id])
        return RuleArrays(
            food_value=jnp.asarray(fv, jnp.float32),
            pain_value=jnp.asarray(pv, jnp.float32),
            edible=jnp.asarray((fv != 0) | (pv != 0)),
            permanent=jnp.asarray(self.permanent),
            holdable=jnp.asarray(self.holdable),
            blocks=jnp.asarray(self.blocks),
            num_uses=jnp.asarray(self.num_uses, jnp.int32),
            map_chance=jnp.asarray(self.map_chance, jnp.float32),
            spawn_weight=jnp.asarray(sw, jnp.float32),
            use_table=jnp.asarray(self.use_table, jnp.int32),
            last_use_table=jnp.asarray(self.last_use_table, jnp.int32),
            trans_new_actor=jnp.asarray(self.trans_new_actor, jnp.int32),
            trans_new_target=jnp.asarray(self.trans_new_target, jnp.int32),
            decay_new=jnp.asarray(self.decay_new, jnp.int32),
            decay_ticks=jnp.asarray(self.decay_ticks, jnp.int32),
            appearance=jnp.asarray(app, jnp.float32),
            agent_appearance=jnp.asarray(appearance_for(AGENT_APPEARANCE_SEED, appearance_dim)),
        )

    def describe(self) -> str:
        lines = [f"{self.size} objects, {len(self.trans_new_actor) - 1} transitions"]
        for i in range(self.size):
            flags = []
            if self.food_value[i]:
                flags.append(f"food={self.food_value[i]:g}")
            if self.pain_value[i]:
                flags.append(f"pain={self.pain_value[i]:g}")
            if self.holdable[i]:
                flags.append("holdable")
            if self.blocks[i]:
                flags.append("blocks")
            if self.num_uses[i] > 1:
                flags.append(f"uses={self.num_uses[i]}")
            if self.decay_new[i] >= 0:
                flags.append(f"decay->{self.names[self.decay_new[i]]} in {self.decay_ticks[i]}t")
            lines.append(f"  [{i}] ohol {self.ohol_id[i]} {self.names[i]} " + " ".join(flags))
        for a in range(self.size):
            for t in range(self.size):
                for table, tag in ((self.use_table, ""), (self.last_use_table, " (last use)")):
                    ti = table[a, t]
                    if ti >= 0:
                        lines.append(f"  {self.names[a]} + {self.names[t]} -> {self.names[self.trans_new_actor[ti]]} + "
                                     f"{self.names[self.trans_new_target[ti]]}{tag}")
        return "\n".join(lines)

    def to_json_dict(self) -> dict:
        return {"ohol_id": self.ohol_id.tolist(), "names": self.names,
                "food_value": self.food_value.tolist(), "pain_value": self.pain_value.tolist()}


class RulesetBuilder:
    """Collects objects and transitions by *ohol id* and builds the compact tables."""

    def __init__(self):
        self.objects: dict[int, dict] = {EMPTY: dict(name="empty", food_value=0.0, pain_value=0.0, permanent=False,
                                                     holdable=False, blocks=False, num_uses=1, map_chance=0.0)}
        self.transitions: list[tuple[int, int, int, int, bool]] = []   # actor, target, new_actor, new_target, last_use
        self.decays: dict[int, tuple[int, int]] = {}                   # target -> (new_target, ticks)

    def add_object(self, ohol_id: int, name: str, food_value: float = 0.0, pain_value: float = 0.0,
                   permanent: bool = False, holdable: bool = True, blocks: bool = False, num_uses: int = 1,
                   map_chance: float = 0.0):
        self.objects[ohol_id] = dict(name=name, food_value=food_value, pain_value=pain_value, permanent=permanent,
                                     holdable=holdable and not permanent, blocks=blocks, num_uses=max(1, num_uses),
                                     map_chance=map_chance)
        return self

    def add_transition(self, actor: int, target: int, new_actor: int, new_target: int, last_use: bool = False):
        self.transitions.append((actor, target, new_actor, new_target, last_use))
        return self

    def add_decay(self, target: int, new_target: int, ticks: int):
        self.decays[target] = (new_target, max(1, int(ticks)))
        return self

    def build(self) -> Ruleset:
        ids = [EMPTY] + sorted(i for i in self.objects if i != EMPTY)
        local = {oid: i for i, oid in enumerate(ids)}
        M = len(ids)

        def col(key, dtype):
            return np.array([self.objects[i][key] for i in ids], dtype=dtype)

        use_table = -np.ones((M, M), np.int32)
        last_use_table = -np.ones((M, M), np.int32)
        new_actor = [EMPTY]      # index 0 is a dummy transition (never referenced)
        new_target = [EMPTY]
        for actor, target, na, nt, last in self.transitions:
            if any(x not in local for x in (actor, target, na, nt)):
                continue
            idx = len(new_actor)
            new_actor.append(local[na])
            new_target.append(local[nt])
            table = last_use_table if last else use_table
            table[local[actor], local[target]] = idx
        decay_new = -np.ones(M, np.int32)
        decay_ticks = -np.ones(M, np.int32)
        for target, (nt, ticks) in self.decays.items():
            if target in local and nt in local:
                decay_new[local[target]] = local[nt]
                decay_ticks[local[target]] = ticks
        return Ruleset(
            ohol_id=np.array(ids, np.int64),
            names=[self.objects[i]["name"] for i in ids],
            food_value=col("food_value", np.float32),
            pain_value=col("pain_value", np.float32),
            permanent=col("permanent", bool),
            holdable=col("holdable", bool),
            blocks=col("blocks", bool),
            num_uses=col("num_uses", np.int32),
            map_chance=col("map_chance", np.float32),
            use_table=use_table,
            last_use_table=last_use_table,
            trans_new_actor=np.array(new_actor, np.int32),
            trans_new_target=np.array(new_target, np.int32),
            decay_new=decay_new,
            decay_ticks=decay_ticks,
        )
