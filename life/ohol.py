"""Parser for One Hour One Life data (docs/OHOL_FORMAT.md) and slicing into a Ruleset (ADR-004)."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import re

from .ruleset import Ruleset, RulesetBuilder, EMPTY

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ohol"
TIME_ACTOR = -1


@dataclass
class OholObject:
    id: int
    name: str
    food_value: float = 0.0
    permanent: bool = False
    holdable: bool = False
    blocks: bool = False
    num_uses: int = 1
    map_chance: float = 0.0
    heat_value: float = 0.0
    biomes: tuple[int, ...] = ()
    person: bool = False


@dataclass
class OholTransition:
    actor: int
    target: int
    new_actor: int
    new_target: int
    decay_seconds: float = 0.0
    last_use_target: bool = False
    last_use_actor: bool = False
    move: int = 0


@dataclass
class OholData:
    objects: dict[int, OholObject] = field(default_factory=dict)
    transitions: list[OholTransition] = field(default_factory=list)
    categories: dict[int, list[int]] = field(default_factory=dict)

    def find(self, name_substring: str) -> list[OholObject]:
        s = name_substring.lower()
        return [o for o in self.objects.values() if s in o.name.lower()]

    def transitions_touching(self, ohol_id: int) -> list[OholTransition]:
        return [t for t in self.transitions if ohol_id in (t.actor, t.target, t.new_actor, t.new_target)]


_KV = re.compile(r"([A-Za-z]+)=([-0-9.]+)")
_TRANS_NAME = re.compile(r"^(-?\d+)_(-?\d+)(_LA|_LT|_L)?\.txt$")


def _parse_object(text: str) -> OholObject | None:
    lines = text.splitlines()
    if len(lines) < 2 or not lines[0].startswith("id="):
        return None
    oid = int(lines[0][3:])
    name = lines[1].strip()
    kv: dict[str, float] = {}
    biomes: tuple[int, ...] = ()
    for line in lines[2:]:
        if line.startswith("mapChance"):
            m = re.match(r"mapChance=([-0-9.]+)#biomes_([0-9,]*)", line)
            if m:
                kv["mapChance"] = float(m.group(1))
                biomes = tuple(int(b) for b in m.group(2).split(",") if b)
            continue
        for k, v in _KV.findall(line):
            kv.setdefault(k, float(v))   # first occurrence wins (later spriteID lines etc. are ignored)
    permanent = kv.get("permanent", 0) > 0
    return OholObject(
        id=oid, name=name,
        food_value=kv.get("foodValue", 0.0),
        permanent=permanent,
        holdable=(kv.get("containable", 0) > 0 or kv.get("heldInHand", 0) > 0) and not permanent,
        blocks=kv.get("blocksWalking", 0) > 0,
        num_uses=max(1, int(kv.get("numUses", 1))),
        map_chance=kv.get("mapChance", 0.0),
        heat_value=kv.get("heatValue", 0.0),
        biomes=biomes,
        person=kv.get("person", 0) > 0,
    )


def load(data_dir: Path | str = DEFAULT_DATA_DIR) -> OholData:
    """Parse objects/, transitions/ and categories/. Takes about a second for the full data set."""
    data_dir = Path(data_dir)
    data = OholData()
    for f in (data_dir / "objects").glob("*.txt"):
        if not f.stem.isdigit():
            continue
        obj = _parse_object(f.read_text(encoding="utf-8", errors="replace"))
        if obj is not None:
            data.objects[obj.id] = obj
    for f in (data_dir / "transitions").glob("*.txt"):
        m = _TRANS_NAME.match(f.name)
        if not m:
            continue
        parts = f.read_text(encoding="utf-8", errors="replace").split()
        if len(parts) < 2:
            continue
        suffix = m.group(3) or ""
        data.transitions.append(OholTransition(
            actor=int(m.group(1)), target=int(m.group(2)),
            new_actor=int(parts[0]), new_target=int(parts[1]),
            decay_seconds=float(parts[2]) if len(parts) > 2 else 0.0,
            last_use_target=suffix in ("_LT", "_L"),
            last_use_actor=suffix == "_LA",
            move=int(float(parts[7])) if len(parts) > 7 else 0,
        ))
    for f in (data_dir / "categories").glob("*.txt"):
        if not f.stem.isdigit():
            continue
        lines = f.read_text(encoding="utf-8", errors="replace").split()
        members = [int(x) for x in lines[2:] if re.fullmatch(r"-?\d+", x)]
        data.categories[int(f.stem)] = members
    return data


def decay_ticks(seconds: float, ticks_per_second: float, max_ticks: int) -> int:
    """OHOL: positive = seconds, negative = hours (ADR-002)."""
    secs = seconds if seconds >= 0 else -seconds * 3600.0
    return int(max(1, min(max_ticks, round(secs * ticks_per_second))))


def slice_ruleset(data: OholData, ohol_ids: list[int], expand_hops: int = 0,
                  ticks_per_second: float = 1.0, max_decay_ticks: int = 600,
                  clones: dict[int, int] | None = None, clone_name_prefix: str = "Variant ",
                  extra_decays: dict[int, tuple[int, int]] | None = None,
                  extra_transitions: list[tuple[int, int, int, int]] | None = None,
                  clone_sets: list[tuple[dict[int, int], str]] | None = None) -> Ruleset:
    """Build a Ruleset from a set of OHOL object ids.

    - Transitions are kept when actor, target, new_actor and new_target are all in the set (0 and -1 count as in).
    - expand_hops > 0 repeatedly adds the products of transitions whose actor and target are in the set.
    - Category ids appearing as actor/target are expanded to their members that are in the set.
    - clones maps original id -> synthetic id (>= 100000): the object and every transition among the cloned
      group are duplicated with the new ids, giving a look-alike object with a different appearance (exp02).
    - clone_sets [(clones, prefix), ...] makes several look-alike copies of the same objects.
    - extra_decays {target: (new_target, ticks)} and extra_transitions [(actor, target, new_actor, new_target)]
      are experiment-level rule patches applied last (they override OHOL). Use them sparingly and document
      them in the experiment docstring; they are also applied to clones.
    """
    ids = set(int(i) for i in ohol_ids)
    for _ in range(expand_hops):
        new = set()
        for t in data.transitions:
            if t.move:
                continue
            if _in(t.actor, ids) and _in(t.target, ids):
                for x in (t.new_actor, t.new_target):
                    if x > 0 and x in data.objects and not data.objects[x].person:
                        new.add(x)
        if new <= ids:
            break
        ids |= new

    b = RulesetBuilder()
    for oid in sorted(ids):
        o = data.objects[oid]
        b.add_object(oid, o.name, food_value=o.food_value, permanent=o.permanent, holdable=o.holdable,
                     blocks=o.blocks, num_uses=o.num_uses, map_chance=o.map_chance, heat_value=o.heat_value)

    def expand(x: int) -> list[int]:
        if x in data.categories:
            return [m for m in data.categories[x] if m in ids]
        return [x]

    kept: list[tuple[int, int, int, int, bool, float, bool]] = []
    for t in data.transitions:
        if t.move or t.last_use_actor or t.actor == -2:
            continue
        for actor in expand(t.actor):
            for target in expand(t.target):
                # a category that survives the transition (tool keeps its category id) resolves to the member
                na = actor if t.new_actor == t.actor and t.actor in data.categories else t.new_actor
                nt = target if t.new_target == t.target and t.target in data.categories else t.new_target
                if all(_in(x, ids) for x in (actor, target, na, nt)):
                    kept.append((actor, target, na, nt, t.last_use_target, t.decay_seconds,
                                 t.actor == TIME_ACTOR))

    def norm(x: int) -> int:
        return EMPTY if x in (EMPTY, -1) else x

    def emit(actor, target, na, nt, last, secs, is_time):
        if is_time:
            b.add_decay(norm(target), norm(nt), decay_ticks(secs, ticks_per_second, max_decay_ticks))
        else:
            b.add_transition(norm(actor), norm(target), norm(na), norm(nt), last_use=last)

    for target, (nt, ticks) in (extra_decays or {}).items():
        kept.append((TIME_ACTOR, target, EMPTY, nt, False, ticks / ticks_per_second, True))
    for actor, target, na, nt in (extra_transitions or []):
        kept.append((actor, target, na, nt, False, 0.0, False))
    for row in kept:
        emit(*row)

    sets = ([(clones, clone_name_prefix)] if clones else []) + list(clone_sets or [])
    for clones, clone_name_prefix in sets:
        for orig, new in clones.items():
            o = data.objects[orig]
            b.add_object(new, clone_name_prefix + o.name, food_value=o.food_value, permanent=o.permanent,
                         holdable=o.holdable, blocks=o.blocks, num_uses=o.num_uses, map_chance=o.map_chance,
                         heat_value=o.heat_value)
        cmap = {k: v for k, v in clones.items()}
        for actor, target, na, nt, last, secs, is_time in kept:
            if any(x in cmap for x in (actor, target, na, nt)):
                emit(cmap.get(actor, actor), cmap.get(target, target), cmap.get(na, na), cmap.get(nt, nt),
                     last, secs, is_time)
    return b.build()


def _in(x: int, ids: set[int]) -> bool:
    return x in (EMPTY, -1) or x in ids
