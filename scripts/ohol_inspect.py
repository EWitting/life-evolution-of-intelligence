"""Look up OHOL objects and their transitions. Usage:
    python scripts/ohol_inspect.py 30            # by id
    python scripts/ohol_inspect.py "Sharp Stone"  # by name substring
    python scripts/ohol_inspect.py --food         # list all edible objects
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from life import ohol  # noqa: E402


def show(data, oid):
    o = data.objects[oid]
    print(f"[{o.id}] {o.name}  food={o.food_value:g} permanent={o.permanent} holdable={o.holdable} "
          f"blocks={o.blocks} uses={o.num_uses} mapChance={o.map_chance:g} biomes={o.biomes}")
    for t in data.transitions_touching(oid):
        def nm(i):
            if i <= 0:
                return {0: "EMPTY", -1: "TIME/GROUND", -2: "DEFAULT"}.get(i, f"?[{i}]")
            return f"{data.objects[i].name}[{i}]" if i in data.objects else f"category?[{i}]"
        tag = " (last use target)" if t.last_use_target else (" (last use actor)" if t.last_use_actor else "")
        extra = f" decay={t.decay_seconds:g}s" if t.actor == -1 else ""
        print(f"    {nm(t.actor)} + {nm(t.target)} -> {nm(t.new_actor)} + {nm(t.new_target)}{tag}{extra}")


def main():
    data = ohol.load()
    arg = sys.argv[1] if len(sys.argv) > 1 else "--food"
    if arg == "--food":
        for o in sorted(data.objects.values(), key=lambda o: -o.food_value):
            if o.food_value > 0:
                print(f"[{o.id}] {o.name} food={o.food_value:g}")
        return
    if arg.lstrip("-").isdigit():
        show(data, int(arg))
    else:
        for o in data.find(arg)[:30]:
            show(data, o.id)


if __name__ == "__main__":
    main()
