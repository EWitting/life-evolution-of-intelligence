# One Hour One Life data format (what we use)

Source: `data/ohol` = jasonrohrer/OneLifeData7. Verified by inspection on 2026-09-18 (see `dataVersionNumber.txt`).

## objects/<id>.txt
Line 1: `id=<int>`. Line 2: the name (may carry a `#comment` suffix, e.g. `Milkweed Debris#Flowering`).
Then `key=value` lines, some with several comma-separated values on one line. Fields we read:

| field | meaning | our use |
|---|---|---|
| `foodValue=<int>` | food restored when eaten | `food_value` |
| `permanent=<0/1>,minPickupAge=<n>` | cannot be picked up | `permanent` |
| `containable=<0/1>` / `heldInHand=<0/1>` | can be held | `holdable = (containable or heldInHand) and not permanent` |
| `blocksWalking=<0/1>,...` | blocks movement | `blocks` |
| `numUses=<n>[,<chance>]` | uses before the last-use transition | `num_uses` |
| `mapChance=<float>#biomes_<list>` | natural spawn density and biomes | `map_chance` (world generation) |
| `person=<0/1>` | player body | excluded from slices |
| `spriteID=<id>` (repeated) | sprite layers | viewer only, later |

## transitions/<actor>_<target>[_LA|_LT|_L].txt
One line: `newActor newTarget [autoDecaySeconds actorMinUseFraction targetMinUseFraction reverseUseActor reverseUseTarget move desiredMoveDist noUseActor noUseTarget]`. Only the first two are guaranteed.

- `actor = -1`: time transition. `autoDecaySeconds > 0` is seconds; `< 0` means hours (`-1` = 1 h). The target turns into `newTarget` when the timer fires.
- `actor = 0`: empty hand used on target.
- `target = -1`: actor used on empty ground (we map it to local 0). `actor = -2` (13 files) is a "default" transition; ignored.
- `_LT` / `_L`: applies instead of the plain transition when the target's last use is consumed. `_LA`: last use of the actor (ignored for now; 98 files).
- `move != 0` (254 files): the target moves (animals). Ignored for now: animals stand still.
- `newActor`/`newTarget` of 0 means "becomes nothing".

Example (gooseberry): `0_30.txt` = `31 30 0` (empty hand + bush 30 gives berry 31 in hand, bush stays, one use consumed; the bush has `numUses=6`). `0_30_L.txt` = `31 279` (last berry: bush becomes empty bush 279). `-1_279.txt` regrows the bush after a delay.

## categories/<id>.txt
`parentID=<id>`, `numObjects=<n>`, then member ids. A transition naming a category id applies to every member. `slice_ruleset` expands these when a category id appears in a transition.

## Looking things up
`.venv\Scripts\python.exe scripts/ohol_inspect.py 30` prints an object and every transition touching it.
`.venv\Scripts\python.exe scripts/ohol_inspect.py "Sharp Stone"` searches by name.
