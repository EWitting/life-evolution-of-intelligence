"""Composite OHOL object sprites into PNG thumbnails for the dashboard (viewer only, never for the brain).

Approximate: layers are drawn in file order at their `pos` (OHOL y axis points up), rotated by `rot` turns,
flipped by `hFlip`, tinted by `color`. Sprite anchor offsets (sprites/<id>.txt: tag, multiplicative-blend flag,
anchor x, anchor y) are applied. Player bodies can be dressed in OHOL clothing (`render_person`). Needs
`git -C data/ohol sparse-checkout add sprites`.
"""
from __future__ import annotations
import base64
import io
import random
import re
from functools import lru_cache
from pathlib import Path

from .ohol import DEFAULT_DATA_DIR

THUMB = 128   # sprites stay sharp up to ~6x world zoom


def sprites_available(data_dir: Path = DEFAULT_DATA_DIR) -> bool:
    return (Path(data_dir) / "sprites").is_dir()


def object_layers(obj_text: str) -> list[dict]:
    layers = []
    cur = None
    for line in obj_text.splitlines():
        if line.startswith("spriteID="):
            cur = dict(sprite=int(line[9:]), pos=(0.0, 0.0), rot=0.0, hflip=0, color=(1.0, 1.0, 1.0), age=(-1.0, -1.0),
                       invis_worn=0)
            layers.append(cur)
        elif cur is None:
            continue
        elif line.startswith("pos="):
            cur["pos"] = tuple(float(v) for v in line[4:].split(","))
        elif line.startswith("rot="):
            cur["rot"] = float(line[4:])
        elif line.startswith("hFlip="):
            cur["hflip"] = int(line[6:])
        elif line.startswith("color="):
            cur["color"] = tuple(float(v) for v in line[6:].split(","))
        elif line.startswith("ageRange="):
            cur["age"] = tuple(float(v) for v in line[9:].split(","))
        elif line.startswith("invisHolding="):   # invisWorn: 1 = hidden when worn, 2 = shown only when worn
            m = re.search(r"invisWorn=(\d+)", line)
            cur["invis_worn"] = int(m.group(1)) if m else 0
    return layers


def object_field(obj_text: str, key: str) -> str | None:
    m = re.search(rf"^{key}=([^\n#]*)", obj_text, re.M)
    return m.group(1).strip() if m else None


def _obj_text(ohol_id: int, data_dir: Path) -> str | None:
    p = Path(data_dir) / "objects" / f"{ohol_id}.txt"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


@lru_cache(maxsize=None)
def _sprite(data_dir: str, sprite_id: int):
    from PIL import Image
    p = Path(data_dir) / "sprites" / f"{sprite_id}.tga"
    if not p.exists():
        return None, (0.0, 0.0)
    im = Image.open(p).convert("RGBA")
    meta = (Path(data_dir) / "sprites" / f"{sprite_id}.txt").read_text().split()
    anchor = (float(meta[2]), float(meta[3])) if len(meta) >= 4 else (0.0, 0.0)
    return im, anchor


def _visible(l: dict, age: float | None, worn: bool = False) -> bool:
    if not (l["age"] == (-1.0, -1.0) or (age is not None and l["age"][0] <= age < l["age"][1])):
        return False
    return not (worn and l["invis_worn"] == 1) and not (not worn and l["invis_worn"] == 2)


def _part(l: dict, data_dir: Path, origin=(0.0, 0.0)):
    """One layer as (image, centre x, centre y) in image coordinates (y down), or None if its sprite is missing.
    `origin` (OHOL coordinates, y up) shifts the layer, e.g. to where a worn item sits on the body."""
    from PIL import Image, ImageChops
    im, anchor = _sprite(str(data_dir), l["sprite"])
    if im is None:
        return None
    if l["color"] != (1.0, 1.0, 1.0):
        tint = Image.new("RGBA", im.size, tuple(int(255 * c) for c in l["color"]) + (255,))
        im = ImageChops.multiply(im, tint)
    if l["hflip"]:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if l["rot"]:
        im = im.rotate(-l["rot"] * 360.0, expand=True, resample=Image.BICUBIC)
    cx = origin[0] + l["pos"][0] + (anchor[0] if not l["hflip"] else -anchor[0])
    cy = -(origin[1] + l["pos"][1]) - anchor[1]
    return im, cx, cy


def _object_parts(ohol_id: int, data_dir: Path, age: float | None = None, origin=(0.0, 0.0), worn: bool = False) -> list:
    text = _obj_text(ohol_id, data_dir)
    if text is None:
        return []
    parts = [_part(l, data_dir, origin) for l in object_layers(text) if _visible(l, age, worn)]
    return [p for p in parts if p]


def _compose(parts: list, thumb: int):
    """Draw parts in order on one canvas, crop to the content and shrink to fit `thumb`."""
    from PIL import Image
    if not parts:
        return None
    x0 = min(cx - im.width / 2 for im, cx, cy in parts)
    x1 = max(cx + im.width / 2 for im, cx, cy in parts)
    y0 = min(cy - im.height / 2 for im, cx, cy in parts)
    y1 = max(cy + im.height / 2 for im, cx, cy in parts)
    canvas = Image.new("RGBA", (int(x1 - x0) + 2, int(y1 - y0) + 2), (0, 0, 0, 0))
    for im, cx, cy in parts:
        canvas.alpha_composite(im, (int(cx - im.width / 2 - x0), int(cy - im.height / 2 - y0)))
    bbox = canvas.getbbox()
    if bbox:
        canvas = canvas.crop(bbox)
    canvas.thumbnail((thumb, thumb), Image.LANCZOS)
    return canvas


def render_object(ohol_id: int, data_dir: Path = DEFAULT_DATA_DIR, thumb: int = THUMB, age: float | None = None):
    """PIL image of the composed object, or None if the object or its sprites are missing.
    `age` (years) selects age-ranged layers, which only people have; None keeps only ageless layers."""
    return _compose(_object_parts(ohol_id, data_dir, age), thumb)


def _age_index(field: str | None, layers: list[dict], age: float) -> int:
    """A body-part index (feet have one entry per age range) -> the layer shown at `age`, -1 if none."""
    if not field:
        return -1
    idx = [int(v) for v in field.split(",") if v.strip().lstrip("-").isdigit() and 0 <= int(v) < len(layers)]
    for i in idx:
        if _visible(layers[i], age):
            return i
    return idx[0] if idx else -1


def render_person(person_id: int, outfit: dict[str, int] | None = None, age: float = 25.0,
                  data_dir: Path = DEFAULT_DATA_DIR, thumb: int = THUMB):
    """A player body dressed as OHOL's drawObject does it: a worn item is drawn at its body part's layer position
    plus the item's clothingOffset, right after that layer (a shoe after each foot, bottom then tunic after the
    body); the hat comes last. `outfit` maps the slots h (hat), t (tunic), b (bottom), s (shoes) to object ids."""
    text = _obj_text(person_id, data_dir)
    if text is None:
        return None
    layers, outfit = object_layers(text), outfit or {}
    at = {k: _age_index(object_field(text, k), layers, age) for k in ("headIndex", "bodyIndex", "backFootIndex", "frontFootIndex")}

    def worn(oid: int, i: int) -> list:
        ct = _obj_text(oid, data_dir)
        if ct is None or i < 0:
            return []
        off = [float(v) for v in (object_field(ct, "clothingOffset") or "0,0").split(",")]
        base = layers[i]["pos"]
        return _object_parts(oid, data_dir, None, (base[0] + off[0], base[1] + off[1]), worn=True)

    parts = []
    for i, l in enumerate(layers):
        if not _visible(l, age):
            continue
        p = _part(l, data_dir)
        if p:
            parts.append(p)
        if i in (at["backFootIndex"], at["frontFootIndex"]) and "s" in outfit:
            parts += worn(outfit["s"], i)
        elif i == at["bodyIndex"]:
            parts += [q for k in ("b", "t") if k in outfit for q in worn(outfit[k], i)]
    if "h" in outfit:
        parts += worn(outfit["h"], at["headIndex"])
    return _compose(parts, thumb)


# tints for colour variants, in the order of experiments.stages.COLOURS (variant 1 = Red, ...)
VARIANT_TINTS = [(255, 110, 110), (110, 150, 255), (255, 235, 90), (190, 110, 255), (255, 255, 255), (80, 80, 80),
                 (255, 150, 200), (255, 170, 70)]


def clone_base(oid: int) -> tuple[int, int]:
    """Synthetic clone id -> (original OHOL id, variant). Clone ids are 100000 + 1000 * variant + original
    (originals < 1000); exp02's single clone 100000 + original is variant 0."""
    rest = oid - 100000
    return rest % 1000, rest // 1000


def _png_url(im) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def sprite_data_urls(ohol_ids, data_dir: Path = DEFAULT_DATA_DIR) -> dict[int, str]:
    """{ohol_id: 'data:image/png;base64,...'} for every id that can be rendered."""
    if not sprites_available(data_dir):
        return {}
    out = {}
    for oid in ohol_ids:
        oid = int(oid)
        base, var = clone_base(oid) if oid >= 100000 else (oid, -1)   # clones reuse the original's sprite
        im = render_object(base, data_dir)
        if im is None:
            continue
        if oid >= 100000:   # tint clones with their colour so they are distinguishable in the viewer
            from PIL import ImageChops, Image
            tint = VARIANT_TINTS[(var - 1) % len(VARIANT_TINTS)] if var >= 1 else (150, 170, 255)
            if tint == (255, 255, 255):   # 'White': lighten instead of multiply
                im = Image.blend(im, Image.new("RGBA", im.size, (255, 255, 255, 255)), 0.45).convert("RGBA")
                im.putalpha(render_object(base, data_dir).getchannel("A"))
            else:
                im = ImageChops.multiply(im, Image.new("RGBA", im.size, tint + (255,)))
        out[oid] = _png_url(im)
    return out


GROUND_TILE = 256   # embedded texture size; OHOL draws one 512px ground texture across 4x4 cells


def ground_data_url(biome: int = 0, data_dir: Path = DEFAULT_DATA_DIR, size: int = GROUND_TILE) -> str | None:
    """Seamless OHOL ground texture (ground/ground_<biome>.tga, 0 = grassland) as a JPEG data URL, or None.
    Needs `git -C data/ohol sparse-checkout add ground`."""
    from PIL import Image
    p = Path(data_dir) / "ground" / f"ground_{biome}.tga"
    if not p.exists():
        return None
    im = Image.open(p).convert("RGB").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def person_ids(data_dir: Path = DEFAULT_DATA_DIR) -> list[int]:
    """OHOL player bodies (objects with person > 0 that can spawn), sorted by id."""
    out = []
    for p in (Path(data_dir) / "objects").glob("*.txt"):
        m = re.search(r"^person=(\d+),noSpawn=(\d+)", p.read_text(encoding="utf-8", errors="replace"), re.M)
        if m and int(m.group(1)) > 0 and m.group(2) == "0":
            out.append(int(p.stem))
    return sorted(out)


# Everyday OHOL clothing for dressing the agents (viewer only). A dress takes the place of tunic and bottom.
WARDROBE = {
    "t": [2882, 2919, 2930, 2938, 2946, 2954,      # button-down shirts
          585, 725, 726, 3443, 3444, 3445,         # wool sweaters
          202, 201, 865],                          # rabbit fur coat, shawl, rag shirt
    "b": [2881, 2918, 2929, 2937, 2945, 2953,      # short trousers
          2880, 2917, 2928, 2936, 2944, 2952,      # pleated skirts
          2878, 2916, 2927, 2935, 2943, 2951,      # long skirts
          128, 200],                               # reed skirt, rabbit fur loincloth
    "dress": [2879, 2922, 2926, 2934, 2942, 2950],
    "s": [2887, 586, 739, 740, 3446, 3447, 3448, 203, 2886, 869],
    "h": [1117, 2884, 2920, 2931, 2939, 2947, 2955, 584, 741, 742, 3440, 3441, 3442, 199, 2181],
}


def outfit_for(i: int) -> dict[str, int]:
    """A fixed, varied outfit for agent i: a top and bottom (or a dress), shoes, and a hat for about half."""
    r = random.Random(7919 * i + 1)
    out = {"s": r.choice(WARDROBE["s"])}
    if r.random() < 0.15:
        out["t"] = r.choice(WARDROBE["dress"])
    else:
        out["t"], out["b"] = r.choice(WARDROBE["t"]), r.choice(WARDROBE["b"])
    if r.random() < 0.5:
        out["h"] = r.choice(WARDROBE["h"])
    return out


def person_data_urls(n: int, age: float = 25.0, data_dir: Path = DEFAULT_DATA_DIR, dressed: bool = True) -> list[str]:
    """Up to n adult body sprites (facing right, as in OHOL) as PNG data URLs; agent i uses [i % len]. Bodies cycle
    through the playable people; with `dressed` each one wears its own outfit (`outfit_for`)."""
    if not sprites_available(data_dir):
        return []
    ids = person_ids(data_dir)
    if not ids:
        return []
    out = []
    for i in range(n if dressed else min(n, len(ids))):
        im = render_person(ids[i % len(ids)], outfit_for(i) if dressed else None, age, data_dir)
        if im is not None:
            out.append(_png_url(im))
    return out
