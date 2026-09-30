"""Composite OHOL object sprites into PNG thumbnails for the dashboard (viewer only, never for the brain).

Approximate: layers are drawn in file order at their `pos` (OHOL y axis points up), rotated by `rot` turns,
flipped by `hFlip`, tinted by `color`. Sprite anchor offsets (sprites/<id>.txt: tag, multiplicative-blend flag,
anchor x, anchor y) are applied. Needs `git -C data/ohol sparse-checkout add sprites`.
"""
from __future__ import annotations
import base64
import io
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
            cur = dict(sprite=int(line[9:]), pos=(0.0, 0.0), rot=0.0, hflip=0, color=(1.0, 1.0, 1.0), age=(-1.0, -1.0))
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
    return layers


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


def render_object(ohol_id: int, data_dir: Path = DEFAULT_DATA_DIR, thumb: int = THUMB, age: float | None = None):
    """PIL image of the composed object, or None if the object or its sprites are missing.
    `age` (years) selects age-ranged layers, which only people have; None keeps only ageless layers."""
    from PIL import Image, ImageChops
    obj_path = Path(data_dir) / "objects" / f"{ohol_id}.txt"
    if not obj_path.exists():
        return None
    layers = [l for l in object_layers(obj_path.read_text(encoding="utf-8", errors="replace"))
              if l["age"] == (-1.0, -1.0) or (age is not None and l["age"][0] <= age < l["age"][1])]
    parts = []
    for l in layers:
        im, anchor = _sprite(str(data_dir), l["sprite"])
        if im is None:
            continue
        if l["color"] != (1.0, 1.0, 1.0):
            tint = Image.new("RGBA", im.size, tuple(int(255 * c) for c in l["color"]) + (255,))
            im = ImageChops.multiply(im, tint)
        if l["hflip"]:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        if l["rot"]:
            im = im.rotate(-l["rot"] * 360.0, expand=True, resample=Image.BICUBIC)
        cx = l["pos"][0] + (anchor[0] if not l["hflip"] else -anchor[0])
        cy = -l["pos"][1] - anchor[1]
        parts.append((im, cx, cy))
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


def sprite_data_urls(ohol_ids, data_dir: Path = DEFAULT_DATA_DIR) -> dict[int, str]:
    """{ohol_id: 'data:image/png;base64,...'} for every id that can be rendered."""
    if not sprites_available(data_dir):
        return {}
    out = {}
    for oid in ohol_ids:
        oid = int(oid)
        base = oid - 100000 if oid >= 100000 else oid   # clones reuse the original's sprite
        im = render_object(base, data_dir)
        if im is None:
            continue
        if oid >= 100000:   # tint clones so they are distinguishable in the viewer
            from PIL import ImageChops, Image
            im = ImageChops.multiply(im, Image.new("RGBA", im.size, (150, 170, 255, 255)))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        out[oid] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
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


def person_data_urls(n: int, age: float = 25.0, data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    """Up to n distinct adult body sprites (facing right, as in OHOL) as PNG data URLs; agent i uses [i % len]."""
    if not sprites_available(data_dir):
        return []
    out = []
    for oid in person_ids(data_dir)[:n]:
        im = render_object(oid, data_dir, age=age)
        if im is None:
            continue
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        out.append("data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())
    return out
