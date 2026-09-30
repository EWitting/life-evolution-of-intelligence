"""Export a self-contained interactive dashboard.html for a run directory (ADR-014).

    python -m life.dashboard runs/<exp>/<timestamp>        # (re)generate dashboard.html, then open it in a browser

The page embeds the recording as base64 typed arrays, OHOL sprites, player bodies and a grassland ground texture if available, the layout (regions) and the
fitness curve. See life/dashboard.html for the page itself (plain HTML + JS, no build step).
"""
from __future__ import annotations
import base64
import json
import sys
from pathlib import Path

import numpy as np

from .ruleset import appearance_for
from . import actions as A

TEMPLATE = Path(__file__).with_name("dashboard.html")


def _b64(a: np.ndarray, dtype) -> dict:
    a = np.ascontiguousarray(a.astype(dtype))
    return {"dtype": np.dtype(dtype).name, "shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode()}


def build_data(run_dir: Path) -> dict:
    rec = np.load(run_dir / "recording.npz")
    cfg = json.loads((run_dir / "config.json").read_text())
    rs = json.loads((run_dir / "ruleset.json").read_text())
    layout = json.loads((run_dir / "layout.json").read_text())
    fitness = []
    if (run_dir / "fitness.csv").exists():
        import csv
        with open(run_dir / "fitness.csv") as f:
            fitness = [{k: float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    k = cfg["vision"]["appearance_dim"]
    colours = [[0.5 + 0.45 * float(c) for c in appearance_for(int(o), 3)] if o else [1, 1, 1] for o in rs["ohol_id"]]
    appearance = [appearance_for(int(o), k).tolist() for o in rs["ohol_id"]]
    w_max = cfg["brain"]["w_max"]
    eta_max = cfg["evolution"]["eta_max"]
    try:
        from .sprites import sprite_data_urls, ground_data_url, person_data_urls
        sprites = {str(i): url for i, url in sprite_data_urls(rs["ohol_id"]).items()}
        ground = ground_data_url(0)   # single biome (grassland) for now
        people = person_data_urls(int(rec["pos"].shape[1]))
    except Exception as e:   # sprites are optional
        print(f"sprites skipped: {e}")
        sprites, ground, people = {}, None, []
    x = np.clip(rec["x"].astype(np.float32), -1, 1)
    return {
        "meta": {"T": int(x.shape[0]), "N": int(x.shape[1]), "H": int(rec["grid"].shape[1]), "W": int(rec["grid"].shape[2]),
                 "best": int(rec["best"]), "names": rs["names"], "ohol_id": rs["ohol_id"], "colours": colours,
                 "appearance": appearance, "food_value": rs["food_value"], "actions": A.NAMES, "layout": layout,
                 "vision": cfg["vision"], "world": cfg["world"], "brain": cfg["brain"], "w_max": w_max,
                 "eta_max": eta_max, "snap_every": int(x.shape[0] // max(1, rec["w_snap"].shape[0])),
                 "fitness": fitness, "config": cfg, "run": run_dir.name, "experiment": cfg["name"]},
        "grid": _b64(rec["grid"], np.int16), "pos": _b64(rec["pos"], np.int16), "dir": _b64(rec["dir"], np.int8),
        "alive": _b64(rec["alive"], np.uint8), "held": _b64(rec["held"], np.int16), "food": _b64(rec["food"], np.float32),
        "pain": _b64(rec["pain"], np.float32), "action": _b64(rec["action"], np.int8), "mod": _b64(rec["mod"], np.float32),
        "x": _b64(np.round(x * 127), np.int8),
        "w_snap": _b64(np.round(np.clip(rec["w_snap"].astype(np.float32) / w_max, -1, 1) * 127), np.int8),
        "w0": _b64(np.round(np.clip(rec["w0"].astype(np.float32) / w_max, -1, 1) * 127), np.int8),
        "eta": _b64(np.round(np.clip(rec["eta"].astype(np.float32) / eta_max, 0, 1) * 255), np.uint8),
        "sprites": sprites, "ground": ground, "people": people,
    }


def export_dashboard(run_dir: Path | str) -> Path:
    run_dir = Path(run_dir)
    data = build_data(run_dir)
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(data))
    out = run_dir / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    return out


if __name__ == "__main__":
    print("wrote", export_dashboard(sys.argv[1]))
