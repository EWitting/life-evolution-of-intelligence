"""Book export and light dashboard (docs/BOOK_BRIEF.md). Run: .venv\\Scripts\\python.exe -m pytest -q tests/test_book.py"""
import base64

import numpy as np

from life import book
from life.dashboard import _light, _b64


def _decode(d):
    return np.frombuffer(base64.b64decode(d["data"]), dtype=d["dtype"]).reshape(d["shape"])


def test_light_grid_is_first_tick_plus_changes():
    rng = np.random.default_rng(0)
    T, H, W, N, NN = 30, 6, 7, 12, 5
    grid = np.zeros((T, H, W), np.int16)
    grid[0] = rng.integers(0, 4, (H, W))
    for t in range(1, T):                       # a few cells change per tick, some ticks none
        grid[t] = grid[t - 1]
        for _ in range(rng.integers(0, 3)):
            grid[t, rng.integers(H), rng.integers(W)] = rng.integers(0, 4)
    x = rng.uniform(-1, 1, (T, N, NN)).astype(np.float32)
    rec = {"best": np.int64(3), "grid": grid, "w_snap": rng.normal(size=(3, N, NN, NN)).astype(np.float16),
           "w0": rng.normal(size=(N, NN, NN)).astype(np.float16), "eta": np.zeros((N, NN, NN), np.float16)}
    data = _light({"meta": {}, "grid": _b64(grid, np.int16)}, rec, x, w_max=4.0, eta_max=0.5)
    assert "grid" not in data
    # rebuild the grid the way dashboard.html does (expandGrid)
    n, ci, cv = _decode(data["grid_n"]), _decode(data["grid_i"]), _decode(data["grid_v"])
    out = np.zeros((T, H * W), np.int16)
    out[0] = _decode(data["grid0"]).ravel()
    k = 0
    for t in range(1, T):
        out[t] = out[t - 1]
        for _ in range(n[t]):
            out[t, ci[k]] = cv[k]
            k += 1
    assert k == len(ci) and (out.reshape(T, H, W) == grid).all()
    agents = data["meta"]["brain_agents"]
    assert 3 in agents and agents == sorted(set(agents)) and len(agents) <= book_agents() + 1
    assert _decode(data["x"]).shape == (T, len(agents), NN)
    assert _decode(data["w_snap"]).shape == (3, len(agents), NN, NN)
    assert _decode(data["w0"]).shape == (len(agents), NN, NN)


def book_agents():
    from life.dashboard import BRAIN_AGENTS
    return BRAIN_AGENTS


def test_projection_indices_and_rounding():
    layout = {"names": ["in", "a", "out"], "offsets": [0, 4, 7], "sizes": [4, 3, 2],
              "in_names": ["vis+0.hit", "vis+0.app0", "pain", "taste"]}
    rows, cols = book._indices(layout, {"src": "in", "dst": "a", "src_select": ["vis*.app*", "pain"]})
    assert rows == [1, 2] and cols == [4, 5, 6]
    rows, cols = book._indices(layout, {"src": "a", "dst": "out", "src_range": [1, 3], "dst_range": [1, 2]})
    assert rows == [5, 6] and cols == [8]
    assert book._round({"a": [1.23456, float("nan"), np.float32(1234567.0)], "n": 3}) == {"a": [1.235, None, 1235000.0], "n": 3}
    assert book._decimals([1.0, 1e-9, 0.12345]) == [1, 0, 0.123]
