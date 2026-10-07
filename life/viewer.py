"""Replay a recording: python -m life.viewer runs/<exp>/<timestamp> [--save out.gif] [--fps 10]

Left: the world (objects coloured by appearance, agents as arrows; the best agent is outlined).
Right: the best agent's neuron activations over time with a cursor, and its food level.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .ruleset import appearance_for
from . import actions as A

ARROWS = {0: (0, -0.4), 1: (0.4, 0), 2: (0, 0.4), 3: (-0.4, 0)}


def colour_table(ohol_ids: list[int]) -> np.ndarray:
    cols = np.ones((len(ohol_ids), 3), np.float32)
    for i, oid in enumerate(ohol_ids):
        if i == 0:
            continue
        v = appearance_for(int(oid), 3)
        cols[i] = 0.5 + 0.45 * v
    return cols


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--save", default=None, help="write a gif instead of showing a window")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--agent", type=int, default=None, help="agent index to focus (default: best)")
    p.add_argument("--frame", type=int, default=None, help="with --save x.png: write this single tick as an image")
    args = p.parse_args(argv)
    run = Path(args.run_dir)
    rec = np.load(run / "recording.npz")
    rs = json.loads((run / "ruleset.json").read_text())
    cfg = json.loads((run / "config.json").read_text())
    grid, pos, dr, alive, x, food, act = (rec[k] for k in ("grid", "pos", "dir", "alive", "x", "food", "action"))
    T = grid.shape[0]
    focus = int(rec["best"]) if args.agent is None else args.agent
    cols = colour_table(rs["ohol_id"])
    n_in = cfg["vision"]["columns"] * (4 + cfg["vision"]["appearance_dim"]) + 4 + cfg["vision"]["appearance_dim"] + 1

    if args.save:
        matplotlib.use("Agg")
    fig, (ax_w, ax_b) = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw=dict(width_ratios=[1, 1]))
    img = ax_w.imshow(cols[grid[0]], interpolation="nearest")
    ax_w.set_xticks([]); ax_w.set_yticks([])
    quiv = ax_w.quiver(pos[0, :, 1], pos[0, :, 0], [ARROWS[int(d)][0] for d in dr[0]], [ARROWS[int(d)][1] for d in dr[0]],
                       color="black", scale=1, scale_units="xy", angles="xy", width=0.008)
    ring = ax_w.plot([pos[0, focus, 1]], [pos[0, focus, 0]], "o", ms=14, mfc="none", mec="red", mew=2)[0]
    title = ax_w.set_title("")
    act_img = ax_b.imshow(x[:, focus, n_in:].T.astype(np.float32), aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                          interpolation="nearest")
    cursor = ax_b.axvline(0, color="k")
    ax_b.set_xlabel("tick"); ax_b.set_ylabel("hidden + output neuron")
    ax_b.set_title(f"agent {focus}: non-input activations")
    fig.colorbar(act_img, ax=ax_b, fraction=0.03)

    def update(t):
        img.set_data(cols[grid[t]])
        al = alive[t]
        u = np.array([ARROWS[int(d)][0] for d in dr[t]]) * al
        v = np.array([ARROWS[int(d)][1] for d in dr[t]]) * al
        quiv.set_offsets(np.stack([pos[t, :, 1], pos[t, :, 0]], 1))
        quiv.set_UVC(u, v)
        ring.set_data([pos[t, focus, 1]], [pos[t, focus, 0]])
        cursor.set_xdata([t, t])
        title.set_text(f"tick {t}  alive {int(al.sum())}/{len(al)}  agent {focus}: food {food[t, focus]:.1f} "
                       f"action {A.NAMES[int(act[t, focus])]}")
        return img, quiv, ring, cursor, title

    if args.save and args.frame is not None:
        update(args.frame)
        fig.savefig(args.save, dpi=100)
        print(f"wrote {args.save}")
        return
    anim = FuncAnimation(fig, update, frames=T, interval=1000 // args.fps, blit=False)
    if args.save:
        anim.save(args.save, writer="pillow", fps=args.fps)
        print(f"wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
