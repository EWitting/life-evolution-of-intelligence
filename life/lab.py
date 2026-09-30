"""Lab: step a world by hand and look inside agents (ADR-014). For Python scripts or Jupyter.

    from life.lab import Lab
    from life.experiments import exp01_evolved_forager as e
    exp = e.make_config(); rs = e.build_ruleset(exp.world)
    lab = Lab(exp, rs)                      # random genomes, or Lab(exp, rs, population=load_population(run))
    lab.step(50)                            # advance 50 ticks
    lab.step(5, override={3: A.FORWARD})    # force agent 3's action for 5 ticks
    lab.obs(3)                              # its observation, split into vision [W, F], body, sound
    lab.brain(3)                            # activations per region, weights, learning rates
    lab.world_summary()                     # what is on the grid, who is alive
    lab.render()                            # matplotlib figure of the current tick
    lab.export("runs/lab_session")          # write recording + dashboard.html of everything stepped so far
"""
from __future__ import annotations
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import brain
from . import actions as A
from .config import ExperimentConfig
from .ruleset import Ruleset, RuleArrays
from .run import make_layout, make_tick, save_recording
from .sensors import VISION_FIXED, BODY_FIXED
from .world import init_world


class Lab:
    def __init__(self, exp: ExperimentConfig, ruleset: Ruleset, population: brain.Genome | None = None,
                 rules: RuleArrays | None = None, seed: int = 0):
        self.exp, self.ruleset = exp, ruleset
        self.layout = make_layout(exp)
        self.rules = rules if rules is not None else ruleset.to_arrays(exp.vision.appearance_dim)
        self.key = jax.random.PRNGKey(seed)
        self.key, k1, k2 = jax.random.split(self.key, 3)
        N = exp.world.num_agents
        self.pop = population if population is not None else \
            jax.vmap(lambda k: brain.init_genome(k, self.layout))(jax.random.split(k1, N))
        self.world = init_world(exp.world, self.rules, k2)
        self.bstate = jax.vmap(brain.init_state)(self.pop)
        self.mod = jnp.ones(N, jnp.float32)
        self._tick = jax.jit(make_tick(exp, self.layout))
        self.history: list[dict] = []
        self.w_snaps: list[np.ndarray] = []
        self.last_obs = None
        self.last_actions = None

    @property
    def tick(self) -> int:
        return int(self.world.tick)

    def step(self, n: int = 1, override: dict[int, int] | None = None):
        """Advance n ticks. override maps agent index -> action forced on every one of these ticks."""
        N = self.exp.world.num_agents
        ov = -np.ones(N, np.int32)
        for i, a in (override or {}).items():
            ov[i] = a
        ov = jnp.asarray(ov)
        every = self.exp.evolution.record_weights_every
        for _ in range(n):
            self.key, k = jax.random.split(self.key)
            self.world, self.bstate, self.mod, acts, obs = self._tick(
                self.rules, self.pop, self.world, self.bstate, self.mod, k, ov)
            self.last_obs, self.last_actions = obs, acts
            w = self.world
            self.history.append(dict(grid=np.asarray(w.grid_obj, np.int16), pos=np.asarray(w.pos, np.int16),
                                     dir=np.asarray(w.dir, np.int8), alive=np.asarray(w.alive),
                                     held=np.asarray(w.held, np.int16), food=np.asarray(w.food),
                                     pain=np.asarray(w.pain), action=np.asarray(acts, np.int8),
                                     mod=np.asarray(self.mod), x=np.asarray(self.bstate.x, np.float16)))
            if self.tick % every == 0:
                self.w_snaps.append(np.asarray(self.bstate.w, np.float16))
        return self

    def obs(self, agent: int) -> dict:
        """The observation the agent received on the last step, split into named channels."""
        if self.last_obs is None:
            raise RuntimeError("call step() first")
        v = self.exp.vision
        o = np.asarray(self.last_obs[agent])
        nv = v.columns * (VISION_FIXED + v.appearance_dim)
        nb = BODY_FIXED + v.appearance_dim
        return {"vision": o[:nv].reshape(v.columns, -1), "body": o[nv:nv + nb], "sound": o[nv + nb:]}

    def brain(self, agent: int) -> dict:
        x = np.asarray(self.bstate.x[agent])
        L = self.layout
        return {"activations": {name: x[L.region(name)] for name in L.names},
                "w": np.asarray(self.bstate.w[agent]), "w0": np.asarray(self.pop.w0[agent]),
                "eta": np.asarray(self.pop.eta[agent]), "mask": np.asarray(self.pop.mask[agent]),
                "action": None if self.last_actions is None else A.NAMES[int(self.last_actions[agent])]}

    def agent(self, agent: int) -> dict:
        w = self.world
        return {"pos": tuple(int(v) for v in w.pos[agent]), "dir": int(w.dir[agent]),
                "held": self.ruleset.names[int(w.held[agent])], "food": float(w.food[agent]),
                "age": int(w.age[agent]), "alive": bool(w.alive[agent]), "pain": float(w.pain[agent]),
                "eaten": float(w.eaten[agent])}

    def world_summary(self) -> dict:
        grid = np.asarray(self.world.grid_obj)
        ids, counts = np.unique(grid, return_counts=True)
        return {"tick": self.tick, "alive": int(self.world.alive.sum()),
                "objects": {self.ruleset.names[int(i)]: int(c) for i, c in zip(ids, counts) if i != 0}}

    def render(self, focus: int | None = None):
        """Matplotlib figure of the current world (colour squares; the dashboard has sprites)."""
        import matplotlib.pyplot as plt
        from .viewer import colour_table, ARROWS
        cols = colour_table(self.ruleset.ohol_id.tolist())
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(cols[np.asarray(self.world.grid_obj)], interpolation="nearest")
        pos, dr, alive = np.asarray(self.world.pos), np.asarray(self.world.dir), np.asarray(self.world.alive)
        u = np.array([ARROWS[int(d)][0] for d in dr]) * alive
        v = np.array([ARROWS[int(d)][1] for d in dr]) * alive
        ax.quiver(pos[:, 1], pos[:, 0], u, v, color="black", scale=1, scale_units="xy", angles="xy", width=0.008)
        if focus is not None:
            ax.plot([pos[focus, 1]], [pos[focus, 0]], "o", ms=14, mfc="none", mec="red", mew=2)
        ax.set_title(f"tick {self.tick}  alive {int(alive.sum())}/{len(alive)}")
        ax.set_xticks([]); ax.set_yticks([])
        return fig

    def export(self, out_dir: str | Path, best: int = 0, dashboard: bool = True) -> Path:
        """Write recording.npz, layout.json, config.json, ruleset.json and dashboard.html for the stepped history."""
        import json
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(self.exp.to_json())
        (out_dir / "ruleset.json").write_text(json.dumps(self.ruleset.to_json_dict()))
        recs = {k: np.stack([h[k] for h in self.history]) for k in self.history[0]}
        snaps = self.w_snaps or [np.asarray(self.bstate.w, np.float16)]
        recs["w_snap"] = np.stack(snaps)
        save_recording(out_dir, self.exp, self.ruleset, self.layout, self.pop, recs, best, dashboard=dashboard)
        return out_dir
