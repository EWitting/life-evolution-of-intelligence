# Setup

## Windows (CPU, development) - done on 2026-09-18

```
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -e ".[dev]"
.venv\Scripts\python.exe -c "import jax; print(jax.devices())"
```

Run tests: `.venv\Scripts\python.exe -m pytest -q`
Run an experiment: `.venv\Scripts\python.exe -m life.experiments.exp01_evolved_forager`
Replay a run: `.venv\Scripts\python.exe -m life.viewer runs/exp01_evolved_forager/<timestamp>`

## OHOL data

`data/ohol` is a sparse, shallow clone of https://github.com/jasonrohrer/OneLifeData7 (public domain)
with only `objects/`, `transitions/`, `categories/` checked out. To add sprites and ground textures for the dashboard:

```
cd data/ohol
git sparse-checkout add sprites ground
```

To refresh: `git -C data/ohol pull --depth 1`.

## WSL2 + CUDA (GPU, scale) - to be done by the user, needs admin + reboot

1. In an **elevated** PowerShell: `wsl --install -d Ubuntu-24.04`, then reboot and finish the Ubuntu user setup.
2. The Windows NVIDIA driver (595.95 is installed) already exposes the GPU inside WSL2. Do **not** install a Linux display driver inside WSL. Check with `nvidia-smi` inside Ubuntu.
3. Inside Ubuntu:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   cd /mnt/c/Users/emiel/Documents/Projects/Life
   uv venv --python 3.12 .venv-wsl
   uv pip install --python .venv-wsl/bin/python -e ".[dev]" "jax[cuda12]"
   .venv-wsl/bin/python -c "import jax; print(jax.devices())"   # expect CudaDevice
   ```
   The RTX 5060 is a Blackwell GPU (sm_120). It needs `jax[cuda12]` >= 0.6, which bundles CUDA 12.8+. If you see
   "no kernel image is available", upgrade jax; if you see the CPU device only, check `nvidia-smi` in WSL first.
4. For speed, keep the repo on the Linux filesystem (`~/Life`) instead of `/mnt/c` when doing long runs; `/mnt/c` file IO is slow but code runs fine.

The same code runs unchanged on both; JAX picks the GPU automatically when present.
