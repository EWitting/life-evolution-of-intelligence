#!/usr/bin/env bash
# Run stage jobs on the GPU in WSL, at most JOBS at a time, each logging to runs/logs/<name>.log.
# Every line of the job file: <log name>|<arguments of life.experiments.stages>.
#   wsl.exe -d Ubuntu-24.04 -- bash scripts/gpu_queue.sh <job file> [JOBS]
set -u
cd "$(dirname "$0")/.."
JOBS=${2:-3}
export XLA_PYTHON_CLIENT_PREALLOCATE=false     # several processes share the GPU; JAX would reserve 75% for each
mkdir -p runs/logs
run() {
  name=${1%%|*}; args=${1#*|}
  ~/.venvs/life/bin/python -m life.experiments.stages $args > "runs/logs/$name.log" 2>&1
  echo "$name: exit $?, $(grep -c '^gen=' "runs/logs/$name.log") generations"
}
export -f run
grep -v '^\s*$' "$1" | xargs -d '\n' -P "$JOBS" -I{} bash -c 'sleep $((RANDOM % 20)); run "$@"' _ {}
