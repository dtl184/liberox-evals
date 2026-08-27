#!/bin/bash
# One-time setup: clone LIBERO-X + openpi, create the isolated conda env this
# pipeline runs in, and install pinned dependencies.
#
# Usage: ./setup_env.sh /path/to/workspace
#
# Layout created under the workspace dir:
#   LIBERO-X/    -- github.com/meituan/LIBERO-X (benchmark + robosuite fork)
#   openpi/      -- github.com/Physical-Intelligence/openpi (VLA + policy server)
#
# A conda env named "liberox" (python 3.9) is created for the CLIENT side
# (LIBERO-X simulation + eval harness). openpi's own uv-managed venv handles
# the SERVER side (JAX policy inference) separately -- see serve_policy.sh.
set -euo pipefail

WORKSPACE="${1:?Usage: setup_env.sh /path/to/workspace}"
mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

if [ ! -d LIBERO-X ]; then
  git clone https://github.com/meituan/LIBERO-X.git
fi
if [ ! -d openpi ]; then
  git clone --depth 1 --shallow-submodules --recurse-submodules \
    https://github.com/Physical-Intelligence/openpi.git
fi

# --- client env (LIBERO-X simulation + this pipeline) ---
if ! conda env list | grep -q '^liberox '; then
  conda create -y -n liberox python=3.9
fi

# IMPORTANT: install with the env's python by absolute path, not via
# `conda activate` in a script -- conda activation does not reliably persist
# across non-interactive shell invocations, which silently leaves packages
# installed into the wrong (system/user) site-packages instead of this env.
# Always invoke $(LIBEROX_PY) by absolute path for every command downstream
# of this script, for the same reason.
LIBEROX_PY="$(conda info --base)/envs/liberox/bin/python"
PYTHONNOUSERSITE=1 "$LIBEROX_PY" -m pip install --no-cache-dir \
  -r "$WORKSPACE/LIBERO-X/requirements.txt"
PYTHONNOUSERSITE=1 "$LIBEROX_PY" -m pip install --no-cache-dir -e "$WORKSPACE/LIBERO-X"

# --- server env (openpi / policy inference) ---
cd "$WORKSPACE/openpi"
uv sync

echo ""
echo "Setup complete."
echo "  LIBERO-X client python: $LIBEROX_PY"
echo "  openpi server env:      $WORKSPACE/openpi/.venv (managed by uv)"
echo "Next: ./download_checkpoint.sh, then serve_policy.sh, then run_sweep.sh"
