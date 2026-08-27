#!/bin/bash
# Start the openpi policy server. Leave this running in its own terminal (or
# background it) for the duration of a sweep -- run_sweep.sh talks to it over
# a local websocket.
#
# Usage: ./serve_policy.sh /path/to/workspace [checkpoint_name] [config_name]
set -euo pipefail

WORKSPACE="${1:?Usage: serve_policy.sh /path/to/workspace [checkpoint_name] [config_name]}"
CHECKPOINT="${2:-pi05_libero}"
CONFIG="${3:-pi05_libero}"

cd "$WORKSPACE/openpi"
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config "$CONFIG" \
  --policy.dir "$WORKSPACE/checkpoints/$CHECKPOINT"
