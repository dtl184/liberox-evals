#!/bin/bash
# Run eval_subgoals.py across LEVEL1-5, sequentially, with resume-on-restart.
#
# Usage:
#   ./run_sweep.sh /path/to/workspace /path/to/results [options]
#
# Options (env vars, all optional):
#   TASK_LIST_DIR   Dir with LEVEL1.txt..LEVEL4.txt from sample_tasks.py.
#                    Unset = every task in the level (full coverage).
#   TRIALS_PER_TASK  Default 5.
#   MAX_STEPS        Default 500.
#   LEVELS           Default "LEVEL1 LEVEL2 LEVEL3 LEVEL4 LEVEL5" -- LEVEL5
#                    reuses LEVEL4's bddl files (see README), so it needs no
#                    separate LEVEL5.txt even when TASK_LIST_DIR is set.
#   HOST / PORT      Policy server address. Default 127.0.0.1:8000.
#
# Throughput on a single RTX 4090: ~14s/episode regardless of level or
# success rate (failures run the full step budget). At that rate:
#   500  episodes  ~ 2 hours
#   1000 episodes  ~ 4 hours
# See the README "Scaling" section before running unattended overnight.
set -uo pipefail  # no -e: one level failing must not stop the rest

WORKSPACE="${1:?Usage: run_sweep.sh /path/to/workspace /path/to/results}"
RESULTS_DIR="${2:?Usage: run_sweep.sh /path/to/workspace /path/to/results}"
TASK_LIST_DIR="${TASK_LIST_DIR:-}"
TRIALS_PER_TASK="${TRIALS_PER_TASK:-5}"
MAX_STEPS="${MAX_STEPS:-500}"
LEVELS="${LEVELS:-LEVEL1 LEVEL2 LEVEL3 LEVEL4 LEVEL5}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LIBEROX_PY="$(conda info --base)/envs/liberox/bin/python"
EVAL="$SCRIPT_DIR/../src/eval_subgoals.py"

cd "$WORKSPACE/LIBERO-X"
export MUJOCO_GL=egl
export PYTHONNOUSERSITE=1

run_level () {
  local scene_group=$1
  local task_list_arg=()
  # LEVEL5 shares LEVEL4's underlying scenes/tasks (only the language
  # instruction changes -- see README), so it uses LEVEL4's task list too.
  local list_level="$scene_group"
  [ "$scene_group" = "LEVEL5" ] && list_level="LEVEL4"
  if [ -n "$TASK_LIST_DIR" ]; then
    task_list_arg=(--task-list-file "$TASK_LIST_DIR/$list_level.txt")
  fi

  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Starting $scene_group ==="
  "$LIBEROX_PY" "$EVAL" \
    --scene-group "$scene_group" --load-mode init \
    "${task_list_arg[@]}" \
    --num-trials-per-task "$TRIALS_PER_TASK" \
    --max-steps "$MAX_STEPS" \
    --host "$HOST" --port "$PORT" \
    --video-out-path "$RESULTS_DIR/$scene_group" \
    --results-out-path "$RESULTS_DIR/$scene_group/results_${scene_group}.jsonl" \
    --save-failure-videos --no-save-all-videos
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Finished $scene_group ==="
}

for level in $LEVELS; do
  run_level "$level"
done

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] ALL LEVELS DONE ==="
