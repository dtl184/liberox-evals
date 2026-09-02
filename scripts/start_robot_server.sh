#!/bin/bash
# Start the openpi policy server in the background for a real-robot session
# (as opposed to serve_policy.sh, which is meant to be run in the foreground
# for local LIBERO-sim sweeps). Binds 0.0.0.0 so the robot's computer can
# reach it over the LAN, not just localhost.
#
# Usage: ./start_robot_server.sh /path/to/workspace [checkpoint_name] [config_name] [port]
set -euo pipefail

WORKSPACE="${1:?Usage: start_robot_server.sh /path/to/workspace [checkpoint_name] [config_name] [port]}"
CHECKPOINT="${2:-pi05_libero}"
CONFIG="${3:-pi05_libero}"
PORT="${4:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/robot_server.pid"
LOG_DIR="$WORKSPACE/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/robot_server_$(date '+%Y%m%d_%H%M%S').log"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Server already running with PID $(cat "$PID_FILE") (log: see $PID_FILE.log path recorded at startup)."
  exit 1
fi

cd "$WORKSPACE/openpi"
nohup uv run scripts/serve_policy.py \
  --port "$PORT" \
  policy:checkpoint \
  --policy.config "$CONFIG" \
  --policy.dir "$WORKSPACE/checkpoints/$CHECKPOINT" \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "$LOG_FILE" > "$PID_FILE.log"
echo "Started policy server, PID $(cat "$PID_FILE"), logging to $LOG_FILE"

echo "Waiting for server to come up on port $PORT..."
for _ in $(seq 1 60); do
  if curl -s -o /dev/null "http://127.0.0.1:$PORT/healthz"; then
    echo "Server is up."
    echo ""
    echo "Reachable from the robot computer at one of (pick the interface on the robot's subnet):"
    hostname -I | tr ' ' '\n' | grep -v '^127\.' | grep -v '^$' | sed "s/^/  ws:\/\//; s/\$/:$PORT/"
    exit 0
  fi
  sleep 2
done

echo "Server did not come up after 120s, check $LOG_FILE" >&2
exit 1
