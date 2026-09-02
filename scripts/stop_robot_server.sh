#!/bin/bash
# Stop the policy server started by start_robot_server.sh.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/robot_server.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "No PID file at $PID_FILE -- server doesn't appear to be running (or wasn't started with start_robot_server.sh)."
  exit 1
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  kill "$PID"
  echo "Sent SIGTERM to PID $PID."
else
  echo "PID $PID from $PID_FILE is not running."
fi

rm -f "$PID_FILE" "$PID_FILE.log"
