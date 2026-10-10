#!/usr/bin/env bash
# Stop the API, coding workspace, and dashboard started by ./start.sh.
#
# Usage: ./stop.sh
#
# First asks the running start.sh to shut down gracefully (via its PID file),
# then kills anything from this repo still listening on the API/workspace/dashboard
# ports. Processes on those ports that don't belong to this repo are left alone.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$ROOT_DIR/.start.pid"
BACKEND_PORT=8000
FRONTEND_PORT=3000
DASHBOARD_PORT=4000

# wait_for_exit <pid> <seconds>: returns 0 if the process exited in time.
wait_for_exit() {
	local pid=$1 tries=$(( $2 * 10 ))
	while kill -0 "$pid" 2>/dev/null && (( tries-- > 0 )); do
		sleep 0.1
	done
	! kill -0 "$pid" 2>/dev/null
}

# Is the process's working directory inside this repo?
belongs_to_repo() {
	local cwd
	cwd="$(lsof -a -p "$1" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')"
	[[ "$cwd" == "$ROOT_DIR" || "$cwd" == "$ROOT_DIR"/* ]]
}

stopped_any=0

if [[ -f "$PID_FILE" ]]; then
	pid="$(cat "$PID_FILE")"
	if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && belongs_to_repo "$pid" && [[ "$(ps -p "$pid" -o command=)" == *start.sh* ]]; then
		echo "==> Stopping start.sh (pid $pid)..."
		kill -TERM "$pid"
		wait_for_exit "$pid" 10 || echo "    start.sh did not exit in time; cleaning up ports directly."
		stopped_any=1
	fi
	rm -f "$PID_FILE"
fi

for port in "$BACKEND_PORT" "$FRONTEND_PORT" "$DASHBOARD_PORT"; do
	for pid in $(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true); do
		kill -0 "$pid" 2>/dev/null || continue  # already exited with its parent
		if ! belongs_to_repo "$pid"; then
			echo "==> Skipping pid $pid on port $port (not part of this repo)."
			continue
		fi
		echo "==> Stopping pid $pid on port $port..."
		kill -TERM "$pid" 2>/dev/null || true
		if ! wait_for_exit "$pid" 5; then
			echo "    pid $pid did not exit; sending SIGKILL."
			kill -KILL "$pid" 2>/dev/null || true
		fi
		stopped_any=1
	done
done

if (( stopped_any )); then
	echo "==> TorchCode stopped."
else
	echo "==> Nothing to stop."
fi
