#!/usr/bin/env bash
# Start TorchCode's API, coding workspace, and daily practice dashboard.
#
# Usage: ./start.sh
# Env overrides:
#   PYTHON=python3        Python interpreter to use (default: ./.venv/bin/python,
#                         created automatically if missing)
#   SKIP_INSTALL=1        Skip pip / npm dependency installation
#
# Note: the frontend talks to the backend at http://localhost:8000 (hardcoded in
# web/src/app/page.tsx), so the backend port is fixed at 8000.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$ROOT_DIR/.venv"
PID_FILE="$ROOT_DIR/.start.pid"
BACKEND_PORT=8000
FRONTEND_PORT=3000
DASHBOARD_PORT=4000

if [[ -z "${PYTHON:-}" ]]; then
	if [[ ! -x "$VENV_DIR/bin/python" ]]; then
		command -v python3 >/dev/null 2>&1 || { echo "Error: python3 not found." >&2; exit 1; }
		echo "==> Creating virtual environment at $VENV_DIR ..."
		python3 -m venv "$VENV_DIR"
	fi
	PYTHON="$VENV_DIR/bin/python"
fi

command -v "$PYTHON" >/dev/null 2>&1 || { echo "Error: $PYTHON not found." >&2; exit 1; }
echo "==> Using Python: $PYTHON"
command -v npm >/dev/null 2>&1 || { echo "Error: npm not found. Install Node.js first." >&2; exit 1; }

port_in_use() {
	lsof -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
}

for port in "$BACKEND_PORT" "$FRONTEND_PORT" "$DASHBOARD_PORT"; do
	if port_in_use "$port"; then
		echo "Error: port $port is already in use. Stop whatever is using it and retry." >&2
		exit 1
	fi
done

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
	echo "==> Installing backend dependencies..."
	"$PYTHON" -m pip install -q -r api/requirements.txt

	if [[ ! -d web/node_modules ]]; then
		echo "==> Installing frontend dependencies..."
		(cd web && npm install)
	fi
fi

BACKEND_PID=""
FRONTEND_PID=""
DASHBOARD_PID=""

cleanup() {
	trap - INT TERM EXIT
	echo
	echo "==> Shutting down..."
	# Kill whole process groups so child processes (uvicorn reloader worker,
	# next dev server spawned by npm) go down too.
	[[ -n "$DASHBOARD_PID" ]] && kill -- -"$DASHBOARD_PID" 2>/dev/null || true
	[[ -n "$FRONTEND_PID" ]] && kill -- -"$FRONTEND_PID" 2>/dev/null || true
	[[ -n "$BACKEND_PID" ]] && kill -- -"$BACKEND_PID" 2>/dev/null || true
	wait 2>/dev/null || true
	rm -f "$PID_FILE"
}
trap cleanup INT TERM EXIT
echo $$ > "$PID_FILE"

# Job control: run each background job in its own process group.
set -m

echo "==> Starting backend on http://localhost:$BACKEND_PORT ..."
"$PYTHON" -m uvicorn api.main:app --port "$BACKEND_PORT" --reload &
BACKEND_PID=$!

echo "==> Starting frontend on http://localhost:$FRONTEND_PORT ..."
(cd web && export TORCHCODE_DASHBOARD=0 && exec npm run dev -- --port "$FRONTEND_PORT") &
FRONTEND_PID=$!

echo "==> Starting practice dashboard on http://localhost:$DASHBOARD_PORT ..."
(cd web && export TORCHCODE_DASHBOARD=1 && exec npm run dev -- --port "$DASHBOARD_PORT") &
DASHBOARD_PID=$!

echo "==> TorchCode is starting:"
echo "    Daily practice: http://localhost:$DASHBOARD_PORT"
echo "    Coding workspace: http://localhost:$FRONTEND_PORT"
echo "    Ctrl+C or ./stop.sh stops all three services."

# Exit (and clean up the other services) as soon as any process dies.
while kill -0 "$BACKEND_PID" 2>/dev/null && kill -0 "$FRONTEND_PID" 2>/dev/null && kill -0 "$DASHBOARD_PID" 2>/dev/null; do
	sleep 1
done
