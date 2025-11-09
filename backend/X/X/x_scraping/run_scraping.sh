#!/usr/bin/env bash
set -euo pipefail

# Go to this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv (one venv, named venv, one level up)
source ../venv/bin/activate

echo "[master] starting engine_service.py in background (silenced)..."
python engine_service.py >/dev/null 2>&1 &
ENGINE_PID=$!

# give Flask + websocket server a moment
sleep 3

echo "[master] starting fetch_and_push_40.py in background (silenced)..."
python fetch_and_push_40.py >/dev/null 2>&1 &
FETCH_PID=$!

# small pause so both are up before listener connects
sleep 2

echo
echo "[master] background processes:"
echo "  engine_service.py pid:      $ENGINE_PID"
echo "  fetch_and_push_40.py pid:   $FETCH_PID"
echo
echo "[master] starting sample_market_listener.py in foreground."
echo "[master] you should only see what the listener receives next."
echo

# listener runs in foreground so its prints show in your terminal
python sample_market_listener.py
