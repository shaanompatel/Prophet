#!/bin/bash

# This script starts all application services in the background
# and monitors them. Pressing Ctrl+C will trigger a graceful
# shutdown of all started services.

echo "--- Starting all services ---"

# Create directories for logs and process IDs (PIDs) if they don't exist
LOG_DIR="logs"
PID_DIR="pids"
mkdir -p "$LOG_DIR"
mkdir -p "$PID_DIR"

# Clear any old PID files
rm -f $PID_DIR/*.pid

# --- Cleanup Function ---
# This function will be called when the script receives a signal to exit (e.g., Ctrl+C)
cleanup() {
    echo ""
    echo "--- Caught interrupt signal. Shutting down all services... ---"
    
    # Loop through all .pid files and kill the processes
    for pid_file in $PID_DIR/*.pid; do
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            service_name=$(basename "$pid_file" .pid)
            
            # Check if a process with this PID is actually running
            if [ -n "$pid" ] && ps -p "$pid" > /dev/null; then
                echo "Stopping '$service_name' (PID: $pid)..."
                # Use kill to send the TERM signal
                kill "$pid"
            fi
        fi
    done
    
    # Wait a moment for processes to terminate
    sleep 1
    
    echo "--- Shutdown complete ---"
    # Exit the script
    exit 0
}

# --- Trap command ---
# This tells the script to call the 'cleanup' function when it receives
# the INT signal (from Ctrl+C) or the TERM signal (from `kill` command).
trap cleanup INT TERM

# 1. Start the frontend HTTP server
echo "Starting frontend server..."
# Added -u flag to python command
(cd ./frontend && python -u -m http.server 8000 > ../$LOG_DIR/frontend.log 2>&1) &
echo $! > $PID_DIR/frontend.pid

# 2. Start the X post broadcaster
echo "Starting X post broadcaster..."
# Added -u flag to python command
(cd ./backend/X && python -u ./broadcast_posts.py > ../../$LOG_DIR/broadcast_posts.log 2>&1) &
echo $! > $PID_DIR/broadcast.pid

# 3. Start the MarketMaker decider
echo "Starting MarketMaker..."
# Added -u flag to python command
python -u ./backend/MarketMaker/decide_markets.py > $LOG_DIR/decide_markets.log 2>&1 &
echo $! > $PID_DIR/marketmaker.pid

# 4. Start the Exchange
echo "Starting Exchange..."
# Added -u flag to python command
python -u ./backend/Exchange/exchange_pool.py > $LOG_DIR/exchange_pool.log 2>&1 &
echo $! > $PID_DIR/exchange.pid

# 5. Start the Agent Manager
echo "Starting Agent Manager..."
# Added -u flag to python command
python -u ./backend/Agents/agent_manager.py > $LOG_DIR/agent_manager.log 2>&1 &
echo $! > $PID_DIR/agent_manager.pid


echo ""
echo "--- All services are running in the background ---"
echo "Logs are being written to the '$LOG_DIR' directory."
echo "Press Ctrl+C to stop all services."
echo ""

# --- Wait indefinitely ---
# Pauses the script until it's interrupted by Ctrl+C
wait