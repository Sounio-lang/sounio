#!/usr/bin/env bash
# darwin-sglang-launch.sh — Start SGLang model servers on DARWIN
#
# Hardware:
#   GPU 0: L4 24GB       → Qwen 2.5 Coder 14B (general code offload, port 30010)
#   GPU 1: RTX 4000 Ada  → Qwen 2.5 7B (scout/classifier, port 30011)
#
# Usage: bash scripts/mcp/darwin-sglang-launch.sh [start|stop|status]
#
# Models are loaded from HuggingFace on first run.
# Recommended: run inside a tmux session on DARWIN.

set -euo pipefail

CODER_MODEL="${CODER_MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct}"
SCOUT_MODEL="${SCOUT_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
CODER_PORT=30010
SCOUT_PORT=30011
LOG_DIR="${HOME}/.darwin-sglang-logs"

mkdir -p "$LOG_DIR"

cmd="${1:-start}"

case "$cmd" in
  start)
    echo "[darwin-sglang] Starting coder model on GPU 0 (port $CODER_PORT)..."
    CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
      --model "$CODER_MODEL" \
      --port "$CODER_PORT" \
      --tp-size 1 \
      --host 0.0.0.0 \
      --served-model-name darwin-coder \
      > "$LOG_DIR/coder.log" 2>&1 &
    echo $! > "$LOG_DIR/coder.pid"
    echo "[darwin-sglang] coder PID $(cat "$LOG_DIR/coder.pid")"

    echo "[darwin-sglang] Starting scout model on GPU 1 (port $SCOUT_PORT)..."
    CUDA_VISIBLE_DEVICES=1 python -m sglang.launch_server \
      --model "$SCOUT_MODEL" \
      --port "$SCOUT_PORT" \
      --tp-size 1 \
      --host 0.0.0.0 \
      --served-model-name darwin-scout \
      > "$LOG_DIR/scout.log" 2>&1 &
    echo $! > "$LOG_DIR/scout.pid"
    echo "[darwin-sglang] scout PID $(cat "$LOG_DIR/scout.pid")"

    echo "[darwin-sglang] Waiting 30s for servers to initialise..."
    sleep 30
    bash "$0" status
    ;;

  stop)
    for model in coder scout; do
      pid_file="$LOG_DIR/${model}.pid"
      if [[ -f "$pid_file" ]]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
          kill "$pid"
          echo "[darwin-sglang] Stopped $model (PID $pid)"
        else
          echo "[darwin-sglang] $model not running (stale PID $pid)"
        fi
        rm -f "$pid_file"
      else
        echo "[darwin-sglang] No PID file for $model"
      fi
    done
    ;;

  status)
    for model in coder scout; do
      port=$( [[ "$model" == "coder" ]] && echo $CODER_PORT || echo $SCOUT_PORT )
      if curl -sf "http://localhost:${port}/health" > /dev/null 2>&1; then
        echo "[darwin-sglang] $model: READY on port $port"
      else
        echo "[darwin-sglang] $model: NOT READY on port $port"
      fi
    done
    ;;

  *)
    echo "Usage: $0 [start|stop|status]"
    exit 1
    ;;
esac
