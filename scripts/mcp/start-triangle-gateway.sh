#!/usr/bin/env bash
# start-triangle-gateway.sh — Start LiteLLM triangle gateway on port 4000
# Run this once per machine boot (or add to ~/.bashrc / systemd unit)

set -euo pipefail
source ~/.sounio-keys.env

PID_FILE="${HOME}/.claude/litellm.pid"

# Stop any existing instance
if [[ -f "$PID_FILE" ]]; then
  old_pid=$(cat "$PID_FILE")
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "[triangle-gateway] Stopping existing instance (PID $old_pid)..."
    kill "$old_pid"
  fi
  rm -f "$PID_FILE"
fi

echo "[triangle-gateway] Starting LiteLLM on port 4000..."
nohup litellm \
  --config "$(dirname "$0")/litellm-triangle.yaml" \
  --port 4000 \
  > "${HOME}/.claude/litellm.log" 2>&1 &

echo $! > "$PID_FILE"
echo "[triangle-gateway] PID $! — log at ~/.claude/litellm.log"
sleep 4
curl -sf http://localhost:4000/health | python3 -m json.tool 2>/dev/null || echo "[triangle-gateway] Health check failed — see log"
