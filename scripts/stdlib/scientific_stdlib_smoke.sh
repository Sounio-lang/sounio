#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-./target/debug/souc}"

run_case() {
  local file="$1"
  local expected_marker="$2"

  echo "SMOKE: running $file"
  local log
  log="$(mktemp)"
  trap 'rm -f "$log"' RETURN

  "$SOUC_BIN" run "$file" >"$log" 2>&1

  if ! grep -q "$expected_marker" "$log"; then
    echo "SMOKE FAIL: expected marker '$expected_marker' not found for $file"
    cat "$log"
    exit 1
  fi

  echo "SMOKE PASS: $file"
  rm -f "$log"
  trap - RETURN
}

echo "Building souc binary for scientific stdlib smoke..."
cargo build -p souc --bin souc >/dev/null

run_case "examples/pbpk_simple.sio" "example: pbpk_simple"
run_case "examples/linalg_demo.sio" "example: linalg_demo"
run_case "examples/ode_demo.sio" "example: ode_demo"

echo "Scientific stdlib smoke checks passed."
