#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOUC_BIN="${SOUC_BIN:-./target/release/souc}"
FILE="${FILE:-examples/minimal.sio}"
GOLDEN_FILE="${GOLDEN_FILE:-tests/golden/minimal.hlir.golden.txt}"
TIMEOUT_SECS="${TIMEOUT_SECS:-180}"
BUILD_SOUC="${BUILD_SOUC:-1}"

run_with_timeout() {
  local seconds="$1"
  shift
  if command -v timeout >/dev/null 2>&1; then
    timeout --preserve-status "${seconds}s" "$@"
  else
    "$@"
  fi
}

if [ "$BUILD_SOUC" = "1" ]; then
  run_with_timeout 900 cargo build -p souc --release
fi

if [ ! -x "$SOUC_BIN" ]; then
  echo "compiler binary not found at $SOUC_BIN" >&2
  exit 1
fi

if [ ! -f "$FILE" ]; then
  echo "input file not found: $FILE" >&2
  exit 1
fi

mkdir -p "$(dirname "$GOLDEN_FILE")"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
if [ -f "$GOLDEN_FILE" ]; then
  backup_file="${GOLDEN_FILE}.bak.${timestamp}"
  cp "$GOLDEN_FILE" "$backup_file"
  echo "Backup created: $backup_file"
fi

tmp_file="$(mktemp /tmp/sounio-hlir-golden.XXXXXX)"
trap 'rm -f "$tmp_file"' EXIT

run_with_timeout "$TIMEOUT_SECS" "$SOUC_BIN" compile --emit hlir "$FILE" -v >"$tmp_file"
mv "$tmp_file" "$GOLDEN_FILE"

echo "Golden updated: $GOLDEN_FILE"
