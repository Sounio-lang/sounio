#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

SOURCE="tests/run-pass/brain_ossm_sigmoid_polarity.sio"
OUT_DIR="${SOUNIO_BRAIN_OSSM_SIGMOID_GATE_DIR:-$(mktemp -d /tmp/sounio-brain-ossm-sigmoid.XXXXXX)}"
BIN="$OUT_DIR/brain_ossm_sigmoid_polarity"
COMPILE_LOG="$OUT_DIR/compile.log"
STDOUT_LOG="$OUT_DIR/stdout.log"
STDERR_LOG="$OUT_DIR/stderr.log"

mkdir -p "$OUT_DIR"

echo "[brain-ossm-sigmoid] source=$SOURCE"
echo "[brain-ossm-sigmoid] out=$OUT_DIR"

SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" "$SOURCE" "$BIN" >"$COMPILE_LOG" 2>&1
chmod +x "$BIN"
"$BIN" >"$STDOUT_LOG" 2>"$STDERR_LOG"

if ! grep -q "BRAIN_OSSM_SIGMOID_POLARITY_PASS" "$STDOUT_LOG"; then
  echo "[brain-ossm-sigmoid] FAIL: pass token missing" >&2
  echo "--- stdout ---" >&2
  cat "$STDOUT_LOG" >&2
  echo "--- stderr ---" >&2
  cat "$STDERR_LOG" >&2
  exit 1
fi

echo "[brain-ossm-sigmoid] PASS: exp_q reciprocal symmetry and sigmoid polarity hold"
