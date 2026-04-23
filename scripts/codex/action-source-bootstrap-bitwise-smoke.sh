#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

SEED=""
for candidate in \
  "$ROOT_DIR/artifacts/bootstrap/boot4.elf" \
  "$ROOT_DIR/artifacts/bootstrap/final_boot4.elf" \
  "$ROOT_DIR/bootstrap/stage0"
do
  if [[ -x "$candidate" ]]; then
    SEED="$candidate"
    break
  fi
done

if [[ -z "$SEED" ]]; then
  echo "error: no bootstrap seed compiler found" >&2
  exit 1
fi

STAGE1="$TMP_DIR/boot4-stage1"
STAGE2="$TMP_DIR/boot4-stage2"
OUT_BIN="$TMP_DIR/bitwise-not-bootstrap-smoke"
LOG1="$TMP_DIR/boot4-stage1.log"
LOG2="$TMP_DIR/boot4-stage2.log"
LOG3="$TMP_DIR/bitwise-not-bootstrap-smoke.log"

# This specifically guards the bootstrap compiler's own parser/codegen against
# dropping unary `~` in compiler source expressions like `a & ~b`.
if ! "$SEED" bootstrap/boot4.sio "$STAGE1" >"$LOG1" 2>&1; then
  tail -n 80 "$LOG1" >&2 || true
  exit 1
fi
chmod +x "$STAGE1"

if ! "$STAGE1" bootstrap/boot4.sio "$STAGE2" >"$LOG2" 2>&1; then
  tail -n 80 "$LOG2" >&2 || true
  exit 1
fi
chmod +x "$STAGE2"

if ! "$STAGE2" tests/run-pass/bitwise_not_bootstrap_regression.sio "$OUT_BIN" >"$LOG3" 2>&1; then
  tail -n 80 "$LOG3" >&2 || true
  exit 1
fi
chmod +x "$OUT_BIN"

set +e
"$OUT_BIN"
RC=$?
set -e

if [[ "$RC" -ne 0 ]]; then
  echo "error: bitwise_not_bootstrap_regression returned $RC, expected 0" >&2
  exit 1
fi

echo "SOURCE_BOOTSTRAP_BITWISE_SMOKE_PASS seed=$(basename "$SEED")"
