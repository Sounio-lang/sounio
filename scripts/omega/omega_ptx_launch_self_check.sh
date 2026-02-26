#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_FILE="$ROOT_DIR/tests/hardware/real_ptx_launch_test.sio"
PASS_MARKER="real_ptx_launch_test PASS"
source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

if [ ! -f "$TEST_FILE" ]; then
  echo "error: missing PTX launch self-check test: $TEST_FILE" >&2
  exit 2
fi

sounio_require_souc

set +e
run_output="$(PATH="$ROOT_DIR:$PATH" "$SOUC_BIN" run "$TEST_FILE" 2>&1)"
run_rc=$?
set -e

if [ "$run_rc" -ne 0 ]; then
  echo "error: souc run failed for PTX launch self-check (exit=$run_rc bin=$SOUC_BIN)" >&2
  printf '%s\n' "$run_output" >&2
  exit "$run_rc"
fi

if ! printf '%s\n' "$run_output" | rg -q "$PASS_MARKER"; then
  echo "error: PTX launch self-check did not emit required marker: $PASS_MARKER" >&2
  printf '%s\n' "$run_output" >&2
  exit 2
fi

printf '%s\n' "$run_output"
