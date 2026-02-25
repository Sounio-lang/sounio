#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEST_FILE="$ROOT_DIR/tests/hardware/real_ptx_launch_test.sio"
PASS_MARKER="real_ptx_launch_test PASS"
DEFAULT_DEBUG_SOUC="$ROOT_DIR/target/debug/souc"
DEFAULT_RELEASE_SOUC="$ROOT_DIR/target/release/souc"

if [ ! -f "$TEST_FILE" ]; then
  echo "error: missing PTX launch self-check test: $TEST_FILE" >&2
  exit 2
fi

resolve_souc_bin() {
  if [ -n "${SOUC_BIN:-}" ]; then
    echo "$SOUC_BIN"
    return
  fi
  if [ -x "$DEFAULT_DEBUG_SOUC" ]; then
    echo "$DEFAULT_DEBUG_SOUC"
    return
  fi
  if [ -x "$DEFAULT_RELEASE_SOUC" ]; then
    echo "$DEFAULT_RELEASE_SOUC"
    return
  fi
  echo "souc"
}

SOUC_BIN="$(resolve_souc_bin)"

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
