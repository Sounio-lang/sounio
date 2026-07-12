#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/zero-event.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

MODULE="$ROOT_DIR/stdlib/epistemic/zero_event.sio"
POSITIVE="$ROOT_DIR/tests/known_failures/zero_event_stdlib_native_v2_probe.sio"
EISA_FLAGS="$ROOT_DIR/tests/known_failures/eisa_zero_flags_native_v2_probe.sio"

fail() {
  echo "[zero-event] FAIL: $*" >&2
  exit 1
}

expect_check_fail() {
  local source="$1"
  local pattern="$2"
  local name="$3"
  local log="$TMP_DIR/$name.log"

  if "$ROOT_DIR/bin/souc" check "$source" >"$log" 2>&1; then
    cat "$log" >&2
    fail "$name unexpectedly passed Madaros check"
  fi
  grep -Fq "$pattern" "$log" || {
    cat "$log" >&2
    fail "$name failed without expected marker: $pattern"
  }
}

"$ROOT_DIR/bin/souc" check "$MODULE" >"$TMP_DIR/module.log" 2>&1 || {
  cat "$TMP_DIR/module.log" >&2
  fail "stdlib module check failed"
}

"$ROOT_DIR/bin/souc" check "$POSITIVE" >"$TMP_DIR/positive-check.log" 2>&1 || {
  cat "$TMP_DIR/positive-check.log" >&2
  fail "positive witness check failed"
}

output="$(SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" run "$POSITIVE" 2>&1)" || {
  printf '%s\n' "$output" >&2
  fail "positive witness lean_single execution failed"
}
printf '%s\n' "$output" | grep -Fq 'ZERO_EVENT_STDLIB PASS' || {
  printf '%s\n' "$output" >&2
  fail "positive witness missing pass marker"
}

"$ROOT_DIR/bin/souc" check "$EISA_FLAGS" >"$TMP_DIR/eisa-check.log" 2>&1 || {
  cat "$TMP_DIR/eisa-check.log" >&2
  fail "EISA zero-flags check failed"
}

eisa_output="$(SOUNIO_SOUC_ENGINE=lean_single "$ROOT_DIR/bin/souc" run "$EISA_FLAGS" 2>&1)" || {
  printf '%s\n' "$eisa_output" >&2
  fail "EISA zero-flags lean_single execution failed"
}
printf '%s\n' "$eisa_output" | grep -Fq 'EISA_ZERO_FLAGS PASS' || {
  printf '%s\n' "$eisa_output" >&2
  fail "EISA zero-flags witness missing pass marker"
}

expect_check_fail \
  "$ROOT_DIR/tests/compile-fail/zero_event_direct_receipt_construction.sio" \
  'error[E176' \
  'receipt-constructor-opacity'

expect_check_fail \
  "$ROOT_DIR/tests/compile-fail/zero_event_direct_erased_construction.sio" \
  'error[E176' \
  'erased-constructor-opacity'

expect_check_fail \
  "$ROOT_DIR/tests/compile-fail/zero_event_erasure_requires_discharge.sio" \
  'expected ErasedZeroF64' \
  'explicit-discharge-type-boundary'

echo '[zero-event] PASS: receipts, evidence, explicit discharge, and derived EISA flags are verified'
