#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/zero-native-matrix.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
  echo "[zero-native-matrix] FAIL: $*" >&2
  exit 1
}

run_capture() {
  local name="$1"
  local source="$2"
  set +e
  "$ROOT_DIR/bin/souc" run "$ROOT_DIR/$source" >"$TMP_DIR/$name.log" 2>&1
  local rc=$?
  set -e
  printf '%s' "$rc" >"$TMP_DIR/$name.rc"
}

require_rc() {
  local name="$1"
  local expected="$2"
  local actual
  actual="$(cat "$TMP_DIR/$name.rc")"
  [[ "$actual" == "$expected" ]] || {
    cat "$TMP_DIR/$name.log" >&2
    fail "$name expected rc=$expected, got rc=$actual"
  }
}

require_marker() {
  local name="$1"
  local marker="$2"
  grep -Fq "$marker" "$TMP_DIR/$name.log" || {
    cat "$TMP_DIR/$name.log" >&2
    fail "$name missing marker: $marker"
  }
}

reject_marker() {
  local name="$1"
  local marker="$2"
  if grep -Fq "$marker" "$TMP_DIR/$name.log"; then
    cat "$TMP_DIR/$name.log" >&2
    fail "$name unexpectedly contained marker: $marker"
  fi
}

run_capture dd64 tests/run-pass/dd64_import_native_v2_smoke.sio
require_rc dd64 0
require_marker dd64 'DD64_IMPORT_NATIVE_V2 PASS'

run_capture qd128 tests/known_failures/qd128_import_native_v2_probe.sio
require_rc qd128 1
require_marker qd128 'lower_array: dep_lower_done 2'
require_marker qd128 'Failed to write native binary'
reject_marker qd128 'Segmentation fault'

run_capture sedenion tests/known_failures/sedenion_import_native_v2_probe.sio
require_rc sedenion 1
require_marker sedenion 'Compilation successful!'
reject_marker sedenion 'Segmentation fault'
reject_marker sedenion 'SEDENION_IMPORT_NATIVE_V2 PASS'

run_capture combined tests/known_failures/zero_provenance_native_v2_probe.sio
require_rc combined 1
require_marker combined 'lower_array: final_fn_count'
require_marker combined 'Failed to write native binary'
reject_marker combined 'Segmentation fault'

run_capture receipt tests/known_failures/zero_event_stdlib_native_v2_probe.sio
require_rc receipt 0
require_marker receipt 'Compilation successful!'
require_marker receipt 'ZERO_EVENT_STDLIB PASS'
reject_marker receipt 'Segmentation fault'

echo '[zero-native-matrix] PASS: dd64 and zero-event run; qd128/combined fail closed in backend; sedenion exits nonzero without crashing'
