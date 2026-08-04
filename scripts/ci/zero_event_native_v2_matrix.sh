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

# Positive controls under default Madaros native-v2
run_capture dd64 tests/run-pass/dd64_import_native_v2_smoke.sio
require_rc dd64 0
require_marker dd64 'DD64_IMPORT_NATIVE_V2 PASS'

run_capture sedenion tests/run-pass/sedenion_import_native_v2_smoke.sio
require_rc sedenion 0
require_marker sedenion 'Compilation successful!'
require_marker sedenion 'SEDENION_IMPORT_NATIVE_V2 PASS'
reject_marker sedenion 'Segmentation fault'

run_capture qd128_core tests/run-pass/qd128_core_import_native_v2_smoke.sio
require_rc qd128_core 0
require_marker qd128_core 'QD128_CORE_IMPORT_NATIVE_V2 PASS'
reject_marker qd128_core 'Segmentation fault'

# Fail-closed residuals: full math::qd128 still pulls qd_mul → ELF write rc=12
run_capture qd128 tests/known_failures/qd128_import_native_v2_probe.sio
require_rc qd128 1
require_marker qd128 'lower_array: into_acc_done 2'
require_marker qd128 'Failed to write native binary'
reject_marker qd128 'Segmentation fault'
reject_marker qd128 'QD128_IMPORT_NATIVE_V2 PASS'

run_capture combined tests/known_failures/zero_provenance_native_v2_probe.sio
require_rc combined 1
require_marker combined 'lower_array: final_fn_count'
require_marker combined 'Failed to write native binary'
reject_marker combined 'Segmentation fault'
reject_marker combined 'ZERO_PROVENANCE PASS'

# Receipt constructors still unsupported under native-v2 (lean_single oracle in zero_event_gate.sh)
run_capture receipt tests/known_failures/zero_event_stdlib_native_v2_probe.sio
require_rc receipt 1
require_marker receipt 'Failed to write native binary'
reject_marker receipt 'Segmentation fault'
reject_marker receipt 'ZERO_EVENT_STDLIB PASS'

echo '[zero-native-matrix] PASS: dd64+sedenion+qd128_core green; full qd128/combined/receipt fail closed (rc=12) without segfault'
