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

run_capture qd128 tests/run-pass/qd128_import_native_v2_smoke.sio
require_rc qd128 0
require_marker qd128 'QD128_IMPORT_NATIVE_V2 PASS'
reject_marker qd128 'Segmentation fault'

run_capture qd128_mul tests/run-pass/qd128_mul_native_v2_smoke.sio
require_rc qd128_mul 0
require_marker qd128_mul 'QD128_MUL_NATIVE_V2 PASS'
reject_marker qd128_mul 'Segmentation fault'

# Compact zero-provenance (sedenion + local f64 kinds) — Madaros-green
run_capture zp_compact tests/run-pass/zero_provenance_native_v2_smoke.sio
require_rc zp_compact 0
require_marker zp_compact 'ZERO_PROVENANCE PASS'
reject_marker zp_compact 'Segmentation fault'

# Full eisa::core_v2 + sedenion combined import still thin-link fail-closed
run_capture combined tests/known_failures/zero_provenance_native_v2_probe.sio
require_rc combined 1
require_marker combined 'Failed to write native binary'
reject_marker combined 'Segmentation fault'
reject_marker combined 'ZERO_PROVENANCE PASS'

# Bool-cmp-in-struct-field shape residual (not an IR fn-count ceiling)
run_capture bool_cmp_precomp tests/run-pass/thinlink_bool_cmp_field_precomp_smoke.sio
require_rc bool_cmp_precomp 0
require_marker bool_cmp_precomp 'BOOL_CMP_FIELD_PRECOMP PASS'
reject_marker bool_cmp_precomp 'Segmentation fault'

run_capture bool_cmp_probe tests/known_failures/thinlink_bool_cmp_field_probe.sio
require_rc bool_cmp_probe 1
require_marker bool_cmp_probe 'Failed to write native binary'
reject_marker bool_cmp_probe 'Segmentation fault'
reject_marker bool_cmp_probe 'BOOL_CMP_FIELD PASS'

# Receipt stdlib probe is green on stock Madaros (main already closed this surface)
run_capture receipt tests/known_failures/zero_event_stdlib_native_v2_probe.sio
require_rc receipt 0
require_marker receipt 'Compilation successful!'
require_marker receipt 'ZERO_EVENT_STDLIB PASS'
reject_marker receipt 'Segmentation fault'

echo '[zero-native-matrix] PASS: dd64+sedenion+qd128(+core,+mul)+zp_compact+bool_cmp_precomp+receipt green; eisa+sedenion combined + bool_cmp_probe fail-closed'
