#!/usr/bin/env bash
# Prove boot4 token recovery preserves crate/super visibility without weakening private access.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${SOUNIO_MADAROS_CRATE_VISIBILITY_BIN:-$ROOT_DIR/bin/madaros}"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
WORK="$(mktemp -d /tmp/sounio-madaros-crate-visibility.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

fail() {
  echo "[madaros-crate-visibility] FAIL: $*" >&2
  exit 1
}

count_marker() {
  local marker="$1"
  local log="$2"
  grep -Fc "$marker" "$log" || true
}

run_check() {
  local label="$1"
  local source="$2"
  local log="$WORK/$label.log"
  local rc=0
  set +e
  MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" check "$source" >"$log" 2>&1
  rc=$?
  set -e
  CHECK_RC="$rc"
  CHECK_LOG="$log"
}

expect_clean() {
  local label="$1"
  local source="$2"
  run_check "$label" "$source"
  [[ "$CHECK_RC" -eq 0 ]] || {
    cat "$CHECK_LOG" >&2
    fail "$label returned rc=$CHECK_RC"
  }
  [[ "$(count_marker 'error[E' "$CHECK_LOG")" -eq 0 ]] || fail "$label emitted diagnostics with rc=0"
  grep -Fq 'check: OK' "$CHECK_LOG" || fail "$label omitted the check success marker"
}

expect_private() {
  local label="$1"
  local source="$2"
  local code="$3"
  run_check "$label" "$source"
  [[ "$CHECK_RC" -eq 1 ]] || {
    cat "$CHECK_LOG" >&2
    fail "$label must reject with rc=1, got rc=$CHECK_RC"
  }
  [[ "$(count_marker 'error[E' "$CHECK_LOG")" -eq 1 ]] || fail "$label must emit exactly one diagnostic"
  [[ "$(count_marker "error[$code" "$CHECK_LOG")" -eq 1 ]] || fail "$label must emit exactly one $code"
}

[[ -x "$MADAROS" ]] || fail "Madaros wrapper is missing or not executable: $MADAROS"
[[ -n "$RAW_MADAROS" ]] || fail 'MADAROS_RAW_BIN must name an explicit current-source Madaros ELF'
[[ -x "$RAW_MADAROS" ]] || fail "current-source Madaros is missing or not executable: $RAW_MADAROS"

FIXTURES="$ROOT_DIR/tests/compiler/madaros_visibility_context"
expect_clean pub-crate-super "$FIXTURES/pub_crate_main.sio"
expect_clean serialize-internal-api "$ROOT_DIR/self-hosted/ir/serialize.sio"
expect_private private-fn "$ROOT_DIR/tests/multimodule/visibility_fn_private_main.sio" E175
expect_private private-struct "$ROOT_DIR/tests/multimodule/visibility_struct_private_main.sio" E176
expect_private private-enum "$ROOT_DIR/tests/multimodule/visibility_enum_private_main.sio" E177

echo '[madaros-crate-visibility] receipt pub_crate=PASS pub_super=PASS serialize_internal_api=PASS private_fn=E175 private_struct=E176 private_enum=E177'
