#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true

MADAROS="${SOUNIO_MADAROS_BIN:-$ROOT/artifacts/self-hosted/madaros}"
SOUC="${SOUC:-$ROOT/bin/souc}"
if [[ ! -x "$MADAROS" ]]; then
  echo "FAIL: current-source Madaros not executable: $MADAROS" >&2
  exit 1
fi
export SOUNIO_MADAROS_BIN="$MADAROS"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== madaros_kl1_parity_small_gate =="

run_case() {
  local src="$1" sentinel="${2:-}" name
  name="$(basename "$src" .sio)"
  "$SOUC" compile "$src" -o "$TMP/$name" >"$TMP/$name.compile.log" 2>&1 || {
    echo "FAIL: compile $src" >&2
    cat "$TMP/$name.compile.log" >&2
    exit 1
  }
  chmod +x "$TMP/$name"
  "$TMP/$name" >"$TMP/$name.run.log" 2>&1 || {
    local rc=$?
    echo "FAIL: run $src (rc=$rc)" >&2
    cat "$TMP/$name.run.log" >&2
    exit 1
  }
  if [[ -n "$sentinel" ]] && ! grep -Fq "$sentinel" "$TMP/$name.run.log"; then
    echo "FAIL: missing '$sentinel' from $src" >&2
    cat "$TMP/$name.run.log" >&2
    exit 1
  fi
}

refuse_case() {
  local src="$1" pattern="$2" name
  name="$(basename "$src" .sio)"
  if "$SOUC" check "$src" >"$TMP/$name.check.log" 2>&1; then
    echo "FAIL: expected refusal for $src" >&2
    cat "$TMP/$name.check.log" >&2
    exit 1
  fi
  grep -Fq "$pattern" "$TMP/$name.check.log" || {
    echo "FAIL: refusal for $src lacked '$pattern'" >&2
    cat "$TMP/$name.check.log" >&2
    exit 1
  }
}

run_case tests/run-pass/bitwise_not_bootstrap_regression.sio
run_case tests/run-pass/kl1_bitwise_not_madaros.sio KL1_BITNOT_OK
run_case tests/run-pass/kl1_epistemic_positive_payload.sio KL1_EPISTEMIC_PAYLOAD_OK
run_case tests/run-pass/kl1_raw_ptr_mut_to_const.sio KL1_RAW_PTR_OK
run_case tests/run-pass/kl1_println_computed_local.sio KL1_PRINTLN_COMPUTED_OK

refuse_case tests/compile-fail/kl1_unary_type_errors.sio E005
refuse_case tests/compile-fail/kl1_epistemic_negative_payload.sio "effect payload must be a non-negative integer literal"
refuse_case tests/compile-fail/kl1_raw_ptr_const_to_mut.sio E009

echo "MADAROS_KL1_PARITY_SMALL_GATE_OK"
