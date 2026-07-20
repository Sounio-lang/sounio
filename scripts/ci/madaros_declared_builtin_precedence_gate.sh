#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="$ROOT_DIR/bin/madaros"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"
WORK="$(mktemp -d /tmp/sounio-madaros-declared-builtin-precedence.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT

fail() {
  echo "[madaros-declared-builtin-precedence] FAIL: $*" >&2
  exit 1
}

[[ -x "$MADAROS" ]] || fail "Madaros launcher is missing: $MADAROS"
[[ -n "$RAW_MADAROS" ]] || fail 'MADAROS_RAW_BIN must name an explicit current-source Madaros ELF'
[[ -x "$RAW_MADAROS" ]] || fail "current-source Madaros is not executable: $RAW_MADAROS"
[[ "$(head -c 2 "$RAW_MADAROS" 2>/dev/null)" != '#!' ]] || fail 'MADAROS_RAW_BIN must be a raw ELF, not a wrapper'

export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

run_witness() {
  local label="$1"
  local source="$2"
  local marker="${3:-}"
  local log="$WORK/$label.log"

  if ! MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" run "$source" >"$log" 2>&1; then
    cat "$log" >&2
    fail "$label did not compile and run"
  fi
  if [[ -n "$marker" ]] && ! grep -Fq "$marker" "$log"; then
    cat "$log" >&2
    fail "$label did not emit its pass marker"
  fi
  printf '[madaros-declared-builtin-precedence] PASS: %s\n' "$label"
}

MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" --version
run_witness declared-local \
  "$ROOT_DIR/tests/packages/measure_declared_precedence_local.sio" \
  'declared measure precedence: passed'
run_witness intrinsic-fallback \
  "$ROOT_DIR/tests/packages/measure_intrinsic_fallback.sio"
run_witness deref-value-field-f64 \
  "$ROOT_DIR/tests/packages/deref_value_field_f64.sio" \
  'deref value field f64: passed'
run_witness contextual-level-struct-literal \
  "$ROOT_DIR/tests/packages/contextual_level_struct_literal.sio" \
  'contextual level literal: passed'

MADAROS_RAW_BIN="$RAW_MADAROS" \
  bash "$ROOT_DIR/scripts/ci/package_import_science_gate.sh"

echo '[madaros-declared-builtin-precedence] PASS: declared functions override intrinsic fallbacks and package imports remain gated'
