#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEEP_WORK="${SOUNIO_MADAROS_DEREF_F64_GATE_KEEP:-0}"
SOURCE="$ROOT_DIR/tests/run-pass/imported_deref_f64_array.sio"

fail() {
  echo "[madaros-deref-f64] FAIL: $*" >&2
  exit 1
}

if [[ -n "${SOUNIO_MADAROS_DEREF_F64_GATE_DIR:-}" ]]; then
  WORK="$SOUNIO_MADAROS_DEREF_F64_GATE_DIR"
  [[ ! -e "$WORK" ]] || fail "refusing existing gate directory: $WORK"
  mkdir "$WORK" || fail "could not create gate directory: $WORK"
else
  WORK="$(mktemp -d /tmp/sounio-madaros-deref-f64.XXXXXX)"
fi

MADAROS_ELF="${SOUNIO_MADAROS_DEREF_F64_GATE_BIN:-$WORK/madaros}"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

if [[ -z "${SOUNIO_MADAROS_DEREF_F64_GATE_BIN:-}" ]]; then
  if ! bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_ELF" >"$WORK/build.log" 2>&1; then
    tail -n 80 "$WORK/build.log" >&2 || true
    fail "current-source Madaros build failed"
  fi
fi
[[ -x "$MADAROS_ELF" ]] || fail "Madaros is missing or not executable: $MADAROS_ELF"

set +e
MADAROS_RAW_BIN="$MADAROS_ELF" "$ROOT_DIR/bin/madaros" run "$SOURCE" >"$WORK/run.log" 2>&1
run_rc=$?
set -e

if [[ "$run_rc" != "0" ]]; then
  cat "$WORK/run.log" >&2
  fail "imported dereferenced f64-array witness exited rc=$run_rc"
fi
grep -Fxq 'PASS imported_deref_f64_array' "$WORK/run.log" || {
  cat "$WORK/run.log" >&2
  fail "exact PASS marker missing"
}

echo "[madaros-deref-f64] PASS: Madaros ELF preserves f64 element typing through imported dereference"
