#!/usr/bin/env bash
# Prove that self-hosted diagnostics retain their source authority context.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MADAROS="${SOUNIO_MADAROS_DIAGNOSTIC_CONTEXT_BIN:-$ROOT_DIR/bin/madaros}"
RAW_MADAROS="${MADAROS_RAW_BIN:-}"

fail() {
  echo "[madaros-diagnostic-context] FAIL: $*" >&2
  exit 1
}

[[ -x "$MADAROS" ]] || fail "Madaros wrapper is missing or not executable: $MADAROS"
[[ -n "$RAW_MADAROS" ]] || fail "MADAROS_RAW_BIN must name an explicit current-source Madaros ELF"
[[ -x "$RAW_MADAROS" ]] || fail "explicit current-source Madaros is missing or not executable: $RAW_MADAROS"

WORK="$(mktemp -d /tmp/sounio-madaros-diagnostic-context.XXXXXX)"
trap 'rm -rf "$WORK"' EXIT
LOG="$WORK/effect-multiple-missing.log"

set +e
MADAROS_RAW_BIN="$RAW_MADAROS" "$MADAROS" check \
  "$ROOT_DIR/tests/compile-fail/effect_multiple_missing.sio" >"$LOG" 2>&1
rc=$?
set -e

grep -Eiq 'segmentation fault|core dumped|terminated by signal|fatal:|bus error|illegal instruction' "$LOG" && {
  cat "$LOG" >&2
  fail "E035 witness produced a fatal compiler/runtime log"
}
[[ "$rc" -eq 1 ]] || {
  cat "$LOG" >&2
  fail "E035 witness must reject with rc=1, got rc=$rc"
}
[[ "$(grep -Fc 'error[E' "$LOG" || true)" -eq 1 ]] || {
  cat "$LOG" >&2
  fail "E035 witness must emit exactly one compiler diagnostic"
}
[[ "$(grep -Ecx 'error\[E035\] in <main>::only_io( at [0-9]+\.\.[0-9]+)?: effect not declared in function signature \(missing: Mut\)' "$LOG" || true)" -eq 1 ]] || {
  cat "$LOG" >&2
  fail "E035 must name <main>::only_io and the exact missing Mut effect once"
}
grep -Fq 'run_check_mode: verdict=1' "$LOG" || {
  cat "$LOG" >&2
  fail "E035 witness rejected without the checker verdict receipt"
}

echo '[madaros-diagnostic-context] receipt E035=<main>::only_io missing=Mut diagnostics=1'
echo '[madaros-diagnostic-context] PASS: effect diagnostics preserve module, function, and missing-effect identity'
