#!/usr/bin/env bash
# Frontier gate for the Madaros "no caveats" lane.
#
# This is intentionally a ratchet, not a declaration of completion: it preserves
# the current proven ceiling while the remaining stack-frame warnings and
# self-check E001 diagnostics are driven down.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MAX_STACK_WARNINGS="${MADAROS_NO_CAVEATS_MAX_STACK_WARNINGS:-43}"
MAX_E001_DIAGNOSTICS="${MADAROS_NO_CAVEATS_MAX_E001_DIAGNOSTICS:-0}"
OUT="${MADAROS_NO_CAVEATS_OUT:-$(mktemp /tmp/sounio-madaros-no-caveats.XXXXXX)}"
LOG="${MADAROS_NO_CAVEATS_LOG:-$(mktemp /tmp/sounio-madaros-no-caveats-log.XXXXXX)}"

fail() {
  echo "[madaros-no-caveats] FAIL: $*" >&2
  if [[ -f "$LOG" ]]; then
    echo "[madaros-no-caveats] log tail:" >&2
    tail -n 80 "$LOG" >&2 || true
  fi
  exit 1
}

count_or_zero() {
  local pattern="$1"
  local file="$2"
  local count
  count="$(rg -c "$pattern" "$file" 2>/dev/null || true)"
  if [[ -z "$count" ]]; then
    count=0
  fi
  printf '%s' "$count"
}

echo "[madaros-no-caveats] out=$OUT"
echo "[madaros-no-caveats] log=$LOG"
echo "[madaros-no-caveats] max_stack_warnings=$MAX_STACK_WARNINGS"
echo "[madaros-no-caveats] max_E001_diagnostics=$MAX_E001_DIAGNOSTICS"

bash scripts/ci/build_modular_madaros.sh "$OUT" >"$LOG" 2>&1

stack_warnings="$(count_or_zero '^warning: stack frame too large' "$LOG")"
e001_diagnostics="$(count_or_zero '^error: error\[E001\]' "$LOG")"

echo "[madaros-no-caveats] stack_warnings=$stack_warnings"
echo "[madaros-no-caveats] E001_diagnostics=$e001_diagnostics"

if (( stack_warnings > MAX_STACK_WARNINGS )); then
  fail "stack warnings regressed: got $stack_warnings, max $MAX_STACK_WARNINGS"
fi

if (( e001_diagnostics > MAX_E001_DIAGNOSTICS )); then
  fail "E001 diagnostics regressed: got $e001_diagnostics, max $MAX_E001_DIAGNOSTICS"
fi

if [[ ! -s "$OUT" ]]; then
  fail "Madaros output missing or empty: $OUT"
fi

echo "[madaros-no-caveats] PASS: frontier preserved"
