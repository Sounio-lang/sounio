#!/usr/bin/env bash
# string_literal_truncation_gate.sh — refuse silent string-literal truncation.
#
# ENGINE: Madaros (default bin/souc after source build). lean_single is not
# the contract surface for this bug (measured on Madaros).
#
# Measured boundary on prebuilt Madaros (2026-08-17):
#   content <= 128 prints full; content >= 129 silently prints 128; rc=0.
# Positive control: 127 and 128 print full length.
# Fix witness: 129 and 200 print full length (Name capacity 384).
# Honesty: oversize literal must error[E258], never shorten quietly.
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

unset SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
if [[ ! -x "$SOUC" ]]; then
  echo "FAIL souc not executable: $SOUC" >&2
  exit 2
fi
if [[ "${SOUNIO_SOUC_ENGINE:-}" == "lean_single" ]]; then
  echo "FAIL this gate asserts Madaros string emission; refuse lean_single" >&2
  exit 2
fi

echo "=== string_literal_truncation_gate ==="
echo "engine=Madaros"
echo "souc=$SOUC"
echo "souc_version=$("$SOUC" --version 2>/dev/null | head -1 || echo unknown)"
if [[ -n "${MADAROS_RAW_BIN:-}" ]]; then
  echo "MADAROS_RAW_BIN=$MADAROS_RAW_BIN"
fi

TMP=$(mktemp -d "${TMPDIR:-/tmp}/strlit-gate.XXXXXX")
trap 'rm -f "$TMP"/*; rmdir "$TMP" 2>/dev/null || true' EXIT

PASS=0
FAIL=0
fail() { FAIL=$((FAIL+1)); echo "FAIL $1" >&2; }
pass() { PASS=$((PASS+1)); echo "PASS $1"; }

count_pure_line() {
  local ch="$2"
  python3 - "$1" "$ch" <<'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(errors="replace")
ch = sys.argv[2]
for line in text.splitlines():
    s = line.lstrip('"').rstrip('"')
    if len(s) >= 100 and set(s) <= {ch}:
        print(len(s))
        raise SystemExit(0)
print(0)
PY
}

run_expect_len() {
  local src="$1"
  local expect="$2"
  local label="$3"
  local ch="${4:-A}"
  local log="$TMP/${label}.log"
  if ! "$SOUC" run "$ROOT/$src" >"$log" 2>&1; then
    if grep -Fq 'error[E258]' "$log"; then
      fail "$label: unexpected E258 (literal should fit Name cap)"
      tail -15 "$log" >&2 || true
      return
    fi
    fail "$label: souc run failed"
    tail -20 "$log" >&2 || true
    return
  fi
  local got
  got="$(count_pure_line "$log" "$ch")"
  if [[ "$got" != "$expect" ]]; then
    fail "$label: printed_len=$got expected=$expect (silent truncation?)"
    rg -n "${ch}{20,}|TRUNCATED|E258|PASS " "$log" | head -20 >&2 || true
    return
  fi
  if ! grep -Eq "PASS string_literal_len" "$log"; then
    fail "$label: missing PASS marker"
    return
  fi
  pass "$label:printed_len=$got"
}

run_expect_e258() {
  local src="$1"
  local label="$2"
  local log="$TMP/${label}.log"
  set +e
  "$SOUC" check "$ROOT/$src" >"$log" 2>&1
  local rc=$?
  set -e
  if grep -Fq 'error[E258]' "$log"; then
    pass "$label:E258"
    return
  fi
  if [[ "$rc" -eq 0 ]]; then
    fail "$label: compiled OK without E258 (must refuse oversize)"
  else
    fail "$label: failed without error[E258]"
  fi
  tail -25 "$log" >&2 || true
}

# Positive controls (must fire on prebuilt AND fixed)
run_expect_len tests/run-pass/string_literal_len127_print.sio 127 string_literal_len127_print A
run_expect_len tests/run-pass/string_literal_len128_print.sio 128 string_literal_len128_print A

# Primary bug witnesses (FAIL on prebuilt: 129→128, 200→128)
run_expect_len tests/run-pass/string_literal_len129_print.sio 129 string_literal_len129_print A
run_expect_len tests/run-pass/string_literal_len200_print.sio 200 string_literal_len200_print B

# Honesty: refuse rather than truncate past Name capacity
run_expect_e258 tests/compile-fail/string_literal_oversize_e258.sio string_literal_oversize_e258

echo "---"
echo "PASS_COUNT=$PASS FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS string_literal_truncation_gate"
  exit 0
fi
echo "FAIL string_literal_truncation_gate" >&2
exit 1
