#!/usr/bin/env bash
# Gate: module-level global arrays mutate correctly when passed by &! / aliased.
# Defect B in docs/audit/MADAROS_NATIVE_V2_F64_REMAINING_BUGS_2026-07-20.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

SOUC="${SOUC:-$ROOT/bin/souc}"
TEST="$ROOT/tests/run-pass/global_array_ref_mut.sio"

if [[ ! -x "$SOUC" ]]; then
  echo "FAIL: souc not executable at $SOUC" >&2
  exit 2
fi
if [[ ! -f "$TEST" ]]; then
  echo "FAIL: missing $TEST" >&2
  exit 2
fi

echo "== madaros_global_array_ref_gate: $TEST =="
out="$("$SOUC" run "$TEST" 2>&1)" || {
  echo "$out"
  echo "FAIL: compile/run non-zero" >&2
  exit 1
}
echo "$out"
if ! grep -q 'GLOBAL_ARRAY_REF_MUT_OK' <<<"$out"; then
  echo "FAIL: missing GLOBAL_ARRAY_REF_MUT_OK marker" >&2
  exit 1
fi
if grep -q 'FAIL ' <<<"$out"; then
  echo "FAIL: assertion marker in output" >&2
  exit 1
fi
echo "PASS madaros_global_array_ref_gate"
echo "MADAROS_GLOBAL_ARRAY_REF_GATE_OK"
