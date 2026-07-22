#!/usr/bin/env bash
# Gate: print_f64 keeps magnitude for negatives (not "-0.000000").
# Residual closeout for #890 / MADAROS_NATIVE_V2_F64_REMAINING_BUGS Note.
# Witness: tests/run-pass/print_f64_negative.sio
# Covers: -0.0, -2.0, -0.5, positive control +2.0; f64_to_bits oracle; print_int.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true

SOUC="${SOUC:-$ROOT/bin/souc}"
TEST="$ROOT/tests/run-pass/print_f64_negative.sio"
# Also keep the earlier println-based regression green.
TEST_PRINTLN="$ROOT/tests/run-pass/println_f64_negative.sio"

if [[ ! -x "$SOUC" ]]; then
  echo "FAIL: souc not executable at $SOUC" >&2
  exit 2
fi
if [[ ! -f "$TEST" || ! -f "$TEST_PRINTLN" ]]; then
  echo "FAIL: missing witness files" >&2
  exit 2
fi

engine_line="$($SOUC --version 2>&1 | head -1 || true)"
if echo "$engine_line" | grep -qi lean_single; then
  echo "FAIL: gate must run under default Madaros, not lean_single ($engine_line)" >&2
  exit 1
fi

echo "== madaros_print_f64_negative_gate =="
echo "engine: $engine_line"
echo "git_sha=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

echo "== print_f64 witness: $TEST =="
out="$("$SOUC" run "$TEST" 2>&1)" || {
  echo "$out"
  echo "FAIL: print_f64 witness compile/run non-zero" >&2
  exit 1
}
echo "$out"

if ! grep -q 'PRINT_F64_NEGATIVE_OK' <<<"$out"; then
  echo "FAIL: missing PRINT_F64_NEGATIVE_OK marker" >&2
  exit 1
fi
if grep -q 'FAIL ' <<<"$out"; then
  echo "FAIL: assertion marker in print_f64 output" >&2
  exit 1
fi

# Exact fixed-format display lines (prefix tags avoid matching compiler noise).
need_line() {
  local needle="$1"
  if ! grep -qF -- "$needle" <<<"$out"; then
    echo "FAIL: missing exact line: $needle" >&2
    exit 1
  fi
}

need_line "POS2=2.000000"
need_line "NEG2=-2.000000"
need_line "NEG05=-0.500000"
need_line "NEG0=-0.000000"

# Bits oracle (signed i64 print of IEEE patterns).
need_line "BITS_POS2=4611686018427387904"
need_line "BITS_NEG2=-4611686018427387904"
need_line "BITS_NEG05=-4620693217682128896"
need_line "BITS_NEG0=-9223372036854775808"

# Regression: pre-fix negatives collapsed to -0.000000 for *every* negative.
# If -2.0 or -0.5 still mangle, NEG2/NEG05 would be -0.000000 and fail above;
# also reject a bare "-0.000000" without the NEG0= tag appearing as the only mag.
if grep -qE 'NEG2=-0\.000000' <<<"$out"; then
  echo "FAIL: NEG2 still mangled to -0.000000" >&2
  exit 1
fi
if grep -qE 'NEG05=-0\.000000' <<<"$out"; then
  echo "FAIL: NEG05 still mangled to -0.000000" >&2
  exit 1
fi

echo "== println_f64 regression: $TEST_PRINTLN =="
out2="$("$SOUC" run "$TEST_PRINTLN" 2>&1)" || {
  echo "$out2"
  echo "FAIL: println_f64_negative compile/run non-zero" >&2
  exit 1
}
echo "$out2"
for want in -0.200000 -0.750000 -1.500000 -3.500000 0.200000 1.500000 3.500000; do
  if ! grep -qF -- "$want" <<<"$out2"; then
    echo "FAIL: println witness missing $want" >&2
    exit 1
  fi
done
# Ensure a known negative did not collapse.
if ! grep -qF -- '-0.200000' <<<"$out2"; then
  echo "FAIL: println -0.2 magnitude missing" >&2
  exit 1
fi

echo "PASS madaros_print_f64_negative_gate"
echo "MADAROS_PRINT_F64_NEGATIVE_GATE_OK"
