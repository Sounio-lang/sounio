#!/usr/bin/env bash
# SIO-207 phase 1: signed integer overflow fails closed on Madaros.
#
# Before this gate, i64 arithmetic wrapped silently (MAX+1 printed
# -9223372036854775808, rc=0) and an i32 variable stored 2^31. The native
# backend now traps (exit 1, the builtin-panic shape) on signed overflow
# of +, -, * and unary negation; wrapping_add/sub/mul keep the explicit
# two's-complement wrap for hash/RNG state evolution; out-of-range
# constants for narrow types are a compile error (E300).
#
# Exit-status readings go through scripts/lib/classify_compile.sh (#2425):
# a crash or infra failure must never masquerade as the expected trap.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
unset SOUNIO_SOUC_ENGINE || true
SOUC="${SOUC:-$ROOT/bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
# shellcheck disable=SC1091
source "$ROOT/scripts/lib/classify_compile.sh"

echo "== madaros_int_overflow_gate =="

# 1) Dynamic signed overflow traps instead of wrapping. The id() indirection
# keeps the operand out of constant folding, so the guard under test is the
# runtime one.
cat > "$OUT/trap_i64.sio" <<'EOF'
fn id(x: i64) -> i64 { x }
fn main() with IO {
  let m: i64 = id(9223372036854775807)
  print_int(m + 1)
  print("NOT_TRAPPED\n")
}
EOF
trap_rc=0
sounio_capture_rc trap_rc "$OUT/trap.log" -- \
  "$SOUC" run "$OUT/trap_i64.sio"
if ! sounio_expect_failed_exit "$trap_rc" "$OUT/trap.log" ""; then
  echo "FAIL: dynamic i64 overflow did not fail closed: $(sounio_refusal_reason) rc=$trap_rc"
  tail -10 "$OUT/trap.log" || true
  exit 1
fi
if grep -qF "NOT_TRAPPED" "$OUT/trap.log"; then
  echo "FAIL: execution continued past the overflowing add"
  tail -10 "$OUT/trap.log" || true
  exit 1
fi
if grep -qF -- "-9223372036854775808" "$OUT/trap.log"; then
  echo "FAIL: the wrapped value was printed"
  tail -10 "$OUT/trap.log" || true
  exit 1
fi

# 2) In-range arithmetic is untouched, and the wrapping builtins reproduce
# the old bit patterns exactly.
cat > "$OUT/ok_wrap.sio" <<'EOF'
fn id(x: i64) -> i64 { x }
fn main() with IO {
  let a: i64 = id(40)
  print_int(a + 2)
  print("\n")
  let m: i64 = id(9223372036854775807)
  print_int(wrapping_add(m, 1))
  print("\n")
  print_int(wrapping_mul(m, 2))
  print("\n")
  print_int(wrapping_sub(0 - 5, 3))
  print("\n")
}
EOF
if ! "$SOUC" run "$OUT/ok_wrap.sio" >"$OUT/ok.log" 2>&1; then
  echo "FAIL: in-range/wrapping probe did not run clean"
  tail -10 "$OUT/ok.log" || true
  exit 1
fi
for want in "42" "-9223372036854775808" "-2" "-8"; do
  grep -qxF -- "$want" "$OUT/ok.log" || {
    echo "FAIL: expected line '$want' missing from probe output"
    cat "$OUT/ok.log" || true
    exit 1
  }
done

# 3) Out-of-range constant for a narrow type is a compile error (E300),
# classified as a refusal, never a crash.
cat > "$OUT/bad_i32.sio" <<'EOF'
fn main() with IO {
  let c: i32 = 2147483648
  print_int(c as i64)
}
EOF
refuse_rc=0
sounio_capture_rc refuse_rc "$OUT/refuse.log" -- \
  "$SOUC" compile "$OUT/bad_i32.sio" -o "$OUT/nope.elf"
if ! sounio_expect_refusal "$refuse_rc" "$OUT/refuse.log" "$OUT/nope.elf" "E300"; then
  echo "FAIL: out-of-range i32 constant not refused with E300: $(sounio_refusal_reason) rc=$refuse_rc"
  tail -10 "$OUT/refuse.log" || true
  exit 1
fi

echo "MADAROS_INT_OVERFLOW_GATE_OK"
