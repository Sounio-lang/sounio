#!/usr/bin/env bash
# SIO-207 phase 1: signed integer overflow fails closed on Madaros.
# SIO-588 extends it: exact unsigned overflow + dynamic i32/u32 width.
#
# Before this gate, i64 arithmetic wrapped silently (MAX+1 printed
# -9223372036854775808, rc=0) and an i32 variable stored 2^31. The native
# backend now traps (exit 1, the builtin-panic shape) on signed overflow
# of +, -, * and unary negation; wrapping_add/sub/mul keep the explicit
# two's-complement wrap for hash/RNG state evolution; out-of-range
# constants for narrow types are a compile error (E300).
#
# SIO-588: the trap fires on signed overflow ONLY. Unsigned +, -, *
# lower to OpUAdd/OpUSub/OpUMul (same bit-producing instructions, no `jo`
# guard) and wrap; dynamic i32 results sign-extend the low 32 bits and
# dynamic u32 results mask them, so no 64-bit value leaks through a
# 32-bit-typed operation.
#
# Two modes: on trees with the SIO-207 fail-closed infra (E300 refusal of
# an out-of-range narrow literal) every section 1-6 runs hard. On pre-207
# trees sections 1-3 and 6 are loud skips (they assert 207 behavior, card
# 207) and only the SIO-588 sections 4-5 run hard. The mode probe below
# decides; a skip names its blocking card and never passes silently.
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

# Mode probe: SIO-207 refuses an out-of-range narrow literal with E300.
# Pre-207 trees compile it silently -> SIO-588 sections only.
cat > "$OUT/probe207.sio" <<'EOF'
fn main() with IO {
  let c: i32 = 2147483648
  print_int(c)
}
EOF
probe207_rc=0
sounio_capture_rc probe207_rc "$OUT/probe207.log" -- \
  "$SOUC" compile "$OUT/probe207.sio" -o "$OUT/probe207.elf"
if sounio_expect_refusal "$probe207_rc" "$OUT/probe207.log" "$OUT/probe207.elf" "E300"; then
  HAS207=1
else
  HAS207=0
fi
echo "SIO-207 fail-closed infra present: $HAS207 (0 = SIO-588 sections 4-5 only)"

if (( HAS207 )); then
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
else
  echo "SKIP section 1: dynamic signed trap requires SIO-207 (card 207)"
fi

if (( HAS207 )); then
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

# 2b) The i64 MIN spelling is a literal, not a runtime negation of the
# 2^63 magnitude token; it must materialise without tripping the guard.
cat > "$OUT/min_literal.sio" <<'EOF'
fn id(x: i64) -> i64 { x }
fn main() with IO {
  print_int(id(-9223372036854775808))
  print("\n")
}
EOF
if ! "$SOUC" run "$OUT/min_literal.sio" >"$OUT/min.log" 2>&1; then
  echo "FAIL: i64 MIN literal probe did not run clean"
  tail -10 "$OUT/min.log" || true
  exit 1
fi
grep -qxF -- "-9223372036854775808" "$OUT/min.log" || {
  echo "FAIL: i64 MIN literal did not materialise"
  cat "$OUT/min.log" || true
  exit 1
}
else
  echo "SKIP section 2: wrapping builtins + MIN literal require SIO-207 (card 207)"
fi

if (( HAS207 )); then
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
else
  echo "SKIP section 3: E300 narrow refusal requires SIO-207 (card 207)"
fi

# 4) SIO-588: unsigned +, -, * wrap instead of trapping. The checker picks
# the UAdd/USub/UMul IR variants for unsigned operands, which emit the same
# bit-producing instructions with no `jo` guard. The id() indirection keeps
# the operands out of const folding, so the runtime (not the folder) wraps.
cat > "$OUT/uwrap.sio" <<'EOF'
fn id(x: u64) -> u64 { x }
fn main() with IO {
  let m: u64 = id(18446744073709551615)
  println(m + 1)
  println(id(0) - 1)
  println(m * 2)
  print("WRAPPED_CLEAN\n")
}
EOF
if ! "$SOUC" run "$OUT/uwrap.sio" >"$OUT/uwrap.log" 2>&1; then
  echo "FAIL: unsigned wrap probe did not run clean (the trap fired on unsigned)"
  tail -10 "$OUT/uwrap.log" || true
  exit 1
fi
for want in "0" "18446744073709551615" "18446744073709551614" "WRAPPED_CLEAN"; do
  grep -qxF -- "$want" "$OUT/uwrap.log" || {
    echo "FAIL: expected line '$want' missing from unsigned wrap output"
    cat "$OUT/uwrap.log" || true
    exit 1
  }
done

# 5) SIO-588: dynamic i32/u32 arithmetic respects 32 bits. A 64-bit word
# slot holding an i32/u32 result is wrapped back to width at runtime:
# i32 sign-extends the low 32 bits, u32 masks them. No trap fires (the
# 64-bit SIO-207 guard cannot trip on 32-bit-range inputs); no 64-bit
# value leaks through either.
cat > "$OUT/narrow32.sio" <<'EOF'
fn id32(x: i32) -> i32 { x }
fn idu32(x: u32) -> u32 { x }
fn bump(x: i32) -> i32 { x + 1 }
fn main() with IO {
  let a: i32 = id32(2147483647)
  println(a + 1)
  let n: i32 = id32(-2147483648)
  println(n - 1)
  println(id32(1000000) * id32(3000))
  let w: u32 = idu32(4294967295)
  println(w + 1)
  println(bump(a))
  println(-n)
}
EOF
if ! "$SOUC" run "$OUT/narrow32.sio" >"$OUT/narrow32.log" 2>&1; then
  echo "FAIL: dynamic narrow-32 probe did not run clean"
  tail -10 "$OUT/narrow32.log" || true
  exit 1
fi
for want in "-2147483648" "2147483647" "-1294967296" "0" "-2147483648"; do
  grep -qxF -- "$want" "$OUT/narrow32.log" || {
    echo "FAIL: expected line '$want' missing from narrow-32 output"
    cat "$OUT/narrow32.log" || true
    exit 1
  }
done
# A 64-bit leak would surface as an exact un-narrowed line: 2147483648
# (i32 MAX+1 kept at 64 bits) or 4294967296 (u32 MAX+1 kept unmasked).
for leaked in "2147483648" "4294967296"; do
  if grep -qxF -- "$leaked" "$OUT/narrow32.log"; then
    echo "FAIL: a narrow-32 result leaked a 64-bit value ($leaked)"
    cat "$OUT/narrow32.log" || true
    exit 1
  fi
done

if (( HAS207 )); then
# 6) SIO-588: the u32 literal boundary is refused with E300 exactly like
# the SIO-207 i32 one (the TyU32 const-range arm predates this change;
# this pins it next to the new runtime probes).
cat > "$OUT/bad_u32.sio" <<'EOF'
fn main() with IO {
  let c: u32 = 4294967296
  print_int(c as i64)
}
EOF
refuse_u32_rc=0
sounio_capture_rc refuse_u32_rc "$OUT/refuse_u32.log" -- \
  "$SOUC" compile "$OUT/bad_u32.sio" -o "$OUT/nope_u32.elf"
if ! sounio_expect_refusal "$refuse_u32_rc" "$OUT/refuse_u32.log" "$OUT/nope_u32.elf" "E300"; then
  echo "FAIL: out-of-range u32 constant not refused with E300: $(sounio_refusal_reason) rc=$refuse_u32_rc"
  tail -10 "$OUT/refuse_u32.log" || true
  exit 1
fi
else
  echo "SKIP section 6: E300 u32 refusal requires SIO-207 (card 207)"
fi

echo "MADAROS_INT_OVERFLOW_GATE_OK"
