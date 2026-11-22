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

# 4) ADR-031 R1, unsigned and narrow widths: each probe overflows its own
# type at run time and must fail closed, with nothing printed after the
# overflowing operation. A guard on the signed 64-bit condition alone would
# let every one of these through (they fit i64).
expect_trap() {
  local name="$1" what="$2" msg="${3:-}"
  local rc=0
  sounio_capture_rc rc "$OUT/$name.log" -- "$SOUC" run "$OUT/$name.sio"
  if ! sounio_expect_failed_exit "$rc" "$OUT/$name.log" ""; then
    echo "FAIL: $what did not fail closed: $(sounio_refusal_reason) rc=$rc"
    tail -10 "$OUT/$name.log" || true
    exit 1
  fi
  if grep -qF "NOT_TRAPPED" "$OUT/$name.log"; then
    echo "FAIL: $what: execution continued past the overflowing operation"
    tail -10 "$OUT/$name.log" || true
    exit 1
  fi
  # ADR-031 R1: the panic names the type and the operation and, for the
  # 64-bit guards, where it is: the overflowing operation is on the probe's
  # last body line, two above the closing brace.
  if [[ -n "$msg" ]]; then
    local opline=$(( $(wc -l < "$OUT/$name.sio") - 2 ))
    local esc; esc=$(printf '%s' "$msg" | sed 's/[][\\.*^$+?(){}|/]/\\&/g')
    local prefix="integer overflow"
    case "$msg" in
      zero\ *) prefix="division by zero"; msg="${msg#zero }"; esc=$(printf '%s' "$msg" | sed 's/[][\\.*^$+?(){}|/]/\\&/g') ;;
    esac
    local want="^sounio: $prefix \\($esc\\)"
    case "$msg" in
      i64*|u64*) want="$want at .*$name\\.sio:$opline:[0-9]+\$" ;;
      *) want="$want( at .*$name\\.sio:$opline:[0-9]+)?\$" ;;
    esac
    if ! grep -qE -- "$want" "$OUT/$name.log"; then
      echo "FAIL: $what: panic line does not match /$want/"
      tail -10 "$OUT/$name.log" || true
      exit 1
    fi
  fi
}
probe() {
  local name="$1" ty="$2" body="$3"
  cat > "$OUT/$name.sio" <<EOF
fn id(x: $ty) -> $ty { x }
fn main() with IO {
  $body
  print("NOT_TRAPPED\n")
}
EOF
}
probe u64_add_max u64 'let m: u64 = id(18446744073709551615)
  println(m + 1)'
expect_trap u64_add_max "u64::MAX + 1" "u64 +"
probe u64_sub_zero u64 'let z: u64 = id(0)
  println(z - 1)'
expect_trap u64_sub_zero "0u64 - 1" "u64 -"
probe u64_mul_two u64 'let m: u64 = id(18446744073709551615)
  println(m * 2)'
expect_trap u64_mul_two "u64::MAX * 2" "u64 *"
probe i32_add_max i32 'let a: i32 = id(2147483647)
  println(a + 1)'
expect_trap i32_add_max "i32::MAX + 1" "i32 +"
probe i32_sub_min i32 'let n: i32 = id(-2147483647) - 1
  println(n - 1)'
expect_trap i32_sub_min "i32::MIN - 1" "i32 -"
probe i32_mul i32 'let a: i32 = id(1000000)
  println(a * id(3000))'
expect_trap i32_mul "1000000i32 * 3000" "i32 *"
probe i32_neg_min i32 'let n: i32 = id(-2147483647) - 1
  println(-n)'
expect_trap i32_neg_min "-i32::MIN" "i32 unary -"
# The call boundary: a non-literal i64 outside 32 bits reaching an i32 param
# panics there (it used to be masked, or kept whole in an i32 local).
probe i32_arg_narrow i32 'let big: i64 = 9999999999
  let r: i32 = id(big)
  println(r as i64)'
expect_trap i32_arg_narrow "i64 9999999999 passed to an i32 param" "i32 argument"
probe u32_add_max u32 'let w: u32 = id(4294967295)
  println(w + 1)'
expect_trap u32_add_max "u32::MAX + 1" "u32 +"
probe i64_add_max i64 'let m: i64 = id(9223372036854775807)
  println(m + 1)'
expect_trap i64_add_max "i64::MAX + 1" "i64 +"
probe i64_sub_min i64 'let n: i64 = id(-9223372036854775808)
  println(n - 1)'
expect_trap i64_sub_min "i64::MIN - 1" "i64 -"
probe i64_mul i64 'let q: i64 = id(4611686018427387904)
  println(q * 2)'
expect_trap i64_mul "2^62 * 2" "i64 *"
probe i64_neg_min i64 'let n: i64 = id(-9223372036854775808)
  println(-n)'
expect_trap i64_neg_min "-i64::MIN" "i64 unary -"

# 5) ADR-031 R2: integer / and % fail on a zero divisor and on MIN / -1,
# through the same trap as overflow (exit 1), not the CPU's SIGFPE. The
# divisors come through id() so neither the checker's constant-zero refusal
# (E056/E057) nor constant propagation sees them.
probe div_zero_i64 i64 'let z: i64 = id(0)
  println(id(7) / z)'
expect_trap div_zero_i64 "7 / 0 (i64)" "zero i64 /"
probe rem_zero_i64 i64 'let z: i64 = id(0)
  println(id(7) % z)'
expect_trap rem_zero_i64 "7 % 0 (i64)" "zero i64 %"
probe div_min_i64 i64 'let m: i64 = id(-9223372036854775807) - 1
  println(m / id(-1))'
expect_trap div_min_i64 "i64::MIN / -1" "i64 /"
probe rem_min_i64 i64 'let m: i64 = id(-9223372036854775807) - 1
  println(m % id(-1))'
expect_trap rem_min_i64 "i64::MIN % -1" "i64 %"
probe div_zero_u64 u64 'let z: u64 = id(0)
  println(id(7) / z)'
expect_trap div_zero_u64 "7 / 0 (u64)" "zero u64 /"
probe div_min_i32 i32 'let m: i32 = id(-2147483647) - 1
  println(m / id(-1))'
expect_trap div_min_i32 "i32::MIN / -1"

cat > "$OUT/div_ok.sio" <<'EOF'
fn id(x: i64) -> i64 { x }
fn idu(x: u64) -> u64 { x }
fn main() with IO {
  let m: i64 = id(-9223372036854775807) - 1
  println(id(-7) / id(2))
  println(id(-7) % id(2))
  println(m / id(1))
  println(m / id(2))
  println(idu(18446744073709551615) / idu(2))
  println(idu(18446744073709551615) % idu(10))
}
EOF
if ! "$SOUC" run "$OUT/div_ok.sio" >"$OUT/div_ok.log" 2>&1; then
  echo "FAIL: in-range division probe did not run clean"
  tail -10 "$OUT/div_ok.log" || true
  exit 1
fi
for want in "-3" "-1" "-9223372036854775808" "-4611686018427387904" "9223372036854775807" "5"; do
  grep -qxF -- "$want" "$OUT/div_ok.log" || {
    echo "FAIL: expected line '$want' missing from the division probe"
    cat "$OUT/div_ok.log" || true
    exit 1
  }
done

# 5) The i32 guard is one fused operation, not a five-op IR window. The window
# (shl 32, sar 32, compare, branch, panic call) cost about 30 machine
# instructions per i32 + - * on the stack-slot emitter and made i32 kernels
# run 1.3x to 2x slower than the same code in i64. The deterministic proxy
# for "the guard is cheap" is code size: a build with 400 dynamic i32 adds must
# not outgrow the same build in i64 by more than two pages (the ELF is
# page-rounded); with the window it outgrew it by about 50 KB.
gen_adds() {
  local ty="$1" n="$2" out="$3"
  {
    echo "fn id(x: $ty) -> $ty { x }"
    echo "fn main() with IO {"
    echo "  var a: $ty = id(1)"
    local i
    for i in $(seq 1 "$n"); do echo "  a = a + id($i)"; done
    echo "  println(a)"
    echo "}"
  } > "$out"
}
elf_size_of() {
  local src="$1" elf="$1.elf"
  if ! "$SOUC" compile "$src" -o "$elf" >"$src.log" 2>&1; then
    echo "FAIL: $(basename "$src") did not compile"; tail -10 "$src.log" || true; exit 1
  fi
  stat -c %s "$elf"
}
gen_adds i32 10 "$OUT/adds_i32_10.sio";   gen_adds i32 410 "$OUT/adds_i32_410.sio"
gen_adds i64 10 "$OUT/adds_i64_10.sio";   gen_adds i64 410 "$OUT/adds_i64_410.sio"
g32=$(( $(elf_size_of "$OUT/adds_i32_410.sio") - $(elf_size_of "$OUT/adds_i32_10.sio") ))
g64=$(( $(elf_size_of "$OUT/adds_i64_410.sio") - $(elf_size_of "$OUT/adds_i64_10.sio") ))
if [ "$g32" -gt $(( g64 + 8192 )) ]; then
  echo "FAIL: 400 i32 adds grew the binary by $g32 bytes against $g64 for i64: the i32 guard is not fused"
  exit 1
fi
if [ "$("$OUT/adds_i32_410.sio.elf")" != "84256" ]; then
  echo "FAIL: i32 add chain printed the wrong value"
  exit 1
fi

# 6) Loop-iteration regions reset in place at every iteration end
# (__rgn_reset). An iteration that allocated nothing has an empty range to
# zero; `rep stosq` with rcx = 0 still pays its microcode start-up, which made
# allocation-free loops that call a function about 1.5x slower. Every zeroing
# `rep stosq` the region runtime emits must be preceded by the empty-range test
# (test rcx, rcx; je over the 9-byte tail): 48 85 c9 0f 84 09 00 00 00, then
# mov rdi, lo; xor eax, eax; rep stosq.
cat > "$OUT/loop_reset.sio" <<'SIO'
fn bump(x: i64) -> i64 { x + 1 }
fn main() with IO {
  var i: i64 = 0
  var s: i64 = 0
  while i < 1000 {
    s = s + bump(i)
    i = i + 1
  }
  println(s)
}
SIO
"$SOUC" compile "$OUT/loop_reset.sio" -o "$OUT/loop_reset.elf" >"$OUT/loop_reset.log" 2>&1 || {
  echo "FAIL: loop_reset probe did not compile"; tail -10 "$OUT/loop_reset.log" || true; exit 1; }
[ "$("$OUT/loop_reset.elf")" = "500500" ] || { echo "FAIL: loop_reset probe printed the wrong sum"; exit 1; }
od -An -v -tx1 -w1000000 "$OUT/loop_reset.elf" | tr -d ' \n' > "$OUT/loop_reset.hex"
guarded=$(grep -o '4885c90f84090000004c89cf4831c0f348ab' "$OUT/loop_reset.hex" | wc -l)
if [ "$guarded" -lt 1 ]; then
  echo "FAIL: the region zeroing does not skip an empty range (no guarded rep stosq in the runtime)"
  exit 1
fi

echo "MADAROS_INT_OVERFLOW_GATE_OK"
