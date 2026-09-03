#!/usr/bin/env bash
# Wide-integer gate: exercise i128/i256 type identity, type safety,
# explicit casts, source compile/run, and the IrWideAdd carry-chain witness.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

MADAROS="${MADAROS_BIN:-$ROOT_DIR/bin/madaros}"
WORK="${SOUNIO_WIDE_INT_GATE_DIR:-$(mktemp -d /tmp/sounio-wide-int.XXXXXX)}"
KEEP_WORK="${SOUNIO_WIDE_INT_GATE_KEEP:-0}"

if [[ "$KEEP_WORK" != "1" ]]; then
  trap 'rm -rf "$WORK"' EXIT
fi

fail() {
  echo "[wide-int] FAIL: $*" >&2
  exit 1
}

pass() {
  echo "[wide-int] PASS: $*"
}

expect_exit() {
  local expected="$1"
  shift
  set +e
  "$@"
  local rc=$?
  set -e
  if [[ "$rc" != "$expected" ]]; then
    fail "expected exit $expected, got $rc: $*"
  fi
}

expect_log_contains() {
  local needle="$1"
  local file="$2"
  if ! grep -Fq "$needle" "$file"; then
    echo "[wide-int] log tail for $file:" >&2
    tail -n 40 "$file" >&2 || true
    fail "missing log marker: $needle"
  fi
}

source_42() {
  local name="$1"
  local src="$2"
  local path="$WORK/${name}.sio"
  local elf="$WORK/${name}.elf"
  printf '%s\n' "$src" > "$path"
  "$MADAROS" check "$path" >"$WORK/${name}_check.log" 2>&1
  expect_log_contains "check: OK" "$WORK/${name}_check.log"
  "$MADAROS" build "$path" -o "$elf" >"$WORK/${name}_build.log" 2>&1
  [[ -s "$elf" ]] || fail "$name did not produce ELF"
  chmod +x "$elf"
  expect_exit 42 "$elf"
  pass "$name source check/build/run exit=42"
}

mkdir -p "$WORK"
printf '[wide-int] madaros=%s\n' "$MADAROS"
printf '[wide-int] work=%s\n' "$WORK"

# ---------------------------------------------------------------------------
# 1. Type identity: i128, i256, u128, u256 all pass check
# ---------------------------------------------------------------------------
printf 'fn main() -> i64 { let x: i128 = 1 as i128; 0 }\n' > "$WORK/i128_basic.sio"
printf 'fn main() -> i64 { let x: i256 = 1 as i256; 0 }\n' > "$WORK/i256_basic.sio"
printf 'fn main() -> i64 { let x: u128 = 1 as u128; 0 }\n' > "$WORK/u128_basic.sio"
printf 'fn main() -> i64 { let x: u256 = 1 as u256; 0 }\n' > "$WORK/u256_basic.sio"

for src in "$WORK"/i128_basic.sio "$WORK"/i256_basic.sio "$WORK"/u128_basic.sio "$WORK"/u256_basic.sio; do
  "$MADAROS" check "$src" >"$WORK/check_$(basename "$src" .sio).log" 2>&1
  expect_log_contains "check: OK" "$WORK/check_$(basename "$src" .sio).log"
done
pass "i128/i256/u128/u256 type identity"

# ---------------------------------------------------------------------------
# 2. Type safety: i128 != i256 (must FAIL check)
# ---------------------------------------------------------------------------
printf 'fn main() -> i64 { let x: i128 = 1 as i256; 0 }\n' > "$WORK/mismatch_128_256.sio"
expect_exit 1 "$MADAROS" check "$WORK/mismatch_128_256.sio" >"$WORK/mismatch_128_256.log" 2>&1
pass "i128 != i256 rejected by checker"

# ---------------------------------------------------------------------------
# 3. Type safety: u128 != i128 (must FAIL check)
# ---------------------------------------------------------------------------
printf 'fn main() -> i64 { let x: u128 = 1 as i128; 0 }\n' > "$WORK/mismatch_u128_i128.sio"
expect_exit 1 "$MADAROS" check "$WORK/mismatch_u128_i128.sio" >"$WORK/mismatch_u128_i128.log" 2>&1
pass "u128 != i128 rejected by checker"

# ---------------------------------------------------------------------------
# 4. Explicit cast: 1 as i128 and 1 as i256 pass check
# ---------------------------------------------------------------------------
printf 'fn main() -> i64 { let a: i128 = 42 as i128; let b: i256 = 99 as i256; 0 }\n' > "$WORK/cast.sio"
"$MADAROS" check "$WORK/cast.sio" >"$WORK/cast.log" 2>&1
expect_log_contains "check: OK" "$WORK/cast.log"
pass "explicit cast (1 as i128, 1 as i256)"

# ---------------------------------------------------------------------------
# 5. Function with i128 param/return: check + build + run
# ---------------------------------------------------------------------------
printf 'fn id(x: i128) -> i128 { x }\nfn main() -> i64 { let r = id(42 as i128); 0 }\n' > "$WORK/i128_fn.sio"
"$MADAROS" check "$WORK/i128_fn.sio" >"$WORK/i128_fn_check.log" 2>&1
expect_log_contains "check: OK" "$WORK/i128_fn_check.log"
"$MADAROS" build "$WORK/i128_fn.sio" -o "$WORK/i128_fn.elf" >"$WORK/i128_fn_build.log" 2>&1
[[ -s "$WORK/i128_fn.elf" ]] || fail "i128 function did not produce ELF"
chmod +x "$WORK/i128_fn.elf"
expect_exit 0 "$WORK/i128_fn.elf"
pass "i128 function check/build/run"

# ---------------------------------------------------------------------------
# 6. Function with i256 param/return: check + build + run
# ---------------------------------------------------------------------------
printf 'fn id(x: i256) -> i256 { x }\nfn main() -> i64 { let r = id(7 as i256); 0 }\n' > "$WORK/i256_fn.sio"
"$MADAROS" check "$WORK/i256_fn.sio" >"$WORK/i256_fn_check.log" 2>&1
expect_log_contains "check: OK" "$WORK/i256_fn_check.log"
"$MADAROS" build "$WORK/i256_fn.sio" -o "$WORK/i256_fn.elf" >"$WORK/i256_fn_build.log" 2>&1
[[ -s "$WORK/i256_fn.elf" ]] || fail "i256 function did not produce ELF"
chmod +x "$WORK/i256_fn.elf"
expect_exit 0 "$WORK/i256_fn.elf"
pass "i256 function check/build/run"

# ---------------------------------------------------------------------------
# 7. Wide-add4 witness: 4-limb carry chain proves real i256 arithmetic.
#    (-1,-1,-1,0) + (1,0,0,0) -> carry cascades 0->1->2->3; high limb = 1.
#    Fake-i64 would give 0. Exit code MUST be 1.
# ---------------------------------------------------------------------------
"$MADAROS" --native-v2-emit-wide-add4 "$WORK/wide_add4.elf" >"$WORK/wide_add4.log" 2>&1
expect_log_contains "wide-add4 rc=0" "$WORK/wide_add4.log"
[[ -s "$WORK/wide_add4.elf" ]] || fail "wide-add4 did not produce ELF"
file "$WORK/wide_add4.elf" | grep -Fq "ELF 64-bit" || fail "wide-add4 not ELF64"
chmod +x "$WORK/wide_add4.elf"
expect_exit 1 "$WORK/wide_add4.elf"
pass "IrWideAdd 4-limb carry chain (exit=1, not fake-i64 exit=0)"

# ---------------------------------------------------------------------------
# 8. Wide-sub witness: i128 (hi=2,lo=0) - (hi=0,lo=1) -> borrow -> hi=1.
#    Fake-i64 would give 2. Exit code MUST be 1.
# ---------------------------------------------------------------------------
"$MADAROS" --native-v2-emit-wide-sub "$WORK/wide_sub.elf" >"$WORK/wide_sub.log" 2>&1
expect_log_contains "wide-sub rc=0" "$WORK/wide_sub.log"
[[ -s "$WORK/wide_sub.elf" ]] || fail "wide-sub did not produce ELF"
file "$WORK/wide_sub.elf" | grep -Fq "ELF 64-bit" || fail "wide-sub not ELF64"
chmod +x "$WORK/wide_sub.elf"
expect_exit 1 "$WORK/wide_sub.elf"
pass "IrWideSub borrow chain (exit=1, not fake-i64 exit=2)"

# ---------------------------------------------------------------------------
# 9. Wide-mul witness: i128 2^32 * 2^32 = 2^64 -> lo=0, hi=1.
#    Fake-i64 (single mul) would give 0. Exit code MUST be 1.
# ---------------------------------------------------------------------------
"$MADAROS" --native-v2-emit-wide-mul "$WORK/wide_mul.elf" >"$WORK/wide_mul.log" 2>&1
expect_log_contains "wide-mul rc=0" "$WORK/wide_mul.log"
[[ -s "$WORK/wide_mul.elf" ]] || fail "wide-mul did not produce ELF"
file "$WORK/wide_mul.elf" | grep -Fq "ELF 64-bit" || fail "wide-mul not ELF64"
chmod +x "$WORK/wide_mul.elf"
expect_exit 1 "$WORK/wide_mul.elf"
pass "IrWideMul schoolbook multiply (exit=1, not fake-i64 exit=0)"

# ---------------------------------------------------------------------------
# 10. Wide-div witness: (15*2^64)/3 = 5*2^64 -> hi limb = 5.
#    Fake-i64 would give 0 (div of lo limbs). Exit code MUST be 5.
# ---------------------------------------------------------------------------
"$MADAROS" --native-v2-emit-wide-div "$WORK/wide_div.elf" >"$WORK/wide_div.log" 2>&1
expect_log_contains "wide-div rc=0" "$WORK/wide_div.log"
[[ -s "$WORK/wide_div.elf" ]] || fail "wide-div did not produce ELF"
chmod +x "$WORK/wide_div.elf"
expect_exit 5 "$WORK/wide_div.elf"
pass "IrWideDiv single-limb divisor (exit=5, not fake-i64 exit=0)"

# ---------------------------------------------------------------------------
# 11. Wide-mod witness: (2^64+1)%7 = 3.
#    Fake-i64 would give 1 (1%7=1). Exit code MUST be 3.
# ---------------------------------------------------------------------------
"$MADAROS" --native-v2-emit-wide-mod "$WORK/wide_mod.elf" >"$WORK/wide_mod.log" 2>&1
expect_log_contains "wide-mod rc=0" "$WORK/wide_mod.log"
[[ -s "$WORK/wide_mod.elf" ]] || fail "wide-mod did not produce ELF"
chmod +x "$WORK/wide_mod.elf"
expect_exit 3 "$WORK/wide_mod.elf"
pass "IrWideMod single-limb divisor (exit=3, not fake-i64 exit=1)"

# ---------------------------------------------------------------------------
# 12. Wide-cmp witness: 2^64 < 2^65 -> 1 (hi limb decides).
#    Fake-i64 would give 0 (lo limbs equal). Exit code MUST be 1.
# ---------------------------------------------------------------------------
"$MADAROS" --native-v2-emit-wide-cmp "$WORK/wide_cmp.elf" >"$WORK/wide_cmp.log" 2>&1
expect_log_contains "wide-cmp rc=0" "$WORK/wide_cmp.log"
[[ -s "$WORK/wide_cmp.elf" ]] || fail "wide-cmp did not produce ELF"
chmod +x "$WORK/wide_cmp.elf"
expect_exit 1 "$WORK/wide_cmp.elf"
pass "IrWideCmp unsigned multi-limb comparison (exit=1, not fake-i64 exit=0)"

# ---------------------------------------------------------------------------
# 13. Wide-shr witness: (2^32*2^32)>>64 = 1 (mul+shr_limb combo).
#    Broken shr would return 0. Exit code MUST be 1.
# ---------------------------------------------------------------------------
"$MADAROS" --native-v2-emit-wide-shr "$WORK/wide_shr.elf" >"$WORK/wide_shr.log" 2>&1
expect_log_contains "wide-shr rc=0" "$WORK/wide_shr.log"
[[ -s "$WORK/wide_shr.elf" ]] || fail "wide-shr did not produce ELF"
chmod +x "$WORK/wide_shr.elf"
expect_exit 1 "$WORK/wide_shr.elf"
pass "IrWideShrLimb limb-aligned shift (exit=1)"

# ---------------------------------------------------------------------------
# 14. Wide-shr-unaligned witness: 2^100 >> 96 = 16 (funnel shift).
#    Limb-aligned-only (>>64) would give 2^36; low-limb -> 0.
# ---------------------------------------------------------------------------
"$MADAROS" --native-v2-emit-wide-shr-unaligned "$WORK/wide_shru.elf" >"$WORK/wide_shru.log" 2>&1
expect_log_contains "wide-shr-unaligned rc=0" "$WORK/wide_shru.log"
[[ -s "$WORK/wide_shru.elf" ]] || fail "wide-shr-unaligned did not produce ELF"
chmod +x "$WORK/wide_shru.elf"
expect_exit 16 "$WORK/wide_shru.elf"
pass "IrWideShr funnel shift (exit=16)"

# ---------------------------------------------------------------------------
# 15. Wide-divfull witness: (5*2^64+105)/(2^64+1) = 5 (multi-limb divisor).
#    Low-limb-only (div by D.lo=1) would give 105.
# ---------------------------------------------------------------------------
"$MADAROS" --native-v2-emit-wide-divfull "$WORK/wide_df.elf" >"$WORK/wide_df.log" 2>&1
expect_log_contains "wide-divfull rc=0" "$WORK/wide_df.log"
[[ -s "$WORK/wide_df.elf" ]] || fail "wide-divfull did not produce ELF"
chmod +x "$WORK/wide_df.elf"
expect_exit 5 "$WORK/wide_df.elf"
pass "IrWideDivFull multi-limb binary long division (exit=5, not 105)"

# ---------------------------------------------------------------------------
# 16. Wide-modfull witness: (7*2^64+200)%(2^64+1) = 193 (multi-limb divisor).
#    Low-limb-only (% D.lo=1) would give 0.
# ---------------------------------------------------------------------------
"$MADAROS" --native-v2-emit-wide-modfull "$WORK/wide_mf.elf" >"$WORK/wide_mf.log" 2>&1
expect_log_contains "wide-modfull rc=0" "$WORK/wide_mf.log"
[[ -s "$WORK/wide_mf.elf" ]] || fail "wide-modfull did not produce ELF"
chmod +x "$WORK/wide_mf.elf"
expect_exit 193 "$WORK/wide_mf.elf"
pass "IrWideModFull multi-limb binary long division remainder (exit=193, not 0)"

# ---------------------------------------------------------------------------
# 17. Source-level wide arithmetic witnesses: lower real user syntax into
#     IrWideAdd/Sub/Mul/Cmp, then build and run native ELFs.
# ---------------------------------------------------------------------------
source_42 "source_i128_mul_gt" 'fn main() -> i64 { let a: i128 = 4294967296 as i128; let b: i128 = 4294967296 as i128; let c: i128 = a * b; let z: i128 = 0 as i128; if c > z { return 42 } 1 }'
source_42 "source_i256_mul_gt" 'fn main() -> i64 { let a: i256 = 4294967296 as i256; let b: i256 = 4294967296 as i256; let c: i256 = a * b; let z: i256 = 0 as i256; if c > z { return 42 } 1 }'
source_42 "source_u128_mul_add_gt" 'fn main() -> i64 { let a: u128 = 4294967296 as u128; let b: u128 = 4294967296 as u128; let c: u128 = a * b; let d: u128 = c + c; if d > c { return 42 } 1 }'
source_42 "source_u256_mul_add_ne" 'fn main() -> i64 { let a: u256 = 4294967296 as u256; let b: u256 = 4294967296 as u256; let c: u256 = a * b; let d: u256 = c + c; if d != c { return 42 } 1 }'
source_42 "source_i128_sub_eq_zero" 'fn main() -> i64 { let a: i128 = 4294967296 as i128; let b: i128 = 4294967296 as i128; let c: i128 = a * b; let d: i128 = c - c; let z: i128 = 0 as i128; if d == z { return 42 } 1 }'

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "[wide-int] PASS: type identity, type safety, casts, i128/i256 compile"
echo "[wide-int] PASS: IrWideAdd/Sub/Mul/Shr/Div/Mod/Cmp/DivFull/ModFull witnesses"
echo "[wide-int] PASS: source-level i128/i256/u128/u256 add/sub/mul/cmp check/build/run"
