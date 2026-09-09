#!/usr/bin/env bash
# madaros_f128_f256_v0e52_language_mul_neg_cmp_gate.sh — V0-E.5.2 language f128 *, unary −, comparisons.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (V0-E.5.2 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
# Claim clock: ADR-008 / ADR-009 — oracle_class=sounio_native_expected
#
# V0-E.5.2 green (this gate):
#   - Seed-run of the F128Bits mul/neg/cmp routines (no language f128; seed still greenwashes)
#   - Madaros-run of language `f128` `*`, unary `-`, `< <= > >= == !=` desugared to stdlib softfloat
#   - Full 113×113-bit RNE product (tiny*tiny ≈ 1e-40) with native hex expecteds
#   - Anti-f64: (1+~1e-20)^2 ≠ 1 and (1+~1e-20) > 1
#   - IEEE corners: +0 == -0, NaN unordered, -0 sign bit
#   - Language f128 `/` and params (f128_v0e2) still V0-E.4.1 fail-closed
#
# Explicitly NOT claimed:
#   - lean_single language f128 (still f64 greenwash)
#   - div, f256, params/returns ABI, print_f128 builtin, GUM
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"
SEED_COMPILER="$(realpath "${SOUNIO_F128_SEED_COMPILER:-$ROOT_DIR/bin/souc-lean-single-x86_64}")"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e52.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

echo "=== madaros_f128_f256_ladder_gate stage=v0e52 ==="
echo "slice=v0e52_language_f128_mul_neg_cmp_desugar"
echo "claim_clock=sounio_native_expected"
echo "adr=ADR-008+ADR-009"
echo "seed_elf=$SEED_COMPILER"

NATIVE_EXPECT=(
  "wire_2*3=0000000000000000:4001800000000000"
  "wire_tiny*tiny=c2d80ca7a598ed48:3f7a16c262777579"
  "wire_(1+tiny)^2=00005e7284324908:3fff000000000000"
  "wire_-1*1.5=0000000000000000:bfff800000000000"
  "wire_two*tiny=3400000000000000:3fbd79ca10c92422"
)

SF=stdlib/math/softfloat_f128.sio
if [[ -f "$SF" ]] \
  && grep -Fq 'pub fn f128_bits_soft_mul' "$SF" \
  && grep -Fq 'pub fn f128_bits_soft_neg' "$SF" \
  && grep -Fq 'pub fn f128_bits_soft_cmp' "$SF" \
  && grep -Fq 'pub fn f128_bits_soft_lt' "$SF" \
  && grep -Fq 'pub fn f128_bits_soft_le' "$SF" \
  && grep -Fq 'pub fn f128_bits_soft_gt' "$SF" \
  && grep -Fq 'pub fn f128_bits_soft_ge' "$SF" \
  && grep -Fq 'pub fn f128_bits_soft_eq' "$SF" \
  && grep -Fq 'pub fn f128_bits_soft_ne' "$SF" \
  && grep -Fq 'fn sf_u128_mul_wide' "$SF"; then
  note_pass "stdlib_mul_neg_cmp_present"
else
  note_fail "stdlib_mul_neg_cmp_missing"
fi

if grep -Fq 'fn lower_f128_binary_target' self-hosted/ir/lower.sio \
  && grep -Fq 'fn lower_f128_unary_ref' self-hosted/ir/lower.sio \
  && grep -Fq 'f128_bits_soft_mul' self-hosted/ir/lower.sio \
  && grep -Fq 'f128_bits_soft_neg' self-hosted/ir/lower.sio \
  && grep -Fq 'f128_bits_soft_lt' self-hosted/ir/lower.sio \
  && grep -Fq 'f128_bits_soft_mul' self-hosted/check/specializer.sio \
  && ! grep -Fq 'f128_bits_soft_div' self-hosted/ir/lower.sio; then
  note_pass "lower_f128_mul_neg_cmp_markers"
else
  note_fail "lower_f128_mul_neg_cmp_markers_missing_or_div_claimed"
fi

SMOKE=tests/run-pass/f128_v0e52_language_mul_neg_cmp.sio
if grep -Fq 'use math::softfloat_f128::{' "$SMOKE" \
  && grep -Fq 'f128_bits_soft_mul' "$SMOKE" \
  && grep -Fq 'f128_bits_soft_neg' "$SMOKE" \
  && grep -Fq 'f128_bits_soft_cmp' "$SMOKE" \
  && grep -Fq 'let one: f128' "$SMOKE" \
  && grep -Fq 'v0e52_main_entered' "$SMOKE" \
  && grep -Fq 'two * three' "$SMOKE" \
  && grep -Fq 'tiny * tiny' "$SMOKE" \
  && grep -Fq '= -one' "$SMOKE" \
  && grep -Fq 'one_plus_tiny > one' "$SMOKE" \
  && grep -Fq 'zero == neg_zero' "$SMOKE" \
  && grep -Fq 'nan != nan' "$SMOKE" \
  && ! grep -Fq 'fn f128_bits_soft_mul' "$SMOKE"; then
  note_pass "smoke_is_language_f128_not_f128bits_copy"
else
  note_fail "smoke_must_be_language_f128_desugar"
fi

REFUSE_SENTINEL='f128/f256 Madaros-run softfloat lowering is not implemented (V0-E.4.1 fail-closed; no f64 greenwash)'

cat >"$TMP_DIR/seed_bits.sio" <<'EOF'
use math::softfloat_f128::{f128_bits_soft_mul, f128_bits_soft_neg, f128_bits_soft_cmp, f128_soft_mul_limbs}
use math::wide_float::{print_f128_bits_hex_wire, F128Bits}

fn main() -> i32 with IO, Mut, Panic, Div {
    let two = F128Bits { lo: 0, hi: 4611686018427387904 }
    let three = F128Bits { lo: 0, hi: 4611826755915743232 }
    let one = F128Bits { lo: 0, hi: 4611404543450677248 }
    let one_and_half = F128Bits { lo: 0, hi: 4611545280939032576 }
    let tiny = F128Bits { lo: 3746994889972252672, hi: 4592679628783035426 }
    let six = f128_bits_soft_mul(two, three)
    let tiny_sq = f128_bits_soft_mul(tiny, tiny)
    let one_plus_tiny = F128Bits { lo: 51922968585348, hi: 4611404543450677248 }
    let square = f128_bits_soft_mul(one_plus_tiny, one_plus_tiny)
    let neg_mul = f128_bits_soft_mul(f128_bits_soft_neg(one), one_and_half)
    let two_tiny = f128_soft_mul_limbs(0, 4611686018427387904, 3746994889972252672, 4592679628783035426)
    if f128_bits_soft_cmp(two, three) != 0 - 1 || f128_bits_soft_cmp(three, two) != 1 || f128_bits_soft_cmp(two, two) != 0 {
        println("FAIL seed_cmp")
        return 3
    }
    print("wire_2*3=")
    print_f128_bits_hex_wire(six)
    println("")
    print("wire_tiny*tiny=")
    print_f128_bits_hex_wire(tiny_sq)
    println("")
    print("wire_(1+tiny)^2=")
    print_f128_bits_hex_wire(square)
    println("")
    print("wire_-1*1.5=")
    print_f128_bits_hex_wire(neg_mul)
    println("")
    print("wire_two*tiny=")
    print_f128_bits_hex_wire(two_tiny)
    println("")
    println("PASS f128_v0e52_seed_bits")
    return 0
}
EOF

ELF="$TMP_DIR/seed_bits.elf"
BLOG="$TMP_DIR/seed.build.log"
RLOG="$TMP_DIR/seed.run.log"

if [[ ! -x "$SEED_COMPILER" ]]; then
  note_fail "seed_missing"
elif [[ "$(head -c2 "$SEED_COMPILER" 2>/dev/null)" == '#!' ]]; then
  note_fail "seed_is_wrapper_not_elf"
else
  set +e
  "$SEED_COMPILER" "$TMP_DIR/seed_bits.sio" "$ELF" >"$BLOG" 2>&1
  b_rc=$?
  set -e
  if [[ "$b_rc" -ne 0 || ! -f "$ELF" ]]; then
    note_fail "seed_build_bits_mul_neg_cmp"
    tail -40 "$BLOG" >&2 || true
  else
    chmod +x "$ELF"
    set +e
    "$ELF" >"$RLOG" 2>&1
    r_rc=$?
    set -e
    if [[ "$r_rc" -ne 0 ]] || ! grep -Fq 'PASS f128_v0e52_seed_bits' "$RLOG"; then
      note_fail "seed_run_bits_mul_neg_cmp"
      cat "$RLOG" >&2 || true
    else
      note_pass "seed_run_bits_mul_neg_cmp"
      for want in "${NATIVE_EXPECT[@]}"; do
        if grep -Fq "$want" "$RLOG"; then
          note_pass "seed_native_hex:${want%%=*}"
        else
          note_fail "seed_native_hex_mismatch:${want%%=*}"
          echo "want $want" >&2
          cat "$RLOG" >&2 || true
        fi
      done
    fi
  fi
fi

cat >"$TMP_DIR/lang_div.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi, f128_bits_soft_mul}

fn main() -> i32 with IO, Mut, Panic, Div {
    let one: f128 = 1.0
    let two: f128 = 2.0
    let q: f128 = one / two
    if f128_to_hi(q) == 0 { return 1 }
    return 0
}
EOF

if [[ -x "$SOUC" ]]; then
  set +e
  "$SOUC" compile tests/run-pass/f128_v0e2_arith_check.sio -o "$TMP_DIR/lang.elf" >"$TMP_DIR/lang.compile.log" 2>&1
  set -e
  if grep -Fq "$REFUSE_SENTINEL" "$TMP_DIR/lang.compile.log" \
    || grep -Fq 'f128 operand is not a language f128 local' "$TMP_DIR/lang.compile.log"; then
    note_pass "language_f128_params_still_fail_closed"
  else
    note_fail "language_f128_v0e2_fail_closed_regression"
    tail -30 "$TMP_DIR/lang.compile.log" >&2 || true
  fi

  set +e
  "$SOUC" compile "$TMP_DIR/lang_div.sio" -o "$TMP_DIR/lang_div.elf" >"$TMP_DIR/lang_div.compile.log" 2>&1
  d_rc=$?
  set -e
  if [[ "$d_rc" -ne 0 ]] && grep -Fq "$REFUSE_SENTINEL" "$TMP_DIR/lang_div.compile.log"; then
    note_pass "language_f128_div_still_fail_closed"
  else
    note_fail "language_f128_div_fail_closed_regression"
    tail -30 "$TMP_DIR/lang_div.compile.log" >&2 || true
  fi

  set +e
  "$SOUC" run "$SMOKE" >"$TMP_DIR/madaros.run.log" 2>&1
  m_rc=$?
  set -e
  if [[ "$m_rc" -eq 0 ]] && grep -Fq 'PASS f128_v0e52_language_mul_neg_cmp' "$TMP_DIR/madaros.run.log"; then
    note_pass "madaros_run_language_f128_mul_neg_cmp"
    for want in "${NATIVE_EXPECT[@]}"; do
      if grep -Fq "$want" "$TMP_DIR/madaros.run.log"; then
        note_pass "madaros_native_hex:${want%%=*}"
      else
        note_fail "madaros_native_hex_mismatch:${want%%=*}"
        echo "want $want" >&2
        cat "$TMP_DIR/madaros.run.log" >&2 || true
      fi
    done
  else
    note_fail "madaros_run_language_f128_mul_neg_cmp rc=$m_rc"
    tail -40 "$TMP_DIR/madaros.run.log" >&2 || true
  fi
else
  note_fail "souc_missing"
fi

echo "NOTE v0e52_deferred div=pending f256=pending params_returns_abi=pending print_builtin=pending gum=pending"
echo "NOTE adr009 python_softfloat=not_claim_clock rust=not_claim_clock lean_single_language=greenwash"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e52_language_mul_neg_cmp general=ieee754_mul_neg_cmp anti_f64=green language_desugar=madaros claim_clock=sounio_native_expected"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e52"
  exit 0
fi
echo "FAIL madaros_f128_f256_ladder_gate stage=v0e52" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
