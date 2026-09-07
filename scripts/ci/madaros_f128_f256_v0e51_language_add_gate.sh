#!/usr/bin/env bash
# madaros_f128_f256_v0e51_language_add_gate.sh — V0-E.5.1 language f128 +/− desugar.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (V0-E.5.1 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
# Claim clock: ADR-008 / ADR-009 — oracle_class=sounio_native_expected
#
# V0-E.5.1 green (this gate):
#   - Seed-run of the four-limb ABI wrappers (no language f128; seed still greenwashes)
#   - Madaros-run of language `f128` `+`/`-` desugared to stdlib softfloat
#   - Honest literals: dyadic table (1, 2, 0.5) plus `f128_from_limbs` for ~1e-20
#   - Anti-f64 recover ((1+~1e-20)-1 ≠ 0) with native hex expecteds
#   - Off-table 0.5+1 and 2+tiny
#   - Language f128 params/mul (f128_v0e2) still V0-E.4.1 fail-closed
#
# Explicitly NOT claimed:
#   - lean_single language f128 (still f64 greenwash)
#   - mul/div/neg/cmp language ops, f256, params/returns ABI, print_f128 builtin, GUM
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"
SEED_COMPILER="$(realpath "${SOUNIO_F128_SEED_COMPILER:-$ROOT_DIR/bin/souc-lean-single-x86_64}")"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e51.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

echo "=== madaros_f128_f256_ladder_gate stage=v0e51 ==="
echo "slice=v0e51_language_f128_add_sub_desugar"
echo "claim_clock=sounio_native_expected"
echo "adr=ADR-008+ADR-009"
echo "seed_elf=$SEED_COMPILER"

NATIVE_EXPECT=(
  "wire_half+one=0000000000000000:3fff800000000000"
  "wire_1+tiny=00002f3942192484:3fff000000000000"
  "wire_recover=0000000000000000:3fbc79ca10c92420"
  "wire_two+tiny=0000179ca10c9242:4000000000000000"
)

SF=stdlib/math/softfloat_f128.sio
WF=stdlib/math/wide_float.sio
if [[ -f "$SF" ]] \
  && grep -Fq 'pub fn f128_soft_add_limbs' "$SF" \
  && grep -Fq 'pub fn f128_soft_sub_limbs' "$SF" \
  && grep -Fq 'pub fn f128_from_limbs' "$SF" \
  && grep -Fq 'pub fn f128_to_lo' "$SF" \
  && grep -Fq 'pub fn f128_to_hi' "$SF" \
  && ! grep -Fq 'f128_bits_soft_add' "$WF"; then
  note_pass "stdlib_limb_abi_and_from_limbs_present"
else
  note_fail "stdlib_limb_abi_missing_or_collides_with_wide_float"
fi

if grep -Fq 'fn lower_f128_call_soft_limbs' self-hosted/ir/lower.sio \
  && grep -Fq 'fn lower_let_stmt_f128_ref' self-hosted/ir/lower.sio \
  && grep -Fq 'no f64 widen' self-hosted/ir/lower.sio; then
  note_pass "lower_f128_desugar_markers"
else
  note_fail "lower_f128_desugar_markers_missing"
fi

SMOKE=tests/run-pass/f128_v0e51_language_add_sub.sio
if grep -Fq 'use math::softfloat_f128::{' "$SMOKE" \
  && grep -Fq 'f128_soft_add_limbs' "$SMOKE" \
  && grep -Fq 'f128_soft_sub_limbs' "$SMOKE" \
  && grep -Fq 'let one: f128' "$SMOKE" \
  && grep -Fq 'f128_from_limbs' "$SMOKE" \
  && grep -Fq 'one + tiny' "$SMOKE" \
  && grep -Fq 'one_plus_tiny - one' "$SMOKE" \
  && ! grep -Fq 'fn f128_bits_soft_add' "$SMOKE"; then
  note_pass "smoke_is_language_f128_not_f128bits_copy"
else
  note_fail "smoke_must_be_language_f128_desugar"
fi

REFUSE_SENTINEL='f128/f256 Madaros-run softfloat lowering is not implemented (V0-E.4.1 fail-closed; no f64 greenwash)'

cat >"$TMP_DIR/seed_limbs.sio" <<'EOF'
use math::softfloat_f128::{f128_soft_add_limbs, f128_soft_sub_limbs}
use math::wide_float::print_f128_bits_hex_wire

fn main() -> i32 with IO, Mut, Panic, Div {
    let half_plus_one = f128_soft_add_limbs(0, 4611123068473966592, 0, 4611404543450677248)
    let one_plus_tiny = f128_soft_add_limbs(0, 4611404543450677248, 3746994889972252672, 4592679628783035426)
    let recover = f128_soft_sub_limbs(one_plus_tiny.lo, one_plus_tiny.hi, 0, 4611404543450677248)
    let two_plus_tiny = f128_soft_add_limbs(0, 4611686018427387904, 3746994889972252672, 4592679628783035426)
    print("wire_half+one=")
    print_f128_bits_hex_wire(half_plus_one)
    println("")
    print("wire_1+tiny=")
    print_f128_bits_hex_wire(one_plus_tiny)
    println("")
    print("wire_recover=")
    print_f128_bits_hex_wire(recover)
    println("")
    print("wire_two+tiny=")
    print_f128_bits_hex_wire(two_plus_tiny)
    println("")
    println("PASS f128_v0e51_seed_limbs")
    return 0
}
EOF

ELF="$TMP_DIR/seed_limbs.elf"
BLOG="$TMP_DIR/seed.build.log"
RLOG="$TMP_DIR/seed.run.log"

if [[ ! -x "$SEED_COMPILER" ]]; then
  note_fail "seed_missing"
elif [[ "$(head -c2 "$SEED_COMPILER" 2>/dev/null)" == '#!' ]]; then
  note_fail "seed_is_wrapper_not_elf"
else
  set +e
  "$SEED_COMPILER" "$TMP_DIR/seed_limbs.sio" "$ELF" >"$BLOG" 2>&1
  b_rc=$?
  set -e
  if [[ "$b_rc" -ne 0 || ! -f "$ELF" ]]; then
    note_fail "seed_build_limb_abi"
    tail -40 "$BLOG" >&2 || true
  else
    chmod +x "$ELF"
    set +e
    "$ELF" >"$RLOG" 2>&1
    r_rc=$?
    set -e
    if [[ "$r_rc" -ne 0 ]] || ! grep -Fq 'PASS f128_v0e51_seed_limbs' "$RLOG"; then
      note_fail "seed_run_limb_abi"
      cat "$RLOG" >&2 || true
    else
      note_pass "seed_run_limb_abi"
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

if [[ -x "$SOUC" ]]; then
  set +e
  "$SOUC" compile tests/run-pass/f128_v0e2_arith_check.sio -o "$TMP_DIR/lang.elf" >"$TMP_DIR/lang.compile.log" 2>&1
  set -e
  if grep -Fq "$REFUSE_SENTINEL" "$TMP_DIR/lang.compile.log"; then
    note_pass "language_f128_params_mul_still_fail_closed"
  else
    note_fail "language_f128_v0e2_fail_closed_regression"
    tail -30 "$TMP_DIR/lang.compile.log" >&2 || true
  fi

  set +e
  "$SOUC" run "$SMOKE" >"$TMP_DIR/madaros.run.log" 2>&1
  m_rc=$?
  set -e
  if [[ "$m_rc" -eq 0 ]] && grep -Fq 'PASS f128_v0e51_language_add_sub' "$TMP_DIR/madaros.run.log"; then
    note_pass "madaros_run_language_f128_add_sub"
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
    note_fail "madaros_run_language_f128_add_sub"
    tail -40 "$TMP_DIR/madaros.run.log" >&2 || true
  fi
else
  note_fail "souc_missing"
fi

echo "NOTE v0e51_deferred mul_div_neg_cmp=pending f256=pending params_returns_abi=pending print_builtin=pending gum=pending"
echo "NOTE adr009 python_softfloat=not_claim_clock rust=not_claim_clock lean_single_language=greenwash"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e51_language_add_sub general=ieee754_add_sub anti_f64=green language_desugar=madaros claim_clock=sounio_native_expected"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e51"
  exit 0
fi
echo "FAIL madaros_f128_f256_ladder_gate stage=v0e51" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
