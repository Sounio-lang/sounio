#!/usr/bin/env bash
# madaros_f128_f256_v0e53_language_div_gate.sh — V0-E.5.3 language f128 `/`.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (V0-E.5.3 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
# Claim clock: ADR-008 / ADR-009 — oracle_class=sounio_native_expected
#
# V0-E.5.3 green (this gate):
#   - Seed-run of the F128Bits div routine (no language f128; seed still greenwashes)
#   - Madaros-run of language `f128` `/` desugared to stdlib softfloat
#   - 117-bit restoring long division with RNE: 1/3, 1/tiny (~1e20), -1/3, tiny/2
#   - Anti-f64: 1/(1+~1e-20) = 1-~1e-20 (f64 gives 1.0)
#   - IEEE corners: x/0 → ±inf, 0/0 → NaN unordered, (1/3)*3 RNE tie → 1
#   - Language f256 params and inexact f128 literals (0.1) still fail-closed
#
# Explicitly NOT claimed:
#   - lean_single language f128 (still f64 greenwash)
#   - f256, params/returns ABI, print_f128 builtin, GUM, sqrt/fma
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"
SEED_COMPILER="$(realpath "${SOUNIO_F128_SEED_COMPILER:-$ROOT_DIR/bin/souc-lean-single-x86_64}")"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e53.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

echo "=== madaros_f128_f256_ladder_gate stage=v0e53 ==="
echo "slice=v0e53_language_f128_div_desugar"
echo "claim_clock=sounio_native_expected"
echo "adr=ADR-008+ADR-009"
echo "seed_elf=$SEED_COMPILER"

NATIVE_EXPECT=(
  "wire_1/3=5555555555555555:3ffd555555555555"
  "wire_1/tiny=01aec5d1d43b6865:40415af1d78b58c4"
  "wire_1/(1+tiny)=ffffa18d7bcdb6f8:3ffeffffffffffff"
  "wire_-1/3=5555555555555555:bffd555555555555"
  "wire_tiny/2=3400000000000000:3fbb79ca10c92422"
)

SF=stdlib/math/softfloat_f128.sio
if [[ -f "$SF" ]] \
  && grep -Fq 'pub fn f128_bits_soft_div' "$SF" \
  && grep -Fq 'pub fn f128_soft_div_limbs' "$SF" \
  && grep -Fq 'fn sf_u128_div_wide' "$SF"; then
  note_pass "stdlib_div_present"
else
  note_fail "stdlib_div_missing"
fi

if grep -Fq 'fn lower_f128_binary_target' self-hosted/ir/lower.sio \
  && grep -Fq 'f128_bits_soft_div' self-hosted/ir/lower.sio \
  && grep -Fq 'f128_bits_soft_div' self-hosted/check/specializer.sio; then
  note_pass "lower_f128_div_markers"
else
  note_fail "lower_f128_div_markers_missing"
fi

SMOKE=tests/run-pass/f128_v0e53_language_div.sio
if grep -Fq 'use math::softfloat_f128::{' "$SMOKE" \
  && grep -Fq 'f128_bits_soft_div' "$SMOKE" \
  && grep -Fq 'let one: f128' "$SMOKE" \
  && grep -Fq 'v0e53_main_entered' "$SMOKE" \
  && grep -Fq 'one / three' "$SMOKE" \
  && grep -Fq 'one / tiny' "$SMOKE" \
  && grep -Fq 'one / one_plus_tiny' "$SMOKE" \
  && grep -Fq 'one / zero' "$SMOKE" \
  && grep -Fq 'zero / zero' "$SMOKE" \
  && ! grep -Fq 'fn f128_bits_soft_div' "$SMOKE"; then
  note_pass "smoke_is_language_f128_not_f128bits_copy"
else
  note_fail "smoke_must_be_language_f128_desugar"
fi

REFUSE_SENTINEL='f128/f256 Madaros-run softfloat lowering is not implemented (V0-E.4.1 fail-closed; no f64 greenwash)'

cat >"$TMP_DIR/seed_bits.sio" <<'EOF'
use math::softfloat_f128::{f128_bits_soft_div, f128_soft_div_limbs}
use math::wide_float::{print_f128_bits_hex_wire, F128Bits}

fn main() -> i32 with IO, Mut, Panic, Div {
    let one = F128Bits { lo: 0, hi: 4611404543450677248 }
    let three = F128Bits { lo: 0, hi: 4611826755915743232 }
    let neg_one = F128Bits { lo: 0, hi: 0 - 4611967493404098560 }
    let tiny = F128Bits { lo: 3746994889972252672, hi: 4592679628783035426 }
    let one_plus_tiny = F128Bits { lo: 51922968585348, hi: 4611404543450677248 }
    let third = f128_bits_soft_div(one, three)
    let big = f128_bits_soft_div(one, tiny)
    let recip = f128_bits_soft_div(one, one_plus_tiny)
    let neg_third = f128_bits_soft_div(neg_one, three)
    let half_tiny = f128_soft_div_limbs(3746994889972252672, 4592679628783035426, 0, 4611686018427387904)
    print("wire_1/3=")
    print_f128_bits_hex_wire(third)
    println("")
    print("wire_1/tiny=")
    print_f128_bits_hex_wire(big)
    println("")
    print("wire_1/(1+tiny)=")
    print_f128_bits_hex_wire(recip)
    println("")
    print("wire_-1/3=")
    print_f128_bits_hex_wire(neg_third)
    println("")
    print("wire_tiny/2=")
    print_f128_bits_hex_wire(half_tiny)
    println("")
    println("PASS f128_v0e53_seed_bits")
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
    note_fail "seed_build_bits_div"
    tail -40 "$BLOG" >&2 || true
  else
    chmod +x "$ELF"
    set +e
    "$ELF" >"$RLOG" 2>&1
    r_rc=$?
    set -e
    if [[ "$r_rc" -ne 0 ]] || ! grep -Fq 'PASS f128_v0e53_seed_bits' "$RLOG"; then
      note_fail "seed_run_bits_div"
      cat "$RLOG" >&2 || true
    else
      note_pass "seed_run_bits_div"
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

# `%` on f128 is refused earlier by check (E050), so the lowering-level
# fail-closed negative is an inexact literal: 0.1 must never be f64-widened.
cat >"$TMP_DIR/lang_inexact.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi, f128_bits_soft_mul}

fn main() -> i32 with IO, Mut, Panic, Div {
    let tenth: f128 = 0.1
    let two: f128 = 2.0
    let fifth: f128 = tenth * two
    if f128_to_hi(fifth) == 0 { return 1 }
    return 0
}
EOF

# V0-E.5.4 landed the f128 params/returns ABI, so the V0-E.2 fixture is no
# longer a fail-closed negative. The lowering-level negative for "wide float
# through a fn boundary" is now f256: same-format ops typecheck, no payload.
cat >"$TMP_DIR/lang_f256_params.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

fn add256(a: f256, b: f256) -> f256 { a + b }

fn main() -> i32 with IO, Mut, Panic, Div {
    let x: f256 = 1.0
    let y: f256 = add256(x, x)
    let z: f256 = y * x
    return 0
}
EOF

if [[ -x "$SOUC" ]]; then
  set +e
  "$SOUC" compile "$TMP_DIR/lang_f256_params.sio" -o "$TMP_DIR/lang.elf" >"$TMP_DIR/lang.compile.log" 2>&1
  set -e
  if grep -Fq "$REFUSE_SENTINEL" "$TMP_DIR/lang.compile.log"; then
    note_pass "language_f256_params_still_fail_closed"
  else
    note_fail "language_f256_params_fail_closed_regression"
    tail -30 "$TMP_DIR/lang.compile.log" >&2 || true
  fi

  set +e
  "$SOUC" compile "$TMP_DIR/lang_inexact.sio" -o "$TMP_DIR/lang_inexact.elf" >"$TMP_DIR/lang_inexact.compile.log" 2>&1
  d_rc=$?
  set -e
  if [[ "$d_rc" -ne 0 ]] && grep -Fq 'no f64 widen' "$TMP_DIR/lang_inexact.compile.log"; then
    note_pass "language_f128_inexact_literal_still_fail_closed"
  else
    note_fail "language_f128_inexact_literal_fail_closed_regression"
    tail -30 "$TMP_DIR/lang_inexact.compile.log" >&2 || true
  fi

  set +e
  "$SOUC" run "$SMOKE" >"$TMP_DIR/madaros.run.log" 2>&1
  m_rc=$?
  set -e
  if [[ "$m_rc" -eq 0 ]] && grep -Fq 'PASS f128_v0e53_language_div' "$TMP_DIR/madaros.run.log"; then
    note_pass "madaros_run_language_f128_div"
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
    note_fail "madaros_run_language_f128_div rc=$m_rc"
    tail -40 "$TMP_DIR/madaros.run.log" >&2 || true
  fi
else
  note_fail "souc_missing"
fi

echo "NOTE v0e53_deferred f256=pending params_returns_abi=see_v0e54 print_builtin=pending gum=pending"
echo "NOTE adr009 python_softfloat=not_claim_clock rust=not_claim_clock lean_single_language=greenwash"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e53_language_div general=ieee754_div anti_f64=green language_desugar=madaros claim_clock=sounio_native_expected"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e53"
  exit 0
fi
echo "FAIL madaros_f128_f256_ladder_gate stage=v0e53" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
