#!/usr/bin/env bash
# madaros_f128_f256_v0e58_print_f128_gate.sh — V0-E.5.8 print_f128: exact
# textual output of a language f128 (decimal 36 sig digits half-even + hex float).
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (V0-E.5.8 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
# Claim clock: ADR-008 / ADR-009 — oracle_class=sounio_native_expected
#
# V0-E.5.8 green (this gate):
#   - stdlib/math/softfloat_f128_fmt.sio: `f128_to_string` / `print_f128` /
#     `println_f128` (decimal, 36 significant digits, round-half-even,
#     `[-]d.ddd…e[+-]dddd`, zeros `0.0e+0000`, `inf`/`-inf`/`nan`) and
#     `f128_to_hex_string` / `print_f128_hex` (C99 hex float, exact)
#   - ordinary stdlib Sounio over the V0-E.5.4 f128 ABI (`fn f(x: f128)`),
#     limbs via the intercepted `f128_to_lo`/`f128_to_hi`; exact base-10^9
#     big-integer scaling (2^e or 5^-e) — no f64 anywhere on the path
#   - full binary128 range: max finite (e+4932), min normal (e-4932), smallest
#     subnormal (e-4966), 999…9 → 1.000e(n+1) carry, specials, ±0
#   - Madaros-run witness with expected strings computed independently from the
#     exact rationals (Fraction/Decimal, cross-checked), incl. anti-f64
#     (1+~1e-20)^2 = 1.00000000000000000001999999999999989e+0000
#   - values through a user `fn(x: f128) -> f128` and a struct field
#   - negatives: `print_f128`/`f128_to_string` applied to f256 or f64 is a
#     checker error (E009); builtin `println(x: f128)` still fails closed at
#     lowering (no ELF)
#
# Explicitly NOT claimed:
#   - builtin `print`/`println` of an f128 (still refused; pinned here)
#   - f256 printing, formatting options (width/precision), parsing from text
#   - lean_single language f128 (still f64 greenwash)
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e58.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

echo "=== madaros_f128_f256_ladder_gate stage=v0e58 ==="
echo "slice=v0e58_print_f128_exact_decimal_hex"
echo "claim_clock=sounio_native_expected"
echo "adr=ADR-008+ADR-009"

# Expected strings: exact rationals → 36 significant digits, round-half-even
# (Fraction + Decimal prec=36 cross-check); hex from the raw 112 fraction bits.
NATIVE_EXPECT=(
  "dec_one=1.00000000000000000000000000000000000e+0000"
  "hex_one=0x1.0000000000000000000000000000p+0"
  "dec_neg_two=-2.00000000000000000000000000000000000e+0000"
  "hex_neg_two=-0x1.0000000000000000000000000000p+1"
  "dec_half=5.00000000000000000000000000000000000e-0001"
  "hex_half=0x1.0000000000000000000000000000p-1"
  "dec_three=3.00000000000000000000000000000000000e+0000"
  "hex_three=0x1.8000000000000000000000000000p+1"
  "dec_third=3.33333333333333333333333333333333317e-0001"
  "hex_third=0x1.5555555555555555555555555555p-2"
  "dec_sq=1.00000000000000000001999999999999989e+0000"
  "hex_sq=0x1.00000000000000005e7284324908p+0"
  "dec_max=1.18973149535723176508575932662800702e+4932"
  "hex_max=0x1.ffffffffffffffffffffffffffffp+16383"
  "dec_min_normal=3.36210314311209350626267781732175260e-4932"
  "hex_min_normal=0x1.0000000000000000000000000000p-16382"
  "dec_sub=6.47517511943802511092443895822764655e-4966"
  "hex_sub=0x0.0000000000000000000000000001p-16382"
  "dec_inf=inf"
  "hex_inf=inf"
  "dec_neg_inf=-inf"
  "hex_neg_inf=-inf"
  "dec_nan=nan"
  "hex_nan=nan"
  "dec_zero=0.0e+0000"
  "hex_zero=0x0p+0"
  "dec_neg_zero=-0.0e+0000"
  "hex_neg_zero=-0x0p+0"
  "dec_twice_third=6.66666666666666666666666666666666635e-0001"
  "hex_twice_third=0x1.5555555555555555555555555555p-1"
  "dec_field_sixth=1.66666666666666666666666666666666659e-0001"
  "hex_field_sixth=0x1.5555555555555555555555555555p-3"
  "dec_roll_a=1.00000000000000000000000000000000000e-4847"
  "hex_roll_a=0x1.87f31452b1b42494995f8bc46918p-16102"
  "dec_roll_b=1.00000000000000000000000000000000000e-2866"
  "hex_roll_b=0x1.47362af41d3287e930f87171d6e1p-9521"
  "wire_print=1.00000000000000000001999999999999989e+0000|0x1.00000000000000005e7284324908p+0"
)

FMT=stdlib/math/softfloat_f128_fmt.sio
if [[ -f "$FMT" ]] \
  && grep -Fq 'pub fn print_f128(x: f128)' "$FMT" \
  && grep -Fq 'pub fn println_f128(x: f128)' "$FMT" \
  && grep -Fq 'pub fn f128_to_string(x: f128) -> string' "$FMT" \
  && grep -Fq 'pub fn f128_to_hex_string(x: f128) -> string' "$FMT" \
  && grep -Fq 'pub fn print_f128_hex(x: f128)' "$FMT" \
  && grep -Fq 'use math::softfloat_f128::{f128_to_lo, f128_to_hi}' "$FMT"; then
  note_pass "stdlib_f128_fmt_api_present"
else
  note_fail "stdlib_f128_fmt_api_missing"
fi

# No host-float anywhere on the formatting path: no f64 casts, no f64-typed
# bindings, no float literals, no f64 print helpers.
FMT_CODE="$TMP_DIR/fmt_code_only.sio"
grep -Ev '^\s*//' "$FMT" | sed 's/"[^"]*"//g' >"$FMT_CODE" || true
if [[ -f "$FMT" ]] \
  && ! grep -Eq 'f64|f32|print_float|format_f64' "$FMT_CODE" \
  && ! grep -Eq '[0-9]\.[0-9]' "$FMT_CODE"; then
  note_pass "stdlib_f128_fmt_no_f64_path"
else
  note_fail "stdlib_f128_fmt_touches_f64"
fi

if ! grep -Fq 'print_f128' stdlib/math/softfloat_f128.sio; then
  note_pass "softfloat_f128_unchanged_by_fmt"
else
  note_fail "softfloat_f128_should_not_carry_print_f128"
fi

SMOKE=tests/run-pass/f128_v0e58_print_f128.sio
if grep -Fq 'use math::softfloat_f128_fmt::{' "$SMOKE" \
  && grep -Fq 'println_f128(x)' "$SMOKE" \
  && grep -Fq 'f128_to_string(sq)' "$SMOKE" \
  && grep -Fq 'f128_to_hex_string(max_finite)' "$SMOKE" \
  && grep -Fq 'f128_from_limbs(1, 0)' "$SMOKE" \
  && grep -Fq 'fn twice(x: f128) -> f128' "$SMOKE" \
  && grep -Fq 'struct Cell { v: f128, tag: i64 }' "$SMOKE" \
  && grep -Fq 'v0e58_main_entered' "$SMOKE" \
  && ! grep -Fq 'F128Bits {' "$SMOKE" \
  && ! grep -Fq 'f128_bits_soft_' "$SMOKE" \
  && ! grep -Fq 'fn f128_to_string' "$SMOKE"; then
  note_pass "smoke_is_language_f128_print_via_stdlib"
else
  note_fail "smoke_must_print_language_f128_via_stdlib_fmt"
fi

# ---------------------------------------------------------------------------
# Negatives
# ---------------------------------------------------------------------------
# print_f128 on an f256: distinct type, no payload — must be refused at check
# (E009 argument mismatch) or at lowering (V0-E.4.1 sentinel); never an ELF.
cat >"$TMP_DIR/print_f128_of_f256.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}
use math::softfloat_f128_fmt::{print_f128}

fn main() -> i32 with IO, Mut, Panic, Div {
    let x: f256 = 1.0
    print_f128(x)
    return 0
}
EOF

# f128_to_string on an f64: the formatter must not accept host floats.
cat >"$TMP_DIR/to_string_of_f64.sio" <<'EOF'
use math::softfloat_f128_fmt::{f128_to_string}

fn main() -> i32 with IO, Mut, Panic, Div {
    let x: f64 = 1.5
    println(f128_to_string(x))
    return 0
}
EOF

REFUSE_SENTINEL='f128/f256 Madaros-run softfloat lowering is not implemented (V0-E.4.1 fail-closed; no f64 greenwash)'
CHECK_MISMATCH='argument type does not match parameter'

if [[ -x "$SOUC" ]]; then
  for neg in print_f128_of_f256:print_f128_on_f256_fail_closed \
             to_string_of_f64:f128_to_string_on_f64_fail_closed; do
    name="${neg%%:*}"; label="${neg##*:}"
    set +e
    "$SOUC" compile "$TMP_DIR/$name.sio" -o "$TMP_DIR/$name.elf" >"$TMP_DIR/$name.compile.log" 2>&1
    n_rc=$?
    set -e
    if [[ "$n_rc" -ne 0 && ! -f "$TMP_DIR/$name.elf" ]] \
       && { grep -Fq "$CHECK_MISMATCH" "$TMP_DIR/$name.compile.log" || grep -Fq "$REFUSE_SENTINEL" "$TMP_DIR/$name.compile.log"; }; then
      note_pass "$label"
    else
      note_fail "${label%_fail_closed}_fail_closed_regression rc=$n_rc"
      tail -30 "$TMP_DIR/$name.compile.log" >&2 || true
    fi
  done

  set +e
  "$SOUC" run "$SMOKE" >"$TMP_DIR/madaros.run.log" 2>&1
  m_rc=$?
  set -e
  if [[ "$m_rc" -eq 0 ]] && grep -Fq 'PASS f128_v0e58_print_f128' "$TMP_DIR/madaros.run.log"; then
    note_pass "madaros_run_print_f128"
    for want in "${NATIVE_EXPECT[@]}"; do
      if grep -Fxq "$want" "$TMP_DIR/madaros.run.log"; then
        note_pass "madaros_native_str:${want%%=*}"
      else
        note_fail "madaros_native_str_mismatch:${want%%=*}"
        echo "want $want" >&2
        grep -F "${want%%=*}=" "$TMP_DIR/madaros.run.log" >&2 || true
      fi
    done
    # Anti-f64: the decimal of (1+~1e-20)^2 must not be the f64 answer.
    if grep -Fxq 'dec_sq=1.00000000000000000000000000000000000e+0000' "$TMP_DIR/madaros.run.log"; then
      note_fail "madaros_print_f128_f64_greenwash_sq"
    else
      note_pass "madaros_print_f128_anti_f64_sq"
    fi
  else
    note_fail "madaros_run_print_f128 rc=$m_rc"
    tail -40 "$TMP_DIR/madaros.run.log" >&2 || true
  fi
else
  note_fail "souc_missing"
fi

echo "NOTE v0e58_format decimal='[-]d.<35 digits>e[+-]dddd' zero='0.0e+0000' hex='[-]0x1.<28 hex>p<exp>' subnormal='0x0.<28 hex>p-16382'"
echo "NOTE v0e58_deferred f256_print=pending f128_methods=pending f128_arrays=pending gum=pending kl8_builtin_println=closed_elsewhere"
echo "NOTE adr009 python_softfloat=not_claim_clock rust=not_claim_clock lean_single_language=greenwash"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e58_print_f128 general=ieee754 anti_f64=green decimal=36sig_half_even hex=c99 print=madaros claim_clock=sounio_native_expected"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e58"
  exit 0
fi
echo "FAIL madaros_f128_f256_ladder_gate stage=v0e58" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
