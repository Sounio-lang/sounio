#!/usr/bin/env bash
# madaros_f128_f256_v0e59_exact_literals_gate.sh — V0-E.5.9 exact f128
# literals: decimal and C99 hex-float text → binary128 limbs, no f64 anywhere.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (V0-E.5.9 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
# Claim clock: ADR-008 / ADR-009 — oracle_class=sounio_native_expected
#
# V0-E.5.9 green (this gate):
#   - the parser stamps every FloatLit node with its exact binary128 identity
#     (Expr.f128_lit_lo / f128_lit_hi / f128_lit_exact), read from the SOURCE
#     BYTES by self-hosted/parser/f128_literal.sio — decimal d.ddd e±N and
#     C99 hex-float 0xh.hhh p±N — never from the literal's f64 value
#   - the lowerer emits those limbs in f128 positions (let, struct literal,
#     operand, argument) and fails closed on f128_lit_exact == false
#   - the V0-E.5.1 8-entry literal table (0, ±1, 2, 3, 0.5, 1.5, 0.25) is gone:
#     `4.0`, `0.125`, `1e3`, `1e23`, `2^60+1`, `1e38` are exact
#   - anti-f64: 1e23 and 1152921504606846977.0 (2^60+1) carry limbs a binary64
#     detour cannot produce; 0x1.0000000000000000000000000001p+0 is 1 + 2^-112
#   - hex-float greenwash closed: 0x1.8p+0 used to lower as the parser's
#     placeholder magnitude; it is 1.5 now, or nothing
#   - inexact text still fails closed: 0.1, 2.5e-3, 3.3, a 26-digit
#     non-dyadic decimal, a 29-hex-digit mantissa, 0x1p+16384
#
# Explicitly NOT claimed:
#   - lean_single language f128 (still f64 greenwash)
#   - the f64 path's hex-float magnitude (still the placeholder; not this rung)
#   - f256 literals (V0-E.4.1 sentinel unchanged), `%`, `+=`, GUM
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e59.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

echo "=== madaros_f128_f256_ladder_gate stage=v0e59 ==="
echo "slice=v0e59_exact_f128_literals_decimal_hexfloat"
echo "claim_clock=sounio_native_expected"
echo "adr=ADR-008+ADR-009"

NATIVE_EXPECT=(
  "wire_e23=404b52d02c7e14af:6800000000000000"
  "wire_hex_max=7ffeffffffffffff:ffffffffffffffff"
  "dec_e23=1.00000000000000000000000000000000000e+0023"
  "dec_two60p1=1.15292150460684697700000000000000000e+0018"
  "dec_hex_1p112=1.00000000000000000000000000000000019e+0000"
  "dec_e38=1.00000000000000000000000000000000000e+0038"
)

LITMOD=self-hosted/parser/f128_literal.sio
LOWER=self-hosted/ir/lower.sio
if [[ -f "$LITMOD" ]] \
  && grep -Fq 'pub fn f128_literal_from_source(start: i64, end: i64) -> F128Lit' "$LITMOD" \
  && grep -Fq 'fn f128lit_from_decimal(' "$LITMOD" \
  && grep -Fq 'fn f128lit_from_hex(' "$LITMOD" \
  && grep -Fq 'fn f128lit_pack(' "$LITMOD" \
  && ! grep -v '^[[:space:]]*//' "$LITMOD" | grep -Eq '(^|[^a-z_])f64([^a-z_0-9]|$)'; then
  note_pass "parser_f128_literal_module_no_f64"
else
  note_fail "parser_f128_literal_module_missing_or_touches_f64"
fi

if grep -Fq 'f128_lit_lo: i64,' self-hosted/parser/ast.sio \
  && grep -Fq 'f128_lit_hi: i64,' self-hosted/parser/ast.sio \
  && grep -Fq 'f128_lit_exact: bool,' self-hosted/parser/ast.sio \
  && grep -Fq 'f128_literal_from_source(span.start, span.end)' self-hosted/parser/exprs.sio \
  && grep -Fq 'f128_literal_from_source(start.start, parser_prev_token_end(p))' self-hosted/parser/exprs.sio; then
  note_pass "expr_carries_f128_literal_identity"
else
  note_fail "expr_f128_literal_identity_fields_missing"
fi

if ! grep -Fq 'fn lower_f128_literal_is_exact' "$LOWER" \
  && ! grep -Fq 'fn lower_f128_literal_hi' "$LOWER" \
  && grep -Fq 'if !(*e).f128_lit_exact {' "$LOWER" \
  && grep -Fq 'self.lower_f128_emit_limbs((*e).f128_lit_lo, (*e).f128_lit_hi)' "$LOWER" \
  && grep -Fq 'V0-E.5.9; no f64 widen' "$LOWER"; then
  note_pass "lower_f128_literal_table_removed"
else
  note_fail "lower_f128_literal_table_still_present_or_consumer_missing"
fi

SMOKE=tests/run-pass/f128_v0e59_language_exact_literals.sio
if grep -Fq 'use math::softfloat_f128::{' "$SMOKE" \
  && grep -Fq 'let four: f128 = 4.0' "$SMOKE" \
  && grep -Fq 'let e23: f128 = 1e23' "$SMOKE" \
  && grep -Fq 'let two60p1: f128 = 1152921504606846977.0' "$SMOKE" \
  && grep -Fq 'let h1p112: f128 = 0x1.0000000000000000000000000001p+0' "$SMOKE" \
  && grep -Fq 'let hmax: f128 = 0x1.ffffffffffffffffffffffffffffp+16383' "$SMOKE" \
  && grep -Fq 'let hsub: f128 = 0x0.0000000000000000000000000001p-16382' "$SMOKE" \
  && grep -Fq 'let h15: f128 = 0x1.8p+0' "$SMOKE" \
  && grep -Fq 'v0e59_main_entered' "$SMOKE" \
  && ! grep -Fq 'F128Bits {' "$SMOKE" \
  && ! grep -Fq 'f128_bits_soft_' "$SMOKE"; then
  note_pass "smoke_is_language_f128_literals_not_f128bits_copy"
else
  note_fail "smoke_must_be_language_f128_literals"
fi

REFUSE_SENTINEL='f128/f256 Madaros-run softfloat lowering is not implemented (V0-E.4.1 fail-closed; no f64 greenwash)'
INEXACT_SENTINEL='V0-E.5.9; no f64 widen'

mk_neg() {
  # $1 = name, $2 = the literal text placed in an f128 let
  cat >"$TMP_DIR/$1.sio" <<EOF
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

fn main() -> i32 with IO, Mut, Panic, Div {
    let x: f128 = $2
    if f128_to_hi(x) == 0 { return 1 }
    return 0
}
EOF
}
mk_neg lit_tenth '0.1'
mk_neg lit_2p5e_3 '2.5e-3'
mk_neg lit_3p3 '3.3'
mk_neg lit_long_nondyadic '1.0000000000000000000000001'
mk_neg lit_hex_29_digits '0x1.00000000000000000000000000001p+0'
mk_neg lit_hex_overflow '0x1p+16384'
mk_neg lit_hex_sub_lost '0x1p-16495'

# f256 literal: still the V0-E.4.1 sentinel (no f256 payload exists).
cat >"$TMP_DIR/lit_f256.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

fn main() -> i32 with IO, Mut, Panic, Div {
    let x: f256 = 4.0
    let y: f256 = x * x
    return 0
}
EOF

# The closed greenwash, as a Madaros-run probe: 0x1.8p+0 must be 1.5 (hi
# 0x3fff800000000000), not the old placeholder (1.0 / 0.0).
cat >"$TMP_DIR/hex_not_placeholder.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

fn main() -> i32 with IO, Mut, Panic, Div {
    let h: f128 = 0x1.8p+0
    if f128_to_lo(h) != 0 { return 1 }
    if f128_to_hi(h) == 4611404543450677248 { return 2 }
    if f128_to_hi(h) == 0 { return 3 }
    if f128_to_hi(h) != 4611545280939032576 { return 4 }
    return 0
}
EOF

if [[ -x "$SOUC" ]]; then
  for neg in lit_tenth:"$INEXACT_SENTINEL":literal_tenth_still_fail_closed \
             lit_2p5e_3:"$INEXACT_SENTINEL":literal_2p5e_3_still_fail_closed \
             lit_3p3:"$INEXACT_SENTINEL":literal_3p3_still_fail_closed \
             lit_long_nondyadic:"$INEXACT_SENTINEL":literal_long_nondyadic_still_fail_closed \
             lit_hex_29_digits:"$INEXACT_SENTINEL":literal_hex_29_digits_still_fail_closed \
             lit_hex_overflow:"$INEXACT_SENTINEL":literal_hex_overflow_still_fail_closed \
             lit_hex_sub_lost:"$INEXACT_SENTINEL":literal_hex_subnormal_bits_lost_still_fail_closed \
             lit_f256:"$REFUSE_SENTINEL":language_f256_literal_still_fail_closed; do
    name="${neg%%:*}"; rest="${neg#*:}"; want="${rest%:*}"; label="${rest##*:}"
    set +e
    "$SOUC" compile "$TMP_DIR/$name.sio" -o "$TMP_DIR/$name.elf" >"$TMP_DIR/$name.compile.log" 2>&1
    n_rc=$?
    set -e
    if [[ "$n_rc" -ne 0 ]] && grep -Fq "$want" "$TMP_DIR/$name.compile.log" && [[ ! -s "$TMP_DIR/$name.elf" ]]; then
      note_pass "$label"
    else
      note_fail "${label%_still_fail_closed}_fail_closed_regression rc=$n_rc"
      tail -30 "$TMP_DIR/$name.compile.log" >&2 || true
    fi
  done

  set +e
  "$SOUC" run "$TMP_DIR/hex_not_placeholder.sio" >"$TMP_DIR/hex_not_placeholder.run.log" 2>&1
  hp_rc=$?
  set -e
  if [[ "$hp_rc" -eq 0 ]]; then
    note_pass "madaros_run_hex_float_is_not_placeholder"
  else
    note_fail "madaros_run_hex_float_is_not_placeholder rc=$hp_rc"
    tail -30 "$TMP_DIR/hex_not_placeholder.run.log" >&2 || true
  fi

  set +e
  "$SOUC" run "$SMOKE" >"$TMP_DIR/madaros.run.log" 2>&1
  m_rc=$?
  set -e
  if [[ "$m_rc" -eq 0 ]] && grep -Fq 'PASS f128_v0e59_language_exact_literals' "$TMP_DIR/madaros.run.log"; then
    note_pass "madaros_run_language_f128_exact_literals"
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
    note_fail "madaros_run_language_f128_exact_literals rc=$m_rc"
    tail -40 "$TMP_DIR/madaros.run.log" >&2 || true
  fi
else
  note_fail "souc_missing"
fi

echo "NOTE v0e59_deferred f256=pending f64_path_hexfloat_magnitude=placeholder compound_assign=pending gum=pending"
echo "NOTE adr009 python_softfloat=not_claim_clock rust=not_claim_clock lean_single_language=greenwash"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e59_exact_literals general=ieee754 anti_f64=green decimal=dyadic113 hex=c99 literal=madaros claim_clock=sounio_native_expected"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e59"
  exit 0
fi
echo "FAIL madaros_f128_f256_ladder_gate stage=v0e59" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
