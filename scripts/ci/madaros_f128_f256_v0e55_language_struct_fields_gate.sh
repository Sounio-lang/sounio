#!/usr/bin/env bash
# madaros_f128_f256_v0e55_language_struct_fields_gate.sh — V0-E.5.5 language
# f128 struct fields + f128 assignment.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (V0-E.5.5 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
# Claim clock: ADR-008 / ADR-009 — oracle_class=sounio_native_expected
#
# V0-E.5.5 green (this gate):
#   - Madaros-run of structs with `f128` fields: the field slot holds the
#     F128Bits handle (one word, like any struct-typed field; layout unchanged)
#   - a field read is an f128 expression: operand, `let` RHS (annotated or not),
#     nested `b.lo.x`, through `&Struct` and by-value struct params
#   - struct-literal initialisers: a bare `1.0` is binary128, never f64 bits
#   - field assignment `v.x = expr`; plain f128 local assignment `acc = 2.0`
#     (a V0-E.5.1 gap: the generic store put f64 bits in the handle slot) and
#     `acc = y` copies limbs (no aliasing)
#   - DCE: a struct with an f128 field marks the softfloat desugar targets even
#     when the program has no f128 let/param/return (struct_only probe)
#   - Anti-f64 through a field: (1+~1e-20)^2 ≠ 1
#   - KL-8 (2026-09-12, #2491): `+=`/`-=` on a plain f128 local desugar to
#     real softfloat add/sub and store back correctly (`acc: f128 = 1.0;
#     acc += one` gives exactly 2.0, no f64 greenwash) -- updated 2026-09-22,
#     this used to be a fail-closed refusal, superseded by KL-8
#   - f256 fields and inexact literals (0.1) still fail-closed
#
# Explicitly NOT claimed:
#   - lean_single language f128 (still f64 greenwash)
#   - f256, methods returning f128 (`v.norm2()`), arrays of f128, print_f128, GUM
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e55.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

echo "=== madaros_f128_f256_ladder_gate stage=v0e55 ==="
echo "slice=v0e55_language_f128_struct_fields_assign"
echo "claim_clock=sounio_native_expected"
echo "adr=ADR-008+ADR-009"

NATIVE_EXPECT=(
  "wire_dot=0000000000000000:4002200000000000"
  "wire_field_sq=00005e7284324908:3fff000000000000"
  "wire_nested_half=0000000000000000:3ffe000000000000"
  "wire_field_assign_y=0000000000000000:4001800000000000"
)

LOWER=self-hosted/ir/lower.sio
if grep -Fq 'fn lower_field_is_f128_ref' "$LOWER" \
  && grep -Fq 'fn lower_expr_is_f128_slot_ref' "$LOWER" \
  && grep -Fq 'fn lower_expr_struct_type_name_ref' "$LOWER" \
  && grep -Fq 'struct field in f128 position is not declared f128 (V0-E.5.5)' "$LOWER"; then
  note_pass "lower_f128_field_markers"
else
  note_fail "lower_f128_field_markers_missing"
fi

if grep -Fq 'fn spec_dce_struct_has_f128_field' self-hosted/check/specializer.sio; then
  note_pass "dce_f128_struct_field_marker"
else
  note_fail "dce_f128_struct_field_marker_missing"
fi

SMOKE=tests/run-pass/f128_v0e55_language_struct_fields.sio
if grep -Fq 'use math::softfloat_f128::{' "$SMOKE" \
  && grep -Fq 'struct Vec2 { x: f128, y: f128 }' "$SMOKE" \
  && grep -Fq 'struct Box2 { lo: Vec2, hi: Vec2, tag: i64 }' "$SMOKE" \
  && grep -Fq 'let a = Vec2 { x: 1.0, y: 2.0 }' "$SMOKE" \
  && grep -Fq 'bx.lo.x * bx.hi.y' "$SMOKE" \
  && grep -Fq 'fn dot(a: &Vec2, b: &Vec2) -> f128' "$SMOKE" \
  && grep -Fq 'v.x = 3.0' "$SMOKE" \
  && grep -Fq 'acc = 2.0' "$SMOKE" \
  && grep -Fq 'v0e55_main_entered' "$SMOKE" \
  && ! grep -Fq 'F128Bits {' "$SMOKE" \
  && ! grep -Fq 'f128_bits_soft_' "$SMOKE"; then
  note_pass "smoke_is_language_f128_fields_not_f128bits_copy"
else
  note_fail "smoke_must_be_language_f128_fields"
fi

REFUSE_SENTINEL='f128/f256 Madaros-run softfloat lowering is not implemented (V0-E.4.1 fail-closed; no f64 greenwash)'

# f256 field: same-format ops typecheck (V0-E.2) but there is no f256 payload —
# must refuse with the V0-E.4.1 sentinel, never emit an ELF.
cat >"$TMP_DIR/lang_f256_field.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

struct W { v: f256 }

fn main() -> i32 with IO, Mut, Panic, Div {
    let x: f256 = 1.0
    let w = W { v: x }
    let y: f256 = w.v * x
    return 0
}
EOF

# Inexact literal into an f128 field initialiser: 0.1 must never be f64-widened.
cat >"$TMP_DIR/lang_inexact_field.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

struct P { x: f128 }

fn main() -> i32 with IO, Mut, Panic, Div {
    let p = P { x: 0.1 }
    if f128_to_hi(p.x) == 0 { return 1 }
    return 0
}
EOF

# Compound assignment on a plain f128 local: since KL-8 (#2491, 2026-09-12)
# this is real softfloat add/sub, not a refusal. `1.0 += 1.0` alone would not
# distinguish real binary128 arithmetic from an f64-widen-then-narrow cheat,
# since 1.0+1.0=2.0 is exact in both formats -- use the same `tiny` anti-f64
# operand as the rest of this ladder (an f128 value with no f64
# representation) so a greenwashed implementation provably diverges: `1.0 +
# tiny` keeps tiny's low-order bits (nonzero lo limb) where f64-then-widen
# would round it away to exactly 1.0 (zero lo limb). Expected limbs match
# scripts/ci/madaros_f128_f256_v0e4_language_lower_gate.sh's own
# `wire_1+tiny` receipt for the identical `1 + tiny` computation. Checked
# separately below, not in the fail-closed loop.
cat >"$TMP_DIR/lang_compound.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}
use math::wide_float::{print_limb_hex16}

fn main() -> i32 with IO, Mut, Panic, Div {
    var acc: f128 = 1.0
    let tiny: f128 = f128_from_limbs(3746994889972252672, 4592679628783035426)
    acc += tiny
    print("wire_compound_assign=")
    print_limb_hex16(f128_to_lo(acc))
    print(":")
    print_limb_hex16(f128_to_hi(acc))
    println("")
    if f128_to_hi(acc) == 0 { return 1 }
    return 0
}
EOF

# DCE trigger: no f128 let / param / return anywhere — only a struct field.
# Without the struct-level mark the desugar targets are dropped and the
# binary hits the body-less stub (SIGILL, exit 132).
cat >"$TMP_DIR/struct_only.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

struct P { x: f128, y: f128 }

fn x_lt_y(p: &P) -> bool { p.x < p.y }

fn main() -> i32 with IO, Mut, Panic, Div {
    let p = P { x: 1.0, y: 2.0 }
    if !x_lt_y(&p) { return 1 }
    if f128_to_hi(p.y) != 4611686018427387904 { return 2 }
    return 0
}
EOF

if [[ -x "$SOUC" ]]; then
  for neg in lang_f256_field:"$REFUSE_SENTINEL":language_f256_field_still_fail_closed \
             lang_inexact_field:"no f64 widen":language_f128_inexact_field_literal_still_fail_closed; do
    name="${neg%%:*}"; rest="${neg#*:}"; want="${rest%:*}"; label="${rest##*:}"
    set +e
    "$SOUC" compile "$TMP_DIR/$name.sio" -o "$TMP_DIR/$name.elf" >"$TMP_DIR/$name.compile.log" 2>&1
    n_rc=$?
    set -e
    if [[ "$n_rc" -ne 0 ]] && grep -Fq "$want" "$TMP_DIR/$name.compile.log"; then
      note_pass "$label"
    else
      note_fail "${label%_still_fail_closed}_fail_closed_regression rc=$n_rc"
      tail -30 "$TMP_DIR/$name.compile.log" >&2 || true
    fi
  done

  # KL-8: compound assignment on a plain f128 local now compiles and runs,
  # computing the real softfloat sum -- 1.0 += tiny must give the exact
  # anti-f64 limbs (00002f3942192484:3fff000000000000), matching v0e4's own
  # wire_1+tiny receipt for the identical computation. Never f64 bits, never
  # a rounded-away-to-1.0 result (which is what an f64-widen-then-narrow
  # cheat would produce, since 1.0+1.0=2.0 alone can't distinguish real
  # binary128 arithmetic from that cheat -- see the lang_compound.sio
  # heredoc above).
  set +e
  "$SOUC" run "$TMP_DIR/lang_compound.sio" >"$TMP_DIR/lang_compound.run.log" 2>&1
  lc_rc=$?
  set -e
  if [[ "$lc_rc" -eq 0 ]] && grep -Fq 'wire_compound_assign=00002f3942192484:3fff000000000000' "$TMP_DIR/lang_compound.run.log"; then
    note_pass "language_f128_compound_assign_kl8_correct"
  else
    note_fail "language_f128_compound_assign_kl8_wrong rc=$lc_rc"
    tail -30 "$TMP_DIR/lang_compound.run.log" >&2 || true
  fi

  set +e
  "$SOUC" run "$TMP_DIR/struct_only.sio" >"$TMP_DIR/struct_only.run.log" 2>&1
  so_rc=$?
  set -e
  if [[ "$so_rc" -eq 0 ]]; then
    note_pass "madaros_run_struct_only_dce_trigger"
  else
    note_fail "madaros_run_struct_only_dce_trigger rc=$so_rc"
    tail -30 "$TMP_DIR/struct_only.run.log" >&2 || true
  fi

  set +e
  "$SOUC" run "$SMOKE" >"$TMP_DIR/madaros.run.log" 2>&1
  m_rc=$?
  set -e
  if [[ "$m_rc" -eq 0 ]] && grep -Fq 'PASS f128_v0e55_language_struct_fields' "$TMP_DIR/madaros.run.log"; then
    note_pass "madaros_run_language_f128_struct_fields"
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
    note_fail "madaros_run_language_f128_struct_fields rc=$m_rc"
    tail -40 "$TMP_DIR/madaros.run.log" >&2 || true
  fi
else
  note_fail "souc_missing"
fi

echo "NOTE v0e55_deferred f256=pending f128_methods=pending f128_arrays=pending compound_assign=pending print_builtin=pending gum=pending"
echo "NOTE adr009 python_softfloat=not_claim_clock rust=not_claim_clock lean_single_language=greenwash"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e55_language_struct_fields general=ieee754 anti_f64=green language_fields=madaros claim_clock=sounio_native_expected"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e55"
  exit 0
fi
echo "FAIL madaros_f128_f256_ladder_gate stage=v0e55" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
