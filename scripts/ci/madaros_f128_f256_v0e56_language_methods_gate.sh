#!/usr/bin/env bash
# madaros_f128_f256_v0e56_language_methods_gate.sh — V0-E.5.6 language f128
# through impl methods (`-> f128` returns, f128 params, self / &self / &!self).
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (V0-E.5.6 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
# Claim clock: ADR-008 / ADR-009 — oracle_class=sounio_native_expected
#
# V0-E.5.6 green (this gate):
#   - Madaros-run impl methods declared `-> f128` / taking `f128` params: the
#     method is the mangled fn `Vec2_norm2`, and its IrFunction carries the same
#     identity a free fn does (return_struct_name == "f128", f128_param_mask)
#   - a method call is an f128 expression: annotated / unannotated `let`,
#     operand, argument to an f128-param fn, comparison operand, assignment RHS,
#     chained `v.scale(two).norm2()`, f128 field off a struct-returning method
#   - explicit args to f128 params lower through the callee's mask with the
#     receiver bit shifted out: `v.scale(2.0)` is binary128, never f64 bits
#   - receivers: by value `self`, `self: &Vec2`, `self: &!Vec2` (mutating an
#     f128 field through the reference); mixed `i64 + f128` params
#   - DCE: a method signature with f128 marks the softfloat desugar targets when
#     the program has no f128 let / struct field / free-fn signature (method_only)
#   - Anti-f64 through a method: (1+~1e-20)^2 ≠ 1
#   - f256 method returns and inexact literal method args (0.1) still fail-closed
#
# Explicitly NOT claimed:
#   - lean_single language f128 (still f64 greenwash)
#   - f256, arrays of f128, `+=` on f128, `%`, print_f128, GUM
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e56.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

echo "=== madaros_f128_f256_ladder_gate stage=v0e56 ==="
echo "slice=v0e56_language_f128_methods"
echo "claim_clock=sounio_native_expected"
echo "adr=ADR-008+ADR-009"

NATIVE_EXPECT=(
  "wire_norm2=0000000000000000:4001400000000000"
  "wire_chained=0000000000000000:4003400000000000"
  "wire_method_sq=00005e7284324908:3fff000000000000"
  "wire_bump_x=0000000000000000:4000c00000000000"
  "wire_axpy=0000000000000000:4001800000000000"
)

LOWER=self-hosted/ir/lower.sio
if grep -Fq 'fn lower_method_call_fn_id_ref' "$LOWER" \
  && grep -Fq 'fn lower_method_call_returns_f128_ref' "$LOWER" \
  && grep -Fq 'method call in f128 position does not return f128 (V0-E.5.6)' "$LOWER" \
  && grep -Fq 'lower_expr_args_masked_ref(&e.args, f128_arg_mask_m)' "$LOWER"; then
  note_pass "lower_f128_method_markers"
else
  note_fail "lower_f128_method_markers_missing"
fi

# Methods are scanned unconditionally by the DCE pre-pass and their signatures
# feed the same f128 trigger as free fns; this pins that the trigger exists.
if grep -Fq 'fn spec_dce_fn_sig_uses_f128' self-hosted/check/specializer.sio; then
  note_pass "dce_f128_signature_marker"
else
  note_fail "dce_f128_signature_marker_missing"
fi

SMOKE=tests/run-pass/f128_v0e56_language_methods.sio
if grep -Fq 'use math::softfloat_f128::{' "$SMOKE" \
  && grep -Fq 'struct Vec2 { x: f128, y: f128 }' "$SMOKE" \
  && grep -Fq 'fn norm2(self) -> f128' "$SMOKE" \
  && grep -Fq 'fn scale(self, k: f128) -> Vec2' "$SMOKE" \
  && grep -Fq 'fn sum_ref(self: &Vec2) -> f128' "$SMOKE" \
  && grep -Fq 'fn bump(self: &!Vec2, d: f128)' "$SMOKE" \
  && grep -Fq 'fn axpy(self, n: i64, k: f128) -> f128' "$SMOKE" \
  && grep -Fq 'let w = v.scale(2.0)' "$SMOKE" \
  && grep -Fq 'v.scale(two).norm2()' "$SMOKE" \
  && grep -Fq 'twice(v.norm2())' "$SMOKE" \
  && grep -Fq 'v0e56_main_entered' "$SMOKE" \
  && ! grep -Fq 'F128Bits {' "$SMOKE" \
  && ! grep -Fq 'f128_bits_soft_' "$SMOKE"; then
  note_pass "smoke_is_language_f128_methods_not_f128bits_copy"
else
  note_fail "smoke_must_be_language_f128_methods"
fi

REFUSE_SENTINEL='f128/f256 Madaros-run softfloat lowering is not implemented (V0-E.4.1 fail-closed; no f64 greenwash)'

# f256 method return: same-format ops typecheck (V0-E.2) but there is no f256
# payload — must refuse with the V0-E.4.1 sentinel, never emit an ELF.
cat >"$TMP_DIR/lang_f256_method.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

struct W { v: f256 }

impl W {
    fn get(self) -> f256 { self.v * self.v }
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let x: f256 = 1.0
    let w = W { v: x }
    let y: f256 = w.get()
    return 0
}
EOF

# Inexact literal as a method argument to an f128 param: 0.1 must never be
# f64-widened into the handle.
cat >"$TMP_DIR/lang_inexact_method_arg.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

struct Vec2 { x: f128, y: f128 }

impl Vec2 {
    fn scale(self, k: f128) -> Vec2 { Vec2 { x: self.x * k, y: self.y * k } }
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let v = Vec2 { x: 1.0, y: 2.0 }
    let w = v.scale(0.1)
    if f128_to_hi(w.x) == 0 { return 1 }
    return 0
}
EOF

# DCE trigger: the ONLY f128 in the program is in a method signature — no f128
# let in main, no f128 struct field, no free fn with f128. Without the
# signature-level mark the desugar targets are dropped and the binary hits the
# body-less stub (SIGILL, exit 132).
cat >"$TMP_DIR/method_only.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

struct P { a: i64 }

impl P {
    fn half(self, k: f128) -> f128 { let h: f128 = 0.5; k * h }
    fn lt(self, a: f128, b: f128) -> bool { a < b }
}

fn main() -> i32 with IO, Mut, Panic, Div {
    let p = P { a: 1 }
    let h = p.half(2.0)
    if f128_to_hi(h) != 4611404543450677248 { return 1 }
    if !p.lt(h, 2.0) { return 2 }
    return 0
}
EOF

if [[ -x "$SOUC" ]]; then
  for neg in lang_f256_method:"$REFUSE_SENTINEL":language_f256_method_still_fail_closed \
             lang_inexact_method_arg:"no f64 widen":language_f128_inexact_method_arg_still_fail_closed; do
    name="${neg%%:*}"; rest="${neg#*:}"; want="${rest%:*}"; label="${rest##*:}"
    set +e
    "$SOUC" compile "$TMP_DIR/$name.sio" -o "$TMP_DIR/$name.elf" >"$TMP_DIR/$name.compile.log" 2>&1
    n_rc=$?
    set -e
    if [[ "$n_rc" -ne 0 ]] && grep -Fq "$want" "$TMP_DIR/$name.compile.log" && [[ ! -e "$TMP_DIR/$name.elf" ]]; then
      note_pass "$label"
    else
      note_fail "${label%_still_fail_closed}_fail_closed_regression rc=$n_rc"
      tail -30 "$TMP_DIR/$name.compile.log" >&2 || true
    fi
  done

  set +e
  "$SOUC" run "$TMP_DIR/method_only.sio" >"$TMP_DIR/method_only.run.log" 2>&1
  mo_rc=$?
  set -e
  if [[ "$mo_rc" -eq 0 ]]; then
    note_pass "madaros_run_method_only_dce_trigger"
  else
    note_fail "madaros_run_method_only_dce_trigger rc=$mo_rc"
    tail -30 "$TMP_DIR/method_only.run.log" >&2 || true
  fi

  set +e
  "$SOUC" run "$SMOKE" >"$TMP_DIR/madaros.run.log" 2>&1
  m_rc=$?
  set -e
  if [[ "$m_rc" -eq 0 ]] && grep -Fq 'PASS f128_v0e56_language_methods' "$TMP_DIR/madaros.run.log"; then
    note_pass "madaros_run_language_f128_methods"
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
    note_fail "madaros_run_language_f128_methods rc=$m_rc"
    tail -40 "$TMP_DIR/madaros.run.log" >&2 || true
  fi
else
  note_fail "souc_missing"
fi

echo "NOTE v0e56_deferred f256=pending f128_arrays=pending compound_assign=pending f128_mod=pending print_builtin=pending gum=pending"
echo "NOTE adr009 python_softfloat=not_claim_clock rust=not_claim_clock lean_single_language=greenwash"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e56_language_methods general=ieee754 anti_f64=green language_methods=madaros claim_clock=sounio_native_expected"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e56"
  exit 0
fi
echo "FAIL madaros_f128_f256_ladder_gate stage=v0e56" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
