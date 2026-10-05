#!/usr/bin/env bash
# madaros_f128_f256_v0e54_language_params_abi_gate.sh — V0-E.5.4 language f128
# params/returns ABI.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (V0-E.5.4 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
# Claim clock: ADR-008 / ADR-009 — oracle_class=sounio_native_expected
#
# V0-E.5.4 green (this gate):
#   - Madaros-run of user fns with f128 params and `-> f128` returns: the value
#     crosses the call boundary as an F128Bits handle (one GPR word)
#   - f128 params are ready softfloat operands (+ - * / unary- comparisons)
#   - a call to a `-> f128` fn is an f128 expression: `let` RHS (annotated or
#     not), operand of another op, argument to another f128 param
#   - a bare literal argument to an f128 param lowers as binary128 (never f64)
#   - mixed param kinds (bool/i64 + f128), early return, recursion, 4 f128 params
#   - Anti-f64 through the ABI: (1+~1e-20)^2 and 1/(1+~1e-20)
#   - f256 params/returns and inexact literals (0.1) still fail-closed
#
# Explicitly NOT claimed:
#   - lean_single language f128 (still f64 greenwash)
#   - f256, f128 struct fields / arrays, print_f128 builtin, GUM, sqrt/fma
#   - f128 literal in tail/return position (check E008: bare literal is f64)
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e54.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

echo "=== madaros_f128_f256_ladder_gate stage=v0e54 ==="
echo "slice=v0e54_language_f128_params_returns_abi"
echo "claim_clock=sounio_native_expected"
echo "adr=ADR-008+ADR-009"

NATIVE_EXPECT=(
  "wire_(1+tiny)^2=00005e7284324908:3fff000000000000"
  "wire_1/(1+tiny)=ffffa18d7bcdb6f8:3ffeffffffffffff"
  "wire_horner=0000000000000000:4002e00000000000"
  "wire_neg1=0000000000000000:bfff000000000000"
)

LOWER=self-hosted/ir/lower.sio
# The semantic ABI rides on the callee's own IrFunction (f128_param_mask +
# return_struct_name), never on a name-keyed side table or a global handoff.
if grep -Fq 'pub f128_param_mask: i64' self-hosted/ir/ir.sio \
  && grep -Fq 'fn lower_f128_param_mask_of' "$LOWER" \
  && grep -Fq 'fn lower_callee_returns_f128_ref' "$LOWER" \
  && grep -Fq 'fn lower_expr_args_masked_ref' "$LOWER" \
  && grep -Fq 'fn lowerer_mark_local_wide_bits_mut' "$LOWER" \
  && ! grep -Fq 'LOWER_F128_SIG_HASH' "$LOWER" \
  && ! grep -Fq 'LOWER_F128_ARG_MASK' "$LOWER"; then
  note_pass "lower_f128_abi_markers"
else
  note_fail "lower_f128_abi_markers_missing"
fi

if grep -Fq 'fn spec_dce_fn_sig_uses_f128' self-hosted/check/specializer.sio \
  && grep -Fq 'fn spec_dce_scan_fn_def' self-hosted/check/specializer.sio; then
  note_pass "dce_f128_signature_markers"
else
  note_fail "dce_f128_signature_markers_missing"
fi

SMOKE=tests/run-pass/f128_v0e54_language_params_abi.sio
if grep -Fq 'use math::softfloat_f128::{' "$SMOKE" \
  && grep -Fq 'fn add128(a: f128, b: f128) -> f128 { a + b }' "$SMOKE" \
  && grep -Fq 'fn neg128(a: f128) -> f128 { -a }' "$SMOKE" \
  && grep -Fq 'fn lt128(a: f128, b: f128) -> bool { a < b }' "$SMOKE" \
  && grep -Fq 'fn scale_i(n: i64, x: f128) -> f128' "$SMOKE" \
  && grep -Fq 'fn horner(a: f128, b: f128, c: f128, x: f128) -> f128' "$SMOKE" \
  && grep -Fq 'add128(1.0, 2.0)' "$SMOKE" \
  && grep -Fq 'let seven = add128(mul128(two, three), one)' "$SMOKE" \
  && grep -Fq 'div128(one, add128(one, tiny))' "$SMOKE" \
  && grep -Fq 'v0e54_main_entered' "$SMOKE" \
  && ! grep -Fq 'F128Bits {' "$SMOKE" \
  && ! grep -Fq 'f128_bits_soft_' "$SMOKE"; then
  note_pass "smoke_is_language_f128_abi_not_f128bits_copy"
else
  note_fail "smoke_must_be_language_f128_abi"
fi

REFUSE_SENTINEL='f128/f256 Madaros-run softfloat lowering is not implemented (V0-E.4.1 fail-closed; no f64 greenwash)'

# f256 params/returns: same-format ops typecheck (V0-E.2) but the lowering has
# no f256 payload — must refuse with the V0-E.4.1 sentinel, never emit an ELF.
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

# Inexact literal argument to an f128 param: 0.1 must never be f64-widened.
cat >"$TMP_DIR/lang_inexact_arg.sio" <<'EOF'
use math::softfloat_f128::{f128_from_limbs, f128_to_lo, f128_to_hi}

fn add128(a: f128, b: f128) -> f128 { a + b }

fn main() -> i32 with IO, Mut, Panic, Div {
    let two: f128 = 2.0
    let bad: f128 = add128(0.1, two)
    if f128_to_hi(bad) == 0 { return 1 }
    return 0
}
EOF

if [[ -x "$SOUC" ]]; then
  set +e
  "$SOUC" compile "$TMP_DIR/lang_f256_params.sio" -o "$TMP_DIR/lang_f256.elf" >"$TMP_DIR/lang_f256.compile.log" 2>&1
  f_rc=$?
  set -e
  if [[ "$f_rc" -ne 0 ]] && grep -Fq "$REFUSE_SENTINEL" "$TMP_DIR/lang_f256.compile.log"; then
    note_pass "language_f256_params_still_fail_closed"
  else
    note_fail "language_f256_params_fail_closed_regression rc=$f_rc"
    tail -30 "$TMP_DIR/lang_f256.compile.log" >&2 || true
  fi

  set +e
  "$SOUC" compile "$TMP_DIR/lang_inexact_arg.sio" -o "$TMP_DIR/lang_inexact_arg.elf" >"$TMP_DIR/lang_inexact_arg.compile.log" 2>&1
  d_rc=$?
  set -e
  if [[ "$d_rc" -ne 0 ]] && grep -Fq 'no f64 widen' "$TMP_DIR/lang_inexact_arg.compile.log"; then
    note_pass "language_f128_inexact_literal_arg_still_fail_closed"
  else
    note_fail "language_f128_inexact_literal_arg_fail_closed_regression rc=$d_rc"
    tail -30 "$TMP_DIR/lang_inexact_arg.compile.log" >&2 || true
  fi

  set +e
  "$SOUC" run "$SMOKE" >"$TMP_DIR/madaros.run.log" 2>&1
  m_rc=$?
  set -e
  if [[ "$m_rc" -eq 0 ]] && grep -Fq 'PASS f128_v0e54_language_params_abi' "$TMP_DIR/madaros.run.log"; then
    note_pass "madaros_run_language_f128_params_abi"
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
    note_fail "madaros_run_language_f128_params_abi rc=$m_rc"
    tail -40 "$TMP_DIR/madaros.run.log" >&2 || true
  fi
else
  note_fail "souc_missing"
fi

echo "NOTE v0e54_deferred f256=pending f128_struct_fields=pending print_builtin=pending gum=pending"
echo "NOTE adr009 python_softfloat=not_claim_clock rust=not_claim_clock lean_single_language=greenwash"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e54_language_params_abi general=ieee754 anti_f64=green language_abi=madaros claim_clock=sounio_native_expected"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e54"
  exit 0
fi
echo "FAIL madaros_f128_f256_ladder_gate stage=v0e54" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
