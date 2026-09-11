#!/usr/bin/env bash
# madaros_kl8_f128_residuals_gate.sh — KL-8 f128 surface residuals closed:
# nested if/else literal tail, `%`, `+=`, builtin println/print → println_f128.
#
# Spec: docs/compiler/KNOWN_LIMITATIONS.md KL-8 (ledger rung)
# Claim clock: ADR-008 / ADR-009 — oracle_class=sounio_native_expected
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/kl8-f128-residuals.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

echo "=== madaros_kl8_f128_residuals_gate ==="
echo "claim_clock=sounio_native_expected"

LOWER=self-hosted/ir/lower.sio
CHECK=self-hosted/check/check.sio
COMPAT=self-hosted/check/compat.sio
SOFT=stdlib/math/softfloat_f128.sio
DRIVER=self-hosted/compiler/module_frontend.sio
SPEC=self-hosted/check/specializer.sio

if grep -Fq 'var LOWER_F128_TAIL_EXPR: bool = false' "$LOWER" \
  && grep -Fq 'if op == BinaryOp::OpRem { return make_name("f128_bits_soft_rem") }' "$LOWER" \
  && grep -Fq 'make_name("println_f128")' "$LOWER"; then
  note_pass "lower_nested_tail_rem_println_f128_wired"
else
  note_fail "lower_kl8_residual_paths_missing"
fi

if grep -Fq 'BinaryOp::OpRem => left' "$COMPAT" \
  && grep -Fq 'fn checker_if_arms_are_float_literal_tails(' "$CHECK" \
  && grep -Fq 'pub fn f128_bits_soft_rem(' "$SOFT"; then
  note_pass "checker_compat_stdlib_rem_and_if_tail"
else
  note_fail "checker_compat_or_stdlib_rem_missing"
fi

if grep -Fq 'implicit_import math/softfloat_f128_fmt.sio reason=f128_in_ast' "$DRIVER" \
  && grep -Fq 'f128_bits_soft_rem' "$SPEC"; then
  note_pass "driver_implicit_fmt_and_dce_rem"
else
  note_fail "driver_implicit_fmt_or_dce_rem_missing"
fi

SMOKE=tests/run-pass/f128_kl8_residuals.sio
if grep -Fq 'fn pick(cond: bool) -> f128' "$SMOKE" \
  && grep -Fq 'eight % three' "$SMOKE" \
  && grep -Fq 'acc += two' "$SMOKE" \
  && grep -Fq 'println(v)' "$SMOKE" \
  && ! grep -Fq 'softfloat_f128_fmt' "$SMOKE"; then
  note_pass "smoke_covers_kl8_residuals"
else
  note_fail "smoke_must_cover_nested_if_rem_add_println"
fi

INEXACT_SENTINEL='V0-E.5.9; no f64 widen'

cat >"$TMP_DIR/neg_nested_tenth.sio" <<'EOF'
fn pick() -> f128 { if true { 0.1 } else { 2.0 } }

fn main() -> i32 with IO, Mut, Panic, Div {
    let x = pick()
    let y: f128 = 1.0
    if x == y { return 1 }
    return 0
}
EOF

cat >"$TMP_DIR/neg_println_f64.sio" <<'EOF'
fn main() -> i32 with IO, Mut, Panic, Div {
    let x: f64 = 3.0
    println(x)
    return 0
}
EOF

if [[ -x "$SOUC" ]]; then
  set +e
  "$SOUC" compile "$TMP_DIR/neg_nested_tenth.sio" -o "$TMP_DIR/neg_nested_tenth.elf" >"$TMP_DIR/neg_nested_tenth.log" 2>&1
  n_rc=$?
  set -e
  if [[ "$n_rc" -ne 0 ]] && grep -Fq "$INEXACT_SENTINEL" "$TMP_DIR/neg_nested_tenth.log" && [[ ! -s "$TMP_DIR/neg_nested_tenth.elf" ]]; then
    note_pass "inexact_nested_if_tail_fail_closed"
  else
    note_fail "inexact_nested_if_tail_fail_closed_regression rc=$n_rc"
    tail -20 "$TMP_DIR/neg_nested_tenth.log" >&2 || true
  fi

  set +e
  "$SOUC" run "$SMOKE" >"$TMP_DIR/smoke.run.log" 2>&1
  s_rc=$?
  set -e
  if [[ "$s_rc" -eq 0 ]] && grep -Fq 'PASS f128_kl8_residuals' "$TMP_DIR/smoke.run.log" \
     && grep -Fq 'nested_if_one=ok' "$TMP_DIR/smoke.run.log" \
     && grep -Fq 'rem_8_mod_3=ok' "$TMP_DIR/smoke.run.log" \
     && grep -Fq 'compound_add=ok' "$TMP_DIR/smoke.run.log" \
     && grep -Fq '3.00000000000000000000000000000000000e+0000' "$TMP_DIR/smoke.run.log"; then
    note_pass "madaros_run_kl8_residuals_witness"
  else
    note_fail "madaros_run_kl8_residuals_witness rc=$s_rc"
    tail -40 "$TMP_DIR/smoke.run.log" >&2 || true
  fi

  set +e
  bash "$ROOT_DIR/scripts/ci/madaros_f128_f256_ladder_gate.sh" --stage v0e510 >"$TMP_DIR/v0e510.log" 2>&1
  l_rc=$?
  set -e
  if [[ "$l_rc" -eq 0 ]] && grep -Fq 'PASS madaros_f128_f256_ladder_gate stage=v0e510' "$TMP_DIR/v0e510.log"; then
    note_pass "ladder_v0e510_still_green"
  else
    note_fail "ladder_v0e510_regression rc=$l_rc"
    tail -30 "$TMP_DIR/v0e510.log" >&2 || true
  fi

  # f64 println must still route to print_f64, not println_f128.
  set +e
  "$SOUC" run "$TMP_DIR/neg_println_f64.sio" >"$TMP_DIR/neg_println_f64.run.log" 2>&1
  f_rc=$?
  set -e
  if [[ "$f_rc" -eq 0 ]] && grep -Fq '3.000000' "$TMP_DIR/neg_println_f64.run.log"; then
    note_pass "println_f64_unchanged"
  else
    note_fail "println_f64_regression rc=$f_rc"
    tail -20 "$TMP_DIR/neg_println_f64.run.log" >&2 || true
  fi
else
  note_fail "souc_missing"
fi

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS madaros_kl8_f128_residuals nested_if_tail=% += builtin_println_f128 claim_clock=sounio_native_expected"
  exit 0
fi
echo "FAIL madaros_kl8_f128_residuals_gate" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
