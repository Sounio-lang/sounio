#!/usr/bin/env bash
# madaros_f128_f256_v0e3_run_ops_gate.sh — V0-E.3 runnable limb softfloat cases.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (V0-E.3 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
# Claim clock: ADR-008 / ADR-009 — oracle_class=sounio_native_expected
#
# V0-E.3 green (this gate):
#   - Seed-run smoke: F128Bits soft_add cases + hex wires
#   - Pass/fail from Sounio sentinels + IEEE hex expecteds hardcoded here
#     (same limb constants as the .sio witness) — NOT Python/Rust/softfloat_limb
#   - Madaros check of language f128 arith still OK (V0-E.2)
#
# Explicitly NOT claimed:
#   - General softfloat-in-sio for all hard corpus rows
#   - Language `f128` op lowering to softfloat at Madaros-run
#   - print_f128 builtin / Knowledge / GUM / MeasuredF256
#   - Python softfloat_limb as claim authority (ADR-009 hard exclusion)
#
# Optional: set SOUNIO_FOREIGN_ORACLE_HARD=1 is ignored here; Python is never
# the claim clock for this gate (external_corroboration_only only if added later).
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"
SEED_COMPILER="$(realpath "${SOUNIO_F128_SEED_COMPILER:-$ROOT_DIR/bin/souc-lean-single-x86_64}")"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e3.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

echo "=== madaros_f128_f256_ladder_gate stage=v0e3 ==="
echo "slice=v0e3_run_softfloat_cases"
echo "claim_clock=sounio_native_expected"
echo "adr=ADR-008+ADR-009"
echo "seed_elf=$SEED_COMPILER"

# Native expected hex wires (IEEE binary128 limb print; mirror of smoke constants).
# These are Sounio-side literals, not a foreign runtime product.
NATIVE_EXPECT=(
  "wire_1+1=0000000000000000:4000000000000000"
  "wire_1+2=0000000000000000:4000800000000000"  # IEEE 3.0
  "wire_half+half=0000000000000000:3fff000000000000"
  "wire_1+neg1=0000000000000000:0000000000000000"
)

SMOKE=tests/run-pass/f128_v0e3_ops_print_smoke.sio
ELF="$TMP_DIR/smoke.elf"
BLOG="$TMP_DIR/build.log"
RLOG="$TMP_DIR/run.log"

if [[ ! -x "$SEED_COMPILER" ]]; then
  note_fail "seed_missing"
else
  set +e
  "$SEED_COMPILER" "$ROOT_DIR/$SMOKE" "$ELF" >"$BLOG" 2>&1
  b_rc=$?
  set -e
  if [[ "$b_rc" -ne 0 || ! -f "$ELF" ]]; then
    note_fail "seed_build_smoke"
    tail -30 "$BLOG" >&2 || true
  else
    chmod +x "$ELF"
    set +e
    "$ELF" >"$RLOG" 2>&1
    r_rc=$?
    set -e
    if [[ "$r_rc" -ne 0 ]] || ! grep -Fq 'PASS f128_v0e3_ops_print_smoke' "$RLOG"; then
      note_fail "seed_run_smoke"
      cat "$RLOG" >&2 || true
    else
      note_pass "seed_run_smoke"
      for want in "${NATIVE_EXPECT[@]}"; do
        if grep -Fq "$want" "$RLOG"; then
          note_pass "native_hex:${want%%=*}"
        else
          note_fail "native_hex_mismatch:${want%%=*}"
          echo "want $want" >&2
          cat "$RLOG" >&2 || true
        fi
      done
    fi
  fi
fi

# Language check still green (V0-E.2)
if [[ -x "$SOUC" ]]; then
  set +e
  "$SOUC" check tests/run-pass/f128_v0e2_arith_check.sio >"$TMP_DIR/lang.check.log" 2>&1
  set -e
  if grep -Fq 'check: OK' "$TMP_DIR/lang.check.log"; then
    note_pass "language_f128_arith_check_still_ok"
  else
    note_fail "language_f128_arith_check"
    tail -20 "$TMP_DIR/lang.check.log" >&2 || true
  fi
else
  note_fail "souc_missing_for_lang_check"
fi

echo "NOTE v0e3_deferred general_softfloat_sio=pending language_f128_lower=pending print_builtin=pending gum=pending"
echo "NOTE adr009 python_softfloat=not_claim_clock rust=not_claim_clock"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e3_run_ops soft_add_cases=green hex_wire=sounio_native_expected language_check=ok lower=deferred"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e3"
  exit 0
fi
echo "FAIL madaros_f128_f256_ladder_gate stage=v0e3" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
