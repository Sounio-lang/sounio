#!/usr/bin/env bash
# madaros_f128_f256_v0f5_softfloat_add_gate.sh — V0-F.5 / KL-15a general f256 add/sub in sio.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-F (V0-F.5 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
# Claim clock: ADR-008 / ADR-009 — oracle_class=sounio_native_expected
#
# V0-F.5 green (this gate):
#   - Seed-run IEEE binary256 add/sub in stdlib/math/softfloat_f256.sio
#   - Madaros-run of the same F256Bits smoke (not language f256 ops)
#   - Cases outside f64 precision (0.5+1, 2+tiny)
#   - Anti-f64 recover ((1+~1e-20)-1 ≠ 0) with native hex expecteds
#   - Language f256 Madaros compile still fail-closed (V0-E.4.1)
#
# Explicitly NOT claimed:
#   - Language `f256` Madaros-run desugar (still V0-E.4.1 sentinel)
#   - mul/div/sqrt, print_f256 builtin, GUM / MeasuredF256 / Knowledge<f128>
#   - f256 fields / params / arrays
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"
SEED_COMPILER="$(realpath "${SOUNIO_F128_SEED_COMPILER:-$ROOT_DIR/bin/souc-lean-single-x86_64}")"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f256-ladder-v0f5.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

echo "=== madaros_f128_f256_ladder_gate stage=v0f5 ==="
echo "slice=v0f5_general_f256_softfloat_add_sub"
echo "claim_clock=sounio_native_expected"
echo "adr=ADR-008+ADR-009"
echo "seed_elf=$SEED_COMPILER"

NATIVE_EXPECT=(
  "wire_half+one=0000000000000000:0000000000000000:0000000000000000:3ffff80000000000"
  "wire_1+tiny=f3d610607aa0167e:446baa23d2ec729a:000002f394219248:3ffff00000000000"
  "wire_recover=0000000000000000:9eb08303d500b3f0:235d511e976394d7:3ffbc79ca10c9242"
  "wire_two+tiny=79eb08303d500b3f:2235d511e976394d:00000179ca10c924:4000000000000000"
)

SF=stdlib/math/softfloat_f256.sio
WF=stdlib/math/wide_float.sio
if [[ -f "$SF" ]] \
  && grep -Fq 'pub fn f256_bits_soft_add' "$SF" \
  && grep -Fq 'pub fn f256_bits_soft_sub' "$SF" \
  && grep -Fq 'IEEE binary256 addition' "$SF" \
  && ! grep -Fq 'f256_bits_soft_add' "$WF"; then
  note_pass "stdlib_softfloat_f256_api_present_not_in_wide_float"
else
  note_fail "stdlib_softfloat_f256_api_missing_or_collides_with_wide_float"
fi

SMOKE=tests/run-pass/f256_v0f5_softfloat_add_sub.sio
if grep -Fq 'use math::softfloat_f256::' "$SMOKE" \
  && grep -Fq 'f256_bits_soft_add' "$SMOKE" \
  && ! grep -Fq 'fn f256_bits_soft_add' "$SMOKE"; then
  note_pass "smoke_imports_stdlib_softfloat_not_local_copy"
else
  note_fail "smoke_must_import_stdlib_softfloat"
fi

ELF="$TMP_DIR/smoke.elf"
BLOG="$TMP_DIR/build.log"
RLOG="$TMP_DIR/run.log"

if [[ ! -x "$SEED_COMPILER" ]]; then
  note_fail "seed_missing"
elif [[ "$(head -c2 "$SEED_COMPILER" 2>/dev/null)" == '#!' ]]; then
  note_fail "seed_is_wrapper_not_elf"
else
  set +e
  "$SEED_COMPILER" "$ROOT_DIR/$SMOKE" "$ELF" >"$BLOG" 2>&1
  b_rc=$?
  set -e
  if [[ "$b_rc" -ne 0 || ! -f "$ELF" ]]; then
    note_fail "seed_build_v0f5_smoke"
    tail -40 "$BLOG" >&2 || true
  else
    chmod +x "$ELF"
    set +e
    "$ELF" >"$RLOG" 2>&1
    r_rc=$?
    set -e
    if [[ "$r_rc" -ne 0 ]] || ! grep -Fq 'PASS f256_v0f5_softfloat_add_sub' "$RLOG"; then
      note_fail "seed_run_v0f5_smoke"
      cat "$RLOG" >&2 || true
    else
      note_pass "seed_run_v0f5_smoke"
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

# Language f256 compile remains fail-closed
REFUSE_SENTINEL='f128/f256 Madaros-run softfloat lowering is not implemented (V0-E.4.1 fail-closed; no f64 greenwash)'
cat >"$TMP_DIR/lang_f256_arith.sio" <<'EOF'
fn main() -> i32 with IO, Mut, Panic, Div {
    let x: f256 = 1.0
    let y: f256 = 2.0
    let z: f256 = x + y
    return 0
}
EOF

if [[ -x "$SOUC" ]]; then
  set +e
  "$SOUC" compile "$TMP_DIR/lang_f256_arith.sio" -o "$TMP_DIR/lang.elf" >"$TMP_DIR/lang.compile.log" 2>&1
  set -e
  if grep -Fq "$REFUSE_SENTINEL" "$TMP_DIR/lang.compile.log"; then
    note_pass "language_f256_arith_still_fail_closed"
  else
    note_fail "language_f256_arith_fail_closed_regression"
    tail -30 "$TMP_DIR/lang.compile.log" >&2 || true
  fi

  set +e
  "$SOUC" run "$SMOKE" >"$TMP_DIR/madaros.run.log" 2>&1
  m_rc=$?
  set -e
  if [[ "$m_rc" -eq 0 ]] && grep -Fq 'PASS f256_v0f5_softfloat_add_sub' "$TMP_DIR/madaros.run.log"; then
    note_pass "madaros_run_v0f5_f256bits_softfloat"
    for want in "${NATIVE_EXPECT[@]}"; do
      if grep -Fq "$want" "$TMP_DIR/madaros.run.log"; then
        note_pass "madaros_native_hex:${want%%=*}"
      else
        note_fail "madaros_native_hex_mismatch:${want%%=*}"
      fi
    done
  else
    note_fail "madaros_run_v0f5_f256bits_softfloat"
    tail -40 "$TMP_DIR/madaros.run.log" >&2 || true
  fi
else
  note_fail "souc_missing_for_fail_closed"
fi

echo "NOTE v0f5_deferred language_desugar=pending mul_div=pending fields_params_arrays=pending print_builtin=pending gum=pending MeasuredF256=pending"
echo "NOTE adr009 python_softfloat=not_claim_clock rust=not_claim_clock"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0f5_softfloat_add_sub general=ieee754_add_sub anti_f64=green madaros_run=f256bits language_desugar=deferred claim_clock=sounio_native_expected"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0f5"
  exit 0
fi
echo "FAIL madaros_f128_f256_ladder_gate stage=v0f5" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
