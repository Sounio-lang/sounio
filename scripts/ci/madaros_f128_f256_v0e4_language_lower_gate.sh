#!/usr/bin/env bash
# madaros_f128_f256_v0e4_language_lower_gate.sh — V0-E.4 anti-f64 softfloat prep.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (V0-E.4 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
# Claim clock: ADR-008 / ADR-009 — oracle_class=sounio_native_expected
#
# V0-E.4 green (this gate):
#   - Seed-run F128Bits soft_add/sub anti-f64 case ((1+~1e-20)-1 ≠ 0)
#   - Native IEEE hex expecteds hardcoded here (not Python/Rust judges)
#   - lean_single language `f128` path is anti-f64: since #2387 it refuses to
#     widen an f64-inexact literal into binary128 rather than greenwash it
#     (negative control; updated 2026-09-22, see the block below)
#   - Madaros check of language f128 arith still OK (V0-E.2)
#   - stdlib wide_float exposes f128_bits_soft_add / soft_sub
#
# Explicitly NOT claimed:
#   - Madaros-run language `f128` op lowering to softfloat
#   - General softfloat-in-sio for all hard corpus rows
#   - print_f128 builtin / Knowledge / GUM / MeasuredF256
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"
SEED_COMPILER="$(realpath "${SOUNIO_F128_SEED_COMPILER:-$ROOT_DIR/bin/souc-lean-single-x86_64}")"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e4.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

echo "=== madaros_f128_f256_ladder_gate stage=v0e4 ==="
echo "slice=v0e4_anti_f64_softfloat_prep"
echo "claim_clock=sounio_native_expected"
echo "adr=ADR-008+ADR-009"
echo "seed_elf=$SEED_COMPILER"

NATIVE_EXPECT=(
  "wire_1+tiny=00002f3942192484:3fff000000000000"
  "wire_recover=0000000000000000:3fbc79ca10c92420"
)

# Structural: stdlib exact-case API (not overstated soft_add) + supported flag
if grep -Fq 'f128_bits_exact_case_add' stdlib/math/wide_float.sio \
  && grep -Fq 'f128_bits_exact_case_sub' stdlib/math/wide_float.sio \
  && grep -Fq 'struct F128ExactCase' stdlib/math/wide_float.sio \
  && grep -Fq 'supported: bool' stdlib/math/wide_float.sio \
  && grep -Fq 'f128_exact_case_unsupported' stdlib/math/wide_float.sio \
  && ! grep -Fq 'f128_bits_soft_add' stdlib/math/wide_float.sio; then
  note_pass "stdlib_exact_case_api_present"
else
  note_fail "stdlib_exact_case_api_missing_or_overstated"
fi

# Behavioral evidence must import stdlib — refuse fixture-local soft_add copies
if grep -Fq 'use math::wide_float::' tests/run-pass/f128_v0e4_anti_f64_softfloat.sio \
  && grep -Fq 'f128_bits_exact_case_add' tests/run-pass/f128_v0e4_anti_f64_softfloat.sio \
  && ! grep -Fq 'fn f128_soft_add' tests/run-pass/f128_v0e4_anti_f64_softfloat.sio \
  && ! grep -Fq 'fn f128_bits_exact_case_add' tests/run-pass/f128_v0e4_anti_f64_softfloat.sio; then
  note_pass "smoke_calls_stdlib_not_local_copy"
else
  note_fail "smoke_must_import_stdlib_exact_case"
fi

SMOKE=tests/run-pass/f128_v0e4_anti_f64_softfloat.sio
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
    note_fail "seed_build_anti_f64_smoke"
    tail -30 "$BLOG" >&2 || true
  else
    chmod +x "$ELF"
    set +e
    "$ELF" >"$RLOG" 2>&1
    r_rc=$?
    set -e
    if [[ "$r_rc" -ne 0 ]] || ! grep -Fq 'PASS f128_v0e4_anti_f64_softfloat' "$RLOG"; then
      note_fail "seed_run_anti_f64_smoke"
      cat "$RLOG" >&2 || true
    else
      note_pass "seed_run_anti_f64_smoke"
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

# Negative control: lean_single language f128 used to greenwash to f64
# ((1e-20 + 1.0) - 1.0) == 0 under f64; must NOT be claimed as f128.
#
# Updated 2026-09-22: commit 8b99209e41 (#2387, 2026-09-13) gave lean_single
# real binary128 lowering for annotated f128 locals -- the same commit that
# made the language check tests/compile-fail/f128_inexact_literal_refused_on_lean_single.sio
# a positive fixture. lean_single now refuses, at compile time, to widen an
# f64-inexact literal like 1e-20 into a binary128 slot at all (see
# self-hosted/compiler/lean_single.sio: f128_widen_refuses_inexact_literal),
# matching the same fail-closed philosophy Madaros enforces on its own
# lowering path since V0-E.5.9 (no f64 approximation is ever fabricated,
# whether by widening a literal or by greenwashing an op). That refusal IS
# the anti-greenwash evidence now, so this negative control accepts either
# shape: a build that fails with the exactness-refusal message (lean_single's
# current, intentional behavior), or -- should the literal ever be relaxed to
# build -- a run that still proves the result is not f64-greenwashed. Only an
# unrelated build failure, or a build that silently greenwashes, still fails
# this check.
GREEN="$TMP_DIR/greenwash.sio"
cat >"$GREEN" <<'EOF'
fn main() -> i32 with IO {
    let tiny: f128 = 0.00000000000000000001
    let one: f128 = 1.0
    let x = (tiny + one) - one
    if x == 0.0 {
        println("FAIL f64_greenwash")
        return 1
    }
    println("PASS anti_f64")
    return 0
}
EOF
GELF="$TMP_DIR/greenwash.elf"
set +e
"$SEED_COMPILER" "$GREEN" "$GELF" >"$TMP_DIR/greenwash.build.log" 2>&1
gb=$?
set -e
if [[ "$gb" -ne 0 ]] && grep -Fq 'f128 literal is not exactly representable in binary64' "$TMP_DIR/greenwash.build.log"; then
  note_pass "lean_single_language_f128_inexact_literal_refused_no_greenwash"
elif [[ "$gb" -eq 0 && -f "$GELF" ]]; then
  # The literal-exactness refusal is not the only guard: if it were ever
  # relaxed to let this program build, the RESULT still has to be anti-f64.
  # The program's own print reports which one happened -- "FAIL f64_greenwash"
  # means the compiler DID collapse to f64 (a genuine greenwash regression,
  # not a pass), "PASS anti_f64" means it computed the real binary128 answer.
  # Treating the former as note_pass (as this branch used to, mirroring the
  # pre-#2387 test where seeing the program's own "FAIL" print WAS the
  # expected/passing outcome) would let a build that reverts BOTH protections
  # at once -- literal refusal gone AND greenwashing back -- still report this
  # stage green.
  chmod +x "$GELF"
  set +e
  "$GELF" >"$TMP_DIR/greenwash.run.log" 2>&1
  gr=$?
  set -e
  if [[ "$gr" -eq 0 ]] && grep -Fq 'PASS anti_f64' "$TMP_DIR/greenwash.run.log"; then
    note_pass "lean_single_language_f128_build_succeeds_anti_f64_confirmed"
  else
    note_fail "lean_single_language_f128_unexpectedly_f64_greenwash rc=$gr"
    cat "$TMP_DIR/greenwash.run.log" >&2 || true
  fi
else
  note_fail "lean_single_language_f128_build"
  tail -20 "$TMP_DIR/greenwash.build.log" >&2 || true
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

echo "NOTE v0e4_deferred madaros_run_language_softfloat=pending general_softfloat_sio=pending print_builtin=pending gum=pending"
echo "NOTE adr009 python_softfloat=not_claim_clock rust=not_claim_clock"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e4_language_lower exact_case=stdlib anti_f64=green unsupported=explicit lean_single_language=inexact_literal_refused_no_greenwash language_check=ok madaros_run=deferred"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e4"
  exit 0
fi
echo "FAIL madaros_f128_f256_ladder_gate stage=v0e4" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
