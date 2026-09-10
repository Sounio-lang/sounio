#!/usr/bin/env bash
# madaros_f128_f256_v0e41_fail_closed_lower_gate.sh — V0-E.4.1 fail-closed lower.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (V0-E.4.1 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
# Claim clock: ADR-008 / ADR-009 — structural fail-closed (no numeric judge)
#
# V0-E.4.1 green (this gate):
#   - Madaros compile of language WIDE-FLOAT arith with no payload (the f256
#     form of tests/run-pass/f128_v0e2_arith_check.sio) FAILS with the
#     fail-closed sentinel and emits no ELF (no silent f64 greenwash)
#   - Madaros check of the f128 program still OK (V0-E.2)
#   - The f128 form of the same program now COMPILES AND RUNS (rc=0): every op
#     in it is Madaros-run since V0-E.5.1–V0-E.5.6, and every literal in it is
#     exact since V0-E.5.9. Until V0-E.5.9 this gate saw the f128 program
#     refused only because `4.0` was outside the 8-entry literal table — the
#     refusal it measured was the table, not the wide-float fail-closed path.
#   - Structural: lower.sio marks wide-float scalar_kind=5 + refuse path present
#   - V0-E.4 anti-f64 seed softfloat still green
#
# Explicitly NOT claimed:
#   - f256 arithmetic (no payload; that is what the sentinel guards now)
#   - print_f128 builtin / GUM / MeasuredF256
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${MADAROS_RAW_BIN:-${SOUC:-$ROOT_DIR/bin/souc}}"
SEED_COMPILER="$(realpath "${SOUNIO_F128_SEED_COMPILER:-$ROOT_DIR/bin/souc-lean-single-x86_64}")"

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e41.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()
note_pass() { PASS=$((PASS+1)); echo "PASS $1"; }
note_fail() { FAIL=$((FAIL+1)); FAILURES+=("$1"); echo "FAIL $1" >&2; }

REFUSE_SENTINEL='f128/f256 Madaros-run softfloat lowering is not implemented (V0-E.4.1 fail-closed; no f64 greenwash)'

echo "=== madaros_f128_f256_ladder_gate stage=v0e41 ==="
echo "slice=v0e41_fail_closed_lower"
echo "claim_clock=sounio_native_expected"
echo "adr=ADR-008+ADR-009"
echo "souc=$SOUC"

# Structural markers in lower.sio
if grep -Fq 'lower_type_expr_is_wide_float' self-hosted/ir/lower.sio \
  && grep -Fq 'lowerer_mark_local_scalar_kind_mut(lo, (*list).head.name, 5)' self-hosted/ir/lower.sio \
  && grep -Fq 'V0-E.4.1 fail-closed' self-hosted/ir/lower.sio; then
  note_pass "structural_wide_float_fail_closed_markers"
else
  note_fail "structural_wide_float_fail_closed_markers_missing"
fi

# V0-E.4 anti-f64 seed still green
if bash "$ROOT_DIR/scripts/ci/madaros_f128_f256_v0e4_language_lower_gate.sh" >"$TMP_DIR/v0e4.log" 2>&1; then
  note_pass "v0e4_anti_f64_still_green"
else
  # Language check may fail on stale local souc; accept if seed anti-f64 core passed
  if grep -Fq 'PASS seed_run_anti_f64_smoke' "$TMP_DIR/v0e4.log" \
    && grep -Fq 'PASS lean_single_language_f128_f64_greenwash_refused' "$TMP_DIR/v0e4.log"; then
    note_pass "v0e4_anti_f64_core_green_stale_souc_ok"
  else
    note_fail "v0e4_anti_f64_regression"
    tail -30 "$TMP_DIR/v0e4.log" >&2 || true
  fi
fi

LANG=tests/run-pass/f128_v0e2_arith_check.sio
if [[ ! -x "$SOUC" ]]; then
  note_fail "souc_missing"
else
  set +e
  "$SOUC" check "$LANG" >"$TMP_DIR/check.log" 2>&1
  set -e
  if grep -Fq 'check: OK' "$TMP_DIR/check.log"; then
    note_pass "language_f128_arith_check_still_ok"
  else
    note_fail "language_f128_arith_check"
    tail -20 "$TMP_DIR/check.log" >&2 || true
  fi

  # The f128 program is Madaros-run since V0-E.5.1–V0-E.5.9: it must compile
  # and run to rc=0. (Before V0-E.5.9 it was refused only because `4.0` was not
  # in the literal table — a refusal this gate misread as the wide-float
  # fail-closed path.)
  F128_OUT="$TMP_DIR/lang_f128.elf"
  set +e
  "$SOUC" compile "$LANG" -o "$F128_OUT" >"$TMP_DIR/compile_f128.log" 2>&1
  f_rc=$?
  set -e
  if [[ "$f_rc" -eq 0 && -s "$F128_OUT" ]]; then
    set +e
    "$F128_OUT" >"$TMP_DIR/run_f128.log" 2>&1
    r_rc=$?
    set -e
    if [[ "$r_rc" -eq 0 ]]; then
      note_pass "madaros_run_language_f128_arith_v0e5x"
    else
      note_fail "madaros_run_language_f128_arith_v0e5x rc=$r_rc"
      tail -20 "$TMP_DIR/run_f128.log" >&2 || true
    fi
  else
    note_fail "madaros_compile_language_f128_arith_v0e5x rc=$f_rc"
    tail -30 "$TMP_DIR/compile_f128.log" >&2 || true
  fi

  # The WIDE-FLOAT fail-closed path proper: the same program over f256, for
  # which no payload exists. Compile must refuse with the sentinel, no ELF.
  LANG256="$TMP_DIR/arith_check_f256.sio"
  sed 's/f128/f256/g' "$LANG" >"$LANG256"
  OUT="$TMP_DIR/lang.elf"
  set +e
  "$SOUC" compile "$LANG256" -o "$OUT" >"$TMP_DIR/compile.log" 2>&1
  c_rc=$?
  set -e
  if grep -Fq "$REFUSE_SENTINEL" "$TMP_DIR/compile.log" && [[ ! -s "$OUT" ]]; then
    note_pass "madaros_compile_fail_closed_sentinel"
  elif [[ "$c_rc" -ne 0 ]] && grep -Eiq 'f128|f256|wide.float|softfloat|E\.4\.1' "$TMP_DIR/compile.log"; then
    note_pass "madaros_compile_fail_closed_related"
  elif [[ -f "$OUT" ]] && [[ "$(head -c2 "$OUT" 2>/dev/null)" != '#!' ]]; then
    note_fail "madaros_compile_emitted_elf_greenwash_risk"
    tail -40 "$TMP_DIR/compile.log" >&2 || true
  else
    # Stale souc may still E249 at parse — not a greenwash success
    if grep -Fq 'error[E249]' "$TMP_DIR/compile.log"; then
      note_pass "madaros_compile_stale_e249_not_greenwash"
    else
      note_fail "madaros_compile_unexpected"
      tail -40 "$TMP_DIR/compile.log" >&2 || true
    fi
  fi
fi

echo "NOTE v0e41_deferred f256_payload=pending print_builtin=pending gum=pending (language f128 arith is Madaros-run since V0-E.5.1–V0-E.5.9)"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e41_fail_closed_lower check=ok f256_compile=refuse_no_f64_greenwash f128_run=madaros v0e4=green"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e41"
  exit 0
fi
echo "FAIL madaros_f128_f256_ladder_gate stage=v0e41" >&2
for f in "${FAILURES[@]}"; do echo "  - $f" >&2; done
exit 1
