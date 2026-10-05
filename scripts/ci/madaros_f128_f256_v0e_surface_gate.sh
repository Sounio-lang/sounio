#!/usr/bin/env bash
# madaros_f128_f256_v0e_surface_gate.sh — V0-E.1 print + stdlib limb surface.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (print/API slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
#
# V0-E.1 green (this gate):
#   - Prior softfloat (V0-D) still holds (oracle runner).
#   - Deterministic print oracle (decimal + hex) from limb softfloat.
#   - stdlib/math/wide_float.sio present and Madaros-run smoke prints hex wire.
#
# Explicitly NOT claimed here (V0-E.2+):
#   - Language source `f128`/`f256` arithmetic (still E004)
#   - Knowledge<f128> / GUM k95 / MeasuredF256 executable
#   - Builtin print_f128 of language f128 values
#
# Usage:
#   bash scripts/ci/madaros_f128_f256_v0e_surface_gate.sh
#   bash scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0e
#
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

unset SOUC_BIN SOUNIO_SOUC_BIN || true
if [[ -n "${SOUNIO_STDLIB_PATH:-}" && ! -d "${SOUNIO_STDLIB_PATH}" ]]; then
  unset SOUNIO_STDLIB_PATH || true
fi
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"

SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
if [[ -n "${MADAROS_RAW_BIN:-}" && -x "${MADAROS_RAW_BIN}" ]]; then
  SOUC="$MADAROS_RAW_BIN"
fi
if [[ ! -x "$SOUC" ]]; then
  echo "FAIL souc not executable: $SOUC" >&2
  exit 2
fi

SEED_COMPILER="$(realpath "${SOUNIO_F128_SEED_COMPILER:-$ROOT_DIR/bin/souc-lean-single-x86_64}")"
if [[ ! -x "$SEED_COMPILER" ]]; then
  echo "FAIL seed compiler not executable: $SEED_COMPILER" >&2
  exit 2
fi
if [[ "$(head -c2 "$SEED_COMPILER" 2>/dev/null)" == '#!' ]]; then
  echo "FAIL seed must be a resolved ELF, not a wrapper: $SEED_COMPILER" >&2
  exit 2
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e.XXXXXX")"
trap 'rm -rf "$TMP_DIR"' EXIT

PASS=0
FAIL=0
FAILURES=()

note_pass() {
  PASS=$((PASS + 1))
  echo "PASS $1"
}

note_fail() {
  FAIL=$((FAIL + 1))
  FAILURES+=("$1")
  echo "FAIL $1" >&2
}

echo "=== madaros_f128_f256_ladder_gate stage=v0e ==="
echo "slice=v0e1_print_stdlib_limb_surface"
echo "souc_check=$SOUC"
echo "engine_scaffold=lean_single_seed_elf"
echo "seed_elf=$SEED_COMPILER"
echo "stdlib=$SOUNIO_STDLIB_PATH"

# ---------------------------------------------------------------------------
# V0-D softfloat still green (print rests on limb softfloat).
# ---------------------------------------------------------------------------
V0D_RUNNER="$ROOT_DIR/scripts/dev/ws_g_v0d_softfloat_corpus_runner.py"
if [[ ! -f "$V0D_RUNNER" ]]; then
  note_fail "missing_v0d_runner"
else
  set +e
  python3 "$V0D_RUNNER" >"$TMP_DIR/v0d_runner.log" 2>&1
  r=$?
  set -e
  if [[ "$r" -eq 0 ]] && grep -Fq 'bit_identity=all_hard_rows' "$TMP_DIR/v0d_runner.log"; then
    note_pass "v0d_softfloat_bit_identity_still_green"
  else
    note_fail "v0d_softfloat_regression"
    tail -20 "$TMP_DIR/v0d_runner.log" >&2 || true
  fi
fi

# ---------------------------------------------------------------------------
# Print oracle (deterministic decimal + hex; rump ≠ f64 path).
# ---------------------------------------------------------------------------
PRINT_ORACLE="$ROOT_DIR/scripts/dev/ws_g_v0e_wide_print_oracle.py"
if [[ ! -f "$PRINT_ORACLE" ]]; then
  note_fail "missing_print_oracle"
else
  set +e
  python3 "$PRINT_ORACLE" >"$TMP_DIR/print_oracle.log" 2>&1
  r=$?
  set -e
  while IFS= read -r line; do
    case "$line" in
      PASS\ *) note_pass "${line#PASS }" ;;
      FAIL\ *) note_fail "${line#FAIL }" ;;
      *) echo "$line" ;;
    esac
  done <"$TMP_DIR/print_oracle.log"
  if [[ "$r" -ne 0 ]]; then
    note_fail "print_oracle_exit=$r"
  fi
fi

# ---------------------------------------------------------------------------
# Stdlib API present.
# ---------------------------------------------------------------------------
WF="$ROOT_DIR/stdlib/math/wide_float.sio"
if [[ -f "$WF" ]] && grep -Fq 'pub struct F128Bits' "$WF" && grep -Fq 'print_f128_bits_hex_wire' "$WF"; then
  note_pass "stdlib_wide_float_api_present"
else
  note_fail "stdlib_wide_float_api_missing"
fi

# ---------------------------------------------------------------------------
# Madaros check + seed-run smoke (limb hex wire).
# Madaros codegen still hits capacity walls on this shape; V0-D scaffolds use
# the same seed-ELF pattern for runnable probes.
# ---------------------------------------------------------------------------
SMOKE="$ROOT_DIR/tests/run-pass/f128_v0e_stdlib_smoke.sio"
ELF="$TMP_DIR/f128_v0e_stdlib_smoke.elf"
BLOG="$TMP_DIR/smoke.build.log"
RLOG="$TMP_DIR/smoke.run.log"
CLOG="$TMP_DIR/smoke.check.log"

if [[ ! -f "$SMOKE" ]]; then
  note_fail "missing_smoke:$SMOKE"
else
  set +e
  "$SOUC" check "$SMOKE" >"$CLOG" 2>&1
  c_rc=$?
  set -e
  if [[ "$c_rc" -eq 0 ]] && grep -Fq 'check: OK' "$CLOG"; then
    note_pass "madaros_v0e_stdlib_smoke_check"
  else
    note_fail "madaros_v0e_stdlib_smoke_check"
    tail -30 "$CLOG" >&2 || true
  fi

  set +e
  "$SEED_COMPILER" "$SMOKE" "$ELF" >"$BLOG" 2>&1
  b_rc=$?
  set -e
  if [[ "$b_rc" -ne 0 || ! -f "$ELF" ]]; then
    note_fail "seed_v0e_stdlib_smoke_build"
    tail -40 "$BLOG" >&2 || true
  else
    chmod +x "$ELF"
    set +e
    "$ELF" >"$RLOG" 2>&1
    r_rc=$?
    set -e
    if [[ "$r_rc" -eq 0 ]] && grep -Fq 'PASS f128_v0e_stdlib_smoke' "$RLOG"; then
      note_pass "seed_v0e_stdlib_smoke_run"
      if grep -Fq 'wire_one=0000000000000000:3fff000000000000' "$RLOG"; then
        note_pass "hex_wire_one_matches_softfloat"
      else
        note_fail "hex_wire_one_mismatch"
        cat "$RLOG" >&2 || true
      fi
      if grep -Fq 'wire_neg=0000000000000000:bfff000000000000' "$RLOG"; then
        note_pass "hex_wire_neg_matches_softfloat"
      else
        note_fail "hex_wire_neg_mismatch"
        cat "$RLOG" >&2 || true
      fi
      if grep -Fq 'wire_f256_one=0000000000000000:0000000000000000:0000000000000000:3ffff00000000000' "$RLOG"; then
        note_pass "hex_wire_f256_one_matches_softfloat"
      else
        note_fail "hex_wire_f256_one_mismatch"
        cat "$RLOG" >&2 || true
      fi
    else
      note_fail "seed_v0e_stdlib_smoke_run"
      cat "$RLOG" >&2 || true
    fi
  fi
fi

# ---------------------------------------------------------------------------
# Deferred claims (informational; do not fail the V0-E.1 slice).
# ---------------------------------------------------------------------------
echo "NOTE v0e2_deferred language_f128_arithmetic=E004 Knowledge_f128=deferred GUM_k95=deferred MeasuredF256=not_executable print_f128_builtin=deferred"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"
echo "slice=v0e1_print_stdlib_limb_surface"

if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e_surface print=deterministic stdlib=wide_float.sio hex_wire=softfloat_match madaros_check=green seed_smoke=green gum=deferred MeasuredF256=deferred"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e"
  exit 0
fi

echo "FAIL madaros_f128_f256_ladder_gate stage=v0e" >&2
echo "first_failures:" >&2
for f in "${FAILURES[@]}"; do
  echo "  - $f" >&2
done
exit 1
