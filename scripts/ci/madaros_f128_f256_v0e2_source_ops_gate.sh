#!/usr/bin/env bash
# madaros_f128_f256_v0e2_source_ops_gate.sh — V0-E.2 source ops through check.
#
# Spec: docs/architecture/F128_F256_LADDER.md §V0-E (V0-E.2 slice)
# Semantic-Lane-ID: WS-G-V0E-STDLIB-GUM-SURFACE
#
# V0-E.2 green (this gate):
#   - Madaros `check` accepts same-format f128/f256 + - * / and comparisons
#   - Mixed f128↔f256 still E004
#   - Casts/implicit widen remain rejected (E248 / type mismatch)
#   - Descriptor probe still green (wide_float ops probe codes 48–51)
#
# Explicitly NOT claimed:
#   - Madaros-run softfloat execution of source ops (codegen lowering)
#   - Builtin print_f128 of language values
#   - Knowledge<f128> / GUM k95 / MeasuredF256 executable
#
# Usage:
#   bash scripts/ci/madaros_f128_f256_v0e2_source_ops_gate.sh
#   bash scripts/ci/madaros_f128_f256_ladder_gate.sh --stage v0e2
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

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/f128-ladder-v0e2.XXXXXX")"
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

echo "=== madaros_f128_f256_ladder_gate stage=v0e2 ==="
echo "slice=v0e2_source_ops_check"
echo "souc=$SOUC"
echo "seed_elf=$SEED_COMPILER"

# Structural: dispatch present
if grep -Fq 'wide_float_binary_result_type' self-hosted/check/compat.sio \
  && grep -Fq 'BinaryOp::OpAdd => left' self-hosted/check/compat.sio; then
  note_pass "structural_wide_float_ops_dispatch"
else
  note_fail "structural_wide_float_ops_dispatch_missing"
fi

# Descriptor probe (includes V0-E.2 probe codes 48–51) via seed
PROBE=self-hosted/compiler/f128_f256_format_descriptor_probe.sio
PELF="$TMP_DIR/desc.elf"
PLOG="$TMP_DIR/desc.log"
if "$SEED_COMPILER" "$ROOT_DIR/$PROBE" "$PELF" >"$PLOG" 2>&1; then
  chmod +x "$PELF"
  if "$PELF" >"$TMP_DIR/desc.run.log" 2>&1 \
    && grep -Fq 'PASS f128_f256_format_descriptor_probe' "$TMP_DIR/desc.run.log"; then
    note_pass "format_descriptor_probe_ops_codes"
  else
    note_fail "format_descriptor_probe_run"
    cat "$TMP_DIR/desc.run.log" >&2 || true
  fi
else
  note_fail "format_descriptor_probe_build"
  tail -30 "$PLOG" >&2 || true
fi

run_check() {
  local src="$1"
  local log="$2"
  "$SOUC" check "$src" >"$log" 2>&1 || true
}

# Positive: arithmetic check OK
for src in \
  tests/run-pass/f128_v0e2_arith_check.sio \
  tests/run-pass/f256_v0e2_arith_check.sio
do
  label="$(basename "$src" .sio)"
  log="$TMP_DIR/${label}.check.log"
  if [[ ! -f "$ROOT_DIR/$src" ]]; then
    note_fail "missing_positive:$src"
    continue
  fi
  run_check "$ROOT_DIR/$src" "$log"
  if grep -Fq 'check: OK' "$log" && ! grep -Eq 'error\[E004\]' "$log"; then
    note_pass "arith_check_ok:$label"
  else
    note_fail "arith_check_failed:$label"
    tail -40 "$log" >&2 || true
  fi
done

# Negatives: mixed still E004; cast/implicit still refused
for src in \
  tests/compile-fail/f128_f256_mixed_arith_rejected.sio \
  tests/compile-fail/f128_v0b_cast_rejected.sio \
  tests/compile-fail/f128_v0b_implicit_conversion_rejected.sio
do
  label="$(basename "$src" .sio)"
  log="$TMP_DIR/${label}.check.log"
  if [[ ! -f "$ROOT_DIR/$src" ]]; then
    note_fail "missing_negative:$src"
    continue
  fi
  run_check "$ROOT_DIR/$src" "$log"
  if grep -Fq 'check: OK' "$log"; then
    note_fail "negative_incorrectly_ok:$label"
    cat "$log" >&2 || true
  else
    note_pass "negative_still_rejected:$label"
  fi
done

echo "NOTE v0e2_deferred madaros_run_softfloat_lowering=pending print_f128_builtin=pending Knowledge_GUM=pending MeasuredF256=pending"

echo "---"
echo "PASS_COUNT=$PASS"
echo "FAIL_COUNT=$FAIL"

if [[ "$FAIL" -eq 0 ]]; then
  echo "PASS f128_f256_v0e2_source_ops check=green same_format=add/sub/mul/div/cmp mixed=E004 cast_implicit=still_rejected madaros_run=deferred"
  echo "PASS madaros_f128_f256_ladder_gate stage=v0e2"
  exit 0
fi

echo "FAIL madaros_f128_f256_ladder_gate stage=v0e2" >&2
for f in "${FAILURES[@]}"; do
  echo "  - $f" >&2
done
exit 1
