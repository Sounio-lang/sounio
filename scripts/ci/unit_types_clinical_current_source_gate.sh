#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

CURRENT_SOUC="$TMP_DIR/sounio-current-source"
BUILD_LOG="$TMP_DIR/build-current-source.log"
POSITIVE="tests/frontend/unit_literal_clinical_current_source.sio"
NEGATIVE="tests/compile-fail/unit_literal_clinical_reject_mass_as_amount_concentration.sio"

if bin/souc-linux-x86_64 self-hosted/compiler/lean_single.sio "$CURRENT_SOUC" >"$BUILD_LOG" 2>&1; then
  chmod +x "$CURRENT_SOUC"
  printf 'PASS  built current-source lean_single compiler for internal label gate\n'
else
  printf 'FAIL  could not build current-source lean_single compiler\n' >&2
  cat "$BUILD_LOG" >&2
  exit 1
fi

POSITIVE_LOG="$TMP_DIR/unit-literal-clinical.log"
# bin/souc translates its verbs only for the engine binaries. With
# SOUNIO_SOUC_BIN set it execs that ELF with the arguments verbatim --
# documented at the top of bin/souc as "raw, lean_single CLI" -- so
# `bin/souc run FILE` reached the compiler as `run FILE`, and it took
# the verb as the source filename and reported E221 no main. The gate
# was reporting the feature missing when it had never invoked it.
if "$CURRENT_SOUC" "$POSITIVE" "$TMP_DIR/positive.elf" >"$POSITIVE_LOG" 2>&1 &&
   chmod +x "$TMP_DIR/positive.elf" &&
   "$TMP_DIR/positive.elf" >>"$POSITIVE_LOG" 2>&1 &&
   grep -q 'unit literal clinical: PASS' "$POSITIVE_LOG"; then
  printf 'PASS  %s accepted clinical numeric literal unit suffixes\n' "$POSITIVE"
else
  printf 'FAIL  %s did not accept clinical numeric literal unit suffixes\n' "$POSITIVE" >&2
  cat "$POSITIVE_LOG" >&2
  exit 1
fi

NEGATIVE_LOG="$TMP_DIR/unit-literal-clinical-reject.log"
if "$CURRENT_SOUC" "$NEGATIVE" "$TMP_DIR/negative.elf" >"$NEGATIVE_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly accepted mass concentration where amount concentration was required\n' "$NEGATIVE" >&2
  cat "$NEGATIVE_LOG" >&2
  exit 1
fi
if grep -q 'unit mismatch in call argument' "$NEGATIVE_LOG"; then
  printf 'PASS  %s rejected incompatible internal label dimension\n' "$NEGATIVE"
else
  printf 'FAIL  %s failed without the expected unit mismatch diagnostic\n' "$NEGATIVE" >&2
  cat "$NEGATIVE_LOG" >&2
  exit 1
fi

cat <<'MSG'
Clinical current-source unit gate passed.
This rebuilds the compiler from self-hosted/compiler/lean_single.sio and shows
declared internal dimension labels such as mg_dL, mmol_L, U_L, and mm_h participate
as internal current-source unit identifiers with no built-in conversion
factors. It does not prove clinical correctness, terminology conformance, unit
conversion safety, dosing safety, modular UnitRegistry coverage, or Knowledge<T>
clinical-unit coverage.
MSG
