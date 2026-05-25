#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

UNIT_ADD="tests/run-pass/unit_same_add.sio"
UNIT_SCALAR_MUL="tests/run-pass/unit_scalar_mul.sio"
UNIT_DIV_CANCEL="tests/run-pass/unit_div_cancel.sio"
UNIT_CAST_COMPAT="tests/run-pass/unit_cast_compatible.sio"
UNIT_MISMATCH_ADD="tests/compile-fail/unit_mismatch_add.sio"
UNIT_MISMATCH_CALL="tests/compile-fail/unit_mismatch_call_arg.sio"
UNIT_CAST_INCOMPAT="tests/compile-fail/unit_cast_incompatible.sio"

ADD_LOG="$TMP_DIR/unit-add.log"
if bin/souc run "$UNIT_ADD" >"$ADD_LOG" 2>&1 &&
   grep -q 'unit add: PASS' "$ADD_LOG"; then
  printf 'PASS  %s preserved unit through same-unit addition\n' "$UNIT_ADD"
else
  printf 'FAIL  %s did not complete same-unit addition witness\n' "$UNIT_ADD" >&2
  cat "$ADD_LOG" >&2
  exit 1
fi

MUL_LOG="$TMP_DIR/unit-scalar-mul.log"
if bin/souc run "$UNIT_SCALAR_MUL" >"$MUL_LOG" 2>&1 &&
   grep -q 'unit scalar mul: PASS' "$MUL_LOG"; then
  printf 'PASS  %s preserved unit through scalar multiplication\n' "$UNIT_SCALAR_MUL"
else
  printf 'FAIL  %s did not complete scalar multiplication witness\n' "$UNIT_SCALAR_MUL" >&2
  cat "$MUL_LOG" >&2
  exit 1
fi

DIV_CANCEL_LOG="$TMP_DIR/unit-div-cancel.log"
if bin/souc run "$UNIT_DIV_CANCEL" >"$DIV_CANCEL_LOG" 2>&1 &&
   grep -q 'unit div cancel: PASS' "$DIV_CANCEL_LOG"; then
  printf 'PASS  %s cancelled same-unit division to a dimensionless scalar\n' "$UNIT_DIV_CANCEL"
else
  printf 'FAIL  %s did not complete same-unit division cancellation witness\n' "$UNIT_DIV_CANCEL" >&2
  cat "$DIV_CANCEL_LOG" >&2
  exit 1
fi

CAST_COMPAT_LOG="$TMP_DIR/unit-cast-compatible.log"
if bin/souc check "$UNIT_CAST_COMPAT" >"$CAST_COMPAT_LOG" 2>&1; then
  printf 'PASS  %s accepted compatible unit cast\n' "$UNIT_CAST_COMPAT"
else
  printf 'FAIL  %s rejected compatible unit cast\n' "$UNIT_CAST_COMPAT" >&2
  cat "$CAST_COMPAT_LOG" >&2
  exit 1
fi

MISMATCH_ADD_LOG="$TMP_DIR/unit-mismatch-add.log"
if bin/souc check "$UNIT_MISMATCH_ADD" >"$MISMATCH_ADD_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly accepted incompatible unit addition\n' "$UNIT_MISMATCH_ADD" >&2
  cat "$MISMATCH_ADD_LOG" >&2
  exit 1
fi
if grep -q 'unit dimension mismatch' "$MISMATCH_ADD_LOG" ||
   grep -q 'unit mismatch' "$MISMATCH_ADD_LOG"; then
  printf 'PASS  %s rejected incompatible unit addition\n' "$UNIT_MISMATCH_ADD"
else
  printf 'FAIL  %s failed without the expected unit mismatch diagnostic\n' "$UNIT_MISMATCH_ADD" >&2
  cat "$MISMATCH_ADD_LOG" >&2
  exit 1
fi

MISMATCH_CALL_LOG="$TMP_DIR/unit-mismatch-call.log"
if bin/souc check "$UNIT_MISMATCH_CALL" >"$MISMATCH_CALL_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly accepted incompatible unit call argument\n' "$UNIT_MISMATCH_CALL" >&2
  cat "$MISMATCH_CALL_LOG" >&2
  exit 1
fi
if grep -q 'unit mismatch in call argument' "$MISMATCH_CALL_LOG"; then
  printf 'PASS  %s rejected incompatible unit call argument\n' "$UNIT_MISMATCH_CALL"
else
  printf 'FAIL  %s failed without the expected unit call diagnostic\n' "$UNIT_MISMATCH_CALL" >&2
  cat "$MISMATCH_CALL_LOG" >&2
  exit 1
fi

CAST_INCOMPAT_LOG="$TMP_DIR/unit-cast-incompatible.log"
if bin/souc check "$UNIT_CAST_INCOMPAT" >"$CAST_INCOMPAT_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly accepted incompatible unit cast\n' "$UNIT_CAST_INCOMPAT" >&2
  cat "$CAST_INCOMPAT_LOG" >&2
  exit 1
fi
if grep -q 'incompatible unit dimensions' "$CAST_INCOMPAT_LOG"; then
  printf 'PASS  %s rejected incompatible unit cast\n' "$UNIT_CAST_INCOMPAT"
else
  printf 'FAIL  %s failed without the expected incompatible-unit diagnostic\n' "$UNIT_CAST_INCOMPAT" >&2
  cat "$CAST_INCOMPAT_LOG" >&2
  exit 1
fi

cat <<'MSG'
Unit types Phase 1 gate passed.
This proves the existing checker unit registry and unit_id type annotations
cover same-unit addition, scalar multiplication, compatible casts, incompatible
addition, incompatible call arguments, incompatible casts, and same-unit
division cancellation to a dimensionless scalar. Derived dimension naming for
non-cancelling products/quotients, current-source named literal unit suffixes,
and current-source explicit-conversion requirements for J/eV are covered by
the separate derived/current-source gate.
Ontology-linked unit metadata is covered by the separate validation-data gate.
MSG
