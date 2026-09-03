#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROBE="self-hosted/compiler/k2_knowledge_unit_constraint_probe.sio"
POSITIVE="tests/frontend/knowledge_static_numeric_constraint_positive.sio"
NEGATIVE_BOOL="tests/frontend/knowledge_static_numeric_constraint_reject_bool.sio"
NEGATIVE_NAME="tests/frontend/knowledge_static_numeric_constraint_reject_name_value.sio"
REFINEMENT_POSITIVE="tests/frontend/knowledge_static_numeric_refinement_positive.sio"
REFINEMENT_NEGATIVE="tests/frontend/knowledge_static_numeric_refinement_reject_weak.sio"

POSITIVE_LOG="$TMP_DIR/knowledge-numeric-positive.log"
if bin/souc run "$PROBE" -- "$POSITIVE" >"$POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_unit_ordinary_verdict=0' "$POSITIVE_LOG" &&
   grep -q 'knowledge_unit_semantic_verdict=0' "$POSITIVE_LOG"; then
  printf 'PASS  %s accepted numeric Knowledge<T> constraint on i64 field\n' "$POSITIVE"
else
  printf 'FAIL  %s did not accept the numeric Knowledge<T> constraint\n' "$POSITIVE" >&2
  cat "$POSITIVE_LOG" >&2
  exit 1
fi

NEGATIVE_BOOL_LOG="$TMP_DIR/knowledge-numeric-reject-bool.log"
if bin/souc run "$PROBE" -- "$NEGATIVE_BOOL" >"$NEGATIVE_BOOL_LOG" 2>&1 &&
   grep -q 'knowledge_unit_ordinary_verdict=1' "$NEGATIVE_BOOL_LOG" &&
   grep -q 'knowledge_unit_semantic_verdict=1' "$NEGATIVE_BOOL_LOG"; then
  printf 'PASS  %s rejected numeric Knowledge<T> constraint on bool field\n' "$NEGATIVE_BOOL"
else
  printf 'FAIL  %s did not reject the numeric Knowledge<T> constraint on bool\n' "$NEGATIVE_BOOL" >&2
  cat "$NEGATIVE_BOOL_LOG" >&2
  exit 1
fi

NEGATIVE_NAME_LOG="$TMP_DIR/knowledge-numeric-reject-name.log"
if bin/souc run "$PROBE" -- "$NEGATIVE_NAME" >"$NEGATIVE_NAME_LOG" 2>&1 &&
   grep -q 'knowledge_unit_ordinary_verdict=1' "$NEGATIVE_NAME_LOG" &&
   grep -q 'knowledge_unit_semantic_verdict=1' "$NEGATIVE_NAME_LOG"; then
  printf 'PASS  %s rejected numeric Knowledge<T> constraint with nonnumeric value\n' "$NEGATIVE_NAME"
else
  printf 'FAIL  %s did not reject the numeric Knowledge<T> constraint with nonnumeric value\n' "$NEGATIVE_NAME" >&2
  cat "$NEGATIVE_NAME_LOG" >&2
  exit 1
fi

REFINEMENT_POSITIVE_LOG="$TMP_DIR/knowledge-numeric-refinement-positive.log"
if bin/souc run "$PROBE" -- "$REFINEMENT_POSITIVE" >"$REFINEMENT_POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_unit_ordinary_verdict=0' "$REFINEMENT_POSITIVE_LOG" &&
   grep -q 'knowledge_unit_semantic_verdict=0' "$REFINEMENT_POSITIVE_LOG"; then
  printf 'PASS  %s accepted Knowledge<T> numeric constraint implied by field refinement\n' "$REFINEMENT_POSITIVE"
else
  printf 'FAIL  %s did not accept the Knowledge<T> constraint implied by refinement\n' "$REFINEMENT_POSITIVE" >&2
  cat "$REFINEMENT_POSITIVE_LOG" >&2
  exit 1
fi

REFINEMENT_NEGATIVE_LOG="$TMP_DIR/knowledge-numeric-refinement-negative.log"
if bin/souc run "$PROBE" -- "$REFINEMENT_NEGATIVE" >"$REFINEMENT_NEGATIVE_LOG" 2>&1 &&
   grep -q 'knowledge_unit_ordinary_verdict=1' "$REFINEMENT_NEGATIVE_LOG" &&
   grep -q 'knowledge_unit_semantic_verdict=1' "$REFINEMENT_NEGATIVE_LOG"; then
  printf 'PASS  %s rejected Knowledge<T> numeric constraint not implied by weak field refinement\n' "$REFINEMENT_NEGATIVE"
else
  printf 'FAIL  %s did not reject the Knowledge<T> constraint not implied by refinement\n' "$REFINEMENT_NEGATIVE" >&2
  cat "$REFINEMENT_NEGATIVE_LOG" >&2
  exit 1
fi

cat <<'MSG'
Knowledge numeric constraint gate passed.
This proves that static `Knowledge<T where { field >= literal }>` constraints
are checked for numeric field/value shape by the ordinary
check::knowledge_context semantic bridge. It also proves that a field refinement
can discharge a Knowledge numeric threshold, and that a weaker refinement is
rejected. The standalone check::check import bridge remains covered by its own
classifier. This does not yet prove threshold
satisfaction for dynamic values.
MSG
