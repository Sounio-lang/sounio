#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

POSITIVE="tests/frontend/knowledge_static_diabetes_context.sio"
NEGATIVE="tests/frontend/knowledge_static_rejects_disjoint_diagnosis.sio"
ORDINARY_POSITIVE="tests/frontend/knowledge_static_subclass_where_ordinary_probe.sio"
ORDINARY_TRANSITIVE="tests/frontend/knowledge_static_transitive_where_ordinary_probe.sio"
ORDINARY_NEGATIVE="tests/frontend/knowledge_static_disjoint_where_ordinary_probe.sio"
ORDINARY_MULTI_POSITIVE="tests/frontend/knowledge_static_multi_constraint_ordinary_probe.sio"
ORDINARY_MULTI_NEGATIVE="tests/frontend/knowledge_static_multi_constraint_reject_ordinary_probe.sio"
ORDINARY_RETURN_POSITIVE="tests/frontend/knowledge_static_return_subclass_where_ordinary_probe.sio"
ORDINARY_RETURN_NEGATIVE="tests/frontend/knowledge_static_return_disjoint_where_ordinary_probe.sio"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

FRONTEND="$TMP_DIR/souc-lean-frontend-knowledge-phase2"
bin/souc self-hosted/compiler/lean_frontend.sio "$FRONTEND"

AST_PROBE="$TMP_DIR/knowledge-context-ast-probe"
bin/souc self-hosted/ci/knowledge_context_ast_probe.sio "$AST_PROBE"

POSITIVE_LOG="$TMP_DIR/positive.log"
if "$FRONTEND" --check "$POSITIVE" >"$POSITIVE_LOG" 2>&1; then
  printf 'PASS  %s accepted by Stage1 checker verdict ontology proof-context bridge\n' "$POSITIVE"
else
  printf 'FAIL  Stage1 rejected positive ontology proof-context witness: %s\n' "$POSITIVE" >&2
  cat "$POSITIVE_LOG" >&2
  exit 1
fi

NEGATIVE_LOG="$TMP_DIR/negative.log"
if "$FRONTEND" --check "$NEGATIVE" >"$NEGATIVE_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly passed Stage1 ontology proof-context checker\n' "$NEGATIVE" >&2
  cat "$NEGATIVE_LOG" >&2
  exit 1
fi

printf 'PASS  %s rejected by Stage1 checker verdict ontology proof-context bridge\n' "$NEGATIVE"

ORDINARY_POSITIVE_AST_LOG="$TMP_DIR/ordinary-positive-ast.log"
if "$AST_PROBE" "$ORDINARY_POSITIVE" >"$ORDINARY_POSITIVE_AST_LOG" 2>&1 &&
   grep -q 'knowledge_constraint_count=1' "$ORDINARY_POSITIVE_AST_LOG"; then
  printf 'PASS  %s parsed by ordinary route with one proof constraint\n' "$ORDINARY_POSITIVE"
else
  printf 'FAIL  ordinary parser/AST probe did not retain proof constraint: %s\n' "$ORDINARY_POSITIVE" >&2
  cat "$ORDINARY_POSITIVE_AST_LOG" >&2
  exit 1
fi

ORDINARY_NEGATIVE_AST_LOG="$TMP_DIR/ordinary-negative-ast.log"
if "$AST_PROBE" "$ORDINARY_NEGATIVE" >"$ORDINARY_NEGATIVE_AST_LOG" 2>&1 &&
   grep -q 'knowledge_constraint_count=1' "$ORDINARY_NEGATIVE_AST_LOG"; then
  printf 'PASS  %s parsed by ordinary route with one proof constraint\n' "$ORDINARY_NEGATIVE"
else
  printf 'FAIL  ordinary parser/AST probe did not retain proof constraint: %s\n' "$ORDINARY_NEGATIVE" >&2
  cat "$ORDINARY_NEGATIVE_AST_LOG" >&2
  exit 1
fi

ORDINARY_TRANSITIVE_AST_LOG="$TMP_DIR/ordinary-transitive-ast.log"
if "$AST_PROBE" "$ORDINARY_TRANSITIVE" >"$ORDINARY_TRANSITIVE_AST_LOG" 2>&1 &&
   grep -q 'knowledge_constraint_count=1' "$ORDINARY_TRANSITIVE_AST_LOG"; then
  printf 'PASS  %s parsed by ordinary route with one proof constraint\n' "$ORDINARY_TRANSITIVE"
else
  printf 'FAIL  ordinary parser/AST probe did not retain proof constraint: %s\n' "$ORDINARY_TRANSITIVE" >&2
  cat "$ORDINARY_TRANSITIVE_AST_LOG" >&2
  exit 1
fi

ORDINARY_RETURN_POSITIVE_AST_LOG="$TMP_DIR/ordinary-return-positive-ast.log"
if "$AST_PROBE" "$ORDINARY_RETURN_POSITIVE" >"$ORDINARY_RETURN_POSITIVE_AST_LOG" 2>&1 &&
   grep -q 'knowledge_constraint_count=1' "$ORDINARY_RETURN_POSITIVE_AST_LOG"; then
  printf 'PASS  %s parsed by ordinary route with one return proof constraint\n' "$ORDINARY_RETURN_POSITIVE"
else
  printf 'FAIL  ordinary parser/AST probe did not retain return proof constraint: %s\n' "$ORDINARY_RETURN_POSITIVE" >&2
  cat "$ORDINARY_RETURN_POSITIVE_AST_LOG" >&2
  exit 1
fi

ORDINARY_RETURN_NEGATIVE_AST_LOG="$TMP_DIR/ordinary-return-negative-ast.log"
if "$AST_PROBE" "$ORDINARY_RETURN_NEGATIVE" >"$ORDINARY_RETURN_NEGATIVE_AST_LOG" 2>&1 &&
   grep -q 'knowledge_constraint_count=1' "$ORDINARY_RETURN_NEGATIVE_AST_LOG"; then
  printf 'PASS  %s parsed by ordinary route with one return proof constraint\n' "$ORDINARY_RETURN_NEGATIVE"
else
  printf 'FAIL  ordinary parser/AST probe did not retain return proof constraint: %s\n' "$ORDINARY_RETURN_NEGATIVE" >&2
  cat "$ORDINARY_RETURN_NEGATIVE_AST_LOG" >&2
  exit 1
fi

ORDINARY_MULTI_POSITIVE_AST_LOG="$TMP_DIR/ordinary-multi-positive-ast.log"
if "$AST_PROBE" "$ORDINARY_MULTI_POSITIVE" >"$ORDINARY_MULTI_POSITIVE_AST_LOG" 2>&1 &&
   grep -q 'knowledge_constraint_count=2' "$ORDINARY_MULTI_POSITIVE_AST_LOG"; then
  printf 'PASS  %s parsed by ordinary route with two proof constraints\n' "$ORDINARY_MULTI_POSITIVE"
else
  printf 'FAIL  ordinary parser/AST probe did not retain two proof constraints: %s\n' "$ORDINARY_MULTI_POSITIVE" >&2
  cat "$ORDINARY_MULTI_POSITIVE_AST_LOG" >&2
  exit 1
fi

ORDINARY_MULTI_NEGATIVE_AST_LOG="$TMP_DIR/ordinary-multi-negative-ast.log"
if "$AST_PROBE" "$ORDINARY_MULTI_NEGATIVE" >"$ORDINARY_MULTI_NEGATIVE_AST_LOG" 2>&1 &&
   grep -q 'knowledge_constraint_count=2' "$ORDINARY_MULTI_NEGATIVE_AST_LOG"; then
  printf 'PASS  %s parsed by ordinary route with two proof constraints\n' "$ORDINARY_MULTI_NEGATIVE"
else
  printf 'FAIL  ordinary parser/AST probe did not retain two proof constraints: %s\n' "$ORDINARY_MULTI_NEGATIVE" >&2
  cat "$ORDINARY_MULTI_NEGATIVE_AST_LOG" >&2
  exit 1
fi

ORDINARY_POSITIVE_LOG="$TMP_DIR/ordinary-positive.log"
if "$FRONTEND" --check "$ORDINARY_POSITIVE" >"$ORDINARY_POSITIVE_LOG" 2>&1; then
  printf 'PASS  %s accepted by ordinary Stage1 checker verdict ontology proof-context bridge\n' "$ORDINARY_POSITIVE"
else
  printf 'FAIL  ordinary Stage1 checker verdict bridge rejected positive witness: %s\n' "$ORDINARY_POSITIVE" >&2
  cat "$ORDINARY_POSITIVE_LOG" >&2
  exit 1
fi

ORDINARY_TRANSITIVE_LOG="$TMP_DIR/ordinary-transitive.log"
if "$FRONTEND" --check "$ORDINARY_TRANSITIVE" >"$ORDINARY_TRANSITIVE_LOG" 2>&1; then
  printf 'PASS  %s accepted by ordinary Stage1 checker verdict ontology proof-context bridge\n' "$ORDINARY_TRANSITIVE"
else
  printf 'FAIL  ordinary Stage1 checker verdict bridge rejected transitive subclass witness: %s\n' "$ORDINARY_TRANSITIVE" >&2
  cat "$ORDINARY_TRANSITIVE_LOG" >&2
  exit 1
fi

ORDINARY_NEGATIVE_LOG="$TMP_DIR/ordinary-negative.log"
if "$FRONTEND" --check "$ORDINARY_NEGATIVE" >"$ORDINARY_NEGATIVE_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly passed ordinary Stage1 checker verdict ontology proof-context bridge\n' "$ORDINARY_NEGATIVE" >&2
  cat "$ORDINARY_NEGATIVE_LOG" >&2
  exit 1
fi

printf 'PASS  %s rejected by ordinary Stage1 checker verdict ontology proof-context bridge\n' "$ORDINARY_NEGATIVE"

ORDINARY_MULTI_POSITIVE_LOG="$TMP_DIR/ordinary-multi-positive.log"
if "$FRONTEND" --check "$ORDINARY_MULTI_POSITIVE" >"$ORDINARY_MULTI_POSITIVE_LOG" 2>&1; then
  printf 'PASS  %s accepted by ordinary Stage1 checker verdict multi-constraint ontology proof-context bridge\n' "$ORDINARY_MULTI_POSITIVE"
else
  printf 'FAIL  ordinary Stage1 checker verdict bridge rejected multi-constraint positive witness: %s\n' "$ORDINARY_MULTI_POSITIVE" >&2
  cat "$ORDINARY_MULTI_POSITIVE_LOG" >&2
  exit 1
fi

ORDINARY_MULTI_NEGATIVE_LOG="$TMP_DIR/ordinary-multi-negative.log"
if "$FRONTEND" --check "$ORDINARY_MULTI_NEGATIVE" >"$ORDINARY_MULTI_NEGATIVE_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly passed ordinary Stage1 checker verdict multi-constraint ontology proof-context bridge\n' "$ORDINARY_MULTI_NEGATIVE" >&2
  cat "$ORDINARY_MULTI_NEGATIVE_LOG" >&2
  exit 1
fi

printf 'PASS  %s rejected by ordinary Stage1 checker verdict multi-constraint ontology proof-context bridge\n' "$ORDINARY_MULTI_NEGATIVE"

ORDINARY_RETURN_POSITIVE_LOG="$TMP_DIR/ordinary-return-positive.log"
if "$FRONTEND" --check "$ORDINARY_RETURN_POSITIVE" >"$ORDINARY_RETURN_POSITIVE_LOG" 2>&1; then
  printf 'PASS  %s accepted by ordinary Stage1 checker verdict return ontology proof-context bridge\n' "$ORDINARY_RETURN_POSITIVE"
else
  printf 'FAIL  ordinary Stage1 checker verdict bridge rejected return positive witness: %s\n' "$ORDINARY_RETURN_POSITIVE" >&2
  cat "$ORDINARY_RETURN_POSITIVE_LOG" >&2
  exit 1
fi

ORDINARY_RETURN_NEGATIVE_LOG="$TMP_DIR/ordinary-return-negative.log"
if "$FRONTEND" --check "$ORDINARY_RETURN_NEGATIVE" >"$ORDINARY_RETURN_NEGATIVE_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly passed ordinary Stage1 checker verdict return ontology proof-context bridge\n' "$ORDINARY_RETURN_NEGATIVE" >&2
  cat "$ORDINARY_RETURN_NEGATIVE_LOG" >&2
  exit 1
fi

printf 'PASS  %s rejected by ordinary Stage1 checker verdict return ontology proof-context bridge\n' "$ORDINARY_RETURN_NEGATIVE"

cat <<'MSG'
Knowledge proof-context Phase 2 ontology gate passed.
This proves static field-level ontology constraints are checked by the
Stage1 checker verdict bridge through the dedicated check::knowledge_context
module for parameter and return types, including multi-constraint proof
contexts, and that the ordinary parser/AST route now retains and validates
proof constraints without the earlier parse segfault, source-text preflight,
frontend-owned AST precheck, or check::mod-local duplicate validator. The
ordinary Stage1 verdict path still uses check::knowledge_context; the separate
K2 import bridge classifier covers the standalone check::check semantic path.
MSG
