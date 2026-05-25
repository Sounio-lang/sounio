#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROBE="self-hosted/compiler/k2_knowledge_runtime_obligation_probe.sio"
STATIC_POSITIVE="tests/frontend/knowledge_static_value_positive.sio"
DYNAMIC_RETURN="tests/frontend/knowledge_static_value_dynamic_gap.sio"
DYNAMIC_CALL="tests/frontend/knowledge_static_value_call_dynamic_gap.sio"
DYNAMIC_ASSIGN="tests/frontend/knowledge_static_value_assign_dynamic_gap.sio"

STATIC_LOG="$TMP_DIR/static-positive.log"
if bin/souc run "$PROBE" -- "$STATIC_POSITIVE" >"$STATIC_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_obligation_verdict=0' "$STATIC_LOG" &&
   grep -q 'knowledge_runtime_obligation_count=0' "$STATIC_LOG"; then
  printf 'PASS  %s has no runtime Knowledge<T> proof obligations after static discharge\n' "$STATIC_POSITIVE"
else
  printf 'FAIL  %s should have zero runtime Knowledge<T> obligations\n' "$STATIC_POSITIVE" >&2
  cat "$STATIC_LOG" >&2
  exit 1
fi

DYNAMIC_RETURN_LOG="$TMP_DIR/dynamic-return.log"
if bin/souc run "$PROBE" -- "$DYNAMIC_RETURN" >"$DYNAMIC_RETURN_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_obligation_verdict=0' "$DYNAMIC_RETURN_LOG" &&
   grep -Eq 'knowledge_runtime_obligation_count=[1-9][0-9]*' "$DYNAMIC_RETURN_LOG"; then
  printf 'PASS  %s emits runtime Knowledge<T> proof obligations for nonliteral return fields\n' "$DYNAMIC_RETURN"
else
  printf 'FAIL  %s should emit runtime Knowledge<T> proof obligations\n' "$DYNAMIC_RETURN" >&2
  cat "$DYNAMIC_RETURN_LOG" >&2
  exit 1
fi

DYNAMIC_CALL_LOG="$TMP_DIR/dynamic-call.log"
if bin/souc run "$PROBE" -- "$DYNAMIC_CALL" >"$DYNAMIC_CALL_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_obligation_verdict=0' "$DYNAMIC_CALL_LOG" &&
   grep -Eq 'knowledge_runtime_obligation_count=[1-9][0-9]*' "$DYNAMIC_CALL_LOG"; then
  printf 'PASS  %s emits runtime Knowledge<T> proof obligations at call boundaries\n' "$DYNAMIC_CALL"
else
  printf 'FAIL  %s should emit runtime Knowledge<T> proof obligations at call boundaries\n' "$DYNAMIC_CALL" >&2
  cat "$DYNAMIC_CALL_LOG" >&2
  exit 1
fi

DYNAMIC_ASSIGN_LOG="$TMP_DIR/dynamic-assign.log"
if bin/souc run "$PROBE" -- "$DYNAMIC_ASSIGN" >"$DYNAMIC_ASSIGN_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_obligation_verdict=0' "$DYNAMIC_ASSIGN_LOG" &&
   grep -Eq 'knowledge_runtime_obligation_count=[1-9][0-9]*' "$DYNAMIC_ASSIGN_LOG"; then
  printf 'PASS  %s emits runtime Knowledge<T> proof obligations at assignment sites\n' "$DYNAMIC_ASSIGN"
else
  printf 'FAIL  %s should emit runtime Knowledge<T> proof obligations at assignment sites\n' "$DYNAMIC_ASSIGN" >&2
  cat "$DYNAMIC_ASSIGN_LOG" >&2
  exit 1
fi

cat <<'MSG'
Knowledge runtime obligation gate passed.
This proves dynamic Knowledge<T where {...}> values are no longer only a
silent static gap: the checker exposes runtime proof obligations for nonliteral
numeric field constraints at return, call-boundary, and assignment sites while
fully static values still discharge to zero obligations. Backend guard/trap
insertion remains future work.
MSG
