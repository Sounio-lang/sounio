#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

SMALL_HEAP_PROBE="self-hosted/compiler/k2_heap_alloc_small_probe.sio"
HEAP_PROBE="self-hosted/compiler/k2_heap_alloc_probe.sio"
BUILTIN_HEAP_PROBE="self-hosted/compiler/k2_heap_alloc_builtin_probe.sio"
LEAF_PROBE="self-hosted/compiler/k2_check_import_leaf_probe.sio"
POINTER_PROBE="self-hosted/compiler/k2_check_import_pointer_probe.sio"
HEAP_CHECKER_PROBE="self-hosted/compiler/k2_check_import_heap_checker_probe.sio"
ACCESSOR_PROBE="self-hosted/compiler/k2_check_import_accessor_probe.sio"
SEMANTIC_PROBE="self-hosted/compiler/k2_check_import_knowledge_context_semantic_probe.sio"
MOD_SEMANTIC_PROBE="self-hosted/compiler/k2_check_mod_semantic_bridge_probe.sio"
COLLECT_ACCESSOR_PROBE="self-hosted/compiler/k2_check_import_collect_accessor_probe.sio"
COLLECT_WITNESS="tests/frontend/knowledge_static_subclass_where_ordinary_probe.sio"
SEMANTIC_MULTI_WITNESS="tests/frontend/knowledge_static_multi_constraint_ordinary_probe.sio"
SEMANTIC_REJECT_WITNESS="tests/frontend/knowledge_static_multi_constraint_reject_ordinary_probe.sio"

bin/souc check "$BUILTIN_HEAP_PROBE"
BUILTIN_HEAP_LOG="$TMP_DIR/builtin-heap.log"
if bin/souc run "$BUILTIN_HEAP_PROBE" >"$BUILTIN_HEAP_LOG" 2>&1 &&
   grep -q 'k2_heap_alloc_builtin_probe=ok' "$BUILTIN_HEAP_LOG"; then
  printf 'PASS  %s allocated through the compiler builtin heap_alloc path\n' "$BUILTIN_HEAP_PROBE"
else
  printf 'FAIL  %s did not complete builtin heap_alloc\n' "$BUILTIN_HEAP_PROBE" >&2
  cat "$BUILTIN_HEAP_LOG" >&2
  exit 1
fi

bin/souc check "$SMALL_HEAP_PROBE"
SMALL_HEAP_LOG="$TMP_DIR/small-heap.log"
if bin/souc run "$SMALL_HEAP_PROBE" >"$SMALL_HEAP_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly completed; expected current mem::box/calloc classifier failure\n' "$SMALL_HEAP_PROBE" >&2
  cat "$SMALL_HEAP_LOG" >&2
  exit 1
fi

if grep -q 'k2_heap_alloc_small_probe=alloc_enter' "$SMALL_HEAP_LOG" &&
   ! grep -q 'k2_heap_alloc_small_probe=alloc_ok' "$SMALL_HEAP_LOG"; then
  printf 'PASS  %s still fails in mem::box/calloc wrapper before alloc_ok\n' "$SMALL_HEAP_PROBE"
else
  printf 'FAIL  %s failed in an unexpected classifier shape\n' "$SMALL_HEAP_PROBE" >&2
  cat "$SMALL_HEAP_LOG" >&2
  exit 1
fi

bin/souc check "$HEAP_PROBE"
HEAP_LOG="$TMP_DIR/heap.log"
if bin/souc run "$HEAP_PROBE" >"$HEAP_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly completed; expected current mem::box/calloc classifier failure\n' "$HEAP_PROBE" >&2
  cat "$HEAP_LOG" >&2
  exit 1
fi

if grep -q 'k2_heap_alloc_probe=alloc_enter' "$HEAP_LOG" &&
   ! grep -q 'k2_heap_alloc_probe=alloc_ok' "$HEAP_LOG"; then
  printf 'PASS  %s still fails in mem::box/calloc wrapper before alloc_ok\n' "$HEAP_PROBE"
else
  printf 'FAIL  %s failed in an unexpected classifier shape\n' "$HEAP_PROBE" >&2
  cat "$HEAP_LOG" >&2
  exit 1
fi

bin/souc check "$LEAF_PROBE"
LEAF_LOG="$TMP_DIR/leaf.log"
if bin/souc run "$LEAF_PROBE" >"$LEAF_LOG" 2>&1 &&
   grep -q 'k2_check_import_leaf_probe=ok' "$LEAF_LOG"; then
  printf 'PASS  %s imported and executed a check::check leaf function\n' "$LEAF_PROBE"
else
  printf 'FAIL  %s did not execute the check::check leaf import\n' "$LEAF_PROBE" >&2
  cat "$LEAF_LOG" >&2
  exit 1
fi

bin/souc check "$POINTER_PROBE"
POINTER_LOG="$TMP_DIR/pointer.log"
if bin/souc run "$POINTER_PROBE" >"$POINTER_LOG" 2>&1 &&
   grep -q 'k2_check_import_pointer_probe=ok' "$POINTER_LOG"; then
  printf 'PASS  %s passed a *mut Checker through a check::check import without dereference\n' "$POINTER_PROBE"
else
  printf 'FAIL  %s did not complete pointer-shaped check::check import\n' "$POINTER_PROBE" >&2
  cat "$POINTER_LOG" >&2
  exit 1
fi

bin/souc check "$HEAP_CHECKER_PROBE"
HEAP_CHECKER_LOG="$TMP_DIR/heap-checker.log"
if bin/souc run "$HEAP_CHECKER_PROBE" >"$HEAP_CHECKER_LOG" 2>&1 &&
   grep -q 'k2_check_import_heap_checker_probe=ok' "$HEAP_CHECKER_LOG"; then
  printf 'PASS  %s allocated heap memory typed as *mut Checker through builtin heap_alloc\n' "$HEAP_CHECKER_PROBE"
else
  printf 'FAIL  %s did not complete builtin Checker heap allocation\n' "$HEAP_CHECKER_PROBE" >&2
  cat "$HEAP_CHECKER_LOG" >&2
  exit 1
fi

bin/souc check "$ACCESSOR_PROBE"
ACCESSOR_LOG="$TMP_DIR/accessor.log"
if bin/souc run "$ACCESSOR_PROBE" >"$ACCESSOR_LOG" 2>&1 &&
   grep -q 'k2_check_import_accessor_probe=ok errors=0' "$ACCESSOR_LOG"; then
  printf 'PASS  %s initialized Checker and read imported scalar accessor\n' "$ACCESSOR_PROBE"
else
  printf 'FAIL  %s did not complete builtin heap/init/accessor bridge\n' "$ACCESSOR_PROBE" >&2
  cat "$ACCESSOR_LOG" >&2
  exit 1
fi

bin/souc check "$SEMANTIC_PROBE"
SEMANTIC_POS_LOG="$TMP_DIR/semantic-positive.log"
if bin/souc run "$SEMANTIC_PROBE" -- "$COLLECT_WITNESS" >"$SEMANTIC_POS_LOG" 2>&1 &&
   grep -q 'k2_check_import_knowledge_context_semantic_probe=ok' "$SEMANTIC_POS_LOG"; then
  printf 'PASS  %s accepted imported in-place ontology semantic witness\n' "$SEMANTIC_PROBE"
else
  printf 'FAIL  %s did not accept the positive ontology semantic witness\n' "$SEMANTIC_PROBE" >&2
  cat "$SEMANTIC_POS_LOG" >&2
  exit 1
fi

SEMANTIC_MULTI_LOG="$TMP_DIR/semantic-multi.log"
if bin/souc run "$SEMANTIC_PROBE" -- "$SEMANTIC_MULTI_WITNESS" >"$SEMANTIC_MULTI_LOG" 2>&1 &&
   grep -q 'k2_check_import_knowledge_context_semantic_probe=ok' "$SEMANTIC_MULTI_LOG"; then
  printf 'PASS  %s accepted imported in-place multi-constraint ontology witness\n' "$SEMANTIC_PROBE"
else
  printf 'FAIL  %s did not accept the multi-constraint ontology semantic witness\n' "$SEMANTIC_PROBE" >&2
  cat "$SEMANTIC_MULTI_LOG" >&2
  exit 1
fi

SEMANTIC_REJECT_LOG="$TMP_DIR/semantic-reject.log"
if bin/souc run "$SEMANTIC_PROBE" -- "$SEMANTIC_REJECT_WITNESS" >"$SEMANTIC_REJECT_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly accepted the negative ontology semantic witness\n' "$SEMANTIC_PROBE" >&2
  cat "$SEMANTIC_REJECT_LOG" >&2
  exit 1
fi

if grep -q 'k2_check_import_knowledge_context_semantic_probe=rejected' "$SEMANTIC_REJECT_LOG"; then
  printf 'PASS  %s rejected imported in-place negative ontology semantic witness\n' "$SEMANTIC_PROBE"
else
  printf 'FAIL  %s failed in an unexpected negative semantic witness shape\n' "$SEMANTIC_PROBE" >&2
  cat "$SEMANTIC_REJECT_LOG" >&2
  exit 1
fi

bin/souc check "$MOD_SEMANTIC_PROBE"
MOD_SEMANTIC_POS_LOG="$TMP_DIR/mod-semantic-positive.log"
if bin/souc run "$MOD_SEMANTIC_PROBE" -- "$SEMANTIC_MULTI_WITNESS" >"$MOD_SEMANTIC_POS_LOG" 2>&1 &&
   grep -q 'k2_check_mod_semantic_bridge_probe=ok' "$MOD_SEMANTIC_POS_LOG"; then
  printf 'PASS  %s positive witness completes through check::mod diagnostic bridge\n' "$MOD_SEMANTIC_PROBE"
else
  printf 'FAIL  %s did not complete the positive check::mod diagnostic bridge\n' "$MOD_SEMANTIC_PROBE" >&2
  cat "$MOD_SEMANTIC_POS_LOG" >&2
  exit 1
fi

MOD_SEMANTIC_REJECT_LOG="$TMP_DIR/mod-semantic-reject.log"
if bin/souc run "$MOD_SEMANTIC_PROBE" -- "$SEMANTIC_REJECT_WITNESS" >"$MOD_SEMANTIC_REJECT_LOG" 2>&1 &&
   grep -q 'k2_check_mod_semantic_bridge_probe=ok' "$MOD_SEMANTIC_REJECT_LOG" &&
   grep -q 'k2_check_mod_semantic_bridge_probe=verdict=0' "$MOD_SEMANTIC_REJECT_LOG"; then
  printf 'PASS  %s still resolves stale through check::mod unqualified bridge and lets the negative witness pass\n' "$MOD_SEMANTIC_PROBE"
else
  printf 'FAIL  %s changed shape; expected current stale unqualified bridge classifier\n' "$MOD_SEMANTIC_PROBE" >&2
  cat "$MOD_SEMANTIC_REJECT_LOG" >&2
  exit 1
fi

bin/souc check "$COLLECT_ACCESSOR_PROBE"
COLLECT_ACCESSOR_LOG="$TMP_DIR/collect-accessor.log"
if bin/souc run "$COLLECT_ACCESSOR_PROBE" -- "$COLLECT_WITNESS" >"$COLLECT_ACCESSOR_LOG" 2>&1; then
  printf 'FAIL  %s unexpectedly completed; expected current imported collect classifier failure\n' "$COLLECT_ACCESSOR_PROBE" >&2
  cat "$COLLECT_ACCESSOR_LOG" >&2
  exit 1
fi

if grep -q 'k2_check_import_collect_accessor_probe=init_ok' "$COLLECT_ACCESSOR_LOG" &&
   ! grep -q 'k2_check_import_collect_accessor_probe=collect_ok' "$COLLECT_ACCESSOR_LOG"; then
  printf 'PASS  %s still segfaults after init and before imported collect_ok\n' "$COLLECT_ACCESSOR_PROBE"
else
  printf 'FAIL  %s failed in an unexpected classifier shape\n' "$COLLECT_ACCESSOR_PROBE" >&2
  cat "$COLLECT_ACCESSOR_LOG" >&2
  exit 1
fi

cat <<'MSG'
K2 check::check import bridge classifier passed.
This proves the compiler builtin heap_alloc path can allocate, explicit imports
from check::check can execute for a leaf function, and a *mut Checker can cross
an imported function boundary. It also proves a standalone driver can allocate
and initialize a Checker and read an imported scalar accessor when it uses the
compiler builtin heap_alloc path. It now also proves the imported in-place
Knowledge<T> ontology semantic path accepts positive and multi-constraint
witnesses while rejecting the negative witness. A check::mod diagnostic bridge
using an unqualified fresh symbol still resolves stale and lets the negative
witness pass, so it remains a classifier rather than an acceptance path. The
mem::box/calloc wrapper remains a separate small-driver failure before alloc_ok.
The remaining imported full Checker collector bridge failure begins at
checker_collect_items_mut after init_ok.
MSG
