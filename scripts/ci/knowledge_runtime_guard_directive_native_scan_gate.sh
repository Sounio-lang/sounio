#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

PROBE="self-hosted/compiler/k2_knowledge_runtime_guard_directive_scan_probe.sio"
POSITIVE_SRC="tests/frontend/knowledge_runtime_guard_directive_positive.sio"
NO_DIRECTIVE_SRC="tests/frontend/knowledge_runtime_guard_call_positive.sio"
MALFORMED_SRC="$TMP_DIR/knowledge-runtime-guards-malformed.sio"
POSITIVE_LOG="$TMP_DIR/knowledge-runtime-guards-positive.log"
NO_DIRECTIVE_LOG="$TMP_DIR/knowledge-runtime-guards-no-directive.log"
MALFORMED_LOG="$TMP_DIR/knowledge-runtime-guards-malformed.log"

printf '%s\n' '//@ knowledge-runtime-guards: raw payload is intentionally malformed' >"$MALFORMED_SRC"
printf '%s\n' 'fn main() -> i64 { 0 }' >>"$MALFORMED_SRC"

if bin/souc run "$PROBE" -- "$POSITIVE_SRC" >"$POSITIVE_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_guard_directive_found=1' "$POSITIVE_LOG" &&
   grep -q 'count=1' "$POSITIVE_LOG" &&
   grep -q 'malformed=0' "$POSITIVE_LOG" &&
   grep -q 'pre_native_guard_expansion=1' "$POSITIVE_LOG" &&
   grep -q 'native_lowering_ready=0' "$POSITIVE_LOG"; then
  printf 'PASS  compiler-side knowledge-runtime-guards directive scanner detected valid raw-source intent\n'
else
  printf 'FAIL  compiler-side knowledge-runtime-guards directive scanner did not detect valid raw-source intent\n' >&2
  cat "$POSITIVE_LOG" >&2
  exit 1
fi

if bin/souc run "$PROBE" -- "$NO_DIRECTIVE_SRC" >"$NO_DIRECTIVE_LOG" 2>&1 &&
   grep -q 'knowledge_runtime_guard_directive_found=0' "$NO_DIRECTIVE_LOG" &&
   grep -q 'count=0' "$NO_DIRECTIVE_LOG" &&
   grep -q 'malformed=0' "$NO_DIRECTIVE_LOG" &&
   grep -q 'pre_native_guard_expansion=0' "$NO_DIRECTIVE_LOG" &&
   grep -q 'native_lowering_ready=0' "$NO_DIRECTIVE_LOG"; then
  printf 'PASS  compiler-side knowledge-runtime-guards directive scanner leaves ordinary sources inactive\n'
else
  printf 'FAIL  compiler-side knowledge-runtime-guards directive scanner unexpectedly activated ordinary source\n' >&2
  cat "$NO_DIRECTIVE_LOG" >&2
  exit 1
fi

if bin/souc run "$PROBE" -- "$MALFORMED_SRC" >"$MALFORMED_LOG" 2>&1; then
  printf 'FAIL  malformed knowledge-runtime-guards directive unexpectedly produced a valid intent\n' >&2
  cat "$MALFORMED_LOG" >&2
  exit 1
fi
if grep -q 'knowledge_runtime_guard_directive_found=0' "$MALFORMED_LOG" &&
   grep -q 'count=0' "$MALFORMED_LOG" &&
   grep -q 'malformed=1' "$MALFORMED_LOG" &&
   grep -q 'malformed_count=1' "$MALFORMED_LOG" &&
   grep -q 'pre_native_guard_expansion=0' "$MALFORMED_LOG" &&
   grep -q 'native_lowering_ready=0' "$MALFORMED_LOG"; then
  printf 'PASS  compiler-side knowledge-runtime-guards directive scanner classified malformed payload-bearing directive\n'
else
  printf 'FAIL  malformed knowledge-runtime-guards directive did not produce expected classifier output\n' >&2
  cat "$MALFORMED_LOG" >&2
  exit 1
fi

cat <<'MSG'
Knowledge runtime guard directive native scan gate passed.
This proves the self-hosted compiler side has a Sounio raw-source scanner for
`//@ knowledge-runtime-guards` before ordinary lexing drops comments. The
scanner currently classifies intent for the existing pre-native expansion path:
valid sources request pre-native guard expansion, ordinary sources remain
inactive, and payload-bearing malformed directives are rejected. The intent
surface explicitly reports native_lowering_ready=0 because direct native
lowering of checker-visible Knowledge<T> proof obligations remains a future
rung.
MSG
