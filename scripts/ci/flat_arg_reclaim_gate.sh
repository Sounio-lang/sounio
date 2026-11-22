#!/usr/bin/env bash
# Linux runtime proof: the modular emitter implements __arena_mark/reset and heap_alloc.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="${1:?pass native Linux raw Madaros}"
OUT="${2:?pass new output directory}"
[[ "$(uname -s)/$(uname -m)" = Linux/x86_64 ]] || { echo 'Linux x86-64 runtime required' >&2; exit 2; }
mkdir -p "$OUT"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="$ROOT/stdlib"
source "$ROOT/scripts/lib/souc_invoke.sh"
_souc_limits
"${CXX:-c++}" -std=c++23 -O2 -Wall -Wextra -Werror scripts/ci/flat_arg_reclaim_probe.cpp -o "$OUT/generate"
"$OUT/generate" "$ROOT" "$OUT/probe.sio"
"$HOST" check "$OUT/probe.sio" >"$OUT/check.log" 2>&1
"$HOST" build "$OUT/probe.sio" --target x86_64-linux -o "$OUT/probe" >"$OUT/build.log" 2>&1
chmod +x "$OUT/probe"
timeout 180 "$OUT/probe" >"$OUT/stdout.log" 2>"$OUT/stderr.log"
grep -Fxq FLAT_ARG_RECLAIM_OK "$OUT/stdout.log"
"$HOST" build "$ROOT/tests/native-v2/box_arena_empty_mark.sio" --target x86_64-linux -o "$OUT/empty-mark" >"$OUT/empty-mark.build.log" 2>&1
chmod +x "$OUT/empty-mark"
timeout 30 "$OUT/empty-mark" >"$OUT/empty-mark.stdout.log" 2>"$OUT/empty-mark.stderr.log"
grep -Fxq BOX_ARENA_EMPTY_MARK_OK "$OUT/empty-mark.stdout.log"
"$HOST" build "$ROOT/tests/native-v2/box_arena_invalid_mark.sio" --target x86_64-linux -o "$OUT/invalid-mark" >"$OUT/invalid-mark.build.log" 2>&1
chmod +x "$OUT/invalid-mark"
set +e
timeout 30 "$OUT/invalid-mark" >"$OUT/invalid-mark.stdout.log" 2>"$OUT/invalid-mark.stderr.log"
invalid_status=$?
set -e
test "$invalid_status" = 183
echo 'PASS: 33000 flat call sites, nine arguments, Box reset/churn, unchanged pool and invalid-range refusal'
