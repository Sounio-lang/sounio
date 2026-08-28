#!/usr/bin/env bash
# bigframe_ops_gate.sh — run-proof gate for stdlib/data/bigframe_ops.sio
#
# Compiles the BigFrame scale-verbs run-proof under the lean_single engine
# (heap works only there), RUNS it (builds 1,000,000 rows, exercises
# bf_filter_gt + bf_groupby_sum against oracle-exact answers), and greps for the
# success token. Emits BIGFRAME_OPS_GATE_OK on success (exit 0) or
# BIGFRAME_OPS_GATE_FAIL (exit 1).
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || { echo "BIGFRAME_OPS_GATE_FAIL: cannot cd repo root"; exit 1; }

SOUC="./bin/souc"
SRC="tests/stdlib/data/test_bigframe_ops_stdlib.sio"
OUT="$(mktemp -d)/test_bigframe_ops"

export SOUNIO_STDLIB_PATH="$REPO_ROOT/stdlib"
export SOUNIO_SOUC_ENGINE=lean_single

if ! "$SOUC" compile "$SRC" -o "$OUT" > /dev/null 2>&1; then
    echo "BIGFRAME_OPS_GATE_FAIL: compile failed"
    exit 1
fi
chmod +x "$OUT"

RUN_OUT="$("$OUT" 2>&1)"
RC=$?
echo "$RUN_OUT"

if [ "$RC" -eq 0 ] && echo "$RUN_OUT" | grep -q "BIGFRAME_OPS_GATE_OK"; then
    echo "BIGFRAME_OPS_GATE_OK"
    exit 0
fi

echo "BIGFRAME_OPS_GATE_FAIL: token missing or nonzero exit (rc=$RC)"
exit 1
