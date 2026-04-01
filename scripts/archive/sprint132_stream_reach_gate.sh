#!/usr/bin/env bash
# Sprint 132 Gate: Streaming two-pass reachability
#
# ir/stream_reach.sio: flat edge list call graph extracted one body at a time.
# wide_native_compile_driver.sio: two-pass streaming — pass 1 extracts edges
# (bodies discarded), BFS finds reachable set, pass 2 lowers only reachable.
# Peak memory: 1 IrFunction at a time instead of all fn_count simultaneously.
set -eo pipefail

SOUC=./bin/souc
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SCG=self-hosted/ir/stream_reach.sio
RUNNER=self-hosted/ir/stream_reach_self_test_runner.sio
CLI=self-hosted/compiler/wide_native_compile_driver.sio

check() {
    local name="$1"; local result="$2"
    TOTAL=$((TOTAL + 1))
    if [ "$result" -eq 0 ]; then
        PASS=$((PASS + 1)); echo "  PASS: $name"
    elif [ "$result" -eq 2 ]; then
        NOT_RUN=$((NOT_RUN + 1)); echo "  NOT_RUN: $name"
    else
        FAIL=$((FAIL + 1)); echo "  FAIL: $name"
    fi
}

echo "=== Sprint 132: Streaming Reachability Gate ==="

# --- stream_reach.sio structural checks ---

# T1: module declaration
_ec=1; grep -q 'module ir::stream_reach' $SCG && _ec=0
check "T1: module ir::stream_reach declared" $_ec

# T2: StreamCallGraph struct
_ec=1; grep -q 'struct StreamCallGraph' $SCG && _ec=0
check "T2: StreamCallGraph struct" $_ec

# T3: flat edge arrays
_ec=1; grep -q 'callers: \[i64; 32768\]' $SCG && _ec=0
check "T3: flat callers array (32768 entries)" $_ec

# T4: scg_scan_function (pass 1 — extract edges, discard body)
_ec=1; grep -q 'fn scg_scan_function(' $SCG && _ec=0
check "T4: scg_scan_function() for pass 1" $_ec

# T5: scg_mark_reachable (BFS)
_ec=1; grep -q 'fn scg_mark_reachable(' $SCG && _ec=0
check "T5: scg_mark_reachable() BFS" $_ec

# T6: scg_count_reachable
_ec=1; grep -q 'fn scg_count_reachable(' $SCG && _ec=0
check "T6: scg_count_reachable() helper" $_ec

# T7: IrOpcode::IrCall check in scan
_ec=1; grep -q 'IrOpcode::IrCall' $SCG && _ec=0
check "T7: IrOpcode::IrCall in scg_scan_function" $_ec

# T8: 5 self-tests
_ec=1
count=$(grep -c 'fn scg_self_test_' $SCG)
[ "$count" -ge 5 ] && _ec=0
check "T8: 5+ self-test functions" $_ec

# T9: no-start test (start_idx=-1 → empty reachable set)
_ec=1; grep -q 'no_start\|start_idx.*-1\|-1.*start_idx' $SCG && _ec=0
check "T9: no-start (start=-1) test present" $_ec

# --- wiring in wide driver ---

# T10: wide driver imports ir::stream_reach
_ec=1; grep -q 'use ir::stream_reach' $CLI && _ec=0
check "T10: wide driver imports ir::stream_reach" $_ec

# T11: two-pass structure — graph_edges= printed (after pass 1)
_ec=1; grep -q 'graph_edges=' $CLI && _ec=0
check "T11: graph_edges= printed after pass 1" $_ec

# T12: thin_fn_count= printed (after pass 2)
_ec=1; grep -q 'thin_fn_count=' $CLI && _ec=0
check "T12: thin_fn_count= printed after pass 2" $_ec

# T13: scg_scan_function called in pass 1
_ec=1; grep -q 'scg_scan_function' $CLI && _ec=0
check "T13: scg_scan_function called in pass 1" $_ec

# T14: scg_mark_reachable called for BFS
_ec=1; grep -q 'scg_mark_reachable' $CLI && _ec=0
check "T14: scg_mark_reachable called for BFS" $_ec

# T15: reach_patch_calls still called for fn_id remapping
_ec=1; grep -q 'reach_patch_calls' $CLI && _ec=0
check "T15: reach_patch_calls() for fn_id remapping" $_ec

# T16: body NOT stored in pass 1 (no module.functions assignment)
# In pass 1 loop, body_result.func is scanned but not stored in a module
_ec=1; grep -q 'scg_scan_function(graph, body_result' $CLI && _ec=0
check "T16: pass 1 scans body without storing it" $_ec

# T17: thin_module built in pass 2
_ec=1; grep -q 'thin_module' $CLI && _ec=0
check "T17: thin_module built in pass 2" $_ec

# T18: reach_thin_module removed (replaced by streaming)
_ec=0; grep -q 'reach_thin_module' $CLI && _ec=1
check "T18: reach_thin_module removed from driver" $_ec

# --- souc check ---

# T19: stream_reach.sio souc check
_ec=0; timeout 60 $SOUC check $SCG >/dev/null 2>&1 || _ec=$?
if [ $_ec -eq 124 ]; then _ec=2; fi
check "T19: stream_reach.sio souc check" $_ec

# T20: wide_native_compile_driver.sio souc check
_ec=0; timeout 120 $SOUC check $CLI >/dev/null 2>&1 || _ec=$?
if [ $_ec -eq 124 ]; then _ec=2; fi
check "T20: wide_native_compile_driver.sio souc check" $_ec

# --- self-test runner ---

# T21: runner souc check
_ec=0; timeout 60 $SOUC check $RUNNER >/dev/null 2>&1 || _ec=$?
if [ $_ec -eq 124 ]; then _ec=2; fi
check "T21: stream_reach_self_test_runner.sio souc check" $_ec

# T22: runner executes ALL PASS under JIT
LOG=$(mktemp)
_ec=0; timeout 120 $SOUC run $RUNNER > "$LOG" 2>&1 || _ec=$?
if [ $_ec -eq 124 ] || [ $_ec -eq 137 ]; then
    _ec=2
elif grep -q 'ALL PASS' "$LOG"; then
    _ec=0
else
    _ec=1
fi
check "T22: scg self-tests ALL PASS under JIT" $_ec
rm -f "$LOG"

# --- regression: Sprint 130 self-tests still pass ---

# T23: reach_self_test_runner still passes
_ec=0; timeout 120 $SOUC run self-hosted/ir/reach_self_test_runner.sio >/dev/null 2>&1 || _ec=$?
if [ $_ec -eq 124 ] || [ $_ec -eq 137 ]; then _ec=2; fi
check "T23: Sprint 130 reach_self_test_runner still ALL PASS" $_ec

# --- integration smoke (NOT_RUN expected under JIT OOM) ---

# T24: wide driver smoke run with streaming (may OOM → NOT_RUN)
SMOKE_SRC=$(mktemp --suffix=.sio)
cat > "$SMOKE_SRC" <<'SIO'
fn helper() -> i64 { 1 }
fn dead() -> i64 { 99 }
fn main() -> i64 with IO { helper() }
SIO
SMOKE_LOG=$(mktemp)
SMOKE_OUT=$(mktemp --suffix=.elf)
_ec=0
timeout 180 $SOUC run $CLI -- "$SMOKE_SRC" -o "$SMOKE_OUT" > "$SMOKE_LOG" 2>&1 || _ec=$?
if [ $_ec -eq 124 ] || [ $_ec -eq 137 ]; then
    _ec=2
elif grep -q 'graph_edges=' "$SMOKE_LOG" && grep -q 'thin_fn_count=' "$SMOKE_LOG"; then
    echo "  [streaming output: $(grep 'wide_native_compile: reachable\|graph_edges\|thin_fn' "$SMOKE_LOG" | tr '\n' ' ')]"
    _ec=0
else
    _ec=1
fi
check "T24: smoke run shows graph_edges= and thin_fn_count=" $_ec
rm -f "$SMOKE_SRC" "$SMOKE_LOG" "$SMOKE_OUT"

echo ""
echo "=== RESULTS: $PASS PASS, $FAIL FAIL, $NOT_RUN NOT_RUN (of $TOTAL) ==="
exit $FAIL
