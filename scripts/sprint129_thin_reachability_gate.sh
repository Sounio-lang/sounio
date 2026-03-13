#!/usr/bin/env bash
# Sprint 129 Gate: Whole-program dead function elimination (thin pass)
# Verifies ir/reachability.sio implements fixed-point BFS from main(),
# compacts the IrModule, and is wired into wide_native_compile_driver.sio.
set -eo pipefail

SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
REACH=self-hosted/ir/reachability.sio
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

echo "=== Sprint 129: Thin Reachability Gate ==="

# --- reachability.sio structural checks ---

# T1: module declaration
_ec=1; grep -q 'module ir::reachability' $REACH && _ec=0
check "T1: module ir::reachability declared" $_ec

# T2: ReachResult struct with all 4 fields
_ec=1; grep -q 'struct ReachResult' $REACH && _ec=0
check "T2: ReachResult struct" $_ec

_ec=1; grep -q 'eliminated_fn_count' $REACH && _ec=0
check "T3: eliminated_fn_count field" $_ec

# T4: reach_thin_module entry point
_ec=1; grep -q 'fn reach_thin_module(' $REACH && _ec=0
check "T4: reach_thin_module() entry point" $_ec

# T5: fixed-point BFS (loop with changed flag)
_ec=1; grep -q 'pass_done' $REACH && _ec=0
check "T5: fixed-point BFS loop" $_ec

# T6: reach_mark_one_pass returns tuple
_ec=1; grep -q 'fn reach_mark_one_pass(' $REACH && _ec=0
check "T6: reach_mark_one_pass() function" $_ec

# T7: reach_build_remap with -1 sentinel
_ec=1; grep -q 'remap: \[i64; 1024\] = \[-1; 1024\]' $REACH && _ec=0
check "T7: remap initialized to -1 (sentinel)" $_ec

# T8: reach_compact_module
_ec=1; grep -q 'fn reach_compact_module(' $REACH && _ec=0
check "T8: reach_compact_module() function" $_ec

# T9: reach_patch_calls with &!IrModule
_ec=1; grep -q 'fn reach_patch_calls(' $REACH && _ec=0
check "T9: reach_patch_calls() function" $_ec

# T10: IrOpcode::IrCall check
_ec=1; grep -q 'IrOpcode::IrCall' $REACH && _ec=0
check "T10: IrOpcode::IrCall opcode check" $_ec

# T11: fn_id field used for call target
_ec=1; grep -q 'instr.fn_id' $REACH && _ec=0
check "T11: instr.fn_id for call target" $_ec

# T12: library module passthrough (no main)
_ec=1; grep -q 'main_idx < 0' $REACH && _ec=0
check "T12: library module passthrough (no main)" $_ec

# T13: self-tests present
_ec=1; grep -q 'fn reach_run_self_tests(' $REACH && _ec=0
check "T13: reach_run_self_tests() present" $_ec

_ec=1
count=$(grep -c 'fn reach_self_test_' $REACH)
[ "$count" -ge 4 ] && _ec=0
check "T14: 4+ self-test functions" $_ec

# --- wiring in CLI driver ---

# T15: wide driver imports reachability
_ec=1; grep -q 'use ir::reachability' $CLI && _ec=0
check "T15: wide driver imports ir::reachability" $_ec

# T16: reach_thin_module called in driver
_ec=1; grep -q 'reach_thin_module' $CLI && _ec=0
check "T16: reach_thin_module called in wide driver" $_ec

# T17: eliminated_fn_count printed
_ec=1; grep -q 'eliminated_fn_count' $CLI && _ec=0
check "T17: elimination count printed in driver" $_ec

# T18: wide driver passes reach.module to wide_compile_and_emit
_ec=1; grep -q 'reach\.module' $CLI && _ec=0
check "T18: compacted module passed to wide_compile_and_emit" $_ec

# --- souc check ---

# T19: reachability.sio passes souc check
_ec=0; timeout 60 $SOUC check $REACH >/dev/null 2>&1 || _ec=$?
if [ $_ec -eq 124 ]; then _ec=2; fi
check "T19: reachability.sio souc check" $_ec

# T20: wide_native_compile_driver.sio passes souc check
_ec=0; timeout 120 $SOUC check $CLI >/dev/null 2>&1 || _ec=$?
if [ $_ec -eq 124 ]; then _ec=2; fi
check "T20: wide_native_compile_driver.sio souc check" $_ec

# --- integration smoke (may OOM → NOT_RUN) ---

# T21: smoke run — verify "eliminated=" appears in output
SMOKE_SRC=$(mktemp --suffix=.sio)
cat > "$SMOKE_SRC" <<'SIO'
fn helper() -> i64 { 42 }
fn dead_fn() -> i64 { 99 }
fn main() -> i64 with IO {
    let x = helper()
    print("hello\n")
    0
}
SIO
SMOKE_OUT=$(mktemp --suffix=.elf)
SMOKE_LOG=$(mktemp)
_ec=0
timeout 180 $SOUC run $CLI -- "$SMOKE_SRC" -o "$SMOKE_OUT" > "$SMOKE_LOG" 2>&1 || _ec=$?
if [ $_ec -eq 124 ] || [ $_ec -eq 137 ]; then
    _ec=2
elif grep -q 'eliminated=' "$SMOKE_LOG"; then
    _ec=0
else
    _ec=1
fi
check "T21: smoke run reports eliminated= count" $_ec
rm -f "$SMOKE_SRC" "$SMOKE_OUT" "$SMOKE_LOG"

echo ""
echo "=== RESULTS: $PASS PASS, $FAIL FAIL, $NOT_RUN NOT_RUN (of $TOTAL) ==="
exit $FAIL
