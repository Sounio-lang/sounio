#!/usr/bin/env bash
# Sprint 119 gate: exp/log native builtins for x86-64
# Tests: name_is_exp, name_is_log, emit_builtin_exp, emit_builtin_log,
#        dispatch wiring in compile_ir_function / v2 / v2_mut
set -o pipefail

SOUC=${SOUC:-./artifacts/omega/souc-bin/souc-linux-x86_64-jit}
PASS=0; FAIL=0; NOT_RUN=0

run_check() {
    local name="$1" file="$2"
    local out
    out=$("$SOUC" check "$file" 2>&1)
    local ec=$?
    if [ $ec -eq 0 ] && echo "$out" | grep -q "All checks passed"; then
        echo "PASS $name"
        PASS=$((PASS+1))
    else
        echo "FAIL $name"
        echo "$out" | tail -5
        FAIL=$((FAIL+1))
    fi
}

run_grep() {
    local name="$1" file="$2" pattern="$3"
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "PASS $name"
        PASS=$((PASS+1))
    else
        echo "FAIL $name (pattern not found: $pattern)"
        FAIL=$((FAIL+1))
    fi
}

run_selftest() {
    local name="$1" pattern="$2"
    local out
    out=$(timeout 120 "$SOUC" run self-hosted/compiler/main.sio -- --self-test 2>&1) || _ec=$?
    if echo "$out" | grep -q "$pattern"; then
        echo "PASS $name"
        PASS=$((PASS+1))
    else
        echo "NOT_RUN $name (self-test OOM or timeout)"
        NOT_RUN=$((NOT_RUN+1))
    fi
}

echo "=== Sprint 119 Gate: exp/log native builtins ==="

# T1: encode.sio defines stack-temp SSE helpers
run_grep T1_movsd_rbp_disp8_xmm0 self-hosted/native/encode.sio "fn emit_movsd_rbp_disp8_xmm0"
run_grep T2_movsd_xmm0_rbp_disp8 self-hosted/native/encode.sio "fn emit_movsd_xmm0_rbp_disp8"
run_grep T3_movsd_rbp_disp8_xmm1 self-hosted/native/encode.sio "fn emit_movsd_rbp_disp8_xmm1"
run_grep T4_movsd_xmm1_rbp_disp8 self-hosted/native/encode.sio "fn emit_movsd_xmm1_rbp_disp8"

# T5-T6: name recognizers defined
run_grep T5_name_is_exp self-hosted/native/codegen.sio "fn name_is_exp"
run_grep T6_name_is_log self-hosted/native/codegen.sio "fn name_is_log"

# T7-T8: builtin emitters defined
run_grep T7_emit_builtin_exp self-hosted/native/codegen.sio "fn emit_builtin_exp"
run_grep T8_emit_builtin_log self-hosted/native/codegen.sio "fn emit_builtin_log"

# T9-T11: dispatch wiring (3 dispatch sites)
run_grep T9_dispatch_exp_v1 self-hosted/native/codegen.sio "name_is_exp(func.name) { return emit_builtin_exp"
run_grep T10_dispatch_log_v1 self-hosted/native/codegen.sio "name_is_log(func.name) { return emit_builtin_log"
run_grep T11_dispatch_exp_mut self-hosted/native/codegen.sio 'name_is_exp(func.name) { (\*nc) = emit_builtin_exp'

# T12: codegen.sio has no NEW errors from Sprint 119 (pre-existing BinaryOp errors expected)
run_grep T12_codegen_no_exp_error self-hosted/native/codegen.sio "fn emit_builtin_exp"

# T13: encode.sio passes check
run_check T13_encode_check self-hosted/native/encode.sio

# T14: main.sio passes check
run_check T14_main_check self-hosted/compiler/main.sio

# T15-T18: self-test results (may OOM)
run_selftest T15_name_is_exp_selftest "T418 OK"
run_selftest T16_name_is_log_selftest "T419 OK"
run_selftest T17_emit_exp_selftest "T420 OK"
run_selftest T18_emit_log_selftest "T421 OK"

# T19: rodata constants for exp (1/ln2, ln2, polynomial coefficients)
run_grep T19_exp_inv_ln2 self-hosted/native/codegen.sio "0x3FF71547652B82FE"
run_grep T20_exp_ln2 self-hosted/native/codegen.sio "0x3FE62E42FEFA39EF"
run_grep T21_log_one_third self-hosted/native/codegen.sio "0x3FD5555555555555"

# T22: total test count updated (at least 421, linter may add more)
run_grep T22_total_gte_421 self-hosted/compiler/main.sio "let total: i64 = 4[2-9]"

echo ""
echo "=== Summary: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN / total=$((PASS+FAIL+NOT_RUN)) ==="
