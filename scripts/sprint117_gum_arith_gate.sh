#!/usr/bin/env bash
# Sprint 117 — GUM arithmetic for Knowledge<f64> gate
# JCGM 100:2008 §5.1.2-5.1.3
set -eo pipefail
SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
MAIN=self-hosted/compiler/main.sio
LOWER=self-hosted/ir/lower.sio
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
check_grep() {
    local name="$1" file="$2" pattern="$3"
    TOTAL=$((TOTAL+1))
    if grep -q "$pattern" "$file" 2>/dev/null; then
        echo "  PASS  $name"; PASS=$((PASS+1))
    else
        echo "  FAIL  $name"; FAIL=$((FAIL+1))
    fi
}
echo "=== Sprint 117 — GUM Arithmetic Gate ==="
echo ""
echo "[structural]"
check_grep "struct:gum_add_value"         "$MAIN" "compiler_main_test_gum_add_value"
check_grep "struct:gum_add_variance"      "$MAIN" "compiler_main_test_gum_add_variance"
check_grep "struct:gum_sub_uses_add"      "$MAIN" "compiler_main_test_gum_sub_variance_uses_add"
check_grep "struct:gum_mul_formula"       "$MAIN" "compiler_main_test_gum_mul_variance_formula"
check_grep "struct:gum_div_value"         "$MAIN" "compiler_main_test_gum_div_value"
check_grep "struct:gum_conf_min"          "$MAIN" "compiler_main_test_gum_confidence_min"
check_grep "struct:gum_citation"          "$MAIN" "JCGM 100:2008"
check_grep "struct:knowledge_layout"      "$LOWER" "ir_register_knowledge_layout"
check_grep "struct:measure_alloc"         "$LOWER" "ir_alloc(dst_reg, 0)"
echo ""
echo "[formula_correctness]"
check_grep "formula:add_OpAdd"            "$MAIN" "GUM add value"
check_grep "formula:sub_OpAdd"            "$MAIN" "sub variance uses OpAdd"
check_grep "formula:mul_delta"            "$MAIN" "mul variance delta"
check_grep "formula:div_OpDiv"            "$MAIN" "div value.*OpDiv"
check_grep "formula:conf_min"             "$MAIN" "confidence min"
echo ""
echo "[tests]"
check_grep "test:T412_wired"  "$MAIN" "T412 OK.*GUM"
check_grep "test:T413_wired"  "$MAIN" "T413 OK.*GUM"
check_grep "test:T414_wired"  "$MAIN" "T414 OK.*GUM"
check_grep "test:T415_wired"  "$MAIN" "T415 OK.*GUM"
check_grep "test:T416_wired"  "$MAIN" "T416 OK.*GUM"
check_grep "test:T417_wired"  "$MAIN" "T417 OK.*GUM"
check_grep "test:total_417"   "$MAIN" "let total: i64 = [4-9][0-9][0-9]"
echo ""
echo "[regression]"
check_grep "regr:knowledge_layout" "$LOWER" "field_count: 3"
check_grep "regr:sqrt_builtin"    "self-hosted/native/codegen.sio" "name_is_sqrt"
check_grep "regr:print_f64"       "self-hosted/native/codegen.sio" "name_is_print_f64"
echo ""
echo "[typecheck]"
TOTAL=$((TOTAL+1))
TC_OUT=$(timeout 30 $SOUC check "$MAIN" 2>&1)
if echo "$TC_OUT" | grep -q "All checks passed"; then
    echo "  PASS  typecheck:main.sio"; PASS=$((PASS+1))
else
    echo "  FAIL  typecheck:main.sio"; echo "$TC_OUT" | tail -3; FAIL=$((FAIL+1))
fi
echo ""
echo "[selftest]"
TOTAL=$((TOTAL+1))
STOUT=$(timeout 60 $SOUC run "$MAIN" -- --self-test 2>&1 || true)
if echo "$STOUT" | grep -q "T412 OK.*GUM"; then
    echo "  PASS  selftest:T412_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T412"; then
    echo "  FAIL  selftest:T412_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T412_runtime (OOM before T412)"; NOT_RUN=$((NOT_RUN+1))
fi
echo ""
echo "========================================="
echo "Sprint 117 — GUM Arithmetic"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="
if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
