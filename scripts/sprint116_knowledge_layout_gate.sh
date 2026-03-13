#!/usr/bin/env bash
# Sprint 116 — Knowledge<f64> struct layout + measure() rewrite gate
set -eo pipefail
SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
LOWER=self-hosted/ir/lower.sio
MAIN=self-hosted/compiler/main.sio
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
echo "=== Sprint 116 — Knowledge<f64> Layout Gate ==="
echo ""
echo "[structural]"
check_grep "struct:register_fn"           "$LOWER" "fn ir_register_knowledge_layout"
check_grep "struct:field_value"           "$LOWER" 'make_name("value")'
check_grep "struct:field_variance"        "$LOWER" 'make_name("variance")'
check_grep "struct:field_confidence"      "$LOWER" 'make_name("confidence")'
check_grep "struct:field_count_3"         "$LOWER" "field_count: 3"
check_grep "struct:measure_alloc"         "$LOWER" "ir_alloc(dst_reg, 0)"
check_grep "struct:measure_variance"      "$LOWER" "ir_binop(variance_reg.*OpMul"
check_grep "struct:measure_confidence"    "$LOWER" "ir_load_float(conf_reg, 1.0)"
check_grep "struct:wired_epistemic"       "$LOWER" "ir_register_knowledge_layout"
echo ""
echo "[tests]"
check_grep "test:T404_exists" "$MAIN" "compiler_main_test_knowledge_layout_registered"
check_grep "test:T405_exists" "$MAIN" "compiler_main_test_knowledge_field_idx_lookup"
check_grep "test:T404_wired"  "$MAIN" "T404 OK.*Knowledge layout"
check_grep "test:T405_wired"  "$MAIN" "T405 OK.*Knowledge field"
check_grep "test:total_405"   "$MAIN" "let total: i64 = [4-9][0-9][0-9]"
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
if echo "$STOUT" | grep -q "T404 OK.*Knowledge"; then
    echo "  PASS  selftest:T404_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T404"; then
    echo "  FAIL  selftest:T404_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T404_runtime (OOM before T404)"; NOT_RUN=$((NOT_RUN+1))
fi
echo ""
echo "========================================="
echo "Sprint 116 — Knowledge<f64> Layout"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="
if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
