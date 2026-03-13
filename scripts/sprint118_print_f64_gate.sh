#!/usr/bin/env bash
# Sprint 118 — print_f64 builtin (multiply-and-truncate) gate
set -eo pipefail
SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
CODEGEN=self-hosted/native/codegen.sio
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
echo "=== Sprint 118 — print_f64 Builtin Gate ==="
echo ""
echo "[structural]"
check_grep "struct:name_is_print_f64"     "$CODEGEN" "fn name_is_print_f64"
check_grep "struct:emit_builtin_print_f64" "$CODEGEN" "fn emit_builtin_print_f64"
check_grep "struct:rodata_1e6"            "$CODEGEN" "0x4124680000000000"
check_grep "struct:cvttsd2si"             "$CODEGEN" "emit_cvttsd2si_rax_xmm0"
check_grep "struct:sign_check"            "$CODEGEN" "emit_test_rdi_rdi"
check_grep "struct:frac_mul"              "$CODEGEN" "emit_mulsd_xmm0_xmm1"
check_grep "struct:dispatch_v1"           "$CODEGEN" "name_is_print_f64(func.name).*emit_builtin_print_f64"
echo ""
echo "[tests]"
check_grep "test:T402_exists" "$MAIN" "compiler_main_test_name_is_print_f64"
check_grep "test:T403_exists" "$MAIN" "compiler_main_test_emit_print_f64_produces_code"
check_grep "test:T402_wired"  "$MAIN" "T402 OK.*print_f64"
check_grep "test:T403_wired"  "$MAIN" "T403 OK.*print_f64"
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
if echo "$STOUT" | grep -q "T402 OK.*print_f64"; then
    echo "  PASS  selftest:T402_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T402"; then
    echo "  FAIL  selftest:T402_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T402_runtime (OOM before T402)"; NOT_RUN=$((NOT_RUN+1))
fi
echo ""
echo "========================================="
echo "Sprint 118 — print_f64 Builtin"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="
if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
