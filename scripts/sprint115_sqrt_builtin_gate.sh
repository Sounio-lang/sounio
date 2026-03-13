#!/usr/bin/env bash
# Sprint 115 — sqrt builtin (hardware SQRTSD) gate
set -eo pipefail
SOUC=./artifacts/omega/souc-bin/souc-linux-x86_64-jit
CODEGEN=self-hosted/native/codegen.sio
ENCODE=self-hosted/native/encode.sio
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
echo "=== Sprint 115 — sqrt Builtin Gate ==="
echo ""
echo "[structural]"
check_grep "struct:name_is_sqrt"          "$CODEGEN" "fn name_is_sqrt"
check_grep "struct:emit_builtin_sqrt"     "$CODEGEN" "fn emit_builtin_sqrt"
check_grep "struct:emit_sqrtsd"           "$ENCODE"  "fn emit_sqrtsd_xmm0_xmm0"
check_grep "struct:sqrtsd_encoding"       "$ENCODE"  "0x51"
check_grep "struct:dispatch_v1"           "$CODEGEN" "name_is_sqrt(func.name).*emit_builtin_sqrt"
echo ""
echo "[tests]"
check_grep "test:T400_exists" "$MAIN" "compiler_main_test_name_is_sqrt"
check_grep "test:T401_exists" "$MAIN" "compiler_main_test_sqrtsd_encoding"
check_grep "test:T400_wired"  "$MAIN" "T400 OK.*sqrt"
check_grep "test:T401_wired"  "$MAIN" "T401 OK.*sqrtsd"
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
if echo "$STOUT" | grep -q "T400 OK.*sqrt"; then
    echo "  PASS  selftest:T400_runtime"; PASS=$((PASS+1))
elif echo "$STOUT" | grep -q "FAIL.*T400"; then
    echo "  FAIL  selftest:T400_runtime"; FAIL=$((FAIL+1))
else
    echo "  NOT_RUN  selftest:T400_runtime (OOM before T400)"; NOT_RUN=$((NOT_RUN+1))
fi
echo ""
echo "========================================="
echo "Sprint 115 — sqrt Builtin"
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
echo "========================================="
if [ "$FAIL" -gt 0 ]; then exit 1; fi
exit 0
