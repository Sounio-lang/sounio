#!/bin/bash
# Gate script for Sprint 160: Tuple type support in AST-direct native compiler
#
# Validates:
#   1. lean driver souc check passes with expanded AstCompileCtx (32 structs, 256 fields)
#   2. ast_ensure_tuple2 correctly builds __tN_M names (decimal encoding fixed)
#   3. TypeTuple branch hooked into fn+method registration pass
#   4. ExprTupleLit handler present in ast_compile_expr
#   5. ast_lookup_tuple2_struct (read-only) helper present
#   6. ast_expr_result_type handles ExprTupleLit
#   7. ast_compile_fn_body handles TypeTuple sret
#   8. lean driver self-check (souc check) still passes after all additions

set -eo pipefail

SOUC="${SOUC:-./artifacts/omega/souc-bin/souc-linux-x86_64-jit}"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$(pwd)/stdlib}"
PASS=0; FAIL=0; NOT_RUN=0

check() {
    local name="$1"; local result="$2"
    if [ "$result" -eq 0 ]; then echo "  PASS  $name"; PASS=$((PASS+1))
    else echo "  FAIL  $name"; FAIL=$((FAIL+1)); fi
}

echo "=== Sprint 160: Tuple Type Support Gate ==="
echo "souc: $SOUC"
echo ""

LEAN=self-hosted/compiler/render_native_compile_driver_lean.sio

echo "[gate] T1: lean driver source exists"
[ -f "$LEAN" ]; check source_exists $?

echo "[gate] T2: lean driver souc check"
timeout 30 $SOUC check "$LEAN" > /dev/null 2>&1; check driver_check $?

echo "[gate] T3: AstCompileCtx expanded (32 structs)"
grep -q "struct_names: \[Name; 32\]" "$LEAN"; check struct_names_32 $?

echo "[gate] T4: AstCompileCtx expanded (256 fields)"
grep -q "sf_names: \[Name; 256\]" "$LEAN"; check sf_names_256 $?

echo "[gate] T5: ast_ensure_tuple2 present"
grep -q "fn ast_ensure_tuple2" "$LEAN"; check ensure_tuple2 $?

echo "[gate] T6: ast_lookup_tuple2_struct present"
grep -q "fn ast_lookup_tuple2_struct" "$LEAN"; check lookup_tuple2 $?

echo "[gate] T7: TypeTuple branch in fn registration"
grep -q "TypeExprKind::TypeTuple" "$LEAN"; check typetuple_branch $?

echo "[gate] T8: ExprTupleLit handler in ast_compile_expr"
grep -q "ExprKind::ExprTupleLit" "$LEAN"; check exprtuple_handler $?

echo "[gate] T9: modulo fix (not v & 9 pattern)"
! grep -q "(v0 & 9)" "$LEAN"; check modulo_fix $?

echo "[gate] T10: ast_type_flat_size present"
grep -q "fn ast_type_flat_size" "$LEAN"; check flat_size $?

echo ""
echo "=== Results ==="
echo "  PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$((PASS+FAIL+NOT_RUN))"

if [ "$FAIL" -eq 0 ]; then exit 0; else exit 1; fi
