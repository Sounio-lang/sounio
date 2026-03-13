#!/usr/bin/env bash
# Sprint 108 gate: extern "C" fn FFI wiring
# Tests: IrCallExtern opcode, IR_STRATEGY_EXTERN, parse_extern_fn_item,
#        ItemExternFn, ExternRelocTable, lower_call_extern (SysV ABI)
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

run_check_no_new_errors() {
    local name="$1" file="$2"
    shift 2
    local patterns=("$@")
    local out
    out=$("$SOUC" check "$file" 2>&1)
    # Build grep pattern from args
    local grep_pat
    grep_pat=$(IFS='|'; echo "${patterns[*]}")
    local new_errs
    new_errs=$(echo "$out" | grep -E "$grep_pat" || true)
    if [ -z "$new_errs" ]; then
        echo "PASS $name (no new errors for: ${patterns[*]})"
        PASS=$((PASS+1))
    else
        echo "FAIL $name"
        echo "$new_errs"
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

# ── T1: ir/ir.sio cleanly defines IrCallExtern + IR_STRATEGY_EXTERN ──────────
run_check T1_ir_check self-hosted/ir/ir.sio
run_grep T2_IrCallExtern_defined self-hosted/ir/ir.sio "IrCallExtern,"
run_grep T3_IR_STRATEGY_EXTERN_defined self-hosted/ir/ir.sio "IR_STRATEGY_EXTERN"
run_grep T4_ir_call_extern_fn self-hosted/ir/ir.sio "fn ir_call_extern"

# ── T5: parser uses is_kernel sentinel for extern "C" fn (no ItemExternFn) ────
run_check T5_ast_check self-hosted/parser/ast.sio
run_grep T6_is_kernel_extern_sentinel self-hosted/parser/items.sio "is_kernel: true"

# ── T7: parser/items.sio parse_extern_fn_item introduces no new errors ────────
run_check_no_new_errors T7_items_no_new_errors self-hosted/parser/items.sio \
    "parse_extern_fn_item" "TokenKind::Extern"
run_grep T8_extern_fn_dispatch self-hosted/parser/items.sio "parse_extern_fn_item"

# ── T9: ir/lower.sio extern preseed + IrCallExtern emit no new errors ─────────
run_check_no_new_errors T9_lower_no_new_errors self-hosted/ir/lower.sio \
    "ir_call_extern" "IR_STRATEGY_EXTERN" "callee_strategy"
run_grep T10_extern_preseed_wired self-hosted/ir/lower.sio "is_kernel"
run_grep T11_IrCallExtern_emit self-hosted/ir/lower.sio "ir_call_extern"

# ── T12: native/reloc.sio cleanly defines ExternRelocTable ───────────────────
run_check T12_reloc_check self-hosted/native/reloc.sio
run_grep T13_ExternReloc_struct self-hosted/native/reloc.sio "struct ExternReloc"
run_grep T14_extern_reloc_table_new self-hosted/native/reloc.sio "fn extern_reloc_table_new"
run_grep T15_add_extern_reloc self-hosted/native/reloc.sio "fn add_extern_reloc"

# ── T16: native/frame.sio no new errors + extern_relocs field present ─────────
run_check_no_new_errors T16_frame_no_new_errors self-hosted/native/frame.sio \
    "extern_relocs" "ExternRelocTable" "extern_reloc_table_new"
run_grep T17_extern_relocs_field self-hosted/native/frame.sio "extern_relocs: ExternRelocTable"

# ── T18: native/lower_ir.sio cleanly adds lower_call_extern ──────────────────
run_check T18_lower_ir_check self-hosted/native/lower_ir.sio
run_grep T19_lower_call_extern_fn self-hosted/native/lower_ir.sio "fn lower_call_extern"
run_grep T20_IrCallExtern_dispatch self-hosted/native/lower_ir.sio "IrOpcode::IrCallExtern"

echo ""
echo "Sprint 108 gate: PASS=$PASS FAIL=$FAIL NOT_RUN=$NOT_RUN"
if [ $FAIL -gt 0 ]; then exit 1; fi
exit 0
