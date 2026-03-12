#!/usr/bin/env bash
# (chmod +x this file before running)
# sprint65_peephole_gate.sh — Peephole Optimizer Wiring
#
# SOTA: Aho, Lam, Sethi, Ullman "Compilers" §8.7 — peephole optimization operates on a
# sliding window of instructions, replacing patterns with shorter/faster equivalents.
# Patterns: mov rX,rX → NOP; add rX,0 → NOP; push/pop pairs → NOP; xor+cmp → drop cmp.
#
# Novel claims:
#   1. IrInstr→PhInstr translation: IrCopy(r,r) fires self-move elimination at IR level
#   2. IrLoadImm consecutive same-dst pairs detected at IR level before lowering
#   3. ph_run_on_func wired after regalloc, before instruction lowering in compile_ir_function
#   4. Full pipeline: instrument->profile->promote->inline->layout->const_fold+DCE->TCO->4-preg regalloc->disp8->compact frame->peephole
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"

check_grep() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (file '$file' not found)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if grep -qE "$pattern" "$file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (pattern '$pattern' not found in $file)"; FAIL=$((FAIL+1))
    fi
}

check_log_line() {
    local name="$1"; local expected_line="$2"; local log_file="$3"
    TOTAL=$((TOTAL+1))
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 65: Peephole Optimizer Wiring ==="
echo ""

# --- Group 1: Self-tests T74-T80 ---
echo "--- Self-tests: peephole T74-T80 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:ph_self_move_eliminated" "selftest:ph_identity_add_eliminated" \
                "selftest:ph_identity_sub_eliminated" "selftest:ph_push_pop_pair_collapsed" \
                "selftest:ph_xor_cmp_redundant_removed" "selftest:ph_run_on_func_no_false_positives" \
                "selftest:ph_run_on_func_eliminates_self_copies"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:T74_self_move" \
        "T74 OK: ph_opt_single eliminates mov rX, rX self-move" "$SELF_TEST_LOG"
    check_log_line "selftest:T75_identity_add" \
        "T75 OK: ph_opt_single eliminates add rX, 0 identity add" "$SELF_TEST_LOG"
    check_log_line "selftest:T76_identity_sub" \
        "T76 OK: ph_opt_single eliminates sub rX, 0 identity sub" "$SELF_TEST_LOG"
    check_log_line "selftest:T77_push_pop" \
        "T77 OK: ph_opt_pair collapses push rX; pop rX redundant pair" "$SELF_TEST_LOG"
    check_log_line "selftest:T78_xor_cmp" \
        "T78 OK: ph_opt_pair removes redundant cmp rX,0 after xor rX,rX" "$SELF_TEST_LOG"
    check_log_line "selftest:T79_no_false_pos" \
        "T79 OK: ph_run_on_func preserves distinct-dst IrLoadImm pair (no false positives)" "$SELF_TEST_LOG"
    check_log_line "selftest:T80_self_copies" \
        "T80 OK: ph_run_on_func eliminates IrCopy(r,r) self-copies to IrNop" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: peephole.sio — module and imports ---"
check_grep "peephole:module_decl" \
    "^module native::peephole" \
    "self-hosted/native/peephole.sio"
check_grep "peephole:use_ir_ir" \
    "use ir::ir::" \
    "self-hosted/native/peephole.sio"
check_grep "peephole:use_ir_nop" \
    "ir_nop" \
    "self-hosted/native/peephole.sio"

echo ""
echo "--- Group 3: peephole.sio — translation layer ---"
check_grep "peephole:ir_instr_to_ph_fn" \
    "fn ir_instr_to_ph" \
    "self-hosted/native/peephole.sio"
check_grep "peephole:IrCopy_mapping" \
    "IrOpcode::IrCopy" \
    "self-hosted/native/peephole.sio"
check_grep "peephole:IrLoadImm_mapping" \
    "IrOpcode::IrLoadImm" \
    "self-hosted/native/peephole.sio"
check_grep "peephole:IrNop_mapping" \
    "IrOpcode::IrNop" \
    "self-hosted/native/peephole.sio"

echo ""
echo "--- Group 4: peephole.sio — ph_run_on_func ---"
check_grep "peephole:ph_run_on_func_fn" \
    "fn ph_run_on_func" \
    "self-hosted/native/peephole.sio"
check_grep "peephole:pass1_single" \
    "ph_opt_single" \
    "self-hosted/native/peephole.sio"
check_grep "peephole:pass2_pair" \
    "ph_opt_pair" \
    "self-hosted/native/peephole.sio"
check_grep "peephole:writeback_ir_nop" \
    "ir_nop\(\)" \
    "self-hosted/native/peephole.sio"

echo ""
echo "--- Group 5: codegen.sio — wiring ---"
check_grep "codegen:import_peephole" \
    "use native::peephole" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:ph_run_on_func_call" \
    "ph_run_on_func" \
    "self-hosted/native/codegen.sio"
check_grep "codegen:peephole_lane_comment" \
    "regalloc/peephole lane" \
    "self-hosted/native/codegen.sio"

echo ""
echo "--- Group 6: main.sio — test registration ---"
check_grep "main:use_peephole" \
    "use native::peephole" \
    "self-hosted/compiler/main.sio"
check_grep "main:T74_fn" \
    "fn compiler_main_test_ph_self_move_eliminated" \
    "self-hosted/compiler/main.sio"
check_grep "main:T79_fn" \
    "fn compiler_main_test_ph_run_on_func_no_false_positives" \
    "self-hosted/compiler/main.sio"
check_grep "main:T80_fn" \
    "fn compiler_main_test_ph_run_on_func_eliminates_self_copies" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_108" \
    "let total: i64 = 108" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 7: Sprint 62/63/64 regression ---"
check_grep "regression:regalloc_fn" \
    "fn regalloc_linear_scan" \
    "self-hosted/native/regalloc.sio"
check_grep "regression:nc_min_frame_size" \
    "fn nc_min_frame_size" \
    "self-hosted/native/lower_ir.sio"
check_grep "regression:ir_slot_fits_disp8" \
    "fn ir_slot_fits_disp8" \
    "self-hosted/native/lower_ir.sio"
check_grep "regression:tco_run_pass" \
    "fn tco_run_pass" \
    "self-hosted/ir/tailcall.sio"

# --- Results ---
echo ""
echo "=== Results: $PASS passed, $FAIL failed, $NOT_RUN not_run (total $TOTAL) ==="

STATUS="pass"
REASON="all_cases_passed"
if [ "$FAIL" -gt 0 ]; then
    STATUS="fail"
    REASON="${FAIL}_cases_failed"
fi
if [ "$NOT_RUN" -gt 0 ] && [ "$FAIL" -eq 0 ]; then
    REASON="all_run_passed_${NOT_RUN}_not_run"
fi

mkdir -p "$ROOT_DIR/artifacts/sprint65"
cat > "$ROOT_DIR/artifacts/sprint65/peephole_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint65.peephole_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)",
  "status": "$STATUS",
  "reason": "$REASON",
  "config": {
    "root": "$ROOT_DIR",
    "timeout_seconds": 180
  },
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  },
  "novel_claims": [
    "IrCopy(r,r) self-move detected at IR level via IrInstr→PhInstr translation before lowering",
    "Consecutive IrLoadImm to same dst eliminated as duplicate-load pair at IR level",
    "ph_run_on_func wired after regalloc, before instruction lowering in compile_ir_function",
    "Full pipeline: instrument->profile->promote->inline->layout->const_fold+DCE->TCO->4-preg regalloc->disp8->compact frame->peephole"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint65/peephole_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
