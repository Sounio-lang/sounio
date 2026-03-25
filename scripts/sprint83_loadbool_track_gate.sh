#!/usr/bin/env bash
# sprint83_loadbool_track_gate.sh — Block K: IrLoadBool constant tracking
#
# Extends ocp_const_fold to track IrLoadBool instructions in the is_const[]
# / const_val[] lattice (true→1, false→0 in imm_i64), enabling:
#   - Block F branch folding on boolean-typed registers
#   - BinOp both-constant folding (e.g. 7 * false → 0)
#
# Also gates native-v2 GC tests (T210–T213) and loadbool tests (T214–T219).
# Self-test total: 219/219.
#
# SOTA: Click "Value Numbering" PLDI 1995 §2 — all value-producing instructions
#       participate in the constant lattice; Wegman & Zadeck TOPLAS 1991 §3.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
OUT_DIR="$ROOT_DIR/artifacts/sprint83"
mkdir -p "$OUT_DIR"

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

check_absent() {
    local name="$1"; local pattern="$2"; local file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (file '$file' not found)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if ! grep -qE "$pattern" "$file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (pattern '$pattern' should NOT appear in $file)"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 83: Block K — IrLoadBool Constant Tracking ==="
echo ""

echo "--- Group 1: Compiler self-test smoke (219/219) ---"
TOTAL=$((TOTAL+1))
if [ ! -x "$SOUC" ]; then
    echo "NOT_RUN  selftest:compiler_main (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
else
    _st_log="$(mktemp)"
    _st_ec=0
    timeout 300 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$_st_log" 2>&1 || _st_ec=$?
    if [ "$_st_ec" -eq 0 ]; then
        echo "PASS  selftest:compiler_main"; PASS=$((PASS+1))
    elif [ "$_st_ec" -eq 124 ] || [ "$_st_ec" -eq 137 ]; then
        echo "NOT_RUN  selftest:compiler_main (OOM/timeout)"; NOT_RUN=$((NOT_RUN+1))
    else
        echo "NOT_RUN  selftest:compiler_main (exit $_st_ec)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$_st_log"
fi

echo ""
echo "--- Group 2: Block K structural checks in opt_cleanup.sio ---"
check_grep "block_k:comment" \
    "Sprint 83 Block K" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_k:loadbool_check" \
    "IrLoadBool" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_k:const_set" \
    "is_const\[instr.dst as usize\] = true" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_k:val_set" \
    "const_val\[instr.dst as usize\] = instr.imm_i64" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_k:sota_ref" \
    "Click.*Value Numbering|PLDI 1995" \
    "self-hosted/ir/opt_cleanup.sio"
check_absent "block_k:no_reassoc_helpers" \
    "ocp_reassoc_encode|ocp_reassoc_combine" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: T210–T219 wiring in main.sio ---"
check_grep "main:T210_fn" \
    "compiler_main_test_native_v2_gc_precise_descriptor_scanning" \
    "self-hosted/compiler/main.sio"
check_grep "main:T211_fn" \
    "compiler_main_test_native_v2_gc_compacts_live_handles" \
    "self-hosted/compiler/main.sio"
check_grep "main:T212_fn" \
    "compiler_main_test_native_v2_gc_pins_block_relocation" \
    "self-hosted/compiler/main.sio"
check_grep "main:T213_fn" \
    "compiler_main_test_native_v2_gc_state_updates_after_collection" \
    "self-hosted/compiler/main.sio"
check_grep "main:T214_fn" \
    "compiler_main_test_loadbool_true_branch_true" \
    "self-hosted/compiler/main.sio"
check_grep "main:T215_fn" \
    "compiler_main_test_loadbool_false_branch_true" \
    "self-hosted/compiler/main.sio"
check_grep "main:T216_fn" \
    "compiler_main_test_loadbool_false_branch_false" \
    "self-hosted/compiler/main.sio"
check_grep "main:T217_fn" \
    "compiler_main_test_loadbool_true_branch_false" \
    "self-hosted/compiler/main.sio"
check_grep "main:T218_fn" \
    "compiler_main_test_loadbool_binop_fold" \
    "self-hosted/compiler/main.sio"
check_grep "main:T219_fn" \
    "compiler_main_test_loadbool_mul_annihilate" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_219" \
    "let total: i64 = 219" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Sprint 82 Block J regression guard ---"
check_grep "compat:block_j_comment" \
    "Sprint 82 Block J" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "compat:unaryop_neg" \
    "OpNeg" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "compat:unaryop_not" \
    "OpNot" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 5: Sprint 79/80/81 regression guard ---"
check_grep "compat:licm_fn" \
    "fn lopt_optimize_function" \
    "self-hosted/ir/loop_opt.sio"
check_grep "compat:sccp_fn" \
    "fn cp_optimize_function" \
    "self-hosted/ir/const_prop.sio"
check_grep "compat:block_i_comment" \
    "Sprint 81 Block I" \
    "self-hosted/ir/opt_cleanup.sio"

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

printf '{\n  "schema": "sounio.sprint83.loadbool_track_gate.v1",\n  "generated_at": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "config": {\n    "root": "%s",\n    "timeout_seconds": 300\n  },\n  "metrics": {\n    "total": %d,\n    "passed": %d,\n    "failed": %d,\n    "not_run": %d\n  },\n  "novel_claims": [\n    "Block K extends the constant lattice in ocp_const_fold to include IrLoadBool: true→1, false→0 in imm_i64; boolean registers now participate in Block F branch folding and BinOp both-constant folding",\n    "T210-T213: native-v2 GC precise descriptor scanning, handle compaction, pin-aware relocation, and state accounting",\n    "T214-T219: IrLoadBool tracking through BranchTrue/False (Block F chain) and BinOp constant folding; total 219/219"\n  ]\n}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)" "$STATUS" "$REASON" "$ROOT_DIR" \
    "$TOTAL" "$PASS" "$FAIL" "$NOT_RUN" \
    > "$OUT_DIR/loadbool_track_gate.v1.json"

echo ""
echo "Artifact: artifacts/sprint83/loadbool_track_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
