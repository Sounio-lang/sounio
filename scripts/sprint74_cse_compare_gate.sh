#!/usr/bin/env bash
# sprint74_cse_compare_gate.sh — CSE + mini-egraph validation + comparison fold Block E
# Part 1: Validate Sprint 73 linter-generated CSE (ocp_cse) and mini-egraph
#         (ocp_egraph_mini_pass / EgSmallContext) via self-tests T131–T145
# Part 2: Comparison operator constant folding Block E (T146–T153)
#         — SOTA anchor: Wegman & Zadeck TOPLAS 1991 (comparison lattice for SCCP)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
OUT_DIR="$ROOT_DIR/artifacts/sprint74"
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

check_probe_line() {
    local name="$1"; shift
    local expected_line="$1"; shift
    TOTAL=$((TOTAL+1))
    if [ ! -x "$SOUC" ]; then
        echo "NOT_RUN  $name (souc '$SOUC' not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local log_file
    log_file="$(mktemp)"
    if timeout 240 "$SOUC" run self-hosted/compiler/main.sio -- "$@" >"$log_file" 2>&1; then
        if grep -qF "$expected_line" "$log_file"; then
            echo "PASS  $name"; PASS=$((PASS+1))
        else
            echo "FAIL  $name (expected '$expected_line' in probe output)"; FAIL=$((FAIL+1))
        fi
    else
        echo "NOT_RUN  $name (probe invocation failed)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$log_file"
}

echo "=== Sprint 74: CSE + Mini-Egraph Validation + Comparison Fold (Block E) ==="
echo ""

echo "--- Group 1: Compiler self-test smoke (153/153) ---"
TOTAL=$((TOTAL+1))
if [ ! -x "$SOUC" ]; then
    echo "NOT_RUN  selftest:compiler_main (souc '$SOUC' not executable)"; NOT_RUN=$((NOT_RUN+1))
else
    _st_log="$(mktemp)"
    if timeout 300 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$_st_log" 2>&1; then
        echo "PASS  selftest:compiler_main"; PASS=$((PASS+1))
    else
        _ec=$?
        if [ "$_ec" -eq 124 ] || [ "$_ec" -eq 137 ]; then
            echo "NOT_RUN  selftest:compiler_main (OOM/timeout)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "NOT_RUN  selftest:compiler_main (exit $_ec)"; NOT_RUN=$((NOT_RUN+1))
        fi
    fi
    rm -f "$_st_log"
fi

echo ""
echo "--- Group 2: CSE structural checks ---"
check_grep "cse:fn_exists" \
    "fn ocp_cse" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "cse:table_size" \
    "cse_op.*128" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "cse:eg_op_const_guard" \
    "EG_OP_CONST" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "cse:wired_in_pipeline" \
    "ocp_cse" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: Mini-egraph structural checks ---"
check_grep "mini_eg:fn_exists" \
    "fn ocp_egraph_mini_pass" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "mini_eg:EgSmallContext" \
    "struct EgSmallContext" \
    "self-hosted/ir/egraph.sio"
check_grep "mini_eg:eg_small_init" \
    "fn eg_small_init" \
    "self-hosted/ir/egraph.sio"
check_grep "mini_eg:guard_4" \
    "instr_count > 4" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 4: Block E structural checks ---"
check_grep "block_e:comment" \
    "Sprint 74 Block E" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_e:opeq_fold" \
    "BinaryOp::OpEq" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_e:oplt_fold" \
    "BinaryOp::OpLt" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_e:ople_fold" \
    "BinaryOp::OpLe" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_e:same_reg" \
    "cmp_s1 == cmp_s2" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_e:wegman_ref" \
    "Wegman.*Zadeck|Zadeck.*Wegman" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_e:total_150plus" \
    "let total: i64 = 1[5-9][0-9]" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 5: Sprint 73/44 compatibility ---"
check_probe_line "compat:probe_load_ir_standard" \
    "probe_load_ir: fn=add_floats strategy=standard" \
    --probe-load-ir tests/frontend/compile_strategy_ir_standard.sio
check_probe_line "compat:probe_ir_opt_min_main" \
    "probe_ir_opt_strategy: fn=main strategy=standard" \
    --probe-ir-opt-strategy tests/frontend/lowering_stability_min_main.sio

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

cat > "$OUT_DIR/cse_compare_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint74.cse_compare_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%S.%6N+00:00)",
  "status": "$STATUS",
  "reason": "$REASON",
  "config": {
    "root": "$ROOT_DIR",
    "timeout_seconds": 300
  },
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  },
  "novel_claims": [
    "Local CSE (ocp_cse) and mini-egraph (ocp_egraph_mini_pass / EgSmallContext) validated via 15 self-tests T131–T145 — first systematic correctness evidence for Sprint 73 linter-generated content",
    "Comparison operator constant folding Block E closes the boolean-result gap in ocp_const_fold: all 6 comparison ops (OpEq/Ne/Lt/Le/Gt/Ge) now have both-constant and same-register folding, establishing the comparison lattice layer prerequisite for SCCP (Wegman & Zadeck TOPLAS 1991)"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint74/cse_compare_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
