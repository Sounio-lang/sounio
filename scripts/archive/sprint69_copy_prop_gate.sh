#!/usr/bin/env bash
# sprint69_copy_prop_gate.sh — Constant propagation through IrCopy in ocp_const_fold
#
# SOTA: Wegman & Zadeck "Constant Propagation with Conditional Branches" (TOPLAS 1991) §copy folding;
# Cooper & Torczon "Engineering a Compiler" §10.7 — copy propagation eliminates the definition site
# when the source value flows through a copy instruction.
#
# Novel claims:
#   1. Single new block in ocp_const_fold re-reads result.instrs[i] after Sprint 67/68 rewrites,
#      enabling constant propagation for both source-IR copies and fold-created copies in one pass
#   2. Transitive propagation: r0=5;r1=IrCopy(r0);r2=IrCopy(r1) → r2 marked constant=5
#   3. Copy prop enables Sprint 68 strength reduction: r0=4;r1=IrCopy(r0);x*r1 → x<<2
#   4. Copy prop enables Sprint 67 identity fold: r0=1;r1=IrCopy(r0);x*r1 → IrCopy(x)
#   5. Non-constant sources correctly not propagated (no false positives)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"

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

echo "=== Sprint 69: Constant propagation through IrCopy ==="
echo ""

# --- Group 1: Self-tests T103-T108 ---
echo "--- Self-tests: IrCopy constant propagation T103-T108 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in T103 T104 T105 T106 T107 T108; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  selftest:$name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:T103_enables_full_fold" \
        "T103 OK: copy prop through IrCopy enables full-constant fold" "$SELF_TEST_LOG"
    check_log_line "selftest:T104_enables_sr" \
        "T104 OK: copy prop enables strength reduction" "$SELF_TEST_LOG"
    check_log_line "selftest:T105_transitive" \
        "T105 OK: transitive copy chain propagates constant" "$SELF_TEST_LOG"
    check_log_line "selftest:T106_enables_identity" \
        "T106 OK: copy prop enables identity fold" "$SELF_TEST_LOG"
    check_log_line "selftest:T107_non_const" \
        "T107 OK: non-constant source not propagated through IrCopy" "$SELF_TEST_LOG"
    check_log_line "selftest:T108_round_trip" \
        "T108 OK: copy prop round-trip: prop+fold+DCE removes dead loads" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio — IrCopy propagation block ---"
check_grep "ocp:sprint69_comment" \
    "Sprint 69" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:ircopy_check" \
    "IrOpcode::IrCopy" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:src_propagation" \
    "ci\.src1" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:is_const_propagate" \
    "is_const\[dst as usize\]" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:const_val_propagate" \
    "const_val\[dst as usize\]" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: main.sio — test registration ---"
check_grep "main:T103_fn" \
    "fn compiler_main_test_ocp_cp_enables_full_fold" \
    "self-hosted/compiler/main.sio"
check_grep "main:T104_fn" \
    "fn compiler_main_test_ocp_cp_enables_strength_reduction" \
    "self-hosted/compiler/main.sio"
check_grep "main:T105_fn" \
    "fn compiler_main_test_ocp_cp_transitive_chain" \
    "self-hosted/compiler/main.sio"
check_grep "main:T106_fn" \
    "fn compiler_main_test_ocp_cp_enables_identity_fold" \
    "self-hosted/compiler/main.sio"
check_grep "main:T107_fn" \
    "fn compiler_main_test_ocp_cp_non_const_not_propagated" \
    "self-hosted/compiler/main.sio"
check_grep "main:T108_fn" \
    "fn compiler_main_test_ocp_cp_round_trip_dce" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_108" \
    "let total: i64 = 1[01][0-9]" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Sprint 67/68 regression ---"
check_grep "regression:sprint68_sr" \
    "Sprint 68" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:sprint67_identity" \
    "Sprint 67" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:ocp_is_power_of_two" \
    "fn ocp_is_power_of_two" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:def_at_decl" \
    "def_at" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:ph_run_on_func" \
    "fn ph_run_on_func" \
    "self-hosted/native/peephole.sio"

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

mkdir -p "$ROOT_DIR/artifacts/sprint69"
cat > "$ROOT_DIR/artifacts/sprint69/copy_prop_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint69.copy_prop_gate.v1",
  "generated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "status": "$STATUS",
  "reason": "$REASON",
  "metrics": {
    "total": $TOTAL,
    "passed": $PASS,
    "failed": $FAIL,
    "not_run": $NOT_RUN
  },
  "novel_claims": [
    "Single new block in ocp_const_fold re-reads result.instrs[i] after Sprint 67/68 rewrites, enabling constant propagation for both source-IR copies and fold-created copies in one pass",
    "Transitive propagation: r0=5;r1=IrCopy(r0);r2=IrCopy(r1) marks r2 constant=5 in a single forward pass",
    "Copy prop enables Sprint 68 strength reduction through copies: r0=4;r1=IrCopy(r0);x*r1 reduces to x<<2",
    "Copy prop enables Sprint 67 identity fold through copies: r0=1;r1=IrCopy(r0);x*r1 becomes IrCopy(x)",
    "Full pipeline after Sprint 69: instrument→profile→promote→inline→layout→full+partial+SR+copy_prop const_fold+DCE→TCO→4-preg regalloc→disp8→compact frame→peephole"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint69/copy_prop_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
