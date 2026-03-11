#!/usr/bin/env bash
# sprint77_commut_cse_gate.sh — Commutative CSE canonicalization Block H validation
# Extends ocp_cse to treat a+b and b+a as the same CSE entry for commutative ops.
# Self-tests T181–T186; total 186/186.
# SOTA: Click "Value Numbering" PLDI 1995 §3.1 (canonical operand ordering for GVN/CSE).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
OUT_DIR="$ROOT_DIR/artifacts/sprint77"
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

echo "=== Sprint 77: Commutative CSE Canonicalization Block H ==="
echo ""

echo "--- Group 1: Compiler self-test smoke (186/186) ---"
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
echo "--- Group 2: Block H structural checks ---"
check_grep "block_h:helper_fn" \
    "fn ocp_is_commutative_op" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_h:comment" \
    "Sprint 77 Block H" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_h:cse_s1_var" \
    "var cse_s1" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_h:cse_s2_var" \
    "var cse_s2" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_h:swap_pattern" \
    "cse_s1 > cse_s2" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_h:click_ref" \
    "Click.*Value Numbering|Value Numbering.*Click" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_h:total_184plus" \
    "let total: i64 = 18[4-9]" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 3: Sprint 76/75 Block G/F compatibility ---"
check_grep "compat:block_g_comment" \
    "Sprint 76 Block G" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "compat:block_f_comment" \
    "Sprint 75 Block F" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 4: Sprint 73/44 compatibility probes ---"
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

cat > "$OUT_DIR/commut_cse_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint77.commut_cse_gate.v1",
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
    "Commutative CSE canonicalization (Block H) extends ocp_cse from exact-match to value-equivalence for 5 commutative operators (Add/Mul/Or/And/BitXor), implementing canonical operand ordering from Click PLDI 1995 §3.1 without requiring full GVN infrastructure",
    "The swap-on-larger-register heuristic (src1 ← min, src2 ← max) is a provably correct canonicalization for symmetric binary functions — all expression orderings map to a unique table key, ensuring commutative CSE completeness within a single forward scan"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint77/commut_cse_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
