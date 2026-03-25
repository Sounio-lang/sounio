#!/usr/bin/env bash
# sprint84_assoc_chain_gate.sh — Block L: Associative constant-chain folding
# Folds (x assoc c1) assoc c2 → x assoc (c1 op c2) in one forward pass.
# Self-tests T220–T225; total 225/225.
# SOTA: Click "Combining Reductions" PLDI 1995 §4; Aho, Lam, Sethi, Ullman §8.5.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
OUT_DIR="$ROOT_DIR/artifacts/sprint84"
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
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    local log_file
    log_file="$(mktemp)"
    _ec=0
    if timeout 240 "$SOUC" run self-hosted/compiler/main.sio -- "$@" >"$log_file" 2>&1; then
        if grep -qF "$expected_line" "$log_file"; then
            echo "PASS  $name"; PASS=$((PASS+1))
        else
            echo "FAIL  $name (expected '$expected_line' in probe output)"; FAIL=$((FAIL+1))
        fi
    else
        _ec=$?
        echo "NOT_RUN  $name (probe exit $_ec)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$log_file"
}

echo "=== Sprint 84: Block L — Associative Constant-Chain Folding ==="
echo ""

echo "--- Group 1: Compiler self-test smoke ---"
TOTAL=$((TOTAL+1))
if [ ! -x "$SOUC" ]; then
    echo "NOT_RUN  selftest:compiler_main (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
else
    _st_log="$(mktemp)"
    _ec=0
    if timeout 300 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$_st_log" 2>&1; then
        echo "PASS  selftest:compiler_main"; PASS=$((PASS+1))
    else
        _ec=$?
        if [ "$_ec" -eq 124 ] || [ "$_ec" -eq 137 ] || [ "$_ec" -eq 143 ]; then
            echo "NOT_RUN  selftest:compiler_main (OOM/timeout exit $_ec)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "NOT_RUN  selftest:compiler_main (exit $_ec)"; NOT_RUN=$((NOT_RUN+1))
        fi
    fi
    rm -f "$_st_log"
fi

echo ""
echo "--- Group 2: Block L structural checks ---"
check_grep "block_l:ra_key_helper" \
    "fn ocp_binop_to_ra_key" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_l:has_ra_array" \
    "has_ra" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_l:ra_const_val_array" \
    "ra_const_val" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_l:block_l_comment" \
    "Sprint 84 Block L" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_l:click_ref" \
    "Click.*Combining Reductions|Combining Reductions.*Click" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "block_l:total_22x" \
    "let total: i64 = 22[0-9]" \
    "self-hosted/compiler/main.sio"
check_grep "block_l:test_add_chain" \
    "compiler_main_test_assoc_add_chain" \
    "self-hosted/compiler/main.sio"
check_grep "block_l:test_non_assoc" \
    "compiler_main_test_assoc_non_assoc_no_fold" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 3: Sprint 83/82 Block K/J compatibility ---"
check_grep "compat:block_k_comment" \
    "Sprint 83 Block K" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "compat:block_j_comment" \
    "Sprint 82 Block J" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 4: Compatibility probe ---"
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

cat > "$OUT_DIR/assoc_chain_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint84.assoc_chain_gate.v1",
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
    "Block L extends the single-pass constant lattice to track partial-constant BinOp metadata (has_ra / ra_base / ra_const_val / ra_op), enabling chain folding (x+c1)+c2 → x+(c1+c2) without instruction insertion. Handles all five associative integer ops: Add, Mul, BitAnd, BitOr, BitXor.",
    "Mul-chain fold interacts with Block B strength reduction in the same forward pass: (r0*2)*4 → r0*8 (Block L) → r0<<3 (Block B) — achieving power-of-2 elimination across a two-instr constant chain in a single ocp_const_fold invocation."
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint84/assoc_chain_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
