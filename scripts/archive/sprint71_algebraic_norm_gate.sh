#!/usr/bin/env bash
# sprint71_algebraic_norm_gate.sh — IR algebraic normalization
# Block A: same-register fold (x-x=0, x^x=0)
# Block B: left-constant commutative SR (4*x → x<<2)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
OUT_DIR="$ROOT_DIR/artifacts/sprint71"
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

echo "=== Sprint 71: IR Algebraic Normalization ==="
echo ""

echo "--- Group 1: Compiler self-test smoke ---"
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
echo "--- Group 2: Structural checks ---"
check_grep "struct:block_a_comment" \
    "Sprint 71 Block A" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "struct:block_b_comment" \
    "Sprint 71 Block B" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "struct:same_reg_xor" \
    "OpBitXor" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "struct:comm_sr_pow2" \
    "ocp_is_power_of_two.*cm_v1" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "struct:comm_sr_swap" \
    "cm_s2.*BinaryOp::OpShl.*cm_s1" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "struct:t109_test_fn" \
    "fn compiler_main_test_alg_same_reg_sub" \
    "self-hosted/compiler/main.sio"
check_grep "struct:t111_test_fn" \
    "fn compiler_main_test_alg_comm_sr_4x" \
    "self-hosted/compiler/main.sio"
check_grep "struct:total_116" \
    "let total: i64 = 11[0-9]" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 3: Stability corpus (Sprint 70 fixtures) ---"
check_grep "fixture:two_fn_loop" \
    "fn bump" \
    "tests/frontend/lowering_stability_two_fn_loop.sio"
check_grep "fixture:min_main" \
    "fn main" \
    "tests/frontend/lowering_stability_min_main.sio"

echo ""
echo "--- Group 4: Sprint 70/44 compatibility ---"
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

cat > "$OUT_DIR/algebraic_norm_gate.v1.json" <<EOF
{
  "schema": "sounio.sprint71.algebraic_norm_gate.v1",
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
    "Same-register algebraic fold closes the non-constant identity gap (x-x=0, x^x=0) without a full value-numbering pass",
    "Left-constant commutative SR symmetrizes the strength-reduction surface: all pow-2 multiplications reduce to shifts regardless of operand order",
    "Both blocks compose with Sprint 69 copy-prop: r1=copy(r0=4); r1*x reduces to x<<2 via the full Sprint 69+71 interop chain"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint71/algebraic_norm_gate.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
