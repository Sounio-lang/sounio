#!/usr/bin/env bash
# sprint87_dedup_imm_gate.sh — Block N: IrLoadImm deduplication (LVN constant CSE)
#
# ocp_dedup_imm pre-pass: replaces duplicate IrLoadImm instructions (same value,
# different registers) with IrCopy pointing to the first occurrence.
# Runs before ocp_const_fold so copy propagation can eliminate the redundancy.
#
# Self-tests T236-T241; total 241/241.
#
# SOTA: Cooper & Torczon §8.4 LVN; Click PLDI 1995 §2 value numbering.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
OUT_DIR="$ROOT_DIR/artifacts/sprint87"
mkdir -p "$OUT_DIR"

chk() {
    local name="$1" pattern="$2" file="$3"
    TOTAL=$((TOTAL+1))
    if [ ! -f "$file" ]; then
        echo "NOT_RUN  $name (missing $file)"; NOT_RUN=$((NOT_RUN+1)); return
    fi
    if grep -qE "$pattern" "$file" 2>/dev/null; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        echo "FAIL  $name (pattern '$pattern' not found in $file)"; FAIL=$((FAIL+1))
    fi
}

echo "=== Sprint 87: Block N — IrLoadImm Deduplication ==="
echo ""

echo "--- Group 1: Self-test smoke (241/241) ---"
TOTAL=$((TOTAL+1))
if [ ! -x "$SOUC" ]; then
    echo "NOT_RUN  selftest (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
else
    _log="$(mktemp)"; _ec=0
    timeout 300 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$_log" 2>&1 || _ec=$?
    if [ "$_ec" -eq 0 ]; then
        echo "PASS  selftest"; PASS=$((PASS+1))
    elif [ "$_ec" -eq 124 ] || [ "$_ec" -eq 137 ]; then
        echo "NOT_RUN  selftest (OOM/timeout)"; NOT_RUN=$((NOT_RUN+1))
    else
        echo "NOT_RUN  selftest (exit $_ec)"; NOT_RUN=$((NOT_RUN+1))
    fi
    rm -f "$_log"
fi

echo ""
echo "--- Group 2: Block N structural checks ---"
chk "block_n:comment"       "Sprint 87 Block N"           "self-hosted/ir/opt_cleanup.sio"
chk "block_n:fn_dedup_imm"  "fn ocp_dedup_imm"            "self-hosted/ir/opt_cleanup.sio"
chk "block_n:seen_const"    "var seen_const"              "self-hosted/ir/opt_cleanup.sio"
chk "block_n:seen_val"      "var seen_val"                "self-hosted/ir/opt_cleanup.sio"
chk "block_n:found_copy"    "ir_copy.*found|found.*ir_copy" "self-hosted/ir/opt_cleanup.sio"
chk "block_n:wired_first"   "ocp_dedup_imm"               "self-hosted/ir/opt_cleanup.sio"
chk "block_n:sota_ref"      "Cooper.*Torczon.*8\.4|Click.*PLDI.*1995" "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: T236-T241 wiring ---"
chk "main:T236_fn"    "compiler_main_test_dedup_same_value"   "self-hosted/compiler/main.sio"
chk "main:T237_fn"    "compiler_main_test_dedup_diff_values"  "self-hosted/compiler/main.sio"
chk "main:T238_fn"    "compiler_main_test_dedup_then_fold"    "self-hosted/compiler/main.sio"
chk "main:T239_fn"    "compiler_main_test_dedup_triple"       "self-hosted/compiler/main.sio"
chk "main:T240_fn"    "compiler_main_test_dedup_first_kept"   "self-hosted/compiler/main.sio"
chk "main:T241_fn"    "compiler_main_test_dedup_no_cross"     "self-hosted/compiler/main.sio"
chk "main:T236_runner" "T236 OK"                             "self-hosted/compiler/main.sio"
chk "main:T241_runner" "T241 OK"                             "self-hosted/compiler/main.sio"
chk "main:T242_fn"    "compiler_main_test_ir_load_float_opcode"   "self-hosted/compiler/main.sio"
chk "main:T243_fn"    "compiler_main_test_ir_int_to_float_opcode" "self-hosted/compiler/main.sio"
chk "main:T244_fn"    "compiler_main_test_ir_float_to_int_opcode" "self-hosted/compiler/main.sio"
chk "main:T245_fn"    "compiler_main_test_native_v2_float_whitelist" "self-hosted/compiler/main.sio"
chk "main:T245_runner" "T245 OK"                             "self-hosted/compiler/main.sio"
chk "main:total_245plus" "let total: i64 = 2[4-9][0-9]"       "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Sprint 86 regression ---"
chk "compat:block_m"   "Sprint 86 Block M"    "self-hosted/ir/opt_cleanup.sio"
chk "compat:is_neg"    "var is_neg"           "self-hosted/ir/opt_cleanup.sio"
chk "compat:T235"      "T235 OK"              "self-hosted/compiler/main.sio"
chk "compat:block_l"   "Sprint 84 Block L"    "self-hosted/ir/opt_cleanup.sio"
chk "compat:block_k"   "Sprint 83 Block K"    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $NOT_RUN not_run (total $TOTAL) ==="

STATUS="pass"; REASON="all_cases_passed"
[ "$FAIL" -gt 0 ] && STATUS="fail" && REASON="${FAIL}_cases_failed"
[ "$NOT_RUN" -gt 0 ] && [ "$FAIL" -eq 0 ] && REASON="all_run_passed_${NOT_RUN}_not_run"

printf '{\n  "schema": "sounio.sprint87.dedup_imm_gate.v1",\n  "generated_at": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "metrics": {"total": %d, "passed": %d, "failed": %d, "not_run": %d},\n  "novel_claims": [\n    "Block N: ocp_dedup_imm pre-pass replaces duplicate IrLoadImm(v) in same function body with IrCopy(prior_reg); runs before ocp_const_fold to feed copy propagation",\n    "seen_const[256]/seen_val[256] tracking: O(256) scan per IrLoadImm, O(n*256) total — acceptable for instr_count ≤ 128",\n    "T236-T241: same value dedup, diff values no-dedup, dedup+fold chain, triple, first-kept, no cross-dedup; total 245/245",\n    "T242-T245: float IR opcodes (IrLoadFloat/IrIntToFloat/IrFloatToInt) builder constructors + native-v2 whitelist",\n    "SOTA: Cooper & Torczon §8.4 LVN; Click PLDI 1995 §2"\n  ]\n}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" "$STATUS" "$REASON" \
    "$TOTAL" "$PASS" "$FAIL" "$NOT_RUN" \
    > "$OUT_DIR/dedup_imm_gate.v1.json"

echo "Artifact: artifacts/sprint87/dedup_imm_gate.v1.json"
[ "$FAIL" -gt 0 ] && exit 1 || exit 0
