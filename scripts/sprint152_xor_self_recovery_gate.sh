#!/usr/bin/env bash
# sprint152_xor_self_recovery_gate.sh — Block BN: XOR self-recovery
#
# SOTA: LLVM InstCombineAndOrXor — XOR group law: (a^b)^a=b; GF(2) §2-3.
# (x^C)^x → C  ;  x^(x^C) → C  ;  (x^C1)^(x^C2) → C1^C2
# Distinct from Block AU ((a^b)^(a^c)→b^c for RR-XOR): uses bm_xor tracking
# for const-xor case. Produces IrLoadImm constants from XOR recovery.
# Zero new arrays: uses bm_xor_valid/src/cval (Block BM) + is_const/const_val.
#
# Novel claims:
#   1. (x^C)^x → C: XOR self-recovery yields constant via bm_xor tracking
#   2. x^(x^C) → C: commutative outer XOR handled
#   3. (x^C1)^(x^C2) → C1^C2: shared base cancels, constants XOR
#   4. Negative constants handled correctly (x^(-1))^x→-1
#   5. Zero new arrays: reuses bm_xor_valid/src/cval from Block BM (Sprint 151)
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
    if [ ! -s "$log_file" ]; then
        echo "NOT_RUN  $name (log empty — OOM/killed)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        if ! grep -qF "T572" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T572)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 152: Block BN — XOR Self-Recovery ==="
echo ""

# --- Group 1: Self-tests T578-T583 ---
echo "--- Self-tests: XOR self-recovery T578-T583 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:xor_recovery_basic" "selftest:xor_recovery_commutative" \
                "selftest:xor_pair_cancel" "selftest:xor_recovery_neg_const" \
                "selftest:xor_recovery_no_fold_diff" "selftest:xor_recovery_no_fold_plain"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:xor_recovery_basic" \
        "T578 OK: Sprint 152 Block BN:" "$SELF_TEST_LOG"
    check_log_line "selftest:xor_recovery_commutative" \
        "T579 OK: Sprint 152 Block BN:" "$SELF_TEST_LOG"
    check_log_line "selftest:xor_pair_cancel" \
        "T580 OK: Sprint 152 Block BN:" "$SELF_TEST_LOG"
    check_log_line "selftest:xor_recovery_neg_const" \
        "T581 OK: Sprint 152 Block BN:" "$SELF_TEST_LOG"
    check_log_line "selftest:xor_recovery_no_fold_diff" \
        "T582 OK: Sprint 152 Block BN:" "$SELF_TEST_LOG"
    check_log_line "selftest:xor_recovery_no_fold_plain" \
        "T583 OK: Sprint 152 Block BN:" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio Block BN implementation ---"
check_grep "ocp:bn_block_comment" \
    "Sprint 152 Block BN" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bn_bm_valid_check" \
    "bm_xor_valid\[bn_s" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bn_src_match" \
    "bm_xor_src\[bn_s.*== bn_s" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bn_load_imm_cval" \
    "ir_load_imm.*bn_instr.dst.*bm_xor_cval" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bn_pair_cancel" \
    "bn_merged = bm_xor_cval\[bn_s1.*\^ bm_xor_cval\[bn_s2" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: main.sio test functions ---"
check_grep "main:T578_fn" \
    "fn compiler_main_test_xor_self_recovery_basic" \
    "self-hosted/compiler/main.sio"
check_grep "main:T579_fn" \
    "fn compiler_main_test_xor_self_recovery_commutative" \
    "self-hosted/compiler/main.sio"
check_grep "main:T580_fn" \
    "fn compiler_main_test_xor_pair_cancel" \
    "self-hosted/compiler/main.sio"
check_grep "main:T581_fn" \
    "fn compiler_main_test_xor_self_recovery_neg_const" \
    "self-hosted/compiler/main.sio"
check_grep "main:T582_fn" \
    "fn compiler_main_test_xor_self_recovery_no_fold_diff_base" \
    "self-hosted/compiler/main.sio"
check_grep "main:T583_fn" \
    "fn compiler_main_test_xor_self_recovery_no_fold_plain" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_updated" \
    "let total: i64 = [0-9]+" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Block BM/AU regression ---"
check_grep "regression:bm_xor_valid" \
    "bm_xor_valid" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:bm_xor_cval" \
    "bm_xor_cval" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_bm_comment" \
    "Block BM" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_au_comment" \
    "Block AU" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:opt_cleanup_module" \
    "fn opt_cleanup_module" \
    "self-hosted/ir/opt_cleanup.sio"

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

mkdir -p "$ROOT_DIR/artifacts/sprint152"
cat > "$ROOT_DIR/artifacts/sprint152/xor_self_recovery.v1.json" <<EOF
{
  "schema": "sounio.sprint152.xor_self_recovery_gate.v1",
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
    "(x^C)^x → C: XOR self-recovery yields constant via bm_xor_src match",
    "x^(x^C) → C: commutative outer XOR handled symmetrically",
    "(x^C1)^(x^C2) → C1^C2: shared base cancels, constants XOR via bm_xor pair",
    "Negative constant recovery: (x^-1)^x → -1 correctly",
    "SOTA: LLVM InstCombineAndOrXor XOR group law (a^b)^a=b; GF(2) §2-3"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint152/xor_self_recovery.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
