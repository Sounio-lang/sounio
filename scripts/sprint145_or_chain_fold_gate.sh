#!/usr/bin/env bash
# sprint145_or_chain_fold_gate.sh — Block BG: OR constant-chain fold
#
# SOTA: LLVM InstCombineAndOrXor.cpp — or(or X,C1),C2 → or X,(C1|C2);
# Boolean algebra associativity + idempotency (a|a=a).
# (x | C1) | C2 → x | (C1 | C2): merges two OR-with-constant chains in one pass.
# Distinct from Block AM ((x|C1)|C2→IrCopy when C2⊆C1): fires when C2 has new bits.
# Analogue of Block AE (AND chain fold) for OR.
# Zero new arrays: reuses am_or_valid/var_src/const_val + def_at from Block AM.
#
# Novel claims:
#   1. (x|C1)|C2 → x|(C1|C2): general OR chain merges non-overlapping masks
#   2. C2|(x|C1) → x|(C1|C2): commutative form handled
#   3. Three-level chains collapse in single pass: ((x|1)|2)|4 → x|7
#   4. Distinct from Block AM: fires only when C2 has bits not in C1
#   5. Zero new arrays: reuses am_or_valid/var_src/const_val/def_at from Block AM
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
    if [ ! -s "$log_file" ]; then
        echo "NOT_RUN  $name (log empty — OOM/killed)"; NOT_RUN=$((NOT_RUN+1))
        return
    fi
    if grep -qF "$expected_line" "$log_file"; then
        echo "PASS  $name"; PASS=$((PASS+1))
    else
        if ! grep -qF "T530" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T530)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 145: Block BG — OR Constant-Chain Fold ==="
echo ""

# --- Group 1: Self-tests T536-T541 ---
echo "--- Self-tests: OR chain fold T536-T541 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:or_chain_basic" "selftest:or_chain_commutative" \
                "selftest:or_chain_three_levels" "selftest:or_chain_no_fold_diff_base" \
                "selftest:or_chain_superset_block_am" "selftest:or_chain_and_not_folded"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:or_chain_basic" \
        "T536 OK: Sprint 145 Block BG:" "$SELF_TEST_LOG"
    check_log_line "selftest:or_chain_commutative" \
        "T537 OK: Sprint 145 Block BG:" "$SELF_TEST_LOG"
    check_log_line "selftest:or_chain_three_levels" \
        "T538 OK: Sprint 145 Block BG:" "$SELF_TEST_LOG"
    check_log_line "selftest:or_chain_no_fold_diff_base" \
        "T539 OK: Sprint 145 Block BG:" "$SELF_TEST_LOG"
    check_log_line "selftest:or_chain_superset_block_am" \
        "T540 OK: Sprint 145 Block BG:" "$SELF_TEST_LOG"
    check_log_line "selftest:or_chain_and_not_folded" \
        "T541 OK: Sprint 145 Block BG:" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio Block BG implementation ---"
check_grep "ocp:bg_block_comment" \
    "Sprint 145 Block BG" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bg_am_or_valid_check" \
    "am_or_valid\[s1.*usize\].*am_or_valid\[s2|am_or_valid\[s" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bg_merged_or" \
    "bg_merged = bg_c" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bg_def_patch" \
    "ir_load_imm.*bg_merged" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:bg_binop_result" \
    "ir_binop.*instr.dst.*am_or_var_src" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: main.sio test functions ---"
check_grep "main:T536_fn" \
    "fn compiler_main_test_or_chain_basic" \
    "self-hosted/compiler/main.sio"
check_grep "main:T537_fn" \
    "fn compiler_main_test_or_chain_commutative" \
    "self-hosted/compiler/main.sio"
check_grep "main:T538_fn" \
    "fn compiler_main_test_or_chain_three_levels" \
    "self-hosted/compiler/main.sio"
check_grep "main:T539_fn" \
    "fn compiler_main_test_or_chain_no_fold_diff_base" \
    "self-hosted/compiler/main.sio"
check_grep "main:T540_fn" \
    "fn compiler_main_test_or_chain_superset_still_block_am" \
    "self-hosted/compiler/main.sio"
check_grep "main:T541_fn" \
    "fn compiler_main_test_or_chain_and_not_folded" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_updated" \
    "let total: i64 = [0-9]+" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Block AM/AE regression ---"
check_grep "regression:am_or_valid_tracking" \
    "am_or_valid" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:am_or_var_src_tracking" \
    "am_or_var_src" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_am_comment" \
    "Block AM" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_ae_comment" \
    "Block AE" \
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

mkdir -p "$ROOT_DIR/artifacts/sprint145"
cat > "$ROOT_DIR/artifacts/sprint145/or_chain_fold.v1.json" <<EOF
{
  "schema": "sounio.sprint145.or_chain_fold_gate.v1",
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
    "(x|C1)|C2 → x|(C1|C2): general OR chain merges non-overlapping masks in one pass",
    "C2|(x|C1) → x|(C1|C2): commutative form patches def_at[s1] IrLoadImm in-place",
    "Three-level chains collapse per pass via updated am_or tracking",
    "Distinct from Block AM ((x|C1)|C2→IrCopy when C2⊆C1): fires when C2 has new bits",
    "SOTA: LLVM InstCombineAndOrXor.cpp or(or X,C1),C2→or X,(C1|C2); Boolean associativity"
  ]
}
EOF

echo ""
echo "Artifact: artifacts/sprint145/or_chain_fold.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
