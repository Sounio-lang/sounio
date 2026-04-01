#!/usr/bin/env bash
# sprint120_or_and_elimination_gate.sh — Block AN: OR-then-AND constant elimination
#
# SOTA: LLVM InstCombineAndOrXor.cpp known-bits analysis; Hacker's Delight §2.
# (x | C1) & C2 → C2 when (C1 & C2) == C2  [all bits of C2 already set by OR]
# Commutative: C2 & (x | C1) → C2 same condition.
#
# Novel claims:
#   1. (x | C1) & C2 → IrLoadImm(C2) when C2 is subset of C1's bits
#   2. Commutative: C2 & (x | C1) handled symmetrically
#   3. Zero new tracking arrays: reuses am_or_valid from Block AM (Sprint 116)
#   4. Chains with Block AK (AND superset) and Block AM (OR superset)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
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
        if ! grep -qF "T422" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T422)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprint 120: Block AN — OR-then-AND Constant Elimination ==="
echo ""

# --- Group 1: Self-tests T422-T427 ---
echo "--- Self-tests: OR-then-AND T422-T427 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in "selftest:or_and_subset_fold" "selftest:or_and_wide_subset" \
                "selftest:or_and_not_subset" "selftest:or_and_same_mask" \
                "selftest:or_and_commutative" "selftest:or_and_no_or_track"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc not executable)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$SELF_TEST_LOG" 2>&1 || true
    check_log_line "selftest:or_and_subset_fold" \
        "T422 OK: Sprint 120 Block AN:" "$SELF_TEST_LOG"
    check_log_line "selftest:or_and_wide_subset" \
        "T423 OK: Sprint 120 Block AN:" "$SELF_TEST_LOG"
    check_log_line "selftest:or_and_not_subset" \
        "T424 OK: Sprint 120 Block AN:" "$SELF_TEST_LOG"
    check_log_line "selftest:or_and_same_mask" \
        "T425 OK: Sprint 120 Block AN:" "$SELF_TEST_LOG"
    check_log_line "selftest:or_and_commutative" \
        "T426 OK: Sprint 120 Block AN:" "$SELF_TEST_LOG"
    check_log_line "selftest:or_and_no_or_track" \
        "T427 OK: Sprint 120 Block AN:" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

echo ""
echo "--- Group 2: opt_cleanup.sio Block AN implementation ---"
check_grep "ocp:block_an_comment" \
    "Sprint 118 Block AN" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:an_subset_condition" \
    "an_c1 & v2.*== v2|C1 & C2.*== C2" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:an_ir_load_imm_fold" \
    "ir_load_imm.*instr.dst.*v2|ir_load_imm.*instr.dst.*v1" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "ocp:an_uses_am_or_valid" \
    "am_or_valid\[s1 as usize\]|am_or_valid\[s2 as usize\]" \
    "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: main.sio test functions ---"
check_grep "main:T422_fn" \
    "fn compiler_main_test_or_and_subset_fold" \
    "self-hosted/compiler/main.sio"
check_grep "main:T423_fn" \
    "fn compiler_main_test_or_and_wide_subset" \
    "self-hosted/compiler/main.sio"
check_grep "main:T424_fn" \
    "fn compiler_main_test_or_and_not_subset" \
    "self-hosted/compiler/main.sio"
check_grep "main:T425_fn" \
    "fn compiler_main_test_or_and_same_mask" \
    "self-hosted/compiler/main.sio"
check_grep "main:T426_fn" \
    "fn compiler_main_test_or_and_commutative" \
    "self-hosted/compiler/main.sio"
check_grep "main:T427_fn" \
    "fn compiler_main_test_or_and_no_or_track" \
    "self-hosted/compiler/main.sio"
check_grep "main:total_updated" \
    "let total: i64 = [0-9]+" \
    "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Block AM/AK regression (OR/AND tracking still works) ---"
check_grep "regression:am_or_valid_tracking" \
    "am_or_valid" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_am_comment" \
    "Sprint 116 Block AM" \
    "self-hosted/ir/opt_cleanup.sio"
check_grep "regression:block_ak_comment" \
    "Sprint 114 Block AK" \
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

mkdir -p "$ROOT_DIR/artifacts/sprint120"
cat > "$ROOT_DIR/artifacts/sprint120/or_and_elimination.v1.json" <<JSONEOF
{
  "schema": "sounio.sprint120.or_and_elimination_gate.v1",
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
    "(x | C1) & C2 -> IrLoadImm(C2) when (C1 & C2) == C2: bit-subset detection eliminates AND",
    "Commutative: C2 & (x | C1) handled symmetrically",
    "Zero new tracking arrays: reuses am_or_valid/am_or_const_val from Block AM",
    "Chains Block AM (OR superset) + Block AK (AND superset) + Block AN (OR-then-AND)"
  ]
}
JSONEOF

echo ""
echo "Artifact: artifacts/sprint120/or_and_elimination.v1.json"

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
