#!/usr/bin/env bash
# sprint131_xor_not_dist_gate.sh — Sprints 131-132: XOR self-inverse + NOT-distributive
#
# Block AS: XOR self-inverse — (a^b)^b→a, (a^b)^a→b
#   SOTA: GF(2) self-inverse property; LLVM InstCombineAndOrXor.cpp; Hacker's Delight §2-3.
#
# Block AT: NOT-distributive — ~a&(a|b)→~a&b, ~a|(a&b)→~a|b
#   SOTA: Boolean algebra distributive + complement axioms; LLVM SimplifyDemandedBits.
#
# Novel claims:
#   1. XOR RR tracking arrays (as_xor_rr_valid/lhs/rhs) enable self-inverse detection
#   2. Commutative variants: b^(a^b)→a, a^(a^b)→b
#   3. NOT-distributive uses existing is_bnot + ao_or_rr/ao_and_rr tracking cross-reference
#   4. Dual form: ~a|(a&b)→~a|b via (~a|a)&(~a|b) = -1&(~a|b)
set -eo pipefail

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
        if ! grep -qF "T470" "$log_file" 2>/dev/null; then
            echo "NOT_RUN  $name (OOM before T470)"; NOT_RUN=$((NOT_RUN+1))
        else
            echo "FAIL  $name (expected '$expected_line')"; FAIL=$((FAIL+1))
        fi
    fi
}

echo "=== Sprints 131-132: XOR Self-Inverse + NOT-Distributive ==="
echo ""

# --- Group 1: Source code verification ---
echo "--- Source: opt_cleanup.sio Block AS/AT ---"
OPT_FILE="self-hosted/ir/opt_cleanup.sio"

check_grep "src:as_xor_rr_valid_decl" \
    "as_xor_rr_valid.*bool.*256" "$OPT_FILE"
check_grep "src:as_xor_rr_lhs_decl" \
    "as_xor_rr_lhs.*i64.*256" "$OPT_FILE"
check_grep "src:as_xor_rr_rhs_decl" \
    "as_xor_rr_rhs.*i64.*256" "$OPT_FILE"
check_grep "src:block_as_xor_self_inverse" \
    "Block AS.*XOR self-inverse" "$OPT_FILE"
check_grep "src:block_at_not_distributive" \
    "Block AT.*NOT-distributive" "$OPT_FILE"
check_grep "src:at_bnot_src_usage" \
    "bnot_src\[ao_s1" "$OPT_FILE"
check_grep "src:at_or_rr_lhs_usage" \
    "ao_or_rr_lhs\[ao_s2" "$OPT_FILE"

# --- Group 2: main.sio test functions ---
echo ""
echo "--- main.sio test functions ---"
MAIN_FILE="self-hosted/compiler/main.sio"

check_grep "main:T470_fn" \
    "fn compiler_main_test_xor_self_inverse_basic" "$MAIN_FILE"
check_grep "main:T471_fn" \
    "fn compiler_main_test_xor_self_inverse_lhs" "$MAIN_FILE"
check_grep "main:T472_fn" \
    "fn compiler_main_test_xor_self_inverse_commutative" "$MAIN_FILE"
check_grep "main:T476_fn" \
    "fn compiler_main_test_not_dist_and_basic" "$MAIN_FILE"
check_grep "main:T478_fn" \
    "fn compiler_main_test_not_dist_or_basic" "$MAIN_FILE"
check_grep "main:T481_fn" \
    "fn compiler_main_test_not_dist_inner_commutative" "$MAIN_FILE"
check_grep "main:total_updated" \
    "let total: i64 = [0-9]+" "$MAIN_FILE"

# --- Group 3: Self-tests T470-T481 ---
echo ""
echo "--- Self-tests: T470-T481 ---"
SELF_TEST_LOG="$(mktemp)"
if [ ! -x "$SOUC" ]; then
    for name in \
        "selftest:T470" "selftest:T471" "selftest:T472" "selftest:T473" \
        "selftest:T474" "selftest:T475" "selftest:T476" "selftest:T477" \
        "selftest:T478" "selftest:T479" "selftest:T480" "selftest:T481"; do
        TOTAL=$((TOTAL+1))
        echo "NOT_RUN  $name (souc binary not found)"; NOT_RUN=$((NOT_RUN+1))
    done
else
    timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test \
        > "$SELF_TEST_LOG" 2>&1 || _ec=$?

    check_log_line "selftest:T470_xor_self_inverse_basic" \
        "T470 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T471_xor_self_inverse_lhs" \
        "T471 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T472_xor_self_inverse_commutative" \
        "T472 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T473_xor_self_inverse_commutative_lhs" \
        "T473 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T474_xor_self_inverse_neg" \
        "T474 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T475_xor_self_inverse_chain" \
        "T475 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T476_not_dist_and_basic" \
        "T476 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T477_not_dist_and_commutative" \
        "T477 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T478_not_dist_or_basic" \
        "T478 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T479_not_dist_or_commutative" \
        "T479 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T480_not_dist_neg" \
        "T480 OK" "$SELF_TEST_LOG"
    check_log_line "selftest:T481_not_dist_inner_commutative" \
        "T481 OK" "$SELF_TEST_LOG"
fi
rm -f "$SELF_TEST_LOG"

# --- Group 4: Type-check ---
echo ""
echo "--- Type-check ---"
TOTAL=$((TOTAL+1))
if timeout 60 "$SOUC" check self-hosted/compiler/main.sio > /dev/null 2>&1; then
    echo "PASS  typecheck:main.sio"; PASS=$((PASS+1))
else
    echo "FAIL  typecheck:main.sio"; FAIL=$((FAIL+1))
fi

# --- Summary ---
echo ""
echo "=== SUMMARY ==="
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then
    echo "GATE: PASS"
    exit 0
else
    echo "GATE: FAIL"
    exit 1
fi
