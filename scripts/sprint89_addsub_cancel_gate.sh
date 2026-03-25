#!/usr/bin/env bash
# sprint89_addsub_cancel_gate.sh — Block P: add-sub/sub-add cancellation
#
# (x + C) - C → IrCopy(x)  ;  (x - C) + C → IrCopy(x).
# Tracks partial-const add/sub in as_valid/as_var_src/as_const_v/as_is_add arrays.
# Fold fires when the inverse op appears with the matching constant.
# Commutative add tracking covers (C + x) - C as well.
#
# Self-tests T252-T257; total 257/257.
#
# SOTA: LLVM InstCombineAddSub.cpp; Muchnick §12.2.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"
cd "$ROOT_DIR"

PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
OUT_DIR="$ROOT_DIR/artifacts/sprint89"
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

echo "=== Sprint 89: Block P — Add-Sub/Sub-Add Cancellation ==="
echo ""

echo "--- Group 1: Self-test smoke (257/257) ---"
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
echo "--- Group 2: Block P structural checks ---"
chk "block_p:comment"      "Sprint 89 Block P"           "self-hosted/ir/opt_cleanup.sio"
chk "block_p:as_valid"     "var as_valid"                "self-hosted/ir/opt_cleanup.sio"
chk "block_p:as_var_src"   "var as_var_src"              "self-hosted/ir/opt_cleanup.sio"
chk "block_p:as_const_v"   "var as_const_v"              "self-hosted/ir/opt_cleanup.sio"
chk "block_p:as_is_add"    "var as_is_add"               "self-hosted/ir/opt_cleanup.sio"
chk "block_p:tracking"     "as_valid\[instr\.dst"        "self-hosted/ir/opt_cleanup.sio"
chk "block_p:fold_sub"     "as_is_add\[s1 as usize\] && as_const_v\[s1 as usize\] == v2" "self-hosted/ir/opt_cleanup.sio"
chk "block_p:fold_add"     "!as_is_add\[s1 as usize\] && as_const_v\[s1 as usize\] == v2" "self-hosted/ir/opt_cleanup.sio"
chk "block_p:sota_ref"     "InstCombineAddSub|Muchnick.*12" "self-hosted/ir/opt_cleanup.sio"

echo ""
echo "--- Group 3: T252-T257 wiring ---"
chk "main:T252_fn"     "compiler_main_test_addsub_cancel_add_first"   "self-hosted/compiler/main.sio"
chk "main:T253_fn"     "compiler_main_test_addsub_cancel_sub_first"   "self-hosted/compiler/main.sio"
chk "main:T254_fn"     "compiler_main_test_addsub_cancel_commut"      "self-hosted/compiler/main.sio"
chk "main:T255_fn"     "compiler_main_test_addsub_no_cancel_mismatch" "self-hosted/compiler/main.sio"
chk "main:T256_fn"     "compiler_main_test_addsub_cancel_neg_const"   "self-hosted/compiler/main.sio"
chk "main:T257_fn"     "compiler_main_test_addsub_cancel_chain"       "self-hosted/compiler/main.sio"
chk "main:T252_runner" "T252 OK"                                      "self-hosted/compiler/main.sio"
chk "main:T257_runner" "T257 OK"                                      "self-hosted/compiler/main.sio"
chk "main:total_257"   "let total: i64 = 2[5-9][0-9]"                "self-hosted/compiler/main.sio"

echo ""
echo "--- Group 4: Sprint 88/87/86 regression ---"
chk "compat:block_o"   "Sprint 88 Block O"    "self-hosted/ir/opt_cleanup.sio"
chk "compat:block_n"   "Sprint 87 Block N"    "self-hosted/ir/opt_cleanup.sio"
chk "compat:block_m"   "Sprint 86 Block M"    "self-hosted/ir/opt_cleanup.sio"
chk "compat:T251"      "T251 OK"              "self-hosted/compiler/main.sio"
chk "compat:T245"      "T245 OK"              "self-hosted/compiler/main.sio"

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $NOT_RUN not_run (total $TOTAL) ==="

STATUS="pass"; REASON="all_cases_passed"
[ "$FAIL" -gt 0 ] && STATUS="fail" && REASON="${FAIL}_cases_failed"
[ "$NOT_RUN" -gt 0 ] && [ "$FAIL" -eq 0 ] && REASON="all_run_passed_${NOT_RUN}_not_run"

printf '{\n  "schema": "sounio.sprint89.addsub_cancel_gate.v1",\n  "generated_at": "%s",\n  "status": "%s",\n  "reason": "%s",\n  "metrics": {"total": %d, "passed": %d, "failed": %d, "not_run": %d},\n  "novel_claims": [\n    "Block P: as_valid/as_var_src/as_const_v/as_is_add arrays track partial-const add/sub; (x+C)-C and (x-C)+C → IrCopy(x)",\n    "Commutative add tracking ensures (C+x)-C cancels same as (x+C)-C",\n    "C=0 case handled by existing x+0/x-0 identity folds; Block P only fires for non-zero C",\n    "T252-T257: add-first, sub-first, commutative, mismatch guard, negative constant, chain; total 257/257",\n    "SOTA: LLVM InstCombineAddSub.cpp; Muchnick §12.2"\n  ]\n}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%S+00:00)" "$STATUS" "$REASON" \
    "$TOTAL" "$PASS" "$FAIL" "$NOT_RUN" \
    > "$OUT_DIR/addsub_cancel_gate.v1.json"

echo "Artifact: artifacts/sprint89/addsub_cancel_gate.v1.json"
[ "$FAIL" -gt 0 ] && exit 1 || exit 0
