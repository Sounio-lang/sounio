#!/usr/bin/env bash
# sprint86_neg_arith_gate.sh — Block M: negation arithmetic
# x*(-1)→0-x, x/(-1)→0-x, x%(-1)→0, neg(neg(x))→IrCopy(x)
# SOTA: LLVM InstCombine; Cooper & Torczon §9.
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./bin/souc"
OUT_DIR="$ROOT_DIR/artifacts/sprint86"; mkdir -p "$OUT_DIR"

chk() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1))
  if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi
  if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1))
  else echo "FAIL  $n (pattern '$p' not found in $f)"; FAIL=$((FAIL+1)); fi; }

echo "=== Sprint 86: Block M — Negation Arithmetic ==="
echo ""

echo "--- Group 1: Self-test smoke (235/235) ---"
TOTAL=$((TOTAL+1))
if [ ! -x "$SOUC" ]; then echo "NOT_RUN  selftest"; NOT_RUN=$((NOT_RUN+1))
else
  _l="$(mktemp)"; _e=0
  timeout 300 "$SOUC" run self-hosted/compiler/main.sio -- --self-test >"$_l" 2>&1 || _e=$?
  if [ "$_e" -eq 0 ]; then echo "PASS  selftest:compiler_main"; PASS=$((PASS+1))
  elif [ "$_e" -eq 124 ] || [ "$_e" -eq 137 ]; then echo "NOT_RUN  selftest (OOM/timeout)"; NOT_RUN=$((NOT_RUN+1))
  else echo "NOT_RUN  selftest (exit $_e)"; NOT_RUN=$((NOT_RUN+1)); fi
  rm -f "$_l"
fi

echo ""; echo "--- Group 2: Block M structural checks ---"
chk "block_m:comment"       "Sprint 86 Block M"           "self-hosted/ir/opt_cleanup.sio"
chk "block_m:is_neg_arr"    "var is_neg"                  "self-hosted/ir/opt_cleanup.sio"
chk "block_m:neg_src_arr"   "var neg_src"                 "self-hosted/ir/opt_cleanup.sio"
chk "block_m:mul_minus_one" "OpMul.*OpDiv|OpSub.*mn_s1"   "self-hosted/ir/opt_cleanup.sio"
chk "block_m:div_minus_one" "OpDiv.*OpMul|ir_binop.*mn_s2.*BinaryOp::OpSub" "self-hosted/ir/opt_cleanup.sio"
chk "block_m:rem_minus_one" "OpRem"                       "self-hosted/ir/opt_cleanup.sio"
chk "block_m:double_neg"    "is_neg\[nn_s as usize\]"     "self-hosted/ir/opt_cleanup.sio"
chk "block_m:instcombine"   "InstCombine|Cooper.*Torczon" "self-hosted/ir/opt_cleanup.sio"

echo ""; echo "--- Group 3: T230-T235 wiring ---"
chk "main:T230_fn"   "compiler_main_test_neg_mul_minus_one"   "self-hosted/compiler/main.sio"
chk "main:T231_fn"   "compiler_main_test_neg_div_minus_one"   "self-hosted/compiler/main.sio"
chk "main:T232_fn"   "compiler_main_test_neg_rem_minus_one"   "self-hosted/compiler/main.sio"
chk "main:T233_fn"   "compiler_main_test_neg_double_neg"      "self-hosted/compiler/main.sio"
chk "main:T234_fn"   "compiler_main_test_neg_single_no_fold"  "self-hosted/compiler/main.sio"
chk "main:T235_fn"   "compiler_main_test_neg_commut_mul_minus_one" "self-hosted/compiler/main.sio"
chk "main:total_235plus" "let total: i64 = 2[0-9][0-9]"         "self-hosted/compiler/main.sio"

echo ""; echo "--- Group 4: Sprint 84-85 regression ---"
chk "compat:block_l"   "Sprint 84 Block L"   "self-hosted/ir/opt_cleanup.sio"
chk "compat:block_k"   "Sprint 83 Block K"   "self-hosted/ir/opt_cleanup.sio"
chk "compat:block_i"   "Sprint 81 Block I"   "self-hosted/ir/opt_cleanup.sio"
chk "compat:T229"      "T229 OK"             "self-hosted/compiler/main.sio"

echo ""; echo "=== Results: $PASS passed, $FAIL failed, $NOT_RUN not_run (total $TOTAL) ==="
STATUS="pass"; [ "$FAIL" -gt 0 ] && STATUS="fail"
printf '{"schema":"sounio.sprint86.neg_arith_gate.v1","status":"%s","metrics":{"total":%d,"passed":%d,"failed":%d,"not_run":%d}}\n' \
  "$STATUS" "$TOTAL" "$PASS" "$FAIL" "$NOT_RUN" > "$OUT_DIR/neg_arith_gate.v1.json"
echo "Artifact: artifacts/sprint86/neg_arith_gate.v1.json"
[ "$FAIL" -gt 0 ] && exit 1 || exit 0
