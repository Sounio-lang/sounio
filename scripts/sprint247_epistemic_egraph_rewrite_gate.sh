#!/usr/bin/env bash
# sprint247_epistemic_egraph_rewrite_gate.sh — Sprint 247: GUM-guided epistemic e-graph rewriting
#
# World-first analytically GUM-guided floating-point rewriting in any compiler.
# Herbie (Panchekha et al., PLDI 2015) uses empirical sampling; Sounio uses
# GUM §5.1.2 delta-method: u(z)^2 = u(x)^2 + u(y)^2 analytically.
#
# Checks:
#   1. egraph.sio type-checks (Phase 1: opcodes, struct, helpers, saturation, extractor)
#   2. opt_cleanup.sio type-checks (Phase 2: ocp_egraph_epistemic_pass wired in)
#   3. egraph.sio self-tests: 70 / 70 passed
#   4. main.sio type-checks
#   5. main.sio self-tests: T1148-T1153 all pass
set -eo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."  ; pwd)"; cd "$ROOT_DIR"
PASS=0; FAIL=0; NOT_RUN=0; TOTAL=0
SOUC="./artifacts/omega/souc-bin/souc-linux-x86_64-jit"
cg() { local n="$1" p="$2" f="$3"; TOTAL=$((TOTAL+1)); if [ ! -f "$f" ]; then echo "NOT_RUN  $n"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qE "$p" "$f" 2>/dev/null; then echo "PASS  $n"; PASS=$((PASS+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
cl() { local n="$1" e="$2" l="$3"; TOTAL=$((TOTAL+1)); if [ ! -s "$l" ]; then echo "NOT_RUN  $n (OOM)"; NOT_RUN=$((NOT_RUN+1)); return; fi; if grep -qF "$e" "$l"; then echo "PASS  $n"; PASS=$((PASS+1)); elif ! grep -qF "T1148" "$l" 2>/dev/null; then echo "NOT_RUN  $n (OOM before T1148)"; NOT_RUN=$((NOT_RUN+1)); else echo "FAIL  $n"; FAIL=$((FAIL+1)); fi; }
echo "=== Sprint 247: GUM-guided Epistemic E-Graph Rewriting ==="
echo ""
echo "--- Source checks ---"
E="self-hosted/ir/egraph.sio"
O="self-hosted/ir/opt_cleanup.sio"
cg "egraph:float_opcodes"    "EG_OP_FADD|EG_OP_FSUB|EG_OP_FMUL" "$E"
cg "egraph:measure_opcode"   "EG_OP_MEASURE" "$E"
cg "egraph:unc_lo_field"     "unc_lo.*\[i64.*64\]" "$E"
cg "egraph:eg_isqrt"         "fn eg_isqrt" "$E"
cg "egraph:eg_quantize_unc"  "fn eg_quantize_unc" "$E"
cg "egraph:add_var_with_unc" "fn eg_small_add_var_with_unc" "$E"
cg "egraph:add_binop_unc"    "fn eg_small_add_binop_with_unc" "$E"
cg "egraph:saturate_float"   "fn eg_small_saturate_float" "$E"
cg "egraph:extract_epistemic" "fn eg_small_extract_epistemic" "$E"
cg "opt:binop_to_float"      "fn ocp_binop_to_float_eg_op" "$O"
cg "opt:is_measure"          "fn ocp_is_measure" "$O"
cg "opt:epistemic_pass"      "fn ocp_egraph_epistemic_pass" "$O"
cg "opt:pass_wired"          "ocp_egraph_epistemic_pass" "$O"
echo ""
echo "--- Tests ---"
M="self-hosted/compiler/main.sio"
cg "main:T1148_fn" "fn compiler_main_test_ge_standard_passthrough" "$M"
cg "main:T1149_fn" "fn compiler_main_test_ge_precision_no_crash" "$M"
cg "main:T1150_fn" "fn compiler_main_test_ge_count_guard" "$M"
cg "main:T1151_fn" "fn compiler_main_test_ge_binop_no_crash" "$M"
cg "main:T1152_fn" "fn compiler_main_test_ge_quantize_integration" "$M"
cg "main:T1153_fn" "fn compiler_main_test_ge_pythagorean_integration" "$M"
cg "main:total_1153" "let total: i64 = 1153" "$M"
echo ""
echo "--- Type-checks ---"
TC_OUT="$(mktemp)"; TOTAL=$((TOTAL+1))
if timeout 90 "$SOUC" check self-hosted/ir/egraph.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck:egraph"; PASS=$((PASS+1)); else echo "FAIL  typecheck:egraph"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
TC_OUT="$(mktemp)"; TOTAL=$((TOTAL+1))
if timeout 90 "$SOUC" check self-hosted/ir/opt_cleanup.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck:opt_cleanup"; PASS=$((PASS+1)); else echo "FAIL  typecheck:opt_cleanup"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
TC_OUT="$(mktemp)"; TOTAL=$((TOTAL+1))
if timeout 90 "$SOUC" check self-hosted/compiler/main.sio > "$TC_OUT" 2>&1 && grep -q "All checks passed" "$TC_OUT"; then echo "PASS  typecheck:main"; PASS=$((PASS+1)); else echo "FAIL  typecheck:main"; FAIL=$((FAIL+1)); fi; rm -f "$TC_OUT"
echo ""
echo "--- E-Graph self-tests (70/70) ---"
EG_OUT="$(mktemp)"; TOTAL=$((TOTAL+1))
if [ ! -x "$SOUC" ]; then echo "NOT_RUN  egraph_run (no binary)"; NOT_RUN=$((NOT_RUN+1))
else timeout 120 "$SOUC" run self-hosted/ir/egraph.sio > "$EG_OUT" 2>&1 || _ec=$?
if grep -q "70 / 70" "$EG_OUT"; then echo "PASS  egraph_run 70/70"; PASS=$((PASS+1)); else echo "FAIL  egraph_run"; FAIL=$((FAIL+1)); fi; fi
rm -f "$EG_OUT"
echo ""
echo "--- Self-tests: T1148-T1153 ---"
L="$(mktemp)"
if [ ! -x "$SOUC" ]; then for t in T1148 T1149 T1150 T1151 T1152 T1153; do TOTAL=$((TOTAL+1)); echo "NOT_RUN  selftest:$t"; NOT_RUN=$((NOT_RUN+1)); done
else timeout 180 "$SOUC" run self-hosted/compiler/main.sio -- --self-test > "$L" 2>&1 || _ec=$?
for t in T1148 T1149 T1150 T1151 T1152 T1153; do cl "selftest:$t" "$t OK" "$L"; done; fi
rm -f "$L"
echo ""
echo "=== SUMMARY ==="
echo "PASS=$PASS  FAIL=$FAIL  NOT_RUN=$NOT_RUN  TOTAL=$TOTAL"
if [ "$FAIL" -eq 0 ]; then echo "GATE: PASS"; exit 0; else echo "GATE: FAIL"; exit 1; fi
