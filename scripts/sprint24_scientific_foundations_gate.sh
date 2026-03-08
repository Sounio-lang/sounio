#!/usr/bin/env bash
# Sprint 24 — Scientific Foundations gate
# 52 cases: Track A (12) + Track B (12) + Track C (10) + Track D (10) + Bridge (4) + Regression (4)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOUC="${ROOT_DIR}/target/debug/souc"
TYPES="${ROOT_DIR}/self-hosted/check/types.sio"
EPISTEMIC="${ROOT_DIR}/self-hosted/check/epistemic.sio"

pass=0
fail=0
results=()

run_case() {
    local name="$1"
    local result="$2"
    if [ "$result" = "pass" ]; then
        pass=$((pass+1))
        results+=("{\"name\":\"${name}\",\"status\":\"pass\"}")
    else
        fail=$((fail+1))
        results+=("{\"name\":\"${name}\",\"status\":\"fail\",\"detail\":\"${result}\"}")
        echo "  FAIL: ${name}: ${result}"
    fi
}

check_grep() {
    local name="$1"
    local file="$2"
    local pattern="$3"
    if grep -q "$pattern" "$file" 2>/dev/null; then
        run_case "$name" "pass"
    else
        run_case "$name" "pattern '${pattern}' not found in ${file##*/}"
    fi
}

check_souc() {
    local name="$1"
    local file="$2"
    if "$SOUC" check "$file" 2>/dev/null; then
        run_case "$name" "pass"
    else
        local err
        err=$("$SOUC" check "$file" 2>&1 | head -3 | tr '\n' ' ')
        run_case "$name" "souc check failed: ${err}"
    fi
}

echo "=== Sprint 24 Scientific Foundations Gate ==="
echo ""

echo "--- Track A: Differentiable Programming Types ---"
check_grep "ir_ty_differentiable"        "$TYPES"     "TyDifferentiable"
check_grep "ir_ty_gradient"              "$TYPES"     "TyGradient"
check_grep "ir_ty_jacobian"              "$TYPES"     "TyJacobian"
check_grep "ty_differentiable_ctor"      "$TYPES"     "fn ty_differentiable"
check_grep "ty_gradient_ctor"            "$TYPES"     "fn ty_gradient"
check_grep "ty_jacobian_ctor"            "$TYPES"     "fn ty_jacobian"
check_grep "is_differentiable_pred"      "$TYPES"     "fn is_differentiable_type"
check_grep "is_gradient_pred"            "$TYPES"     "fn is_gradient_type"
check_grep "is_jacobian_pred"            "$TYPES"     "fn is_jacobian_type"
check_grep "check_differentiable_fn"     "$EPISTEMIC" "fn check_differentiable_type"
check_grep "check_ad_mode_compat_fn"     "$EPISTEMIC" "fn check_ad_mode_compat"
check_souc "check_differentiable_basic"  "${ROOT_DIR}/tests/frontend/differentiable_types_basic.sio"

echo ""
echo "--- Track B: Stochastic Process Types ---"
check_grep "ir_ty_markov_chain"          "$TYPES"     "TyMarkovChain"
check_grep "ir_ty_sde"                   "$TYPES"     "TySDE"
check_grep "ir_ty_martingale"            "$TYPES"     "TyMartingale"
check_grep "ir_ty_stationary_dist"       "$TYPES"     "TyStationaryDist"
check_grep "ty_markov_chain_ctor"        "$TYPES"     "fn ty_markov_chain"
check_grep "ty_sde_ctor"                 "$TYPES"     "fn ty_sde"
check_grep "ty_martingale_ctor"          "$TYPES"     "fn ty_martingale"
check_grep "ty_stationary_dist_ctor"     "$TYPES"     "fn ty_stationary_dist"
check_grep "is_markov_chain_pred"        "$TYPES"     "fn is_markov_chain_type"
check_grep "is_sde_pred"                 "$TYPES"     "fn is_sde_type"
check_grep "is_martingale_pred"          "$TYPES"     "fn is_martingale_type"
check_souc "check_stochastic_basic"      "${ROOT_DIR}/tests/frontend/stochastic_process_basic.sio"

echo ""
echo "--- Track C: Complexity/Resource Types ---"
check_grep "ir_ty_bigO"                  "$TYPES"     "TyBigO"
check_grep "ir_ty_amortized"             "$TYPES"     "TyAmortized"
check_grep "ty_bigO_ctor"                "$TYPES"     "fn ty_bigO"
check_grep "ty_amortized_ctor"           "$TYPES"     "fn ty_amortized"
check_grep "is_bigO_pred"                "$TYPES"     "fn is_bigO_type"
check_grep "is_amortized_pred"           "$TYPES"     "fn is_amortized_type"
check_grep "check_bigO_fn"               "$EPISTEMIC" "fn check_bigO_type"
check_grep "check_amortized_fn"          "$EPISTEMIC" "fn check_amortized_type"
check_grep "matmul_complexity_bridge"    "$EPISTEMIC" "fn matmul_complexity"
check_souc "check_complexity_basic"      "${ROOT_DIR}/tests/frontend/complexity_types_basic.sio"

echo ""
echo "--- Track D: Proof Certificate Types ---"
check_grep "ir_ty_proof"                 "$TYPES"     "TyProof"
check_grep "ir_ty_lemma"                 "$TYPES"     "TyLemma"
check_grep "ir_ty_axiom"                 "$TYPES"     "TyAxiom"
check_grep "ty_proof_ctor"               "$TYPES"     "fn ty_proof"
check_grep "ty_lemma_ctor"               "$TYPES"     "fn ty_lemma"
check_grep "ty_axiom_ctor"               "$TYPES"     "fn ty_axiom"
check_grep "is_proof_pred"               "$TYPES"     "fn is_proof_type"
check_grep "check_proof_completeness_fn" "$EPISTEMIC" "fn check_proof_completeness"
check_grep "proof_to_validated_fn"       "$EPISTEMIC" "fn proof_to_validated"
check_souc "check_proof_basic"           "${ROOT_DIR}/tests/frontend/proof_types_basic.sio"

echo ""
echo "--- Bridge Functions ---"
check_grep "langevin_from_gradient_fn"       "$EPISTEMIC" "fn langevin_from_gradient"
check_grep "sde_stationary_to_dist_fn"       "$EPISTEMIC" "fn sde_stationary_to_dist"
check_grep "jacobian_from_matrix_shape_fn"   "$EPISTEMIC" "fn jacobian_from_matrix_shape"
if [ -f "${ROOT_DIR}/stdlib/epistemic/gradient_bridge.sio" ]; then
    run_case "gradient_bridge_stdlib_exists" "pass"
else
    run_case "gradient_bridge_stdlib_exists" "stdlib/epistemic/gradient_bridge.sio missing"
fi

echo ""
echo "--- Regression ---"
if grep -q '"status": "pass"' "${ROOT_DIR}/artifacts/sprint23/statistical_types_gate.v1.json" 2>/dev/null; then
    run_case "sprint23_gate_passes" "pass"
else
    run_case "sprint23_gate_passes" "sprint23 artifact missing or not pass"
fi
if grep -q '"status": "pass"' "${ROOT_DIR}/artifacts/sprint22/responsible_ai_gate.v1.json" 2>/dev/null; then
    run_case "sprint22_gate_passes" "pass"
else
    run_case "sprint22_gate_passes" "sprint22 artifact missing or not pass"
fi
s24_pass=0
for f in differentiable_types_basic stochastic_process_basic complexity_types_basic proof_types_basic; do
    if "$SOUC" check "${ROOT_DIR}/tests/frontend/${f}.sio" 2>/dev/null; then
        s24_pass=$((s24_pass+1))
    fi
done
if [ "$s24_pass" -eq 4 ]; then
    run_case "sprint24_frontend_regression" "pass"
else
    run_case "sprint24_frontend_regression" "${s24_pass}/4 tests pass"
fi
if [ -f "${ROOT_DIR}/stdlib/epistemic/gradient_bridge.sio" ]; then
    run_case "stdlib_gradient_bridge_exists" "pass"
else
    run_case "stdlib_gradient_bridge_exists" "missing"
fi

echo ""
echo "=== Results: ${pass} pass, ${fail} fail / $((pass+fail)) total ==="

mkdir -p "${ROOT_DIR}/artifacts/sprint24"
ARTIFACT="${ROOT_DIR}/artifacts/sprint24/scientific_foundations_gate.v1.json"
STATUS="pass"
[ "$fail" -gt 0 ] && STATUS="fail"

{
    echo "{"
    echo "  \"gate\": \"sprint24_scientific_foundations\","
    echo "  \"sprint\": 24,"
    echo "  \"version\": \"v1\","
    echo "  \"status\": \"${STATUS}\","
    echo "  \"pass\": ${pass},"
    echo "  \"fail\": ${fail},"
    echo "  \"total\": $((pass+fail)),"
    echo "  \"tracks\": {\"A\": \"differentiable_programming\", \"B\": \"stochastic_processes\", \"C\": \"complexity_resource\", \"D\": \"proof_certificates\", \"moonshot\": \"sgd_langevin_gibbs_bridge\"},"
    echo "  \"typekind_additions\": 12,"
    echo "  \"typekind_total\": 86,"
    echo "  \"error_codes\": \"E100-E113\","
    echo "  \"references\": ["
    echo "    \"Elliott (2018) ICFP Simple Essence of Automatic Differentiation\","
    echo "    \"Huot et al. (2020) LICS Correctness of Automatic Differentiation\","
    echo "    \"Pearlmutter & Siskind (2008) POPL Reverse-Mode AD\","
    echo "    \"Oksendal (2003) Stochastic Differential Equations\","
    echo "    \"Doob (1953) Stochastic Processes\","
    echo "    \"Aguirre et al. (2018) POPL Denotational Semantics for Probabilistic Programs\","
    echo "    \"Hoffmann & Hofmann (2010) POPL Amortized Resource Analysis\","
    echo "    \"Hoffmann et al. (2012) POPL Multivariate Amortized Resource Analysis\","
    echo "    \"Wadler (2015) CACM Propositions as Types\","
    echo "    \"Martin-Lof (1984) Intuitionistic Type Theory\""
    echo "  ],"
    echo "  \"cases\": ["
    first=1
    for r in "${results[@]}"; do
        if [ "$first" -eq 1 ]; then
            echo "    ${r}"
            first=0
        else
            echo "    ,${r}"
        fi
    done
    echo "  ]"
    echo "}"
} > "$ARTIFACT"

echo "Artifact: ${ARTIFACT}"
[ "$fail" -eq 0 ]
