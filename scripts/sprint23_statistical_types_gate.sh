#!/usr/bin/env bash
# Sprint 23 — Epistemic Statistical Core gate
# 48 cases: Track P (12) + Track I (12) + Track V (12) + Track X (8) + Regression (4)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOUC="${ROOT_DIR}/target/debug/souc"
TYPES="${ROOT_DIR}/self-hosted/check/types.sio"
EPISTEMIC="${ROOT_DIR}/self-hosted/check/epistemic.sio"
COMPAT="${ROOT_DIR}/self-hosted/check/compat.sio"
CHECK="${ROOT_DIR}/self-hosted/check/check.sio"

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

echo "=== Sprint 23 Epistemic Statistical Core Gate ==="
echo ""

echo "--- Track P: Probabilistic Types ---"

# P1: TypeKind declarations
check_grep "ir_ty_distribution_declared"    "$TYPES"     "TyDistribution"
check_grep "ir_ty_sample_declared"          "$TYPES"     "TySample"
check_grep "ir_ty_conditional_dist_declared" "$TYPES"    "TyConditionalDist"

# P2: Constructors
check_grep "ty_distribution_constructor"   "$TYPES"     "fn ty_distribution"
check_grep "ty_sample_constructor"         "$TYPES"     "fn ty_sample"
check_grep "ty_conditional_dist_constructor" "$TYPES"   "fn ty_conditional_dist"

# P3: Predicates
check_grep "is_distribution_predicate"     "$TYPES"     "fn is_distribution_type"
check_grep "is_sample_predicate"           "$TYPES"     "fn is_sample_type"
check_grep "is_conditional_dist_predicate" "$TYPES"     "fn is_conditional_dist_type"

# P4: Checker functions
check_grep "check_distribution_fn"         "$EPISTEMIC" "fn check_distribution_type"
check_grep "check_conjugate_update_fn"     "$EPISTEMIC" "fn check_conjugate_update"

# P5: Frontend test
check_souc "check_probabilistic_basic"     "${ROOT_DIR}/tests/frontend/probabilistic_types_basic.sio"

echo ""
echo "--- Track I: Information-Theoretic Types ---"

# I1: TypeKind declarations
check_grep "ir_ty_entropic_declared"       "$TYPES"     "TyEntropic"
check_grep "ir_ty_mutual_info_declared"    "$TYPES"     "TyMutualInfo"
check_grep "ir_ty_kl_bounded_declared"     "$TYPES"     "TyKLBounded"

# I2: Constructors
check_grep "ty_entropic_constructor"       "$TYPES"     "fn ty_entropic"
check_grep "ty_mutual_info_constructor"    "$TYPES"     "fn ty_mutual_info"
check_grep "ty_kl_bounded_constructor"     "$TYPES"     "fn ty_kl_bounded"

# I3: Predicates
check_grep "is_entropic_predicate"         "$TYPES"     "fn is_entropic_type"
check_grep "is_mutual_info_predicate"      "$TYPES"     "fn is_mutual_info_type"
check_grep "is_kl_bounded_predicate"       "$TYPES"     "fn is_kl_bounded_type"

# I4: Checker functions
check_grep "check_entropic_fn"             "$EPISTEMIC" "fn check_entropic_type"
check_grep "check_dpi_step_fn"             "$EPISTEMIC" "fn check_dpi_step"

# I5: Frontend test
check_souc "check_information_theory_basic" "${ROOT_DIR}/tests/frontend/information_theory_basic.sio"

echo ""
echo "--- Track V: Shape-Typed Linear Algebra ---"

# V1: TypeKind declarations
check_grep "ir_ty_vec_shaped_declared"     "$TYPES"     "TyVecShaped"
check_grep "ir_ty_matrix_shaped_declared"  "$TYPES"     "TyMatrixShaped"
check_grep "ir_ty_broadcastable_declared"  "$TYPES"     "TyBroadcastable"

# V2: Constructors
check_grep "ty_vec_shaped_constructor"     "$TYPES"     "fn ty_vec_shaped"
check_grep "ty_matrix_shaped_constructor"  "$TYPES"     "fn ty_matrix_shaped"
check_grep "ty_broadcastable_constructor"  "$TYPES"     "fn ty_broadcastable"

# V3: Predicates
check_grep "is_vec_shaped_predicate"       "$TYPES"     "fn is_vec_shaped_type"
check_grep "is_matrix_shaped_predicate"    "$TYPES"     "fn is_matrix_shaped_type"
check_grep "is_broadcastable_predicate"    "$TYPES"     "fn is_broadcastable_type"

# V4: Checker functions
check_grep "check_matmul_dims_fn"          "$EPISTEMIC" "fn check_matmul_dims"
check_grep "check_decomp_shape_fn"         "$EPISTEMIC" "fn check_decomp_shape"

# V5: Frontend test
check_souc "check_shaped_linalg_basic"     "${ROOT_DIR}/tests/frontend/shaped_linalg_basic.sio"

echo ""
echo "--- Track X: Variational Inference Types ---"

# X1: TypeKind declarations
check_grep "ir_ty_elbo_declared"                  "$TYPES"     "TyELBO"
check_grep "ir_ty_variational_family_declared"    "$TYPES"     "TyVariationalFamily"

# X2: Constructors
check_grep "ty_elbo_constructor"                  "$TYPES"     "fn ty_elbo"
check_grep "ty_variational_family_constructor"    "$TYPES"     "fn ty_variational_family"

# X3: Predicates
check_grep "is_elbo_predicate"                    "$TYPES"     "fn is_elbo_type"
check_grep "is_variational_family_predicate"      "$TYPES"     "fn is_variational_family_type"

# X4: Bridge function
check_grep "vi_to_contest_fn"                     "$EPISTEMIC" "fn vi_to_contest"

# X5: Frontend test
check_souc "check_variational_inference_basic"    "${ROOT_DIR}/tests/frontend/variational_inference_basic.sio"

echo ""
echo "--- Regression ---"

# R1: Sprint 22 gate (check artifact)
if grep -q '"status": "pass"' "${ROOT_DIR}/artifacts/sprint22/responsible_ai_gate.v1.json" 2>/dev/null; then
    run_case "sprint22_responsible_ai_gate_passes" "pass"
else
    run_case "sprint22_responsible_ai_gate_passes" "sprint22 artifact missing or status != pass"
fi

# R2: Sprint 21 gate (check artifact)
if grep -q '"status": "pass"' "${ROOT_DIR}/artifacts/sprint21/dp_gate.v1.json" 2>/dev/null; then
    run_case "sprint21_dp_gate_passes" "pass"
else
    run_case "sprint21_dp_gate_passes" "sprint21 artifact missing or status != pass"
fi

# R3: Frontend regression (5 sprint23 tests all pass)
s23_pass=0
for f in probabilistic_types_basic information_theory_basic shaped_linalg_basic variational_inference_basic; do
    if "$SOUC" check "${ROOT_DIR}/tests/frontend/${f}.sio" 2>/dev/null; then
        s23_pass=$((s23_pass+1))
    fi
done
if [ "$s23_pass" -eq 4 ]; then
    run_case "sprint23_frontend_regression" "pass"
else
    run_case "sprint23_frontend_regression" "${s23_pass}/4 tests pass"
fi

# R4: Stdlib statistical core exists
if [ -f "${ROOT_DIR}/stdlib/epistemic/statistical_core.sio" ]; then
    run_case "stdlib_epistemic_statistical_core_exists" "pass"
else
    run_case "stdlib_epistemic_statistical_core_exists" "stdlib/epistemic/statistical_core.sio missing"
fi

echo ""
echo "=== Results: ${pass} pass, ${fail} fail / $((pass+fail)) total ==="

# Generate artifact
mkdir -p "${ROOT_DIR}/artifacts/sprint23"
ARTIFACT="${ROOT_DIR}/artifacts/sprint23/statistical_types_gate.v1.json"

STATUS="pass"
[ "$fail" -gt 0 ] && STATUS="fail"

{
    echo "{"
    echo "  \"gate\": \"sprint23_statistical_types\","
    echo "  \"sprint\": 23,"
    echo "  \"version\": \"v1\","
    echo "  \"status\": \"${STATUS}\","
    echo "  \"pass\": ${pass},"
    echo "  \"fail\": ${fail},"
    echo "  \"total\": $((pass+fail)),"
    echo "  \"tracks\": {\"P\": \"probabilistic\", \"I\": \"information_theory\", \"V\": \"shape_linalg\", \"X\": \"variational_inference\"},"
    echo "  \"typekind_additions\": 11,"
    echo "  \"error_codes\": \"E083-E096\","
    echo "  \"references\": ["
    echo "    \"Gelman et al. (2013) BDA3\","
    echo "    \"Cover & Thomas (2006) Elements of Information Theory\","
    echo "    \"Henriksen et al. (2017) PLDI Futhark\","
    echo "    \"Blei et al. (2017) JASA Variational Inference Review\","
    echo "    \"Kingma & Welling (2014) VAE\""
    echo "  ],"
    echo "  \"cases\": ["
    local_ifs="${IFS}"
    IFS=','
    first=1
    for r in "${results[@]}"; do
        if [ "$first" -eq 1 ]; then
            echo "    ${r}"
            first=0
        else
            echo "    ,${r}"
        fi
    done
    IFS="${local_ifs}"
    echo "  ]"
    echo "}"
} > "$ARTIFACT"

echo "Artifact: ${ARTIFACT}"

[ "$fail" -eq 0 ]
