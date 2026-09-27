#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d /tmp/statistical-coverage-binding-d9-gate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    echo "[statistical-coverage-empirical-binding-d9] FAIL: $*" >&2
    exit 1
}

require_line() {
    local expected="$1"
    local log="$2"
    grep -Fq "$expected" "$log" || {
        cat "$log" >&2
        fail "missing exact receipt: $expected"
    }
}

if [[ -n "${SOUNIO_SOUC_ENGINE:-}" && "${SOUNIO_SOUC_ENGINE}" != "madaros" ]]; then
    fail "SOUNIO_SOUC_ENGINE requests a non-Madaros fallback"
fi

bin/souc --version >"$TMP_DIR/version.log" 2>&1 || {
    cat "$TMP_DIR/version.log" >&2
    fail "canonical bin/souc did not report its version"
}
grep -Fq "Madaros" "$TMP_DIR/version.log" || {
    cat "$TMP_DIR/version.log" >&2
    fail "canonical bin/souc did not resolve to Madaros"
}

kernel="stdlib/epistemic/proof_carrying_statistical_coverage_empirical_binding.sio"
grep -Fq 'use epistemic::proof_carrying_path_conditioned_identification::*' \
    "$kernel" || fail "D8 import must use the stable wildcard path"

check_sources=(
    "$kernel"
    stdlib/ontology/statistical_coverage_empirical_binding.sio
    tests/run-pass/clinical_statistical_coverage_empirical_binding_witness.sio
)
for source in "${check_sources[@]}"; do
    log="$TMP_DIR/$(basename "$source").check.log"
    if ! bin/souc check "$source" >"$log" 2>&1; then
        cat "$log" >&2
        fail "$source did not typecheck"
    fi
    grep -q "check: OK" "$log" || fail "$source lacked the check receipt"
done

if ! bin/souc run \
    tests/run-pass/clinical_statistical_coverage_empirical_binding_native_witness.sio \
    >"$TMP_DIR/native.log" 2>&1; then
    cat "$TMP_DIR/native.log" >&2
    fail "native D9 witness did not execute"
fi
require_line "D9-W0 target_mask=3 procedure_masks=3,1,2,7 outcomes=4 total_mass=12" "$TMP_DIR/native.log"
require_line "D9-W1 design_a=5,1,1,5 whole=10/12 memberwise=11/12 permille=833 remainder=4 adequate=true" "$TMP_DIR/native.log"
require_line "D9-W2 design_b=1,5,5,1 whole=2/12 memberwise=7/12 permille=166 remainder=8 adequate=false" "$TMP_DIR/native.log"
require_line "D9-W3 same_realized_region=3 different_procedure_coverage=true identified_set_is_confidence_region=false" "$TMP_DIR/native.log"
require_line "D9-W4 total_designs=455 positive=165 positivity_failures=290 support_histogram=4,66,220,165" "$TMP_DIR/native.log"
require_line "D9-W5 positive_coverage_histogram=2:9,3:16,4:21,5:24,6:25,7:24,8:21,9:16,10:9 adequate_at_3/4=25" "$TMP_DIR/native.log"
require_line "D9-W6 marginal=9/10 rare_group=0/1 selected=0/1 marginal_is_selected=false" "$TMP_DIR/native.log"
require_line "D9-W7 eligibility_combinations=8 eligible=1 abstain=7 failure_masks=1,2,4" "$TMP_DIR/native.log"
require_line "D9-W8 same_table_bytes=true same_numeric_region=true lineage_substitutable=false integrity=false custody=false" "$TMP_DIR/native.log"
require_line "D9-W9 predictive_set_is_confidence_region=false patient_state=0 clinical_authority=0" "$TMP_DIR/native.log"
require_line "D9-E0 rows=1885 eligible=1877 semeron_excluded=8 development=1125 calibration=377 evaluation=375" "$TMP_DIR/native.log"
require_line "D9-E1 calibration_covered=286/377 permille=758 remainder=234 adequate=true" "$TMP_DIR/native.log"
require_line "D9-E2 evaluation_covered=265/375 permille=706 remainder=250 adequate=false" "$TMP_DIR/native.log"
require_line "D9-E3 support_compatible=true empirical_binding=false abstention_reason_mask=230" "$TMP_DIR/native.log"
require_line "D9-E4 custody=false sealed=false patient_state=false clinical_authority=false" "$TMP_DIR/native.log"
require_line "STATISTICAL COVERAGE AND EMPIRICAL BINDING D9 PASS" "$TMP_DIR/native.log"

if ! bin/souc run tests/run-pass/ontology_statistical_coverage_empirical_binding_types.sio \
    >"$TMP_DIR/ontology.log" 2>&1; then
    cat "$TMP_DIR/ontology.log" >&2
    fail "parallel D9 ontology witness did not execute"
fi
require_line "statistical coverage empirical binding parallel ontology OK" \
    "$TMP_DIR/ontology.log"

oracle="scripts/research/proof_carrying_statistical_coverage_empirical_binding_oracle.py"
if ! python3 "$oracle" >"$TMP_DIR/oracle.log" 2>&1; then
    cat "$TMP_DIR/oracle.log" >&2
    fail "independent D9 oracle failed"
fi
require_line "ORACLE_D9_W0 target_mask=3 procedure_masks=3,1,2,7 outcomes=4 total_mass=12" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_W1 design_a=5,1,1,5 whole=10/12 memberwise=11/12 permille=833 remainder=4 adequate=true" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_W2 design_b=1,5,5,1 whole=2/12 memberwise=7/12 permille=166 remainder=8 adequate=false" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_W3 same_realized_region=3 different_procedure_coverage=true identified_set_is_confidence_region=false" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_W4 total_designs=455 positive=165 positivity_failures=290 support_histogram=4,66,220,165" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_W5 positive_coverage_histogram=2:9,3:16,4:21,5:24,6:25,7:24,8:21,9:16,10:9 adequate_at_3/4=25" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_W6 marginal=9/10 rare_group=0/1 selected=0/1 marginal_is_selected=false" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_W7 eligibility_combinations=8 eligible=1 abstain=7 failure_masks=1,2,4" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_W8 same_table_bytes=true same_numeric_region=true lineage_substitutable=false integrity=false custody=false" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_W9 predictive_set_is_confidence_region=false patient_state=0 clinical_authority=0" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_E0 rows=1885 eligible=1877 semeron_excluded=8 development=1125 calibration=377 evaluation=375" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_E1 calibration_covered=286/377 permille=758 remainder=234 adequate=true recent=58 set_masks=1:122,2:109,3:146" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_E2 evaluation_covered=265/375 permille=706 remainder=250 adequate=false recent=58 set_masks=1:124,2:136,3:115" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_E3 support_compatible=true calibration_bands=3 evaluation_bands=3" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_E4 empirical_binding=false abstention_reason_mask=230 custody=false sealed=false patient_state=false clinical_authority=false" "$TMP_DIR/oracle.log"
require_line "ORACLE_D9_E5 dataset_sha256=90b8cf500b07ad455baf9fe1dc519998c75a1df6d87f6bd7069176f0826ea8c1 protocol_sha256=b443191335bda3eb0eaa3bd8fee47a30cebc16080ad9e0862d85d9734fee4a1e" "$TMP_DIR/oracle.log"
require_line "STATISTICAL COVERAGE AND EMPIRICAL BINDING D9 ORACLE PASS" "$TMP_DIR/oracle.log"

dataset="tests/fixtures/psychiatric_d9/uci_drug_consumption_373.data"
manifest="tests/fixtures/psychiatric_d9/dataset_manifest.v1.json"
protocol="tests/fixtures/psychiatric_d9/evaluation_protocol.v1.json"
cp "$dataset" "$TMP_DIR/tampered.data"
printf '\n' >>"$TMP_DIR/tampered.data"
if python3 "$oracle" --dataset "$TMP_DIR/tampered.data" \
    --manifest "$manifest" --protocol "$protocol" \
    >"$TMP_DIR/data-tamper.log" 2>&1; then
    fail "dataset byte tamper unexpectedly passed"
fi
grep -Fq "dataset SHA-256 mismatch" "$TMP_DIR/data-tamper.log" || {
    cat "$TMP_DIR/data-tamper.log" >&2
    fail "dataset tamper lacked its SHA-256 refusal"
}

cp "$protocol" "$TMP_DIR/tampered-protocol.json"
printf ' ' >>"$TMP_DIR/tampered-protocol.json"
if python3 "$oracle" --dataset "$dataset" --manifest "$manifest" \
    --protocol "$TMP_DIR/tampered-protocol.json" \
    >"$TMP_DIR/protocol-tamper.log" 2>&1; then
    fail "protocol byte tamper unexpectedly passed"
fi
grep -Fq "protocol SHA-256 mismatch" "$TMP_DIR/protocol-tamper.log" || {
    cat "$TMP_DIR/protocol-tamper.log" >&2
    fail "protocol tamper lacked its SHA-256 refusal"
}

expect_type_rejection() {
    local source="$1"
    local expected="$2"
    local found="$3"
    local log="$TMP_DIR/$(basename "$source").reject.log"
    if bin/souc check "$source" >"$log" 2>&1; then
        cat "$log" >&2
        fail "$source unexpectedly typechecked"
    fi
    grep -q "expected $expected" "$log" || {
        cat "$log" >&2
        fail "$source lacked expected type $expected"
    }
    grep -q "found $found" "$log" || {
        cat "$log" >&2
        fail "$source lacked found type $found"
    }
}

expect_private_constructor_rejection() {
    local source="$1"
    local log="$TMP_DIR/$(basename "$source").private.log"
    if bin/souc check "$source" >"$log" 2>&1; then
        cat "$log" >&2
        fail "$source unexpectedly constructed a reserved authority receipt"
    fi
    grep -q 'error\[E176' "$log" || {
        cat "$log" >&2
        fail "$source lacked E176 private-constructor refusal"
    }
    grep -Fq "struct constructor is private in its defining module" "$log" || {
        cat "$log" >&2
        fail "$source lacked the private-constructor diagnostic"
    }
}

expect_type_rejection tests/compile-fail/clinical_d9_identified_set_cannot_replace_confidence_region_d9.sio D9ConfidenceRegionReceipt ExactABIdentifiedSetReceipt
expect_type_rejection tests/compile-fail/clinical_d9_confidence_region_cannot_replace_identified_set_d9.sio ExactABIdentifiedSetReceipt D9ConfidenceRegionReceipt
expect_type_rejection tests/compile-fail/clinical_d9_confidence_region_cannot_replace_predictive_set_d9.sio D9PredictiveSetReceipt D9ConfidenceRegionReceipt
expect_type_rejection tests/compile-fail/clinical_d9_predictive_set_cannot_replace_confidence_region_d9.sio D9ConfidenceRegionReceipt D9PredictiveSetReceipt
expect_type_rejection tests/compile-fail/clinical_d9_whole_set_cannot_replace_memberwise_coverage_d9.sio D9MemberwiseFiniteCoverageAReceipt D9WholeSetFiniteCoverageAReceipt
expect_type_rejection tests/compile-fail/clinical_d9_memberwise_cannot_replace_whole_set_coverage_d9.sio D9WholeSetFiniteCoverageAReceipt D9MemberwiseFiniteCoverageAReceipt
expect_type_rejection tests/compile-fail/clinical_d9_adequate_cannot_replace_insufficient_coverage_d9.sio D9InsufficientCoverageReceipt D9AdequateCoverageReceipt
expect_type_rejection tests/compile-fail/clinical_d9_design_a_cannot_replace_design_b_d9.sio D9SamplingDesignBReceipt D9SamplingDesignAReceipt
expect_type_rejection tests/compile-fail/clinical_d9_region_a_cannot_replace_region_b_d9.sio D9AlternateConfidenceRegionReceipt D9ConfidenceRegionReceipt
expect_type_rejection tests/compile-fail/clinical_d9_marginal_cannot_replace_selection_conditioned_failure_d9.sio D9SelectionConditionedCoverageFailureReceipt D9MarginalCoverageReceipt
expect_type_rejection tests/compile-fail/clinical_d9_selection_failure_cannot_issue_negative_prediction_d9.sio D9NegativePredictionReceipt D9SelectionConditionedCoverageFailureReceipt
expect_type_rejection tests/compile-fail/clinical_d9_selection_abstention_cannot_escalate_d9.sio D9ClinicalEscalationReceipt D9SelectionAbstentionReceipt
expect_type_rejection tests/compile-fail/clinical_d9_lineage_cannot_claim_integrity_d9.sio D9IntegrityReceipt D9DatasetProvenanceAReceipt
expect_type_rejection tests/compile-fail/clinical_d9_lineage_cannot_claim_authenticity_d9.sio D9AuthenticityReceipt D9DatasetProvenanceAReceipt
expect_type_rejection tests/compile-fail/clinical_d9_provenance_mismatch_cannot_replace_matched_d9.sio D9MatchedProvenanceReceipt D9ProvenanceMismatchReceipt
expect_type_rejection tests/compile-fail/clinical_d9_matched_provenance_cannot_claim_custody_d9.sio D9ExternalDataCustodyReceipt D9MatchedProvenanceReceipt
expect_type_rejection tests/compile-fail/clinical_d9_statistical_calibration_cannot_replace_instrument_calibration_d9.sio D9InstrumentCalibrationReceipt D9FiniteCoverageCalibrationReceipt
expect_type_rejection tests/compile-fail/clinical_d9_instrument_calibration_cannot_replace_statistical_calibration_d9.sio D9FiniteCoverageCalibrationReceipt D9InstrumentCalibrationReceipt
expect_type_rejection tests/compile-fail/clinical_d9_sampling_positivity_cannot_claim_causal_effect_d9.sio D9CausalTreatmentEffectReceipt D9SamplingPositivityReceipt
expect_type_rejection tests/compile-fail/clinical_d9_positivity_failure_cannot_replace_positivity_d9.sio D9SamplingPositivityReceipt D9SamplingPositivityFailureReceipt
expect_type_rejection tests/compile-fail/clinical_d9_calibration_failure_cannot_replace_binding_d9.sio D9DeclaredContextBindingReceipt D9CoverageCalibrationFailureReceipt
expect_type_rejection tests/compile-fail/clinical_d9_incompatibility_cannot_replace_binding_d9.sio D9DeclaredContextBindingReceipt D9InstrumentPopulationMismatchReceipt
expect_type_rejection tests/compile-fail/clinical_d9_declared_binding_cannot_replace_empirical_binding_d9.sio D9EmpiricalBindingReceipt D9DeclaredContextBindingReceipt
expect_type_rejection tests/compile-fail/clinical_d9_declared_binding_cannot_replace_patient_state_d9.sio D9PatientStateReceipt D9DeclaredContextBindingReceipt
expect_type_rejection tests/compile-fail/clinical_d9_declared_binding_cannot_authorize_action_d9.sio D9ClinicalActionAuthorityReceipt D9DeclaredContextBindingReceipt
expect_type_rejection tests/compile-fail/clinical_d9_external_evaluation_cannot_replace_empirical_binding_d9.sio D9EmpiricalBindingReceipt D9ExternalEvaluationReceipt
expect_type_rejection tests/compile-fail/clinical_d9_external_abstention_cannot_replace_empirical_binding_d9.sio D9EmpiricalBindingReceipt D9ExternalBindingAbstentionReceipt
expect_type_rejection tests/compile-fail/clinical_d9_external_abstention_cannot_replace_patient_state_d9.sio D9PatientStateReceipt D9ExternalBindingAbstentionReceipt
expect_type_rejection tests/compile-fail/clinical_d9_external_abstention_cannot_authorize_action_d9.sio D9ClinicalActionAuthorityReceipt D9ExternalBindingAbstentionReceipt
expect_type_rejection tests/compile-fail/clinical_d9_predictive_set_cannot_replace_patient_state_d9.sio D9PatientStateReceipt D9PredictiveSetReceipt
expect_type_rejection tests/compile-fail/clinical_d9_causal_ambiguity_cannot_replace_causal_effect_d9.sio D9CausalTreatmentEffectReceipt D9CausalActionAmbiguityReceipt
expect_type_rejection tests/compile-fail/clinical_d9_summary_cannot_replace_patient_state_d9.sio D9PatientStateReceipt D9StatisticalCoverageSummaryReceipt
expect_type_rejection tests/compile-fail/clinical_d9_summary_cannot_authorize_action_d9.sio D9ClinicalActionAuthorityReceipt D9StatisticalCoverageSummaryReceipt

expect_private_constructor_rejection tests/compile-fail/clinical_d9_cannot_construct_empirical_binding_d9.sio
expect_private_constructor_rejection tests/compile-fail/clinical_d9_cannot_construct_patient_state_d9.sio
expect_private_constructor_rejection tests/compile-fail/clinical_d9_cannot_construct_clinical_action_authority_d9.sio

expect_type_rejection tests/compile-fail/ontology_d9_exact_identified_set_cannot_replace_confidence_region_d9.sio ConfidenceRegion ExactIdentifiedSet
expect_type_rejection tests/compile-fail/ontology_d9_confidence_region_cannot_replace_predictive_set_d9.sio PredictiveSet ConfidenceRegion
expect_type_rejection tests/compile-fail/ontology_d9_whole_set_cannot_replace_memberwise_coverage_d9.sio MemberwiseCoverage WholeSetCoverage
expect_type_rejection tests/compile-fail/ontology_d9_marginal_cannot_replace_selection_conditioned_coverage_d9.sio SelectionConditionedCoverage MarginalCoverage
expect_type_rejection tests/compile-fail/ontology_d9_declared_lineage_cannot_replace_integrity_d9.sio IntegrityEvidence DeclaredLineage
expect_type_rejection tests/compile-fail/ontology_d9_declared_lineage_cannot_replace_custody_d9.sio ExternalDataCustody DeclaredLineage
expect_type_rejection tests/compile-fail/ontology_d9_statistical_calibration_cannot_replace_instrument_calibration_d9.sio InstrumentCalibration StatisticalCoverageCalibration
expect_type_rejection tests/compile-fail/ontology_d9_sampling_positivity_cannot_replace_causal_effect_d9.sio CausalTreatmentEffect SamplingPositivity
expect_type_rejection tests/compile-fail/ontology_d9_declared_binding_cannot_replace_empirical_binding_d9.sio EmpiricalBinding DeclaredContextBinding
expect_type_rejection tests/compile-fail/ontology_d9_external_abstention_cannot_replace_empirical_binding_d9.sio EmpiricalBinding ExternalBindingAbstention
expect_type_rejection tests/compile-fail/ontology_d9_empirical_binding_cannot_replace_patient_state_d9.sio PatientState EmpiricalBinding
expect_type_rejection tests/compile-fail/ontology_d9_patient_state_cannot_replace_clinical_authority_d9.sio ClinicalActionAuthority PatientState
expect_type_rejection tests/compile-fail/ontology_d9_association_cannot_replace_causal_effect_d9.sio CausalTreatmentEffect AssociationEvidence
expect_type_rejection tests/compile-fail/ontology_d9_predictive_set_cannot_replace_patient_state_d9.sio PatientState PredictiveSet
expect_type_rejection tests/compile-fail/ontology_d9_confidence_region_cannot_authorize_action_d9.sio ClinicalActionAuthority ConfidenceRegion

negative_count="$(find tests/compile-fail -maxdepth 1 -name '*_d9.sio' | wc -l | tr -d ' ')"
[[ "$negative_count" == "51" ]] || fail "D9 negative matrix is not exactly 51 files"

d9_count="$(awk -F '\t' '$1 == "SOUNIO-PROOF-CARRYING-STATISTICAL-COVERAGE-EMPIRICAL-BINDING" { count++ } END { print count + 0 }' docs/internal/concepts/registry.tsv)"
[[ "$d9_count" == "1" ]] || fail "D9 concept row is not exactly one"
grep -Fqx $'SOUNIO-PROOF-CARRYING-STATISTICAL-COVERAGE-EMPIRICAL-BINDING\texecutable\tfounder\tdocs/internal/concepts/proof-carrying-statistical-coverage-empirical-binding.md\tstdlib/epistemic/proof_carrying_statistical_coverage_empirical_binding.sio\texternal-custody-and-sealed-validation' \
    docs/internal/concepts/registry.tsv || fail "D9 concept row is not exact"

required_bindings=(
    $'docs/internal/concepts/proof-carrying-statistical-coverage-empirical-binding.md\tconcept-contract'
    $'docs/research/proof_carrying_statistical_coverage_empirical_binding_d9_2026-07-19.md\tresearch-specification'
    $'docs/superpowers/specs/2026-07-19-proof-carrying-statistical-coverage-empirical-binding-d9-design.md\tdesign-specification'
    $'docs/handoff/d9_current_main_d4_ast_closure_regression_2026-07-19.md\tintegration-blocker'
    $'stdlib/epistemic/proof_carrying_statistical_coverage_empirical_binding.sio\tcanonical-model-kernel'
    $'stdlib/ontology/statistical_coverage_empirical_binding.sio\tparallel-ontology-boundary'
    $'tests/fixtures/psychiatric_d9/*\texternal-candidate-fixture'
    $'tests/run-pass/clinical_statistical_coverage_empirical_binding_*\tpositive-evidence'
    $'tests/run-pass/ontology_statistical_coverage_empirical_binding_types.sio\tparallel-ontology-evidence'
    $'tests/compile-fail/clinical_d9_*_d9.sio\tnegative-evidence'
    $'tests/compile-fail/ontology_d9_*_d9.sio\tnegative-evidence'
    $'scripts/research/proof_carrying_statistical_coverage_empirical_binding_oracle.py\tindependent-oracle'
    $'scripts/ci/proof_carrying_statistical_coverage_empirical_binding_gate.sh\tacceptance-gate'
)
for binding in "${required_bindings[@]}"; do
    grep -Fqx $'SOUNIO-PROOF-CARRYING-STATISTICAL-COVERAGE-EMPIRICAL-BINDING\t'"$binding" \
        docs/internal/concepts/bindings.tsv || fail "missing D9 binding: $binding"
done

grep -Fq "coverage belongs to a procedure" \
    docs/internal/concepts/proof-carrying-statistical-coverage-empirical-binding.md || \
    fail "concept contract lacks the procedure-level coverage boundary"
grep -Fq "does not bind a patient state" \
    docs/internal/concepts/proof-carrying-statistical-coverage-empirical-binding.md || \
    fail "concept contract lacks the patient-state boundary"
grep -Fq "No positive constructor" \
    docs/internal/concepts/proof-carrying-statistical-coverage-empirical-binding.md || \
    fail "concept contract lacks the authority-constructor boundary"

echo "[statistical-coverage-empirical-binding-d9] PASS: coverage, provenance, abstention, and authority remain nominally separated"
