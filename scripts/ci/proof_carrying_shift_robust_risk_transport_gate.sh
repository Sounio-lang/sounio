#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d /tmp/shift-robust-risk-transport-d11-gate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    echo "[shift-robust-risk-transport-d11] FAIL: $*" >&2
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
grep -Eq "Madares|Madaros" "$TMP_DIR/version.log" || {
    cat "$TMP_DIR/version.log" >&2
    fail "canonical bin/souc did not resolve to Madaros"
}

kernel="stdlib/epistemic/proof_carrying_shift_robust_risk_transport.sio"
grep -Fq 'use epistemic::proof_carrying_deployment_validity_revocable_authority::{' \
    "$kernel" || fail "D11 kernel lacks the D10 nominal boundary import"
grep -Fq '    observe_d10_fixture_canary_lease,' "$kernel" || \
    fail "D11 kernel lacks its explicit D10 observation import"
grep -Fq '    replay_d10_complete_fixture_canary,' "$kernel" || \
    fail "D11 kernel lacks its explicit D10 replay import"
if grep -Fq 'proof_carrying_deployment_validity_revocable_authority::*' "$kernel"; then
    fail "D11 kernel uses a wildcard D10 import"
fi
if grep -Eq '^pub struct D11[A-Za-z0-9_]*Token' "$kernel"; then
    fail "an authority-bearing D11 token was exposed as a public struct"
fi
for private_token in \
    D11LabeledExactJointFixtureToken \
    D11FixtureScopeSubsetToken \
    D11ActiveCalibratorDriftToken \
    D11TargetAmbiguityTriggerToken \
    D11ConceptShiftTriggerToken \
    D11NominalTraceToken; do
    grep -Fq "struct $private_token" "$kernel" || \
        fail "D11 kernel lacks private token $private_token"
done
grep -Fq 'struct D11ProductionDeploymentAuthorityReceipt' "$kernel" || \
    fail "production authority is not reserved as a private type"
grep -Fq 'struct D11ClinicalActionAuthorityReceipt' "$kernel" || \
    fail "clinical authority is not reserved as a private type"
grep -Fq 'struct D11InstitutionalRevocationAuthorityReceipt' "$kernel" || \
    fail "institutional revocation authority is not reserved as a private type"
grep -Fq 'struct D11ExternalTransportValidationReceipt' "$kernel" || \
    fail "external transport validation is not reserved as a private type"
grep -Fq 'struct D11NoShiftReceipt' "$kernel" || \
    fail "NoShift is not reserved as a private type"
grep -Fq 'struct D11PatientStateReceipt' "$kernel" || \
    fail "patient state is not reserved as a private type"
for authority in \
    D11ExternalTransportValidationReceipt \
    D11NoShiftReceipt \
    D11ProductionDeploymentAuthorityReceipt \
    D11PatientStateReceipt \
    D11ClinicalActionAuthorityReceipt \
    D11InstitutionalRevocationAuthorityReceipt; do
    if grep -Eq -- "-> ${authority}([[:space:]]|$)" "$kernel"; then
        fail "D11 kernel exposes a producer for reserved authority $authority"
    fi
done

grep -Fq 'assert(token.joint_law_fingerprint == 92352734845403155)' \
    "$kernel" || fail "D11 kernel lacks the exact-law fingerprint wall"
grep -Fq 'd10_attests_d11_scope_subset: false' "$kernel" || \
    fail "D11 kernel lets D10 attest a D11 scope fact"
for index in 0 1 2 3; do
    grep -Fq "assert(token.target_member_$index == token.source_member_$index)" \
        "$kernel" || fail "D11 scope proof lacks member $index equality"
done
grep -Fq 'subset_established: false' "$kernel" || \
    fail "D11 kernel lacks the smaller-disjoint-scope wall"
grep -Fq 'let observation = observe_d11_active_calibrator_drift(31604)' \
    "$kernel" || fail "D11 degradation trigger is not internally fixed"
grep -Fq 'let observation = observe_d11_unlabeled_target_ambiguity(31403)' \
    "$kernel" || fail "D11 suspension trigger is not internally fixed"
grep -Fq 'let observation = observe_d11_concept_shift_failure(31402)' \
    "$kernel" || fail "D11 revocation trigger is not internally fixed"
grep -Fq 'globally_absorbing: false' "$kernel" || \
    fail "D11 kernel overclaims global absorption"
grep -Fq 'runtime_canary_disabled: false' "$kernel" || \
    fail "D11 kernel overclaims runtime disablement"
grep -Fq 'fixture_replayable: true' "$kernel" || \
    fail "D11 kernel hides fixture replayability"
grep -Fq 'single_chain_proven: false' "$kernel" || \
    fail "D11 kernel overclaims a unique execution chain"

ontology="stdlib/ontology/shift_robust_risk_transport.sio"
grep -Fq 'class ScopeEvidenceArtifact subclass_of EpistemicArtifact' \
    "$ontology" || fail "D11 scope evidence is not rooted outside warrant state"
grep -Fq 'class TargetScopeSubsetEvidence subclass_of ScopeEvidenceArtifact' \
    "$ontology" || fail "D11 ontology lacks scope-subset evidence"
grep -Fq 'class ReservedAuthorityArtifact subclass_of EpistemicArtifact' \
    "$ontology" || fail "D11 reserved authority is not rooted outside warrant state"
if grep -Fq 'class ReservedAuthorityArtifact subclass_of WarrantStateArtifact' \
    "$ontology"; then
    fail "D11 reserved authority transitively enters warrant state"
fi
grep -Fq 'class SourceCanaryWarrant subclass_of SourceScopedWarrantArtifact' \
    "$ontology" || fail "D11 ontology lacks source-scoped warrants"
grep -Fq 'class TargetCanaryContinuation subclass_of TargetScopedWarrantArtifact' \
    "$ontology" || fail "D11 ontology lacks target-scoped warrants"
for authority in ProductionDeploymentAuthority ClinicalActionAuthority InstitutionalRevocationAuthority; do
    grep -Fq "class $authority subclass_of ReservedAuthorityArtifact" \
        "$ontology" || fail "$authority is not reserved outside warrant state"
    if grep -Fq "class $authority subclass_of WarrantStateArtifact" "$ontology"; then
        fail "$authority incorrectly descends from warrant state"
    fi
done
grep -Fq 'role calibrated_in_source domain SourceMarginalCalibration range SourcePopulation' \
    "$ontology" || fail "D11 ontology conflates source calibration"
grep -Fq 'role calibrated_in_target domain TargetLocalCalibration range TargetPopulation' \
    "$ontology" || fail "D11 ontology conflates target calibration"
grep -Fq 'role establishes_source_scope domain TargetScopeSubsetEvidence range SourcePopulation' \
    "$ontology" || fail "D11 ontology lacks source-side subset evidence"
grep -Fq 'role establishes_target_scope domain TargetScopeSubsetEvidence range TargetPopulation' \
    "$ontology" || fail "D11 ontology lacks target-side subset evidence"
grep -Fq 'role scopes_source_to domain SourceScopedWarrantArtifact range SourcePopulation' \
    "$ontology" || fail "D11 ontology lacks source-warrant scope"
grep -Fq 'role scopes_target_to domain TargetScopedWarrantArtifact range TargetPopulation' \
    "$ontology" || fail "D11 ontology lacks target-warrant scope"

check_sources=(
    "$kernel"
    "$ontology"
    tests/run-pass/clinical_shift_robust_risk_transport_witness.sio
)
for source in "${check_sources[@]}"; do
    log="$TMP_DIR/$(basename "$source").check.log"
    if ! bin/souc check "$source" >"$log" 2>&1; then
        cat "$log" >&2
        fail "$source did not typecheck"
    fi
    grep -q "check: OK" "$log" || fail "$source lacked the check receipt"
done

native="tests/run-pass/clinical_shift_robust_risk_transport_native_witness.sio"
if ! bin/souc run "$native" >"$TMP_DIR/native.log" 2>&1; then
    cat "$TMP_DIR/native.log" >&2
    fail "standalone D11 witness did not execute"
fi
require_line "D11-W0 d10_lease=30821 source_rank=3 canary_only=true production=false clinical=false" "$TMP_DIR/native.log"
require_line "D11-W1 covariate source_mass=3,9 target_mass=6,6 weights=2,2/3 source_risk=1/4 target_risk=1/2 weighted=1/2" "$TMP_DIR/native.log"
require_line "D11-W2 overlap source_mass=4,0 target_mass=2,2 target_risk_interval=[0,1/2] point_identified=false" "$TMP_DIR/native.log"
require_line "D11-W3 label src=3,9 tgt=6,6 risk=1/4->1/2 probe=31311 loss=31711 singular=1/4,3/4" "$TMP_DIR/native.log"
require_line "D11-W4 concept same_unlabeled_inputs=true stable_risk=2/4 shifted_risk=4/4 labels_required=true" "$TMP_DIR/native.log"
require_line "D11-W5 subgroup marginal_a=6/12 marginal_b=6/12 worst_a=1/2 worst_b=1" "$TMP_DIR/native.log"
require_line "D11-W6 calibration diag=0->1/4 local=0 active_later=-1/2 diagnostic_transition=false" "$TMP_DIR/native.log"
require_line "D11-W7 conformal source_risk=1/4 tv=1/4 target_risk=1/2 bound_tight=true general_theorem=false" "$TMP_DIR/native.log"
require_line "D11-W8 rank=3,3,2,1,0 scope=8,4,2,0,0 w=31121..31124 up=0 nominal=true global=false" "$TMP_DIR/native.log"
require_line "D11-W8-LIMIT runtime_disabled=false replayable=true stale_alias_invalidated=false" "$TMP_DIR/native.log"
require_line "PROOF-CARRYING SHIFT-ROBUST RISK TRANSPORT D11 PASS" "$TMP_DIR/native.log"

if ! bin/souc run tests/run-pass/ontology_shift_robust_risk_transport_types.sio \
    >"$TMP_DIR/ontology.log" 2>&1; then
    cat "$TMP_DIR/ontology.log" >&2
    fail "parallel D11 ontology witness did not execute"
fi
require_line "shift robust risk transport parallel ontology OK" \
    "$TMP_DIR/ontology.log"

oracle="scripts/research/proof_carrying_shift_robust_risk_transport_oracle.py"
if ! python3 "$oracle" >"$TMP_DIR/oracle.log" 2>&1; then
    cat "$TMP_DIR/oracle.log" >&2
    fail "independent D11 oracle failed"
fi
require_line "ORACLE_D11_W0 d10_lease=30821 source_rank=3 canary_only=true production=false clinical=false" "$TMP_DIR/oracle.log"
require_line "ORACLE_D11_W1 covariate source_mass=3,9 target_mass=6,6 weights=2,2/3 source_risk=1/4 target_risk=1/2 weighted=1/2" "$TMP_DIR/oracle.log"
require_line "ORACLE_D11_W2 overlap source_mass=4,0 target_mass=2,2 target_risk_interval=[0,1/2] point_identified=false" "$TMP_DIR/oracle.log"
require_line "ORACLE_D11_W3 label src=3,9 tgt=6,6 risk=1/4->1/2 probe=31311 loss=31711 singular=1/4,3/4" "$TMP_DIR/oracle.log"
require_line "ORACLE_D11_W4 concept same_unlabeled_inputs=true stable_risk=2/4 shifted_risk=4/4 labels_required=true" "$TMP_DIR/oracle.log"
require_line "ORACLE_D11_W5 subgroup marginal_a=6/12 marginal_b=6/12 worst_a=1/2 worst_b=1" "$TMP_DIR/oracle.log"
require_line "ORACLE_D11_W6 calibration diag=0->1/4 local=0 active_later=-1/2 diagnostic_transition=false" "$TMP_DIR/oracle.log"
require_line "ORACLE_D11_W7 conformal source_risk=1/4 tv=1/4 target_risk=1/2 bound_tight=true general_theorem=false" "$TMP_DIR/oracle.log"
require_line "ORACLE_D11_W8 rank=3,3,2,1,0 scope=8,4,2,0,0 w=31121..31124 up=0 nominal=true global=false" "$TMP_DIR/oracle.log"
require_line "ORACLE_D11_W8_LIMIT runtime_disabled=false replayable=true stale_alias_invalidated=false" "$TMP_DIR/oracle.log"
require_line "PROOF-CARRYING SHIFT-ROBUST RISK TRANSPORT D11 ORACLE PASS" "$TMP_DIR/oracle.log"

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
        fail "$source unexpectedly constructed a reserved private receipt"
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

expect_type_rejection tests/compile-fail/clinical_d11_label_cannot_replace_covariate_d11.sio D11CovariateTransportToken D11LabelTransportToken
expect_type_rejection tests/compile-fail/clinical_d11_covariate_cannot_replace_label_d11.sio D11LabelTransportToken D11CovariateTransportToken
expect_type_rejection tests/compile-fail/clinical_d11_concept_shift_cannot_replace_stability_d11.sio D11ConceptStabilityToken D11ConceptShiftFailureReceipt
expect_type_rejection tests/compile-fail/clinical_d11_overlap_failure_cannot_replace_transport_d11.sio D11CovariateTransportToken D11CovariateOverlapFailureReceipt
expect_type_rejection tests/compile-fail/clinical_d11_singular_label_cannot_replace_transport_d11.sio D11LabelTransportToken D11LabelShiftNonidentifiabilityReceipt
expect_type_rejection tests/compile-fail/clinical_d11_marginal_cannot_replace_subgroup_d11.sio D11SubgroupRiskBoundToken D11MarginalTargetRiskReceipt
expect_type_rejection tests/compile-fail/clinical_d11_source_calibration_cannot_replace_target_local_d11.sio D11TargetLocalCalibrationToken D11SourceMarginalCalibrationReceipt
expect_type_rejection tests/compile-fail/clinical_d11_source_conformal_observation_cannot_replace_target_risk_d11.sio D11ShiftRobustTargetRiskToken D11SourceConformalRiskReceipt
expect_type_rejection tests/compile-fail/clinical_d11_estimated_weight_cannot_replace_given_tv_d11.sio D11JointTVRadiusToken D11EstimatedShiftWeightPointReceipt
expect_type_rejection tests/compile-fail/clinical_d11_no_detected_cannot_replace_no_shift_d11.sio D11NoShiftReceipt D11NoDetectedTargetShiftReceipt
expect_type_rejection tests/compile-fail/clinical_d11_unlabeled_ambiguity_cannot_replace_target_evidence_d11.sio D11TargetEvidenceBoundToken D11UnlabeledTargetAmbiguityReceipt
expect_type_rejection tests/compile-fail/clinical_d11_d10_observation_cannot_replace_source_canary_d11.sio D11SourceCanaryBoundToken D11D10BoundaryObservationReceipt
expect_type_rejection tests/compile-fail/clinical_d11_target_evidence_observation_cannot_replace_token_d11.sio D11TargetEvidenceBoundToken D11TargetEvidenceBoundObservationReceipt
expect_type_rejection tests/compile-fail/clinical_d11_degraded_cannot_replace_continuation_d11.sio D11TargetCanaryContinuationToken D11DegradedCanaryToken
expect_type_rejection tests/compile-fail/clinical_d11_suspended_cannot_replace_continuation_d11.sio D11TargetCanaryContinuationToken D11SuspendedCanaryToken
expect_type_rejection tests/compile-fail/clinical_d11_revoked_cannot_replace_continuation_d11.sio D11TargetCanaryContinuationToken D11RevokedCanaryToken
expect_type_rejection tests/compile-fail/clinical_d11_suspended_cannot_replace_degraded_d11.sio D11DegradedCanaryToken D11SuspendedCanaryToken
expect_type_rejection tests/compile-fail/clinical_d11_revoked_cannot_replace_suspended_d11.sio D11SuspendedCanaryToken D11RevokedCanaryToken
expect_type_rejection tests/compile-fail/clinical_d11_continuation_cannot_replace_degraded_d11.sio D11DegradedCanaryToken D11TargetCanaryContinuationToken
expect_type_rejection tests/compile-fail/clinical_d11_continuation_cannot_replace_production_d11.sio D11ProductionDeploymentAuthorityReceipt D11TargetCanaryContinuationToken
expect_type_rejection tests/compile-fail/clinical_d11_continuation_cannot_authorize_action_d11.sio D11ClinicalActionAuthorityReceipt D11TargetCanaryContinuationToken
expect_type_rejection tests/compile-fail/clinical_d11_degraded_cannot_authorize_action_d11.sio D11ClinicalActionAuthorityReceipt D11DegradedCanaryToken
expect_type_rejection tests/compile-fail/clinical_d11_suspended_cannot_authorize_action_d11.sio D11ClinicalActionAuthorityReceipt D11SuspendedCanaryToken
expect_type_rejection tests/compile-fail/clinical_d11_revoked_cannot_authorize_action_d11.sio D11ClinicalActionAuthorityReceipt D11RevokedCanaryToken
expect_type_rejection tests/compile-fail/clinical_d11_target_evidence_cannot_replace_external_validation_d11.sio D11ExternalTransportValidationReceipt D11TargetEvidenceBoundToken
expect_type_rejection tests/compile-fail/clinical_d11_source_conformal_token_cannot_replace_target_risk_d11.sio D11ShiftRobustTargetRiskToken D11SourceConformalRiskToken
expect_type_rejection tests/compile-fail/clinical_d11_concept_stability_cannot_replace_covariate_d11.sio D11CovariateTransportToken D11ConceptStabilityToken
expect_type_rejection tests/compile-fail/clinical_d11_target_calibration_cannot_replace_conformal_risk_d11.sio D11SourceConformalRiskToken D11TargetLocalCalibrationToken
expect_type_rejection tests/compile-fail/clinical_d11_given_tv_cannot_replace_target_risk_d11.sio D11ShiftRobustTargetRiskToken D11JointTVRadiusToken
expect_type_rejection tests/compile-fail/clinical_d11_subgroup_cannot_replace_target_calibration_d11.sio D11TargetLocalCalibrationToken D11SubgroupRiskBoundToken
expect_type_rejection tests/compile-fail/clinical_d11_public_joint_observation_cannot_replace_exact_law_d11.sio D11LabeledExactJointFixtureToken D11CovariateShiftTransportObservationReceipt
expect_type_rejection tests/compile-fail/clinical_d11_smaller_disjoint_scope_cannot_replace_subset_d11.sio D11FixtureScopeSubsetToken D11SmallerDisjointScopeObservationReceipt
expect_type_rejection tests/compile-fail/clinical_d11_diagnostic_drift_cannot_replace_active_trigger_d11.sio D11ActiveCalibratorDriftToken D11TargetCalibrationDriftReceipt
expect_type_rejection tests/compile-fail/clinical_d11_public_ambiguity_cannot_replace_private_trigger_d11.sio D11TargetAmbiguityTriggerToken D11UnlabeledTargetAmbiguityReceipt
expect_type_rejection tests/compile-fail/clinical_d11_public_concept_shift_cannot_replace_private_trigger_d11.sio D11ConceptShiftTriggerToken D11ConceptShiftFailureReceipt

expect_private_constructor_rejection tests/compile-fail/clinical_d11_cannot_construct_external_validation_d11.sio
expect_private_constructor_rejection tests/compile-fail/clinical_d11_cannot_construct_no_shift_d11.sio
expect_private_constructor_rejection tests/compile-fail/clinical_d11_cannot_construct_production_authority_d11.sio
expect_private_constructor_rejection tests/compile-fail/clinical_d11_cannot_construct_patient_state_d11.sio
expect_private_constructor_rejection tests/compile-fail/clinical_d11_cannot_construct_clinical_authority_d11.sio
expect_private_constructor_rejection tests/compile-fail/clinical_d11_cannot_construct_institutional_revocation_d11.sio

expect_type_rejection tests/compile-fail/ontology_d11_covariate_cannot_replace_label_d11.sio LabelShiftEvidence CovariateShiftEvidence
expect_type_rejection tests/compile-fail/ontology_d11_label_cannot_replace_concept_d11.sio ConceptShiftEvidence LabelShiftEvidence
expect_type_rejection tests/compile-fail/ontology_d11_concept_cannot_replace_covariate_d11.sio CovariateShiftEvidence ConceptShiftEvidence
expect_type_rejection tests/compile-fail/ontology_d11_overlap_failure_cannot_replace_overlap_d11.sio OverlapEvidence OverlapFailure
expect_type_rejection tests/compile-fail/ontology_d11_singular_label_cannot_replace_identifiability_d11.sio LabelShiftIdentifiability SingularLabelShift
expect_type_rejection tests/compile-fail/ontology_d11_marginal_cannot_replace_subgroup_d11.sio PrespecifiedSubgroupRisk MarginalTargetRisk
expect_type_rejection tests/compile-fail/ontology_d11_source_conformal_cannot_replace_target_risk_d11.sio ShiftRobustTargetRisk SourceConformalRisk
expect_type_rejection tests/compile-fail/ontology_d11_source_calibration_cannot_replace_target_local_d11.sio TargetLocalCalibration SourceMarginalCalibration
expect_type_rejection tests/compile-fail/ontology_d11_unlabeled_cannot_replace_labeled_d11.sio LabeledTargetEvidence UnlabeledTargetEvidence
expect_type_rejection tests/compile-fail/ontology_d11_no_detected_cannot_replace_no_shift_d11.sio NoShiftEvidence NoDetectedShiftEvidence
expect_type_rejection tests/compile-fail/ontology_d11_degraded_cannot_replace_continuation_d11.sio TargetCanaryContinuation DegradedCanary
expect_type_rejection tests/compile-fail/ontology_d11_suspended_cannot_replace_continuation_d11.sio TargetCanaryContinuation SuspendedCanary
expect_type_rejection tests/compile-fail/ontology_d11_revoked_cannot_replace_continuation_d11.sio TargetCanaryContinuation RevokedCanary
expect_type_rejection tests/compile-fail/ontology_d11_continuation_cannot_replace_production_d11.sio ProductionDeploymentAuthority TargetCanaryContinuation
expect_type_rejection tests/compile-fail/ontology_d11_continuation_cannot_replace_clinical_d11.sio ClinicalActionAuthority TargetCanaryContinuation
expect_type_rejection tests/compile-fail/ontology_d11_revoked_cannot_replace_institutional_authority_d11.sio InstitutionalRevocationAuthority RevokedCanary
expect_type_rejection tests/compile-fail/ontology_d11_production_authority_cannot_replace_warrant_d11.sio WarrantStateArtifact ProductionDeploymentAuthority
expect_type_rejection tests/compile-fail/ontology_d11_clinical_authority_cannot_replace_warrant_d11.sio WarrantStateArtifact ClinicalActionAuthority
expect_type_rejection tests/compile-fail/ontology_d11_institutional_authority_cannot_replace_warrant_d11.sio WarrantStateArtifact InstitutionalRevocationAuthority
expect_type_rejection tests/compile-fail/ontology_d11_warrant_cannot_replace_reserved_authority_d11.sio ReservedAuthorityArtifact WarrantStateArtifact
expect_type_rejection tests/compile-fail/ontology_d11_scope_evidence_cannot_replace_warrant_d11.sio WarrantStateArtifact TargetScopeSubsetEvidence
expect_type_rejection tests/compile-fail/ontology_d11_target_warrant_cannot_replace_source_scoped_d11.sio SourceScopedWarrantArtifact TargetCanaryContinuation
expect_type_rejection tests/compile-fail/ontology_d11_source_warrant_cannot_replace_target_scoped_d11.sio TargetScopedWarrantArtifact SourceCanaryWarrant

clinical_count="$(find tests/compile-fail -maxdepth 1 -name 'clinical_d11_*_d11.sio' | wc -l | tr -d ' ')"
ontology_count="$(find tests/compile-fail -maxdepth 1 -name 'ontology_d11_*_d11.sio' | wc -l | tr -d ' ')"
[[ "$clinical_count" == "41" ]] || fail "D11 clinical negative matrix is not exactly 41 files"
[[ "$ontology_count" == "23" ]] || fail "D11 ontology negative matrix is not exactly 23 files"

d11_count="$(awk -F '\t' '$1 == "SOUNIO-PROOF-CARRYING-SHIFT-ROBUST-RISK-TRANSPORT" { count++ } END { print count + 0 }' docs/internal/concepts/registry.tsv)"
[[ "$d11_count" == "1" ]] || fail "D11 concept row is not exactly one"
grep -Fqx $'SOUNIO-PROOF-CARRYING-SHIFT-ROBUST-RISK-TRANSPORT\texecutable\tfounder\tdocs/internal/concepts/proof-carrying-shift-robust-risk-transport.md\tstdlib/epistemic/proof_carrying_shift_robust_risk_transport.sio\tcompiler-enforced-affine-attenuation-and-trusted-target-monitor' \
    docs/internal/concepts/registry.tsv || fail "D11 concept row is not exact"

required_bindings=(
    $'docs/internal/concepts/proof-carrying-shift-robust-risk-transport.md\tconcept-contract'
    $'docs/research/proof_carrying_shift_robust_risk_transport_d11_2026-07-19.md\tresearch-specification'
    $'docs/superpowers/specs/2026-07-19-proof-carrying-shift-robust-risk-transport-d11-design.md\tdesign-specification'
    $'docs/superpowers/plans/2026-07-19-proof-carrying-shift-robust-risk-transport-d11.md\timplementation-plan'
    $'stdlib/epistemic/proof_carrying_shift_robust_risk_transport.sio\tcanonical-model-kernel'
    $'stdlib/ontology/shift_robust_risk_transport.sio\tparallel-ontology-boundary'
    $'tests/run-pass/clinical_shift_robust_risk_transport_*\tpositive-evidence'
    $'tests/run-pass/ontology_shift_robust_risk_transport_types.sio\tparallel-ontology-evidence'
    $'tests/compile-fail/clinical_d11_*_d11.sio\tnegative-evidence'
    $'tests/compile-fail/ontology_d11_*_d11.sio\tnegative-evidence'
    $'scripts/research/proof_carrying_shift_robust_risk_transport_oracle.py\tindependent-oracle'
    $'scripts/ci/proof_carrying_shift_robust_risk_transport_gate.sh\tacceptance-gate'
)
for binding in "${required_bindings[@]}"; do
    grep -Fqx $'SOUNIO-PROOF-CARRYING-SHIFT-ROBUST-RISK-TRANSPORT\t'"$binding" \
        docs/internal/concepts/bindings.tsv || fail "missing D11 binding: $binding"
done

contract="docs/internal/concepts/proof-carrying-shift-robust-risk-transport.md"
grep -Fq "source canary rank 3" "$contract" || fail "contract lacks non-expansive rank boundary"
grep -Fq "does not statically" "$contract" || fail "contract lacks stale-alias boundary"
grep -Fq "no real transportability" "$contract" || fail "contract lacks real-transport boundary"
grep -Fq "one exact labeled joint law" "$contract" || \
    fail "contract lacks the shared exact-law boundary"
grep -Fq "not attest this D11 subset fact" "$contract" || \
    fail "contract lets D10 invent D11 scope evidence"
grep -Fq "not globally" "$contract" || \
    fail "contract overclaims nominal revocation"
grep -Fq "can replay the same fixture" "$contract" || \
    fail "contract hides fixture replayability"

for mode in default rebuilt; do
    ontology_log="$TMP_DIR/ontology-$mode.log"
    if ! SOUNIO_ONTOLOGY_COMPILE_GATES=0 SOUNIO_ONTOLOGY_PREPARE_GENERATED=0 \
        bash scripts/ci/run_ontology_validation.sh --mode "$mode" --quiet \
        ontology_shift_robust_risk_transport \
        >"$ontology_log" 2>&1; then
        cat "$ontology_log" >&2
        fail "$mode ontology validation failed"
    fi
    require_line "Pass: 1" "$ontology_log"
    require_line "Fail: 0" "$ontology_log"
    require_line "All tests passed!" "$ontology_log"
done
require_line "Using rebuilt wrapper:" "$TMP_DIR/ontology-rebuilt.log"

if ! bash scripts/ci/proof_carrying_deployment_validity_revocable_authority_gate.sh \
    >"$TMP_DIR/d10-recursive.log" 2>&1; then
    cat "$TMP_DIR/d10-recursive.log" >&2
    fail "recursive D10-D0 boundary gate failed"
fi
require_line "[deployment-validity-revocable-authority-d10] PASS" \
    "$TMP_DIR/d10-recursive.log"

echo "[shift-robust-risk-transport-d11] PASS: target risk stays assumption-bound and canary authority only preserves or attenuates"
