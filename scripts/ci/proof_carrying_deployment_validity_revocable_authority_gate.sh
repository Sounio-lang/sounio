#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d /tmp/deployment-validity-d10-gate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    echo "[deployment-validity-revocable-authority-d10] FAIL: $*" >&2
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

kernel="stdlib/epistemic/proof_carrying_deployment_validity_revocable_authority.sio"
grep -Fq 'use epistemic::proof_carrying_statistical_coverage_empirical_binding::*' \
    "$kernel" || fail "D10 kernel lacks the D9 nominal boundary import"
if grep -Eq '^pub struct D10[A-Za-z0-9_]*Token' "$kernel"; then
    fail "an authority-bearing D10 token was exposed as a public struct"
fi
grep -Fq 'struct D10ProductionDeploymentAuthorityReceipt' "$kernel" || \
    fail "production authority is not reserved as a private type"
grep -Fq 'struct D10ClinicalActionAuthorityReceipt' "$kernel" || \
    fail "clinical authority is not reserved as a private type"

check_sources=(
    "$kernel"
    stdlib/ontology/deployment_validity_revocable_authority.sio
    tests/run-pass/clinical_deployment_validity_revocable_authority_witness.sio
)
for source in "${check_sources[@]}"; do
    log="$TMP_DIR/$(basename "$source").check.log"
    if ! bin/souc check "$source" >"$log" 2>&1; then
        cat "$log" >&2
        fail "$source did not typecheck"
    fi
    grep -q "check: OK" "$log" || fail "$source lacked the check receipt"
done

native="tests/run-pass/clinical_deployment_validity_revocable_authority_native_witness.sio"
if ! bin/souc run "$native" >"$TMP_DIR/native.log" 2>&1; then
    cat "$TMP_DIR/native.log" >&2
    fail "standalone D10 witness did not execute"
fi
require_line "D10-W0 brier_a=1/40 brier_b=29/160 ranking_a=4/4 ranking_b=4/4 decisions=0,0,1,1 threshold=1/2" "$TMP_DIR/native.log"
require_line "D10-W1 fixed_look1=12/16 fixed_look2=12/16 stopped=9/16 path_masses=9,3,3,1" "$TMP_DIR/native.log"
require_line "D10-W2 bounded_time_uniform_simultaneous=12/16 stopped=12/16 general_theorem=false" "$TMP_DIR/native.log"
require_line "D10-W3 e0=1 expected_e1=1 expected_e2=1 expected_stopped=1" "$TMP_DIR/native.log"
require_line "D10-W4 same_abstention=true site_a=ready,safe site_b=quarantined,unsafe capacity_a=2 capacity_b=1 ack_a=2 ack_b=0" "$TMP_DIR/native.log"
require_line "D10-W5 metrics_equal=true authorized=true out_of_protocol=quarantined authorization_truth_table=1/16" "$TMP_DIR/native.log"
require_line "D10-W6 drift_categories=input,performance,calibration distinct=true brier_delta=5/32 no_detected_is_no_shift=false" "$TMP_DIR/native.log"
require_line "D10-W7 ledger=40+60=100 capacity=100 remaining=0 repeat=reuse_refused overspend10=overspend_refused" "$TMP_DIR/native.log"
require_line "D10-W8 epoch1=live epoch2=revoked epoch3=revoked old_facet_statically_invalidated=false" "$TMP_DIR/native.log"
require_line "PROOF-CARRYING DEPLOYMENT VALIDITY AND REVOCABLE AUTHORITY D10 PASS" "$TMP_DIR/native.log"

if ! bin/souc run tests/run-pass/ontology_deployment_validity_revocable_authority_types.sio \
    >"$TMP_DIR/ontology.log" 2>&1; then
    cat "$TMP_DIR/ontology.log" >&2
    fail "parallel D10 ontology witness did not execute"
fi
require_line "deployment validity revocable authority parallel ontology OK" \
    "$TMP_DIR/ontology.log"

oracle="scripts/research/proof_carrying_deployment_validity_revocable_authority_oracle.py"
if ! python3 "$oracle" >"$TMP_DIR/oracle.log" 2>&1; then
    cat "$TMP_DIR/oracle.log" >&2
    fail "independent D10 oracle failed"
fi
require_line "ORACLE_D10_W0 brier_a=1/40 brier_b=29/160 ranking_a=4/4 ranking_b=4/4 decisions=0,0,1,1 threshold=1/2" "$TMP_DIR/oracle.log"
require_line "ORACLE_D10_W1 fixed_look1=12/16 fixed_look2=12/16 stopped=9/16 path_masses=9,3,3,1" "$TMP_DIR/oracle.log"
require_line "ORACLE_D10_W2 bounded_time_uniform_simultaneous=12/16 stopped=12/16 general_theorem=false" "$TMP_DIR/oracle.log"
require_line "ORACLE_D10_W3 e0=1 expected_e1=1 expected_e2=1 expected_stopped=1" "$TMP_DIR/oracle.log"
require_line "ORACLE_D10_W4 same_abstention=true site_a=ready,safe site_b=quarantined,unsafe capacity_a=2 capacity_b=1 ack_a=2 ack_b=0" "$TMP_DIR/oracle.log"
require_line "ORACLE_D10_W5 metrics_equal=true authorized=true out_of_protocol=quarantined authorization_truth_table=1/16" "$TMP_DIR/oracle.log"
require_line "ORACLE_D10_W6 drift_categories=input,performance,calibration distinct=true brier_delta=5/32 no_detected_is_no_shift=false" "$TMP_DIR/oracle.log"
require_line "ORACLE_D10_W7 ledger=40+60=100 capacity=100 remaining=0 repeat=reuse_refused overspend10=overspend_refused" "$TMP_DIR/oracle.log"
require_line "ORACLE_D10_W8 epoch1=live epoch2=revoked epoch3=revoked old_facet_statically_invalidated=false" "$TMP_DIR/oracle.log"
require_line "PROOF-CARRYING DEPLOYMENT VALIDITY AND REVOCABLE AUTHORITY D10 ORACLE PASS" "$TMP_DIR/oracle.log"

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

expect_type_rejection tests/compile-fail/clinical_d10_discrimination_cannot_replace_calibration_d10.sio D10CalibrationProfileAReceipt D10DiscriminationReceipt
expect_type_rejection tests/compile-fail/clinical_d10_fixed_horizon_cannot_replace_time_uniform_d10.sio D10TimeUniformValidityToken D10FixedHorizonCoverageReceipt
expect_type_rejection tests/compile-fail/clinical_d10_fixed_e_value_cannot_replace_e_process_d10.sio D10EProcessValidityToken D10FixedTimeEValueReceipt
expect_type_rejection tests/compile-fail/clinical_d10_manufacturer_safety_cannot_replace_local_safety_d10.sio D10LocalDeploymentSafetyToken D10ManufacturerSafetyCaseReceipt
expect_type_rejection tests/compile-fail/clinical_d10_model_abstention_cannot_replace_safe_deferral_d10.sio D10SafeDeferralToken D10ModelAbstentionReceipt
expect_type_rejection tests/compile-fail/clinical_d10_pending_deferral_cannot_replace_safe_deferral_d10.sio D10SafeDeferralToken D10PendingDeferralReceipt
expect_type_rejection tests/compile-fail/clinical_d10_human_present_cannot_replace_effective_oversight_d10.sio D10EffectiveOversightToken D10HumanPresentReceipt
expect_type_rejection tests/compile-fail/clinical_d10_site_a_validation_cannot_replace_site_b_d10.sio D10ResearchValidationSiteBReceipt D10ResearchValidationSiteAReceipt
expect_type_rejection tests/compile-fail/clinical_d10_research_validation_cannot_replace_external_validation_d10.sio D10ExternalValidationReceipt D10ResearchValidationSiteAReceipt
expect_type_rejection tests/compile-fail/clinical_d10_input_shift_cannot_replace_performance_drift_d10.sio D10PerformanceDriftReceipt D10InputDistributionShiftReceipt
expect_type_rejection tests/compile-fail/clinical_d10_no_detected_shift_cannot_replace_no_shift_d10.sio D10NoShiftReceipt D10NoDetectedShiftReceipt
expect_type_rejection tests/compile-fail/clinical_d10_pccp_plan_cannot_replace_authorized_modification_d10.sio D10AuthorizedModificationToken D10PCCPPlanReceipt
expect_type_rejection tests/compile-fail/clinical_d10_unauthorized_change_cannot_replace_authorized_modification_d10.sio D10AuthorizedModificationToken D10UnauthorizedChangeReceipt
expect_type_rejection tests/compile-fail/clinical_d10_site_b_failure_cannot_replace_local_safety_d10.sio D10LocalDeploymentSafetyToken D10SiteBSafetyFailureReceipt
expect_type_rejection tests/compile-fail/clinical_d10_quarantined_cannot_replace_advisory_ready_d10.sio D10AdvisoryReadyToken D10QuarantinedDeploymentReceipt
expect_type_rejection tests/compile-fail/clinical_d10_revoked_lease_cannot_replace_live_lease_d10.sio D10FixtureCanaryLeaseToken D10RevokedCanaryLeaseReceipt
expect_type_rejection tests/compile-fail/clinical_d10_safe_deferral_cannot_authorize_action_d10.sio D10ClinicalActionAuthorityReceipt D10SafeDeferralObservationReceipt
expect_type_rejection tests/compile-fail/clinical_d10_advisory_observation_cannot_authorize_action_d10.sio D10ClinicalActionAuthorityReceipt D10AdvisoryReadyObservationReceipt
expect_type_rejection tests/compile-fail/clinical_d10_coverage_stage_cannot_replace_provenance_stage_d10.sio D10ProvenanceBoundFixtureToken D10CoverageCheckedFixtureToken
expect_type_rejection tests/compile-fail/clinical_d10_provenance_stage_cannot_replace_sequential_stage_d10.sio D10SequentialValidityToken D10ProvenanceBoundFixtureToken
expect_type_rejection tests/compile-fail/clinical_d10_spent_warrant_cannot_replace_unspent_d10.sio D10UnspentFixtureWarrantToken D10SpentFixtureWarrantToken
expect_type_rejection tests/compile-fail/clinical_d10_canary_lease_cannot_replace_production_authority_d10.sio D10ProductionDeploymentAuthorityReceipt D10FixtureCanaryLeaseToken
expect_type_rejection tests/compile-fail/clinical_d10_canary_lease_cannot_authorize_clinical_action_d10.sio D10ClinicalActionAuthorityReceipt D10FixtureCanaryLeaseToken

expect_private_constructor_rejection tests/compile-fail/clinical_d10_cannot_construct_external_validation_d10.sio
expect_private_constructor_rejection tests/compile-fail/clinical_d10_cannot_construct_no_shift_d10.sio
expect_private_constructor_rejection tests/compile-fail/clinical_d10_cannot_construct_production_authority_d10.sio
expect_private_constructor_rejection tests/compile-fail/clinical_d10_cannot_construct_patient_state_d10.sio
expect_private_constructor_rejection tests/compile-fail/clinical_d10_cannot_construct_clinical_action_authority_d10.sio

expect_type_rejection tests/compile-fail/ontology_d10_discrimination_cannot_replace_calibration_d10.sio CalibrationEvidence DiscriminationEvidence
expect_type_rejection tests/compile-fail/ontology_d10_fixed_horizon_cannot_replace_time_uniform_d10.sio TimeUniformValidity FixedHorizonValidity
expect_type_rejection tests/compile-fail/ontology_d10_fixed_e_value_cannot_replace_e_process_d10.sio EProcess FixedEValue
expect_type_rejection tests/compile-fail/ontology_d10_manufacturer_safety_cannot_replace_local_safety_d10.sio LocalDeploymentSafetyCase ManufacturerSafetyCase
expect_type_rejection tests/compile-fail/ontology_d10_model_abstention_cannot_replace_safe_deferral_d10.sio SafeDeferral ModelAbstention
expect_type_rejection tests/compile-fail/ontology_d10_input_shift_cannot_replace_performance_drift_d10.sio PerformanceDrift InputShift
expect_type_rejection tests/compile-fail/ontology_d10_no_detected_shift_cannot_replace_no_shift_d10.sio NoShift NoDetectedShift
expect_type_rejection tests/compile-fail/ontology_d10_pccp_plan_cannot_replace_authorized_modification_d10.sio AuthorizedModification PCCPPlan
expect_type_rejection tests/compile-fail/ontology_d10_unauthorized_change_cannot_replace_authorized_modification_d10.sio AuthorizedModification UnauthorizedChange
expect_type_rejection tests/compile-fail/ontology_d10_canary_lease_cannot_replace_production_authority_d10.sio ProductionDeploymentAuthority FixtureCanaryLease
expect_type_rejection tests/compile-fail/ontology_d10_quarantined_cannot_replace_advisory_ready_d10.sio AdvisoryReady Quarantined
expect_type_rejection tests/compile-fail/ontology_d10_advisory_ready_cannot_replace_clinical_authority_d10.sio ClinicalActionAuthority AdvisoryReady

clinical_count="$(find tests/compile-fail -maxdepth 1 -name 'clinical_d10_*_d10.sio' | wc -l | tr -d ' ')"
ontology_count="$(find tests/compile-fail -maxdepth 1 -name 'ontology_d10_*_d10.sio' | wc -l | tr -d ' ')"
[[ "$clinical_count" == "28" ]] || fail "D10 clinical negative matrix is not exactly 28 files"
[[ "$ontology_count" == "12" ]] || fail "D10 ontology negative matrix is not exactly 12 files"

d10_count="$(awk -F '\t' '$1 == "SOUNIO-PROOF-CARRYING-DEPLOYMENT-VALIDITY-REVOCABLE-AUTHORITY" { count++ } END { print count + 0 }' docs/internal/concepts/registry.tsv)"
[[ "$d10_count" == "1" ]] || fail "D10 concept row is not exactly one"
grep -Fqx $'SOUNIO-PROOF-CARRYING-DEPLOYMENT-VALIDITY-REVOCABLE-AUTHORITY\texecutable\tfounder\tdocs/internal/concepts/proof-carrying-deployment-validity-revocable-authority.md\tstdlib/epistemic/proof_carrying_deployment_validity_revocable_authority.sio\taffine-warrant-consumption-and-trusted-live-institutional-authority-adapter' \
    docs/internal/concepts/registry.tsv || fail "D10 concept row is not exact"

required_bindings=(
    $'docs/internal/concepts/proof-carrying-deployment-validity-revocable-authority.md\tconcept-contract'
    $'docs/research/proof_carrying_deployment_validity_revocable_authority_d10_2026-07-19.md\tresearch-specification'
    $'docs/superpowers/specs/2026-07-19-proof-carrying-deployment-validity-revocable-authority-d10-design.md\tdesign-specification'
    $'docs/superpowers/plans/2026-07-19-proof-carrying-deployment-validity-revocable-authority-d10.md\timplementation-plan'
    $'stdlib/epistemic/proof_carrying_deployment_validity_revocable_authority.sio\tcanonical-model-kernel'
    $'stdlib/ontology/deployment_validity_revocable_authority.sio\tparallel-ontology-boundary'
    $'tests/run-pass/clinical_deployment_validity_revocable_authority_*\tpositive-evidence'
    $'tests/run-pass/ontology_deployment_validity_revocable_authority_types.sio\tparallel-ontology-evidence'
    $'tests/compile-fail/clinical_d10_*_d10.sio\tnegative-evidence'
    $'tests/compile-fail/ontology_d10_*_d10.sio\tnegative-evidence'
    $'scripts/research/proof_carrying_deployment_validity_revocable_authority_oracle.py\tindependent-oracle'
    $'scripts/ci/proof_carrying_deployment_validity_revocable_authority_gate.sh\tacceptance-gate'
)
for binding in "${required_bindings[@]}"; do
    grep -Fqx $'SOUNIO-PROOF-CARRYING-DEPLOYMENT-VALIDITY-REVOCABLE-AUTHORITY\t'"$binding" \
        docs/internal/concepts/bindings.tsv || fail "missing D10 binding: $binding"
done

contract="docs/internal/concepts/proof-carrying-deployment-validity-revocable-authority.md"
grep -Fq "synthetic canary lease" "$contract" || fail "contract lacks canary-only boundary"
grep -Fq "does not claim static no-double-spend" "$contract" || fail "contract lacks affine boundary"
grep -Fq "external validation, production deployment authority" "$contract" || fail "contract lacks external-authority boundary"

for mode in default rebuilt; do
    ontology_log="$TMP_DIR/ontology-$mode.log"
    if ! SOUNIO_ONTOLOGY_COMPILE_GATES=0 SOUNIO_ONTOLOGY_PREPARE_GENERATED=0 \
        bash scripts/ci/run_ontology_validation.sh --mode "$mode" --quiet \
        ontology_deployment_validity_revocable_authority \
        >"$ontology_log" 2>&1; then
        cat "$ontology_log" >&2
        fail "$mode ontology validation failed"
    fi
    require_line "Pass: 1" "$ontology_log"
    require_line "Fail: 0" "$ontology_log"
    require_line "All tests passed!" "$ontology_log"
done
require_line "Using rebuilt wrapper:" "$TMP_DIR/ontology-rebuilt.log"

if ! bash scripts/ci/proof_carrying_statistical_coverage_empirical_binding_gate.sh \
    >"$TMP_DIR/d9-recursive.log" 2>&1; then
    cat "$TMP_DIR/d9-recursive.log" >&2
    fail "recursive D9-D0 boundary gate failed"
fi
require_line "[statistical-coverage-empirical-binding-d9] PASS" \
    "$TMP_DIR/d9-recursive.log"

echo "[deployment-validity-revocable-authority-d10] PASS: sequential warrant, local safety, deferral, change, spending, revocation, and authority remain bounded"
