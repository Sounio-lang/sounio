#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d /tmp/endogenous-observability-d4-gate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    echo "[endogenous-observability-d4] FAIL: $*" >&2
    exit 1
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

check_sources=(
    stdlib/epistemic/proof_carrying_endogenous_observability.sio
    stdlib/ontology/endogenous_observability.sio
    tests/run-pass/clinical_proof_carrying_endogenous_observability_witness.sio
)

for source in "${check_sources[@]}"; do
    log="$TMP_DIR/$(basename "$source").check.log"
    if ! bin/souc check "$source" >"$log" 2>&1; then
        cat "$log" >&2
        fail "$source did not typecheck"
    fi
    grep -q "check: OK" "$log" || fail "$source lacked the check receipt"
done

if ! bin/souc run tests/run-pass/clinical_proof_carrying_endogenous_observability_native_witness.sio \
    >"$TMP_DIR/native.log" 2>&1; then
    cat "$TMP_DIR/native.log" >&2
    fail "native D4 witness did not execute"
fi
grep -q "D4-W0 legacy_missing mask=15 mechanisms=4" "$TMP_DIR/native.log"
grep -q "D4-W1 custody scheduled=1 delivered=1 opportunity=1 response=0 mask=6 ambiguous=independent,dependent" "$TMP_DIR/native.log"
grep -q "D4-W2 equivalence original=same hidden=2|8 retry=response2|nonresponse recoverability=false" "$TMP_DIR/native.log"
grep -q "D4-W3 retry_response mask=2 selected=311 burden=9 fingerprint=227233" "$TMP_DIR/native.log"
grep -q "D4-W4 retry_nonresponse mask=4 selected=312 target_observed=false" "$TMP_DIR/native.log"
grep -q "D4-W5 delayed prompt_tick=2 arrival_tick=3 aligned_tick=3 retroactive=false" "$TMP_DIR/native.log"
grep -q "D4-W6 policy_erased=abstain disconnected=abstain mask=6" "$TMP_DIR/native.log"
grep -q "D4-W7 relabel=invariant clinical=false suffering=false fingerprint_max2=32000000" "$TMP_DIR/native.log"
grep -q "PROOF-CARRYING ENDOGENOUS OBSERVABILITY D4 PASS" "$TMP_DIR/native.log"

if ! bin/souc run tests/run-pass/ontology_endogenous_observability_types.sio \
    >"$TMP_DIR/ontology.log" 2>&1; then
    cat "$TMP_DIR/ontology.log" >&2
    fail "endogenous observability ontology witness did not execute"
fi
grep -q "endogenous observability ontology OK" "$TMP_DIR/ontology.log"

if ! python3 scripts/research/proof_carrying_endogenous_observability_oracle.py \
    >"$TMP_DIR/oracle.log" 2>&1; then
    cat "$TMP_DIR/oracle.log" >&2
    fail "independent D4 oracle rejected the fixture"
fi
grep -q "ORACLE_D4_W0 legacy=missing mechanisms=4 custody_partitions=1|6|8" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D4_W1 custody=1,1,1,1,0,0,0,-1 survivors=independent,dependent mask=6" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D4_W2 original_equal=true hidden=2|8 retry_predictions=different recoverability=false" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D4_W3 retry_response=2 retry_nonresponse=4 burden=9 fingerprint=227233" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D4_W4 delayed=2->3 retroactive=false" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D4_W5 relabelings=24 partitions=invariant policy_erased=6 disconnected=6" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D4_W6 fingerprint_max2=32000000 exhaustive_traces=384 i64_safe=true" "$TMP_DIR/oracle.log"
grep -q "PROOF-CARRYING ENDOGENOUS OBSERVABILITY D4 ORACLE PASS" "$TMP_DIR/oracle.log"

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

expect_type_rejection \
    tests/compile-fail/clinical_legacy_missing_cannot_update_observability_contest.sio \
    AdmissibleObservationCustodyTraceReceipt LegacyCoarsenedMissingTokenReceipt
expect_type_rejection \
    tests/compile-fail/clinical_policy_erased_custody_cannot_update_observability_contest.sio \
    AdmissibleObservationCustodyTraceReceipt PolicyErasedCustodyTraceReceipt
expect_type_rejection \
    tests/compile-fail/clinical_disconnected_retry_cannot_update_observability_contest.sio \
    AdmissibleRetryTraceReceipt DisconnectedRetryTraceReceipt
expect_type_rejection \
    tests/compile-fail/clinical_window_nonresponse_cannot_replace_observed_value.sio \
    ContemporaneousObservedValueReceipt WindowNonresponseReceipt
expect_type_rejection \
    tests/compile-fail/clinical_delayed_response_cannot_replace_contemporaneous_value.sio \
    ContemporaneousObservedValueReceipt DelayedResponseReceipt
expect_type_rejection \
    tests/compile-fail/clinical_observability_ambiguity_cannot_claim_mechanism_identification.sio \
    DeclaredResponseMechanismIdentificationReceipt EndogenousObservabilityAmbiguityReceipt
expect_type_rejection \
    tests/compile-fail/clinical_observed_equivalence_cannot_claim_global_recoverability.sio \
    GlobalRecoverabilityReceipt ObservedTraceEquivalenceReceipt
expect_type_rejection \
    tests/compile-fail/clinical_declared_response_identification_cannot_claim_biological_mechanism.sio \
    BiologicalResponseMechanismReceipt DeclaredResponseMechanismIdentificationReceipt
expect_type_rejection \
    tests/compile-fail/clinical_declared_response_identification_cannot_claim_missingness_taxonomy.sio \
    EmpiricalMissingnessTaxonomyReceipt DeclaredResponseMechanismIdentificationReceipt
expect_type_rejection \
    tests/compile-fail/clinical_retry_assignment_cannot_claim_real_randomization.sio \
    RealPersonRandomizationReceipt SyntheticExogenousRetryAssignmentReceipt
expect_type_rejection \
    tests/compile-fail/clinical_observation_burden_cannot_claim_suffering.sio \
    SubjectiveSufferingReceipt AdmissibleRetryTraceReceipt
expect_type_rejection \
    tests/compile-fail/clinical_policy_withholding_cannot_replace_participant_nonresponse.sio \
    WindowNonresponseReceipt ObservationPolicyDecisionReceipt
expect_type_rejection \
    tests/compile-fail/clinical_delivery_failure_cannot_replace_participant_nonresponse.sio \
    WindowNonresponseReceipt DeliveryEventReceipt
expect_type_rejection \
    tests/compile-fail/ontology_coarsened_missing_cannot_replace_observed_value.sio \
    ContemporaneousObservedValue LegacyCoarsenedMissingToken
expect_type_rejection \
    tests/compile-fail/ontology_delayed_response_cannot_replace_contemporaneous_value.sio \
    ContemporaneousObservedValue DelayedResponse
expect_type_rejection \
    tests/compile-fail/ontology_declared_response_mechanism_cannot_replace_biological_mechanism.sio \
    BiologicalResponseMechanismReceipt DeclaredResponseMechanismIdentification

awk -F '\t' '$1 == "SOUNIO-ENDOGENOUS-OBSERVABILITY" && $2 == "executable" { found = 1 } END { exit !found }' \
    docs/internal/concepts/registry.tsv || fail "executable concept registry row is missing"

grep -q "does not identify MAR or MNAR" \
    docs/research/proof_carrying_endogenous_observability_d4_2026-07-15.md || \
    fail "specification lacks the missingness-identifiability boundary"
grep -q "cannot be coerced into a numeric target value" \
    docs/research/proof_carrying_endogenous_observability_d4_2026-07-15.md || \
    fail "specification lacks the absence/value boundary"

# D4 extends D3. The D3 gate recursively includes D2, D1, and D0 surfaces.
bash scripts/ci/proof_carrying_reflexive_inquiry_gate.sh >"$TMP_DIR/d3.log" 2>&1 || {
    cat "$TMP_DIR/d3.log" >&2
    fail "D3/D2/D1/D0 regression gate failed"
}
grep -q "\[reflexive-inquiry-d3\] PASS" "$TMP_DIR/d3.log"

echo "[endogenous-observability-d4] PASS: absence retains its production process without becoming a value or mechanism"
