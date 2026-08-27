#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d /tmp/reflexive-inquiry-d3-gate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    echo "[reflexive-inquiry-d3] FAIL: $*" >&2
    exit 1
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

check_sources=(
    stdlib/epistemic/proof_carrying_reflexive_inquiry.sio
    stdlib/ontology/reflexive_inquiry.sio
    tests/run-pass/clinical_proof_carrying_reflexive_inquiry_witness.sio
)

for source in "${check_sources[@]}"; do
    log="$TMP_DIR/$(basename "$source").check.log"
    if ! bin/souc check "$source" >"$log" 2>&1; then
        cat "$log" >&2
        fail "$source did not typecheck"
    fi
    grep -q "check: OK" "$log" || fail "$source lacked the check receipt"
done

if ! bin/souc run tests/run-pass/clinical_proof_carrying_reflexive_inquiry_native_witness.sio \
    >"$TMP_DIR/native.log" 2>&1; then
    cat "$TMP_DIR/native.log" >&2
    fail "native D3 witness did not execute"
fi
grep -q "D3-W0 passive PQ=1,1 QP=1,1 layers_and_values_commute=true" "$TMP_DIR/native.log"
grep -q "D3-W1 relation PQ=2,3 QP=2,4 locus=relational commutator=-1" "$TMP_DIR/native.log"
grep -q "D3-W2 instrument PQ=2,3 QP=2,4 locus=instrument commutator=-1" "$TMP_DIR/native.log"
grep -q "D3-W3 overlap Q,R final=4 traces=2,4|3,4 layer_commutator=0 value_reorder=false" "$TMP_DIR/native.log"
grep -q "D3-W4 order_trace mask=6 ambiguous=relational,instrument" "$TMP_DIR/native.log"
grep -q "D3-W5 audit relational=3 mask=2 selected=211 burden=7 fingerprint=195233" "$TMP_DIR/native.log"
grep -q "D3-W6 full_state=deterministic projection_sufficient=false nonassociativity=false clinical=false" "$TMP_DIR/native.log"
grep -q "D3-W7 missing=7 unaudited=7 relabel=invariant fingerprint_max2=32000000" "$TMP_DIR/native.log"
grep -q "PROOF-CARRYING REFLEXIVE INQUIRY D3 PASS" "$TMP_DIR/native.log"

if ! bin/souc run tests/run-pass/ontology_reflexive_inquiry_types.sio \
    >"$TMP_DIR/ontology.log" 2>&1; then
    cat "$TMP_DIR/ontology.log" >&2
    fail "reflexive inquiry ontology witness did not execute"
fi
grep -q "reflexive inquiry ontology OK" "$TMP_DIR/ontology.log"

if ! python3 scripts/research/proof_carrying_reflexive_inquiry_oracle.py \
    >"$TMP_DIR/oracle.log" 2>&1; then
    cat "$TMP_DIR/oracle.log" >&2
    fail "independent D3 oracle rejected the fixture"
fi
grep -q "ORACLE_D3_W0 schedules=6 passive_all_pairs=layers_and_values_commute" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D3_W1 P,Q relation=-1 instrument=-1 projected_traces_identical" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D3_W2 Q,R overlap=true final_layers_commute emitted_values_commute=false" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D3_W3 order_trace=2,3,2,4 survivors=relational,instrument" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D3_W4 relational_audit=3 survivor=relational instrument_audit=2 survivor=instrument" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D3_W5 relabelings=6 partitions=invariant missing=7 unaudited=7" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D3_W6 fingerprint=195233 fingerprint_max2=32000000 i64_safe=true" "$TMP_DIR/oracle.log"
grep -q "PROOF-CARRYING REFLEXIVE INQUIRY D3 ORACLE PASS" "$TMP_DIR/oracle.log"

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
    tests/compile-fail/clinical_missing_order_trace_cannot_update_reflexive_contest.sio \
    AdmissibleOrderTraceObservationReceipt MissingOrderTraceObservationReceipt
expect_type_rejection \
    tests/compile-fail/clinical_unaudited_order_trace_cannot_update_reflexive_contest.sio \
    AdmissibleOrderTraceObservationReceipt UnauditedOrderTraceObservationReceipt
expect_type_rejection \
    tests/compile-fail/clinical_inquiry_interference_cannot_claim_noncommutation.sio \
    ExactInquiryNonCommutationWitnessReceipt PotentialInquiryInterferenceReceipt
expect_type_rejection \
    tests/compile-fail/clinical_inquiry_noncommutation_cannot_claim_nonassociativity.sio \
    NonAssociativeInquiryReceipt ExactInquiryNonCommutationWitnessReceipt
expect_type_rejection \
    tests/compile-fail/clinical_state_only_commutation_cannot_authorize_inquiry_reorder.sio \
    InquiryReorderAuthorizationReceipt OverlappingWriteStateCommutationReceipt
expect_type_rejection \
    tests/compile-fail/clinical_reactive_locus_ambiguity_cannot_claim_biological_mechanism.sio \
    BiologicalReactivityMechanismReceipt ReactiveLocusAmbiguityReceipt
expect_type_rejection \
    tests/compile-fail/clinical_declared_locus_identification_cannot_authorize_action.sio \
    ClinicalInquiryActionReceipt DeclaredLocusIdentificationReceipt
expect_type_rejection \
    tests/compile-fail/clinical_matched_synthetic_pair_cannot_claim_real_counterfactual.sio \
    RealPersonCounterfactualReceipt MatchedSyntheticProtocolPairReceipt
expect_type_rejection \
    tests/compile-fail/clinical_synthetic_order_trace_cannot_claim_physical_observation.sio \
    PhysicalInquiryObservationReceipt AdmissibleOrderTraceObservationReceipt
expect_type_rejection \
    tests/compile-fail/clinical_inquiry_burden_cannot_claim_suffering.sio \
    SubjectiveInquirySufferingReceipt SyntheticLayerAuditProbeReceipt
expect_type_rejection \
    tests/compile-fail/ontology_inquiry_noncommutation_cannot_replace_nonassociativity.sio \
    NonAssociativityReceipt ExactNonCommutationReceipt
expect_type_rejection \
    tests/compile-fail/ontology_reactive_locus_ambiguity_cannot_replace_biological_mechanism.sio \
    BiologicalMechanismReceipt ReactiveLocusAmbiguity

awk -F '\t' '$1 == "SOUNIO-REFLEXIVE-INQUIRY" && $2 == "executable" { found = 1 } END { exit !found }' \
    docs/internal/concepts/registry.tsv || fail "executable concept registry row is missing"

# D3 extends D2. The D2 gate itself includes the D1 relational-associator gate.
bash scripts/ci/proof_carrying_model_contest_gate.sh >"$TMP_DIR/d2.log" 2>&1 || {
    cat "$TMP_DIR/d2.log" >&2
    fail "D2/D1 regression gate failed"
}
grep -q "\[proof-carrying-d2\] PASS" "$TMP_DIR/d2.log"

echo "[reflexive-inquiry-d3] PASS: inquiry order is retained without reifying its locus or authority"
