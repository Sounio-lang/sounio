#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d /tmp/policy-observation-d6-gate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    echo "[policy-observation-associator-d6] FAIL: $*" >&2
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
    stdlib/epistemic/proof_carrying_policy_observation_associator.sio
    stdlib/ontology/policy_observation_associator.sio
    tests/run-pass/clinical_proof_carrying_policy_observation_associator_witness.sio
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
    tests/run-pass/clinical_proof_carrying_policy_observation_associator_native_witness.sio \
    >"$TMP_DIR/native.log" 2>&1; then
    cat "$TMP_DIR/native.log" >&2
    fail "native D6 witness did not execute"
fi
grep -q "D6-W0 atoms=policy,boundary,probe ids=9101,9102,9103 flat_fp=93767706" "$TMP_DIR/native.log"
grep -q "D6-W1 left=((a\*b)\*c) target=withheld mask=3 burden=3 evidence_fp=8101 tree_fp=9037326" "$TMP_DIR/native.log"
grep -q "D6-W2 right=(a\*(b\*c)) target=8 mask=2 burden=7 evidence_fp=259234 tree_fp=573396" "$TMP_DIR/native.log"
grep -q "D6-W3 same_operands=true same_operator=9601 both_defined=true difference_bitset=15" "$TMP_DIR/native.log"
grep -q "D6-W4 committed_before_policy=true committed_after_policy=true retroactive_erasure=false" "$TMP_DIR/native.log"
grep -q "D6-W5 flat_control=equal label_orders=6 checksum_collisions=0 grouping_state=distinct irreducible_memory=false" "$TMP_DIR/native.log"
grep -q "D6-W6 scope=partial invalid_pairs=2 total_magma=false causal=false empirical=false clinical=false" "$TMP_DIR/native.log"
grep -q "PROOF-CARRYING POLICY-OBSERVATION ASSOCIATOR D6 PASS" "$TMP_DIR/native.log"

if ! bin/souc run tests/run-pass/ontology_policy_observation_associator_types.sio \
    >"$TMP_DIR/ontology.log" 2>&1; then
    cat "$TMP_DIR/ontology.log" >&2
    fail "policy-observation ontology witness did not execute"
fi
grep -q "policy observation associator ontology OK" "$TMP_DIR/ontology.log"

if ! python3 scripts/research/proof_carrying_policy_observation_associator_oracle.py \
    >"$TMP_DIR/oracle.log" 2>&1; then
    cat "$TMP_DIR/oracle.log" >&2
    fail "independent D6 oracle rejected the fixture"
fi
grep -q "ORACLE_D6_W0 trees=2 applications=4 invalid_pairs=5 flat_fp=93767706" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D6_W1 left=status2,mask3,burden3,fp8101 right=status3,mask2,burden7,fp259234" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D6_W2 difference_bitset=15 committed_input_cases=1 erasures=0" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D6_W3 flat_concat=associative label_orders=6 checksum_collisions=0 grouping_trees=9037326,573396" "$TMP_DIR/oracle.log"
grep -q "PROOF-CARRYING POLICY-OBSERVATION ASSOCIATOR D6 ORACLE PASS" "$TMP_DIR/oracle.log"

expect_rejection() {
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

expect_rejection tests/compile-fail/clinical_policy_observation_associator_cannot_claim_causal_mechanism.sio \
    CausalPolicyObservationMechanismReceipt PolicyObservationNonAssociativityReceipt
expect_rejection tests/compile-fail/clinical_policy_observation_associator_cannot_claim_empirical_order_effect.sio \
    EmpiricalPsychiatricOrderEffectReceipt PolicyObservationNonAssociativityReceipt
expect_rejection tests/compile-fail/clinical_policy_observation_associator_cannot_authorize_action.sio \
    ClinicalPolicyObservationActionReceipt PolicyObservationNonAssociativityReceipt
expect_rejection tests/compile-fail/clinical_partial_policy_observation_operator_cannot_claim_total_magma.sio \
    TotalAssociativeMagmaReceipt PartialPolicyObservationOperatorReceipt
expect_rejection tests/compile-fail/clinical_noncomposable_policy_observation_cannot_compose.sio \
    PolicyObservationComposabilityReceipt PolicyObservationNonComposabilityReceipt
expect_rejection tests/compile-fail/clinical_policy_withheld_target_cannot_replace_committed_observation.sio \
    CommittedSyntheticTargetObservationReceipt PolicyWithheldTargetReceipt
expect_rejection tests/compile-fail/clinical_committed_policy_observation_cannot_claim_positivity.sio \
    StatisticalPolicyPositivityReceipt CommittedSyntheticTargetObservationReceipt
expect_rejection tests/compile-fail/clinical_policy_observation_associator_cannot_claim_suffering.sio \
    PolicyObservationSufferingReceipt PolicyObservationNonAssociativityReceipt
expect_rejection tests/compile-fail/clinical_policy_observation_associator_cannot_claim_consent.sio \
    PolicyObservationConsentReceipt PolicyObservationNonAssociativityReceipt
expect_rejection tests/compile-fail/clinical_flat_policy_trace_cannot_replace_grouping_state.sio \
    GroupingRetainedPolicyObservationStateReceipt FlatOrderedPolicyObservationTraceReceipt
expect_rejection tests/compile-fail/clinical_flat_policy_trace_cannot_claim_complete_state_equivalence.sio \
    CompletePolicyObservationStateEquivalenceReceipt FlatOrderedPolicyObservationTraceReceipt
expect_rejection tests/compile-fail/clinical_policy_withheld_target_cannot_replace_participant_nonresponse_d6.sio \
    RealParticipantNonresponseReceipt PolicyWithheldTargetReceipt
expect_rejection tests/compile-fail/clinical_committed_evidence_monotonicity_cannot_claim_causality.sio \
    CausalPolicyObservationMechanismReceipt CommittedEvidenceMonotonicityReceipt
expect_rejection tests/compile-fail/ontology_policy_withheld_outcome_cannot_replace_committed_observation.sio \
    CommittedSyntheticObservation PolicyWithheldOutcome
expect_rejection tests/compile-fail/ontology_policy_observation_associator_cannot_replace_causal_mechanism.sio \
    CausalPolicyObservationMechanism PolicyObservationNonAssociativityWitness
expect_rejection tests/compile-fail/ontology_partial_policy_composition_cannot_replace_total_magma.sio \
    TotalAssociativeMagma PartialCompositionOperator
expect_rejection tests/compile-fail/ontology_flat_policy_trace_cannot_replace_grouping_state.sio \
    GroupingRetainedState FlatOrderedTrace
expect_rejection tests/compile-fail/ontology_committed_observation_cannot_replace_statistical_positivity_d6.sio \
    StatisticalPositivityReceipt CommittedSyntheticObservation

grep -q $'^SOUNIO-POLICY-OBSERVATION-ASSOCIATOR\texecutable\tfounder\t' \
    docs/internal/concepts/registry.tsv || fail "Concept-ID is not executable"
grep -q "This is a counterexample for a partial operation on one composable triple" \
    docs/research/proof_carrying_policy_observation_associator_d6_2026-07-15.md || \
    fail "specification lacks the partial-operation boundary"
grep -q "ordinary function" \
    docs/research/proof_carrying_policy_observation_associator_d6_2026-07-15.md && \
grep -q "monadic bind" \
    docs/research/proof_carrying_policy_observation_associator_d6_2026-07-15.md || \
    fail "specification lacks the associative-computation rival"
grep -q "monotone only inside the declared fixture" \
    docs/research/proof_carrying_policy_observation_associator_d6_2026-07-15.md || \
    fail "specification overstates evidence monotonicity"
grep -q "does not establish an empirical psychiatric order effect" \
    docs/research/proof_carrying_policy_observation_associator_d6_2026-07-15.md || \
    fail "specification lacks the empirical boundary"

# D6 extends D5. D5 recursively includes D4, D3, D2, D1, and D0.
bash scripts/ci/proof_carrying_policy_state_feedback_gate.sh >"$TMP_DIR/d5.log" 2>&1 || {
    cat "$TMP_DIR/d5.log" >&2
    fail "D5/D4/D3/D2/D1/D0 regression gate failed"
}
grep -q "\[policy-state-feedback-d5\] PASS" "$TMP_DIR/d5.log"

echo "[policy-observation-associator-d6] PASS: grouping scope cannot erase committed evidence"
