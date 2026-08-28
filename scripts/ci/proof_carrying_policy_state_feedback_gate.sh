#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d /tmp/policy-state-feedback-d5-gate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    echo "[policy-state-feedback-d5] FAIL: $*" >&2
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
    stdlib/epistemic/proof_carrying_policy_state_feedback.sio
    stdlib/ontology/policy_state_feedback.sio
    tests/run-pass/clinical_proof_carrying_policy_state_feedback_witness.sio
)

for source in "${check_sources[@]}"; do
    log="$TMP_DIR/$(basename "$source").check.log"
    if ! bin/souc check "$source" >"$log" 2>&1; then
        cat "$log" >&2
        fail "$source did not typecheck"
    fi
    grep -q "check: OK" "$log" || fail "$source lacked the check receipt"
done

if ! bin/souc run tests/run-pass/clinical_proof_carrying_policy_state_feedback_native_witness.sio \
    >"$TMP_DIR/native.log" 2>&1; then
    cat "$TMP_DIR/native.log" >&2
    fail "native D5 witness did not execute"
fi
grep -q "D5-W0 anchor value=2 mask=3 summary=2 ambiguous=true burden=3" "$TMP_DIR/native.log"
grep -q "D5-W1 feedback decision=withhold eligible=0 mask=3 hidden=2|8 trace_equal=true" "$TMP_DIR/native.log"
grep -q "D5-W2 coverage support=absent positivity=false policy_value_identified=false" "$TMP_DIR/native.log"
grep -q "D5-W3 exogenous_low value=2 mask=1 selected=510 burden=7 fingerprint=259234" "$TMP_DIR/native.log"
grep -q "D5-W4 exogenous_high value=8 mask=2 selected=511 burden=7 fingerprint=259234" "$TMP_DIR/native.log"
grep -q "D5-W5 budget cap=7 remaining=4 probe=4 excess=refused" "$TMP_DIR/native.log"
grep -q "D5-W6 policy_erased=abstain disconnected=abstain mask=3" "$TMP_DIR/native.log"
grep -q "D5-W7 relabel=invariant consent=false suffering=false clinical=false fingerprint_max2=32000000" "$TMP_DIR/native.log"
grep -q "PROOF-CARRYING POLICY-STATE FEEDBACK D5 PASS" "$TMP_DIR/native.log"

if ! bin/souc run tests/run-pass/ontology_policy_state_feedback_types.sio \
    >"$TMP_DIR/ontology.log" 2>&1; then
    cat "$TMP_DIR/ontology.log" >&2
    fail "policy-state feedback ontology witness did not execute"
fi
grep -q "policy state feedback ontology OK" "$TMP_DIR/ontology.log"

if ! python3 scripts/research/proof_carrying_policy_state_feedback_oracle.py \
    >"$TMP_DIR/oracle.log" 2>&1; then
    cat "$TMP_DIR/oracle.log" >&2
    fail "independent D5 oracle rejected the fixture"
fi
grep -q "ORACLE_D5_W0 anchor=2 mask=3 summary=2 targets=2|8" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D5_W1 feedback=withhold,withhold traces=equal absorbing=bounded" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D5_W2 coverage=0/0 positivity=false policy_value=false" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D5_W3 exogenous=2->1,8->2 burden=3+4=7 fingerprint=259234" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D5_W4 budget_before=4 budget_after=0 second_probe=refused" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D5_W5 relabelings=2 policy_erased=3 disconnected=3" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D5_W6 exhaustive_actions=48 valid_probes=2 fingerprint_max2=32000000 i64_safe=true" "$TMP_DIR/oracle.log"
grep -q "PROOF-CARRYING POLICY-STATE FEEDBACK D5 ORACLE PASS" "$TMP_DIR/oracle.log"

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
    tests/compile-fail/clinical_policy_summary_cannot_replace_observed_target_value.sio \
    ObservedTargetValueReceipt PolicyStateSummaryReceipt
expect_type_rejection \
    tests/compile-fail/clinical_policy_withholding_cannot_replace_observed_target_value.sio \
    ObservedTargetValueReceipt AdaptiveObservationPolicyDecisionReceipt
expect_type_rejection \
    tests/compile-fail/clinical_observed_anchor_cannot_replace_observed_target_value.sio \
    ObservedTargetValueReceipt ObservedAnchorValueReceipt
expect_type_rejection \
    tests/compile-fail/clinical_policy_coverage_gap_cannot_claim_statistical_positivity.sio \
    StatisticalPositivityReceipt PolicyCoverageGapReceipt
expect_type_rejection \
    tests/compile-fail/clinical_policy_nonidentification_cannot_claim_off_policy_value.sio \
    OffPolicyValueIdentificationReceipt PolicyComparisonNonidentificationReceipt
expect_type_rejection \
    tests/compile-fail/clinical_observation_budget_cannot_claim_consent.sio \
    ConsentReceipt DeclaredSyntheticObservationBudgetReceipt
expect_type_rejection \
    tests/compile-fail/clinical_observation_budget_cannot_claim_ethics_approval.sio \
    EthicsApprovalReceipt DeclaredSyntheticObservationBudgetReceipt
expect_type_rejection \
    tests/compile-fail/clinical_observation_budget_cannot_claim_suffering_d5.sio \
    SubjectiveSufferingReceipt DeclaredSyntheticObservationBudgetReceipt
expect_type_rejection \
    tests/compile-fail/clinical_budget_exceeded_action_cannot_execute_coverage_probe.sio \
    AdmissibleExogenousCoverageProbeReceipt BudgetExceededObservationActionReceipt
expect_type_rejection \
    tests/compile-fail/clinical_policy_feedback_witness_cannot_claim_causal_mechanism.sio \
    CausalFeedbackMechanismReceipt BoundedPolicyStateFeedbackReceipt
expect_type_rejection \
    tests/compile-fail/clinical_within_family_target_identification_cannot_authorize_policy_action.sio \
    ClinicalPolicyActionReceipt DeclaredWithinFamilyTargetIdentificationReceipt
expect_type_rejection \
    tests/compile-fail/clinical_synthetic_coverage_assignment_cannot_claim_real_randomization.sio \
    RealPersonRandomizationReceipt SyntheticExogenousCoverageAssignmentReceipt
expect_type_rejection \
    tests/compile-fail/clinical_disconnected_coverage_probe_cannot_update_policy_state.sio \
    AdmissibleExogenousCoverageProbeReceipt DisconnectedCoverageProbeReceipt
expect_type_rejection \
    tests/compile-fail/clinical_policy_erased_feedback_cannot_replace_bounded_feedback.sio \
    BoundedPolicyStateFeedbackReceipt PolicyErasedFeedbackTraceReceipt
expect_type_rejection \
    tests/compile-fail/ontology_policy_withholding_cannot_replace_observed_target_value.sio \
    ObservedTargetValue PolicyWithholding
expect_type_rejection \
    tests/compile-fail/ontology_policy_coverage_gap_cannot_replace_statistical_positivity.sio \
    StatisticalPositivityReceipt PolicyCoverageGap
expect_type_rejection \
    tests/compile-fail/ontology_observation_budget_cannot_replace_consent.sio \
    ConsentReceipt DeclaredSyntheticObservationBudget

awk -F '\t' '$1 == "SOUNIO-POLICY-STATE-FEEDBACK" && $2 == "executable" { found = 1 } END { exit !found }' \
    docs/internal/concepts/registry.tsv || fail "executable concept registry row is missing"

grep -q "does not establish statistical positivity or global overlap" \
    docs/research/proof_carrying_policy_state_feedback_d5_2026-07-15.md || \
    fail "specification lacks the coverage boundary"
grep -q "A budget refusal is not consent withdrawal" \
    docs/research/proof_carrying_policy_state_feedback_d5_2026-07-15.md || \
    fail "specification lacks the budget/consent boundary"
grep -q "does not yet claim a formal associativity counterexample" \
    docs/research/proof_carrying_policy_state_feedback_d5_2026-07-15.md || \
    fail "specification overstates the non-associativity result"

# D5 extends D4. The D4 gate recursively includes D3, D2, D1, and D0.
bash scripts/ci/proof_carrying_endogenous_observability_gate.sh >"$TMP_DIR/d4.log" 2>&1 || {
    cat "$TMP_DIR/d4.log" >&2
    fail "D4/D3/D2/D1/D0 regression gate failed"
}
grep -q "\[endogenous-observability-d4\] PASS" "$TMP_DIR/d4.log"

echo "[policy-state-feedback-d5] PASS: adaptive observation policy cannot turn its own blind spot into evidence"
