#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d /tmp/path-conditioned-identification-d8-gate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    echo "[path-conditioned-identification-d8] FAIL: $*" >&2
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

check_sources=(
    stdlib/epistemic/proof_carrying_path_conditioned_identification.sio
    stdlib/ontology/path_conditioned_partial_identification.sio
    tests/run-pass/clinical_path_conditioned_partial_identification_witness.sio
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
    tests/run-pass/clinical_path_conditioned_partial_identification_native_witness.sio \
    >"$TMP_DIR/native.log" 2>&1; then
    cat "$TMP_DIR/native.log" >&2
    fail "native D8 witness did not execute"
fi
require_line "D8-W0 states=3 scalar=877 collision_mask=7 family_fingerprint=11917026 collision_checksum=369428683" "$TMP_DIR/native.log"
require_line "D8-W1 history_ab=448033 history_ba=448063 initial_ab=3 initial_ba=6 exact_intersection=2" "$TMP_DIR/native.log"
require_line "D8-W2 evidence_mask=5 refined_ab=1 refined_ba=4 subset_checksums=15515,18492 point_checksums=384337994,384369739" "$TMP_DIR/native.log"
require_line "D8-W3 initial_separation=false refusal_checksum=2650573642 refined_separation=true separation_checksum=12401023" "$TMP_DIR/native.log"
require_line "D8-W4 outer_intersection=2 completion_pairs=9 overlap_completions=4 disjoint_completions=5 undecided_checksum=12090976311463" "$TMP_DIR/native.log"
require_line "D8-W5 missing_result=3 missing_abstention_checksum=12373740403 conflict_result=0 conflict_checksum=24113281431937 nearest_state=false" "$TMP_DIR/native.log"
require_line "D8-W6 provenance=477185883,477229017,477259800 policy_decisions=111567302,111574037,111580772" "$TMP_DIR/native.log"
require_line "D8-W7 d7_cross_occurrence_refusal=2175470118" "$TMP_DIR/native.log"
require_line "D8-W8 model_checksum=11728748010 rebracketing=false compiler_rewrites=0 contest_ir=0 ontology_transport=0" "$TMP_DIR/native.log"
require_line "D8-W9 association=1 intervention=0 counterfactual=0 clinical_action=0 human_suffering=0 sealed=false summary_checksum=386060470608" "$TMP_DIR/native.log"
require_line "PATH-CONDITIONED PARTIAL IDENTIFICATION D8 PASS" "$TMP_DIR/native.log"

if ! bin/souc run tests/run-pass/ontology_path_conditioned_partial_identification_types.sio \
    >"$TMP_DIR/ontology.log" 2>&1; then
    cat "$TMP_DIR/ontology.log" >&2
    fail "path-conditioned partial-identification ontology witness did not execute"
fi
require_line "path conditioned partial identification parallel ontology OK" "$TMP_DIR/ontology.log"

if ! python3 scripts/research/proof_carrying_path_conditioned_identification_oracle.py \
    >"$TMP_DIR/oracle.log" 2>&1; then
    cat "$TMP_DIR/oracle.log" >&2
    fail "independent D8 oracle rejected the fixture"
fi
require_line "ORACLE_D8_W0 states=3 scalar=877 collision_mask=7 family_fingerprint=11917026 collision_checksum=369428683" "$TMP_DIR/oracle.log"
require_line "ORACLE_D8_W1 subset_pairs=64 history_ab=448033 history_ba=448063 initial=3,6 intersection=2" "$TMP_DIR/oracle.log"
require_line "ORACLE_D8_W2 evidence=5 refined=1,4 subset_checksums=15515,18492 point_checksums=384337994,384369739" "$TMP_DIR/oracle.log"
require_line "ORACLE_D8_W3 refinement_triples=343 identity=343 post_disjoint=174 initial_overlap_to_post_disjoint=90" "$TMP_DIR/oracle.log"
require_line "ORACLE_D8_W4 both_refinements_nonempty=205 both_nonempty_post_disjoint=36" "$TMP_DIR/oracle.log"
require_line "ORACLE_D8_W5 sound_outer_tuples=361 disjoint_outer=24 soundness_violations=0 overlap_outer_exact_disjoint=120 overlap_outer_exact_overlap=217" "$TMP_DIR/oracle.log"
require_line "ORACLE_D8_W6 completion_pairs=9 overlap=4 disjoint=5 outer_only_undecided_checksum=12090976311463" "$TMP_DIR/oracle.log"
require_line "ORACLE_D8_W7 missing_result=3 conflict_result=0 conflict_checksum=24113281431937 nearest_state=false" "$TMP_DIR/oracle.log"
require_line "ORACLE_D8_W8 provenance=477185883,477229017,477259800 policy_decisions=111567302,111574037,111580772" "$TMP_DIR/oracle.log"
require_line "ORACLE_D8_W9 d7_reuse_refusal=2175470118 rebracketing=false" "$TMP_DIR/oracle.log"
require_line "ORACLE_D8_W10 compiler_rewrites=0 contest_ir=0 ontology_transport=0" "$TMP_DIR/oracle.log"
require_line "ORACLE_D8_W11 association=1 intervention=0 counterfactual=0 clinical_action=0 human_suffering=0" "$TMP_DIR/oracle.log"
require_line "PATH-CONDITIONED PARTIAL IDENTIFICATION D8 ORACLE PASS" "$TMP_DIR/oracle.log"

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

expect_rejection tests/compile-fail/clinical_scalar_projection_cannot_replace_latent_state_d8.sio \
    D8LatentStateReceipt ScalarProjectionObservationReceipt
expect_rejection tests/compile-fail/clinical_ordered_history_ab_cannot_replace_ba_d8.sio \
    OrderedHistoryBAReceipt OrderedHistoryABReceipt
expect_rejection tests/compile-fail/clinical_ab_set_cannot_enter_ba_path_d8.sio \
    ExactBAIdentifiedSetReceipt ExactABIdentifiedSetReceipt
expect_rejection tests/compile-fail/clinical_sound_outer_ab_cannot_replace_exact_ab_d8.sio \
    ExactABIdentifiedSetReceipt SoundOuterABIdentifiedSetReceipt
expect_rejection tests/compile-fail/clinical_witnessed_inner_cannot_replace_exact_ab_d8.sio \
    ExactABIdentifiedSetReceipt WitnessedInnerIdentifiedSetReceipt
expect_rejection tests/compile-fail/clinical_sound_outer_ab_cannot_replace_sound_outer_ba_d8.sio \
    SoundOuterBAIdentifiedSetReceipt SoundOuterABIdentifiedSetReceipt
expect_rejection tests/compile-fail/clinical_outer_only_overlap_cannot_claim_real_overlap_d8.sio \
    RealIdentifiedSetOverlapReceipt OuterOnlyOverlapUndecidedReceipt
expect_rejection tests/compile-fail/clinical_not_measured_cannot_be_observed_false_d8.sio \
    ObservedFalseReceipt NotMeasuredReceipt
expect_rejection tests/compile-fail/clinical_missing_under_policy_cannot_replace_observation_d8.sio \
    SyntheticDiscriminatingObservationReceipt MissingUnderPolicyReceipt
expect_rejection tests/compile-fail/clinical_association_cannot_claim_intervention_d8.sio \
    InterventionReceipt PathAssociationReceipt
expect_rejection tests/compile-fail/clinical_association_cannot_claim_counterfactual_d8.sio \
    CounterfactualReceipt PathAssociationReceipt
expect_rejection tests/compile-fail/clinical_identified_set_cannot_replace_confidence_region_d8.sio \
    ConfidenceRegionReceipt ExactABIdentifiedSetReceipt
expect_rejection tests/compile-fail/clinical_identified_set_cannot_replace_predictive_set_d8.sio \
    PredictiveSetReceipt ExactABIdentifiedSetReceipt
expect_rejection tests/compile-fail/clinical_identified_set_cannot_replace_value_interval_d8.sio \
    ValueIntervalReceipt ExactABIdentifiedSetReceipt
expect_rejection tests/compile-fail/clinical_identified_set_cannot_replace_pbox_d8.sio \
    PBoxBoundaryReceipt ExactABIdentifiedSetReceipt
expect_rejection tests/compile-fail/clinical_identified_set_cannot_replace_credal_set_d8.sio \
    CredalSetBoundaryReceipt ExactABIdentifiedSetReceipt
expect_rejection tests/compile-fail/clinical_identified_set_cannot_replace_posterior_d8.sio \
    PosteriorReceipt ExactABIdentifiedSetReceipt
expect_rejection tests/compile-fail/clinical_value_interval_cannot_replace_pbox_d8.sio \
    PBoxBoundaryReceipt ValueIntervalReceipt
expect_rejection tests/compile-fail/clinical_pbox_cannot_replace_credal_set_d8.sio \
    CredalSetBoundaryReceipt PBoxBoundaryReceipt
expect_rejection tests/compile-fail/clinical_credal_set_cannot_replace_posterior_d8.sio \
    PosteriorReceipt CredalSetBoundaryReceipt
expect_rejection tests/compile-fail/clinical_nonsingleton_ab_cannot_replace_point_identification_d8.sio \
    ABFinitePointIdentificationReceipt ExactABIdentifiedSetReceipt
expect_rejection tests/compile-fail/clinical_ab_finite_point_cannot_claim_global_state_truth_d8.sio \
    GlobalFunctionalStateTruthReceipt ABFinitePointIdentificationReceipt
expect_rejection tests/compile-fail/clinical_ba_finite_point_cannot_claim_global_state_truth_d8.sio \
    GlobalFunctionalStateTruthReceipt BAFinitePointIdentificationReceipt
expect_rejection tests/compile-fail/clinical_ordered_history_cannot_replace_d7_decision_d8.sio \
    LocalRebracketingEqualityDecisionReceipt OrderedHistoryABReceipt
expect_rejection tests/compile-fail/clinical_individual_context_cannot_replace_dyadic_d8.sio \
    DyadicContextReceipt IndividualContextReceipt
expect_rejection tests/compile-fail/clinical_suffering_proxy_cannot_claim_human_suffering_d8.sio \
    HumanSufferingReceipt SyntheticSufferingProxyReceipt
expect_rejection tests/compile-fail/clinical_model_conflict_cannot_select_latent_state_d8.sio \
    D8LatentStateReceipt ModelEvidenceConflictReceipt
expect_rejection tests/compile-fail/clinical_d8_summary_cannot_authorize_action_d8.sio \
    ClinicalActionReceipt PathConditionedIdentificationSummaryReceipt
expect_rejection tests/compile-fail/clinical_d8_summary_cannot_replace_compiler_authority_d8.sio \
    CompilerOwnedRebracketingCapabilityBoundary PathConditionedIdentificationSummaryReceipt
expect_rejection tests/compile-fail/clinical_d8_summary_cannot_replace_native_contest_d8.sio \
    Contest PathConditionedIdentificationSummaryReceipt
expect_rejection tests/compile-fail/clinical_exact_set_separation_cannot_claim_intervention_d8.sio \
    InterventionReceipt ExactSeparationReceipt
expect_rejection tests/compile-fail/clinical_exact_separation_cannot_authorize_action_d8.sio \
    ClinicalActionReceipt ExactSeparationReceipt
expect_rejection tests/compile-fail/clinical_initial_compatibility_cannot_replace_exact_separation_d8.sio \
    ExactSeparationReceipt WitnessedInitialCompatibilityReceipt
expect_rejection tests/compile-fail/clinical_heuristic_set_cannot_replace_exact_ab_d8.sio \
    ExactABIdentifiedSetReceipt HeuristicIdentifiedSetReceipt

expect_rejection tests/compile-fail/ontology_exact_identified_set_cannot_replace_confidence_region_d8.sio \
    ConfidenceRegion ExactIdentifiedSet
expect_rejection tests/compile-fail/ontology_exact_identified_set_cannot_replace_pbox_d8.sio \
    PBox ExactIdentifiedSet
expect_rejection tests/compile-fail/ontology_sound_outer_cannot_replace_exact_identified_set_d8.sio \
    ExactIdentifiedSet SoundOuterIdentifiedSet
expect_rejection tests/compile-fail/ontology_witnessed_inner_cannot_replace_exact_identified_set_d8.sio \
    ExactIdentifiedSet WitnessedInnerIdentifiedSet
expect_rejection tests/compile-fail/ontology_outer_only_undecided_cannot_replace_real_overlap_d8.sio \
    RealIdentifiedSetOverlap OuterOnlyOverlapUndecided
expect_rejection tests/compile-fail/ontology_not_measured_cannot_replace_observed_false_d8.sio \
    ObservedFalse NotMeasured
expect_rejection tests/compile-fail/ontology_association_cannot_replace_intervention_d8.sio \
    InterventionReceipt AssociationReceipt
expect_rejection tests/compile-fail/ontology_association_cannot_replace_counterfactual_d8.sio \
    CounterfactualReceipt AssociationReceipt
expect_rejection tests/compile-fail/ontology_individual_context_cannot_replace_dyadic_d8.sio \
    DyadicContext IndividualContext
expect_rejection tests/compile-fail/ontology_suffering_proxy_cannot_replace_human_suffering_d8.sio \
    HumanSuffering SufferingProxy
expect_rejection tests/compile-fail/ontology_history_ab_cannot_replace_history_ba_d8.sio \
    OrderedHistoryBA OrderedHistoryAB
expect_rejection tests/compile-fail/ontology_exact_separation_cannot_authorize_clinical_action_d8.sio \
    ClinicalAction ExactSeparation
expect_rejection tests/compile-fail/ontology_model_conflict_cannot_select_latent_state_d8.sio \
    LatentState ModelEvidenceConflict

d7_count="$(awk -F '\t' '$1 == "SOUNIO-PROOF-CARRYING-REBRACKETING-PROTOCOL" { count++ } END { print count + 0 }' docs/internal/concepts/registry.tsv)"
d8_count="$(awk -F '\t' '$1 == "SOUNIO-PATH-CONDITIONED-PARTIAL-IDENTIFICATION" { count++ } END { print count + 0 }' docs/internal/concepts/registry.tsv)"
[[ "$d7_count" == "1" ]] || fail "D7 protocol row count changed"
[[ "$d8_count" == "1" ]] || fail "D8 concept row count is not exactly one"
grep -Fqx $'SOUNIO-PROOF-CARRYING-REBRACKETING-PROTOCOL\texecutable\tfounder\tdocs/internal/concepts/proof-carrying-rebracketing-protocol.md\tstdlib/epistemic/proof_carrying_rebracketing_protocol.sio\tsealed-receipt-and-compiler-capability-bridge' \
    docs/internal/concepts/registry.tsv || fail "D7 protocol row changed"
grep -Fqx $'SOUNIO-PATH-CONDITIONED-PARTIAL-IDENTIFICATION\texecutable\tfounder\tdocs/internal/concepts/path-conditioned-partial-identification.md\tstdlib/epistemic/proof_carrying_path_conditioned_identification.sio\tstatistical-coverage-and-empirical-state-binding' \
    docs/internal/concepts/registry.tsv || fail "D8 concept row is not exact"

required_bindings=(
    $'docs/internal/concepts/path-conditioned-partial-identification.md\tconcept-contract'
    $'docs/research/proof_carrying_path_conditioned_identification_d8_2026-07-18.md\tresearch-specification'
    $'stdlib/epistemic/proof_carrying_path_conditioned_identification.sio\tcanonical-model-kernel'
    $'stdlib/ontology/path_conditioned_partial_identification.sio\tparallel-ontology-boundary'
    $'tests/run-pass/clinical_path_conditioned_partial_identification_*\tpositive-evidence'
    $'tests/run-pass/ontology_path_conditioned_partial_identification_types.sio\tparallel-ontology-evidence'
    $'tests/compile-fail/clinical_*_d8.sio\tnegative-evidence'
    $'tests/compile-fail/ontology_*_d8.sio\tnegative-evidence'
    $'scripts/research/proof_carrying_path_conditioned_identification_oracle.py\tindependent-oracle'
    $'scripts/ci/proof_carrying_path_conditioned_identification_gate.sh\tacceptance-gate'
)
for binding in "${required_bindings[@]}"; do
    grep -Fqx $'SOUNIO-PATH-CONDITIONED-PARTIAL-IDENTIFICATION\t'"$binding" \
        docs/internal/concepts/bindings.tsv || fail "missing D8 binding: $binding"
done

grep -q "enumeration of all compatible members" \
    docs/internal/concepts/path-conditioned-partial-identification.md || \
    fail "concept contract lacks the finite exactness boundary"
grep -q "not a confidence region" \
    docs/internal/concepts/path-conditioned-partial-identification.md || \
    fail "concept contract lacks the statistical-set boundary"
grep -q "alternative branches from the same root" \
    docs/internal/concepts/path-conditioned-partial-identification.md || \
    fail "concept contract lacks the provenance branch boundary"
grep -q "does not select a nearest state" \
    docs/internal/concepts/path-conditioned-partial-identification.md || \
    fail "concept contract lacks the empty-set conflict boundary"
grep -q "No reviewed source" \
    docs/research/proof_carrying_path_conditioned_identification_d8_2026-07-18.md || \
    fail "specification lacks the novelty boundary"
grep -q "complete D8 graph separately carries" \
    docs/research/proof_carrying_path_conditioned_identification_d8_2026-07-18.md || \
    fail "specification overstates outer-only undecidedness"
grep -q "not a cryptographic seal" \
    docs/research/proof_carrying_path_conditioned_identification_d8_2026-07-18.md || \
    fail "specification lacks the checksum boundary"
grep -q "not two jointly observed outcomes" \
    docs/internal/concepts/path-conditioned-partial-identification.md || \
    fail "concept contract lacks the causal boundary"
grep -q "imported reusable module is check-only" \
    docs/research/proof_carrying_path_conditioned_identification_d8_2026-07-18.md || \
    fail "specification lacks the imported-runtime boundary"

# D8 extends D7. D7 recursively includes D6, D5, D4, D3, D2, D1, and D0.
bash scripts/ci/proof_carrying_rebracketing_protocol_gate.sh \
    >"$TMP_DIR/d7.log" 2>&1 || {
    cat "$TMP_DIR/d7.log" >&2
    fail "D7/D6/D5/D4/D3/D2/D1/D0 regression gate failed"
}
grep -q "\[rebracketing-protocol-d7\] PASS" "$TMP_DIR/d7.log" || \
    fail "recursive D7 gate lacked its PASS receipt"

echo "[path-conditioned-identification-d8] PASS: path-conditioned sets cannot become patient or causal authority"
