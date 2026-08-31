#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d /tmp/rebracketing-protocol-d7-gate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    echo "[rebracketing-protocol-d7] FAIL: $*" >&2
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

check_sources=(
    stdlib/epistemic/proof_carrying_rebracketing_protocol.sio
    stdlib/ontology/proof_carrying_rebracketing_protocol.sio
    tests/run-pass/clinical_proof_carrying_rebracketing_protocol_witness.sio
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
    tests/run-pass/clinical_proof_carrying_rebracketing_protocol_native_witness.sio \
    >"$TMP_DIR/native.log" 2>&1; then
    cat "$TMP_DIR/native.log" >&2
    fail "native D7 witness did not execute"
fi
require_line "D7-W0 recorded_cases=3 local_decisions=1 semantic_refusals=1 promotion_abstentions=1 model_replays=1 mismatch_refusals=1" "$TMP_DIR/native.log"
require_line "D7-W1 flat_left=93767706 flat_right=93767706 tree_left=9037326 tree_right=573396 decision_checksum=322833 replay_checksum=10018124" "$TMP_DIR/native.log"
require_line "D7-W2 semantic_source=62559008101 semantic_target=91514259234 difference_bitset=15 reason_mask=31 refusal_checksum=310581870" "$TMP_DIR/native.log"
require_line "D7-W3 promotion_checksum=10018324 reason_mask=63 abstention_checksum=39443487978 global_law=false" "$TMP_DIR/native.log"
require_line "D7-W4 protocol=8700 public=true sealed=false compiler_capability=0 compiler_rewrites=0 contest_ir=0 ontology_transport=0" "$TMP_DIR/native.log"
require_line "D7-W5 requested_occurrence=11003 mismatch_reason=3 model_replay=false refusal_checksum=2174160852" "$TMP_DIR/native.log"
require_line "PROOF-CARRYING REBRACKETING PROTOCOL D7 PASS" "$TMP_DIR/native.log"

if ! bin/souc run tests/run-pass/ontology_proof_carrying_rebracketing_protocol_types.sio \
    >"$TMP_DIR/ontology.log" 2>&1; then
    cat "$TMP_DIR/ontology.log" >&2
    fail "rebracketing protocol ontology witness did not execute"
fi
require_line "proof carrying rebracketing protocol parallel ontology OK" "$TMP_DIR/ontology.log"

if ! python3 scripts/research/proof_carrying_rebracketing_protocol_oracle.py \
    >"$TMP_DIR/oracle.log" 2>&1; then
    cat "$TMP_DIR/oracle.log" >&2
    fail "independent D7 oracle rejected the fixture"
fi
require_line "ORACLE_D7_W0 predicate_vectors=64 admitted_vectors=1 declared_cases=3 recorded_cases=3" "$TMP_DIR/oracle.log"
require_line "ORACLE_D7_W1 flat=93767706,93767706 trees=9037326,573396 label_orders=6 checksum_collisions=0" "$TMP_DIR/oracle.log"
require_line "ORACLE_D7_W2 semantic=62559008101,91514259234 difference_bitset=15 reason_mask=31 refusal_checksum=310581870" "$TMP_DIR/oracle.log"
require_line "ORACLE_D7_W3 decision_checksum=322833 replay_checksum=10018124 wrong_occurrence_refusal_checksum=2174160852" "$TMP_DIR/oracle.log"
require_line "ORACLE_D7_W4 promotion_checksum=10018324 reason_mask=63 abstention_checksum=39443487978" "$TMP_DIR/oracle.log"
require_line "ORACLE_D7_W5 compiler_capabilities=0 compiler_rewrites=0 contest_ir=0 ontology_transport=0" "$TMP_DIR/oracle.log"
require_line "PROOF-CARRYING REBRACKETING PROTOCOL D7 ORACLE PASS" "$TMP_DIR/oracle.log"

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

expect_rejection tests/compile-fail/clinical_local_rebracketing_decision_cannot_claim_global_law.sio \
    GlobalAssociativityLawReceipt LocalRebracketingEqualityDecisionReceipt
expect_rejection tests/compile-fail/clinical_local_rebracketing_decision_cannot_replace_compiler_authority.sio \
    CompilerOwnedRebracketingCapabilityBoundary LocalRebracketingEqualityDecisionReceipt
expect_rejection tests/compile-fail/clinical_flat_rebracketing_decision_cannot_rewrite_semantic_carrier.sio \
    SemanticCarrierRewriteCapabilityBoundary LocalRebracketingEqualityDecisionReceipt
expect_rejection tests/compile-fail/clinical_rebracketing_refusal_cannot_replace_local_decision.sio \
    LocalRebracketingEqualityDecisionReceipt SemanticRebracketingRefusalReceipt
expect_rejection tests/compile-fail/clinical_rebracketing_refusal_cannot_replace_compiler_authority.sio \
    CompilerOwnedRebracketingCapabilityBoundary SemanticRebracketingRefusalReceipt
expect_rejection tests/compile-fail/clinical_compiler_authority_abstention_cannot_claim_global_law.sio \
    GlobalAssociativityLawReceipt CompilerAuthorityPromotionAbstentionReceipt
expect_rejection tests/compile-fail/clinical_local_rebracketing_model_replay_cannot_replace_compiler_authority.sio \
    CompilerOwnedRebracketingCapabilityBoundary LocalRebracketingModelReplayReceipt
expect_rejection tests/compile-fail/clinical_rebracketing_request_cannot_replace_decision.sio \
    LocalRebracketingEqualityDecisionReceipt RebracketingProtocolRequestReceipt
expect_rejection tests/compile-fail/clinical_rebracketing_summary_cannot_claim_proof_assistant_theorem.sio \
    ProofAssistantRebracketingTheoremReceipt RebracketingProtocolSummaryReceipt
expect_rejection tests/compile-fail/clinical_local_rebracketing_decision_cannot_claim_empirical_equivalence.sio \
    EmpiricalRebracketingEquivalenceReceipt LocalRebracketingEqualityDecisionReceipt
expect_rejection tests/compile-fail/clinical_rebracketing_refusal_cannot_claim_causal_mechanism.sio \
    CausalRebracketingMechanismReceipt SemanticRebracketingRefusalReceipt
expect_rejection tests/compile-fail/clinical_local_rebracketing_model_replay_cannot_authorize_action.sio \
    ClinicalRebracketingActionReceipt LocalRebracketingModelReplayReceipt
expect_rejection tests/compile-fail/clinical_fixture_occurrence_refusal_cannot_replace_model_replay.sio \
    LocalRebracketingModelReplayReceipt FixtureOccurrenceReplayRefusalReceipt
expect_rejection tests/compile-fail/clinical_rebracketing_protocol_receipt_cannot_replace_native_contest.sio \
    Contest LocalRebracketingEqualityDecisionReceipt
expect_rejection tests/compile-fail/ontology_local_rebracketing_decision_cannot_replace_global_law.sio \
    GlobalAssociativityLaw LocalRebracketingEqualityDecision
expect_rejection tests/compile-fail/ontology_rebracketing_refusal_cannot_replace_local_decision.sio \
    LocalRebracketingEqualityDecision RebracketingRefusal
expect_rejection tests/compile-fail/ontology_compiler_authority_abstention_cannot_replace_compiler_authority.sio \
    CompilerRebracketingCapabilityClaim CompilerAuthorityPromotionAbstention
expect_rejection tests/compile-fail/ontology_local_equality_witness_cannot_replace_global_law.sio \
    GlobalAssociativityLaw LocalEqualityWitness
expect_rejection tests/compile-fail/ontology_semantic_inequality_witness_cannot_authorize_clinical_action.sio \
    ClinicalRebracketingAction SemanticInequalityWitness

canonical_count="$(awk -F '\t' '$1 == "SOUNIO-REBRACKETING-AUTHORITY" { count++ } END { print count + 0 }' docs/internal/concepts/registry.tsv)"
protocol_count="$(awk -F '\t' '$1 == "SOUNIO-PROOF-CARRYING-REBRACKETING-PROTOCOL" { count++ } END { print count + 0 }' docs/internal/concepts/registry.tsv)"
[[ "$canonical_count" == "1" ]] || fail "canonical rebracketing authority row count changed"
[[ "$protocol_count" == "1" ]] || fail "D7 protocol row count is not exactly one"
grep -Fqx $'SOUNIO-REBRACKETING-AUTHORITY\thypothesis\tfounder\tdocs/internal/concepts/rebracketing-authority.md\tself-hosted/ir/opt_cleanup.sio\tloop-phi-reaching-definition-certificate' \
    docs/internal/concepts/registry.tsv || fail "canonical compiler authority row changed"
grep -Fqx $'SOUNIO-PROOF-CARRYING-REBRACKETING-PROTOCOL\texecutable\tfounder\tdocs/internal/concepts/proof-carrying-rebracketing-protocol.md\tstdlib/epistemic/proof_carrying_rebracketing_protocol.sio\tsealed-receipt-and-compiler-capability-bridge' \
    docs/internal/concepts/registry.tsv || fail "D7 protocol row is not exact"
grep -Fq $'SOUNIO-REBRACKETING-AUTHORITY\tself-hosted/ir/opt_cleanup.sio\tcanonical-ir-transaction' \
    docs/internal/concepts/bindings.tsv || fail "canonical compiler authority binding changed"
grep -Fq $'SOUNIO-PROOF-CARRYING-REBRACKETING-PROTOCOL\tstdlib/epistemic/proof_carrying_rebracketing_protocol.sio\tcanonical-model-protocol' \
    docs/internal/concepts/bindings.tsv || fail "D7 protocol binding is absent"
grep -q "public model receipts are not compiler capabilities" \
    docs/internal/concepts/proof-carrying-rebracketing-protocol.md || \
    fail "concept contract lacks the public-receipt boundary"
grep -q "does not instantiate native.*Contest" \
    docs/internal/concepts/proof-carrying-rebracketing-protocol.md || \
    fail "concept contract lacks the Contest boundary"
grep -q "does not formalize source and target semantics" \
    docs/research/proof_carrying_rebracketing_protocol_d7_2026-07-15.md || \
    fail "specification overstates translation validation"
grep -q "refinement relation" \
    docs/research/proof_carrying_rebracketing_protocol_d7_2026-07-15.md || \
    fail "specification lacks the refinement boundary"
grep -q "linear or single-use consumption" \
    docs/research/proof_carrying_rebracketing_protocol_d7_2026-07-15.md || \
    fail "specification lacks the receipt-reuse boundary"
grep -q "authenticity or unforgeability" \
    docs/research/proof_carrying_rebracketing_protocol_d7_2026-07-15.md || \
    fail "specification lacks the receipt-authenticity boundary"
grep -q "no actual compiler rewrite" \
    docs/research/proof_carrying_rebracketing_protocol_d7_2026-07-15.md || \
    fail "specification lacks the compiler execution boundary"

# D7 extends D6. D6 recursively includes D5, D4, D3, D2, D1, and D0.
bash scripts/ci/proof_carrying_policy_observation_associator_gate.sh \
    >"$TMP_DIR/d6.log" 2>&1 || {
    cat "$TMP_DIR/d6.log" >&2
    fail "D6/D5/D4/D3/D2/D1/D0 regression gate failed"
}
grep -q "\[policy-observation-associator-d6\] PASS" "$TMP_DIR/d6.log"

echo "[rebracketing-protocol-d7] PASS: local evidence cannot become global rewrite authority"
