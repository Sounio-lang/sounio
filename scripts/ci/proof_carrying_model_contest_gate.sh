#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d /tmp/proof-carrying-d2-gate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    echo "[proof-carrying-d2] FAIL: $*" >&2
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
    stdlib/epistemic/proof_carrying_model_contest.sio
    stdlib/ontology/proof_carrying_inference.sio
    tests/run-pass/clinical_proof_carrying_model_contest_witness.sio
)

for source in "${check_sources[@]}"; do
    log="$TMP_DIR/$(basename "$source").check.log"
    if ! bin/souc check "$source" >"$log" 2>&1; then
        cat "$log" >&2
        fail "$source did not typecheck"
    fi
    grep -q "check: OK" "$log" || fail "$source lacked the check receipt"
done

if ! bin/souc run tests/run-pass/clinical_proof_carrying_model_contest_native_witness.sio \
    >"$TMP_DIR/native.log" 2>&1; then
    cat "$TMP_DIR/native.log" >&2
    fail "native D2 witness did not execute"
fi
grep -q "D2-W0 initial_mask=15 missing_mask=15 unaudited_mask=15 disconnected_mask=15 abstentions=typed" "$TMP_DIR/native.log"
grep -q "D2-W1 A600 mask=12 count=2 burden=2 fingerprint=9101" "$TMP_DIR/native.log"
grep -q "D2-W2 C650 mask=8 count=1 selected=113 burden=7 fingerprint=291233" "$TMP_DIR/native.log"
grep -q "D2-W3 policy=A_to_B_or_C worst=7 preset=10 exhaustive=depth2" "$TMP_DIR/native.log"
grep -q "D2-W4 B700 mask=0 family_refuted=true nearest=false" "$TMP_DIR/native.log"
grep -q "D2-W5 relabel=partition_invariant global=false causal=false clinical=false suffering=false" "$TMP_DIR/native.log"
grep -q "D2-W6 fingerprint_max8=28429701248000000 i64_safe=true" "$TMP_DIR/native.log"
grep -q "PROOF-CARRYING EPISTEMIC CONTEST D2 PASS" "$TMP_DIR/native.log"

if ! bin/souc run tests/run-pass/ontology_proof_carrying_inference_types.sio \
    >"$TMP_DIR/ontology.log" 2>&1; then
    cat "$TMP_DIR/ontology.log" >&2
    fail "proof-carrying inference ontology witness did not execute"
fi
grep -q "proof carrying inference ontology OK" "$TMP_DIR/ontology.log"

if ! python3 scripts/research/proof_carrying_model_contest_oracle.py \
    >"$TMP_DIR/oracle.log" 2>&1; then
    cat "$TMP_DIR/oracle.log" >&2
    fail "independent D2 oracle rejected the fixture"
fi
grep -q "ORACLE_D2_W0 policies=15 complete=1 root=A children=B,C" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D2_W1 worst=7 preset=10 one_probe=false" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D2_W2 A600=H2,H3 C650=H3 fingerprint=291233" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D2_W3 missing=unchanged unaudited=unchanged disconnected=unchanged B700=family_refuted" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D2_W4 relabel=partition_invariant scope=declared_family" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D2_W5 fingerprint_max8=28429701248000000 i64_safe=true" "$TMP_DIR/oracle.log"
grep -q "PROOF-CARRYING MODEL CONTEST D2 ORACLE PASS" "$TMP_DIR/oracle.log"

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
    tests/compile-fail/clinical_missing_observation_cannot_update_model_contest.sio \
    AdmissibleContestObservationReceipt MissingContestObservationReceipt
expect_type_rejection \
    tests/compile-fail/clinical_unaudited_observation_cannot_update_model_contest.sio \
    AdmissibleContestObservationReceipt UnauditedContestObservationReceipt
expect_type_rejection \
    tests/compile-fail/clinical_finite_family_identification_cannot_claim_global_truth.sio \
    GlobalModelTruthReceipt FiniteFamilyIdentificationReceipt
expect_type_rejection \
    tests/compile-fail/clinical_finite_family_identification_cannot_claim_causality.sio \
    CausalContestMechanismReceipt FiniteFamilyIdentificationReceipt
expect_type_rejection \
    tests/compile-fail/clinical_finite_family_identification_cannot_authorize_action.sio \
    ClinicalContestActionReceipt FiniteFamilyIdentificationReceipt
expect_type_rejection \
    tests/compile-fail/clinical_declared_probe_burden_cannot_claim_suffering.sio \
    SubjectiveSufferingReceipt AdaptiveBurdenPolicyReceipt
expect_type_rejection \
    tests/compile-fail/clinical_synthetic_probe_cannot_claim_physical_intervention.sio \
    PhysicalContestInterventionReceipt SyntheticContestProbeReceipt
expect_type_rejection \
    tests/compile-fail/clinical_epistemic_abstention_cannot_claim_identification.sio \
    FiniteFamilyIdentificationReceipt EpistemicAbstentionReceipt
expect_type_rejection \
    tests/compile-fail/clinical_declared_family_refutation_cannot_claim_global_truth.sio \
    GlobalModelTruthReceipt DeclaredFamilyRefutationReceipt
expect_type_rejection \
    tests/compile-fail/ontology_missing_observation_cannot_replace_admissible_observation.sio \
    AdmissibleObservation MissingObservation
expect_type_rejection \
    tests/compile-fail/ontology_family_refutation_cannot_replace_global_truth.sio \
    GlobalTruthReceipt DeclaredFamilyRefutation

awk -F '\t' '$1 == "SOUNIO-PROOF-CARRYING-INFERENCE" && $2 == "executable" { found = 1 } END { exit !found }' \
    docs/internal/concepts/registry.tsv || fail "executable concept registry row is missing"

# D2 extends the ontology-bound D1 lane and must preserve its exact boundaries.
bash scripts/ci/dyadic_relational_associator_gate.sh >"$TMP_DIR/d1.log" 2>&1 || {
    cat "$TMP_DIR/d1.log" >&2
    fail "D1 regression gate failed"
}
grep -q "\[dyadic-d1\] PASS" "$TMP_DIR/d1.log"

echo "[proof-carrying-d2] PASS: admissible evidence updates exact version spaces without authority leakage"
