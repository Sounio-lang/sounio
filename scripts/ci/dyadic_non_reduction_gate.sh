#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

KERNEL="stdlib/epistemic/dyadic_non_reduction.sio"
API_WITNESS="tests/run-pass/clinical_dyadic_non_reduction_witness.sio"
NATIVE_WITNESS="tests/run-pass/clinical_dyadic_non_reduction_native_witness.sio"
ORACLE="scripts/research/dyadic_non_reduction_oracle.py"
TMP_DIR="$(mktemp -d /tmp/dyadic-d0-gate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    echo "[dyadic-d0] FAIL: $*" >&2
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

for source in "$KERNEL" "$API_WITNESS"; do
    log="$TMP_DIR/$(basename "$source").check.log"
    if ! bin/souc check "$source" >"$log" 2>&1; then
        cat "$log" >&2
        fail "$source did not typecheck"
    fi
    grep -q "check: OK" "$log" || fail "$source lacked the check receipt"
done

if ! bin/souc run "$NATIVE_WITNESS" >"$TMP_DIR/native.log" 2>&1; then
    cat "$TMP_DIR/native.log" >&2
    fail "native D0 witness did not execute"
fi
grep -q "D0-W0 current=500/1000 common_probe=A future=400/1000,600/1000 difference=-1/5" "$TMP_DIR/native.log"
grep -q "D0-W1 factorable_null=true residual_blocks=1 global_equivalence=false" "$TMP_DIR/native.log"
grep -q "D0-W2 permutation=true history_id_leak=rejected reverse_order=true" "$TMP_DIR/native.log"
grep -q "D0-W3 one_step_separates=false adaptive_cost=2 preset_cost=3 incomplete_refused=true" "$TMP_DIR/native.log"
grep -q "D0-W4 context_rival=explains-control markov_expansion=restores-factorability" "$TMP_DIR/native.log"
grep -q "boundary=participant_product_nonreduction history_unbounded_irreducibility=false" "$TMP_DIR/native.log"
grep -q "synthetic=true relationship=false suffering=false consent=false clinical=false" "$TMP_DIR/native.log"
grep -q "SYNTHETIC DYADIC NON-REDUCTION D0 PASS" "$TMP_DIR/native.log"

if ! python3 "$ORACLE" >"$TMP_DIR/oracle.log" 2>&1; then
    cat "$TMP_DIR/oracle.log" >&2
    fail "independent D0 oracle rejected the fixture"
fi
grep -q "ORACLE_D0_W0 difference=-1/5 common_probe=A" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D0_W3 one_step=false adaptive_cost=2 preset_cost=3 incomplete=true" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D0_W4 context_rival=true markov_expansion=true unbounded=false" "$TMP_DIR/oracle.log"
grep -q "DYADIC NON-REDUCTION D0 ORACLE PASS" "$TMP_DIR/oracle.log"

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
    tests/compile-fail/clinical_participant_pair_cannot_replace_dyadic_state.sio \
    DyadicCandidateStateReceipt PairOfParticipantStates
expect_type_rejection \
    tests/compile-fail/clinical_dyadic_collision_cannot_claim_relational_identity.sio \
    RelationalIdentityReceipt DyadicProjectionCollisionReceipt
expect_type_rejection \
    tests/compile-fail/clinical_dyadic_ambiguity_cannot_claim_global_equivalence.sio \
    GlobalDyadicEquivalenceReceipt HorizonLimitedDyadicAmbiguityReceipt
expect_type_rejection \
    tests/compile-fail/clinical_dyadic_non_reduction_cannot_claim_causal_mechanism.sio \
    CausalRelationalMechanismReceipt DyadicNonReductionWitness
expect_type_rejection \
    tests/compile-fail/clinical_dyadic_non_reduction_cannot_claim_suffering.sio \
    SubjectiveSufferingReceipt DyadicNonReductionWitness
expect_type_rejection \
    tests/compile-fail/clinical_dyadic_non_reduction_cannot_claim_consent.sio \
    DyadicConsentReceipt DyadicNonReductionWitness
expect_type_rejection \
    tests/compile-fail/clinical_dyadic_non_reduction_cannot_authorize_clinical_action.sio \
    ClinicalRelationalActionReceipt DyadicNonReductionWitness
expect_type_rejection \
    tests/compile-fail/clinical_incomplete_dyadic_search_cannot_claim_minimality.sio \
    BoundedDyadicMinimalityReceipt DyadicSearchIncompleteReceipt
expect_type_rejection \
    tests/compile-fail/clinical_leaking_dyadic_schema_cannot_claim_valid_observation.sio \
    DyadicObservationSchemaReceipt LeakingDyadicObservationSchemaReceipt

echo "[dyadic-d0] PASS: retained relational history separates bounded synthetic dyads without authority leakage"
