#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d /tmp/dyadic-d1-gate.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT

fail() {
    echo "[dyadic-d1] FAIL: $*" >&2
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
    stdlib/epistemic/dyadic_relational_associator.sio
    stdlib/ontology/relational_dynamics.sio
    tests/run-pass/clinical_dyadic_relational_associator_witness.sio
)

for source in "${check_sources[@]}"; do
    log="$TMP_DIR/$(basename "$source").check.log"
    if ! bin/souc check "$source" >"$log" 2>&1; then
        cat "$log" >&2
        fail "$source did not typecheck"
    fi
    grep -q "check: OK" "$log" || fail "$source lacked the check receipt"
done

if ! bin/souc run tests/run-pass/clinical_dyadic_relational_associator_native_witness.sio \
    >"$TMP_DIR/native.log" 2>&1; then
    cat "$TMP_DIR/native.log" >&2
    fail "native D1 witness did not execute"
fi
grep -q "D1-W0 left=2550/4500=17/30 right=1950/4500=13/30 associator=2700000/20250000=2/15" "$TMP_DIR/native.log"
grep -q "D1-W1 associative_control=9/5,9/5 associator=0" "$TMP_DIR/native.log"
grep -q "D1-W2 grouping=explicit same_rule=true same_ordered_leaves=true" "$TMP_DIR/native.log"
grep -q "D1-W3 expanded_grouping_state=distinct transition_table=total outputs=reproduced" "$TMP_DIR/native.log"
grep -q "boundary=synthetic empirical_law=false causal=false clinical=false irreducible_memory=false" "$TMP_DIR/native.log"
grep -q "SYNTHETIC DYADIC RELATIONAL ASSOCIATOR D1 PASS" "$TMP_DIR/native.log"

if ! bin/souc run tests/run-pass/ontology_relational_dynamics_types.sio \
    >"$TMP_DIR/ontology.log" 2>&1; then
    cat "$TMP_DIR/ontology.log" >&2
    fail "relational-dynamics ontology witness did not execute"
fi
grep -q "relational dynamics ontology OK" "$TMP_DIR/ontology.log"

if ! python3 scripts/research/dyadic_relational_associator_oracle.py \
    >"$TMP_DIR/oracle.log" 2>&1; then
    cat "$TMP_DIR/oracle.log" >&2
    fail "independent D1 oracle rejected the fixture"
fi
grep -q "ORACLE_D1_W0 left=17/30 right=13/30 associator=2/15" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D1_W1 associative_control=9/5 associator=0" "$TMP_DIR/oracle.log"
grep -q "ORACLE_D1_W3 expansion=restores-factorability irreducible=false" "$TMP_DIR/oracle.log"
grep -q "DYADIC RELATIONAL ASSOCIATOR D1 ORACLE PASS" "$TMP_DIR/oracle.log"

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

expect_rejection \
    tests/compile-fail/ontology_participant_state_cannot_replace_dyadic_state.sio \
    DyadicRelationalState ParticipantState
expect_rejection \
    tests/compile-fail/ontology_associative_control_cannot_replace_grouping_sensitive.sio \
    GroupingSensitiveProcess AssociativeControlProcess
expect_rejection \
    tests/compile-fail/ontology_associator_witness_cannot_replace_causal_mechanism.sio \
    CausalMechanismReceipt BoundedAssociatorWitness
expect_rejection \
    tests/compile-fail/clinical_relational_associator_cannot_claim_causal_mechanism.sio \
    CausalRelationalMechanismReceipt ExactRelationalAssociatorReceipt
expect_rejection \
    tests/compile-fail/clinical_relational_associator_cannot_authorize_action.sio \
    ClinicalRelationalAuthorityReceipt ExactRelationalAssociatorReceipt

# D1 extends the D0 lane; it must not invalidate the earlier non-reduction gate.
bash scripts/ci/dyadic_non_reduction_gate.sh >"$TMP_DIR/d0.log" 2>&1 || {
    cat "$TMP_DIR/d0.log" >&2
    fail "D0 regression gate failed"
}
grep -q "\[dyadic-d0\] PASS" "$TMP_DIR/d0.log"

echo "[dyadic-d1] PASS: ontology-bound grouping sensitivity preserves exact claim boundaries"
