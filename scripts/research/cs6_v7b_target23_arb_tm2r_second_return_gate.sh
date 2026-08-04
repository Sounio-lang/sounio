#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
receipt="$root/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_second_return_v1"
worker="$root/scripts/research/cs6_v7b_target23_arb_tm2r_second_return_worker.py"
verifier="$root/scripts/research/cs6_v7b_target23_arb_tm2r_second_return_verify.py"
mkdir -p "$receipt"

: "${CS6_ARB_PYTHONPATH:?set CS6_ARB_PYTHONPATH to a python-flint 0.8.0 target directory}"
PYTHONPATH="$CS6_ARB_PYTHONPATH" PYTHONDONTWRITEBYTECODE=1 \
  python3 -B "$worker" > "$receipt/worker.stdout.txt"
PYTHONDONTWRITEBYTECODE=1 python3 -B "$verifier" \
  "$receipt/worker.stdout.txt" > "$receipt/verification.txt"
grep -qx 'FIRST_RETURN_END_STEP=617' "$receipt/verification.txt"
grep -qx 'PURE_SOURCE_MONOMIALS_RETAINED=15' "$receipt/verification.txt"
grep -qx 'SECOND_PHASE_ATTEMPTED_STEPS=708' "$receipt/verification.txt"
grep -qx 'SECOND_PHASE_COMPLETED_STEPS=707' "$receipt/verification.txt"
grep -qx 'SECOND_PHASE_COMPLETED_TIME_Q=707/256' "$receipt/verification.txt"
grep -qx 'FAILURE_CLASS=ENDPOINT_ESCAPES_PICARD' "$receipt/verification.txt"
grep -qx 'INTERVAL_NEWTON_TIME_SLAB_VERIFIED=true' "$receipt/verification.txt"
grep -qx 'SIGNED_PICARD_EVENT_FLOW_VERIFIED=true' "$receipt/verification.txt"
grep -qx 'EVENT_POSITION_SLAB_CONTAINMENT_VERIFIED=true' "$receipt/verification.txt"
grep -qx 'SOURCE_VARIABLE_RETENTION_VERIFIED=true' "$receipt/verification.txt"
grep -qx 'EXACT_SECTION_CARRIER_VERIFIED=true' "$receipt/verification.txt"
grep -qx 'SECOND_PHASE_FAIL_CLOSED_VERIFIED=true' "$receipt/verification.txt"
grep -qx 'INTERVAL_NEWTON_EVENT_PROJECTION_CERTIFICATE=true' "$receipt/verification.txt"
grep -qx 'FULL_LEAF_SECOND_RETURN_CERTIFICATE=false' "$receipt/verification.txt"
grep -qx 'CHAOS_PROVED=false' "$receipt/verification.txt"
grep -qx 'OPEN_PROBLEM_SOLVED=false' "$receipt/verification.txt"

PYTHONDONTWRITEBYTECODE=1 python3 -B \
  "$root/scripts/research/cs6_v7b_target23_arb_tm2r_second_return_mutations.py" \
  "$receipt/worker.stdout.txt" "$verifier" > "$receipt/mutation-summary.txt"
grep -qx 'MUTATION_TESTS=35' "$receipt/mutation-summary.txt"
grep -qx 'MUTATIONS_REJECTED=35' "$receipt/mutation-summary.txt"
grep -qx 'MUTATIONS_ESCAPED=0' "$receipt/mutation-summary.txt"
grep -qx 'MUTATION_GATE_PASS=true' "$receipt/mutation-summary.txt"

(
  cd "$root"
  sha256sum \
    scripts/research/cs6_v7b_target23_arb_tm2r_second_return_contract_v1.txt \
    scripts/research/cs6_v7b_target23_arb_tm2r_second_return_worker.py \
    scripts/research/cs6_v7b_target23_arb_tm2r_second_return_verify.py \
    scripts/research/cs6_v7b_target23_arb_tm2r_second_return_mutations.py \
    scripts/research/receipts/cs6_v7b_target23_arb_tm2r_second_return_v1/worker.stdout.txt \
    scripts/research/receipts/cs6_v7b_target23_arb_tm2r_second_return_v1/verification.txt \
    scripts/research/receipts/cs6_v7b_target23_arb_tm2r_second_return_v1/mutation-summary.txt
) > "$receipt/files.sha256"

echo 'CS6_V7B_TARGET23_ARB_TM2R_SECOND_RETURN_GATE_PASS=true'
