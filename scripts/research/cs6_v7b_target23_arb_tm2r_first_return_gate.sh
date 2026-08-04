#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
receipt="$root/scripts/research/receipts/cs6_v7b_target23_arb_tm2r_first_return_v1"
worker="$root/scripts/research/cs6_v7b_target23_arb_tm2r_first_return_worker.py"
verifier="$root/scripts/research/cs6_v7b_target23_arb_tm2r_first_return_verify.py"
mkdir -p "$receipt"

: "${CS6_ARB_PYTHONPATH:?set CS6_ARB_PYTHONPATH to a python-flint 0.8.0 target directory}"
PYTHONPATH="$CS6_ARB_PYTHONPATH" PYTHONDONTWRITEBYTECODE=1 \
  python3 -B "$worker" > "$receipt/worker.stdout.txt"
PYTHONDONTWRITEBYTECODE=1 python3 -B "$verifier" \
  "$receipt/worker.stdout.txt" > "$receipt/verification.txt"
grep -qx 'FIRST_RETURN_END_STEP=617' "$receipt/verification.txt"
grep -qx 'FIRST_RETURN_TIME_LOWER_Q=77/32' "$receipt/verification.txt"
grep -qx 'FIRST_RETURN_TIME_UPPER_Q=617/256' "$receipt/verification.txt"
grep -qx 'GENERATOR_RECONSTRUCTIONS=15810' "$receipt/verification.txt"
grep -qx 'INITIAL_DEPARTURE_TUBES=1' "$receipt/verification.txt"
grep -qx 'PRIOR_DOWNWARD_TUBES=1' "$receipt/verification.txt"
grep -qx 'ZERO_FREE_PRIOR_TUBES=614' "$receipt/verification.txt"
grep -qx 'LEAF_GEOMETRY_EXACTLY_RECONSTRUCTED=true' "$receipt/verification.txt"
grep -qx 'STRICT_EVENT_BRACKET_VERIFIED=true' "$receipt/verification.txt"
grep -qx 'STRICT_EVENT_TRANSVERSALITY_VERIFIED=true' "$receipt/verification.txt"
grep -qx 'NO_PRIOR_POSITIVE_RETURN_VERIFIED=true' "$receipt/verification.txt"
grep -qx 'UNIQUE_TARGET_STEP_CROSSING_VERIFIED=true' "$receipt/verification.txt"
grep -qx 'FULL_LEAF_FIRST_RETURN_CERTIFICATE=true' "$receipt/verification.txt"
grep -qx 'FULL_LEAF_SECOND_RETURN_CERTIFICATE=false' "$receipt/verification.txt"
grep -qx 'CHAOS_PROVED=false' "$receipt/verification.txt"
grep -qx 'CHAOTIC_ATTRACTOR_PROVED=false' "$receipt/verification.txt"
grep -qx 'OPEN_PROBLEM_SOLVED=false' "$receipt/verification.txt"

PYTHONDONTWRITEBYTECODE=1 python3 -B \
  "$root/scripts/research/cs6_v7b_target23_arb_tm2r_first_return_mutations.py" \
  "$receipt/worker.stdout.txt" "$verifier" > "$receipt/mutation-summary.txt"
grep -qx 'MUTATION_TESTS=28' "$receipt/mutation-summary.txt"
grep -qx 'MUTATIONS_REJECTED=28' "$receipt/mutation-summary.txt"
grep -qx 'MUTATIONS_ESCAPED=0' "$receipt/mutation-summary.txt"
grep -qx 'MUTATION_GATE_PASS=true' "$receipt/mutation-summary.txt"

(
  cd "$root"
  sha256sum \
    scripts/research/cs6_v7b_target23_arb_tm2r_first_return_contract_v1.txt \
    scripts/research/cs6_v7b_target23_arb_tm2r_first_return_worker.py \
    scripts/research/cs6_v7b_target23_arb_tm2r_first_return_verify.py \
    scripts/research/cs6_v7b_target23_arb_tm2r_first_return_mutations.py \
    scripts/research/receipts/cs6_v7b_target23_arb_tm2r_first_return_v1/worker.stdout.txt \
    scripts/research/receipts/cs6_v7b_target23_arb_tm2r_first_return_v1/verification.txt \
    scripts/research/receipts/cs6_v7b_target23_arb_tm2r_first_return_v1/mutation-summary.txt
) > "$receipt/files.sha256"

echo 'CS6_V7B_TARGET23_ARB_TM2R_FIRST_RETURN_GATE_PASS=true'
