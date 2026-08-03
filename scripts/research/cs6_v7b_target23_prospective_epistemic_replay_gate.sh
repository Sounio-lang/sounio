#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
receipt="$root/scripts/research/receipts/cs6_v7b_target23_prospective_epistemic_replay_v1"
binding="$receipt/execution-binding.txt"
[[ -f $binding ]] || { echo "missing prospective execution binding" >&2; exit 1; }
source_commit=$(awk -F= '$1 == "PRE_EXECUTION_GIT_COMMIT" {print $2}' "$binding")
[[ $source_commit =~ ^[0-9a-f]{40}$ ]] || { echo "invalid pre-execution commit" >&2; exit 1; }

PYTHONDONTWRITEBYTECODE=1 python3 -B \
  "$root/scripts/research/cs6_v7b_target23_prospective_epistemic_replay_verify.py" \
  "$receipt/result" --source-commit "$source_commit" > "$receipt/local-verification.txt"
grep -qx 'PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED=true' "$receipt/local-verification.txt"
grep -qx 'ATTEMPTS_VERIFIED=662' "$receipt/local-verification.txt"
grep -qx 'LEAF_PAIRS_VERIFIED=331' "$receipt/local-verification.txt"
grep -qx 'GLOBAL_HPG_CERTIFICATE=false' "$receipt/local-verification.txt"
grep -qx 'V7_B_ELIGIBILITY=false' "$receipt/local-verification.txt"
grep -qx 'OPEN_PROBLEM_SOLVED=false' "$receipt/local-verification.txt"
grep -qx 'FPGA_EXECUTION=false' "$receipt/local-verification.txt"

mutation_dir=$(mktemp -d /tmp/cs6-v7b-t23-prospective-mutations.XXXXXX)
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  "$root/scripts/research/cs6_v7b_target23_prospective_epistemic_replay_mutations.py" \
  "$receipt/result" --source-commit "$source_commit" --out-dir "$mutation_dir"
cp "$mutation_dir/mutations.tsv" "$receipt/mutations.tsv"
cp "$mutation_dir/mutation-summary.txt" "$receipt/mutation-summary.txt"
grep -qx 'MUTATION_TESTS=14' "$receipt/mutation-summary.txt"
grep -qx 'MUTATIONS_REJECTED=14' "$receipt/mutation-summary.txt"
grep -qx 'MUTATIONS_ESCAPED=0' "$receipt/mutation-summary.txt"
grep -qx 'MUTATION_GATE_PASS=true' "$receipt/mutation-summary.txt"

echo 'CS6_V7B_TARGET23_PROSPECTIVE_EPISTEMIC_REPLAY_GATE_PASS=true'
