#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
receipt="$root/scripts/research/receipts/cs6_v7b_target23_decimal_center_replay_v1"
binding="$receipt/execution-binding.txt"
[[ -f $binding ]] || { echo "missing Decimal replay execution binding" >&2; exit 1; }
source_commit=$(awk -F= '$1 == "PRE_EXECUTION_GIT_COMMIT" {print $2}' "$binding")
[[ $source_commit =~ ^[0-9a-f]{40}$ ]] || { echo "invalid pre-execution commit" >&2; exit 1; }
result_dir=$(mktemp -d /tmp/cs6-v7b-t23-decimal-result.XXXXXX)
tar -xzf "$receipt/full-result.tar.gz" -C "$result_dir"

PYTHONDONTWRITEBYTECODE=1 python3 -B \
  "$root/scripts/research/cs6_v7b_target23_decimal_center_replay_verify.py" \
  "$result_dir" --source-commit "$source_commit" > "$receipt/local-verification.txt"
grep -qx 'INDEPENDENT_POINTWISE_SCOUT_COMPLETED=true' "$receipt/local-verification.txt"
grep -qx 'LEAVES_VERIFIED=331' "$receipt/local-verification.txt"
grep -qx 'RIGOROUS_INTERVAL_CERTIFICATE=false' "$receipt/local-verification.txt"
grep -qx 'INDEPENDENT_INTERVAL_ENGINE=false' "$receipt/local-verification.txt"
grep -qx 'GLOBAL_HPG_CERTIFICATE=false' "$receipt/local-verification.txt"
grep -qx 'V7_B_ELIGIBILITY=false' "$receipt/local-verification.txt"
grep -qx 'OPEN_PROBLEM_SOLVED=false' "$receipt/local-verification.txt"
grep -qx 'FPGA_EXECUTION=false' "$receipt/local-verification.txt"

mutation_dir=$(mktemp -d /tmp/cs6-v7b-t23-decimal-mutations.XXXXXX)
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  "$root/scripts/research/cs6_v7b_target23_decimal_center_replay_mutations.py" \
  "$result_dir" --source-commit "$source_commit" --out-dir "$mutation_dir"
cp "$mutation_dir/mutations.tsv" "$receipt/mutations.tsv"
cp "$mutation_dir/mutation-summary.txt" "$receipt/mutation-summary.txt"
grep -qx 'MUTATION_TESTS=14' "$receipt/mutation-summary.txt"
grep -qx 'MUTATIONS_REJECTED=14' "$receipt/mutation-summary.txt"
grep -qx 'MUTATIONS_ESCAPED=0' "$receipt/mutation-summary.txt"
grep -qx 'MUTATION_GATE_PASS=true' "$receipt/mutation-summary.txt"

echo 'CS6_V7B_TARGET23_DECIMAL_CENTER_REPLAY_GATE_PASS=true'
