#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
receipt="$root/scripts/research/receipts/cs6_v7b_target23_arb_validated_minmargin_v1"
binding="$receipt/execution-binding.txt"
[[ -f $binding ]] || { echo "missing Arb validated execution binding" >&2; exit 1; }
source_commit=$(awk -F= '$1 == "PRE_EXECUTION_GIT_COMMIT" {print $2}' "$binding")
wheel_sha=$(awk -F= '$1 == "PYTHON_FLINT_WHEEL_SHA256" {print $2}' "$binding")
[[ $source_commit =~ ^[0-9a-f]{40}$ && $wheel_sha =~ ^[0-9a-f]{64}$ ]] || {
  echo "invalid Arb validated execution binding" >&2; exit 1;
}
result_dir=$(mktemp -d /tmp/cs6-v7b-t23-arb-result.XXXXXX)
tar -xzf "$receipt/full-result.tar.gz" -C "$result_dir"

PYTHONDONTWRITEBYTECODE=1 python3 -B \
  "$root/scripts/research/cs6_v7b_target23_arb_validated_minmargin_verify.py" \
  "$result_dir" --source-commit "$source_commit" --wheel-sha256 "$wheel_sha" \
  > "$receipt/local-verification.txt"
grep -qx 'INDEPENDENT_VALIDATED_CENTER_ORBIT_CERTIFICATE=true' "$receipt/local-verification.txt"
grep -qx 'CAPD_COMPATIBLE_CENTER_ENCLOSURE=true' "$receipt/local-verification.txt"
grep -qx 'LEAF_WIDE_CERTIFICATE=false' "$receipt/local-verification.txt"
grep -qx 'INDEPENDENT_FULL_LEAF_INTERVAL_ENGINE=false' "$receipt/local-verification.txt"
grep -qx 'GLOBAL_HPG_CERTIFICATE=false' "$receipt/local-verification.txt"
grep -qx 'V7_B_ELIGIBILITY=false' "$receipt/local-verification.txt"
grep -qx 'OPEN_PROBLEM_SOLVED=false' "$receipt/local-verification.txt"
grep -qx 'FPGA_EXECUTION=false' "$receipt/local-verification.txt"

mutation_dir=$(mktemp -d /tmp/cs6-v7b-t23-arb-mutations.XXXXXX)
PYTHONDONTWRITEBYTECODE=1 python3 -B \
  "$root/scripts/research/cs6_v7b_target23_arb_validated_minmargin_mutations.py" \
  "$result_dir" --source-commit "$source_commit" --wheel-sha256 "$wheel_sha" \
  --out-dir "$mutation_dir"
cp "$mutation_dir/mutations.tsv" "$receipt/mutations.tsv"
cp "$mutation_dir/mutation-summary.txt" "$receipt/mutation-summary.txt"
grep -qx 'MUTATION_TESTS=14' "$receipt/mutation-summary.txt"
grep -qx 'MUTATIONS_REJECTED=14' "$receipt/mutation-summary.txt"
grep -qx 'MUTATIONS_ESCAPED=0' "$receipt/mutation-summary.txt"
grep -qx 'MUTATION_GATE_PASS=true' "$receipt/mutation-summary.txt"

echo 'CS6_V7B_TARGET23_ARB_VALIDATED_MINMARGIN_GATE_PASS=true'
