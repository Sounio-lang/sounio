#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
receipt="$root/scripts/research/receipts/cs6_u250_target23_picard_step_v1"
generated=$(mktemp -d /tmp/cs6-u250-target23-picard.XXXXXXXX)
trap 'rm -rf "$generated"' EXIT

python3 "$root/scripts/research/cs6_u250_target23_picard_step_generate.py" --out-dir "$generated"
cmp "$generated/inputs.bin" "$receipt/inputs.bin"
cmp "$generated/expected.bin" "$receipt/expected.bin"
cmp "$generated/cases.tsv" "$receipt/cases.tsv"
cmp "$generated/summary.txt" "$receipt/summary.txt"
python3 "$root/scripts/research/cs6_u250_target23_picard_step_verify.py" --receipt "$receipt"
python3 "$root/scripts/research/cs6_u250_target23_picard_step_mutations.py" --receipt "$receipt"
echo TARGET23_PICARD_STEP_GATE_PASS=true
