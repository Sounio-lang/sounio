#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
receipt="$root/scripts/research/receipts/cs6_u250_target23_scaled_taylor16_v1"
generated=$(mktemp -d /tmp/cs6-scaled-taylor16.XXXXXXXX)
trap 'rm -rf "$generated"' EXIT

bash "$root/scripts/research/cs6_u250_target23_picard_step_gate.sh" >/dev/null
python3 "$root/scripts/research/cs6_u250_target23_scaled_taylor16_generate.py" --out-dir "$generated"
cmp "$generated/inputs.bin" "$receipt/inputs.bin"
cmp "$generated/expected.bin" "$receipt/expected.bin"
cmp "$generated/cases.tsv" "$receipt/cases.tsv"
cmp "$generated/summary.txt" "$receipt/summary.txt"
python3 "$root/scripts/research/cs6_u250_target23_scaled_taylor16_verify.py" --receipt "$receipt"
python3 "$root/scripts/research/cs6_u250_target23_scaled_taylor16_mutations.py" --receipt "$receipt"
echo TARGET23_SCALED_TAYLOR16_GATE_PASS=true
