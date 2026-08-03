#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
receipt="$root/scripts/research/receipts/cs6_u250_validated_dyadic_v1"
generated=$(mktemp -d /tmp/cs6-u250-validated-dyadic.XXXXXXXX)
trap 'rm -rf "$generated"' EXIT

python3 "$root/scripts/research/cs6_u250_validated_dyadic_generate.py" --out-dir "$generated"
cmp "$generated/inputs.bin" "$receipt/inputs.bin"
cmp "$generated/expected.bin" "$receipt/expected.bin"
cmp "$generated/cases.tsv" "$receipt/cases.tsv"
cmp "$generated/summary.txt" "$receipt/summary.txt"
python3 "$root/scripts/research/cs6_u250_validated_dyadic_verify.py" --receipt "$receipt"
python3 "$root/scripts/research/cs6_u250_validated_dyadic_mutations.py" --receipt "$receipt"
echo VALIDATED_DYADIC_GATE_PASS=true
