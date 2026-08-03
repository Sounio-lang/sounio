#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
receipt="$root/scripts/research/receipts/cs6_v7b_target23_u250_batch_v1"
reference=$(mktemp -d /tmp/cs6-v7b-target23-u250-reference.XXXXXXXX)
trap 'rm -rf "$reference"' EXIT

python3 "$root/scripts/research/cs6_v7b_target23_u250_batch_generate.py" \
  --out-dir "$reference"
cmp "$reference/reference.tsv" "$receipt/reference.tsv"
python3 "$root/scripts/research/cs6_v7b_target23_u250_batch_verify.py" \
  --receipt "$receipt"
python3 "$root/scripts/research/cs6_v7b_target23_u250_batch_mutations.py" \
  --receipt "$receipt"
echo TARGET23_U250_BATCH_GATE_PASS=true
