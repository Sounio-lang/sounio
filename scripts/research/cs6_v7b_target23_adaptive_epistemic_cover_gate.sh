#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
d4="$root/scripts/research/receipts/cs6_v7b_target23_depth4_cover_v1/full-result.tar.gz"
d5="$root/scripts/research/receipts/cs6_v7b_target23_depth5_boundary_refine_v1/full-result.tar.gz"
receipt="$root/scripts/research/receipts/cs6_v7b_target23_adaptive_epistemic_cover_v1"
analyzer="$root/scripts/research/cs6_v7b_target23_adaptive_epistemic_cover_analyze.py"
verifier="$root/scripts/research/cs6_v7b_target23_adaptive_epistemic_cover_verify.py"
mutations="$root/scripts/research/cs6_v7b_target23_adaptive_epistemic_cover_mutations.py"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

python3 "$analyzer" "$d4" "$d5" "$tmp/generated"
cmp "$tmp/generated/certificates.tsv" "$receipt/certificates.tsv"
cmp "$tmp/generated/leaves.tsv" "$receipt/leaves.tsv"
cmp "$tmp/generated/summary.txt" "$receipt/summary.txt"
python3 "$verifier" "$d4" "$d5" "$receipt"
python3 "$mutations" "$d4" "$d5" "$receipt" "$tmp/mutations"
cmp "$tmp/mutations/mutations.tsv" "$receipt/mutations.tsv"
cmp "$tmp/mutations/mutation-summary.txt" "$receipt/mutation-summary.txt"
sha256sum -c "$receipt/committed-files.sha256"
echo "CS6_V7B_TARGET23_ADAPTIVE_EPISTEMIC_COVER_GATE_PASS=true"
