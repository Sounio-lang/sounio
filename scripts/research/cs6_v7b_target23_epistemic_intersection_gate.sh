#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
archive="$repo_root/scripts/research/receipts/cs6_v7b_target23_depth5_boundary_refine_v1/full-result.tar.gz"
receipt="$repo_root/scripts/research/receipts/cs6_v7b_target23_epistemic_intersection_v1"
analyzer="$repo_root/scripts/research/cs6_v7b_target23_epistemic_intersection_analyze.py"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

python3 "$analyzer" "$archive" "$tmp"
cmp "$tmp/summary.txt" "$receipt/summary.txt"
cmp "$tmp/attempt_intersections.tsv" "$receipt/attempt_intersections.tsv"
sha256sum -c "$receipt/committed-files.sha256"
echo "CS6_V7B_EPISTEMIC_INTERSECTION_GATE_PASS=true"
