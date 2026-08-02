#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

receipt_dir=scripts/research/receipts/cs6_v7b_full_hpg_bridge_anatomy_v1
mkdir -p "$receipt_dir"
rm -f "$receipt_dir/anatomy.tsv" "$receipt_dir/files.sha256" "$receipt_dir/summary.txt" "$receipt_dir/run.stdout.txt"

PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/research/cs6_v7b_full_hpg_bridge_analyze.py \
  --execution-dir scripts/research/receipts/cs6_v7b_full_hpg_bridge_execution_v1 \
  --out-dir "$receipt_dir" | tee "$receipt_dir/run.stdout.txt"

grep -Fxq 'RUN_VALID=true' "$receipt_dir/summary.txt"
grep -Fxq 'ATTEMPTS_ANALYZED=6' "$receipt_dir/summary.txt"
grep -Fxq 'WORKER_SUMMARIES_EMITTED=4' "$receipt_dir/summary.txt"
grep -Fxq 'SECTION_RESIDENT_CROSSING_UNAVAILABLE=2' "$receipt_dir/summary.txt"
grep -Fxq 'C1_C2_ORIENTATION_UNRESOLVED=4' "$receipt_dir/summary.txt"
grep -Fxq 'ORIENTATION_CERTIFICATE_REJECTED=0' "$receipt_dir/summary.txt"
grep -Fxq 'UNKNOWN_ANATOMY=0' "$receipt_dir/summary.txt"
grep -Fxq 'NEXT_EXPERIMENT_CLASS=c1_c2_orientation_and_section_crossing_reparameterization' "$receipt_dir/summary.txt"
grep -Fxq 'PROMOTION_ELIGIBLE=false' "$receipt_dir/summary.txt"
grep -Fxq 'OPEN_PROBLEM_SOLVED=false' "$receipt_dir/summary.txt"

pycache_dir=$(mktemp -d)
trap 'rm -rf "$pycache_dir"' EXIT
PYTHONPYCACHEPREFIX=$pycache_dir python3 -m py_compile scripts/research/cs6_v7b_full_hpg_bridge_analyze.py
git diff --check

echo "V7B_BRIDGE_ANATOMY_GATE_PASS=true"
