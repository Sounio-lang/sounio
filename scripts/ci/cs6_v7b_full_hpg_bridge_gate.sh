#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

receipt_dir=scripts/research/receipts/cs6_v7b_full_hpg_bridge_freeze_v1
mkdir -p "$receipt_dir"
rm -f \
  "$receipt_dir/blocker.txt" \
  "$receipt_dir/contract.sha256" \
  "$receipt_dir/files.sha256" \
  "$receipt_dir/gate.stdout.txt" \
  "$receipt_dir/ledger.sha256" \
  "$receipt_dir/summary.txt"

PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/research/cs6_v7b_full_hpg_bridge_gate.py \
  --receipt-dir "$receipt_dir" | tee "$receipt_dir/gate.stdout.txt"

grep -Fxq 'BRIDGE_LEDGER_VALID=true' "$receipt_dir/summary.txt"
grep -Fxq 'PARENT_V7A1_EVIDENCE_BOUND=true' "$receipt_dir/summary.txt"
grep -Fxq 'SATISFIED_BY_V7A1=6' "$receipt_dir/summary.txt"
grep -Fxq 'REQUIRED_UNRUN=18' "$receipt_dir/summary.txt"
grep -Fxq 'V7_B_ELIGIBILITY=false' "$receipt_dir/summary.txt"
grep -Fxq 'V7_B_WINNER=NONE' "$receipt_dir/summary.txt"
grep -Fxq 'PROMOTION_ELIGIBLE=false' "$receipt_dir/summary.txt"
grep -Fxq 'OPEN_PROBLEM_SOLVED=false' "$receipt_dir/summary.txt"
grep -Fxq 'FPGA_EXECUTION=false' "$receipt_dir/summary.txt"

pycache_dir=$(mktemp -d)
trap 'rm -rf "$pycache_dir"' EXIT
PYTHONPYCACHEPREFIX=$pycache_dir python3 -m py_compile scripts/research/cs6_v7b_full_hpg_bridge_gate.py
git diff --check

echo "V7B_BRIDGE_FREEZE_GATE_PASS=true"
