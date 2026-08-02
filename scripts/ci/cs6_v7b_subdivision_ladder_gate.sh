#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

receipt_dir=scripts/research/receipts/cs6_v7b_subdivision_ladder_v1
mkdir -p "$receipt_dir"
find "$receipt_dir" -mindepth 1 -maxdepth 1 \
  ! -name docs_registry_blocker.txt \
  ! -name llm-offload \
  -exec rm -rf {} +

if [[ ! -x ${CS6_CAPD_CONFIG:-/tmp/capd-build/bin/capd-config} ]]; then
  echo "V7B_SUBDIVISION_LADDER_GATE_SKIPPED=no_capd_config"
  exit 0
fi

PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/research/cs6_v7b_subdivision_ladder_run.py \
  --out-dir "$receipt_dir" \
  --capd-config "${CS6_CAPD_CONFIG:-/tmp/capd-build/bin/capd-config}" \
  --timeout "${CS6_V7B_LADDER_TIMEOUT:-120}" \
  --jobs "${CS6_V7B_LADDER_JOBS:-4}" | tee "$receipt_dir/run.stdout.txt"
rm -f "$receipt_dir/worker-binary"

PYTHONDONTWRITEBYTECODE=1 python3 -B scripts/research/cs6_v7b_subdivision_ladder_verify.py \
  "$receipt_dir" | tee "$receipt_dir/verification.txt"

grep -Fxq 'RUN_VALID=true' "$receipt_dir/summary.txt"
grep -Fxq 'ATTEMPTS_COMPLETED=24' "$receipt_dir/summary.txt"
grep -Fxq 'UNKNOWN_FAILURE=0' "$receipt_dir/summary.txt"
grep -Fxq 'DESCENDANT_CANDIDATE_DISCOVERED=true' "$receipt_dir/summary.txt"
grep -Fxq 'PARENT_COVER_EVALUATED=false' "$receipt_dir/summary.txt"
grep -Fxq 'V7_B_ELIGIBILITY=false' "$receipt_dir/summary.txt"
grep -Fxq 'PROMOTION_ELIGIBLE=false' "$receipt_dir/summary.txt"
grep -Fxq 'OPEN_PROBLEM_SOLVED=false' "$receipt_dir/summary.txt"

pycache_dir=$(mktemp -d)
trap 'rm -rf "$pycache_dir"' EXIT
PYTHONPYCACHEPREFIX=$pycache_dir python3 -m py_compile \
  scripts/research/cs6_v7b_subdivision_ladder_run.py \
  scripts/research/cs6_v7b_subdivision_ladder_verify.py
git diff --check

echo "V7B_SUBDIVISION_LADDER_GATE_PASS=true"
