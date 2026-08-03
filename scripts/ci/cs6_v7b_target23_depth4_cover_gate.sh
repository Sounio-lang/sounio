#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

contract=scripts/research/cs6_v7b_target23_depth4_cover_contract_v1.txt
runner=scripts/research/cs6_v7b_target23_depth4_cover_run.py
verifier=scripts/research/cs6_v7b_target23_depth4_cover_verify.py
job=scripts/research/cs6_v7b_target23_depth4_cover_slurm_job.sh

grep -Fxq 'CONTRACT_STATE=PRE_EXECUTION_FROZEN' "$contract"
grep -Fxq 'CHILD_COUNT=256' "$contract"
grep -Fxq 'ATTEMPT_COUNT=512' "$contract"
grep -Fxq 'PARENT_COVER_EVALUATED=false' "$contract"
grep -Fxq 'V7_B_ELIGIBILITY=false' "$contract"
grep -Fxq 'OPEN_PROBLEM_SOLVED=false' "$contract"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
PYTHONDONTWRITEBYTECODE=1 python3 -B "$runner" \
  --out-dir "$work/plan" --plan-only | tee "$work/plan.stdout.txt"
grep -Fxq 'PLAN_VALID=true' "$work/plan/plan-summary.txt"
grep -Fxq 'CHILD_CELLS=256' "$work/plan/plan-summary.txt"
grep -Fxq 'ATTEMPTS=512' "$work/plan/plan-summary.txt"

python3 -B - "$work/plan/coordinate-manifest.tsv" <<'PY'
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open(encoding="ascii"), delimiter="\t"))
assert len(rows) == 256
keys = {(int(row["CHILD_U_OFFSET"]), int(row["CHILD_S_OFFSET"])) for row in rows}
assert keys == {(u, s) for u in range(16) for s in range(16)}
assert len({row["NODE_ID"] for row in rows}) == 256
assert len({row["INPUT_SHA256"] for row in rows}) == 256
assert all(row["U_DEPTH"] == "7" and row["S_DEPTH"] == "8" for row in rows)
assert rows[0]["NODE_ID"] == "U07-0000000096_S08-0000000160"
assert rows[-1]["NODE_ID"] == "U07-0000000111_S08-0000000175"
PY

bash -n "$job"
pycache_dir=$(mktemp -d)
PYTHONPYCACHEPREFIX=$pycache_dir python3 -m py_compile "$runner" "$verifier"
rm -rf "$pycache_dir"
git diff --check

echo "V7B_TARGET23_DEPTH4_COVER_GATE_PASS=true"
