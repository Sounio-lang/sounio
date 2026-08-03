#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root"

contract=scripts/research/cs6_v7b_target23_depth5_boundary_refine_contract_v1.txt
runner=scripts/research/cs6_v7b_target23_depth5_boundary_refine_run.py
verifier=scripts/research/cs6_v7b_target23_depth5_boundary_refine_verify.py
job=scripts/research/cs6_v7b_target23_depth5_boundary_refine_slurm_job.sh

grep -Fxq 'CONTRACT_STATE=PRE_EXECUTION_FROZEN' "$contract"
grep -Fxq 'SOURCE_REJECTED_PARENT_COUNT=25' "$contract"
grep -Fxq 'REFINED_ORIGINAL_PARENT_DEPTH_DELTA=5' "$contract"
grep -Fxq 'REFINEMENT_STEP_DELTA=1' "$contract"
grep -Fxq 'GRANDCHILD_COUNT=100' "$contract"
grep -Fxq 'ATTEMPT_COUNT=200' "$contract"
grep -Fxq 'ADAPTIVE_PARENT_PROBE_COVER_EVALUATED=false' "$contract"
grep -Fxq 'V7_B_ELIGIBILITY=false' "$contract"
grep -Fxq 'OPEN_PROBLEM_SOLVED=false' "$contract"

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
PYTHONDONTWRITEBYTECODE=1 python3 -B "$runner" --out-dir "$work/plan" --plan-only |
  tee "$work/plan.stdout.txt"
grep -Fxq 'PLAN_VALID=true' "$work/plan/plan-summary.txt"
grep -Fxq 'SOURCE_REJECTED_PARENT_CELLS=25' "$work/plan/plan-summary.txt"
grep -Fxq 'GRANDCHILD_CELLS=100' "$work/plan/plan-summary.txt"
grep -Fxq 'ATTEMPTS=200' "$work/plan/plan-summary.txt"

python3 -B - "$work/plan/coordinate-manifest.tsv" <<'PY'
import csv
import sys
from collections import defaultdict
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open(encoding="ascii"), delimiter="\t"))
assert len(rows) == 100
assert len({row["NODE_ID"] for row in rows}) == 100
assert len({row["INPUT_SHA256"] for row in rows}) == 100
assert all(row["U_DEPTH"] == "8" and row["S_DEPTH"] == "9" for row in rows)
children = defaultdict(set)
for row in rows:
    children[int(row["SOURCE_CELL_INDEX"])].add(
        (int(row["SUB_U_OFFSET"]), int(row["SUB_S_OFFSET"]))
    )
assert len(children) == 25
assert all(value == {(0, 0), (0, 1), (1, 0), (1, 1)} for value in children.values())
assert rows[0]["NODE_ID"] == "U08-0000000198_S09-0000000320"
assert rows[-1]["NODE_ID"] == "U08-0000000223_S09-0000000325"
PY

bash -n "$job"
pycache_dir=$(mktemp -d)
PYTHONPYCACHEPREFIX=$pycache_dir python3 -m py_compile "$runner" "$verifier"
rm -rf "$pycache_dir"
git diff --check

echo "V7B_TARGET23_DEPTH5_BOUNDARY_REFINE_GATE_PASS=true"
