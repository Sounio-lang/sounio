#!/usr/bin/env bash
# Gate campaign -> preflight batch -> refute attempt sweep ledger.
#
# This is a bounded local search loop. It records the next action after trying
# selected cube refutations, while preserving the no-claim boundary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

SWEEP="$ROOT/examples/erdos/chi6_frontier_refute_sweep.py"
ATTEMPT="$ROOT/examples/erdos/chi6_frontier_refute_attempt.py"
CAMPAIGN="$ROOT/examples/erdos/chi6_rational_frontier_campaign.py"
BATCH="$ROOT/examples/erdos/chi6_frontier_campaign_preflight_batch.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$SWEEP" "$ATTEMPT" "$CAMPAIGN" "$BATCH"
mkdir -p "$WORK"

echo "chi6_frontier_refute_sweep_gate: workdir=$WORK"
python3 "$SWEEP" "$WORK/sweep" \
  --candidate-prefix sweeptest \
  --max-den-list 1,5 \
  --layers-list 1 \
  --max-points-list 16 \
  --split-depth-list 1 \
  --min-split-degree-list 2 \
  --dsatur-node-limit-list 1 \
  --preflight-limit 1 \
  --refute-limit 1 \
  > "$WORK/sweep.out"

rg -q '^chi6_frontier_refute_sweep v1$' "$WORK/sweep.out"
rg -q '^cell_count=1$' "$WORK/sweep.out"
rg -q '^campaign_manifest_count=1$' "$WORK/sweep.out"
rg -q '^preflight_batch_count=1$' "$WORK/sweep.out"
rg -q '^attempt_manifest_count=1$' "$WORK/sweep.out"
rg -q '^campaign_total_count=2$' "$WORK/sweep.out"
rg -q '^preflight_selected_count=1$' "$WORK/sweep.out"
rg -q '^preflight_refute_ready_count=1$' "$WORK/sweep.out"
rg -q '^attempt_count=1$' "$WORK/sweep.out"
rg -q '^refute_success_count=0$' "$WORK/sweep.out"
rg -q '^refute_failed_count=1$' "$WORK/sweep.out"
rg -q '^first_success_candidate=NONE$' "$WORK/sweep.out"
rg -q '^first_failed_candidate=sweeptest_c000_d5_l1_p16_001$' "$WORK/sweep.out"
rg -q '^recommended_next_action=mutate_or_expand_frontier$' "$WORK/sweep.out"
rg -q '^claim_scope=frontier_refute_sweep_only$' "$WORK/sweep.out"
rg -q '^sat_claim=none$' "$WORK/sweep.out"
rg -q '^chromatic_claim=none$' "$WORK/sweep.out"
rg -q '^global_unsat_claim=none$' "$WORK/sweep.out"
rg -q '^verified_claim=none$' "$WORK/sweep.out"
rg -q '^promotable=0$' "$WORK/sweep.out"
rg -q '^status=FRONTIER_REFUTE_SWEEP_RECORDED$' "$WORK/sweep.out"

SWEEP_JSON="$(rg '^sweep_json=' "$WORK/sweep.out" | cut -d= -f2-)"
[[ -s "$SWEEP_JSON" ]]
python3 - "$SWEEP_JSON" <<'PY'
import json
import sys
from pathlib import Path

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["schema"] == "chi6_frontier_refute_sweep.v1"
assert meta["cell_count"] == 1
assert meta["campaign_manifest_count"] == 1
assert meta["preflight_batch_count"] == 1
assert meta["attempt_manifest_count"] == 1
assert meta["campaign_total_count"] == 2
assert meta["preflight_selected_count"] == 1
assert meta["preflight_refute_ready_count"] == 1
assert meta["attempt_count"] == 1
assert meta["refute_success_count"] == 0
assert meta["refute_failed_count"] == 1
assert meta["first_success_candidate"] == "NONE"
assert meta["first_failed_candidate"] == "sweeptest_c000_d5_l1_p16_001"
assert meta["status_counts"] == {"REFUTE_SAT_MUTATE_FRONTIER": 1}
assert meta["recommended_next_action"] == "mutate_or_expand_frontier"
assert meta["claim_scope"] == "frontier_refute_sweep_only"
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["global_unsat_claim"] == "none"
assert meta["verified_claim"] == "none"
assert meta["promotable"] == 0
assert len(meta["cells"]) == 1
cell = meta["cells"][0]
assert cell["cell_index"] == 0
assert cell["candidate_prefix"] == "sweeptest_c000"
assert cell["status_counts"] == {"REFUTE_SAT_MUTATE_FRONTIER": 1}
for key in ("campaign_json", "preflight_batch_json", "refute_attempt_json"):
    assert Path(cell[key]).is_file()
PY

python3 "$SWEEP" "$WORK/skip-coloring" \
  --candidate-prefix skipsweep \
  --max-den-list 1 \
  --layers-list 1 \
  --max-points-list 16 \
  --split-depth 1 \
  --dsatur-node-limit 100000 \
  --skip-coloring-found \
  > "$WORK/skip-coloring.out" 2>&1 && {
  echo "error: sweep accepted a batch with no selected candidates" >&2
  exit 1
}
rg -q 'no campaign candidates selected for preflight' "$WORK/skip-coloring.out"

python3 "$SWEEP" "$WORK/no-viable-cell" \
  --candidate-prefix noviable \
  --max-den-list 1 \
  --layers-list 1 \
  --max-points-list 16 \
  --split-depth-list 2 \
  --min-split-degree-list 2 \
  --dsatur-node-limit-list 1 \
  --preflight-limit 1 \
  --refute-limit 1 \
  > "$WORK/no-viable-cell.out"
rg -q '^cell_count=1$' "$WORK/no-viable-cell.out"
rg -q '^campaign_manifest_count=0$' "$WORK/no-viable-cell.out"
rg -q '^preflight_batch_count=0$' "$WORK/no-viable-cell.out"
rg -q '^attempt_manifest_count=0$' "$WORK/no-viable-cell.out"
rg -q '^campaign_total_count=0$' "$WORK/no-viable-cell.out"
rg -q '^attempt_count=0$' "$WORK/no-viable-cell.out"
rg -q '^recommended_next_action=adjust_split_parameters_or_expand_frontier$' "$WORK/no-viable-cell.out"
NO_VIABLE_SWEEP_JSON="$(rg '^sweep_json=' "$WORK/no-viable-cell.out" | cut -d= -f2-)"
[[ -s "$NO_VIABLE_SWEEP_JSON" ]]
python3 - "$NO_VIABLE_SWEEP_JSON" <<'PY'
import json
import sys
from pathlib import Path

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["schema"] == "chi6_frontier_refute_sweep.v1"
assert meta["cell_count"] == 1
assert meta["campaign_manifest_count"] == 0
assert meta["preflight_batch_count"] == 0
assert meta["attempt_manifest_count"] == 0
assert meta["campaign_total_count"] == 0
assert meta["preflight_selected_count"] == 0
assert meta["preflight_refute_ready_count"] == 0
assert meta["attempt_count"] == 0
assert meta["refute_success_count"] == 0
assert meta["refute_failed_count"] == 0
assert meta["status_counts"] == {"CAMPAIGN_NO_VIABLE_SCOUTS": 1}
assert meta["recommended_next_action"] == "adjust_split_parameters_or_expand_frontier"
cell = meta["cells"][0]
assert cell["preflight_status"] == "CAMPAIGN_NO_VIABLE_SCOUTS"
assert cell["campaign_json"] == "NONE"
assert cell["preflight_batch_json"] == "NONE"
assert cell["refute_attempt_json"] == "NONE"
assert Path(cell["campaign_stdout"]).is_file()
assert Path(cell["campaign_stderr"]).is_file()
PY

if python3 "$SWEEP" "$WORK/bad-limit" --preflight-limit 0 \
    > "$WORK/bad-limit.out" 2>&1; then
  echo "error: sweep accepted zero preflight limit" >&2
  exit 1
fi
rg -q -- '--preflight-limit must be positive' "$WORK/bad-limit.out"

if python3 "$SWEEP" "$WORK/bad-split" --split-depth-list 0 \
    > "$WORK/bad-split.out" 2>&1; then
  echo "error: sweep accepted zero split-depth list" >&2
  exit 1
fi
rg -q "bad --split-depth-list token: '0'" "$WORK/bad-split.out"

if python3 "$SWEEP" "$WORK/bad-sample" --max-cubes 4 --sample-hard-cubes 5 \
    > "$WORK/bad-sample.out" 2>&1; then
  echo "error: sweep accepted sample-hard-cubes > max-cubes" >&2
  exit 1
fi
rg -q -- '--sample-hard-cubes cannot exceed --max-cubes' "$WORK/bad-sample.out"

mkdir -p "$WORK/nonempty"
touch "$WORK/nonempty/existing"
if python3 "$SWEEP" "$WORK/nonempty" \
    > "$WORK/nonempty.out" 2>&1; then
  echo "error: sweep accepted a non-empty out-dir without --resume" >&2
  exit 1
fi
rg -q 'out_dir already exists and is non-empty' "$WORK/nonempty.out"

echo "chi6_frontier_refute_sweep_gate: PASS"
