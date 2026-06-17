#!/usr/bin/env bash
# Gate the adaptive recursive chi6 frontier sweep ledger.
#
# This is search scheduling only. It proves the recursive driver records a
# bounded campaign/preflight/refute loop, handles invalid child branches as
# ledger rows, and preserves the no-claim boundary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

FRACTAL="$ROOT/examples/erdos/chi6_fractal_frontier_sweep.py"
SWEEP="$ROOT/examples/erdos/chi6_frontier_refute_sweep.py"
ATTEMPT="$ROOT/examples/erdos/chi6_frontier_refute_attempt.py"
CAMPAIGN="$ROOT/examples/erdos/chi6_rational_frontier_campaign.py"
BATCH="$ROOT/examples/erdos/chi6_frontier_campaign_preflight_batch.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$FRACTAL" "$SWEEP" "$ATTEMPT" "$CAMPAIGN" "$BATCH"
mkdir -p "$WORK"

echo "chi6_fractal_frontier_sweep_gate: workdir=$WORK"
python3 "$FRACTAL" "$WORK/fractal" \
  --candidate-prefix fractest \
  --max-den-list 1 \
  --layers-list 1 \
  --max-points-list 16 \
  --min-vertices 2 \
  --min-edges 1 \
  --split-depth 1 \
  --min-split-degree 2 \
  --dsatur-node-limit 1 \
  --preflight-limit 1 \
  --max-cubes 200 \
  --sample-hard-cubes 2 \
  --refute-limit 1 \
  --timeout-seconds 30 \
  --max-generations 2 \
  --beam-width 2 \
  --children-per-cell 2 \
  --max-den-cap 2 \
  --layers-cap 2 \
  --max-points-cap 32 \
  --split-depth-cap 2 \
  --max-cubes-cap 1000 \
  --sample-hard-cubes-cap 4 \
  --point-growth 8 \
  --sample-growth 2 \
  --colour-guided-mutations \
  --colour-max-den 5 \
  --colour-top-points 5 \
  --colour-emit-mutations 2 \
  --colour-add-points 1 \
  > "$WORK/fractal.out"

rg -q '^chi6_fractal_frontier_sweep v1$' "$WORK/fractal.out"
rg -q '^generation_count=2$' "$WORK/fractal.out"
rg -q '^cell_count=3$' "$WORK/fractal.out"
rg -q '^expanded_cell_count=1$' "$WORK/fractal.out"
rg -q '^frontier_leaf_count=2$' "$WORK/fractal.out"
rg -q '^cell_failure_count=1$' "$WORK/fractal.out"
rg -q '^attempt_count=2$' "$WORK/fractal.out"
rg -q '^refute_success_count=0$' "$WORK/fractal.out"
rg -q '^refute_failed_count=2$' "$WORK/fractal.out"
rg -q '^first_success_candidate=NONE$' "$WORK/fractal.out"
rg -q '^attention_policy=recursive_fractal_frontier_locus_coeruleus_gain_heuristic_not_chromatic_evidence$' \
  "$WORK/fractal.out"
rg -q '^colour_guided_mutation_count=2$' "$WORK/fractal.out"
rg -q '^colour_guided_single_point_full_blocker_count=0$' "$WORK/fractal.out"
rg -q '^recommended_next_action=continue_fractal_attention_or_raise_refuter_budget$' "$WORK/fractal.out"
rg -q '^claim_scope=fractal_frontier_sweep_only$' "$WORK/fractal.out"
rg -q '^sat_claim=none$' "$WORK/fractal.out"
rg -q '^chromatic_claim=none$' "$WORK/fractal.out"
rg -q '^global_unsat_claim=none$' "$WORK/fractal.out"
rg -q '^verified_claim=none$' "$WORK/fractal.out"
rg -q '^promotable=0$' "$WORK/fractal.out"
rg -q '^status=FRACTAL_FRONTIER_SWEEP_RECORDED$' "$WORK/fractal.out"

FRACTAL_JSON="$(rg '^fractal_json=' "$WORK/fractal.out" | cut -d= -f2-)"
[[ -s "$FRACTAL_JSON" ]]
CHECKPOINT_JSON="$WORK/fractal/fractal_frontier_sweep.checkpoint.json"
[[ -s "$CHECKPOINT_JSON" ]]
python3 - "$FRACTAL_JSON" <<'PY'
import json
import sys
from pathlib import Path

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["schema"] == "chi6_fractal_frontier_sweep.v1"
assert meta["generation_count"] == 2
assert meta["requested_max_generations"] == 2
assert meta["beam_width"] == 2
assert meta["children_per_cell"] == 2
assert meta["cell_count"] == 3
assert meta["expanded_cell_count"] == 1
assert meta["frontier_leaf_count"] == 2
assert meta["cell_failure_count"] == 1
assert meta["attempt_count"] == 2
assert meta["refute_success_count"] == 0
assert meta["refute_failed_count"] == 2
assert meta["first_success_candidate"] == "NONE"
assert meta["status_counts"] == {
    "FRACTAL_CELL_FAILED_INFRA": 1,
    "REFUTE_SAT_MUTATE_FRONTIER": 2,
}
assert meta["colour_guided_mutations_enabled"] == 1
assert meta["colour_guided_mutation_count"] == 2
assert meta["colour_guided_single_point_full_blocker_count"] == 0
assert meta["attention_policy"] == "recursive_fractal_frontier_locus_coeruleus_gain_heuristic_not_chromatic_evidence"
assert meta["recommended_next_action"] == "continue_fractal_attention_or_raise_refuter_budget"
assert meta["claim_scope"] == "fractal_frontier_sweep_only"
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["global_unsat_claim"] == "none"
assert meta["verified_claim"] == "none"
assert meta["promotable"] == 0
assert len(meta["cells"]) == 3
seed, bad_child, good_child = meta["cells"]
assert seed["generation"] == 0
assert seed["branch"] == "seed"
assert seed["preflight_status"] == "PREFLIGHT_REFUTE_READY"
assert len(seed["children"]) == 2
assert seed["colour_guided_mutation"]["status"] == "COLOUR_GUIDED_MUTATION_RECORDED"
assert seed["colour_guided_mutation"]["mutation_count"] == 2
assert seed["colour_guided_mutation"]["first_mutation_new_n"] == 6
assert Path(seed["colour_guided_mutation"]["mutation_json"]).is_file()
assert bad_child["generation"] == 1
assert bad_child["branch"] == "recursive_split_gain"
assert bad_child["preflight_status"] == "FRACTAL_CELL_FAILED_INFRA"
assert bad_child["recommended_next_action"] == "discard_invalid_branch"
assert Path(bad_child["failure_path"]).is_file()
assert bad_child["colour_guided_mutation"]["status"] == "COLOUR_GUIDED_MUTATION_SKIPPED"
assert good_child["generation"] == 1
assert good_child["branch"] == "fractal_layer_growth"
assert good_child["preflight_status"] == "PREFLIGHT_REFUTE_READY"
assert good_child["recommended_next_action"] == "mutate_or_expand_frontier"
assert good_child["attention_score"] == meta["best_attention_score"]
assert good_child["colour_guided_mutation"]["status"] == "COLOUR_GUIDED_MUTATION_RECORDED"
assert good_child["colour_guided_mutation"]["mutation_count"] == 2
assert Path(good_child["colour_guided_mutation"]["mutation_json"]).is_file()
for cell in (seed, good_child):
    for key in ("campaign_json", "preflight_batch_json", "refute_attempt_json"):
        assert Path(cell[key]).is_file()
PY

python3 - "$CHECKPOINT_JSON" <<'PY'
import json
import sys

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["schema"] == "chi6_fractal_frontier_sweep_checkpoint.v1"
assert meta["completed_cell_count"] == 3
assert meta["cell_failure_count"] == 1
assert meta["attempt_count"] == 2
assert meta["refute_success_count"] == 0
assert meta["refute_failed_count"] == 2
assert meta["colour_guided_mutation_count"] == 2
assert meta["colour_guided_single_point_full_blocker_count"] == 0
assert meta["claim_scope"] == "fractal_frontier_sweep_checkpoint_only"
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["global_unsat_claim"] == "none"
assert meta["verified_claim"] == "none"
assert meta["promotable"] == 0
PY

if python3 "$FRACTAL" "$WORK/bad-gen" --max-generations 0 \
    > "$WORK/bad-gen.out" 2>&1; then
  echo "error: fractal sweep accepted zero generations" >&2
  exit 1
fi
rg -q -- '--max-generations must be positive' "$WORK/bad-gen.out"

mkdir -p "$WORK/nonempty"
touch "$WORK/nonempty/existing"
if python3 "$FRACTAL" "$WORK/nonempty" \
    > "$WORK/nonempty.out" 2>&1; then
  echo "error: fractal sweep accepted a non-empty out-dir without --resume" >&2
  exit 1
fi
rg -q 'out_dir already exists and is non-empty' "$WORK/nonempty.out"

echo "chi6_fractal_frontier_sweep_gate: PASS"
