#!/usr/bin/env bash
# Gate for the shardable persistent colour-guided beam campaign runner.
#
# This verifies append-only ledger rows, resume semantics, shard selection, and
# in-campaign refute attempts without making any SAT/chromatic/global claim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

CAMPAIGN="$ROOT/examples/erdos/chi6_colour_guided_beam_campaign.py"
BEAM="$ROOT/examples/erdos/chi6_colour_guided_beam.py"
EVOLVE="$ROOT/examples/erdos/chi6_colour_guided_evolution.py"
MUTATOR="$ROOT/examples/erdos/chi6_colour_guided_mutation.py"
SCOUT="$ROOT/examples/erdos/chi6_rational_frontier_scout.py"
PREFLIGHT="$ROOT/examples/erdos/chi6_frontier_campaign_preflight.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$CAMPAIGN" "$BEAM" "$EVOLVE" "$MUTATOR" "$SCOUT" "$PREFLIGHT"
mkdir -p "$WORK"

cat > "$WORK/coords.csv" <<'EOF'
id,x,y
0,1,0
1,0,1
2,-1,0
3,0,-1
4,3/5,4/5
EOF

cat > "$WORK/colourings.txt" <<'EOF'
0:0,1:1,2:2,3:3,4:4
EOF

COMMON_ARGS=(
  --coords-csv "$WORK/coords.csv"
  --colourings-file "$WORK/colourings.txt"
  --candidate-prefix campaign
  --generations-list 1
  --beam-width-list 1
  --branch-width-list 1
  --mutation-max-den-list 5
  --mutation-top-points-list 4
  --mutation-emit-mutations 1
  --mutation-add-points 1
  --mutation-min-neighbor-count 1
  --split-depth 1
  --min-split-degree 1
  --max-cubes 100
  --sample-hard-cubes 2
)

echo "chi6_colour_guided_beam_campaign_gate: workdir=$WORK"
python3 "$CAMPAIGN" "$WORK/campaign" "${COMMON_ARGS[@]}" \
  --dsatur-node-limit-list 100000 \
  > "$WORK/campaign.out"

rg -q '^chi6_colour_guided_beam_campaign v1$' "$WORK/campaign.out"
rg -q '^shard_index=0$' "$WORK/campaign.out"
rg -q '^shard_count=1$' "$WORK/campaign.out"
rg -q '^selected_cell_count=1$' "$WORK/campaign.out"
rg -q '^executed_cell_count=1$' "$WORK/campaign.out"
rg -q '^skipped_resume_cell_count=0$' "$WORK/campaign.out"
rg -q '^mutation_min_neighbor_count=1$' "$WORK/campaign.out"
rg -q '^mutation_edge_gain_pool_points=0$' "$WORK/campaign.out"
rg -q '^mutation_edge_gain_max_combinations=250000$' "$WORK/campaign.out"
rg -q '^mutation_edge_gain_combination_offset=0$' "$WORK/campaign.out"
rg -q '^mutation_edge_gain_combination_stride=1$' "$WORK/campaign.out"
rg -q '^mutation_edge_gain_emit_mutations=1$' "$WORK/campaign.out"
rg -q '^completed_cell_count=1$' "$WORK/campaign.out"
rg -q '^failed_cell_count=0$' "$WORK/campaign.out"
rg -q '^evaluated_child_count=1$' "$WORK/campaign.out"
rg -q '^coloured_child_count=1$' "$WORK/campaign.out"
rg -q '^refute_attempt_child_count=0$' "$WORK/campaign.out"
rg -q '^refute_success_count=0$' "$WORK/campaign.out"
rg -q '^refute_failed_count=0$' "$WORK/campaign.out"
rg -q '^best_cell_key=i000000_g1_bw1_br1_d5_tp4_ds100000$' "$WORK/campaign.out"
rg -q '^best_child_dsatur_status=K_COLORING_FOUND$' "$WORK/campaign.out"
rg -q '^best_child_recommended_next_action=reject_or_mutate_frontier$' "$WORK/campaign.out"
rg -q '^claim_scope=colour_guided_beam_campaign_only$' "$WORK/campaign.out"
rg -q '^sat_claim=none$' "$WORK/campaign.out"
rg -q '^chromatic_claim=none$' "$WORK/campaign.out"
rg -q '^global_unsat_claim=none$' "$WORK/campaign.out"
rg -q '^verified_claim=none$' "$WORK/campaign.out"
rg -q '^promotable=0$' "$WORK/campaign.out"
rg -q '^status=COLOUR_GUIDED_BEAM_CAMPAIGN_RECORDED$' "$WORK/campaign.out"

CAMPAIGN_JSON="$(rg '^campaign_json=' "$WORK/campaign.out" | cut -d= -f2-)"
LEDGER_JSONL="$(rg '^ledger_jsonl=' "$WORK/campaign.out" | cut -d= -f2-)"
[[ -s "$CAMPAIGN_JSON" ]]
[[ -s "$LEDGER_JSONL" ]]
python3 - "$CAMPAIGN_JSON" "$LEDGER_JSONL" <<'PY'
import json
import sys
from pathlib import Path

campaign_path, ledger_path = map(Path, sys.argv[1:])
meta = json.load(open(campaign_path, encoding="ascii"))
rows = [json.loads(line) for line in open(ledger_path, encoding="ascii") if line.strip()]
assert meta["schema"] == "chi6_colour_guided_beam_campaign.v1"
assert meta["selected_cell_count"] == 1
assert meta["executed_cell_count"] == 1
assert meta["skipped_resume_cell_count"] == 0
assert meta["mutation_min_neighbor_count"] == 1
assert meta["mutation_edge_gain_pool_points"] == 0
assert meta["mutation_edge_gain_max_combinations"] == 250000
assert meta["mutation_edge_gain_combination_offset"] == 0
assert meta["mutation_edge_gain_combination_stride"] == 1
assert meta["mutation_edge_gain_emit_mutations"] == 1
assert meta["completed_cell_count"] == 1
assert meta["failed_cell_count"] == 0
assert meta["evaluated_child_count"] == 1
assert meta["coloured_child_count"] == 1
assert meta["refute_attempt_child_count"] == 0
assert meta["refute_success_count"] == 0
assert meta["refute_failed_count"] == 0
assert meta["best_cell_key"] == "i000000_g1_bw1_br1_d5_tp4_ds100000"
assert meta["best_child_dsatur_status"] == "K_COLORING_FOUND"
assert meta["best_child_recommended_next_action"] == "reject_or_mutate_frontier"
assert meta["claim_scope"] == "colour_guided_beam_campaign_only"
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["global_unsat_claim"] == "none"
assert meta["verified_claim"] == "none"
assert meta["promotable"] == 0
assert len(meta["cells"]) == 1
assert len(rows) == 1
row = rows[0]
assert row["schema"] == "chi6_colour_guided_beam_campaign_cell.v1"
assert row["campaign_cell_key"] == "i000000_g1_bw1_br1_d5_tp4_ds100000"
assert row["mutation_min_neighbor_count"] == 1
assert row["mutation_edge_gain_pool_points"] == 0
assert row["mutation_edge_gain_max_combinations"] == 250000
assert row["mutation_edge_gain_combination_offset"] == 0
assert row["mutation_edge_gain_combination_stride"] == 1
assert row["mutation_edge_gain_emit_mutations"] == 1
assert row["status"] == "BEAM_CAMPAIGN_CELL_RECORDED"
assert row["claim_scope"] == "colour_guided_beam_campaign_cell_only"
assert row["sat_claim"] == "none"
assert row["chromatic_claim"] == "none"
assert row["global_unsat_claim"] == "none"
assert row["verified_claim"] == "none"
assert row["promotable"] == 0
assert Path(row["beam_json"]).is_file()
PY

if python3 "$CAMPAIGN" "$WORK/campaign" "${COMMON_ARGS[@]}" \
    --dsatur-node-limit-list 100000 \
    > "$WORK/no-resume.out" 2>&1; then
  echo "error: campaign accepted a non-empty out-dir without --resume" >&2
  exit 1
fi
rg -q 'out_dir already exists and is non-empty; pass --resume to reuse it' "$WORK/no-resume.out"

python3 "$CAMPAIGN" "$WORK/campaign" "${COMMON_ARGS[@]}" \
  --dsatur-node-limit-list 100000 \
  --resume \
  > "$WORK/resume.out"

rg -q '^executed_cell_count=0$' "$WORK/resume.out"
rg -q '^skipped_resume_cell_count=1$' "$WORK/resume.out"
rg -q '^completed_cell_count=1$' "$WORK/resume.out"
python3 - "$LEDGER_JSONL" <<'PY'
import sys
from pathlib import Path

rows = [line for line in Path(sys.argv[1]).read_text(encoding="ascii").splitlines() if line]
assert len(rows) == 1
PY

python3 "$CAMPAIGN" "$WORK/shard" "${COMMON_ARGS[@]}" \
  --dsatur-node-limit-list 1,100000 \
  --shard-index 1 \
  --shard-count 2 \
  > "$WORK/shard.out"

rg -q '^shard_index=1$' "$WORK/shard.out"
rg -q '^shard_count=2$' "$WORK/shard.out"
rg -q '^selected_cell_count=1$' "$WORK/shard.out"
rg -q '^best_cell_key=i000001_g1_bw1_br1_d5_tp4_ds100000$' "$WORK/shard.out"

python3 "$CAMPAIGN" "$WORK/refute" "${COMMON_ARGS[@]}" \
  --dsatur-node-limit-list 1 \
  --run-refute-ready \
  --refute-limit 1 \
  > "$WORK/refute.out"

rg -q '^completed_cell_count=1$' "$WORK/refute.out"
rg -q '^refute_attempt_child_count=1$' "$WORK/refute.out"
rg -q '^refute_success_count=0$' "$WORK/refute.out"
rg -q '^refute_failed_count=1$' "$WORK/refute.out"
rg -q '^best_cell_key=i000000_g1_bw1_br1_d5_tp4_ds1$' "$WORK/refute.out"
rg -q '^best_child_dsatur_status=UNKNOWN_NODE_LIMIT$' "$WORK/refute.out"
rg -q '^best_child_recommended_next_action=prepare_cube_refute_batch$' "$WORK/refute.out"
rg -q '^promotable=0$' "$WORK/refute.out"

REFUTE_JSON="$(rg '^campaign_json=' "$WORK/refute.out" | cut -d= -f2-)"
python3 - "$REFUTE_JSON" <<'PY'
import json
import sys

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["refute_attempt_child_count"] == 1
assert meta["refute_success_count"] == 0
assert meta["refute_failed_count"] == 1
cell = meta["cells"][0]
assert cell["status"] == "BEAM_CAMPAIGN_CELL_RECORDED"
assert cell["best_child_dsatur_status"] == "UNKNOWN_NODE_LIMIT"
assert cell["best_child_recommended_next_action"] == "prepare_cube_refute_batch"
assert cell["sat_claim"] == "none"
assert cell["chromatic_claim"] == "none"
assert cell["global_unsat_claim"] == "none"
assert cell["verified_claim"] == "none"
assert cell["promotable"] == 0
PY

if python3 "$CAMPAIGN" "$WORK/bad-branch" "${COMMON_ARGS[@]}" \
    --branch-width-list 2 \
    --mutation-emit-mutations 1 \
    > "$WORK/bad-branch.out" 2>&1; then
  echo "error: campaign accepted branch-width above emitted mutations" >&2
  exit 1
fi
rg -q -- '--branch-width-list cannot exceed --mutation-emit-mutations' "$WORK/bad-branch.out"

if python3 "$CAMPAIGN" "$WORK/bad-neighbors" "${COMMON_ARGS[@]}" \
    --mutation-min-neighbor-count 0 \
    > "$WORK/bad-neighbors.out" 2>&1; then
  echo "error: campaign accepted non-positive mutation-min-neighbor-count" >&2
  exit 1
fi
rg -q -- '--mutation-min-neighbor-count must be positive' "$WORK/bad-neighbors.out"

if python3 "$CAMPAIGN" "$WORK/bad-edge-gain" "${COMMON_ARGS[@]}" \
    --mutation-edge-gain-pool-points -1 \
    > "$WORK/bad-edge-gain.out" 2>&1; then
  echo "error: campaign accepted negative mutation-edge-gain-pool-points" >&2
  exit 1
fi
rg -q -- '--mutation-edge-gain-pool-points must be non-negative' \
  "$WORK/bad-edge-gain.out"

if python3 "$CAMPAIGN" "$WORK/bad-edge-gain-count" "${COMMON_ARGS[@]}" \
    --mutation-edge-gain-emit-mutations -1 \
    > "$WORK/bad-edge-gain-count.out" 2>&1; then
  echo "error: campaign accepted negative mutation-edge-gain-emit-mutations" >&2
  exit 1
fi
rg -q -- '--mutation-edge-gain-emit-mutations must be non-negative' \
  "$WORK/bad-edge-gain-count.out"

if python3 "$CAMPAIGN" "$WORK/bad-edge-gain-offset" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --branch-width-list 1 \
    --mutation-emit-mutations 1 \
    --mutation-edge-gain-combination-offset -1 \
    > "$WORK/bad-edge-gain-offset.out" 2>&1; then
  echo "error: campaign accepted negative mutation-edge-gain-combination-offset" >&2
  exit 1
fi
rg -q -- '--mutation-edge-gain-combination-offset must be non-negative' \
  "$WORK/bad-edge-gain-offset.out"

if python3 "$CAMPAIGN" "$WORK/bad-edge-gain-stride" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --branch-width-list 1 \
    --mutation-emit-mutations 1 \
    --mutation-edge-gain-combination-stride 0 \
    > "$WORK/bad-edge-gain-stride.out" 2>&1; then
  echo "error: campaign accepted zero mutation-edge-gain-combination-stride" >&2
  exit 1
fi
rg -q -- '--mutation-edge-gain-combination-stride must be positive' \
  "$WORK/bad-edge-gain-stride.out"

echo "chi6_colour_guided_beam_campaign_gate: PASS"
