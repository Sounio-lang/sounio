#!/usr/bin/env bash
# Gate for beam-search colour-guided frontier evolution.
#
# This evaluates multiple exact-rational mutation children per generation and
# ranks them as search signals only. It emits no SAT, chromatic, global UNSAT,
# or verified proof claim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

BEAM="$ROOT/examples/erdos/chi6_colour_guided_beam.py"
EVOLVE="$ROOT/examples/erdos/chi6_colour_guided_evolution.py"
MUTATOR="$ROOT/examples/erdos/chi6_colour_guided_mutation.py"
SCOUT="$ROOT/examples/erdos/chi6_rational_frontier_scout.py"
PREFLIGHT="$ROOT/examples/erdos/chi6_frontier_campaign_preflight.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$BEAM" "$EVOLVE" "$MUTATOR" "$SCOUT" "$PREFLIGHT"
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

echo "chi6_colour_guided_beam_gate: workdir=$WORK"
python3 "$BEAM" "$WORK/beam" \
  --coords-csv "$WORK/coords.csv" \
  --colourings-file "$WORK/colourings.txt" \
  --candidate-prefix beamtest \
  --generations 2 \
  --beam-width 2 \
  --branch-width 2 \
  --mutation-max-den 5 \
  --mutation-top-points 8 \
  --mutation-emit-mutations 2 \
  --mutation-add-points 1 \
  --mutation-min-neighbor-count 1 \
  --split-depth 1 \
  --min-split-degree 1 \
  --max-cubes 100 \
  --sample-hard-cubes 3 \
  > "$WORK/beam.out"

rg -q '^chi6_colour_guided_beam v1$' "$WORK/beam.out"
rg -q '^requested_generations=2$' "$WORK/beam.out"
rg -q '^completed_generation_count=2$' "$WORK/beam.out"
rg -q '^beam_width=2$' "$WORK/beam.out"
rg -q '^branch_width=2$' "$WORK/beam.out"
rg -q '^mutation_min_neighbor_count=1$' "$WORK/beam.out"
rg -q '^mutation_edge_gain_pool_points=0$' "$WORK/beam.out"
rg -q '^mutation_edge_gain_max_combinations=250000$' "$WORK/beam.out"
rg -q '^mutation_edge_gain_combination_offset=0$' "$WORK/beam.out"
rg -q '^mutation_edge_gain_combination_stride=1$' "$WORK/beam.out"
rg -q '^mutation_edge_gain_emit_mutations=1$' "$WORK/beam.out"
rg -q '^stopped_reason=generation_budget_exhausted$' "$WORK/beam.out"
rg -q '^evaluated_child_count=6$' "$WORK/beam.out"
rg -q '^coloured_child_count=6$' "$WORK/beam.out"
rg -q '^refute_attempt_child_count=0$' "$WORK/beam.out"
rg -q '^refute_success_count=0$' "$WORK/beam.out"
rg -q '^refute_failed_count=0$' "$WORK/beam.out"
rg -q '^best_child_dsatur_status=K_COLORING_FOUND$' "$WORK/beam.out"
rg -q '^best_child_recommended_next_action=reject_or_mutate_frontier$' "$WORK/beam.out"
rg -q '^total_selected_killed_colouring_count_by_unit_neighborhood=2$' "$WORK/beam.out"
rg -q '^claim_scope=colour_guided_frontier_beam_search_only$' "$WORK/beam.out"
rg -q '^sat_claim=none$' "$WORK/beam.out"
rg -q '^chromatic_claim=none$' "$WORK/beam.out"
rg -q '^global_unsat_claim=none$' "$WORK/beam.out"
rg -q '^verified_claim=none$' "$WORK/beam.out"
rg -q '^promotable=0$' "$WORK/beam.out"
rg -q '^status=COLOUR_GUIDED_BEAM_RECORDED$' "$WORK/beam.out"

BEAM_JSON="$(rg '^beam_json=' "$WORK/beam.out" | cut -d= -f2-)"
[[ -s "$BEAM_JSON" ]]
python3 - "$BEAM_JSON" <<'PY'
import json
import sys
from pathlib import Path

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["schema"] == "chi6_colour_guided_beam.v1"
assert meta["requested_generations"] == 2
assert meta["completed_generation_count"] == 2
assert meta["beam_width"] == 2
assert meta["branch_width"] == 2
assert meta["mutation_min_neighbor_count"] == 1
assert meta["mutation_edge_gain_pool_points"] == 0
assert meta["mutation_edge_gain_max_combinations"] == 250000
assert meta["mutation_edge_gain_combination_offset"] == 0
assert meta["mutation_edge_gain_combination_stride"] == 1
assert meta["mutation_edge_gain_emit_mutations"] == 1
assert meta["stopped_reason"] == "generation_budget_exhausted"
assert meta["evaluated_child_count"] == 6
assert meta["coloured_child_count"] == 6
assert meta["refute_attempt_child_count"] == 0
assert meta["refute_success_count"] == 0
assert meta["refute_failed_count"] == 0
assert meta["best_child_dsatur_status"] == "K_COLORING_FOUND"
assert meta["best_child_recommended_next_action"] == "reject_or_mutate_frontier"
assert meta["total_selected_killed_colouring_count_by_unit_neighborhood"] == 2
assert meta["claim_scope"] == "colour_guided_frontier_beam_search_only"
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["global_unsat_claim"] == "none"
assert meta["verified_claim"] == "none"
assert meta["promotable"] == 0
assert len(meta["generations"]) == 2
g0, g1 = meta["generations"]
assert g0["generation"] == 0
assert g0["input_parent_count"] == 1
assert g0["evaluated_child_count"] == 2
assert g0["next_beam_count"] == 2
assert g1["generation"] == 1
assert g1["input_parent_count"] == 2
assert g1["evaluated_child_count"] == 4
assert g1["next_beam_count"] == 2
first = g0["children"][0]
assert first["selected_old_n"] == 5
assert first["selected_new_n"] == 6
assert first["selected_edge_gain_after_mutation"] == 5
assert first["selected_existing_neighbor_count_total"] == 5
assert first["selected_existing_neighbor_count_min"] == 5
assert first["selected_existing_neighbor_count_max"] == 5
assert first["mutation_min_neighbor_count"] == 1
assert first["mutation_edge_gain_pool_points"] == 0
assert first["mutation_edge_gain_emit_mutations"] == 1
assert first["mutation_edge_gain_combination_offset"] == 0
assert first["mutation_edge_gain_combination_stride"] == 1
assert first["mutation_edge_gain_considered_combination_count"] == 0
assert first["mutation_edge_gain_combination_truncated"] is False
assert first["selected_mutation_strategy"] == "colour_greedy"
assert first["selected_killed_colouring_count_by_unit_neighborhood"] == 1
assert first["next_colouring_history_count"] == 1
assert first["next_extended_parent_colouring_count"] == 0
assert first["dsatur_status"] == "K_COLORING_FOUND"
assert first["recommended_next_action"] == "reject_or_mutate_frontier"
assert (
    first["beam_score"] == meta["best_child_score"]
    or meta["best_child_score"] >= first["beam_score"]
)
for gen in meta["generations"]:
    scores = [row["beam_score"] for row in gen["children"]]
    assert scores == sorted(scores, reverse=True)
    for row in gen["children"]:
        assert Path(row["selected_coords_csv"]).is_file()
        assert Path(row["frontier_scout"]).is_file()
        assert Path(row["campaign_preflight_json"]).is_file()
        assert Path(row["next_colourings_file"]).is_file()
        assert row["next_colouring_history_count"] >= 1
        assert row["next_extended_parent_colouring_count"] >= 0
        if gen["generation"] == 1:
            assert row["next_extended_parent_colouring_count"] >= 1
        assert row["refute_attempt"]["status"] == "REFUTE_ATTEMPT_SKIPPED"
        assert row["refute_attempt"]["reason"] == "disabled"
        assert row["claim_scope"] == "colour_guided_beam_child_only"
        assert row["sat_claim"] == "none"
        assert row["chromatic_claim"] == "none"
        assert row["global_unsat_claim"] == "none"
        assert row["verified_claim"] == "none"
        assert row["promotable"] == 0
PY

python3 "$BEAM" "$WORK/refute-ready-beam" \
  --coords-csv "$WORK/coords.csv" \
  --colourings-file "$WORK/colourings.txt" \
  --candidate-prefix beamrefute \
  --generations 1 \
  --beam-width 1 \
  --branch-width 1 \
  --mutation-max-den 5 \
  --mutation-top-points 4 \
  --mutation-emit-mutations 1 \
  --mutation-add-points 1 \
  --mutation-min-neighbor-count 1 \
  --split-depth 1 \
  --min-split-degree 1 \
  --dsatur-node-limit 1 \
  --max-cubes 100 \
  --sample-hard-cubes 2 \
  --run-refute-ready \
  --refute-limit 1 \
  > "$WORK/refute-ready-beam.out"

rg -q '^completed_generation_count=1$' "$WORK/refute-ready-beam.out"
rg -q '^evaluated_child_count=1$' "$WORK/refute-ready-beam.out"
rg -q '^refute_attempt_child_count=1$' "$WORK/refute-ready-beam.out"
rg -q '^refute_success_count=0$' "$WORK/refute-ready-beam.out"
rg -q '^refute_failed_count=1$' "$WORK/refute-ready-beam.out"
rg -q '^best_child_dsatur_status=UNKNOWN_NODE_LIMIT$' "$WORK/refute-ready-beam.out"
rg -q '^best_child_recommended_next_action=prepare_cube_refute_batch$' \
  "$WORK/refute-ready-beam.out"
rg -q '^promotable=0$' "$WORK/refute-ready-beam.out"

REFUTE_BEAM_JSON="$(rg '^beam_json=' "$WORK/refute-ready-beam.out" | cut -d= -f2-)"
[[ -s "$REFUTE_BEAM_JSON" ]]
python3 - "$REFUTE_BEAM_JSON" <<'PY'
import json
import sys
from pathlib import Path

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["schema"] == "chi6_colour_guided_beam.v1"
assert meta["completed_generation_count"] == 1
assert meta["evaluated_child_count"] == 1
assert meta["refute_attempt_child_count"] == 1
assert meta["refute_success_count"] == 0
assert meta["refute_failed_count"] == 1
child = meta["generations"][0]["children"][0]
assert child["dsatur_status"] == "UNKNOWN_NODE_LIMIT"
assert child["recommended_next_action"] == "prepare_cube_refute_batch"
attempt = child["refute_attempt"]
assert attempt["status"] == "REFUTE_ATTEMPT_RECORDED"
assert attempt["reason"] == "refute_ready_child"
assert attempt["attempt_count"] == 1
assert attempt["refute_success_count"] == 0
assert attempt["refute_failed_count"] == 1
assert attempt["status_counts"] == {"REFUTE_SAT_MUTATE_FRONTIER": 1}
assert Path(attempt["preflight_batch_json"]).is_file()
assert Path(attempt["refute_attempt_json"]).is_file()
assert child["sat_claim"] == "none"
assert child["chromatic_claim"] == "none"
assert child["global_unsat_claim"] == "none"
assert child["verified_claim"] == "none"
assert child["promotable"] == 0
PY

if python3 "$BEAM" "$WORK/bad" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --branch-width 3 \
    --mutation-emit-mutations 2 \
    > "$WORK/bad.out" 2>&1; then
  echo "error: beam accepted branch-width above emitted mutations" >&2
  exit 1
fi
rg -q -- '--branch-width cannot exceed --mutation-emit-mutations' "$WORK/bad.out"

if python3 "$BEAM" "$WORK/bad-neighbors" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --branch-width 1 \
    --mutation-emit-mutations 1 \
    --mutation-min-neighbor-count 0 \
    > "$WORK/bad-neighbors.out" 2>&1; then
  echo "error: beam accepted non-positive mutation-min-neighbor-count" >&2
  exit 1
fi
rg -q -- '--mutation-min-neighbor-count must be positive' "$WORK/bad-neighbors.out"

if python3 "$BEAM" "$WORK/bad-edge-gain" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --branch-width 1 \
    --mutation-emit-mutations 1 \
    --mutation-edge-gain-pool-points -1 \
    > "$WORK/bad-edge-gain.out" 2>&1; then
  echo "error: beam accepted negative mutation-edge-gain-pool-points" >&2
  exit 1
fi
rg -q -- '--mutation-edge-gain-pool-points must be non-negative' \
  "$WORK/bad-edge-gain.out"

if python3 "$BEAM" "$WORK/bad-edge-gain-count" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --branch-width 1 \
    --mutation-emit-mutations 1 \
    --mutation-edge-gain-emit-mutations -1 \
    > "$WORK/bad-edge-gain-count.out" 2>&1; then
  echo "error: beam accepted negative mutation-edge-gain-emit-mutations" >&2
  exit 1
fi
rg -q -- '--mutation-edge-gain-emit-mutations must be non-negative' \
  "$WORK/bad-edge-gain-count.out"

if python3 "$BEAM" "$WORK/bad-edge-gain-offset" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --branch-width 1 \
    --mutation-emit-mutations 1 \
    --mutation-edge-gain-combination-offset -1 \
    > "$WORK/bad-edge-gain-offset.out" 2>&1; then
  echo "error: beam accepted negative mutation-edge-gain-combination-offset" >&2
  exit 1
fi
rg -q -- '--mutation-edge-gain-combination-offset must be non-negative' \
  "$WORK/bad-edge-gain-offset.out"

if python3 "$BEAM" "$WORK/bad-edge-gain-stride" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --branch-width 1 \
    --mutation-emit-mutations 1 \
    --mutation-edge-gain-combination-stride 0 \
    > "$WORK/bad-edge-gain-stride.out" 2>&1; then
  echo "error: beam accepted zero mutation-edge-gain-combination-stride" >&2
  exit 1
fi
rg -q -- '--mutation-edge-gain-combination-stride must be positive' \
  "$WORK/bad-edge-gain-stride.out"

echo "chi6_colour_guided_beam_gate: PASS"
