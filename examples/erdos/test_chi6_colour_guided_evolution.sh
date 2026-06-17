#!/usr/bin/env bash
# Gate for iterative colour-guided frontier evolution.
#
# This is search orchestration only. It checks that SAT/DSATUR colourings can be
# fed into exact-rational mutation, then back through scout/preflight, without
# emitting any SAT, chromatic, global UNSAT, or verified proof claim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

EVOLVE="$ROOT/examples/erdos/chi6_colour_guided_evolution.py"
MUTATOR="$ROOT/examples/erdos/chi6_colour_guided_mutation.py"
SCOUT="$ROOT/examples/erdos/chi6_rational_frontier_scout.py"
PREFLIGHT="$ROOT/examples/erdos/chi6_frontier_campaign_preflight.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$EVOLVE" "$MUTATOR" "$SCOUT" "$PREFLIGHT"
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

echo "chi6_colour_guided_evolution_gate: workdir=$WORK"
python3 "$EVOLVE" "$WORK/evolution" \
  --coords-csv "$WORK/coords.csv" \
  --colourings-file "$WORK/colourings.txt" \
  --candidate-prefix evotest \
  --generations 2 \
  --mutation-max-den 5 \
  --mutation-top-points 8 \
  --mutation-emit-mutations 2 \
  --mutation-add-points 1 \
  --mutation-min-neighbor-count 1 \
  --split-depth 1 \
  --min-split-degree 1 \
  --max-cubes 100 \
  --sample-hard-cubes 3 \
  > "$WORK/evolution.out"

rg -q '^chi6_colour_guided_evolution v1$' "$WORK/evolution.out"
rg -q '^requested_generations=2$' "$WORK/evolution.out"
rg -q '^completed_generation_count=2$' "$WORK/evolution.out"
rg -q '^stopped_reason=generation_budget_exhausted$' "$WORK/evolution.out"
rg -q '^last_dsatur_status=K_COLORING_FOUND$' "$WORK/evolution.out"
rg -q '^last_recommended_next_action=reject_or_mutate_frontier$' "$WORK/evolution.out"
rg -q '^colouring_feedback_count=2$' "$WORK/evolution.out"
rg -q '^total_selected_killed_colouring_count_by_unit_neighborhood=1$' "$WORK/evolution.out"
rg -q '^claim_scope=colour_guided_frontier_evolution_only$' "$WORK/evolution.out"
rg -q '^sat_claim=none$' "$WORK/evolution.out"
rg -q '^chromatic_claim=none$' "$WORK/evolution.out"
rg -q '^global_unsat_claim=none$' "$WORK/evolution.out"
rg -q '^verified_claim=none$' "$WORK/evolution.out"
rg -q '^promotable=0$' "$WORK/evolution.out"
rg -q '^status=COLOUR_GUIDED_EVOLUTION_RECORDED$' "$WORK/evolution.out"

EVOLUTION_JSON="$(rg '^evolution_json=' "$WORK/evolution.out" | cut -d= -f2-)"
[[ -s "$EVOLUTION_JSON" ]]
python3 - "$EVOLUTION_JSON" <<'PY'
import json
import sys
from pathlib import Path

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["schema"] == "chi6_colour_guided_evolution.v1"
assert meta["requested_generations"] == 2
assert meta["completed_generation_count"] == 2
assert meta["stopped_reason"] == "generation_budget_exhausted"
assert meta["colouring_feedback_count"] == 2
assert meta["total_single_point_full_blocker_count"] >= 1
assert meta["total_selected_killed_colouring_count_by_unit_neighborhood"] == 1
assert meta["claim_scope"] == "colour_guided_frontier_evolution_only"
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["global_unsat_claim"] == "none"
assert meta["verified_claim"] == "none"
assert meta["promotable"] == 0
assert len(meta["generations"]) == 2
g0, g1 = meta["generations"]
assert g0["generation"] == 0
assert g0["selected_old_n"] == 5
assert g0["selected_new_n"] == 6
assert g0["selected_new_m"] == 5
assert g0["selected_edge_gain_after_mutation"] == 5
assert g0["selected_existing_neighbor_count_total"] == 5
assert g0["selected_existing_neighbor_count_min"] == 5
assert g0["selected_existing_neighbor_count_max"] == 5
assert g0["mutation_min_neighbor_count"] == 1
assert g0["mutation_edge_gain_pool_points"] == 0
assert g0["mutation_edge_gain_emit_mutations"] == 1
assert g0["mutation_edge_gain_combination_offset"] == 0
assert g0["mutation_edge_gain_combination_stride"] == 1
assert g0["mutation_edge_gain_considered_combination_count"] == 0
assert g0["mutation_edge_gain_combination_truncated"] is False
assert g0["selected_mutation_strategy"] == "colour_greedy"
assert g0["selected_added_point_count"] == 1
assert g0["selected_killed_colouring_count_by_unit_neighborhood"] == 1
assert g0["dsatur_status"] == "K_COLORING_FOUND"
assert g0["recommended_next_action"] == "reject_or_mutate_frontier"
assert g0["next_colouring_vertex_count"] == 6
assert Path(g0["next_colourings_file"]).is_file()
assert Path(g0["frontier_scout"]).is_file()
assert Path(g0["campaign_preflight_json"]).is_file()
assert g1["generation"] == 1
assert g1["selected_old_n"] == 6
assert g1["selected_new_n"] == meta["last_selected_new_n"]
assert g1["selected_new_m"] == meta["last_selected_new_m"]
assert g1["dsatur_status"] == "K_COLORING_FOUND"
assert g1["recommended_next_action"] == "reject_or_mutate_frontier"
assert g1["next_colouring_vertex_count"] == g1["selected_new_n"]
assert Path(g1["next_colourings_file"]).is_file()
for row in (g0, g1):
    assert row["claim_scope"] == "colour_guided_evolution_generation_only"
    assert row["sat_claim"] == "none"
    assert row["chromatic_claim"] == "none"
    assert row["global_unsat_claim"] == "none"
    assert row["verified_claim"] == "none"
    assert row["promotable"] == 0
PY

if python3 "$EVOLVE" "$WORK/bad" --coords-csv "$WORK/coords.csv" \
    > "$WORK/bad.out" 2>&1; then
  echo "error: evolution accepted missing colourings" >&2
  exit 1
fi
rg -q 'pass --satfanout-json or both --coords-csv and --colourings-file' "$WORK/bad.out"

if python3 "$EVOLVE" "$WORK/bad-neighbors" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --mutation-min-neighbor-count 0 \
    > "$WORK/bad-neighbors.out" 2>&1; then
  echo "error: evolution accepted non-positive mutation-min-neighbor-count" >&2
  exit 1
fi
rg -q -- '--mutation-min-neighbor-count must be positive' "$WORK/bad-neighbors.out"

if python3 "$EVOLVE" "$WORK/bad-offset" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --mutation-edge-gain-combination-offset -1 \
    > "$WORK/bad-offset.out" 2>&1; then
  echo "error: evolution accepted negative mutation-edge-gain-combination-offset" >&2
  exit 1
fi
rg -q -- '--mutation-edge-gain-combination-offset must be non-negative' \
  "$WORK/bad-offset.out"

if python3 "$EVOLVE" "$WORK/bad-stride" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --mutation-edge-gain-combination-stride 0 \
    > "$WORK/bad-stride.out" 2>&1; then
  echo "error: evolution accepted zero mutation-edge-gain-combination-stride" >&2
  exit 1
fi
rg -q -- '--mutation-edge-gain-combination-stride must be positive' \
  "$WORK/bad-stride.out"

echo "chi6_colour_guided_evolution_gate: PASS"
