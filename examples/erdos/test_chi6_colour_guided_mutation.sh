#!/usr/bin/env bash
# Gate for colour-guided frontier mutation.
#
# This is search steering only: observed SAT colourings suggest rational points
# to add to the exact coordinate frontier.  It emits no SAT/LRAT/Lean claim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

MUTATOR="$ROOT/examples/erdos/chi6_colour_guided_mutation.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$MUTATOR"
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

echo "chi6_colour_guided_mutation_gate: workdir=$WORK"
python3 "$MUTATOR" "$WORK/mutations" \
  --coords-csv "$WORK/coords.csv" \
  --colourings-file "$WORK/colourings.txt" \
  --candidate-prefix unit_star \
  --max-den 5 \
  --top-points 5 \
  --emit-mutations 3 \
  --add-points 1 \
  --edge-gain-pool-points 5 \
  --edge-gain-combination-offset 1 \
  --edge-gain-combination-stride 2 \
  --edge-gain-emit-mutations 2 \
  > "$WORK/mutation.out"

rg -q '^chi6_colour_guided_mutation v1$' "$WORK/mutation.out"
rg -q '^source_candidate_id=unit_star$' "$WORK/mutation.out"
rg -q '^n=5$' "$WORK/mutation.out"
rg -q '^observed_colouring_count=1$' "$WORK/mutation.out"
rg -q '^single_point_full_blocker_count=[1-9][0-9]*$' "$WORK/mutation.out"
rg -q '^first_mutation_new_n=6$' "$WORK/mutation.out"
rg -q '^first_mutation_new_m=5$' "$WORK/mutation.out"
rg -q '^first_mutation_killed_colouring_count_by_unit_neighborhood=1$' "$WORK/mutation.out"
rg -q '^sat_claim=none$' "$WORK/mutation.out"
rg -q '^chromatic_claim=none$' "$WORK/mutation.out"
rg -q '^global_unsat_claim=none$' "$WORK/mutation.out"
rg -q '^verified_claim=none$' "$WORK/mutation.out"
rg -q '^promotable=0$' "$WORK/mutation.out"
rg -q '^status=COLOUR_GUIDED_MUTATION_RECORDED$' "$WORK/mutation.out"

MUTATION_JSON="$(rg '^mutation_json=' "$WORK/mutation.out" | cut -d= -f2-)"
[[ -s "$MUTATION_JSON" ]]
python3 - "$MUTATION_JSON" <<'PY'
import json
import sys
from pathlib import Path

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["schema"] == "chi6_colour_guided_mutation.v1"
assert meta["input_mode"] == "explicit_colourings"
assert meta["n"] == 5
assert meta["m"] == 0
assert meta["k"] == 5
assert meta["observed_colouring_count"] == 1
assert meta["min_neighbor_count"] == 1
assert meta["edge_gain_pool_points"] == 5
assert meta["edge_gain_max_combinations"] == 250000
assert meta["edge_gain_combination_offset"] == 1
assert meta["edge_gain_combination_stride"] == 2
assert meta["edge_gain_emit_mutations"] == 2
assert meta["edge_gain_considered_combination_count"] == 2
assert meta["edge_gain_combination_truncated"] is False
assert meta["single_point_full_blocker_count"] >= 1
assert meta["claim_scope"] == "colour_guided_frontier_mutation_only"
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["global_unsat_claim"] == "none"
assert meta["verified_claim"] == "none"
assert meta["promotable"] == 0
first = meta["mutations"][0]
assert first["selection_strategy"] == "colour_greedy"
assert first["old_n"] == 5
assert first["old_m"] == 0
assert first["new_n"] == 6
assert first["new_m"] == 5
assert first["edge_gain_after_mutation"] == 5
assert first["selected_existing_neighbor_count_total"] == 5
assert first["selected_existing_neighbor_count_min"] == 5
assert first["selected_existing_neighbor_count_max"] == 5
assert first["killed_colouring_count_by_unit_neighborhood"] == 1
point = first["points"][0]
assert point["x"] == "0"
assert point["y"] == "0"
assert point["neighbor_count"] == 5
assert point["killed_colouring_count"] == 1
assert Path(first["coords_csv"]).is_file()
second = meta["mutations"][1]
assert second["selection_strategy"] == "edge_gain_batch"
third = meta["mutations"][2]
assert third["selection_strategy"] == "edge_gain_batch"
PY

if python3 "$MUTATOR" "$WORK/bad" --coords-csv "$WORK/coords.csv" \
    > "$WORK/bad.out" 2>&1; then
  echo "error: mutator accepted missing colourings" >&2
  exit 1
fi
rg -q 'pass --satfanout-json or both --coords-csv and --colourings-file' "$WORK/bad.out"

if python3 "$MUTATOR" "$WORK/bad-neighbors" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --candidate-prefix unit_star \
    --max-den 5 \
    --min-neighbor-count 6 \
    > "$WORK/bad-neighbors.out" 2>&1; then
  echo "error: mutator accepted impossible neighbor-count filter" >&2
  exit 1
fi
rg -q 'no adjacent rational unit-step mutation candidates found with min_neighbor_count=6' \
  "$WORK/bad-neighbors.out"

if python3 "$MUTATOR" "$WORK/bad-offset" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --edge-gain-combination-offset -1 \
    > "$WORK/bad-offset.out" 2>&1; then
  echo "error: mutator accepted negative edge-gain-combination-offset" >&2
  exit 1
fi
rg -q -- '--edge-gain-combination-offset must be non-negative' "$WORK/bad-offset.out"

if python3 "$MUTATOR" "$WORK/bad-stride" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --edge-gain-combination-stride 0 \
    > "$WORK/bad-stride.out" 2>&1; then
  echo "error: mutator accepted zero edge-gain-combination-stride" >&2
  exit 1
fi
rg -q -- '--edge-gain-combination-stride must be positive' "$WORK/bad-stride.out"

echo "chi6_colour_guided_mutation_gate: PASS"
