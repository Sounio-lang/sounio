#!/usr/bin/env bash
# Gate the scout -> cube-campaign preflight planner.
#
# This classifies the deterministic cube workload after an exact-rational scout.
# It is still non-promotable and emits no SAT/LRAT/Lean no-5 certificate.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

SCOUT="$ROOT/examples/erdos/chi6_rational_frontier_scout.py"
PLANNER="$ROOT/examples/erdos/chi6_frontier_campaign_preflight.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$SCOUT" "$PLANNER"
mkdir -p "$WORK"

COORDS="$WORK/square.coords.csv"
cat > "$COORDS" <<'EOF'
id,x,y
0,0,0
1,1,0
2,1,1
3,0,1
EOF

echo "chi6_frontier_campaign_preflight_gate: workdir=$WORK"
python3 "$SCOUT" \
  --coords-csv "$COORDS" \
  --candidate-id square_campaign_frontier \
  --out-dir "$WORK/scout" \
  --split-depth 1 \
  > "$WORK/scout.out"
rg -q '^status=SCOUT_SOURCE_PACKAGE_READY$' "$WORK/scout.out"

SCOUT_JSON="$WORK/scout/square_campaign_frontier.frontier-scout.json"
python3 "$PLANNER" "$SCOUT_JSON" "$WORK/campaign" > "$WORK/campaign.out"

rg -q '^chi6_frontier_campaign_preflight v1$' "$WORK/campaign.out"
rg -q '^candidate_id=square_campaign_frontier$' "$WORK/campaign.out"
rg -q '^campaign_id=chi6camp_[0-9a-f]{32}$' "$WORK/campaign.out"
rg -q '^campaign_mode=split_product_frontier_preflight$' "$WORK/campaign.out"
rg -q '^source_status=PASS$' "$WORK/campaign.out"
rg -q '^n=4$' "$WORK/campaign.out"
rg -q '^m=4$' "$WORK/campaign.out"
rg -q '^k=5$' "$WORK/campaign.out"
rg -q '^split_vertices=0$' "$WORK/campaign.out"
rg -q '^cube_count=5$' "$WORK/campaign.out"
rg -q '^propagation_conflict_count=0$' "$WORK/campaign.out"
rg -q '^propagation_hard_count=5$' "$WORK/campaign.out"
rg -q '^recommended_next_action=reject_or_mutate_frontier$' "$WORK/campaign.out"
rg -q '^recommended_next_gate=reject_or_mutate_frontier$' "$WORK/campaign.out"
rg -q '^next_action_execution_status=not_run_by_preflight$' "$WORK/campaign.out"
rg -q '^claim_scope=deterministic_campaign_preflight_only$' "$WORK/campaign.out"
rg -q '^sat_claim=none$' "$WORK/campaign.out"
rg -q '^chromatic_claim=none$' "$WORK/campaign.out"
rg -q '^global_unsat_claim=none$' "$WORK/campaign.out"
rg -q '^verified_claim=none$' "$WORK/campaign.out"
rg -q '^promotable=0$' "$WORK/campaign.out"
rg -q '^status=FRONTIER_CAMPAIGN_PREFLIGHT_READY$' "$WORK/campaign.out"

MANIFEST="$(rg '^campaign_preflight_json=' "$WORK/campaign.out" | cut -d= -f2-)"
[[ -s "$MANIFEST" ]]
python3 - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
meta = json.loads(manifest.read_text(encoding="ascii"))
assert meta["schema"] == "chi6_frontier_campaign_preflight.v1"
assert meta["candidate_id"] == "square_campaign_frontier"
assert meta["campaign_id"].startswith("chi6camp_")
assert meta["campaign_mode"] == "split_product_frontier_preflight"
assert meta["source_status"] == "PASS"
assert meta["n"] == 4
assert meta["m"] == 4
assert meta["k"] == 5
assert meta["split_vertices"] == [0]
assert meta["split_depth"] == 1
assert meta["cube_count"] == 5
assert meta["propagation_conflict_count"] == 0
assert meta["propagation_hard_count"] == 5
assert meta["hard_cube_prefix_sample_requested"] == 5
assert meta["hard_cube_prefix_sample_count"] == 5
assert meta["hard_cube_prefix_sample_truncated"] == 0
assert meta["hard_cube_prefix_sample_exhaustive"] == 1
assert len(meta["hard_cube_prefix_sample"]) == 5
assert len(meta["hard_cube_prefix_assignment_sample"]) == 5
assert meta["hard_cube_prefix_assignment_sample"][0]["id"] == meta["hard_cube_prefix_sample"][0]
assert meta["hard_cube_prefix_assignment_sample"][0]["assignments"].startswith("0:")
assert meta["estimated_vars"] == 20
assert meta["estimated_repo_colourCNF_base_clause_count"] == 24
assert meta["estimated_repo_colourCNF_base_clause_count_scope"] == (
    "repo_colourCNF_base_only_atleast_one_plus_edge_clauses_only"
    "_no_cube_units_lrat_or_cover_clauses"
)
assert meta["standard_at_most_one_colour_clause_count_included"] == 0
assert meta["estimated_leaf_lrat_required_if_all_hard_cubes_unsat"] == 5
assert meta["recommended_next_action"] == "reject_or_mutate_frontier"
assert meta["recommended_next_gate"] == "reject_or_mutate_frontier"
assert meta["next_action_execution_status"] == "not_run_by_preflight"
assert meta["foundry_handoff_recommended"] == 1
assert "cube_sieve_refute_batch.py" in meta["refute_command"]
assert "cube_cover_complement_cnf.py" in meta["complement_cnf_command"]
assert "make_chi6_integrated_candidate_preflight.sh" in meta["integrated_preflight_command"]
assert meta["claim_scope"] == "deterministic_campaign_preflight_only"
assert meta["geometry_claim"] == "exact_rational_squared_distance_edges_only_from_source_validator"
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["global_unsat_claim"] == "none"
assert meta["verified_claim"] == "none"
assert meta["promotable"] == 0
assert Path(meta["cube_batch_path"]).is_file()
assert Path(meta["propagation_summary_path"]).is_file()
assert Path(meta["hard_cube_list_path"]).is_file()
PY

HARD_LIST="$(python3 - "$MANIFEST" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="ascii"))["hard_cube_list_path"])
PY
)"
hard_count="$(wc -l < "$HARD_LIST" | tr -d ' ')"
if [[ "$hard_count" != "5" ]]; then
  echo "error: expected 5 hard cubes, got $hard_count" >&2
  exit 1
fi

if python3 "$PLANNER" "$SCOUT_JSON" "$WORK/bad-cap" --max-cubes 4 \
    > "$WORK/bad-cap.out" 2>&1; then
  echo "error: campaign preflight ignored max-cubes cap" >&2
  exit 1
fi
rg -q 'split product would emit 5 cubes' "$WORK/bad-cap.out"

if python3 "$PLANNER" "$SCOUT_JSON" "$WORK/bad-zero-cap" --max-cubes 0 \
    > "$WORK/bad-zero-cap.out" 2>&1; then
  echo "error: campaign preflight accepted zero max-cubes" >&2
  exit 1
fi
rg -q -- '--max-cubes must be positive' "$WORK/bad-zero-cap.out"

if python3 "$PLANNER" "$SCOUT_JSON" "$WORK/bad-sample" --sample-hard-cubes 1001 \
    > "$WORK/bad-sample.out" 2>&1; then
  echo "error: campaign preflight accepted an excessive hard-cube sample" >&2
  exit 1
fi
rg -q -- '--sample-hard-cubes must be <= 1000' "$WORK/bad-sample.out"

PYTHONPATH="$ROOT/examples/erdos" python3 - <<'PY'
from chi6_frontier_campaign_preflight import choose_next_gate

assert choose_next_gate("K_COLORING_FOUND", 5, 0, 5) == "reject_or_mutate_frontier"
assert choose_next_gate("UNKNOWN_NODE_LIMIT", 1, 4, 5) == "prepare_cube_refute_batch"
assert (
    choose_next_gate("UNKNOWN_NODE_LIMIT", 0, 5, 5)
    == "propagation_conflicts_require_lrat_lean_upgrade"
)
try:
    choose_next_gate("UNKNOWN_NODE_LIMIT", 0, 0, 0)
except RuntimeError as exc:
    assert "zero cubes" in str(exc)
else:
    raise AssertionError("zero-cube campaign was not fatal")
try:
    choose_next_gate("UNKNOWN_NODE_LIMIT", 0, 0, 5)
except RuntimeError as exc:
    assert "do not cover cube count" in str(exc)
else:
    raise AssertionError("uncovered cube counts were not fatal")
PY

python3 - "$SCOUT_JSON" "$WORK/bad-scout.json" <<'PY'
import json
import sys
src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["promotable"] = 1
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
if python3 "$PLANNER" "$WORK/bad-scout.json" "$WORK/bad-scout" \
    > "$WORK/bad-scout.out" 2>&1; then
  echo "error: campaign preflight accepted a promotable scout sidecar" >&2
  exit 1
fi
rg -q 'frontier scout must be non-promotable' "$WORK/bad-scout.out"

echo "chi6_frontier_campaign_preflight_gate: PASS"
