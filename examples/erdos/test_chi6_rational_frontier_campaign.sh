#!/usr/bin/env bash
# Gate the deterministic rational-frontier campaign layer.
#
# The campaign runs multiple exact-rational scouts, validates their source
# packages through integrated preflight, and emits a ranked manifest. It remains
# non-promotable: no SAT/LRAT no-5 certificate and no chi(R^2)>=6 claim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

CAMPAIGN="$ROOT/examples/erdos/chi6_rational_frontier_campaign.py"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_solver_candidate_package.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$CAMPAIGN"
mkdir -p "$WORK"

echo "chi6_rational_frontier_campaign_gate: workdir=$WORK"
python3 "$CAMPAIGN" "$WORK/campaign" \
  --candidate-prefix camp \
  --max-den-list 1,5 \
  --layers-list 1 \
  --max-points-list 16 \
  --split-depth 1 \
  > "$WORK/campaign.out"

rg -q '^chi6_rational_frontier_campaign v1$' "$WORK/campaign.out"
rg -q '^campaign_count=2$' "$WORK/campaign.out"
rg -q '^preflight_enabled=1$' "$WORK/campaign.out"
rg -q '^claim_scope=solver_candidate_campaign_only$' "$WORK/campaign.out"
rg -q '^priority_claim_scope=noncertifying_solver_priority_only_not_chromatic_evidence$' "$WORK/campaign.out"
rg -q '^priority_policy=status_bonus_plus_edge_degree_pressure_plus_capped_dsatur_nodes$' "$WORK/campaign.out"
rg -q '^dsatur_node_score_cap=10000$' "$WORK/campaign.out"
rg -q '^sat_claim=none$' "$WORK/campaign.out"
rg -q '^chromatic_claim=none$' "$WORK/campaign.out"
rg -q '^promotable=0$' "$WORK/campaign.out"
rg -q '^promotable_candidate_count=0$' "$WORK/campaign.out"
rg -q '^candidate rank=0 id=camp_d[0-9]+_l1_p16_[0-9]{3} n=[0-9]+ m=[0-9]+ priority=[0-9]+ dsatur_status=K_COLORING_FOUND source_status=PASS geometry_status=PASS sat_status=FAIL integrated_status=INCOMPLETE first_blocker=sat_no5_cube_cover_refutation_absent source=.* source_sha256=[0-9a-f]{64}$' "$WORK/campaign.out"
rg -q '^candidate rank=1 id=camp_d[0-9]+_l1_p16_[0-9]{3} n=[0-9]+ m=[0-9]+ priority=[0-9]+ dsatur_status=K_COLORING_FOUND source_status=PASS geometry_status=PASS sat_status=FAIL integrated_status=INCOMPLETE first_blocker=sat_no5_cube_cover_refutation_absent source=.* source_sha256=[0-9a-f]{64}$' "$WORK/campaign.out"
rg -q '^status=RATIONAL_FRONTIER_CAMPAIGN_READY$' "$WORK/campaign.out"

MANIFEST="$WORK/campaign/campaign.json"
[[ -s "$MANIFEST" ]]
python3 - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
meta = json.loads(manifest.read_text(encoding="ascii"))
assert meta["schema"] == "chi6_rational_frontier_campaign.v1"
assert meta["candidate_prefix"] == "camp"
assert meta["claim_scope"] == "solver_candidate_campaign_only"
assert meta["priority_claim_scope"] == "noncertifying_solver_priority_only_not_chromatic_evidence"
assert meta["priority_policy"] == "status_bonus_plus_edge_degree_pressure_plus_capped_dsatur_nodes"
assert meta["dsatur_node_score_cap"] == 10000
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["promotable"] == 0
assert meta["promotable_candidate_count"] == 0
assert meta["preflight_skipped"] == 0
assert meta["preflight_enabled"] == 1
assert meta["campaign_count"] == 2
ranking = meta["ranking"]
assert len(ranking) == 2
scores = [row["solver_heuristic_priority"] for row in ranking]
assert scores == sorted(scores, reverse=True)
ids = {row["candidate_id"] for row in ranking}
assert ids == {"camp_d1_l1_p16_000", "camp_d5_l1_p16_001"}
for row in ranking:
    assert row["source_status"] == "PASS"
    assert row["geometry_status"] == "PASS"
    assert row["sat_status"] == "FAIL"
    assert row["integrated_status"] == "INCOMPLETE"
    assert row["first_blocker"] == "sat_no5_cube_cover_refutation_absent"
    assert row["promotable"] == 0
    assert row["preflight_skipped"] == 0
    assert row["dsatur_status"] == "K_COLORING_FOUND"
    assert Path(row["candidate_source"]).is_file()
    assert Path(row["frontier_scout"]).is_file()
    assert Path(row["scout_out"]).is_file()
    assert Path(row["scout_stderr"]).is_file()
    assert Path(row["preflight_out"]).is_file()
    assert Path(row["preflight_stderr"]).is_file()
PY

while IFS= read -r source; do
  "$VALIDATOR" "$source" > "$WORK/$(basename "$source").validator.out"
  rg -q '^status=VALID_SOLVER_CANDIDATE_PACKAGE$' "$WORK/$(basename "$source").validator.out"
done < <(python3 - "$MANIFEST" <<'PY'
import json
import sys
for row in json.load(open(sys.argv[1], encoding="ascii"))["ranking"]:
    print(row["candidate_source"])
PY
)

python3 "$CAMPAIGN" "$WORK/skip" \
  --candidate-prefix skipcamp \
  --max-den-list 1 \
  --layers-list 1 \
  --max-points-list 8 \
  --skip-preflight \
  > "$WORK/skip.out"
rg -q '^campaign_count=1$' "$WORK/skip.out"
rg -q '^preflight_enabled=0$' "$WORK/skip.out"
rg -q 'source_status=SKIPPED geometry_status=SKIPPED sat_status=SKIPPED integrated_status=SKIPPED first_blocker=preflight_skipped' \
  "$WORK/skip.out"
python3 - "$WORK/skip/campaign.json" <<'PY'
import json
import sys
meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["preflight_skipped"] == 1
assert meta["ranking"][0]["preflight_skipped"] == 1
PY

python3 "$CAMPAIGN" "$WORK/mixed-infeasible" \
  --candidate-prefix mixedcamp \
  --max-den-list 1,5 \
  --layers-list 1,2 \
  --max-points-list 16 \
  --split-depth 2 \
  --skip-preflight \
  > "$WORK/mixed-infeasible.out"
rg -q '^attempted_scout_count=4$' "$WORK/mixed-infeasible.out"
rg -q '^failed_scout_count=3$' "$WORK/mixed-infeasible.out"
rg -q '^campaign_count=1$' "$WORK/mixed-infeasible.out"
rg -q '^candidate rank=0 id=mixedcamp_d1_l2_p16_001 .* source_status=SKIPPED geometry_status=SKIPPED sat_status=SKIPPED integrated_status=SKIPPED first_blocker=preflight_skipped ' \
  "$WORK/mixed-infeasible.out"
python3 - "$WORK/mixed-infeasible/campaign.json" <<'PY'
import json
import sys
from pathlib import Path

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["attempted_scout_count"] == 4
assert meta["failed_scout_count"] == 3
assert meta["campaign_count"] == 1
assert meta["promotable"] == 0
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert [row["candidate_id"] for row in meta["ranking"]] == ["mixedcamp_d1_l2_p16_001"]
failed_ids = [row["candidate_id"] for row in meta["failed_scouts"]]
assert failed_ids == [
    "mixedcamp_d1_l1_p16_000",
    "mixedcamp_d5_l1_p16_002",
    "mixedcamp_d5_l2_p16_003",
]
for row in meta["failed_scouts"]:
    assert row["status"] == "SCOUT_INFEASIBLE_SKIPPED"
    assert row["claim_scope"] == "scout_infeasible_parameter_row_only"
    assert row["sat_claim"] == "none"
    assert row["chromatic_claim"] == "none"
    assert row["promotable"] == 0
    assert Path(row["scout_out"]).is_file()
    assert Path(row["scout_stderr"]).is_file()
PY

if python3 "$CAMPAIGN" "$WORK/bad" \
    --candidate-prefix bad \
    --max-den-list 1,x \
    > "$WORK/bad.out" 2>&1; then
  echo "error: rational frontier campaign accepted a bad max-den list" >&2
  exit 1
fi
rg -q "bad --max-den-list token: 'x'" "$WORK/bad.out"

echo "chi6_rational_frontier_campaign_gate: PASS"
