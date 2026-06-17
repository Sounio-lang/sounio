#!/usr/bin/env bash
# Gate for colour-guided mutation density-envelope probing.
#
# This is search instrumentation only: it summarizes exact-rational candidate
# neighbour-count availability across denominator budgets without emitting any
# SAT, chromatic, global UNSAT, or verified proof claim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

PROBE="$ROOT/examples/erdos/chi6_colour_guided_density_probe.py"
MUTATOR="$ROOT/examples/erdos/chi6_colour_guided_mutation.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$PROBE" "$MUTATOR"
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

echo "chi6_colour_guided_density_probe_gate: workdir=$WORK"
python3 "$PROBE" "$WORK/probe" \
  --coords-csv "$WORK/coords.csv" \
  --colourings-file "$WORK/colourings.txt" \
  --candidate-prefix densitytest \
  --max-den-list 1,5 \
  --min-neighbor-count-list 1,5,6 \
  --max-candidates 200 \
  --top-points 5 \
  > "$WORK/probe.out"

rg -q '^chi6_colour_guided_density_probe v1$' "$WORK/probe.out"
rg -q '^source_candidate_id=densitytest$' "$WORK/probe.out"
rg -q '^n=5$' "$WORK/probe.out"
rg -q '^m=0$' "$WORK/probe.out"
rg -q '^observed_colouring_count=1$' "$WORK/probe.out"
rg -q '^max_observed_neighbor_count=5$' "$WORK/probe.out"
rg -q '^max_observed_neighbor_count_den=5$' "$WORK/probe.out"
rg -q '^first_den_with_min_neighbor_count_1=1$' "$WORK/probe.out"
rg -q '^first_den_with_min_neighbor_count_5=1$' "$WORK/probe.out"
rg -q '^first_den_with_min_neighbor_count_6=NONE$' "$WORK/probe.out"
rg -q '^claim_scope=colour_guided_density_envelope_only$' "$WORK/probe.out"
rg -q '^sat_claim=none$' "$WORK/probe.out"
rg -q '^chromatic_claim=none$' "$WORK/probe.out"
rg -q '^global_unsat_claim=none$' "$WORK/probe.out"
rg -q '^verified_claim=none$' "$WORK/probe.out"
rg -q '^promotable=0$' "$WORK/probe.out"
rg -q '^status=COLOUR_GUIDED_DENSITY_PROBE_RECORDED$' "$WORK/probe.out"

PROBE_JSON="$(rg '^density_probe_json=' "$WORK/probe.out" | cut -d= -f2-)"
[[ -s "$PROBE_JSON" ]]
python3 - "$PROBE_JSON" <<'PY'
import json
import sys

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["schema"] == "chi6_colour_guided_density_probe.v1"
assert meta["input_mode"] == "explicit_colourings"
assert meta["n"] == 5
assert meta["m"] == 0
assert meta["k"] == 5
assert meta["observed_colouring_count"] == 1
assert meta["max_den_list"] == [1, 5]
assert meta["min_neighbor_count_list"] == [1, 5, 6]
assert meta["max_observed_neighbor_count"] == 5
assert meta["max_observed_neighbor_count_den"] == 5
assert meta["first_den_by_min_neighbor_count"] == {"1": 1, "5": 1, "6": "NONE"}
assert meta["claim_scope"] == "colour_guided_density_envelope_only"
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["global_unsat_claim"] == "none"
assert meta["verified_claim"] == "none"
assert meta["promotable"] == 0
assert len(meta["probes"]) == 2
first = meta["probes"][0]
assert first["max_den"] == 1
assert first["max_neighbor_count"] == 5
assert first["candidate_point_count_scored"] > 0
assert first["single_point_full_blocker_count"] >= 1
assert first["neighbor_count_histogram"]
rows = {row["min_neighbor_count"]: row for row in first["thresholds"]}
assert rows[1]["available_count"] > 0
assert rows[5]["available_count"] >= 1
assert rows[6]["available_count"] == 0
top = first["top_points"][0]
assert top["neighbor_count"] == 5
assert top["killed_colouring_count"] == 1
PY

if python3 "$PROBE" "$WORK/bad" \
    --coords-csv "$WORK/coords.csv" \
    --colourings-file "$WORK/colourings.txt" \
    --max-den-list 1,0 \
    > "$WORK/bad.out" 2>&1; then
  echo "error: density probe accepted non-positive max-den-list token" >&2
  exit 1
fi
rg -q "bad --max-den-list token: '0'" "$WORK/bad.out"

if python3 "$PROBE" "$WORK/bad-missing" \
    --coords-csv "$WORK/coords.csv" \
    > "$WORK/bad-missing.out" 2>&1; then
  echo "error: density probe accepted missing colourings" >&2
  exit 1
fi
rg -q 'pass --satfanout-json or both --coords-csv and --colourings-file' \
  "$WORK/bad-missing.out"

echo "chi6_colour_guided_density_probe_gate: PASS"
