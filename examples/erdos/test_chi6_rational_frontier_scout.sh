#!/usr/bin/env bash
# Gate the CPU exact-rational frontier scout for chi>=6 source candidates.
#
# This is a search/plumbing gate only: it proves the scout can derive an exact
# rational unit graph, run a bounded local hardness probe, and hand a source
# package to the existing integrated preflight without claiming chi(R^2) >= 6.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

SCOUT="$ROOT/examples/erdos/chi6_rational_frontier_scout.py"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_solver_candidate_package.py"
PREFLIGHT="$ROOT/examples/erdos/make_chi6_integrated_candidate_preflight.sh"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$SCOUT"
mkdir -p "$WORK"

echo "chi6_rational_frontier_scout_gate: workdir=$WORK"
GEN_WORK="$WORK/generated"
python3 "$SCOUT" \
  --candidate-id generated_frontier_smoke \
  --out-dir "$GEN_WORK" \
  --max-den 5 \
  --layers 1 \
  --max-points 32 \
  --split-depth 1 \
  > "$WORK/generated.out"

rg -q '^chi6_rational_frontier_scout v1$' "$WORK/generated.out"
rg -q '^candidate_id=generated_frontier_smoke$' "$WORK/generated.out"
rg -q '^mode=generated_unit_step_cloud$' "$WORK/generated.out"
rg -q '^k=5$' "$WORK/generated.out"
rg -q '^split_vertices=0$' "$WORK/generated.out"
rg -q '^dsatur_status=K_COLORING_FOUND$' "$WORK/generated.out"
rg -q '^claim_scope=solver_candidate_frontier_only$' "$WORK/generated.out"
rg -q '^sat_claim=none$' "$WORK/generated.out"
rg -q '^chromatic_claim=none$' "$WORK/generated.out"
rg -q '^promotable=0$' "$WORK/generated.out"
rg -q '^status=SCOUT_SOURCE_PACKAGE_READY$' "$WORK/generated.out"

GEN_SOURCE="$GEN_WORK/source/generated_frontier_smoke.candidate-source.json"
GEN_SIDECAR="$GEN_WORK/generated_frontier_smoke.frontier-scout.json"
[[ -s "$GEN_SOURCE" ]]
[[ -s "$GEN_SIDECAR" ]]
"$VALIDATOR" "$GEN_SOURCE" > "$WORK/generated-validator.out"
rg -q '^status=VALID_SOLVER_CANDIDATE_PACKAGE$' "$WORK/generated-validator.out"

python3 - "$GEN_SIDECAR" "$GEN_SOURCE" <<'PY'
import json
import sys
from pathlib import Path

sidecar, source = map(Path, sys.argv[1:])
meta = json.loads(sidecar.read_text(encoding="ascii"))
assert meta["schema"] == "chi6_rational_frontier_scout.v1"
assert meta["candidate_id"] == "generated_frontier_smoke"
assert meta["candidate_source"] == str(source)
assert meta["mode"] == "generated_unit_step_cloud"
assert meta["k"] == 5
assert meta["m"] >= 1
assert meta["split_vertices"] == [0]
assert meta["split_vertex_degrees"]["0"] >= 2
assert meta["split_vertices_adjacent_pairs"] == []
assert meta["split_vertices_induced_edge_count"] == 0
assert meta["split_vertices_induced_is_clique"] is True
assert meta["split_vertices_incident_edge_count"] >= meta["split_vertex_degrees"]["0"]
assert meta["split_cover_note"] == "split product enumerates assignments; SAT handles adjacent split conflicts"
assert meta["dsatur_status"] == "K_COLORING_FOUND"
assert meta["dsatur_claim_scope"] == "bounded_cpu_search_probe_only"
assert meta["dsatur_warning"] == "negative DSATUR probe statuses are not SAT/LRAT/Lean certificates"
assert meta["claim_scope"] == "solver_candidate_frontier_only"
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["promotable"] == 0
PY

GEN_PREFLIGHT="$WORK/generated-preflight"
WORK="$GEN_PREFLIGHT" "$PREFLIGHT" "$GEN_SOURCE" > "$WORK/generated-preflight.out"
rg -q '^source_status=PASS$' "$WORK/generated-preflight.out"
rg -q '^geometry_status=PASS$' "$WORK/generated-preflight.out"
rg -q '^sat_status=FAIL$' "$WORK/generated-preflight.out"
rg -q '^promotable=0$' "$WORK/generated-preflight.out"

SQUARE="$WORK/square.coords.csv"
cat > "$SQUARE" <<'EOF'
id,x,y
0,0,0
1,1,0
2,1,1
3,0,1
EOF
INGEST_WORK="$WORK/ingest"
python3 "$SCOUT" \
  --coords-csv "$SQUARE" \
  --candidate-id square_frontier_smoke \
  --out-dir "$INGEST_WORK" \
  --split-depth 2 \
  > "$WORK/ingest.out"

rg -q '^mode=ingest_csv$' "$WORK/ingest.out"
rg -q '^n=4$' "$WORK/ingest.out"
rg -q '^m=4$' "$WORK/ingest.out"
rg -q '^split_vertices=0,1$' "$WORK/ingest.out"
rg -q '^dsatur_status=K_COLORING_FOUND$' "$WORK/ingest.out"
rg -q '^status=SCOUT_SOURCE_PACKAGE_READY$' "$WORK/ingest.out"

INGEST_SOURCE="$INGEST_WORK/source/square_frontier_smoke.candidate-source.json"
"$VALIDATOR" "$INGEST_SOURCE" > "$WORK/ingest-validator.out"
rg -q '^status=VALID_SOLVER_CANDIDATE_PACKAGE$' "$WORK/ingest-validator.out"

if python3 "$SCOUT" \
    --coords-csv "$SQUARE" \
    --candidate-id bad_split_depth \
    --out-dir "$WORK/bad-depth" \
    --split-depth 5 \
    > "$WORK/bad-depth.out" 2>&1; then
  echo "error: frontier scout accepted excessive split depth" >&2
  exit 1
fi
rg -q 'cannot choose split depth 5' "$WORK/bad-depth.out"

echo "chi6_rational_frontier_scout_gate: PASS"
