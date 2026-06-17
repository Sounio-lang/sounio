#!/usr/bin/env bash
# Gate the rational-coordinate unit-graph source-package front door.
#
# This derives the DIMACS edge list from exact rational coordinates before
# packaging, so downstream SAT/geometry gates consume one consistent source
# bundle. It remains non-promotable: no SAT/LRAT and no chromatic claim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

MAKER="$ROOT/examples/erdos/make_chi6_rational_unit_graph_source_package.py"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_solver_candidate_package.py"
PREFLIGHT="$ROOT/examples/erdos/make_chi6_integrated_candidate_preflight.sh"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }
python3 -m py_compile "$MAKER" "$VALIDATOR"
mkdir -p "$WORK"

COORDS="$WORK/square.coords.csv"
cat > "$COORDS" <<'EOF'
id,x,y
0,0,0
1,1,0
2,1,1
3,0,1
EOF

echo "chi6_rational_unit_graph_source_package_gate: workdir=$WORK"
PKG_WORK="$WORK/pkg"
python3 "$MAKER" "$COORDS" square_unit_graph_source 0,1 "$PKG_WORK" \
  > "$WORK/maker.out"

rg -q '^chi6_rational_unit_graph_source_package v1$' "$WORK/maker.out"
rg -q '^candidate_id=square_unit_graph_source$' "$WORK/maker.out"
rg -q '^n=4$' "$WORK/maker.out"
rg -q '^m=4$' "$WORK/maker.out"
rg -q '^k=5$' "$WORK/maker.out"
rg -q '^split_vertices=0,1$' "$WORK/maker.out"
rg -q '^split_vertex_indexing=zero_based$' "$WORK/maker.out"
rg -q '^edge_vertex_indexing=one_based_dimacs$' "$WORK/maker.out"
rg -q '^split_vertex_min_degree=2$' "$WORK/maker.out"
rg -q '^max_vertices=4096$' "$WORK/maker.out"
rg -q '^edge_derivation=all_pairs_exact_rational_dist2_eq_1$' "$WORK/maker.out"
rg -q '^claim_scope=solver_candidate_source_only$' "$WORK/maker.out"
rg -q '^sat_claim=none$' "$WORK/maker.out"
rg -q '^chromatic_claim=none$' "$WORK/maker.out"
rg -q '^promotable=0$' "$WORK/maker.out"
rg -q '^status=FORMAT_VALID_RATIONAL_UNIT_GRAPH_SOURCE_PACKAGE$' "$WORK/maker.out"

SOURCE="$PKG_WORK/square_unit_graph_source.candidate-source.json"
EDGE="$PKG_WORK/package/square_unit_graph_source.edge"
PKG_COORDS="$PKG_WORK/package/square_unit_graph_source.coords.csv"
[[ -s "$SOURCE" ]]
[[ -s "$EDGE" ]]
[[ -s "$PKG_COORDS" ]]
cmp -s "$COORDS" "$PKG_COORDS"
rg -q '^p edge 4 4$' "$EDGE"
rg -q '^e 1 2$' "$EDGE"
rg -q '^e 1 4$' "$EDGE"
rg -q '^e 2 3$' "$EDGE"
rg -q '^e 3 4$' "$EDGE"
if rg -q '^e 1 3$|^e 2 4$' "$EDGE"; then
  echo "error: derived square unit graph included a diagonal" >&2
  exit 1
fi

"$VALIDATOR" "$SOURCE" > "$WORK/source-validator.out"
rg -q '^status=VALID_SOLVER_CANDIDATE_PACKAGE$' "$WORK/source-validator.out"
rg -q '^validated_edge_count=4$' "$WORK/source-validator.out"
rg -q '^coordinate_row_count=4$' "$WORK/source-validator.out"
rg -q '^geometry_claim=exact_rational_squared_distance_edges_only$' "$WORK/source-validator.out"

python3 - "$SOURCE" "$EDGE" "$PKG_COORDS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source, edge, coords = map(Path, sys.argv[1:])
meta = json.loads(source.read_text(encoding="ascii"))

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert meta["schema"] == "chi6_solver_candidate_package.v1"
assert meta["candidate_id"] == "square_unit_graph_source"
assert meta["edge_path"] == "package/square_unit_graph_source.edge"
assert meta["coords_path"] == "package/square_unit_graph_source.coords.csv"
assert meta["edge_sha256"] == digest(edge)
assert meta["coords_sha256"] == digest(coords)
assert meta["coordinate_domain"] == "rational_xy"
assert meta["n"] == 4
assert meta["m"] == 4
assert meta["k"] == 5
assert meta["split_vertices"] == [0, 1]
assert meta["claim_scope"] == "solver_candidate_source_only"
assert meta["promotion_gate"] == "requires_cube_cover_lrat_lean_exact_geometry_and_real_bridge"
assert "sat_claim" not in meta
assert "chromatic_claim" not in meta
assert "promotable" not in meta
PY

PREFLIGHT_WORK="$WORK/preflight"
WORK="$PREFLIGHT_WORK" "$PREFLIGHT" "$SOURCE" > "$WORK/preflight.out"
rg -q '^source_status=PASS$' "$WORK/preflight.out"
rg -q '^geometry_status=PASS$' "$WORK/preflight.out"
rg -q '^sat_status=FAIL$' "$WORK/preflight.out"
rg -q '^integrated_status=INCOMPLETE$' "$WORK/preflight.out"
rg -q '^first_blocker=sat_no5_cube_cover_refutation_absent$' "$WORK/preflight.out"
rg -q '^promotable=0$' "$WORK/preflight.out"
rg -q '^chromatic_claim=none$' "$WORK/preflight.out"

EQUIV_COORDS="$WORK/equivalent-rationals.coords.csv"
cat > "$EQUIV_COORDS" <<'EOF'
id,x,y
0,0,0
1,2/2,0
2,3/3,5/5
3,0,7/7
EOF
python3 "$MAKER" "$EQUIV_COORDS" square_unit_graph_equiv 0 "$WORK/equiv" \
  > "$WORK/equiv.out"
rg -q '^status=FORMAT_VALID_RATIONAL_UNIT_GRAPH_SOURCE_PACKAGE$' "$WORK/equiv.out"
rg -q '^m=4$' "$WORK/equiv.out"

if python3 "$MAKER" "$COORDS" 'bad/id' 0 "$WORK/bad-id" \
    > "$WORK/bad-id.out" 2>&1; then
  echo "error: unit-graph maker accepted invalid candidate id" >&2
  exit 1
fi
rg -q "^error: candidate-id must not contain path separators$" \
  "$WORK/bad-id.out"

if python3 "$MAKER" "$COORDS" '..' 0 "$WORK/bad-parent-id" \
    > "$WORK/bad-parent-id.out" 2>&1; then
  echo "error: unit-graph maker accepted parent-like candidate id" >&2
  exit 1
fi
rg -q "^error: candidate-id must not be '.' or contain '..'$" \
  "$WORK/bad-parent-id.out"

BAD_MISSING="$WORK/bad-missing.coords.csv"
cat > "$BAD_MISSING" <<'EOF'
id,x,y
0,0,0
2,1,1
EOF
if python3 "$MAKER" "$BAD_MISSING" missing_id 0 "$WORK/bad-missing" \
    > "$WORK/bad-missing.out" 2>&1; then
  echo "error: unit-graph maker accepted missing coordinate id" >&2
  exit 1
fi
rg -q 'missing coordinate rows for vertices 1' "$WORK/bad-missing.out"

BAD_HUGE_ID="$WORK/bad-huge-id.coords.csv"
cat > "$BAD_HUGE_ID" <<'EOF'
id,x,y
0,0,0
4096,1,0
EOF
if python3 "$MAKER" "$BAD_HUGE_ID" huge_id 0 "$WORK/bad-huge-id" \
    > "$WORK/bad-huge-id.out" 2>&1; then
  echo "error: unit-graph maker accepted an excessive sparse coordinate id" >&2
  exit 1
fi
rg -q 'vertex id 4096 exceeds --max-vertices limit 4096' "$WORK/bad-huge-id.out"

BAD_LEADING_ZERO="$WORK/bad-leading-zero.coords.csv"
cat > "$BAD_LEADING_ZERO" <<'EOF'
id,x,y
0,0,0
01,1,0
EOF
if python3 "$MAKER" "$BAD_LEADING_ZERO" leading_zero 0 "$WORK/bad-leading-zero" \
    > "$WORK/bad-leading-zero.out" 2>&1; then
  echo "error: unit-graph maker accepted a leading-zero vertex id" >&2
  exit 1
fi
rg -q "vertex id must not have leading zeros: '01'" "$WORK/bad-leading-zero.out"

BAD_DUP_ID="$WORK/bad-duplicate-id.coords.csv"
cat > "$BAD_DUP_ID" <<'EOF'
id,x,y
0,0,0
1,1,0
1,0,1
EOF
if python3 "$MAKER" "$BAD_DUP_ID" duplicate_id 0 "$WORK/bad-dup-id" \
    > "$WORK/bad-dup-id.out" 2>&1; then
  echo "error: unit-graph maker accepted duplicate coordinate id" >&2
  exit 1
fi
rg -q 'duplicate vertex id: 1' "$WORK/bad-dup-id.out"

BAD_COLLAPSED="$WORK/bad-collapsed.coords.csv"
cat > "$BAD_COLLAPSED" <<'EOF'
id,x,y
0,0,0
1,0,0
EOF
if python3 "$MAKER" "$BAD_COLLAPSED" collapsed 0 "$WORK/bad-collapsed" \
    > "$WORK/bad-collapsed.out" 2>&1; then
  echo "error: unit-graph maker accepted collapsed coordinates" >&2
  exit 1
fi
rg -q 'duplicate coordinates .* for vertices 0 and 1' "$WORK/bad-collapsed.out"

NO_UNIT="$WORK/no-unit.coords.csv"
cat > "$NO_UNIT" <<'EOF'
id,x,y
0,0,0
1,2,0
EOF
if python3 "$MAKER" "$NO_UNIT" no_unit 0 "$WORK/no-unit" \
    > "$WORK/no-unit.out" 2>&1; then
  echo "error: unit-graph maker accepted a coordinate set with no unit edges" >&2
  exit 1
fi
rg -q 'derived unit graph has 0 edges, below required minimum 1' "$WORK/no-unit.out"

ISOLATED_SPLIT="$WORK/isolated-split.coords.csv"
cat > "$ISOLATED_SPLIT" <<'EOF'
id,x,y
0,0,0
1,1,0
2,3,0
EOF
if python3 "$MAKER" "$ISOLATED_SPLIT" isolated_split 2 "$WORK/isolated-split" \
    > "$WORK/isolated-split.out" 2>&1; then
  echo "error: unit-graph maker accepted an isolated split vertex" >&2
  exit 1
fi
rg -q 'split vertex 2 has degree 0, below required minimum 2' "$WORK/isolated-split.out"

NEAR_UNIT="$WORK/near-unit.coords.csv"
cat > "$NEAR_UNIT" <<'EOF'
id,x,y
0,0,0
1,1000000000000001/1000000000000000,0
2,0,1
EOF
if python3 "$MAKER" "$NEAR_UNIT" near_unit 0 "$WORK/near-unit" --min-edges 2 \
    > "$WORK/near-unit.out" 2>&1; then
  echo "error: unit-graph maker rounded a near-unit edge to unit" >&2
  exit 1
fi
rg -q 'derived unit graph has 1 edges, below required minimum 2' "$WORK/near-unit.out"

if python3 "$MAKER" "$COORDS" bad_split 0,4 "$WORK/bad-split" \
    > "$WORK/bad-split.out" 2>&1; then
  echo "error: unit-graph maker accepted out-of-range split vertex" >&2
  exit 1
fi
rg -q 'split vertex out of range: 4' "$WORK/bad-split.out"

echo "chi6_rational_unit_graph_source_package_gate: PASS"
