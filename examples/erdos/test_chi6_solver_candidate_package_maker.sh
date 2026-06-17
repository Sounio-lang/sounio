#!/usr/bin/env bash
# Gate the canonical source-package maker for solver-produced chi>=6 candidates.
#
# The square fixture is intentionally not a chromatic witness. The gate proves
# only that a producer's exact rational edge/coordinate output can be packaged,
# validated, and handed to the integrated preflight without changing the claim
# boundary.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

MAKER="$ROOT/examples/erdos/make_chi6_solver_candidate_package.sh"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_solver_candidate_package.py"
PREFLIGHT="$ROOT/examples/erdos/make_chi6_integrated_candidate_preflight.sh"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }
mkdir -p "$WORK"

EDGE="$WORK/square.edge"
COORDS="$WORK/square.coords.csv"
cat > "$EDGE" <<'EOF'
p edge 4 4
e 1 2
e 2 3
e 3 4
e 4 1
EOF

cat > "$COORDS" <<'EOF'
id,x,y
0,0,0
1,1,0
2,1,1
3,0,1
EOF

echo "chi6_solver_candidate_package_maker_gate: workdir=$WORK"
PKG_WORK="$WORK/pkg"
WORK="$PKG_WORK" "$MAKER" "$EDGE" "$COORDS" square_solver_source_maker 0,1 \
  > "$WORK/maker.out"
rg -q '^chi6_solver_candidate_package v1$' "$WORK/maker.out"
rg -q '^candidate_id=square_solver_source_maker$' "$WORK/maker.out"
rg -q '^n=4$' "$WORK/maker.out"
rg -q '^m=4$' "$WORK/maker.out"
rg -q '^k=5$' "$WORK/maker.out"
rg -q '^validated_edge_count=4$' "$WORK/maker.out"
rg -q '^coordinate_row_count=4$' "$WORK/maker.out"
rg -q '^split_vertices=0,1$' "$WORK/maker.out"
rg -q '^geometry_claim=exact_rational_squared_distance_edges_only$' "$WORK/maker.out"
rg -q '^sat_claim=none$' "$WORK/maker.out"
rg -q '^chromatic_claim=none$' "$WORK/maker.out"
rg -q '^promotable=0$' "$WORK/maker.out"
rg -q '^status=VALID_SOLVER_CANDIDATE_PACKAGE$' "$WORK/maker.out"
rg -q '^chi6_solver_candidate_package_maker: PASS$' "$WORK/maker.out"

SOURCE="$PKG_WORK/square_solver_source_maker.candidate-source.json"
PKG_EDGE="$PKG_WORK/package/square_solver_source_maker.edge"
PKG_COORDS="$PKG_WORK/package/square_solver_source_maker.coords.csv"
[[ -s "$SOURCE" ]] || { echo "error: missing source package JSON" >&2; exit 1; }
[[ -s "$PKG_EDGE" ]] || { echo "error: missing packaged edge file" >&2; exit 1; }
[[ -s "$PKG_COORDS" ]] || { echo "error: missing packaged coords file" >&2; exit 1; }
cmp -s "$EDGE" "$PKG_EDGE"
cmp -s "$COORDS" "$PKG_COORDS"

"$VALIDATOR" "$SOURCE" > "$WORK/source-validator.out"
rg -q '^status=VALID_SOLVER_CANDIDATE_PACKAGE$' "$WORK/source-validator.out"
rg -q '^validated_edge_count=4$' "$WORK/source-validator.out"
rg -q '^coordinate_row_count=4$' "$WORK/source-validator.out"

python3 - "$SOURCE" "$PKG_EDGE" "$PKG_COORDS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

source, edge, coords = map(Path, sys.argv[1:])
meta = json.loads(source.read_text(encoding="ascii"))

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert meta["schema"] == "chi6_solver_candidate_package.v1"
assert meta["candidate_id"] == "square_solver_source_maker"
assert meta["edge_path"] == "package/square_solver_source_maker.edge"
assert meta["coords_path"] == "package/square_solver_source_maker.coords.csv"
assert meta["edge_sha256"] == digest(edge)
assert meta["coords_sha256"] == digest(coords)
assert meta["coordinate_domain"] == "rational_xy"
assert meta["split_vertices"] == [0, 1]
assert meta["claim_scope"] == "solver_candidate_source_only"
assert meta["promotion_gate"] == "requires_cube_cover_lrat_lean_exact_geometry_and_real_bridge"
assert "sat_claim" not in meta
assert "chromatic_claim" not in meta
assert "promotable" not in meta
PY

BAD_EXTRA="$WORK/bad-extra-claim.candidate-source.json"
cp "$SOURCE" "$BAD_EXTRA"
python3 - "$BAD_EXTRA" <<'PY'
import json
import sys
path = sys.argv[1]
meta = json.load(open(path, encoding="ascii"))
meta["sat_claim"] = "found_no5_certificate"
meta["chromatic_claim"] = "chi_ge_6"
meta["promotable"] = 1
open(path, "w", encoding="ascii").write(json.dumps(meta, indent=2, sort_keys=True) + "\n")
PY
if "$VALIDATOR" "$BAD_EXTRA" > "$WORK/bad-extra-claim.out" 2>&1; then
  echo "error: source validator accepted extra claim/promotion fields" >&2
  exit 1
fi
rg -q 'unexpected keys: chromatic_claim,promotable,sat_claim' "$WORK/bad-extra-claim.out"

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
EQUIV_WORK="$WORK/equiv"
WORK="$EQUIV_WORK" "$MAKER" "$EDGE" "$EQUIV_COORDS" square_equiv_rationals 0 \
  > "$WORK/equiv.out"
rg -q '^status=VALID_SOLVER_CANDIDATE_PACKAGE$' "$WORK/equiv.out"
rg -q '^candidate_id=square_equiv_rationals$' "$WORK/equiv.out"

if WORK="$WORK/bad_id" "$MAKER" "$EDGE" "$COORDS" 'bad/id' 0 \
    > "$WORK/bad-id.out" 2>&1; then
  echo "error: maker accepted invalid candidate id" >&2
  exit 1
fi
rg -q "^error: candidate-id must use only letters, digits, '.', '_', or '-'$" \
  "$WORK/bad-id.out"

if WORK="$WORK/bad_traversal_id" "$MAKER" "$EDGE" "$COORDS" '../bad' 0 \
    > "$WORK/bad-traversal-id.out" 2>&1; then
  echo "error: maker accepted path-traversal candidate id" >&2
  exit 1
fi
rg -q "^error: candidate-id must use only letters, digits, '.', '_', or '-'$" \
  "$WORK/bad-traversal-id.out"

if WORK="$WORK/bad_split" "$MAKER" "$EDGE" "$COORDS" square_bad_split 0,0 \
    > "$WORK/bad-split.out" 2>&1; then
  echo "error: maker accepted duplicate split vertex" >&2
  exit 1
fi
rg -q 'duplicate split vertex: 0' "$WORK/bad-split.out"

BAD_DUP_COORDS="$WORK/bad-duplicate-id.coords.csv"
cat > "$BAD_DUP_COORDS" <<'EOF'
id,x,y
0,0,0
1,1,0
1,1,1
3,0,1
EOF
if WORK="$WORK/bad_dup_coords" "$MAKER" "$EDGE" "$BAD_DUP_COORDS" square_bad_dup_coords 0 \
    > "$WORK/bad-dup-coords.out" 2>&1; then
  echo "error: maker accepted duplicate coordinate vertex id" >&2
  exit 1
fi
rg -q 'duplicate vertex id: 1' "$WORK/bad-dup-coords.out"

BAD_COLLAPSED_COORDS="$WORK/bad-collapsed.coords.csv"
cat > "$BAD_COLLAPSED_COORDS" <<'EOF'
id,x,y
0,0,0
1,0,0
2,1,1
3,0,1
EOF
if WORK="$WORK/bad_collapsed" "$MAKER" "$EDGE" "$BAD_COLLAPSED_COORDS" square_bad_collapsed 0 \
    > "$WORK/bad-collapsed.out" 2>&1; then
  echo "error: maker accepted collapsed coordinates" >&2
  exit 1
fi
rg -q 'duplicate coordinates for vertices 0 and 1' "$WORK/bad-collapsed.out"

BAD_COORDS="$WORK/bad.coords.csv"
cp "$COORDS" "$BAD_COORDS"
sed -i 's/^1,1,0/1,2,0/' "$BAD_COORDS"
if WORK="$WORK/bad_geom" "$MAKER" "$EDGE" "$BAD_COORDS" square_bad_geom 0 \
    > "$WORK/bad-geom.out" 2>&1; then
  echo "error: maker accepted non-unit rational geometry" >&2
  exit 1
fi
rg -q 'edge 0,1 has dist2=4, expected 1' "$WORK/bad-geom.out"

BAD_NEAR_UNIT="$WORK/bad-near-unit.coords.csv"
cat > "$BAD_NEAR_UNIT" <<'EOF'
id,x,y
0,0,0
1,1000000000000001/1000000000000000,0
2,1,1
3,0,1
EOF
if WORK="$WORK/bad_near_unit" "$MAKER" "$EDGE" "$BAD_NEAR_UNIT" square_bad_near_unit 0 \
    > "$WORK/bad-near-unit.out" 2>&1; then
  echo "error: maker accepted near-unit rational geometry" >&2
  exit 1
fi
rg -q 'expected 1' "$WORK/bad-near-unit.out"

BAD_DIMACS="$WORK/bad-count.edge"
cat > "$BAD_DIMACS" <<'EOF'
p edge 4 5
e 1 2
e 2 3
e 3 4
e 4 1
EOF
if WORK="$WORK/bad_dimacs" "$MAKER" "$BAD_DIMACS" "$COORDS" square_bad_dimacs 0 \
    > "$WORK/bad-dimacs.out" 2>&1; then
  echo "error: maker accepted DIMACS edge-count mismatch" >&2
  exit 1
fi
rg -q 'p edge declares 5 edges but found 4' "$WORK/bad-dimacs.out"

echo "chi6_solver_candidate_package_maker_gate: PASS"
