#!/usr/bin/env bash
# Gate the solver-candidate source package contract.
#
# This is a source/provenance gate: it binds a DIMACS edge file and exact
# rational coordinates before SAT/LRAT promotion. The square fixture is not a
# chi>=6 witness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
VALIDATOR="$ROOT/examples/erdos/validate_chi6_solver_candidate_package.py"
MAKER="$ROOT/examples/erdos/make_chi6_rational_geometry_candidate_manifest.sh"
SCHEMA="$ROOT/examples/erdos/schemas/chi6_solver_candidate_package.v1.schema.json"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }
mkdir -p "$WORK"

EDGE="$WORK/square.edge"
COORD="$WORK/square.coords.csv"
SOURCE="$WORK/square.candidate-source.json"
cat > "$EDGE" <<'EOF'
p edge 4 4
e 1 2
e 2 3
e 3 4
e 4 1
EOF

cat > "$COORD" <<'EOF'
id,x,y
0,0,0
1,1,0
2,1,1
3,0,1
EOF

python3 -m json.tool "$SCHEMA" > "$WORK/schema.pretty.json"
python3 - "$SOURCE" "$EDGE" "$COORD" <<'PY'
import json
import sys
from pathlib import Path
from hashlib import sha256

source, edge, coord = map(Path, sys.argv[1:])

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

meta = {
    "schema": "chi6_solver_candidate_package.v1",
    "candidate_id": "square_solver_source_smoke",
    "edge_path": edge.name,
    "edge_sha256": digest(edge),
    "coords_path": coord.name,
    "coords_sha256": digest(coord),
    "coordinate_domain": "rational_xy",
    "n": 4,
    "m": 4,
    "k": 5,
    "split_vertices": [0, 1],
    "producer_command": "test fixture: rational unit square",
    "claim_scope": "solver_candidate_source_only",
    "promotion_gate": "requires_cube_cover_lrat_lean_exact_geometry_and_real_bridge",
}
source.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY

echo "chi6_solver_candidate_package_gate: workdir=$WORK"
"$VALIDATOR" "$SOURCE" > "$WORK/source-validator.out"
rg -q '^chi6_solver_candidate_package v1$' "$WORK/source-validator.out"
rg -q '^candidate_id=square_solver_source_smoke$' "$WORK/source-validator.out"
rg -q '^split_vertices=0,1$' "$WORK/source-validator.out"
rg -q '^geometry_claim=exact_rational_squared_distance_edges_only$' "$WORK/source-validator.out"
rg -q '^sat_claim=none$' "$WORK/source-validator.out"
rg -q '^chromatic_claim=none$' "$WORK/source-validator.out"
rg -q '^promotable=0$' "$WORK/source-validator.out"
rg -q '^status=VALID_SOLVER_CANDIDATE_PACKAGE$' "$WORK/source-validator.out"

PKG_WORK="$WORK/from_source"
WORK="$PKG_WORK" "$MAKER" "$SOURCE" > "$WORK/maker.out"
rg -q '^chi6_rational_geometry_candidate: PASS$' "$WORK/maker.out"
rg -q '^source_validator=.*/source_validator\.out$' "$WORK/maker.out"
rg -q '^candidate_id=square_solver_source_smoke$' "$PKG_WORK/candidate.manifest"
rg -q '^geometry_source_path=package/square_solver_source_smoke\.coords\.csv$' \
  "$PKG_WORK/candidate.manifest"
rg -q '^cube_batch_path=package/square_solver_source_smoke\.cubes$' \
  "$PKG_WORK/candidate.manifest"
rg -q '^square_solver_source_smoke_v0_c0_v1_c1: 0:0 1:1$' \
  "$PKG_WORK/package/square_solver_source_smoke.cubes"

BAD_HASH="$WORK/bad-hash.candidate-source.json"
cp "$SOURCE" "$BAD_HASH"
sed -i 's/"coords_sha256": "[0-9a-f]*"/"coords_sha256": "0000000000000000000000000000000000000000000000000000000000000000"/' \
  "$BAD_HASH"
if "$VALIDATOR" "$BAD_HASH" > "$WORK/bad-hash.out" 2>&1; then
  echo "error: source validator accepted bad coords hash" >&2
  exit 1
fi
rg -q 'coords_sha256 mismatch' "$WORK/bad-hash.out"

BAD_SPLIT="$WORK/bad-split.candidate-source.json"
cp "$SOURCE" "$BAD_SPLIT"
python3 - "$BAD_SPLIT" <<'PY'
import json
import sys
path = sys.argv[1]
meta = json.load(open(path, encoding="ascii"))
meta["split_vertices"] = [0, 4]
open(path, "w", encoding="ascii").write(json.dumps(meta, indent=2, sort_keys=True) + "\n")
PY
if "$VALIDATOR" "$BAD_SPLIT" > "$WORK/bad-split.out" 2>&1; then
  echo "error: source validator accepted out-of-range split vertex" >&2
  exit 1
fi
rg -q 'split_vertices must be non-negative integers below n' "$WORK/bad-split.out"

BAD_DUP_SPLIT="$WORK/bad-duplicate-split.candidate-source.json"
cp "$SOURCE" "$BAD_DUP_SPLIT"
python3 - "$BAD_DUP_SPLIT" <<'PY'
import json
import sys
path = sys.argv[1]
meta = json.load(open(path, encoding="ascii"))
meta["split_vertices"] = [0, 0]
open(path, "w", encoding="ascii").write(json.dumps(meta, indent=2, sort_keys=True) + "\n")
PY
if "$VALIDATOR" "$BAD_DUP_SPLIT" > "$WORK/bad-duplicate-split.out" 2>&1; then
  echo "error: source validator accepted duplicate split vertices" >&2
  exit 1
fi
rg -q 'split_vertices must not contain duplicates' "$WORK/bad-duplicate-split.out"

BAD_ESCAPE="$WORK/bad-escape.candidate-source.json"
cp "$SOURCE" "$BAD_ESCAPE"
python3 - "$BAD_ESCAPE" <<'PY'
import json
import sys
path = sys.argv[1]
meta = json.load(open(path, encoding="ascii"))
meta["edge_path"] = "../square.edge"
open(path, "w", encoding="ascii").write(json.dumps(meta, indent=2, sort_keys=True) + "\n")
PY
if "$VALIDATOR" "$BAD_ESCAPE" > "$WORK/bad-escape.out" 2>&1; then
  echo "error: source validator accepted path traversal" >&2
  exit 1
fi
rg -q 'candidate package path escapes source directory' "$WORK/bad-escape.out"

BAD_COORD="$WORK/bad.coords.csv"
BAD_SOURCE="$WORK/bad-nonunit.candidate-source.json"
cp "$COORD" "$BAD_COORD"
sed -i 's/^1,1,0/1,2,0/' "$BAD_COORD"
python3 - "$SOURCE" "$BAD_SOURCE" "$BAD_COORD" <<'PY'
import json
import sys
from pathlib import Path
from hashlib import sha256

src, dst, bad_coord = map(Path, sys.argv[1:])
meta = json.loads(src.read_text(encoding="ascii"))
meta["coords_path"] = bad_coord.name
meta["coords_sha256"] = sha256(bad_coord.read_bytes()).hexdigest()
dst.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY
if "$VALIDATOR" "$BAD_SOURCE" > "$WORK/bad-nonunit.out" 2>&1; then
  echo "error: source validator accepted non-unit coordinates" >&2
  exit 1
fi
rg -q 'edge 0,1 has dist2=4, expected 1' "$WORK/bad-nonunit.out"

echo "chi6_solver_candidate_package_gate: PASS"
