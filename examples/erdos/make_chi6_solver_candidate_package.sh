#!/usr/bin/env bash
# Package a solver-produced edge/coordinate pair as a chi>=6 source package.
#
# This is the canonical local front door for a real search producer:
#
#   DIMACS p edge graph + exact rational id,x,y coordinates + split vertices
#     -> hash-pinned chi6_solver_candidate_package.v1 JSON
#     -> validator output proving exact listed unit edges only
#
# It deliberately emits no SAT/LRAT proof and no chromatic claim. Downstream
# geometry/SAT/Lean promotion must consume this source JSON.
set -euo pipefail

usage() {
  cat <<'EOF' >&2
usage: make_chi6_solver_candidate_package.sh <edge-file> <coords-csv> <candidate-id> <split-vertices>

Creates in $WORK (or a fresh temp dir):
  package/<candidate-id>.edge
  package/<candidate-id>.coords.csv
  <candidate-id>.candidate-source.json
  source_validator.out

The coordinate CSV must be exact rational `id,x,y`, zero-based vertex ids.
EOF
}

if [[ $# -ne 4 ]]; then
  usage
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EDGE_IN="$1"
COORDS_IN="$2"
CANDIDATE_ID="$3"
SPLIT_VERTICES="$4"
WORK="${WORK:-$(mktemp -d)}"

VALIDATOR="$ROOT/examples/erdos/validate_chi6_solver_candidate_package.py"
GEOM_GEN="$ROOT/examples/erdos/gen_lean_rational_geometry.py"
EDGE_PARSER="$ROOT/examples/erdos/cube_sieve_propagation_manifest.py"
SPLIT_HELPER="$ROOT/examples/erdos/cube_split_batch.py"

[[ -s "$EDGE_IN" ]] || { echo "error: missing/empty edge file: $EDGE_IN" >&2; exit 2; }
[[ -s "$COORDS_IN" ]] || { echo "error: missing/empty coordinate CSV: $COORDS_IN" >&2; exit 2; }
[[ -n "$SPLIT_VERTICES" ]] || { echo "error: split-vertices cannot be empty" >&2; exit 2; }
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }

python3 -m py_compile "$VALIDATOR" "$GEOM_GEN" "$EDGE_PARSER" "$SPLIT_HELPER"

mkdir -p "$WORK/package"
SOURCE_JSON="$WORK/$CANDIDATE_ID.candidate-source.json"
SOURCE_OUT="$WORK/source_validator.out"

PYTHONPATH="$ROOT/examples/erdos${PYTHONPATH:+:$PYTHONPATH}" python3 - \
  "$EDGE_IN" "$COORDS_IN" "$CANDIDATE_ID" "$SPLIT_VERTICES" "$WORK" <<'PY'
import json
import re
import shutil
import sys
from pathlib import Path

from cube_sieve_batch_manifest import sha256_file
from cube_sieve_propagation_manifest import parse_edge_file
from cube_split_batch import parse_split_vertices
from gen_lean_rational_geometry import parse_coords, validate_geometry

edge_in = Path(sys.argv[1])
coords_in = Path(sys.argv[2])
candidate_id = sys.argv[3]
split_raw = sys.argv[4]
work = Path(sys.argv[5])

if not re.fullmatch(r"[A-Za-z0-9_.-]+", candidate_id):
    raise SystemExit("error: candidate-id must use only letters, digits, '.', '_', or '-'")

pkg_dir = work / "package"
pkg_dir.mkdir(parents=True, exist_ok=True)
pkg_edge = pkg_dir / f"{candidate_id}.edge"
pkg_coords = pkg_dir / f"{candidate_id}.coords.csv"
source_json = work / f"{candidate_id}.candidate-source.json"

shutil.copyfile(edge_in, pkg_edge)
shutil.copyfile(coords_in, pkg_coords)

n, m, edges = parse_edge_file(pkg_edge)
split_vertices = parse_split_vertices(split_raw, n)
coords = parse_coords(pkg_coords, n)
validate_geometry(coords, edges)

meta = {
    "schema": "chi6_solver_candidate_package.v1",
    "candidate_id": candidate_id,
    "edge_path": f"package/{candidate_id}.edge",
    "edge_sha256": sha256_file(pkg_edge),
    "coords_path": f"package/{candidate_id}.coords.csv",
    "coords_sha256": sha256_file(pkg_coords),
    "coordinate_domain": "rational_xy",
    "n": n,
    "m": m,
    "k": 5,
    "split_vertices": split_vertices,
    "producer_command": "make_chi6_solver_candidate_package.sh "
        f"{edge_in} {coords_in} {candidate_id} {split_raw}",
    "claim_scope": "solver_candidate_source_only",
    "promotion_gate": "requires_cube_cover_lrat_lean_exact_geometry_and_real_bridge",
}
source_json.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="ascii")

print(f"candidate_id={candidate_id}")
print(f"n={n}")
print(f"m={m}")
print("k=5")
print(f"split_vertices={','.join(str(v) for v in split_vertices)}")
print(f"candidate_source={source_json}")
print(f"edge={pkg_edge}")
print(f"edge_sha256={meta['edge_sha256']}")
print(f"coords={pkg_coords}")
print(f"coords_sha256={meta['coords_sha256']}")
print("claim_scope=solver_candidate_source_only")
print("chromatic_claim=none")
print("promotable=0")
PY

"$VALIDATOR" "$SOURCE_JSON" > "$SOURCE_OUT"
rg -q '^status=VALID_SOLVER_CANDIDATE_PACKAGE$' "$SOURCE_OUT"

echo "chi6_solver_candidate_package_maker: workdir=$WORK"
cat "$SOURCE_OUT"
echo "candidate_source=$SOURCE_JSON"
echo "source_validator=$SOURCE_OUT"
echo "edge=$WORK/package/$CANDIDATE_ID.edge"
echo "coords=$WORK/package/$CANDIDATE_ID.coords.csv"
echo "chi6_solver_candidate_package_maker: PASS"
