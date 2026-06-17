#!/usr/bin/env bash
# Package an external DIMACS graph plus exact rational coordinates as a
# geometry-only chi>=6 candidate manifest.
#
# This bridge is intentionally non-promotable: it proves the listed edges are
# exact unit edges over Rat^2, but it attaches no SAT/LRAT no-5-colouring proof
# and makes no chromatic-number claim.
set -euo pipefail

usage() {
  echo "usage: $0 <candidate-source.json>" >&2
  echo "   or: $0 <edge-file> <coords-csv> <candidate-id>" >&2
  echo "example: WORK=/tmp/square $0 square.candidate-source.json" >&2
}

if [[ $# -ne 1 && $# -ne 3 ]]; then
  usage
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-$(mktemp -d)}"
K=5

SEARCH="$ROOT/examples/erdos/chi6_candidate_search_manifest.py"
GEOM_GEN="$ROOT/examples/erdos/gen_lean_rational_geometry.py"
SOURCE_VALIDATOR="$ROOT/examples/erdos/validate_chi6_solver_candidate_package.py"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"
LOCK="$ROOT/scripts/dev/souc-build-lock.sh"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi

[[ -n "$LAKE" ]] || { echo "error: lake not found; set LAKE=/path/to/lake" >&2; exit 127; }
[[ -x "$LOCK" ]] || { echo "error: missing build lock helper: $LOCK" >&2; exit 1; }
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }

python3 -m py_compile "$SEARCH" "$GEOM_GEN" "$SOURCE_VALIDATOR"

mkdir -p "$WORK/package"
PACKAGE_OUT="$WORK/package.out"
OUT_LEAN="$WORK/SounioChi6RationalGeometryCandidate.lean"
MANIFEST="$WORK/candidate.manifest"
SOURCE_OUT="$WORK/source_validator.out"

get_source_field() {
  local key="$1"
  awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}' "$SOURCE_OUT"
}

if [[ $# -eq 1 ]]; then
  "$SOURCE_VALIDATOR" "$1" > "$SOURCE_OUT"
  rg -q '^status=VALID_SOLVER_CANDIDATE_PACKAGE$' "$SOURCE_OUT"
  EDGE_IN="$(get_source_field edge_path_abs)"
  COORD_IN="$(get_source_field coords_path_abs)"
  CANDIDATE_ID="$(get_source_field candidate_id)"
  SPLIT_VERTICES="$(get_source_field split_vertices)"
else
  EDGE_IN="$1"
  COORD_IN="$2"
  CANDIDATE_ID="$3"
  SPLIT_VERTICES="0"
fi

[[ -s "$EDGE_IN" ]] || { echo "error: missing/empty edge file: $EDGE_IN" >&2; exit 2; }
[[ -s "$COORD_IN" ]] || { echo "error: missing/empty coordinate CSV: $COORD_IN" >&2; exit 2; }
[[ "$CANDIDATE_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "error: candidate-id must use only letters, digits, '.', '_', or '-'" >&2
  exit 2
}
[[ -n "$SPLIT_VERTICES" ]] || { echo "error: split vertices cannot be empty" >&2; exit 2; }
PKG_COORD="$WORK/package/$CANDIDATE_ID.coords.csv"

echo "chi6_rational_geometry_candidate: workdir=$WORK"
python3 "$SEARCH" "$WORK/package" --edge-file "$EDGE_IN" --k "$K" \
  --candidate-id "$CANDIDATE_ID" --split-vertices "$SPLIT_VERTICES" > "$PACKAGE_OUT"
rg -q '^family=external_dimacs_edge$' "$PACKAGE_OUT"
rg -q '^finite_graph_search_claim=none_external_graph_packaging_only$' "$PACKAGE_OUT"
rg -q '^status=EXTERNAL_GRAPH_PACKAGED_UNPROMOTABLE$' "$PACKAGE_OUT"

PKG_EDGE="$WORK/package/$CANDIDATE_ID.edge"
PKG_CUBES="$WORK/package/$CANDIDATE_ID.cubes"
PKG_META="$WORK/package/$CANDIDATE_ID.meta.json"
[[ -s "$PKG_EDGE" ]] || { echo "error: missing packaged edge: $PKG_EDGE" >&2; exit 1; }
[[ -s "$PKG_CUBES" ]] || { echo "error: missing packaged cube batch: $PKG_CUBES" >&2; exit 1; }
[[ -s "$PKG_META" ]] || { echo "error: missing packaged metadata: $PKG_META" >&2; exit 1; }
cp "$COORD_IN" "$PKG_COORD"

python3 "$GEOM_GEN" "$PKG_EDGE" "$PKG_COORD" "$OUT_LEAN" \
  --module SounioChi6RationalGeometryCandidate \
  --namespace SounioChi6RationalGeometryCandidate \
  --prefix chi6rat > "$WORK/gen_geometry.out"
rg -q '^geometry_claim=exact_rational_squared_distance_edges_only$' "$WORK/gen_geometry.out"
rg -q '^sat_claim=none$' "$WORK/gen_geometry.out"
rg -q '^chromatic_claim=none$' "$WORK/gen_geometry.out"
rg -q '^promotable=0$' "$WORK/gen_geometry.out"

(
  cd "$ROOT/formal/lean4"
  "$LOCK" "$LAKE" env lean "$OUT_LEAN" > "$WORK/geometry_lean.log" 2>&1
)
if rg -q '\b(sorry|admit)\b|#exit|#eval|#check' "$OUT_LEAN"; then
  echo "error: generated rational geometry module contains forbidden proof/debug marker" >&2
  exit 1
fi
if rg -q 'sorryAx|error:' "$WORK/geometry_lean.log"; then
  cat "$WORK/geometry_lean.log" >&2
  echo "error: generated rational geometry module failed Lean/no-sorry gate" >&2
  exit 1
fi

sha() {
  [[ -f "$1" ]] || { echo "error: file not found for hashing: $1" >&2; exit 1; }
  local out
  out="$(sha256sum "$1")" || { echo "error: failed to hash $1" >&2; exit 1; }
  awk '{print $1}' <<< "$out"
}

read -r N M < <(PYTHONPATH="$ROOT/examples/erdos${PYTHONPATH:+:$PYTHONPATH}" python3 - "$PKG_EDGE" <<'PY'
import sys
from pathlib import Path
from cube_sieve_propagation_manifest import parse_edge_file

path = Path(sys.argv[1])
headers = 0
with path.open("r", encoding="ascii") as f:
    for raw in f:
        parts = raw.strip().split()
        if len(parts) >= 2 and parts[0] == "p" and parts[1] == "edge":
            headers += 1
if headers != 1:
    raise SystemExit(f"error: expected exactly one p edge header, found {headers}")
n, m, _edges = parse_edge_file(path)
print(n, m)
PY
)
[[ "$N" =~ ^[1-9][0-9]*$ && "$M" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: packaged edge has invalid p edge header" >&2
  exit 1
}
GENERATOR_COMMIT="${SOUNIO_GENERATOR_COMMIT:-}"
if [[ -z "$GENERATOR_COMMIT" ]]; then
  GENERATOR_COMMIT="$(git -C "$ROOT" rev-parse --verify HEAD 2>/dev/null)" || {
    echo "error: unable to resolve generator git commit; set SOUNIO_GENERATOR_COMMIT for payload snapshots without .git" >&2
    exit 1
  }
fi
[[ "$GENERATOR_COMMIT" =~ ^[0-9a-fA-F]{40}$ ]] || {
  echo "error: generator commit must be a full 40-hex SHA" >&2
  exit 1
}

cat > "$MANIFEST" <<EOF
candidate_manifest_version=1
promotable=0
candidate_id=$CANDIDATE_ID
n=$N
m=$M
k=$K
edge_path=package/$CANDIDATE_ID.edge
edge_sha256=$(sha "$PKG_EDGE")
cnf_path=NONE
cnf_sha256=NONE
drat_or_lrat_path=NONE
drat_or_lrat_sha256=NONE
lean_sat_module_path=NONE
lean_sat_module_sha256=NONE
geometry_module_path=SounioChi6RationalGeometryCandidate.lean
geometry_module_sha256=$(sha "$OUT_LEAN")
geometry_source_path=package/$CANDIDATE_ID.coords.csv
geometry_source_sha256=$(sha "$PKG_COORD")
geometry_proof_type=euclidean
sat_proof_route=none
triangle_sb=none
generator_commit=$GENERATOR_COMMIT
producer_command=WORK=$WORK LAKE=$LAKE examples/erdos/make_chi6_rational_geometry_candidate_manifest.sh $EDGE_IN $COORD_IN $CANDIDATE_ID
lean_build_command=lake env lean SounioChi6RationalGeometryCandidate.lean
offload_review_raw=NONE
offload_review_sha256=NONE
source_meta_path=package/$CANDIDATE_ID.meta.json
source_meta_sha256=$(sha "$PKG_META")
cube_batch_path=package/$CANDIDATE_ID.cubes
cube_batch_sha256=$(sha "$PKG_CUBES")
cube_refutation_summary_path=NONE
cube_refutation_summary_sha256=NONE
cube_cover_certificate_path=NONE
cube_cover_certificate_sha256=NONE
cube_cover_complement_cnf_path=NONE
cube_cover_complement_cnf_sha256=NONE
cube_cover_complement_lrat_path=NONE
cube_cover_complement_lrat_sha256=NONE
lean_module=SounioChi6RationalGeometryCandidate
lean_point_type=UnitDistanceChromatic.SounioChi6RationalGeometryCandidate.chi6rat_point_type
lean_unit_term=UnitDistanceChromatic.SounioChi6RationalGeometryCandidate.chi6rat_unit
lean_geometry_term=UnitDistanceChromatic.SounioChi6RationalGeometryCandidate.chi6rat_geometry
lean_edges_sync_term=UnitDistanceChromatic.SounioChi6RationalGeometryCandidate.edgesSyncSelf
lean_real_unit_term=UnitDistanceChromatic.SounioChi6RationalGeometryCandidate.chi6rat_real_unit
lean_real_emb_term=UnitDistanceChromatic.SounioChi6RationalGeometryCandidate.chi6rat_real_emb
lean_real_unit_edges_term=UnitDistanceChromatic.SounioChi6RationalGeometryCandidate.chi6rat_real_unit_edges
lean_real_unit_iff_standard=UnitDistanceChromatic.SounioChi6RationalGeometryCandidate.chi6rat_real_unit_iff_standard
lean_real_final_theorem=NONE
EOF

"$VALIDATOR" "$MANIFEST" | tee "$WORK/manifest_validator.log"

echo "manifest=$MANIFEST"
echo "package_manifest=$PACKAGE_OUT"
echo "edge=$PKG_EDGE"
echo "coords=$PKG_COORD"
echo "source_meta=$PKG_META"
echo "cube_batch=$PKG_CUBES"
echo "geometry_module=$OUT_LEAN"
if [[ -s "$SOURCE_OUT" ]]; then
  echo "source_validator=$SOURCE_OUT"
fi
echo "chi6_rational_geometry_candidate: PASS"
