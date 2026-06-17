#!/usr/bin/env bash
# Gate the integrated source-bundle preflight classifier.
#
# The rational square is geometry-valid but trivially 5-colourable because it is
# 2-colourable, so the expected result is geometry PASS, SAT FAIL, and an
# incomplete integrated status. That is the honest shape until a real chi6
# source bundle exists.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
PREFLIGHT="$ROOT/examples/erdos/make_chi6_integrated_candidate_preflight.sh"

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
    "candidate_id": "square_integrated_preflight_smoke",
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

echo "chi6_integrated_candidate_preflight_gate: workdir=$WORK"
PREFLIGHT_WORK="$WORK/preflight"
WORK="$PREFLIGHT_WORK" "$PREFLIGHT" "$SOURCE" > "$WORK/preflight.out"

rg -q '^source_status=PASS$' "$WORK/preflight.out"
rg -q '^geometry_status=PASS$' "$WORK/preflight.out"
rg -q '^sat_status=FAIL$' "$WORK/preflight.out"
rg -q '^integrated_status=INCOMPLETE$' "$WORK/preflight.out"
rg -q '^first_blocker=sat_no5_cube_cover_refutation_absent$' "$WORK/preflight.out"
rg -q '^sat_route_mode=split_vertices_product_cube_cover$' "$WORK/preflight.out"
rg -q '^promotable=0$' "$WORK/preflight.out"
rg -q '^chromatic_claim=none$' "$WORK/preflight.out"
rg -q '^chi6_integrated_candidate_preflight: PASS$' "$WORK/preflight.out"

GEOM_MANIFEST="$(awk -F= '$1 == "geometry_manifest" {print $2; exit}' "$WORK/preflight.out")"
SAT_MANIFEST="$(awk -F= '$1 == "sat_manifest" {print $2; exit}' "$WORK/preflight.out")"
[[ "$GEOM_MANIFEST" != "NONE" ]] || { echo "error: geometry manifest unexpectedly NONE" >&2; exit 1; }
[[ -s "$GEOM_MANIFEST" ]] || { echo "error: geometry manifest missing: $GEOM_MANIFEST" >&2; exit 1; }
[[ "$SAT_MANIFEST" == "NONE" ]] || { echo "error: square smoke unexpectedly produced SAT manifest: $SAT_MANIFEST" >&2; exit 1; }
rg -q '^geometry_proof_type=euclidean$' "$GEOM_MANIFEST"
rg -q '^sat_proof_route=none$' "$GEOM_MANIFEST"

SAT_ERR="$(awk -F= '$1 == "sat_stderr" {print $2; exit}' "$WORK/preflight.out")"
SAT_OUT="$(awk -F= '$1 == "sat_stdout" {print $2; exit}' "$WORK/preflight.out")"
[[ -s "$SAT_ERR" || -s "$SAT_OUT" ]] || {
  echo "error: expected SAT failure logs to be present" >&2
  exit 1
}
rg -q 'without UNSAT marker|failed with exit|not.*UNSAT|satisf' "$SAT_ERR" "$SAT_OUT"

ARBITRARY_CUBES="$WORK/square_arbitrary.cubes"
ARBITRARY_DRUP="$WORK/square_arbitrary_cover.drup"
cat > "$ARBITRARY_CUBES" <<'EOF'
square_v0_c0: 0:0
EOF
printf '0\n' > "$ARBITRARY_DRUP"

WORK="$WORK/preflight_arbitrary" "$PREFLIGHT" "$SOURCE" "$ARBITRARY_CUBES" "$ARBITRARY_DRUP" \
  > "$WORK/preflight_arbitrary.out"
rg -q '^source_status=PASS$' "$WORK/preflight_arbitrary.out"
rg -q '^geometry_status=PASS$' "$WORK/preflight_arbitrary.out"
rg -q '^sat_route_mode=arbitrary_complement_cube_cover$' "$WORK/preflight_arbitrary.out"
rg -q "^cube_batch_input=$ARBITRARY_CUBES$" "$WORK/preflight_arbitrary.out"
rg -q "^cover_drup_or_rup_input=$ARBITRARY_DRUP$" "$WORK/preflight_arbitrary.out"
rg -q '^sat_status=FAIL$' "$WORK/preflight_arbitrary.out"
rg -q '^sat_blocker=sat_arbitrary_cube_cover_refutation_absent$' "$WORK/preflight_arbitrary.out"
rg -q '^integrated_status=INCOMPLETE$' "$WORK/preflight_arbitrary.out"
rg -q '^first_blocker=sat_arbitrary_cube_cover_refutation_absent$' "$WORK/preflight_arbitrary.out"
rg -q '^promotable=0$' "$WORK/preflight_arbitrary.out"
rg -q '^chromatic_claim=none$' "$WORK/preflight_arbitrary.out"

EMPTY_CUBES="$WORK/empty.cubes"
: > "$EMPTY_CUBES"
WORK="$WORK/preflight_arbitrary_missing" "$PREFLIGHT" "$SOURCE" "$EMPTY_CUBES" "$ARBITRARY_DRUP" \
  > "$WORK/preflight_arbitrary_missing.out"
rg -q '^source_status=PASS$' "$WORK/preflight_arbitrary_missing.out"
rg -q '^geometry_status=PASS$' "$WORK/preflight_arbitrary_missing.out"
rg -q '^sat_route_mode=arbitrary_complement_cube_cover$' "$WORK/preflight_arbitrary_missing.out"
rg -q '^sat_status=FAIL$' "$WORK/preflight_arbitrary_missing.out"
rg -q '^sat_blocker=sat_arbitrary_cube_cover_inputs_absent$' "$WORK/preflight_arbitrary_missing.out"
rg -q '^first_blocker=sat_arbitrary_cube_cover_inputs_absent$' "$WORK/preflight_arbitrary_missing.out"

BAD_SOURCE="$WORK/bad-source.candidate-source.json"
cp "$SOURCE" "$BAD_SOURCE"
sed -i 's/"edge_sha256": "[0-9a-f]*"/"edge_sha256": "0000000000000000000000000000000000000000000000000000000000000000"/' \
  "$BAD_SOURCE"
if WORK="$WORK/bad_preflight" "$PREFLIGHT" "$BAD_SOURCE" > "$WORK/bad-preflight.out" 2>&1; then
  echo "error: preflight accepted source bundle with bad edge hash" >&2
  exit 1
fi
rg -q 'edge_sha256 mismatch' "$WORK/bad-preflight.out"

echo "chi6_integrated_candidate_preflight_gate: PASS"
