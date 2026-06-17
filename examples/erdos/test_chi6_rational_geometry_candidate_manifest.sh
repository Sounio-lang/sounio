#!/usr/bin/env bash
# Gate the rational-coordinate geometry-only candidate.manifest bridge.
#
# The square fixture is a unit-distance geometry smoke, not a no-5-colouring
# witness. This gate checks that exact geometry can be packaged into the same
# candidate manifest shape used by downstream SAT/LRAT routes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
MAKER="$ROOT/examples/erdos/make_chi6_rational_geometry_candidate_manifest.sh"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"
HANDOFF_MAKER="$ROOT/examples/erdos/make_chi6_foundry_handoff_package.sh"
HANDOFF_VALIDATOR="$ROOT/examples/erdos/validate_chi6_foundry_handoff_package.sh"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }
mkdir -p "$WORK"

EDGE="$WORK/square.edge"
COORD="$WORK/square.coords.csv"
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

echo "chi6_rational_geometry_candidate_gate: workdir=$WORK"
PKG_WORK="$WORK/ratgeom"
WORK="$PKG_WORK" "$MAKER" "$EDGE" "$COORD" square_ratgeom_smoke \
  > "$WORK/maker.out"

MANIFEST="$PKG_WORK/candidate.manifest"
GEOM="$PKG_WORK/SounioChi6RationalGeometryCandidate.lean"
[[ -s "$MANIFEST" ]] || { echo "error: maker did not emit candidate.manifest" >&2; exit 1; }
[[ -s "$GEOM" ]] || { echo "error: maker did not emit geometry module" >&2; exit 1; }

rg -q '^candidate_id=square_ratgeom_smoke$' "$MANIFEST"
rg -q '^promotable=0$' "$MANIFEST"
rg -q '^geometry_proof_type=euclidean$' "$MANIFEST"
rg -q '^sat_proof_route=none$' "$MANIFEST"
rg -q '^geometry_module_path=SounioChi6RationalGeometryCandidate\.lean$' "$MANIFEST"
rg -q '^geometry_source_path=package/square_ratgeom_smoke\.coords\.csv$' "$MANIFEST"
rg -q '^lean_sat_module_path=NONE$' "$MANIFEST"
rg -q '^lean_geometry_term=UnitDistanceChromatic\.SounioChi6RationalGeometryCandidate\.chi6rat_geometry$' \
  "$MANIFEST"
rg -q '^lean_edges_sync_term=UnitDistanceChromatic\.SounioChi6RationalGeometryCandidate\.edgesSyncSelf$' \
  "$MANIFEST"
rg -q '^lean_real_unit_term=UnitDistanceChromatic\.SounioChi6RationalGeometryCandidate\.chi6rat_real_unit$' \
  "$MANIFEST"
rg -q '^lean_real_emb_term=UnitDistanceChromatic\.SounioChi6RationalGeometryCandidate\.chi6rat_real_emb$' \
  "$MANIFEST"
rg -q '^lean_real_unit_edges_term=UnitDistanceChromatic\.SounioChi6RationalGeometryCandidate\.chi6rat_real_unit_edges$' \
  "$MANIFEST"
rg -q '^lean_real_unit_iff_standard=UnitDistanceChromatic\.SounioChi6RationalGeometryCandidate\.chi6rat_real_unit_iff_standard$' \
  "$MANIFEST"
rg -q '^lean_real_final_theorem=NONE$' "$MANIFEST"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=square_ratgeom_smoke$' \
  "$WORK/maker.out"
rg -q '^chi6_rational_geometry_candidate: PASS$' "$WORK/maker.out"

cmp -s "$EDGE" "$PKG_WORK/package/square_ratgeom_smoke.edge"
cmp -s "$COORD" "$PKG_WORK/package/square_ratgeom_smoke.coords.csv"
rg -q '^def euclideanGeometry : EuclideanNatEdgeExactGeometry 4' "$GEOM"
rg -q '^theorem edgesSyncSelf : euclideanGeometry\.exact\.edges = edges := rfl$' "$GEOM"
rg -q '^abbrev chi6rat_geometry := euclideanGeometry$' "$GEOM"
rg -q '^abbrev chi6rat_real_unit := realUnit$' "$GEOM"
rg -q '^abbrev chi6rat_real_unit_iff_standard := realUnit_iff_standard$' "$GEOM"
rg -q '^abbrev chi6rat_real_emb := realEmb$' "$GEOM"
rg -q '^abbrev chi6rat_real_unit_edges := realUnitEdges$' "$GEOM"
if rg -q '\b(sorry|admit)\b|#exit|#eval|#check' "$GEOM"; then
  echo "error: generated geometry module contains forbidden proof/debug marker" >&2
  exit 1
fi

"$VALIDATOR" "$MANIFEST" > "$WORK/validator.out"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=square_ratgeom_smoke$' \
  "$WORK/validator.out"

BAD_HASH="$PKG_WORK/bad-geometry-source.env"
cp "$MANIFEST" "$BAD_HASH"
sed -i 's/^geometry_source_sha256=.*/geometry_source_sha256=0000000000000000000000000000000000000000000000000000000000000000/' \
  "$BAD_HASH"
if "$VALIDATOR" "$BAD_HASH" > "$WORK/bad-geometry-source.out" 2>&1; then
  echo "error: validator accepted a bad geometry_source hash" >&2
  exit 1
fi
rg -q 'geometry_source SHA256 mismatch' "$WORK/bad-geometry-source.out"

BAD_COORD="$WORK/bad.coords.csv"
cp "$COORD" "$BAD_COORD"
sed -i 's/^1,1,0/1,2,0/' "$BAD_COORD"
if WORK="$WORK/bad_nonunit" "$MAKER" "$EDGE" "$BAD_COORD" bad_square_ratgeom \
    > "$WORK/bad_nonunit.out" 2>&1; then
  echo "error: maker accepted non-unit rational geometry" >&2
  exit 1
fi
rg -q 'edge 0,1 has dist2=4, expected 1' "$WORK/bad_nonunit.out"

HANDOFF_OUT="$WORK/handoff"
"$HANDOFF_MAKER" "$MANIFEST" "$HANDOFF_OUT" > "$WORK/handoff-maker.out"
"$HANDOFF_VALIDATOR" "$HANDOFF_OUT" > "$WORK/handoff-validator.out"
rg -q '^geometry_proof_type: euclidean$' "$HANDOFF_OUT/handoff.txt"
rg -q '^geometry_source_path: package/square_ratgeom_smoke\.coords\.csv$' \
  "$HANDOFF_OUT/handoff.txt"
rg -q '^geometry_source_sha256: [0-9a-f]{64}$' "$HANDOFF_OUT/handoff.txt"
rg -q '^chi6-package/package/square_ratgeom_smoke\.coords\.csv$' \
  <(awk '{print $2}' "$HANDOFF_OUT/SHA256SUMS")
rg -q '^chi6_foundry_handoff_package: VALID candidate=square_ratgeom_smoke promotable=0$' \
  "$WORK/handoff-validator.out"

echo "chi6_rational_geometry_candidate_gate: PASS"
