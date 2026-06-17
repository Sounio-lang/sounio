#!/usr/bin/env bash
# Fail-closed gate for the chi>=6 promotable candidate assembler.
#
# There is no known real chi>=6 witness in this repo. This gate therefore proves
# the assembler refuses incomplete or mismatched inputs instead of manufacturing
# a `promotable=1` package from geometry-only evidence or unrelated SAT smokes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREFLIGHT="$ROOT/examples/erdos/make_chi6_integrated_candidate_preflight.sh"
GEOM_MAKER="$ROOT/examples/erdos/make_chi6_rational_geometry_candidate_manifest.sh"
SAT_MAKER="$ROOT/examples/erdos/make_chi6_external_cube_cover_candidate_manifest.sh"
GEN_JOIN="$ROOT/examples/erdos/gen_lean_chi6_promotable_candidate.py"
ASSEMBLER="$ROOT/examples/erdos/make_chi6_promotable_candidate_manifest.sh"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }
python3 -m py_compile "$GEN_JOIN"
bash -n "$ASSEMBLER"

OWN_WORK=0
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  OWN_WORK=1
fi
cleanup() {
  if [[ "$OWN_WORK" == "1" ]]; then
    rm -rf "$WORK"
  fi
}
trap cleanup EXIT INT TERM
mkdir -p "$WORK"

cat > "$WORK/square.edge" <<'EOF'
p edge 4 4
e 1 2
e 2 3
e 3 4
e 4 1
EOF

cat > "$WORK/square.coords.csv" <<'EOF'
id,x,y
0,0,0
1,1,0
2,1,1
3,0,1
EOF

cat > "$WORK/square.candidate-source.json" <<EOF
{
  "schema": "chi6_solver_candidate_package.v1",
  "candidate_id": "square_assembler_refuses",
  "edge_path": "square.edge",
  "edge_sha256": "$(sha256sum "$WORK/square.edge" | awk '{print $1}')",
  "coords_path": "square.coords.csv",
  "coords_sha256": "$(sha256sum "$WORK/square.coords.csv" | awk '{print $1}')",
  "coordinate_domain": "rational_xy",
  "n": 4,
  "m": 4,
  "k": 5,
  "split_vertices": [0],
  "producer_command": "test_chi6_promotable_candidate_assembler_gate square fixture",
  "claim_scope": "solver_candidate_source_only",
  "promotion_gate": "requires_cube_cover_lrat_lean_exact_geometry_and_real_bridge"
}
EOF

echo "chi6_promotable_candidate_assembler_gate: workdir=$WORK"
WORK="$WORK/preflight_square" "$PREFLIGHT" "$WORK/square.candidate-source.json" \
  > "$WORK/preflight_square.out"
rg -q '^geometry_status=PASS$' "$WORK/preflight_square.out"
rg -q '^sat_status=FAIL$' "$WORK/preflight_square.out"
rg -q '^integrated_status=INCOMPLETE$' "$WORK/preflight_square.out"
rg -q '^first_blocker=sat_no5_cube_cover_refutation_absent$' "$WORK/preflight_square.out"

if WORK="$WORK/assembler_square" "$ASSEMBLER" "$WORK/preflight_square.out" \
    > "$WORK/assembler_square.out" 2>&1; then
  echo "error: assembler accepted an incomplete square preflight" >&2
  exit 1
fi
rg -q 'preflight is not promotion-ready: geometry_status=PASS sat_status=FAIL integrated_status=INCOMPLETE first_blocker=sat_no5_cube_cover_refutation_absent' \
  "$WORK/assembler_square.out"

WORK="$WORK/geometry_square" "$GEOM_MAKER" "$WORK/square.edge" "$WORK/square.coords.csv" \
  square_assembler_refuses > "$WORK/geometry_square.out"
GEOM_MANIFEST="$WORK/geometry_square/candidate.manifest"
[[ -s "$GEOM_MANIFEST" ]]

cat > "$WORK/k6.edge" <<'EOF'
p edge 6 15
e 1 2
e 1 3
e 1 4
e 1 5
e 1 6
e 2 3
e 2 4
e 2 5
e 2 6
e 3 4
e 3 5
e 3 6
e 4 5
e 4 6
e 5 6
EOF

WORK="$WORK/sat_k6" "$SAT_MAKER" "$WORK/k6.edge" k6_assembler_sat_smoke 0 \
  > "$WORK/sat_k6.out"
SAT_MANIFEST="$WORK/sat_k6/candidate.manifest"
[[ -s "$SAT_MANIFEST" ]]

if python3 "$GEN_JOIN" "$GEOM_MANIFEST" "$SAT_MANIFEST" "$WORK/bad_join.lean" \
    --module SounioChi6BadJoin > "$WORK/bad_join.out" 2>&1; then
  echo "error: join generator accepted mismatched geometry and SAT manifests" >&2
  exit 1
fi
rg -q 'manifest metadata mismatch for candidate_id|manifest metadata mismatch for n|manifest metadata mismatch for edge_sha256' \
  "$WORK/bad_join.out"

BAD_GEOM="$WORK/bad_missing_real.env"
cp "$GEOM_MANIFEST" "$BAD_GEOM"
sed -i '/^lean_real_unit_term=/d' "$BAD_GEOM"
if python3 "$GEN_JOIN" "$BAD_GEOM" "$SAT_MANIFEST" "$WORK/bad_missing_real.lean" \
    --module SounioChi6BadMissingReal > "$WORK/bad_missing_real.out" 2>&1; then
  echo "error: join generator accepted geometry without Real unit term" >&2
  exit 1
fi
rg -q 'geometry manifest missing lean_real_unit_term' "$WORK/bad_missing_real.out"

for missing_real_field in lean_real_emb_term lean_real_unit_edges_term; do
  BAD_REAL="$WORK/bad_missing_${missing_real_field}.env"
  cp "$GEOM_MANIFEST" "$BAD_REAL"
  sed -i "/^${missing_real_field}=/d" "$BAD_REAL"
  if python3 "$GEN_JOIN" "$BAD_REAL" "$SAT_MANIFEST" "$WORK/bad_missing_${missing_real_field}.lean" \
      --module "SounioChi6BadMissing${missing_real_field}" \
      > "$WORK/bad_missing_${missing_real_field}.out" 2>&1; then
    echo "error: join generator accepted geometry without $missing_real_field" >&2
    exit 1
  fi
  rg -q "geometry manifest missing $missing_real_field" \
    "$WORK/bad_missing_${missing_real_field}.out"
done

cat > "$WORK/RouteGeometry.lean" <<'EOF'
-- Static generator fixture only. The generated join is not promoted here.
EOF

cat > "$WORK/route_geometry.env" <<'EOF'
candidate_manifest_version=1
promotable=0
candidate_id=route_fixture
n=4
m=4
k=5
edge_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
geometry_proof_type=euclidean
sat_proof_route=none
geometry_module_path=RouteGeometry.lean
lean_module=RouteGeometry
lean_point_type=RouteGeometry.Point
lean_unit_term=RouteGeometry.unit
lean_geometry_term=RouteGeometry.geometry
lean_real_unit_term=RouteGeometry.fixture_real_unit
lean_real_unit_iff_standard=RouteGeometry.fixture_real_unit_iff_standard
lean_real_emb_term=RouteGeometry.route_real_emb_explicit
lean_real_unit_edges_term=RouteGeometry.route_real_unit_edges_explicit
EOF

cat > "$WORK/PlainSat.lean" <<'EOF'
import SounioSatColouringBridge

def plain_edges : List (Nat × Nat) := []
def plain_cnf : Std.Sat.CNF Nat := SounioSatColouring.colourCNF 4 5 plain_edges
theorem plain_unsat : plain_cnf.Unsat := by
  intro _h
  contradiction
EOF

cat > "$WORK/plain_sat.env" <<'EOF'
candidate_manifest_version=1
promotable=0
candidate_id=route_fixture
n=4
m=4
k=5
edge_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
sat_proof_route=plain_lrat
lean_sat_module_path=PlainSat.lean
EOF

python3 "$GEN_JOIN" "$WORK/route_geometry.env" "$WORK/plain_sat.env" "$WORK/plain_join.lean" \
  --module SounioChi6PlainJoin > "$WORK/plain_join.out"
rg -q '^sat_proof_route=plain_lrat$' "$WORK/plain_join.out"
rg -q 'geometry\.noFiveWitnessOfColourCNFUnsat plainUnsatOnGeometry' \
  "$WORK/plain_join.lean"
rg -q 'NatEdgeUnitDistanceCertificate\.ofColourCNFUnsat' "$WORK/plain_join.lean"
rg -q 'RouteGeometry\.route_real_emb_explicit' "$WORK/plain_join.lean"
rg -q 'RouteGeometry\.route_real_unit_edges_explicit' "$WORK/plain_join.lean"
if rg -q 'noFiveWitnessOfCubeCoverUnsat|colourCNFWithUnit' "$WORK/plain_join.lean"; then
  echo "error: plain_lrat join emitted cube-cover route surface" >&2
  exit 1
fi

cat > "$WORK/TriangleSat.lean" <<'EOF'
import SounioSatColouringSB

def tri_edges : List (Nat × Nat) := [(0, 1), (0, 2), (1, 2), (2, 3)]
def tri_cnf : Std.Sat.CNF Nat := colourCNFsb5 0 1 2 4 tri_edges
theorem tri_unsat : tri_cnf.Unsat := by
  intro _h
  contradiction
EOF

cat > "$WORK/triangle_sat.env" <<'EOF'
candidate_manifest_version=1
promotable=0
candidate_id=route_fixture
n=4
m=4
k=5
edge_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
sat_proof_route=triangle_sb5_lrat
triangle_sb=0,1,2
lean_sat_module_path=TriangleSat.lean
EOF

python3 "$GEN_JOIN" "$WORK/route_geometry.env" "$WORK/triangle_sat.env" "$WORK/triangle_join.lean" \
  --module SounioChi6TriangleJoin > "$WORK/triangle_join.out"
rg -q '^sat_proof_route=triangle_sb5_lrat$' "$WORK/triangle_join.out"
rg -q '^sat_triangle_sb=0,1,2$' "$WORK/triangle_join.out"
rg -q '^sat_sb5_cnf_term=tri_cnf$' "$WORK/triangle_join.out"
rg -q '^sat_sb5_unsat_term=tri_unsat$' "$WORK/triangle_join.out"
rg -q 'geometry\.noFiveWitnessOfColourCNFsb5UnsatTri' "$WORK/triangle_join.lean"
rg -q 'NatEdgeUnitDistanceCertificate\.noFiveWitnessOfColourCNFsb5UnsatTri' \
  "$WORK/triangle_join.lean"
rg -q 'triangleUnsatOnGeometry' "$WORK/triangle_join.lean"
rg -q 'RouteGeometry\.route_real_emb_explicit' "$WORK/triangle_join.lean"
rg -q 'RouteGeometry\.route_real_unit_edges_explicit' "$WORK/triangle_join.lean"
if rg -q 'noFiveWitnessOfCubeCoverUnsat|colourCNFWithUnit|ofColourCNFUnsat' \
    "$WORK/triangle_join.lean"; then
  echo "error: triangle_sb5_lrat join emitted plain/cube-cover route surface" >&2
  exit 1
fi

cp "$WORK/triangle_sat.env" "$WORK/triangle_sat_mismatch.env"
sed -i 's/^triangle_sb=.*/triangle_sb=0,1,3/' "$WORK/triangle_sat_mismatch.env"
if python3 "$GEN_JOIN" "$WORK/route_geometry.env" "$WORK/triangle_sat_mismatch.env" \
    "$WORK/triangle_mismatch_join.lean" --module SounioChi6TriangleMismatch \
    > "$WORK/triangle_mismatch_join.out" 2>&1; then
  echo "error: triangle_sb5_lrat join accepted manifest/Lean triangle mismatch" >&2
  exit 1
fi
rg -q 'sat manifest triangle_sb does not match Lean SAT module' \
  "$WORK/triangle_mismatch_join.out"

cp "$WORK/PlainSat.lean" "$WORK/PlainSatBadSB.lean"
printf '\n-- bad route marker\n#check SounioSatColouringSB.colourCNFsb5\n' \
  >> "$WORK/PlainSatBadSB.lean"
cp "$WORK/plain_sat.env" "$WORK/plain_sat_bad_sb.env"
sed -i 's/^lean_sat_module_path=.*/lean_sat_module_path=PlainSatBadSB.lean/' \
  "$WORK/plain_sat_bad_sb.env"
if python3 "$GEN_JOIN" "$WORK/route_geometry.env" "$WORK/plain_sat_bad_sb.env" \
    "$WORK/plain_bad_sb_join.lean" --module SounioChi6PlainBadSB \
    > "$WORK/plain_bad_sb_join.out" 2>&1; then
  echo "error: plain_lrat join accepted SB5 SAT module" >&2
  exit 1
fi
rg -q 'plain_lrat route must use plain colourCNF only' "$WORK/plain_bad_sb_join.out"

cat > "$WORK/SplitSat.lean" <<'EOF'
import SounioSatCubeCover

def split_edges : List (Nat × Nat) := []
theorem split_v2_c0_unsat :
    (SounioSatCubeCover.colourCNFWithUnit 4 5 split_edges 2 0).Unsat := by
  intro _h
  contradiction
theorem split_v2_c1_unsat :
    (SounioSatCubeCover.colourCNFWithUnit 4 5 split_edges 2 1).Unsat := by
  intro _h
  contradiction
theorem split_v2_c2_unsat :
    (SounioSatCubeCover.colourCNFWithUnit 4 5 split_edges 2 2).Unsat := by
  intro _h
  contradiction
theorem split_v2_c3_unsat :
    (SounioSatCubeCover.colourCNFWithUnit 4 5 split_edges 2 3).Unsat := by
  intro _h
  contradiction
theorem split_v2_c4_unsat :
    (SounioSatCubeCover.colourCNFWithUnit 4 5 split_edges 2 4).Unsat := by
  intro _h
  contradiction
theorem split_unsat_from_v2_split :
    (SounioSatColouring.colourCNF 4 5 split_edges).Unsat :=
  SounioSatCubeCover.unsat_of_split_vertex5
    (n := 4) (edges := split_edges) (v := 2)
    (by decide)
    split_v2_c0_unsat split_v2_c1_unsat split_v2_c2_unsat
    split_v2_c3_unsat split_v2_c4_unsat
EOF

cat > "$WORK/split_sat.env" <<'EOF'
candidate_manifest_version=1
promotable=0
candidate_id=route_fixture
n=4
m=4
k=5
edge_sha256=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
sat_proof_route=cube_cover_split5
lean_sat_module_path=SplitSat.lean
EOF

python3 "$GEN_JOIN" "$WORK/route_geometry.env" "$WORK/split_sat.env" "$WORK/split_join.lean" \
  --module SounioChi6SplitJoin > "$WORK/split_join.out"
rg -q '^sat_proof_route=cube_cover_split5$' "$WORK/split_join.out"
rg -q '^sat_split_vertex=2$' "$WORK/split_join.out"
rg -q 'geometry\.noFiveWitnessOfSplitVertex5Unsat' "$WORK/split_join.lean"
rg -q 'NatEdgeUnitDistanceCertificate\.noFiveWitnessOfSplitVertex5Unsat' \
  "$WORK/split_join.lean"
rg -q 'splitUnsat0OnGeometry' "$WORK/split_join.lean"
rg -q 'splitUnsat4OnGeometry' "$WORK/split_join.lean"
if rg -q 'noFiveWitnessOfCubeCoverUnsat|cubeCoverOnGeometry|cubeUnsatOnGeometry|unsat_of_cube_cover' \
    "$WORK/split_join.lean"; then
  echo "error: cube_cover_split5 join emitted generic cube-cover route surface" >&2
  exit 1
fi

sed '/split_v2_c4_unsat/,+3d' "$WORK/SplitSat.lean" > "$WORK/SplitSatMissing.lean"
cp "$WORK/split_sat.env" "$WORK/split_sat_missing.env"
sed -i 's/^lean_sat_module_path=.*/lean_sat_module_path=SplitSatMissing.lean/' \
  "$WORK/split_sat_missing.env"
if python3 "$GEN_JOIN" "$WORK/route_geometry.env" "$WORK/split_sat_missing.env" \
    "$WORK/split_missing_join.lean" --module SounioChi6SplitMissing \
    > "$WORK/split_missing_join.out" 2>&1; then
  echo "error: cube_cover_split5 join accepted missing leaf colour" >&2
  exit 1
fi
rg -q 'cube_cover_split5 route expected five leaf UNSAT theorems' \
  "$WORK/split_missing_join.out"

READY_FAKE="$WORK/fake_ready_missing_sat.out"
cp "$WORK/preflight_square.out" "$READY_FAKE"
sed -i 's/^sat_status=.*/sat_status=PASS/' "$READY_FAKE"
sed -i 's/^integrated_status=.*/integrated_status=READY_FOR_CANDIDATE_PROMOTION_WIRING/' \
  "$READY_FAKE"
sed -i 's/^first_blocker=.*/first_blocker=none/' "$READY_FAKE"
sed -i 's|^sat_manifest=.*|sat_manifest=NONE|' "$READY_FAKE"
if WORK="$WORK/assembler_fake_ready" "$ASSEMBLER" "$READY_FAKE" \
    > "$WORK/assembler_fake_ready.out" 2>&1; then
  echo "error: assembler accepted fake READY preflight without sat_manifest" >&2
  exit 1
fi
rg -q 'promotion-ready preflight lacks concrete sat_manifest' "$WORK/assembler_fake_ready.out"

echo "chi6_promotable_candidate_assembler_gate: PASS"
