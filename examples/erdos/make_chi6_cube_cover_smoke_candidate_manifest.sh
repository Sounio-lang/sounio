#!/usr/bin/env bash
# Produce a concrete, non-promotable chi>=6-lane manifest smoke through the
# cube-cover SAT route.
#
# This is the K6/k=5 calibration path:
#   cube refutation batch -> cover certificate -> generated Lean leaf checks ->
#   SounioSatCubeCover.unsat_of_split_vertex5 / unsat_of_cube_cover.
#
# The five cubes split one vertex over the five colours; they cover every
# satisfying assignment of the base CNF because `colourCNF` contains the
# at-least-one colour clause for that vertex. They are not meant to enumerate all
# graph vertices.
#
# It proves only a finite `colourCNF` smoke in Lean after the generated module
# attaches the leaf LRAT checks. It is not a Euclidean chi>=6 witness and
# intentionally emits promotable=0 / geometry_proof_type=none.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK="${WORK:-$(mktemp -d)}"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi
LOCK="$ROOT/scripts/dev/souc-build-lock.sh"
REFUTER="$ROOT/examples/erdos/cube_sieve_refute_batch.py"
COVER="$ROOT/examples/erdos/cube_cover_certificate.py"
COMP_CNF="$ROOT/examples/erdos/cube_cover_complement_cnf.py"
CONVERTER="$ROOT/examples/erdos/drup_to_lrat_rup.py"
GEN="$ROOT/examples/erdos/gen_lean_cube_cover_reflect.py"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"
ROUTE="${CUBE_COVER_ROUTE:-cube_cover_split5}"
case "$ROUTE" in
  cube_cover_split5)
    COMPOSITION=split5
    MANIFEST_ROUTE=cube_cover_split5
    CANDIDATE_ID=k6_cube_cover_smoke_not_planar
    ;;
  cube_cover_generic)
    COMPOSITION=generic
    MANIFEST_ROUTE=cube_cover_generic
    CANDIDATE_ID=k6_cube_cover_generic_smoke_not_planar
    ;;
  cube_cover_arbitrary_complement)
    COMPOSITION=arbitrary
    MANIFEST_ROUTE=cube_cover_generic
    CANDIDATE_ID=k6_cube_cover_arbitrary_complement_smoke_not_planar
    ;;
  *)
    echo "error: CUBE_COVER_ROUTE must be cube_cover_split5, cube_cover_generic, or cube_cover_arbitrary_complement" >&2
    exit 2
    ;;
esac

[[ -n "$LAKE" ]] || { echo "error: lake not found; set LAKE=/path/to/lake" >&2; exit 127; }
[[ -x "$LOCK" ]] || { echo "error: missing build lock helper: $LOCK" >&2; exit 1; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$REFUTER" "$COVER" "$COMP_CNF" "$CONVERTER" "$GEN"

mkdir -p "$WORK/refute"
EDGE="$WORK/k6.edge"
CUBES="$WORK/k6_v0_cover.cubes"
REFUTE_OUT="$WORK/cube_refute.out"
COVER_OUT="$WORK/cube_cover.out"
COVER_COMP_CNF="$WORK/cube_cover_complement.cnf"
COVER_COMP_LRAT="$WORK/cube_cover_complement.lrat"
OUT_LEAN="$WORK/SounioSatK65CubeCoverReflect.lean"
MANIFEST="$WORK/candidate.manifest"

cat > "$EDGE" <<'EOF'
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

if [[ "$COMPOSITION" == "arbitrary" ]]; then
  for c0 in 0 1 2 3 4; do
    for c1 in 0 1 2 3 4; do
      printf 'v0_c%s_v1_c%s: 0:%s 1:%s\n' "$c0" "$c1" "$c0" "$c1"
    done
  done > "$CUBES"
else
  cat > "$CUBES" <<'EOF'
v0_c0: 0:0
v0_c1: 0:1
v0_c2: 0:2
v0_c3: 0:3
v0_c4: 0:4
EOF
fi

echo "chi6_cube_cover_smoke_manifest: workdir=$WORK"
python3 "$REFUTER" "$EDGE" 5 "$CUBES" "$WORK/refute" > "$REFUTE_OUT"
rg -q '^formula_kind=colourCNF$' "$REFUTE_OUT"
if [[ "$COMPOSITION" == "arbitrary" ]]; then
  rg -q '^cube_count=25$' "$REFUTE_OUT"
  rg -q '^solver_unsat_count=25$' "$REFUTE_OUT"
  rg -q '^lrat_artifact_count=25$' "$REFUTE_OUT"
else
  rg -q '^cube_count=5$' "$REFUTE_OUT"
  rg -q '^solver_unsat_count=5$' "$REFUTE_OUT"
  rg -q '^lrat_artifact_count=5$' "$REFUTE_OUT"
fi
rg -q '^formal_proof_checker=none$' "$REFUTE_OUT"
rg -q '^global_unsat_claim=none$' "$REFUTE_OUT"
rg -q '^status=subproblem_lrat_artifacts_emitted_unpromotable$' "$REFUTE_OUT"
GEN_ARGS=(
  "$EDGE" 5 "$CUBES" "$REFUTE_OUT" "$OUT_LEAN"
  --module SounioSatK65CubeCoverReflect
  --prefix k65cube
  --composition "$COMPOSITION"
)
if [[ "$COMPOSITION" == "arbitrary" ]]; then
  python3 "$COMP_CNF" "$EDGE" 5 "$CUBES" "$COVER_COMP_CNF" > "$WORK/cube_cover_complement_cnf.out"
  rg -q '^cube_cover_complement_cnf v1$' "$WORK/cube_cover_complement_cnf.out"
  rg -q '^cube_count=25$' "$WORK/cube_cover_complement_cnf.out"
  rg -q '^clause_count=106$' "$WORK/cube_cover_complement_cnf.out"
  cat > "$WORK/cube_cover_complement.drup" <<'EOF'
-1 0
-2 0
-3 0
-4 0
-5 0
0
EOF
  python3 "$CONVERTER" "$COVER_COMP_CNF" "$WORK/cube_cover_complement.drup" \
    "$COVER_COMP_LRAT" > "$WORK/cube_cover_complement_lrat.out" 2> "$WORK/cube_cover_complement_lrat.err"
  rg -q 'empty=1' "$WORK/cube_cover_complement_lrat.err"
  GEN_ARGS+=(--cover-cnf "$COVER_COMP_CNF" --cover-lrat "$COVER_COMP_LRAT")
else
  python3 "$COVER" "$EDGE" 5 "$CUBES" "$REFUTE_OUT" > "$COVER_OUT"
  rg -q '^cover_rule=single_vertex_atleast_one_split$' "$COVER_OUT"
  rg -q '^cover_complete_for_split_vertex=1$' "$COVER_OUT"
  rg -q '^cover_claim=atleast_one_cover_for_split_vertex$' "$COVER_OUT"
  rg -q '^global_unsat_claim=none$' "$COVER_OUT"
fi
python3 "$GEN" "${GEN_ARGS[@]}" > "$WORK/gen_lean_cube_cover_reflect.out"

(
  cd "$ROOT/formal/lean4"
  "$LOCK" "$LAKE" env lean "$OUT_LEAN" > "$WORK/lean_build.log" 2>&1
)

if rg -q '\b(sorry|admit)\b|#exit' "$OUT_LEAN"; then
  echo "error: generated Lean cube-cover module contains sorry/admit/#exit" >&2
  exit 1
fi
if rg -q '#eval|#check' "$OUT_LEAN"; then
  echo "error: generated Lean cube-cover module contains #eval/#check" >&2
  exit 1
fi
if [[ "$ROUTE" == "cube_cover_generic" ]]; then
  rg -q '^theorem k65cube_unsat_from_generic_cube_cover' "$OUT_LEAN" || {
    echo "error: generated Lean cube-cover module lacks generic composed UNSAT theorem" >&2
    exit 1
  }
  rg -q 'SounioSatCubeCover.unsat_of_cube_cover' "$OUT_LEAN" || {
    echo "error: generated Lean cube-cover module lacks generic composition theorem" >&2
    exit 1
  }
elif [[ "$ROUTE" == "cube_cover_arbitrary_complement" ]]; then
  rg -q '^theorem k65cube_unsat_from_arbitrary_cube_cover' "$OUT_LEAN" || {
    echo "error: generated Lean cube-cover module lacks arbitrary composed UNSAT theorem" >&2
    exit 1
  }
  rg -q 'SounioSatCubeCover.cube_cover_of_complement_unsat' "$OUT_LEAN" || {
    echo "error: generated Lean cube-cover module lacks complement-cover proof theorem" >&2
    exit 1
  }
  rg -q 'SounioSatCubeCover.unsat_of_cube_cover' "$OUT_LEAN" || {
    echo "error: generated Lean cube-cover module lacks generic composition theorem" >&2
    exit 1
  }
else
  rg -q '^theorem k65cube_unsat_from_v0_split' "$OUT_LEAN" || {
    echo "error: generated Lean cube-cover module lacks split composed UNSAT theorem" >&2
    exit 1
  }
fi
rg -q '^    \(colourCNF 6 5 k65cube_edges\)\.Unsat :=$' "$OUT_LEAN" || {
  echo "error: generated Lean cube-cover theorem has unexpected statement" >&2
  exit 1
}
rg -q "'k65cube_unsat_from_(v0_split|generic_cube_cover|arbitrary_cube_cover)' depends on axioms:" "$WORK/lean_build.log" || {
  echo "error: Lean run did not print composed UNSAT axiom surface" >&2
  exit 1
}
if rg -q 'sorryAx' "$WORK/lean_build.log"; then
  echo "error: generated Lean cube-cover module depends on sorryAx" >&2
  exit 1
fi

sha() {
  [[ -f "$1" ]] || { echo "error: file not found for hashing: $1" >&2; exit 1; }
  sha256sum "$1" | awk '{print $1}'
}

GENERATOR_COMMIT="$(git -C "$ROOT" rev-parse --verify HEAD 2>/dev/null || echo UNKNOWN)"

cat > "$MANIFEST" <<EOF
candidate_manifest_version=1
promotable=0
candidate_id=$CANDIDATE_ID
n=6
m=15
k=5
edge_path=k6.edge
edge_sha256=$(sha "$EDGE")
cnf_path=NONE
cnf_sha256=NONE
drat_or_lrat_path=NONE
drat_or_lrat_sha256=NONE
lean_sat_module_path=SounioSatK65CubeCoverReflect.lean
lean_sat_module_sha256=$(sha "$OUT_LEAN")
geometry_module_path=NONE
geometry_module_sha256=NONE
geometry_proof_type=none
sat_proof_route=$MANIFEST_ROUTE
triangle_sb=none
generator_commit=$GENERATOR_COMMIT
producer_command=examples/erdos/cube_sieve_refute_batch.py k6.edge 5 k6_v0_cover.cubes refute && examples/erdos/gen_lean_cube_cover_reflect.py k6.edge 5 k6_v0_cover.cubes cube_refute.out SounioSatK65CubeCoverReflect.lean
lean_build_command=lake env lean SounioSatK65CubeCoverReflect.lean
offload_review_raw=NONE
offload_review_sha256=NONE
cube_batch_path=k6_v0_cover.cubes
cube_batch_sha256=$(sha "$CUBES")
cube_refutation_summary_path=cube_refute.out
cube_refutation_summary_sha256=$(sha "$REFUTE_OUT")
EOF

if [[ "$COMPOSITION" == "arbitrary" ]]; then
  cat >> "$MANIFEST" <<EOF
cube_cover_certificate_path=NONE
cube_cover_certificate_sha256=NONE
cube_cover_complement_cnf_path=cube_cover_complement.cnf
cube_cover_complement_cnf_sha256=$(sha "$COVER_COMP_CNF")
cube_cover_complement_lrat_path=cube_cover_complement.lrat
cube_cover_complement_lrat_sha256=$(sha "$COVER_COMP_LRAT")
EOF
else
  cat >> "$MANIFEST" <<EOF
cube_cover_certificate_path=cube_cover.out
cube_cover_certificate_sha256=$(sha "$COVER_OUT")
cube_cover_complement_cnf_path=NONE
cube_cover_complement_cnf_sha256=NONE
cube_cover_complement_lrat_path=NONE
cube_cover_complement_lrat_sha256=NONE
EOF
fi

"$VALIDATOR" "$MANIFEST" | tee "$WORK/manifest_validator.log"

echo "manifest=$MANIFEST"
echo "edge=$EDGE"
echo "cube_batch=$CUBES"
echo "cube_refutation_summary=$REFUTE_OUT"
if [[ "$COMPOSITION" == "arbitrary" ]]; then
  echo "cube_cover_complement_cnf=$COVER_COMP_CNF"
  echo "cube_cover_complement_lrat=$COVER_COMP_LRAT"
else
  echo "cube_cover_certificate=$COVER_OUT"
fi
echo "lean_sat_module=$OUT_LEAN"
echo "cube_cover_route=$ROUTE"
echo "chi6_cube_cover_smoke_manifest: PASS"
