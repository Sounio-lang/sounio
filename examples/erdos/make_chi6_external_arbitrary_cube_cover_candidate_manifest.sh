#!/usr/bin/env bash
# Package an external DIMACS graph plus arbitrary cube-cover proof into candidate.manifest.
#
# This is the reusable bridge for the generic complement-cover SAT lane:
#
#   external edge file + arbitrary cube batch + complement DRUP/RUP proof
#     -> per-cube LRAT leaves
#     -> complement-cover CNF/LRAT
#     -> generated Lean SAT module using cube_cover_of_complement_unsat
#     -> non-promotable candidate.manifest
#
# It deliberately emits `promotable=0`. A real chi(R^2)>=6 promotion still needs
# candidate-owned exact Euclidean geometry, Real-plane bridge fields, and the
# promotable Lean gate.
set -euo pipefail

usage() {
  echo "usage: $0 <edge-file> <candidate-id> <cube-batch> <cover-drup>" >&2
  echo "example: WORK=/tmp/chi6-arb $0 graph.edge candidate cubes.txt cover.drup" >&2
}

if [[ $# -ne 4 ]]; then
  usage
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EDGE_IN="$1"
CANDIDATE_ID="$2"
CUBE_IN="$3"
COVER_DRUP_IN="$4"
WORK="${WORK:-$(mktemp -d)}"
K=5

REFUTER="$ROOT/examples/erdos/cube_sieve_refute_batch.py"
COMP_CNF="$ROOT/examples/erdos/cube_cover_complement_cnf.py"
CONVERTER="$ROOT/examples/erdos/drup_to_lrat_rup.py"
GEN="$ROOT/examples/erdos/gen_lean_cube_cover_reflect.py"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"
LOCK="$ROOT/scripts/dev/souc-build-lock.sh"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi

[[ -s "$EDGE_IN" ]] || { echo "error: missing/empty edge file: $EDGE_IN" >&2; exit 2; }
[[ -s "$CUBE_IN" ]] || { echo "error: missing/empty cube batch: $CUBE_IN" >&2; exit 2; }
[[ -s "$COVER_DRUP_IN" ]] || { echo "error: missing/empty cover DRUP/RUP proof: $COVER_DRUP_IN" >&2; exit 2; }
[[ "$CANDIDATE_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "error: candidate-id must use only letters, digits, '.', '_', or '-'" >&2
  exit 2
}
[[ -n "$LAKE" ]] || { echo "error: lake not found; set LAKE=/path/to/lake" >&2; exit 127; }
[[ -x "$LOCK" ]] || { echo "error: missing build lock helper: $LOCK" >&2; exit 1; }
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }

python3 -m py_compile "$REFUTER" "$COMP_CNF" "$CONVERTER" "$GEN"

mkdir -p "$WORK/package" "$WORK/refute"
PKG_EDGE="$WORK/package/$CANDIDATE_ID.edge"
PKG_CUBES="$WORK/package/$CANDIDATE_ID.cubes"
cp "$EDGE_IN" "$PKG_EDGE"
cp "$CUBE_IN" "$PKG_CUBES"

REFUTE_OUT="$WORK/cube_refute.out"
COVER_CNF="$WORK/cover_complement.cnf"
COVER_LRAT="$WORK/cover_complement.lrat"
OUT_LEAN="$WORK/SounioSatChi6ExternalArbitraryCoverReflect.lean"
MANIFEST="$WORK/candidate.manifest"

echo "chi6_external_arbitrary_cube_cover_candidate: workdir=$WORK"
python3 "$REFUTER" "$PKG_EDGE" "$K" "$PKG_CUBES" "$WORK/refute" > "$REFUTE_OUT"
rg -q '^formula_kind=colourCNF$' "$REFUTE_OUT"
rg -q '^failed_count=0$' "$REFUTE_OUT"
rg -q '^formal_proof_checker=none$' "$REFUTE_OUT"
rg -q '^global_unsat_claim=none$' "$REFUTE_OUT"
rg -q '^status=subproblem_lrat_artifacts_emitted_unpromotable$' "$REFUTE_OUT"

python3 "$COMP_CNF" "$PKG_EDGE" "$K" "$PKG_CUBES" "$COVER_CNF" > "$WORK/cover_complement.out"
rg -q '^claim=base_plus_cube_blockers_dimacs_only$' "$WORK/cover_complement.out"
rg -q '^status=complement_cnf_emitted$' "$WORK/cover_complement.out"

python3 "$CONVERTER" "$COVER_CNF" "$COVER_DRUP_IN" "$COVER_LRAT" \
  > "$WORK/cover_lrat.out" 2> "$WORK/cover_lrat.err"
[[ -s "$COVER_LRAT" ]] || { echo "error: converter emitted empty complement LRAT" >&2; exit 1; }
rg -q 'empty=1' "$WORK/cover_lrat.err"
cover_clause_count="$(awk -F= '$1 == "clause_count" {print $2; exit}' "$WORK/cover_complement.out")"
converter_original_count="$(rg -o 'original=[0-9]+' "$WORK/cover_lrat.err" | sed -n '1s/original=//p')"
if [[ -z "$cover_clause_count" || "$cover_clause_count" != "$converter_original_count" ]]; then
  echo "error: complement LRAT conversion did not read the emitted complement CNF clause count" >&2
  echo "cover_clause_count=${cover_clause_count:-missing} converter_original_count=${converter_original_count:-missing}" >&2
  exit 1
fi

# The converted LRAT is still not trusted here. The generated Lean module below
# replays it against SounioSatCubeCover.cubeCoverComplementCNF before the
# manifest is emitted.
python3 "$GEN" "$PKG_EDGE" "$K" "$PKG_CUBES" "$REFUTE_OUT" "$OUT_LEAN" \
  --module SounioSatChi6ExternalArbitraryCoverReflect \
  --prefix chi6arb \
  --composition arbitrary \
  --cover-cnf "$COVER_CNF" \
  --cover-lrat "$COVER_LRAT" \
  > "$WORK/gen_lean_cube_cover_reflect.out"
rg -q '^composition=arbitrary$' "$WORK/gen_lean_cube_cover_reflect.out"
rg -q '^cover_claim=base_plus_cube_blockers_unsat$' "$WORK/gen_lean_cube_cover_reflect.out"

rg -q '^theorem chi6arb_unsat_from_arbitrary_cube_cover' "$OUT_LEAN" || {
  echo "error: generated Lean lacks arbitrary cube-cover theorem" >&2
  exit 1
}
rg -q 'SounioSatCubeCover.cubeCoverComplementCNF' "$OUT_LEAN" || {
  echo "error: generated Lean lacks complement CNF term" >&2
  exit 1
}
rg -q '^theorem chi6arb_cover_complement_check :' "$OUT_LEAN" || {
  echo "error: generated Lean lacks concrete complement LRAT check theorem" >&2
  exit 1
}
rg -q 'Std\.Tactic\.BVDecide\.LRAT\.check_sound _ \(SounioSatCubeCover\.cubeCoverComplementCNF' "$OUT_LEAN" || {
  echo "error: generated Lean does not replay complement LRAT against cubeCoverComplementCNF" >&2
  exit 1
}
rg -q 'SounioSatCubeCover.cube_cover_of_complement_unsat' "$OUT_LEAN" || {
  echo "error: generated Lean lacks complement-cover theorem" >&2
  exit 1
}
rg -q '^theorem chi6arb_cube_cover :' "$OUT_LEAN" || {
  echo "error: generated Lean lacks concrete CubeCover theorem" >&2
  exit 1
}
rg -q 'SounioSatCubeCover.unsat_of_cube_cover' "$OUT_LEAN" || {
  echo "error: generated Lean lacks generic cube-cover composition" >&2
  exit 1
}
rg -q 'Std.Tactic.BVDecide.LRAT.check' "$OUT_LEAN" || {
  echo "error: generated Lean lacks LRAT checker calls" >&2
  exit 1
}
if rg -q 'EuclideanNatEdgeExactGeometry|ExactSquaredDistancePlane|split_vertices_cubes_cover|splitVerticesCubes|unsat_of_split_vertex5|colourCNFWithUnit' "$OUT_LEAN"; then
  echo "error: arbitrary-cover Lean used geometry or split-cover-only surface" >&2
  exit 1
fi
if rg -q '\b(sorry|admit)\b|#exit|#eval|#check' "$OUT_LEAN"; then
  echo "error: generated Lean contains forbidden proof/debug marker" >&2
  exit 1
fi

(
  cd "$ROOT/formal/lean4"
  "$LOCK" "$LAKE" env lean "$OUT_LEAN" > "$WORK/lean_build.log" 2>&1
)
if rg -q 'error:|sorryAx|warning:|deprecated:' "$WORK/lean_build.log"; then
  cat "$WORK/lean_build.log" >&2
  echo "error: generated Lean reported errors, warnings, or sorryAx" >&2
  exit 1
fi

sha() {
  [[ -f "$1" ]] || { echo "error: file not found for hashing: $1" >&2; exit 1; }
  sha256sum "$1" | awk '{print $1}'
}

read -r N M < <(awk '$1 == "p" && $2 == "edge" {print $3, $4; exit}' "$PKG_EDGE")
[[ "$N" =~ ^[1-9][0-9]*$ && "$M" =~ ^[1-9][0-9]*$ ]] || {
  echo "error: packaged edge has invalid p edge header" >&2
  exit 1
}
GENERATOR_COMMIT="$(git -C "$ROOT" rev-parse --verify HEAD 2>/dev/null)" || {
  echo "error: unable to resolve generator git commit" >&2
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
lean_sat_module_path=SounioSatChi6ExternalArbitraryCoverReflect.lean
lean_sat_module_sha256=$(sha "$OUT_LEAN")
geometry_module_path=NONE
geometry_module_sha256=NONE
geometry_proof_type=none
sat_proof_route=cube_cover_generic
triangle_sb=none
generator_commit=$GENERATOR_COMMIT
producer_command=WORK=$WORK LAKE=$LAKE examples/erdos/make_chi6_external_arbitrary_cube_cover_candidate_manifest.sh $EDGE_IN $CANDIDATE_ID $CUBE_IN $COVER_DRUP_IN
lean_build_command=lake env lean SounioSatChi6ExternalArbitraryCoverReflect.lean
offload_review_raw=NONE
offload_review_sha256=NONE
cube_batch_path=package/$CANDIDATE_ID.cubes
cube_batch_sha256=$(sha "$PKG_CUBES")
cube_refutation_summary_path=cube_refute.out
cube_refutation_summary_sha256=$(sha "$REFUTE_OUT")
cube_cover_certificate_path=NONE
cube_cover_certificate_sha256=NONE
cube_cover_complement_cnf_path=cover_complement.cnf
cube_cover_complement_cnf_sha256=$(sha "$COVER_CNF")
cube_cover_complement_lrat_path=cover_complement.lrat
cube_cover_complement_lrat_sha256=$(sha "$COVER_LRAT")
chromatic_claim=none
geometry_claim=none
EOF

"$VALIDATOR" "$MANIFEST" | tee "$WORK/manifest_validator.log"

echo "manifest=$MANIFEST"
echo "edge=$PKG_EDGE"
echo "cube_batch=$PKG_CUBES"
echo "cube_refutation_summary=$REFUTE_OUT"
echo "cube_cover_complement_cnf=$COVER_CNF"
echo "cube_cover_complement_lrat=$COVER_LRAT"
echo "lean_sat_module=$OUT_LEAN"
echo "chi6_external_arbitrary_cube_cover_candidate: PASS"
