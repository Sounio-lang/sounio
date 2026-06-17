#!/usr/bin/env bash
# Package an external DIMACS candidate through the generic cube-cover SAT lane.
#
# This is the handoff bridge from "candidate edge file" to the existing
# candidate.manifest format:
#
#   external DIMACS package -> split-product cubes -> per-cube LRAT artifacts ->
#   checked cube-cover certificate -> generated Lean SAT module -> manifest
#
# It is deliberately non-promotable. A real chi(R^2)>=6 promotion still needs a
# candidate-owned exact Euclidean geometry module, Real-plane bridge fields, and
# the promotable Lean gate.
set -euo pipefail

usage() {
  echo "usage: $0 <edge-file> <candidate-id> <split-vertices>" >&2
  echo "example: WORK=/tmp/k6 $0 k6.edge k6_external 0" >&2
}

if [[ $# -ne 3 ]]; then
  usage
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EDGE_IN="$1"
CANDIDATE_ID="$2"
SPLIT_VERTICES="$3"
WORK="${WORK:-$(mktemp -d)}"
K=5

SEARCH="$ROOT/examples/erdos/chi6_candidate_search_manifest.py"
REFUTER="$ROOT/examples/erdos/cube_sieve_refute_batch.py"
COVER="$ROOT/examples/erdos/cube_cover_certificate.py"
GEN="$ROOT/examples/erdos/gen_lean_cube_cover_reflect.py"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"
LOCK="$ROOT/scripts/dev/souc-build-lock.sh"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi

[[ -s "$EDGE_IN" ]] || { echo "error: missing/empty edge file: $EDGE_IN" >&2; exit 2; }
[[ "$CANDIDATE_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "error: candidate-id must use only letters, digits, '.', '_', or '-'" >&2
  exit 2
}
[[ -n "$SPLIT_VERTICES" ]] || { echo "error: split-vertices cannot be empty" >&2; exit 2; }
[[ -n "$LAKE" ]] || { echo "error: lake not found; set LAKE=/path/to/lake" >&2; exit 127; }
[[ -x "$LOCK" ]] || { echo "error: missing build lock helper: $LOCK" >&2; exit 1; }
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }

python3 -m py_compile "$SEARCH" "$REFUTER" "$COVER" "$GEN"

mkdir -p "$WORK/package" "$WORK/refute"
PACKAGE_OUT="$WORK/package.out"
REFUTE_OUT="$WORK/cube_refute.out"
COVER_OUT="$WORK/cube_cover.out"
OUT_LEAN="$WORK/SounioSatChi6ExternalCubeCoverReflect.lean"
MANIFEST="$WORK/candidate.manifest"

echo "chi6_external_cube_cover_candidate: workdir=$WORK"
python3 "$SEARCH" "$WORK/package" --edge-file "$EDGE_IN" --k "$K" \
  --candidate-id "$CANDIDATE_ID" --split-vertices "$SPLIT_VERTICES" > "$PACKAGE_OUT"
rg -q '^family=external_dimacs_edge$' "$PACKAGE_OUT"
rg -q '^finite_graph_search_claim=none_external_graph_packaging_only$' "$PACKAGE_OUT"
rg -q '^status=EXTERNAL_GRAPH_PACKAGED_UNPROMOTABLE$' "$PACKAGE_OUT"
rg -q '^candidate index=0 .* not_k_colourable_claim=none geometry_claim=none .* cover_route=split_vertices_atleast_one_product$' \
  "$PACKAGE_OUT"

PKG_EDGE="$WORK/package/$CANDIDATE_ID.edge"
PKG_CUBES="$WORK/package/$CANDIDATE_ID.cubes"
PKG_META="$WORK/package/$CANDIDATE_ID.meta.json"
[[ -s "$PKG_EDGE" ]] || { echo "error: missing packaged edge: $PKG_EDGE" >&2; exit 1; }
[[ -s "$PKG_CUBES" ]] || { echo "error: missing packaged cube batch: $PKG_CUBES" >&2; exit 1; }
[[ -s "$PKG_META" ]] || { echo "error: missing packaged metadata: $PKG_META" >&2; exit 1; }

python3 "$REFUTER" "$PKG_EDGE" "$K" "$PKG_CUBES" "$WORK/refute" > "$REFUTE_OUT"
rg -q '^formula_kind=colourCNF$' "$REFUTE_OUT"
rg -q '^failed_count=0$' "$REFUTE_OUT"
rg -q '^formal_proof_checker=none$' "$REFUTE_OUT"
rg -q '^global_unsat_claim=none$' "$REFUTE_OUT"
rg -q '^status=subproblem_lrat_artifacts_emitted_unpromotable$' "$REFUTE_OUT"

python3 "$COVER" "$PKG_EDGE" "$K" "$PKG_CUBES" "$REFUTE_OUT" \
  --cover-rule split_vertices_atleast_one_product \
  --split-vertices "$SPLIT_VERTICES" > "$COVER_OUT"
rg -q '^cover_rule=split_vertices_atleast_one_product$' "$COVER_OUT"
rg -q '^cover_claim=atleast_one_product_cover_for_split_vertices$' "$COVER_OUT"
rg -q '^global_unsat_claim=none$' "$COVER_OUT"

python3 "$GEN" "$PKG_EDGE" "$K" "$PKG_CUBES" "$REFUTE_OUT" "$OUT_LEAN" \
  --module SounioSatChi6ExternalCubeCoverReflect \
  --prefix chi6ext \
  --composition generic > "$WORK/gen_lean_cube_cover_reflect.out"

(
  cd "$ROOT/formal/lean4"
  "$LOCK" "$LAKE" env lean "$OUT_LEAN" > "$WORK/lean_build.log" 2>&1
)

if rg -q '\b(sorry|admit)\b|#exit|#eval|#check' "$OUT_LEAN"; then
  echo "error: generated Lean cube-cover module contains forbidden proof/debug marker" >&2
  exit 1
fi
rg -q '^theorem chi6ext_unsat_from_generic_cube_cover' "$OUT_LEAN" || {
  echo "error: generated Lean cube-cover module lacks generic composed UNSAT theorem" >&2
  exit 1
}
rg -q 'SounioSatCubeCover.unsat_of_cube_cover' "$OUT_LEAN" || {
  echo "error: generated Lean cube-cover module lacks generic composition theorem" >&2
  exit 1
}
if rg -q 'sorryAx' "$WORK/lean_build.log"; then
  echo "error: generated Lean cube-cover module depends on sorryAx" >&2
  exit 1
fi
if rg -q 'error:' "$WORK/lean_build.log"; then
  cat "$WORK/lean_build.log" >&2
  echo "error: generated Lean cube-cover module reported Lean errors" >&2
  exit 1
fi
if rg -q 'warning:|deprecated:' "$WORK/lean_build.log"; then
  cat "$WORK/lean_build.log" >&2
  echo "error: generated Lean cube-cover module reported Lean warnings" >&2
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
lean_sat_module_path=SounioSatChi6ExternalCubeCoverReflect.lean
lean_sat_module_sha256=$(sha "$OUT_LEAN")
geometry_module_path=NONE
geometry_module_sha256=NONE
geometry_proof_type=none
sat_proof_route=cube_cover_generic
triangle_sb=none
generator_commit=$GENERATOR_COMMIT
producer_command=WORK=$WORK LAKE=$LAKE examples/erdos/make_chi6_external_cube_cover_candidate_manifest.sh $EDGE_IN $CANDIDATE_ID $SPLIT_VERTICES
lean_build_command=lake env lean SounioSatChi6ExternalCubeCoverReflect.lean
offload_review_raw=NONE
offload_review_sha256=NONE
source_meta_path=package/$CANDIDATE_ID.meta.json
source_meta_sha256=$(sha "$PKG_META")
cube_batch_path=package/$CANDIDATE_ID.cubes
cube_batch_sha256=$(sha "$PKG_CUBES")
cube_refutation_summary_path=cube_refute.out
cube_refutation_summary_sha256=$(sha "$REFUTE_OUT")
cube_cover_certificate_path=cube_cover.out
cube_cover_certificate_sha256=$(sha "$COVER_OUT")
cube_cover_complement_cnf_path=NONE
cube_cover_complement_cnf_sha256=NONE
cube_cover_complement_lrat_path=NONE
cube_cover_complement_lrat_sha256=NONE
EOF

"$VALIDATOR" "$MANIFEST" | tee "$WORK/manifest_validator.log"

echo "manifest=$MANIFEST"
echo "package_manifest=$PACKAGE_OUT"
echo "edge=$PKG_EDGE"
echo "source_meta=$PKG_META"
echo "cube_batch=$PKG_CUBES"
echo "cube_refutation_summary=$REFUTE_OUT"
echo "cube_cover_certificate=$COVER_OUT"
echo "lean_sat_module=$OUT_LEAN"
echo "chi6_external_cube_cover_candidate: PASS"
