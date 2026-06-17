#!/usr/bin/env bash
# Gate for the candidate-manifest smoke that uses cube-cover Lean composition.
#
# This is intentionally non-promotable: it validates that the manifest can carry
# the cube batch/refutation/cover artifacts and the generated Lean SAT module
# without claiming Euclidean geometry or chi>=6.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
MAKER="$ROOT/examples/erdos/make_chi6_cube_cover_smoke_candidate_manifest.sh"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
mkdir -p "$WORK"

WORK="$WORK" "$MAKER" > "$WORK/maker.out"

MANIFEST="$WORK/candidate.manifest"
[[ -s "$MANIFEST" ]] || { echo "error: maker did not emit candidate.manifest" >&2; exit 1; }
rg -q '^candidate_id=k6_cube_cover_smoke_not_planar$' "$MANIFEST"
rg -q '^promotable=0$' "$MANIFEST"
rg -q '^geometry_proof_type=none$' "$MANIFEST"
rg -q '^sat_proof_route=cube_cover_split5$' "$MANIFEST"
rg -q '^triangle_sb=none$' "$MANIFEST"
rg -q '^lean_sat_module_path=SounioSatK65CubeCoverReflect\.lean$' "$MANIFEST"
rg -q '^cube_batch_path=k6_v0_cover\.cubes$' "$MANIFEST"
rg -q '^cube_refutation_summary_path=cube_refute\.out$' "$MANIFEST"
rg -q '^cube_cover_certificate_path=cube_cover\.out$' "$MANIFEST"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=k6_cube_cover_smoke_not_planar$' \
  "$WORK/maker.out"
rg -q '^chi6_cube_cover_smoke_manifest: PASS$' "$WORK/maker.out"
rg -q '^theorem k65cube_unsat_from_v0_split' "$WORK/SounioSatK65CubeCoverReflect.lean"

"$VALIDATOR" "$MANIFEST" > "$WORK/validator.out"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=k6_cube_cover_smoke_not_planar$' \
  "$WORK/validator.out"

GENERIC_WORK="$WORK/generic"
mkdir -p "$GENERIC_WORK"
CUBE_COVER_ROUTE=cube_cover_generic WORK="$GENERIC_WORK" "$MAKER" > "$GENERIC_WORK/maker.out"
GENERIC_MANIFEST="$GENERIC_WORK/candidate.manifest"
[[ -s "$GENERIC_MANIFEST" ]] || { echo "error: generic maker did not emit candidate.manifest" >&2; exit 1; }
rg -q '^candidate_id=k6_cube_cover_generic_smoke_not_planar$' "$GENERIC_MANIFEST"
rg -q '^promotable=0$' "$GENERIC_MANIFEST"
rg -q '^geometry_proof_type=none$' "$GENERIC_MANIFEST"
rg -q '^sat_proof_route=cube_cover_generic$' "$GENERIC_MANIFEST"
rg -q '^triangle_sb=none$' "$GENERIC_MANIFEST"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=k6_cube_cover_generic_smoke_not_planar$' \
  "$GENERIC_WORK/maker.out"
rg -q '^chi6_cube_cover_smoke_manifest: PASS$' "$GENERIC_WORK/maker.out"
rg -q '^theorem k65cube_unsat_from_generic_cube_cover' \
  "$GENERIC_WORK/SounioSatK65CubeCoverReflect.lean"
rg -q 'SounioSatCubeCover.unsat_of_cube_cover' \
  "$GENERIC_WORK/SounioSatK65CubeCoverReflect.lean"
"$VALIDATOR" "$GENERIC_MANIFEST" > "$GENERIC_WORK/validator.out"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=k6_cube_cover_generic_smoke_not_planar$' \
  "$GENERIC_WORK/validator.out"

cp "$MANIFEST" "$WORK/bad-cube-cover.env"
sed -i 's/^cube_cover_certificate_sha256=.*/cube_cover_certificate_sha256=0000000000000000000000000000000000000000000000000000000000000000/' \
  "$WORK/bad-cube-cover.env"
if "$VALIDATOR" "$WORK/bad-cube-cover.env" > "$WORK/bad-cube-cover.out" 2>&1; then
  echo "error: validator accepted a bad cube-cover certificate hash" >&2
  exit 1
fi
rg -q 'cube_cover_certificate SHA256 mismatch' "$WORK/bad-cube-cover.out"

cp "$MANIFEST" "$WORK/bad-triangle.env"
sed -i 's/^triangle_sb=.*/triangle_sb=0,1,2/' "$WORK/bad-triangle.env"
if "$VALIDATOR" "$WORK/bad-triangle.env" > "$WORK/bad-triangle.out" 2>&1; then
  echo "error: validator accepted triangle_sb on plain cube-cover SAT module" >&2
  exit 1
fi
rg -q 'sat_proof_route=cube_cover_split5 requires triangle_sb=none' "$WORK/bad-triangle.out"

cp "$MANIFEST" "$WORK/bad-route.env"
sed -i 's/^sat_proof_route=.*/sat_proof_route=triangle_sb5_lrat/' "$WORK/bad-route.env"
if "$VALIDATOR" "$WORK/bad-route.env" > "$WORK/bad-route.out" 2>&1; then
  echo "error: validator accepted triangle_sb5 route without triangle_sb metadata" >&2
  exit 1
fi
rg -q 'sat_proof_route=triangle_sb5_lrat requires triangle_sb metadata' "$WORK/bad-route.out"

cp "$MANIFEST" "$WORK/missing-cube-artifact.env"
sed -i 's/^cube_cover_certificate_path=.*/cube_cover_certificate_path=NONE/' \
  "$WORK/missing-cube-artifact.env"
sed -i 's/^cube_cover_certificate_sha256=.*/cube_cover_certificate_sha256=NONE/' \
  "$WORK/missing-cube-artifact.env"
if "$VALIDATOR" "$WORK/missing-cube-artifact.env" > "$WORK/missing-cube-artifact.out" 2>&1; then
  echo "error: validator accepted cube-cover route without cover artifact" >&2
  exit 1
fi
rg -q 'sat_proof_route=cube_cover_split5 requires concrete cube_cover_certificate artifact' \
  "$WORK/missing-cube-artifact.out"

cp "$GENERIC_MANIFEST" "$GENERIC_WORK/bad-triangle.env"
sed -i 's/^triangle_sb=.*/triangle_sb=0,1,2/' "$GENERIC_WORK/bad-triangle.env"
if "$VALIDATOR" "$GENERIC_WORK/bad-triangle.env" > "$GENERIC_WORK/bad-triangle.out" 2>&1; then
  echo "error: validator accepted triangle_sb on generic cube-cover SAT module" >&2
  exit 1
fi
rg -q 'sat_proof_route=cube_cover_generic requires triangle_sb=none' \
  "$GENERIC_WORK/bad-triangle.out"

cp "$GENERIC_MANIFEST" "$GENERIC_WORK/missing-cube-artifact.env"
sed -i 's/^cube_cover_certificate_path=.*/cube_cover_certificate_path=NONE/' \
  "$GENERIC_WORK/missing-cube-artifact.env"
sed -i 's/^cube_cover_certificate_sha256=.*/cube_cover_certificate_sha256=NONE/' \
  "$GENERIC_WORK/missing-cube-artifact.env"
if "$VALIDATOR" "$GENERIC_WORK/missing-cube-artifact.env" > "$GENERIC_WORK/missing-cube-artifact.out" 2>&1; then
  echo "error: validator accepted generic cube-cover route without cover artifact" >&2
  exit 1
fi
rg -q 'sat_proof_route=cube_cover_generic requires concrete cube_cover_certificate artifact' \
  "$GENERIC_WORK/missing-cube-artifact.out"

ARBITRARY_WORK="$WORK/arbitrary"
mkdir -p "$ARBITRARY_WORK"
CUBE_COVER_ROUTE=cube_cover_arbitrary_complement WORK="$ARBITRARY_WORK" "$MAKER" \
  > "$ARBITRARY_WORK/maker.out"
ARBITRARY_MANIFEST="$ARBITRARY_WORK/candidate.manifest"
[[ -s "$ARBITRARY_MANIFEST" ]] || {
  echo "error: arbitrary maker did not emit candidate.manifest" >&2
  exit 1
}
rg -q '^candidate_id=k6_cube_cover_arbitrary_complement_smoke_not_planar$' \
  "$ARBITRARY_MANIFEST"
rg -q '^promotable=0$' "$ARBITRARY_MANIFEST"
rg -q '^geometry_proof_type=none$' "$ARBITRARY_MANIFEST"
rg -q '^sat_proof_route=cube_cover_generic$' "$ARBITRARY_MANIFEST"
rg -q '^cube_cover_certificate_path=NONE$' "$ARBITRARY_MANIFEST"
rg -q '^cube_cover_complement_cnf_path=cube_cover_complement\.cnf$' "$ARBITRARY_MANIFEST"
rg -q '^cube_cover_complement_lrat_path=cube_cover_complement\.lrat$' "$ARBITRARY_MANIFEST"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=k6_cube_cover_arbitrary_complement_smoke_not_planar$' \
  "$ARBITRARY_WORK/maker.out"
rg -q '^chi6_cube_cover_smoke_manifest: PASS$' "$ARBITRARY_WORK/maker.out"
rg -q '^theorem k65cube_unsat_from_arbitrary_cube_cover' \
  "$ARBITRARY_WORK/SounioSatK65CubeCoverReflect.lean"
rg -q 'SounioSatCubeCover.cubeCoverComplementCNF' \
  "$ARBITRARY_WORK/SounioSatK65CubeCoverReflect.lean"
rg -q 'SounioSatCubeCover.cube_cover_of_complement_unsat' \
  "$ARBITRARY_WORK/SounioSatK65CubeCoverReflect.lean"
"$VALIDATOR" "$ARBITRARY_MANIFEST" > "$ARBITRARY_WORK/validator.out"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=k6_cube_cover_arbitrary_complement_smoke_not_planar$' \
  "$ARBITRARY_WORK/validator.out"

cp "$ARBITRARY_MANIFEST" "$ARBITRARY_WORK/missing-complement.env"
sed -i 's/^cube_cover_complement_lrat_path=.*/cube_cover_complement_lrat_path=NONE/' \
  "$ARBITRARY_WORK/missing-complement.env"
sed -i 's/^cube_cover_complement_lrat_sha256=.*/cube_cover_complement_lrat_sha256=NONE/' \
  "$ARBITRARY_WORK/missing-complement.env"
if "$VALIDATOR" "$ARBITRARY_WORK/missing-complement.env" \
    > "$ARBITRARY_WORK/missing-complement.out" 2>&1; then
  echo "error: validator accepted complement-cover route without complement LRAT" >&2
  exit 1
fi
rg -q 'sat_proof_route=cube_cover_generic requires concrete cube_cover_complement_lrat artifact' \
  "$ARBITRARY_WORK/missing-complement.out"

cp "$ARBITRARY_MANIFEST" "$ARBITRARY_WORK/bad-complement-hash.env"
sed -i 's/^cube_cover_complement_lrat_sha256=.*/cube_cover_complement_lrat_sha256=0000000000000000000000000000000000000000000000000000000000000000/' \
  "$ARBITRARY_WORK/bad-complement-hash.env"
if "$VALIDATOR" "$ARBITRARY_WORK/bad-complement-hash.env" \
    > "$ARBITRARY_WORK/bad-complement-hash.out" 2>&1; then
  echo "error: validator accepted complement-cover route with bad complement LRAT hash" >&2
  exit 1
fi
rg -q 'cube_cover_complement_lrat SHA256 mismatch' \
  "$ARBITRARY_WORK/bad-complement-hash.out"

cp "$MANIFEST" "$WORK/split-module-as-generic.env"
sed -i 's/^sat_proof_route=.*/sat_proof_route=cube_cover_generic/' \
  "$WORK/split-module-as-generic.env"
if "$VALIDATOR" "$WORK/split-module-as-generic.env" > "$WORK/split-module-as-generic.out" 2>&1; then
  echo "error: validator accepted split5 Lean module as generic cube-cover route" >&2
  exit 1
fi
rg -q 'sat_proof_route=cube_cover_generic requires colourCNFWithCube in Lean SAT module' \
  "$WORK/split-module-as-generic.out"

echo "chi6_cube_cover_smoke_candidate_manifest_gate: PASS"
