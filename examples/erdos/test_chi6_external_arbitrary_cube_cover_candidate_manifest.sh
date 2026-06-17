#!/usr/bin/env bash
# Gate a non-promotable candidate.manifest through the arbitrary complement-cover route.
#
# K6 is only a SAT plumbing calibration graph here. The cube family is deliberately
# non-product-shaped: five singleton cubes cover via the v0 at-least-one clause,
# while extra two-vertex cubes force the generic cube-cover manifest path to carry
# arbitrary complement CNF/LRAT artifacts. No Euclidean geometry or chi(R^2)>=6
# claim is made.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

LEAN_DIR="$ROOT/formal/lean4"
LOCK="$ROOT/scripts/dev/souc-build-lock.sh"
LAKE="${LAKE:-$(command -v lake || true)}"
if [[ -z "$LAKE" && -x /workspace/.home/openvscode-server/.elan/bin/lake ]]; then
  LAKE=/workspace/.home/openvscode-server/.elan/bin/lake
fi

REFUTER="$ROOT/examples/erdos/cube_sieve_refute_batch.py"
COMP_CNF="$ROOT/examples/erdos/cube_cover_complement_cnf.py"
CONVERTER="$ROOT/examples/erdos/drup_to_lrat_rup.py"
GEN="$ROOT/examples/erdos/gen_lean_cube_cover_reflect.py"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"
MAKER="$ROOT/examples/erdos/make_chi6_external_arbitrary_cube_cover_candidate_manifest.sh"

[[ -x "$LOCK" ]] || { echo "error: missing build lock helper: $LOCK" >&2; exit 1; }
[[ -n "$LAKE" ]] || { echo "error: lake not found; set LAKE=/path/to/lake" >&2; exit 127; }
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }

python3 -m py_compile "$REFUTER" "$COMP_CNF" "$CONVERTER" "$GEN"

mkdir -p "$WORK/refute"
CANDIDATE_ID="k6_external_arbitrary_cover_smoke"
EDGE="$WORK/$CANDIDATE_ID.edge"
CUBES="$WORK/$CANDIDATE_ID.cubes"
REFUTE_OUT="$WORK/cube_refute.out"
COVER_CNF="$WORK/cover_complement.cnf"
COVER_LRAT="$WORK/cover_complement.lrat"
LEAN_OUT="$WORK/SounioSatChi6ExternalArbitraryCoverReflect.lean"
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

python3 - "$CUBES" "$CANDIDATE_ID" <<'PY'
from pathlib import Path
import sys

out = Path(sys.argv[1])
prefix = sys.argv[2]
with out.open("w", encoding="ascii") as f:
    for c0 in range(5):
        f.write(f"{prefix}_v0_c{c0}: 0:{c0}\n")
    for c1 in range(5):
        for c2 in range(5):
            f.write(f"{prefix}_v1_c{c1}_v2_c{c2}: 1:{c1} 2:{c2}\n")
PY

cube_count="$(wc -l < "$CUBES" | tr -d ' ')"
if [[ "$cube_count" != "30" ]]; then
  echo "error: expected 30 non-product cubes, got $cube_count" >&2
  exit 1
fi

echo "chi6_external_arbitrary_cube_cover_candidate_gate: workdir=$WORK"
python3 "$REFUTER" "$EDGE" 5 "$CUBES" "$WORK/refute" > "$REFUTE_OUT"
rg -q '^formula_kind=colourCNF$' "$REFUTE_OUT"
rg -q '^cube_count=30$' "$REFUTE_OUT"
rg -q '^solver_unsat_count=30$' "$REFUTE_OUT"
rg -q '^lrat_artifact_count=30$' "$REFUTE_OUT"
rg -q '^failed_count=0$' "$REFUTE_OUT"
rg -q '^formal_proof_checker=none$' "$REFUTE_OUT"
rg -q '^global_unsat_claim=none$' "$REFUTE_OUT"

python3 "$COMP_CNF" "$EDGE" 5 "$CUBES" "$COVER_CNF" > "$WORK/cover_complement.out"
rg -q '^cube_count=30$' "$WORK/cover_complement.out"
rg -q '^clause_count=111$' "$WORK/cover_complement.out"
rg -q '^claim=base_plus_cube_blockers_dimacs_only$' "$WORK/cover_complement.out"

printf '0\n' > "$WORK/cover_complement.drup"
python3 "$CONVERTER" "$COVER_CNF" "$WORK/cover_complement.drup" "$COVER_LRAT" \
  > "$WORK/cover_lrat.out" 2> "$WORK/cover_lrat.err"
[[ -s "$COVER_LRAT" ]]
rg -q 'original=111' "$WORK/cover_lrat.err"
rg -q 'empty=1' "$WORK/cover_lrat.err"

python3 "$GEN" "$EDGE" 5 "$CUBES" "$REFUTE_OUT" "$LEAN_OUT" \
  --module SounioSatChi6ExternalArbitraryCoverReflect \
  --prefix chi6arb \
  --composition arbitrary \
  --cover-cnf "$COVER_CNF" \
  --cover-lrat "$COVER_LRAT" \
  > "$WORK/gen.out"

rg -q '^leaf_count=30$' "$WORK/gen.out"
rg -q '^composition=arbitrary$' "$WORK/gen.out"
rg -q '^cover_claim=base_plus_cube_blockers_unsat$' "$WORK/gen.out"
rg -q '^theorem chi6arb_unsat_from_arbitrary_cube_cover' "$LEAN_OUT"
leaf_check_count="$(rg -c '^theorem chi6arb_leaf[0-9]+_check' "$LEAN_OUT")"
if [[ "$leaf_check_count" != "30" ]]; then
  echo "error: expected 30 generated LRAT leaf checks, got $leaf_check_count" >&2
  exit 1
fi
rg -q 'SounioSatCubeCover.cubeCoverComplementCNF' "$LEAN_OUT"
rg -q '^theorem chi6arb_cover_complement_check :' "$LEAN_OUT"
rg -q 'Std\.Tactic\.BVDecide\.LRAT\.check_sound _ \(SounioSatCubeCover\.cubeCoverComplementCNF' \
  "$LEAN_OUT"
rg -q 'SounioSatCubeCover.cube_cover_of_complement_unsat' "$LEAN_OUT"
rg -q '^theorem chi6arb_cube_cover :' "$LEAN_OUT"
rg -q 'SounioSatCubeCover.unsat_of_cube_cover' "$LEAN_OUT"
rg -q 'Std.Tactic.BVDecide.LRAT.check' "$LEAN_OUT"
if rg -q 'EuclideanNatEdgeExactGeometry|ExactSquaredDistancePlane|split_vertices_cubes_cover|splitVerticesCubes|unsat_of_split_vertex5|colourCNFWithUnit' "$LEAN_OUT"; then
  echo "error: arbitrary-cover Lean used geometry or split-cover-only surface" >&2
  exit 1
fi
if rg -q '\b(sorry|admit)\b|#exit|#eval|#check' "$LEAN_OUT"; then
  echo "error: generated arbitrary-cover Lean contains forbidden proof/debug marker" >&2
  exit 1
fi

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" env lean "$LEAN_OUT" > "$WORK/lean_build.log" 2>&1
)
if rg -q 'error:|sorryAx|warning:|deprecated:' "$WORK/lean_build.log"; then
  cat "$WORK/lean_build.log" >&2
  echo "error: generated arbitrary-cover Lean failed strict log check" >&2
  exit 1
fi

sha() {
  [[ -f "$1" ]] || { echo "error: file not found for hashing: $1" >&2; exit 1; }
  sha256sum "$1" | awk '{print $1}'
}

GENERATOR_COMMIT="$(git -C "$ROOT" rev-parse --verify HEAD 2>/dev/null)" || {
  echo "error: unable to resolve generator git commit" >&2
  exit 1
}

cat > "$MANIFEST" <<EOF
candidate_manifest_version=1
promotable=0
candidate_id=$CANDIDATE_ID
n=6
m=15
k=5
edge_path=$CANDIDATE_ID.edge
edge_sha256=$(sha "$EDGE")
cnf_path=NONE
cnf_sha256=NONE
drat_or_lrat_path=NONE
drat_or_lrat_sha256=NONE
lean_sat_module_path=SounioSatChi6ExternalArbitraryCoverReflect.lean
lean_sat_module_sha256=$(sha "$LEAN_OUT")
geometry_module_path=NONE
geometry_module_sha256=NONE
geometry_proof_type=none
sat_proof_route=cube_cover_generic
triangle_sb=none
generator_commit=$GENERATOR_COMMIT
producer_command=WORK=$WORK LAKE=$LAKE examples/erdos/test_chi6_external_arbitrary_cube_cover_candidate_manifest.sh
lean_build_command=lake env lean SounioSatChi6ExternalArbitraryCoverReflect.lean
offload_review_raw=NONE
offload_review_sha256=NONE
cube_batch_path=$CANDIDATE_ID.cubes
cube_batch_sha256=$(sha "$CUBES")
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

"$VALIDATOR" "$MANIFEST" > "$WORK/validator.out"
rg -q "^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=$CANDIDATE_ID$" \
  "$WORK/validator.out"

cp "$MANIFEST" "$WORK/bad-complement-hash.manifest"
sed -i 's/^cube_cover_complement_lrat_sha256=.*/cube_cover_complement_lrat_sha256=0000000000000000000000000000000000000000000000000000000000000000/' \
  "$WORK/bad-complement-hash.manifest"
if "$VALIDATOR" "$WORK/bad-complement-hash.manifest" > "$WORK/bad-complement-hash.out" 2>&1; then
  echo "error: validator accepted corrupted complement LRAT hash" >&2
  exit 1
fi
rg -q 'cube_cover_complement_lrat SHA256 mismatch' "$WORK/bad-complement-hash.out"

cp "$MANIFEST" "$WORK/missing-complement.manifest"
sed -i 's/^cube_cover_complement_cnf_path=.*/cube_cover_complement_cnf_path=NONE/' \
  "$WORK/missing-complement.manifest"
sed -i 's/^cube_cover_complement_cnf_sha256=.*/cube_cover_complement_cnf_sha256=NONE/' \
  "$WORK/missing-complement.manifest"
if "$VALIDATOR" "$WORK/missing-complement.manifest" > "$WORK/missing-complement.out" 2>&1; then
  echo "error: validator accepted complement-cover Lean without complement CNF" >&2
  exit 1
fi
rg -q 'requires concrete cube_cover_complement_cnf artifact' "$WORK/missing-complement.out"

cp "$MANIFEST" "$WORK/missing-complement-lrat.manifest"
sed -i 's/^cube_cover_complement_lrat_path=.*/cube_cover_complement_lrat_path=NONE/' \
  "$WORK/missing-complement-lrat.manifest"
sed -i 's/^cube_cover_complement_lrat_sha256=.*/cube_cover_complement_lrat_sha256=NONE/' \
  "$WORK/missing-complement-lrat.manifest"
if "$VALIDATOR" "$WORK/missing-complement-lrat.manifest" > "$WORK/missing-complement-lrat.out" 2>&1; then
  echo "error: validator accepted complement-cover Lean without complement LRAT" >&2
  exit 1
fi
rg -q 'requires concrete cube_cover_complement_lrat artifact' "$WORK/missing-complement-lrat.out"

MAKER_WORK="$WORK/maker_gate"
WORK="$MAKER_WORK" "$MAKER" "$EDGE" "$CANDIDATE_ID" "$CUBES" "$WORK/cover_complement.drup" \
  > "$WORK/maker.out"
rg -q '^chi6_external_arbitrary_cube_cover_candidate: PASS$' "$WORK/maker.out"
MAKER_MANIFEST="$MAKER_WORK/candidate.manifest"
[[ -s "$MAKER_MANIFEST" ]] || { echo "error: maker did not emit candidate.manifest" >&2; exit 1; }
rg -q '^promotable=0$' "$MAKER_MANIFEST"
rg -q '^geometry_proof_type=none$' "$MAKER_MANIFEST"
rg -q '^sat_proof_route=cube_cover_generic$' "$MAKER_MANIFEST"
rg -q '^cube_cover_certificate_path=NONE$' "$MAKER_MANIFEST"
rg -q '^cube_cover_complement_cnf_path=cover_complement\.cnf$' "$MAKER_MANIFEST"
rg -q '^cube_cover_complement_lrat_path=cover_complement\.lrat$' "$MAKER_MANIFEST"
rg -q "^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=$CANDIDATE_ID$" \
  "$MAKER_WORK/manifest_validator.log"
rg -q 'SounioSatCubeCover.cube_cover_of_complement_unsat' \
  "$MAKER_WORK/SounioSatChi6ExternalArbitraryCoverReflect.lean"
rg -q 'Std\.Tactic\.BVDecide\.LRAT\.check_sound _ \(SounioSatCubeCover\.cubeCoverComplementCNF' \
  "$MAKER_WORK/SounioSatChi6ExternalArbitraryCoverReflect.lean"
rg -q '^theorem chi6arb_cube_cover :' \
  "$MAKER_WORK/SounioSatChi6ExternalArbitraryCoverReflect.lean"
if rg -q 'split_vertices_cubes_cover|splitVerticesCubes' \
    "$MAKER_WORK/SounioSatChi6ExternalArbitraryCoverReflect.lean"; then
  echo "error: maker Lean used split-product cover theorem on arbitrary route" >&2
  exit 1
fi

if WORK="$WORK/bad_id" "$MAKER" "$EDGE" 'bad/id' "$CUBES" "$WORK/cover_complement.drup" \
    > "$WORK/bad_id.out" 2>&1; then
  echo "error: arbitrary maker accepted bad candidate id" >&2
  exit 1
fi
rg -q 'candidate-id must use only' "$WORK/bad_id.out"

echo "manifest=$MANIFEST"
echo "lean_sat_module=$LEAN_OUT"
echo "cube_refutation_summary=$REFUTE_OUT"
echo "cube_cover_complement_cnf=$COVER_CNF"
echo "cube_cover_complement_lrat=$COVER_LRAT"
echo "chi6_external_arbitrary_cube_cover_candidate_gate: PASS"
