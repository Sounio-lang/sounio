#!/usr/bin/env bash
# Temp-only tests for validate_chi6_candidate_manifest.sh.
#
# The positive fixture is an explicitly non-promotable K6/k=5 SB5 smoke
# manifest. It validates manifest structure, hashes, edge metadata, and triangle
# precolour wiring only. It is not a Euclidean chi>=6 witness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

mkdir -p "$WORK"
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }

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

echo "chi6_manifest_validator: workdir=$WORK"
SB_MODE=1 WORK="$WORK" \
  "$ROOT/examples/erdos/make_graph_reflect_certificate.sh" \
    "$WORK/k6.edge" 5 "$WORK/SounioSatK65ManifestReflect.lean" \
    k65manifest SounioSatK65ManifestReflect "$WORK" \
  > "$WORK/producer.log"

cat > "$WORK/manifest.env" <<EOF
candidate_manifest_version=1
promotable=0
candidate_id=k6_sb5_smoke_not_planar
n=6
m=15
k=5
edge_path=k6.edge
edge_sha256=$(sha256sum "$WORK/k6.edge" | awk '{print $1}')
cnf_path=souc_sat_worker.cnf
cnf_sha256=$(sha256sum "$WORK/souc_sat_worker.cnf" | awk '{print $1}')
drat_or_lrat_path=k65manifest.lrat
drat_or_lrat_sha256=$(sha256sum "$WORK/k65manifest.lrat" | awk '{print $1}')
lean_sat_module_path=SounioSatK65ManifestReflect.lean
lean_sat_module_sha256=$(sha256sum "$WORK/SounioSatK65ManifestReflect.lean" | awk '{print $1}')
geometry_module_path=NONE
geometry_module_sha256=NONE
geometry_proof_type=none
sat_proof_route=triangle_sb5_lrat
triangle_sb=0,1,2
generator_commit=$(git -C "$ROOT" rev-parse --verify HEAD 2>/dev/null || echo UNKNOWN)
producer_command=SB_MODE=1 examples/erdos/make_graph_reflect_certificate.sh k6.edge 5 SounioSatK65ManifestReflect.lean k65manifest SounioSatK65ManifestReflect
lean_build_command=lake env lean SounioSatK65ManifestReflect.lean
offload_review_raw=NONE
offload_review_sha256=NONE
EOF

"$VALIDATOR" "$WORK/manifest.env" | tee "$WORK/validator.out"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=k6_sb5_smoke_not_planar$' "$WORK/validator.out"

cp "$WORK/SounioSatK65ManifestReflect.lean" "$WORK/SounioSatK65ManifestReflectSpaced.lean"
sed -i $'s/colourCNFsb5 0 1 2 6/colourCNFsb5   0\\t1  2   6/' \
  "$WORK/SounioSatK65ManifestReflectSpaced.lean"
cp "$WORK/manifest.env" "$WORK/spaced-sb5.env"
sed -i 's/^lean_sat_module_path=.*/lean_sat_module_path=SounioSatK65ManifestReflectSpaced.lean/' \
  "$WORK/spaced-sb5.env"
sed -i "s/^lean_sat_module_sha256=.*/lean_sat_module_sha256=$(sha256sum "$WORK/SounioSatK65ManifestReflectSpaced.lean" | awk '{print $1}')/" \
  "$WORK/spaced-sb5.env"
"$VALIDATOR" "$WORK/spaced-sb5.env" > "$WORK/spaced-sb5.out"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=k6_sb5_smoke_not_planar$' \
  "$WORK/spaced-sb5.out"

cp "$WORK/manifest.env" "$WORK/bad-hash.env"
sed -i 's/^edge_sha256=.*/edge_sha256=0000000000000000000000000000000000000000000000000000000000000000/' "$WORK/bad-hash.env"
if "$VALIDATOR" "$WORK/bad-hash.env" > "$WORK/bad-hash.out" 2>&1; then
  echo "error: bad edge hash unexpectedly validated" >&2
  exit 1
fi
rg -q 'edge SHA256 mismatch' "$WORK/bad-hash.out"

cp "$WORK/manifest.env" "$WORK/bad-route-alias.env"
sed -i 's/^sat_proof_route=.*/sat_proof_route=triangle_sb5/' "$WORK/bad-route-alias.env"
if "$VALIDATOR" "$WORK/bad-route-alias.env" > "$WORK/bad-route-alias.out" 2>&1; then
  echo "error: legacy triangle_sb5 route alias unexpectedly validated" >&2
  exit 1
fi
rg -q 'sat_proof_route must be none, plain_lrat, triangle_sb5_lrat, cube_cover_split5, or cube_cover_generic' \
  "$WORK/bad-route-alias.out"

cp "$WORK/manifest.env" "$WORK/bad-candidate-id.env"
sed -i 's/^candidate_id=.*/candidate_id=bad id/' "$WORK/bad-candidate-id.env"
if "$VALIDATOR" "$WORK/bad-candidate-id.env" > "$WORK/bad-candidate-id.out" 2>&1; then
  echo "error: invalid candidate_id unexpectedly validated" >&2
  exit 1
fi
rg -q "candidate_id must use only letters, digits, '.', '_', or '-'" \
  "$WORK/bad-candidate-id.out"

cp "$WORK/SounioSatK65ManifestReflect.lean" "$WORK/SounioSatK65ManifestReflectEval.lean"
printf '\n#eval 1\n' >> "$WORK/SounioSatK65ManifestReflectEval.lean"
cp "$WORK/manifest.env" "$WORK/bad-eval.env"
sed -i 's/^lean_sat_module_path=.*/lean_sat_module_path=SounioSatK65ManifestReflectEval.lean/' \
  "$WORK/bad-eval.env"
sed -i "s/^lean_sat_module_sha256=.*/lean_sat_module_sha256=$(sha256sum "$WORK/SounioSatK65ManifestReflectEval.lean" | awk '{print $1}')/" \
  "$WORK/bad-eval.env"
if "$VALIDATOR" "$WORK/bad-eval.env" > "$WORK/bad-eval.out" 2>&1; then
  echo "error: Lean module with #eval unexpectedly validated" >&2
  exit 1
fi
rg -q 'sorry/admit/#exit/#eval/#check' "$WORK/bad-eval.out"

cp "$WORK/k6.edge" "$WORK/leading-zero.edge"
sed -i 's/^e 1 2$/e 01 2/' "$WORK/leading-zero.edge"
cp "$WORK/manifest.env" "$WORK/leading-zero.env"
sed -i 's/^edge_path=.*/edge_path=leading-zero.edge/' "$WORK/leading-zero.env"
sed -i "s/^edge_sha256=.*/edge_sha256=$(sha256sum "$WORK/leading-zero.edge" | awk '{print $1}')/" \
  "$WORK/leading-zero.env"
if "$VALIDATOR" "$WORK/leading-zero.env" > "$WORK/leading-zero.out" 2>&1; then
  echo "error: leading-zero DIMACS vertex id unexpectedly validated" >&2
  exit 1
fi
rg -q 'edge file has malformed/out-of-range/self-loop edge' "$WORK/leading-zero.out"

cat > "$WORK/reversed-triangle.edge" <<'EOF'
p edge 3 3
e 2 1
e 3 2
e 1 3
EOF

cat > "$WORK/reversed-triangle.env" <<EOF
candidate_manifest_version=1
promotable=0
candidate_id=reversed_triangle_nonpromotable
n=3
m=3
k=5
edge_path=reversed-triangle.edge
edge_sha256=$(sha256sum "$WORK/reversed-triangle.edge" | awk '{print $1}')
cnf_path=NONE
cnf_sha256=NONE
drat_or_lrat_path=NONE
drat_or_lrat_sha256=NONE
lean_sat_module_path=NONE
lean_sat_module_sha256=NONE
geometry_module_path=NONE
geometry_module_sha256=NONE
geometry_proof_type=none
sat_proof_route=triangle_sb5_lrat
triangle_sb=0,1,2
generator_commit=$(git -C "$ROOT" rev-parse --verify HEAD 2>/dev/null || echo UNKNOWN)
producer_command=NONE
lean_build_command=NONE
offload_review_raw=NONE
offload_review_sha256=NONE
EOF

"$VALIDATOR" "$WORK/reversed-triangle.env" | tee "$WORK/reversed-triangle.out"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=reversed_triangle_nonpromotable$' \
  "$WORK/reversed-triangle.out"

cp "$WORK/manifest.env" "$WORK/bad-triangle.env"
sed -i 's/^triangle_sb=.*/triangle_sb=0,1,5/' "$WORK/bad-triangle.env"
if "$VALIDATOR" "$WORK/bad-triangle.env" > "$WORK/bad-triangle.out" 2>&1; then
  echo "error: bad triangle unexpectedly validated" >&2
  exit 1
fi
rg -q 'unordered triangle_sb edge missing|Lean SAT module does not use matching colourCNFsb5 triangle' \
  "$WORK/bad-triangle.out"

cp "$WORK/manifest.env" "$WORK/bad-promote.env"
sed -i 's/^promotable=.*/promotable=1/' "$WORK/bad-promote.env"
if "$VALIDATOR" "$WORK/bad-promote.env" > "$WORK/bad-promote.out" 2>&1; then
  echo "error: promotable manifest without geometry unexpectedly validated" >&2
  exit 1
fi
rg -q 'promotable=1 requires geometry_proof_type=euclidean' "$WORK/bad-promote.out"

cp "$WORK/manifest.env" "$WORK/bad-geometry-type.env"
sed -i 's/^geometry_proof_type=.*/geometry_proof_type=finite_smoke/' "$WORK/bad-geometry-type.env"
if "$VALIDATOR" "$WORK/bad-geometry-type.env" > "$WORK/bad-geometry-type.out" 2>&1; then
  echo "error: finite_smoke geometry type without geometry unexpectedly validated" >&2
  exit 1
fi
rg -q 'geometry_proof_type=finite_smoke requires concrete geometry_module artifact' \
  "$WORK/bad-geometry-type.out"

echo "chi6_manifest_validator: PASS"
