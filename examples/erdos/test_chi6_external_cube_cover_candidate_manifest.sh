#!/usr/bin/env bash
# Gate the external-DIMACS -> cube-cover candidate.manifest bridge.
#
# K6 is used only as a calibration graph: it is not planar and not a Euclidean
# chi>=6 witness. The gate proves the packaging/refutation/Lean-SAT plumbing for
# an external edge file with the same interface a real solver candidate will use.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
MAKER="$ROOT/examples/erdos/make_chi6_external_cube_cover_candidate_manifest.sh"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }
mkdir -p "$WORK"

EDGE="$WORK/k6_external.edge"
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

echo "chi6_external_cube_cover_candidate_gate: workdir=$WORK"
PACKAGE_WORK="$WORK/package_gate"
WORK="$PACKAGE_WORK" "$MAKER" "$EDGE" k6_external_cube_cover_smoke 0 \
  > "$WORK/maker.out"

MANIFEST="$PACKAGE_WORK/candidate.manifest"
[[ -s "$MANIFEST" ]] || { echo "error: maker did not emit candidate.manifest" >&2; exit 1; }
rg -q '^candidate_id=k6_external_cube_cover_smoke$' "$MANIFEST"
rg -q '^promotable=0$' "$MANIFEST"
rg -q '^geometry_proof_type=none$' "$MANIFEST"
rg -q '^sat_proof_route=cube_cover_generic$' "$MANIFEST"
rg -q '^source_meta_path=package/k6_external_cube_cover_smoke\.meta\.json$' "$MANIFEST"
rg -q '^cube_batch_path=package/k6_external_cube_cover_smoke\.cubes$' "$MANIFEST"
rg -q '^cube_cover_certificate_path=cube_cover\.out$' "$MANIFEST"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=k6_external_cube_cover_smoke$' \
  "$WORK/maker.out"
rg -q '^chi6_external_cube_cover_candidate: PASS$' "$WORK/maker.out"

cmp -s "$EDGE" "$PACKAGE_WORK/package/k6_external_cube_cover_smoke.edge"
rg -q '^k6_external_cube_cover_smoke_v0_c0: 0:0$' \
  "$PACKAGE_WORK/package/k6_external_cube_cover_smoke.cubes"
rg -q '^k6_external_cube_cover_smoke_v0_c4: 0:4$' \
  "$PACKAGE_WORK/package/k6_external_cube_cover_smoke.cubes"
rg -q '"schema": "chi6_external_dimacs_edge_package.v1"' \
  "$PACKAGE_WORK/package/k6_external_cube_cover_smoke.meta.json"
rg -q '"provenance_scope": "edge_packaging_only"' \
  "$PACKAGE_WORK/package/k6_external_cube_cover_smoke.meta.json"
rg -q '"promotion_gate": "requires_lrat_lean_and_exact_euclidean_geometry"' \
  "$PACKAGE_WORK/package/k6_external_cube_cover_smoke.meta.json"
rg -q '^theorem chi6ext_unsat_from_generic_cube_cover' \
  "$PACKAGE_WORK/SounioSatChi6ExternalCubeCoverReflect.lean"
rg -q 'SounioSatCubeCover.unsat_of_cube_cover' \
  "$PACKAGE_WORK/SounioSatChi6ExternalCubeCoverReflect.lean"

"$VALIDATOR" "$MANIFEST" > "$WORK/validator.out"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=k6_external_cube_cover_smoke$' \
  "$WORK/validator.out"

cp "$MANIFEST" "$PACKAGE_WORK/bad-source-meta.env"
sed -i 's/^source_meta_sha256=.*/source_meta_sha256=0000000000000000000000000000000000000000000000000000000000000000/' \
  "$PACKAGE_WORK/bad-source-meta.env"
if "$VALIDATOR" "$PACKAGE_WORK/bad-source-meta.env" > "$WORK/bad-source-meta.out" 2>&1; then
  echo "error: validator accepted bad source_meta hash" >&2
  exit 1
fi
rg -q 'source_meta SHA256 mismatch' "$WORK/bad-source-meta.out"

BAD_META_WORK="$WORK/bad_meta_semantic"
cp -a "$PACKAGE_WORK" "$BAD_META_WORK"
python3 - "$BAD_META_WORK/package/k6_external_cube_cover_smoke.meta.json" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, encoding="ascii") as f:
    meta = json.load(f)
meta["candidate_id"] = "wrong_candidate"
with open(path, "w", encoding="ascii") as f:
    json.dump(meta, f, indent=2, sort_keys=True)
    f.write("\n")
PY
bad_meta_sha="$(sha256sum "$BAD_META_WORK/package/k6_external_cube_cover_smoke.meta.json" | awk '{print $1}')"
sed -i "s/^source_meta_sha256=.*/source_meta_sha256=$bad_meta_sha/" \
  "$BAD_META_WORK/candidate.manifest"
if "$VALIDATOR" "$BAD_META_WORK/candidate.manifest" > "$WORK/bad-source-meta-semantic.out" 2>&1; then
  echo "error: validator accepted source_meta with mismatched candidate_id" >&2
  exit 1
fi
rg -q 'source_meta candidate_id mismatch' "$WORK/bad-source-meta-semantic.out"

if WORK="$WORK/bad_id" "$MAKER" "$EDGE" 'bad/id' 0 > "$WORK/bad_id.out" 2>&1; then
  echo "error: maker accepted bad candidate id" >&2
  exit 1
fi
rg -q 'candidate-id must use only' "$WORK/bad_id.out"

if WORK="$WORK/bad_split" "$MAKER" "$EDGE" k6_bad_split 0,6 \
    > "$WORK/bad_split.out" 2>&1; then
  echo "error: maker accepted out-of-range split vertex" >&2
  exit 1
fi
rg -q 'split vertex out of range: 6' "$WORK/bad_split.out"

echo "chi6_external_cube_cover_candidate_gate: PASS"
