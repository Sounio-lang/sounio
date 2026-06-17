#!/usr/bin/env bash
# Gate for a multi-vertex split-product cube cover.
#
# K6/k=5 remains only a finite SAT calibration target. This test splits two
# vertices, producing 25 cube leaves, checks the product-cover certificate shape,
# and composes the Lean-checked LRAT leaves through the generic CubeCover theorem.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEAN_DIR="$ROOT/formal/lean4"
LOCK="$ROOT/scripts/dev/souc-build-lock.sh"
LAKE="${LAKE:-/workspace/.home/openvscode-server/.elan/bin/lake}"
REFUTER="$ROOT/examples/erdos/cube_sieve_refute_batch.py"
COVER="$ROOT/examples/erdos/cube_cover_certificate.py"
GEN="$ROOT/examples/erdos/gen_lean_cube_cover_reflect.py"

if [[ ! -x "$LOCK" ]]; then
  echo "error: missing build lock helper: $LOCK" >&2
  exit 1
fi
if [[ ! -x "$LAKE" ]]; then
  echo "error: missing lake executable: $LAKE" >&2
  exit 1
fi
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$REFUTER" "$COVER" "$GEN"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM
mkdir -p "$WORK/refute"

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

for c0 in 0 1 2 3 4; do
  for c1 in 0 1 2 3 4; do
    printf 'v0_c%s_v1_c%s: 0:%s 1:%s\n' "$c0" "$c1" "$c0" "$c1"
  done
done > "$WORK/k6_v0_v1_cover.cubes"

echo "cube_cover_product_lean_reflect_pipeline_gate: workdir=$WORK"
python3 "$REFUTER" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1_cover.cubes" "$WORK/refute" \
  > "$WORK/refute.out"
python3 "$COVER" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1_cover.cubes" "$WORK/refute.out" \
  --cover-rule split_vertices_atleast_one_product \
  --split-vertices 0,1 \
  > "$WORK/cover.out"

rg -q '^cube_cover_certificate v1$' "$WORK/cover.out"
rg -q '^cover_rule=split_vertices_atleast_one_product$' "$WORK/cover.out"
rg -q '^split_vertices=0,1$' "$WORK/cover.out"
rg -q '^base_clause=atleast_one_colour_for_each_split_vertex$' "$WORK/cover.out"
rg -q '^leaf_count=25$' "$WORK/cover.out"
rg -q '^covered_cube_count=25$' "$WORK/cover.out"
rg -q '^lrat_artifact_count=25$' "$WORK/cover.out"
rg -q '^cover_complete_for_split_vertices=1$' "$WORK/cover.out"
rg -q '^cover_claim=atleast_one_product_cover_for_split_vertices$' "$WORK/cover.out"
rg -q '^lean_cover_obligation=CubeCover n k edges splitVerticesCubes$' "$WORK/cover.out"
rg -q '^promotion_gate=REJECT_LEAN_CUBECOVER_PROOF_NOT_ATTACHED$' "$WORK/cover.out"
rg -q '^promotable=0$' "$WORK/cover.out"
rg -q '^leaf index=0 cube_id=v0_c0_v1_c0 assignment_count=2 assignments=0:0,1:0 cube_sha256=[0-9a-f]{64} cnf_sha256=[0-9a-f]{64} lrat_sha256=[0-9a-f]{64}$' \
  "$WORK/cover.out"
rg -q '^leaf index=24 cube_id=v0_c4_v1_c4 assignment_count=2 assignments=0:4,1:4 cube_sha256=[0-9a-f]{64} cnf_sha256=[0-9a-f]{64} lrat_sha256=[0-9a-f]{64}$' \
  "$WORK/cover.out"

python3 "$GEN" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1_cover.cubes" "$WORK/refute.out" \
  "$WORK/SounioSatK65ProductCoverReflect.lean" \
  --module SounioSatK65ProductCoverReflect \
  --prefix k65prod \
  --composition generic \
  > "$WORK/gen.out"

rg -q '^split_vertices=0,1$' "$WORK/gen.out"
rg -q '^leaf_count=25$' "$WORK/gen.out"
rg -q '^theorem k65prod_unsat_from_generic_cube_cover' \
  "$WORK/SounioSatK65ProductCoverReflect.lean"
rg -q 'SounioSatCubeCover.splitVerticesCubes 5 \[0, 1\]' \
  "$WORK/SounioSatK65ProductCoverReflect.lean"
rg -q 'SounioSatCubeCover.split_vertices_cubes_cover' \
  "$WORK/SounioSatK65ProductCoverReflect.lean"
rg -q 'SounioSatCubeCover.unsat_of_cube_cover' \
  "$WORK/SounioSatK65ProductCoverReflect.lean"

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" env lean "$WORK/SounioSatK65ProductCoverReflect.lean" \
    > "$WORK/lean_build.log" 2>&1
)
if rg -q 'sorryAx' "$WORK/lean_build.log"; then
  echo "error: product cover generated Lean depends on sorryAx" >&2
  exit 1
fi

sed '/v0_c4_v1_c4/d' "$WORK/k6_v0_v1_cover.cubes" > "$WORK/missing.cubes"
if python3 "$COVER" "$WORK/k6.edge" 5 "$WORK/missing.cubes" "$WORK/refute.out" \
    --cover-rule split_vertices_atleast_one_product \
    --split-vertices 0,1 \
    > "$WORK/missing.out" 2>&1; then
  echo "error: product cover accepted a missing leaf" >&2
  exit 1
fi
rg -q 'refutation summary cube_batch_sha256 mismatch|split product cover needs exactly 25 cubes' \
  "$WORK/missing.out"

echo "cube_cover_product_lean_reflect_pipeline_gate: PASS"
