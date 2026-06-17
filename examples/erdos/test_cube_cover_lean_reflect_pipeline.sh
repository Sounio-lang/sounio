#!/usr/bin/env bash
# End-to-end finite SAT smoke: cube-unit LRAT leaves -> Lean cover composition.
#
# K6/k=5 is only a calibration graph. The gate proves that the pipeline can
# generate Lean-checked cube-leaf UNSAT facts and compose them back to the plain
# colourCNF via SounioSatCubeCover.unsat_of_split_vertex5.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
REFUTER="$ROOT/examples/erdos/cube_sieve_refute_batch.py"
COVER="$ROOT/examples/erdos/cube_cover_certificate.py"
GEN="$ROOT/examples/erdos/gen_lean_cube_cover_reflect.py"
LEAN_DIR="$ROOT/formal/lean4"
LOCK="$ROOT/scripts/dev/souc-build-lock.sh"
LAKE="${LAKE:-/workspace/.home/openvscode-server/.elan/bin/lake}"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
[[ -x "$LOCK" ]] || { echo "error: missing build lock helper: $LOCK" >&2; exit 1; }
[[ -x "$LAKE" ]] || { echo "error: missing lake executable: $LAKE" >&2; exit 1; }
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

cat > "$WORK/k6_cover.cubes" <<'EOF'
v0_c0: 0:0
v0_c1: 0:1
v0_c2: 0:2
v0_c3: 0:3
v0_c4: 0:4
EOF

echo "cube_cover_lean_reflect_pipeline_gate: workdir=$WORK"
python3 "$REFUTER" "$WORK/k6.edge" 5 "$WORK/k6_cover.cubes" "$WORK/refute" \
  > "$WORK/refute.out"
python3 "$COVER" "$WORK/k6.edge" 5 "$WORK/k6_cover.cubes" "$WORK/refute.out" \
  > "$WORK/cover.out"
python3 "$GEN" "$WORK/k6.edge" 5 "$WORK/k6_cover.cubes" "$WORK/refute.out" \
  "$WORK/SounioSatK65CubeCoverReflect.lean" \
  --module SounioSatK65CubeCoverReflect \
  --prefix k65cube \
  > "$WORK/gen.out"

rg -q '^lean_cube_cover_reflect v1$' "$WORK/gen.out"
rg -q '^claim=finite_colourCNF_unsat_from_checked_cube_lrat_leaves$' "$WORK/gen.out"
rg -q '^geometry_claim=none$' "$WORK/gen.out"
rg -q '^status=lean_cube_cover_reflect_emitted$' "$WORK/gen.out"
rg -q '^theorem k65cube_unsat_from_v0_split' "$WORK/SounioSatK65CubeCoverReflect.lean"
if rg -n '\b(sorry|admit)\b' "$WORK/SounioSatK65CubeCoverReflect.lean"; then
  echo "error: generated Lean file contains an incomplete proof marker" >&2
  exit 1
fi

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" env lean "$WORK/SounioSatK65CubeCoverReflect.lean" > "$WORK/lean.out"
)

rg -q "'k65cube_unsat_from_v0_split' depends on axioms:" "$WORK/lean.out"
for c in 0 1 2 3 4; do
  rg -q "k65cube_v0_c${c}_check\\._native\\.native_decide\\.ax" "$WORK/lean.out"
done
if rg -q 'sorryAx' "$WORK/lean.out"; then
  echo "error: generated cube-cover theorem depends on sorryAx" >&2
  exit 1
fi

echo "cube_cover_lean_reflect_pipeline_gate: PASS"
