#!/usr/bin/env bash
# Gate for the Lean-side finite cube-cover composition adapter.
#
# This does not claim a chi(R^2)>=6 witness. It checks only the reusable theorem
# that five UNSAT one-literal split leaves for one vertex imply UNSAT of the
# plain k=5 graph-colouring CNF.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEAN_DIR="$ROOT/formal/lean4"
LOCK="$ROOT/scripts/dev/souc-build-lock.sh"
LAKE="${LAKE:-/workspace/.home/openvscode-server/.elan/bin/lake}"

if [[ ! -x "$LOCK" ]]; then
  echo "error: missing build lock helper: $LOCK" >&2
  exit 1
fi
if [[ ! -x "$LAKE" ]]; then
  echo "error: missing lake executable: $LAKE" >&2
  exit 1
fi
command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }

if rg -n '\b(sorry|admit)\b' "$LEAN_DIR/SounioSatCubeCover.lean"; then
  echo "error: SounioSatCubeCover contains an incomplete proof marker" >&2
  exit 1
fi

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" build SounioSatCubeCover
)

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT INT TERM
cat > "$WORK/check_cube_cover.lean" <<'EOF'
import SounioSatCubeCover

#check SounioSatCubeCover.Cube
#check SounioSatCubeCover.CubeCover
#check SounioSatCubeCover.colourCNFWithUnit
#check SounioSatCubeCover.colourCNFWithCube
#check SounioSatCubeCover.cubeCoverComplementCNF
#check SounioSatCubeCover.splitVerticesCubes
#check SounioSatCubeCover.unsat_of_split_vertex5
#check SounioSatCubeCover.unsat_of_cube_cover
#check SounioSatCubeCover.cube_cover_of_complement_unsat
#check SounioSatCubeCover.split_vertex5_cubes_cover
#check SounioSatCubeCover.split_vertices_cubes_cover
#check SounioSatCubeCover.unsat_of_split_vertex5_cube_cover
#print axioms SounioSatCubeCover.unsat_of_split_vertex5
#print axioms SounioSatCubeCover.unsat_of_cube_cover
#print axioms SounioSatCubeCover.cube_cover_of_complement_unsat
#print axioms SounioSatCubeCover.split_vertices_cubes_cover
#print axioms SounioSatCubeCover.unsat_of_split_vertex5_cube_cover
EOF

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" env lean "$WORK/check_cube_cover.lean" > "$WORK/check.out"
)

rg -q '^SounioSatCubeCover\.Cube' "$WORK/check.out"
rg -q '^SounioSatCubeCover\.CubeCover' "$WORK/check.out"
rg -q '^SounioSatCubeCover\.colourCNFWithUnit' "$WORK/check.out"
rg -q '^SounioSatCubeCover\.colourCNFWithCube' "$WORK/check.out"
rg -q '^SounioSatCubeCover\.cubeCoverComplementCNF' "$WORK/check.out"
rg -q '^SounioSatCubeCover\.splitVerticesCubes' "$WORK/check.out"
rg -q '^SounioSatCubeCover\.unsat_of_split_vertex5' "$WORK/check.out"
rg -q '^SounioSatCubeCover\.unsat_of_cube_cover' "$WORK/check.out"
rg -q '^SounioSatCubeCover\.cube_cover_of_complement_unsat' "$WORK/check.out"
rg -q '^SounioSatCubeCover\.split_vertex5_cubes_cover' "$WORK/check.out"
rg -q '^SounioSatCubeCover\.split_vertices_cubes_cover' "$WORK/check.out"
rg -q '^SounioSatCubeCover\.unsat_of_split_vertex5_cube_cover' "$WORK/check.out"
rg -q "depends on axioms: \\[propext, Quot.sound\\]" "$WORK/check.out"
if rg -q 'sorryAx' "$WORK/check.out"; then
  echo "error: cube-cover composition theorem depends on sorryAx" >&2
  exit 1
fi

echo "cube_cover_lean_composition_gate: PASS"
