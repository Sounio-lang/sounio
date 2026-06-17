#!/usr/bin/env bash
# Gate for arbitrary cube-cover composition via a complement-cover LRAT proof.
#
# The cube family below happens to be the small K6 split-product smoke, but this
# gate deliberately does not use the split-product theorem. Instead it emits
# `base ∧ cube-blockers`, converts a tiny RUP proof of that complement to LRAT,
# and lets generated Lean prove `CubeCover` via `cube_cover_of_complement_unsat`.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEAN_DIR="$ROOT/formal/lean4"
LOCK="$ROOT/scripts/dev/souc-build-lock.sh"
LAKE="${LAKE:-/workspace/.home/openvscode-server/.elan/bin/lake}"
REFUTER="$ROOT/examples/erdos/cube_sieve_refute_batch.py"
COMP_CNF="$ROOT/examples/erdos/cube_cover_complement_cnf.py"
CONVERTER="$ROOT/examples/erdos/drup_to_lrat_rup.py"
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
python3 -m py_compile "$REFUTER" "$COMP_CNF" "$CONVERTER" "$GEN"

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

echo "cube_cover_arbitrary_complement_lean_reflect_pipeline_gate: workdir=$WORK"
python3 "$REFUTER" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1_cover.cubes" "$WORK/refute" \
  > "$WORK/refute.out"
python3 "$COMP_CNF" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1_cover.cubes" "$WORK/cover_complement.cnf" \
  > "$WORK/cover_complement.out"

rg -q '^cube_cover_complement_cnf v1$' "$WORK/cover_complement.out"
rg -q '^cube_count=25$' "$WORK/cover_complement.out"
rg -q '^clause_count=106$' "$WORK/cover_complement.out"

cat > "$WORK/cover_complement.drup" <<'EOF'
-1 0
-2 0
-3 0
-4 0
-5 0
0
EOF

python3 "$CONVERTER" "$WORK/cover_complement.cnf" "$WORK/cover_complement.drup" \
  "$WORK/cover_complement.lrat" > "$WORK/cover_lrat.out" 2> "$WORK/cover_lrat.err"
rg -q 'empty=1' "$WORK/cover_lrat.err"

python3 "$GEN" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1_cover.cubes" "$WORK/refute.out" \
  "$WORK/SounioSatK65ArbitraryCoverReflect.lean" \
  --module SounioSatK65ArbitraryCoverReflect \
  --prefix k65arb \
  --composition arbitrary \
  --cover-cnf "$WORK/cover_complement.cnf" \
  --cover-lrat "$WORK/cover_complement.lrat" \
  > "$WORK/gen.out"

rg -q '^composition=arbitrary$' "$WORK/gen.out"
rg -q '^cover_claim=base_plus_cube_blockers_unsat$' "$WORK/gen.out"
rg -q '^theorem k65arb_unsat_from_arbitrary_cube_cover' \
  "$WORK/SounioSatK65ArbitraryCoverReflect.lean"
rg -q 'SounioSatCubeCover.cubeCoverComplementCNF' \
  "$WORK/SounioSatK65ArbitraryCoverReflect.lean"
rg -q 'SounioSatCubeCover.cube_cover_of_complement_unsat' \
  "$WORK/SounioSatK65ArbitraryCoverReflect.lean"
rg -q 'SounioSatCubeCover.unsat_of_cube_cover' \
  "$WORK/SounioSatK65ArbitraryCoverReflect.lean"

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" env lean "$WORK/SounioSatK65ArbitraryCoverReflect.lean" \
    > "$WORK/lean_build.log" 2>&1
)
if rg -q 'sorryAx' "$WORK/lean_build.log"; then
  echo "error: arbitrary cover generated Lean depends on sorryAx" >&2
  exit 1
fi

if python3 "$GEN" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1_cover.cubes" "$WORK/refute.out" \
    "$WORK/bad.lean" --composition arbitrary > "$WORK/bad.out" 2>&1; then
  echo "error: arbitrary composition accepted missing complement proof" >&2
  exit 1
fi
rg -q -- '--composition arbitrary requires --cover-cnf and --cover-lrat' "$WORK/bad.out"

echo "cube_cover_arbitrary_complement_lean_reflect_pipeline_gate: PASS"
