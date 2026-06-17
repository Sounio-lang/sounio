#!/usr/bin/env bash
# Larger arbitrary cube-cover smoke: K6, k=5, 125 three-vertex cubes.
#
# This deliberately sends a product-shaped fixture through the arbitrary
# complement-cover path, not through the split-product theorem. It is a
# generated-Lean/LRAT volume gate, not a search-hardness benchmark.
#
# In the complement proof, the binary blockers for v0/v1 are not assumptions:
# each one is a RUP consequence of v2's at-least-one clause and the five ternary
# cube blockers for that v0/v1 pair. The proof then derives unary blockers for
# v0 and finally the empty clause.
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

python3 - "$WORK/k6_v0_v1_v2_cover.cubes" <<'PY'
from pathlib import Path
import sys

out = Path(sys.argv[1])
with out.open("w", encoding="ascii") as f:
    for c0 in range(5):
        for c1 in range(5):
            for c2 in range(5):
                f.write(
                    f"v0_c{c0}_v1_c{c1}_v2_c{c2}: "
                    f"0:{c0} 1:{c1} 2:{c2}\n"
                )
PY

echo "cube_cover_arbitrary_complement_scale_pipeline_gate: workdir=$WORK"
python3 "$REFUTER" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1_v2_cover.cubes" "$WORK/refute" \
  > "$WORK/refute.out"
python3 "$COMP_CNF" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1_v2_cover.cubes" "$WORK/cover_complement.cnf" \
  > "$WORK/cover_complement.out"

rg -q '^cube_count=125$' "$WORK/refute.out"
rg -q '^solver_unsat_count=125$' "$WORK/refute.out"
rg -q '^lrat_artifact_count=125$' "$WORK/refute.out"
rg -q '^failed_count=0$' "$WORK/refute.out"
rg -q '^cube_count=125$' "$WORK/cover_complement.out"
rg -q '^clause_count=206$' "$WORK/cover_complement.out"

python3 - "$WORK/cover_complement.drup" <<'PY'
from pathlib import Path
import sys

def lit(v: int, c: int, k: int = 5) -> int:
    return v * k + c + 1

out = Path(sys.argv[1])
with out.open("w", encoding="ascii") as f:
    for c0 in range(5):
        for c1 in range(5):
            f.write(f"-{lit(0, c0)} -{lit(1, c1)} 0\n")
    for c0 in range(5):
        f.write(f"-{lit(0, c0)} 0\n")
    f.write("0\n")
PY

python3 "$CONVERTER" "$WORK/cover_complement.cnf" "$WORK/cover_complement.drup" \
  "$WORK/cover_complement.lrat" > "$WORK/cover_lrat.out" 2> "$WORK/cover_lrat.err"
[[ -s "$WORK/cover_complement.lrat" ]]
rg -q 'original=206' "$WORK/cover_lrat.err"
rg -q 'additions=31' "$WORK/cover_lrat.err"
rg -q 'empty=1' "$WORK/cover_lrat.err"

python3 "$GEN" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1_v2_cover.cubes" "$WORK/refute.out" \
  "$WORK/SounioSatK65Arbitrary125CoverReflect.lean" \
  --module SounioSatK65Arbitrary125CoverReflect \
  --prefix k65arb125 \
  --composition arbitrary \
  --cover-cnf "$WORK/cover_complement.cnf" \
  --cover-lrat "$WORK/cover_complement.lrat" \
  --max-lrat-bytes 0 \
  > "$WORK/gen.out"

rg -q '^leaf_count=125$' "$WORK/gen.out"
rg -q '^composition=arbitrary$' "$WORK/gen.out"
rg -q '^cover_claim=base_plus_cube_blockers_unsat$' "$WORK/gen.out"
rg -q '^theorem k65arb125_unsat_from_arbitrary_cube_cover' \
  "$WORK/SounioSatK65Arbitrary125CoverReflect.lean"
rg -q '^set_option maxHeartbeats 0$' "$WORK/SounioSatK65Arbitrary125CoverReflect.lean"
rg -q 'SounioSatCubeCover.cubeCoverComplementCNF' \
  "$WORK/SounioSatK65Arbitrary125CoverReflect.lean"
rg -q 'SounioSatCubeCover.cube_cover_of_complement_unsat' \
  "$WORK/SounioSatK65Arbitrary125CoverReflect.lean"
rg -q 'SounioSatCubeCover.unsat_of_cube_cover' \
  "$WORK/SounioSatK65Arbitrary125CoverReflect.lean"
if rg -q '\b(sorry|admit)\b' "$WORK/SounioSatK65Arbitrary125CoverReflect.lean"; then
  echo "error: scale arbitrary cover generated Lean contains incomplete proof marker" >&2
  exit 1
fi

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" env lean "$WORK/SounioSatK65Arbitrary125CoverReflect.lean" \
    > "$WORK/lean_build.log" 2>&1
)
if rg -q 'sorryAx' "$WORK/lean_build.log"; then
  echo "error: scale arbitrary cover generated Lean depends on sorryAx" >&2
  exit 1
fi

echo "cube_cover_arbitrary_complement_scale_pipeline_gate: PASS"
