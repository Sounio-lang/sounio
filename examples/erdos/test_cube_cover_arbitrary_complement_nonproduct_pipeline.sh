#!/usr/bin/env bash
# Arbitrary cube-cover smoke with a non-product-shaped cube list.
#
# The first five singleton cubes cover by the at-least-one clause for v0. The
# extra 125 three-vertex cubes are deliberately redundant for coverage, but they
# force the arbitrary generator to replay and dispatch a larger cube list that
# is not equal to a split-product family. This is a Lean/LRAT composition gate,
# not a search-hardness benchmark and not Euclidean geometry evidence.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
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

WORK="$(mktemp -d)" || { echo "error: failed to create temp dir" >&2; exit 1; }
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

python3 - "$WORK/k6_nonproduct_cover.cubes" <<'PY'
from pathlib import Path
import sys

out = Path(sys.argv[1])
with out.open("w", encoding="ascii") as f:
    for c0 in range(5):
        f.write(f"v0_c{c0}: 0:{c0}\n")
    for c0 in range(5):
        for c1 in range(5):
            for c2 in range(5):
                f.write(
                    f"v0_c{c0}_v1_c{c1}_v2_c{c2}: "
                    f"0:{c0} 1:{c1} 2:{c2}\n"
                )
PY

line_count="$(wc -l < "$WORK/k6_nonproduct_cover.cubes" | tr -d ' ')"
if [[ "$line_count" != 130 ]]; then
  echo "error: expected 130 cubes, got $line_count" >&2
  exit 1
fi

echo "cube_cover_arbitrary_complement_nonproduct_pipeline_gate: workdir=$WORK"
python3 "$REFUTER" "$WORK/k6.edge" 5 "$WORK/k6_nonproduct_cover.cubes" "$WORK/refute" \
  > "$WORK/refute.out"
python3 "$COMP_CNF" "$WORK/k6.edge" 5 "$WORK/k6_nonproduct_cover.cubes" "$WORK/cover_complement.cnf" \
  > "$WORK/cover_complement.out"

rg -q '^cube_count=130$' "$WORK/refute.out"
rg -q '^solver_unsat_count=130$' "$WORK/refute.out"
rg -q '^lrat_artifact_count=130$' "$WORK/refute.out"
rg -q '^failed_count=0$' "$WORK/refute.out"
refute_cube_lines="$(rg -c '^cube index=' "$WORK/refute.out")"
if [[ "$refute_cube_lines" != 130 ]]; then
  echo "error: expected 130 cube refutation rows, got $refute_cube_lines" >&2
  exit 1
fi
rg -q '^cube index=0 .*cube_assignment_count=1' "$WORK/refute.out"
rg -q '^cube index=5 .*cube_assignment_count=3' "$WORK/refute.out"
rg -q '^cube index=129 ' "$WORK/refute.out"
rg -q 'cube_assignment_count=1' "$WORK/refute.out"
rg -q 'cube_assignment_count=3' "$WORK/refute.out"
rg -q '^cube_count=130$' "$WORK/cover_complement.out"
rg -q '^clause_count=211$' "$WORK/cover_complement.out"

printf '0\n' > "$WORK/cover_complement.drup"
python3 "$CONVERTER" "$WORK/cover_complement.cnf" "$WORK/cover_complement.drup" \
  "$WORK/cover_complement.lrat" > "$WORK/cover_lrat.out" 2> "$WORK/cover_lrat.err"
[[ -s "$WORK/cover_complement.lrat" ]]
rg -q 'original=211' "$WORK/cover_lrat.err"
rg -q 'additions=1' "$WORK/cover_lrat.err"
rg -q 'empty=1' "$WORK/cover_lrat.err"

python3 "$GEN" "$WORK/k6.edge" 5 "$WORK/k6_nonproduct_cover.cubes" "$WORK/refute.out" \
  "$WORK/SounioSatK65ArbitraryNonProduct130CoverReflect.lean" \
  --module SounioSatK65ArbitraryNonProduct130CoverReflect \
  --prefix k65arbnp130 \
  --composition arbitrary \
  --cover-cnf "$WORK/cover_complement.cnf" \
  --cover-lrat "$WORK/cover_complement.lrat" \
  > "$WORK/gen.out"

rg -q '^leaf_count=130$' "$WORK/gen.out"
rg -q '^composition=arbitrary$' "$WORK/gen.out"
rg -q '^cover_claim=base_plus_cube_blockers_unsat$' "$WORK/gen.out"
rg -q '^theorem k65arbnp130_unsat_from_arbitrary_cube_cover' \
  "$WORK/SounioSatK65ArbitraryNonProduct130CoverReflect.lean"
rg -q '^theorem k65arbnp130_leaf129_check' \
  "$WORK/SounioSatK65ArbitraryNonProduct130CoverReflect.lean"
leaf_check_count="$(
  rg -c '^theorem k65arbnp130_leaf[0-9]+_check' \
    "$WORK/SounioSatK65ArbitraryNonProduct130CoverReflect.lean"
)"
if [[ "$leaf_check_count" != 130 ]]; then
  echo "error: expected 130 generated leaf checks, got $leaf_check_count" >&2
  exit 1
fi
rg -q '^set_option maxHeartbeats 0$' "$WORK/SounioSatK65ArbitraryNonProduct130CoverReflect.lean"
rg -q 'SounioSatCubeCover.cubeCoverComplementCNF' \
  "$WORK/SounioSatK65ArbitraryNonProduct130CoverReflect.lean"
rg -q 'SounioSatCubeCover.cube_cover_of_complement_unsat' \
  "$WORK/SounioSatK65ArbitraryNonProduct130CoverReflect.lean"
rg -q 'SounioSatCubeCover.unsat_of_cube_cover' \
  "$WORK/SounioSatK65ArbitraryNonProduct130CoverReflect.lean"
if rg -q '\b(sorry|admit)\b' "$WORK/SounioSatK65ArbitraryNonProduct130CoverReflect.lean"; then
  echo "error: non-product arbitrary cover generated Lean contains incomplete proof marker" >&2
  exit 1
fi

(
  cd "$LEAN_DIR"
  "$LOCK" "$LAKE" env lean "$WORK/SounioSatK65ArbitraryNonProduct130CoverReflect.lean" \
    > "$WORK/lean_build.log" 2>&1
)
if rg -q 'sorryAx' "$WORK/lean_build.log"; then
  echo "error: non-product arbitrary cover generated Lean depends on sorryAx" >&2
  exit 1
fi

echo "cube_cover_arbitrary_complement_nonproduct_pipeline_gate: PASS"
