#!/usr/bin/env bash
# Gate for souc_sat cube-unit worker input.
#
# This is producer-side proof plumbing: a zero-based cube file is added as
# original unit clauses before `n_orig`, so the emitted CNF/DRAT/LRAT refutes
# CNF(edge,k) plus the cube units. It is not a Euclidean chi>=6 witness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

SOUC="${SOUC:-$ROOT/artifacts/self-hosted/souc-self-hosted-x86_64}"
CONV="$ROOT/examples/erdos/drup_to_lrat_rup.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
[[ -x "$SOUC" ]] || { echo "error: SOUC is not executable: $SOUC" >&2; exit 2; }
[[ -x "$CONV" ]] || { echo "error: converter is not executable: $CONV" >&2; exit 2; }

mkdir -p "$WORK"
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

cat > "$WORK/k6.cube" <<'EOF'
# zero-based vertex/colour assignments; labels are accepted.
vertex=0 colour=0
1 1
2 2
v=3 color=3
4 4
EOF

echo "souc_sat_cube_units_gate: workdir=$WORK"
"$SOUC" "$ROOT/examples/erdos/souc_sat.sio" "$WORK/souc_sat.elf"
chmod +x "$WORK/souc_sat.elf"

(
  cd "$WORK"
  ./souc_sat.elf 0 5 1 0 "$WORK/k6.edge" "$WORK/k6.cube" > souc_sat.stdout 2>&1
)

rg -q '^\[cube units=5\]$' "$WORK/souc_sat.stdout"
rg -q 'graph-file k-col cube-file vars=30 clauses=86' "$WORK/souc_sat.stdout"
rg -q 'UNSAT' "$WORK/souc_sat.stdout"
rg -q '^p cnf 30 86$' "$WORK/souc_sat_worker.cnf"
for unit in '1 0' '7 0' '13 0' '19 0' '25 0'; do
  rg -q "^$unit$" "$WORK/souc_sat_worker.cnf"
done
if rg -q '^[[:space:]]*d[[:space:]]' "$WORK/souc_sat_worker.drat"; then
  echo "error: cube smoke emitted deletion records" >&2
  exit 1
fi
python3 "$CONV" \
  "$WORK/souc_sat_worker.cnf" \
  "$WORK/souc_sat_worker.drat" \
  "$WORK/k6_cube.lrat" \
  > "$WORK/converter.stdout" 2> "$WORK/converter.stderr"
rg -q 'empty=1' "$WORK/converter.stderr"
rg -q '^[0-9]+ 0 ' "$WORK/k6_cube.lrat"

cat > "$WORK/k6_duplicate.cube" <<'EOF'
0 0
0 1
EOF
if (
  cd "$WORK"
  ./souc_sat.elf 0 5 1 0 "$WORK/k6.edge" "$WORK/k6_duplicate.cube" \
    > duplicate.stdout 2>&1
); then
  echo "error: duplicate cube assignment was accepted" >&2
  exit 1
fi
rg -q 'invalid cube file code=-6' "$WORK/duplicate.stdout"

cat > "$WORK/k6_bad.edge" <<'EOF'
p edge 6 1
e 6 7
EOF
if (
  cd "$WORK"
  ./souc_sat.elf 0 5 1 0 "$WORK/k6_bad.edge" "$WORK/k6.cube" \
    > bad_edge.stdout 2>&1
); then
  echo "error: out-of-range edge endpoint was accepted" >&2
  exit 1
fi
rg -q 'invalid edge file code=-10' "$WORK/bad_edge.stdout"

cat "$WORK/converter.stderr"
echo "souc_sat_cube_units_gate: PASS"

# X>6 extension smoke (chi>=6 / chi>=7 lane): k=6 on K6 complete (should SAT even with cube)
# This exercises the cube path for the erdos X>6 search (cube-and-conquer subproblems on
# larger unit-distance graphs or higher-k colourings for frontier candidates toward χ(ℝ²)>6).
cat > "$WORK/k6_for_xgt6.cube" <<'EOF'
# non-conflicting for k=6 on K6 (one possible proper 6-col)
0 0
1 1
2 2
3 3
4 4
5 5
EOF
if (
  cd "$WORK"
  ./souc_sat.elf 42 6 0 0 "$WORK/k6.edge" "$WORK/k6_for_xgt6.cube" > k6_xgt6.stdout 2>&1
); then
  k6_xgt6_rc=0
else
  k6_xgt6_rc=$?
fi
if [[ "$k6_xgt6_rc" != "1" ]]; then
  echo "error: expected SAT worker rc=1 for k=6 cube smoke, got $k6_xgt6_rc" >&2
  exit 1
fi
rg -q '\[cube units=6\]' "$WORK/k6_xgt6.stdout"
rg -q 'SAT' "$WORK/k6_xgt6.stdout" || rg -q 'SAT colouring' "$WORK/k6_xgt6.stdout"
echo "X>6 k=6 cube smoke: PASS (owned for erdos chi>6 frontier)"
