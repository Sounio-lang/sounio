#!/usr/bin/env bash
# Gate for the data-driven cube-propagation manifest producer.
#
# This is still producer-side search plumbing. It proves that a DIMACS graph plus
# cube can emit replayable propagation metadata; it does not certify SAT UNSAT or
# any Euclidean geometry claim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
PRODUCER="$ROOT/examples/erdos/cube_sieve_propagation_manifest.py"
VALIDATOR="$ROOT/examples/erdos/validate_cube_sieve_manifest.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
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
0 0
1 1
2 2
3 3
4 4
EOF

echo "cube_sieve_propagation_manifest_gate: workdir=$WORK"
python3 "$PRODUCER" "$WORK/k6.edge" 5 "$WORK/k6.cube" > "$WORK/k6.out"
python3 "$VALIDATOR" "$WORK/k6.out" > "$WORK/k6.validator.log"
rg -q '^cube_sieve_propagation_manifest v1$' "$WORK/k6.out"
rg -q '^section=dimacs_cube_propagation$' "$WORK/k6.out"
rg -q '^  graph_family=dimacs_edge$' "$WORK/k6.out"
rg -q '^  n=6$' "$WORK/k6.out"
rg -q '^  m=15$' "$WORK/k6.out"
rg -q '^  k=5$' "$WORK/k6.out"
rg -q '^  cube_assignment_count=5$' "$WORK/k6.out"
rg -q '^  trail_len=5$' "$WORK/k6.out"
rg -q '^  conflict=1$' "$WORK/k6.out"
rg -q '^  conflict_vertex=5$' "$WORK/k6.out"
rg -q '^  hard_cube=0$' "$WORK/k6.out"
rg -q '^  final_domains=1,2,4,8,16,0$' "$WORK/k6.out"
rg -q '^promotable=0$' "$WORK/k6.out"

cat > "$WORK/path3.edge" <<'EOF'
p edge 3 2
e 1 2
e 2 3
EOF

cat > "$WORK/path3.cube" <<'EOF'
0 0
EOF

python3 "$PRODUCER" "$WORK/path3.edge" 3 "$WORK/path3.cube" > "$WORK/path3.out"
python3 "$VALIDATOR" "$WORK/path3.out" > "$WORK/path3.validator.log"
rg -q '^  graph_family=dimacs_edge$' "$WORK/path3.out"
rg -q '^  n=3$' "$WORK/path3.out"
rg -q '^  m=2$' "$WORK/path3.out"
rg -q '^  k=3$' "$WORK/path3.out"
rg -q '^  trail_len=1$' "$WORK/path3.out"
rg -q '^  conflict=0$' "$WORK/path3.out"
rg -q '^  conflict_vertex=-1$' "$WORK/path3.out"
rg -q '^  hard_cube=1$' "$WORK/path3.out"
rg -q '^  final_domains=1,6,7$' "$WORK/path3.out"

sed 's/^    edge 0 1$/    edge 0 0/' "$WORK/path3.out" > "$WORK/bad_edge.out"
if python3 "$VALIDATOR" "$WORK/bad_edge.out" > "$WORK/bad_edge.validator.log" 2>&1; then
  echo "error: validator accepted a self-loop edge row" >&2
  exit 1
fi
rg -q 'bad edge row' "$WORK/bad_edge.validator.log"

sed 's/^  hard_cube=1$/  hard_cube=0/' "$WORK/path3.out" > "$WORK/bad_hard_cube.out"
if python3 "$VALIDATOR" "$WORK/bad_hard_cube.out" > "$WORK/bad_hard_cube.validator.log" 2>&1; then
  echo "error: validator accepted a wrong hard_cube summary" >&2
  exit 1
fi
rg -q 'bad hard_cube' "$WORK/bad_hard_cube.validator.log"

cat "$WORK/k6.validator.log"
cat "$WORK/path3.validator.log"
echo "cube_sieve_propagation_manifest_gate: PASS"
