#!/usr/bin/env bash
# Gate for the smallest finite cube-cover certificate smoke.
#
# K6/k=5 is used only as a finite SAT calibration target. Five one-literal cubes
# covering vertex 0's colours are enough to cover the base colourCNF search space
# by the at-least-one colour clause for vertex 0.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
REFUTER="$ROOT/examples/erdos/cube_sieve_refute_batch.py"
COVER="$ROOT/examples/erdos/cube_cover_certificate.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
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

echo "cube_cover_certificate_gate: workdir=$WORK"
python3 "$REFUTER" "$WORK/k6.edge" 5 "$WORK/k6_cover.cubes" "$WORK/refute" \
  > "$WORK/refute.out"
python3 "$COVER" "$WORK/k6.edge" 5 "$WORK/k6_cover.cubes" "$WORK/refute.out" \
  > "$WORK/cover.out"

rg -q '^cube_cover_certificate v1$' "$WORK/cover.out"
rg -q '^cover_rule=single_vertex_atleast_one_split$' "$WORK/cover.out"
rg -q '^formula_kind=colourCNF$' "$WORK/cover.out"
rg -q '^n=6$' "$WORK/cover.out"
rg -q '^m=15$' "$WORK/cover.out"
rg -q '^k=5$' "$WORK/cover.out"
rg -q '^split_vertex=0$' "$WORK/cover.out"
rg -q '^base_clause=atleast_one_colour_for_split_vertex$' "$WORK/cover.out"
rg -q '^refutation_batch_sha256=[0-9a-f]{64}$' "$WORK/cover.out"
rg -q '^leaf_count=5$' "$WORK/cover.out"
rg -q '^covered_cube_count=5$' "$WORK/cover.out"
rg -q '^lrat_artifact_count=5$' "$WORK/cover.out"
rg -q '^cover_complete_for_split_vertex=1$' "$WORK/cover.out"
rg -q '^cover_claim=atleast_one_cover_for_split_vertex$' "$WORK/cover.out"
rg -q '^verified_claim=none$' "$WORK/cover.out"
rg -q '^global_unsat_claim=none$' "$WORK/cover.out"
rg -q '^geometry_claim=none$' "$WORK/cover.out"
rg -q '^promotion_gate=REJECT_LEAN_CHECKED_LEAF_UNSAT_NOT_ATTACHED$' "$WORK/cover.out"
rg -q '^promotable=0$' "$WORK/cover.out"
for c in 0 1 2 3 4; do
  rg -q "^leaf index=$c colour=$c cube_id=v0_c$c assignment=0:$c cube_sha256=[0-9a-f]{64} lrat_sha256=[0-9a-f]{64}$" \
    "$WORK/cover.out"
done
rg -q '^status=cover_certificate_emitted_unpromotable$' "$WORK/cover.out"

cat > "$WORK/k6_missing.cubes" <<'EOF'
v0_c0: 0:0
v0_c1: 0:1
v0_c2: 0:2
v0_c3: 0:3
EOF
if python3 "$COVER" "$WORK/k6.edge" 5 "$WORK/k6_missing.cubes" "$WORK/refute.out" \
    > "$WORK/missing-cover.out" 2>&1; then
  echo "error: cover accepted a missing colour leaf" >&2
  exit 1
fi
rg -q 'refutation summary cube_batch_sha256 mismatch' "$WORK/missing-cover.out"

sed 's/assignments=0:4/assignments=1:4/' "$WORK/refute.out" > "$WORK/bad-refute.out"
if python3 "$COVER" "$WORK/k6.edge" 5 "$WORK/k6_cover.cubes" "$WORK/bad-refute.out" \
    > "$WORK/bad-refute-cover.out" 2>&1; then
  echo "error: cover accepted a mismatched refutation row" >&2
  exit 1
fi
rg -q 'wrong assignments' "$WORK/bad-refute-cover.out"

echo "cube_cover_certificate_gate: PASS"
