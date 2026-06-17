#!/usr/bin/env bash
# Gate for the canonical split-product cube-batch producer.
#
# This is the front door a real chi6 cube-and-conquer search can use before
# local/cluster refutation. It emits cube rows only; proof and geometry claims
# remain downstream LRAT/Lean obligations.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
SPLIT="$ROOT/examples/erdos/cube_split_batch.py"
BATCH="$ROOT/examples/erdos/cube_sieve_batch_manifest.py"
COVER="$ROOT/examples/erdos/cube_cover_certificate.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$SPLIT"
mkdir -p "$WORK/out"

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

echo "cube_split_batch_gate: workdir=$WORK"
python3 "$SPLIT" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1.cubes" \
  --split-vertices 0,1 > "$WORK/split.out"

rg -q '^cube_split_batch v1$' "$WORK/split.out"
rg -q '^output=split_product_cube_batch$' "$WORK/split.out"
rg -q '^n=6$' "$WORK/split.out"
rg -q '^m=15$' "$WORK/split.out"
rg -q '^k=5$' "$WORK/split.out"
rg -q '^split_vertices=0,1$' "$WORK/split.out"
rg -q '^split_depth=2$' "$WORK/split.out"
rg -q '^cube_count=25$' "$WORK/split.out"
rg -q '^cube_batch_sha256=[0-9a-f]{64}$' "$WORK/split.out"
rg -q '^first_cube_id=v0_c0_v1_c0$' "$WORK/split.out"
rg -q '^last_cube_id=v0_c4_v1_c4$' "$WORK/split.out"
rg -q '^cover_route=split_vertices_atleast_one_product$' "$WORK/split.out"
rg -q '^promotable=0$' "$WORK/split.out"
rg -q '^status=cube_batch_emitted_unpromotable$' "$WORK/split.out"

rows="$(rg -c '^[A-Za-z0-9_.-]+:' "$WORK/k6_v0_v1.cubes")"
if [[ "$rows" != "25" ]]; then
  echo "error: expected 25 generated cube rows, got $rows" >&2
  exit 1
fi
rg -q '^v0_c0_v1_c0: 0:0 1:0$' "$WORK/k6_v0_v1.cubes"
rg -q '^v0_c0_v1_c4: 0:0 1:4$' "$WORK/k6_v0_v1.cubes"
rg -q '^v0_c4_v1_c4: 0:4 1:4$' "$WORK/k6_v0_v1.cubes"

python3 "$BATCH" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1.cubes" "$WORK/out/batch" \
  > "$WORK/batch.out"
rg -q '^cube_count=25$' "$WORK/batch.out"
rg -q '^conflict_count=5$' "$WORK/batch.out"
rg -q '^hard_count=20$' "$WORK/batch.out"

cat > "$WORK/fake-refute.out" <<EOF
cube_sieve_refute_batch v1
formula_kind=colourCNF
n=6
m=15
k=5
expected_vars=30
base_clause_count=81
edge_sha256=$(sha256sum "$WORK/k6.edge" | awk '{print $1}')
cube_batch_sha256=$(sha256sum "$WORK/k6_v0_v1.cubes" | awk '{print $1}')
failed_count=0
sb_mode=0
promotable=0
out_dir=$WORK/fake-refute
EOF
mkdir -p "$WORK/fake-refute"
while IFS= read -r line; do
  [[ "$line" =~ ^#|^$ ]] && continue
  id="${line%%:*}"
  rest="${line#*: }"
  mkdir -p "$WORK/fake-refute/$id"
  printf '%s\n' "$rest" | tr ' ' '\n' | sed 's/:/ /' > "$WORK/fake-refute/$id/$id.cube"
  printf 'p cnf 30 0\n' > "$WORK/fake-refute/$id/$id.cnf"
  printf '1 0 0\n' > "$WORK/fake-refute/$id/$id.lrat"
  cube_sha="$(sha256sum "$WORK/fake-refute/$id/$id.cube" | awk '{print $1}')"
  cnf_sha="$(sha256sum "$WORK/fake-refute/$id/$id.cnf" | awk '{print $1}')"
  lrat_sha="$(sha256sum "$WORK/fake-refute/$id/$id.lrat" | awk '{print $1}')"
  printf 'cube id=%s assignments=%s cube_assignment_count=2 drat_deletions=0 cnf_clauses=83 expected_cnf_clauses=83 cube=%s/%s.cube cube_sha256=%s cnf=%s/%s.cnf cnf_sha256=%s lrat=%s/%s.lrat lrat_sha256=%s\n' \
    "$id" "${rest// /,}" "$id" "$id" "$cube_sha" "$id" "$id" "$cnf_sha" "$id" "$id" "$lrat_sha" \
    >> "$WORK/fake-refute.out"
done < "$WORK/k6_v0_v1.cubes"

python3 "$COVER" "$WORK/k6.edge" 5 "$WORK/k6_v0_v1.cubes" "$WORK/fake-refute.out" \
  --cover-rule split_vertices_atleast_one_product \
  --split-vertices 0,1 > "$WORK/cover.out"
rg -q '^cover_rule=split_vertices_atleast_one_product$' "$WORK/cover.out"
rg -q '^leaf_count=25$' "$WORK/cover.out"
rg -q '^cover_complete_for_split_vertices=1$' "$WORK/cover.out"

if python3 "$SPLIT" "$WORK/k6.edge" 5 "$WORK/too_many.cubes" \
    --split-vertices 0,1,2 --max-cubes 100 > "$WORK/too_many.out" 2>&1; then
  echo "error: split producer ignored max-cubes cap" >&2
  exit 1
fi
rg -q 'would emit 125 cubes' "$WORK/too_many.out"

if python3 "$SPLIT" "$WORK/k6.edge" 5 "$WORK/bad_vertex.cubes" \
    --split-vertices 0,6 > "$WORK/bad_vertex.out" 2>&1; then
  echo "error: split producer accepted out-of-range split vertex" >&2
  exit 1
fi
rg -q 'split vertex out of range: 6' "$WORK/bad_vertex.out"

echo "cube_split_batch_gate: PASS"
