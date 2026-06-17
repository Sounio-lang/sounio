#!/usr/bin/env bash
# Gate for a bounded finite-graph candidate-search handoff.
#
# This is deliberately non-promotable: it proves that the search lane can emit
# absence/candidate manifests and hand candidate edge/cube artifacts into the
# existing cube tooling, not that any graph is Euclidean or chi(R^2)>=6.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi
SEARCH="$ROOT/examples/erdos/chi6_candidate_search_manifest.py"
BATCH="$ROOT/examples/erdos/cube_sieve_batch_manifest.py"
DEGREY_EDGE="$ROOT/examples/erdos/data/degrey_529.edge"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$SEARCH"
mkdir -p "$WORK/absent" "$WORK/found" "$WORK/truncated" "$WORK/external"

echo "chi6_candidate_search_manifest_gate: workdir=$WORK"
python3 "$SEARCH" "$WORK/absent" --n 5 --k 5 --max-graphs 2000 \
  > "$WORK/absent.out"
rg -q '^chi6_candidate_search_manifest v1$' "$WORK/absent.out"
rg -q '^family=all_simple_graphs$' "$WORK/absent.out"
rg -q '^n=5$' "$WORK/absent.out"
rg -q '^k=5$' "$WORK/absent.out"
rg -q '^graphs_examined=1024$' "$WORK/absent.out"
rg -q '^search_truncated=0$' "$WORK/absent.out"
rg -q '^candidate_count=0$' "$WORK/absent.out"
rg -q '^finite_graph_search_claim=untrusted_backtracking_only$' "$WORK/absent.out"
rg -q '^promotable=0$' "$WORK/absent.out"
rg -q '^status=FINITE_GRAPH_CANDIDATE_ABSENT_WITHIN_BOUND$' "$WORK/absent.out"

python3 "$SEARCH" "$WORK/truncated" --n 5 --k 5 --max-graphs 10 \
  > "$WORK/truncated.out"
rg -q '^graphs_examined=10$' "$WORK/truncated.out"
rg -q '^search_truncated=1$' "$WORK/truncated.out"
rg -q '^candidate_count=0$' "$WORK/truncated.out"
rg -q '^status=FINITE_GRAPH_SEARCH_TRUNCATED_NO_CANDIDATE$' "$WORK/truncated.out"

python3 "$SEARCH" "$WORK/found" --n 6 --k 5 --max-graphs 10 \
  --split-vertices 0,1 > "$WORK/found.out"
rg -q '^n=6$' "$WORK/found.out"
rg -q '^k=5$' "$WORK/found.out"
rg -q '^graphs_examined=1$' "$WORK/found.out"
rg -q '^candidate_count=1$' "$WORK/found.out"
rg -q '^split_vertices=0,1$' "$WORK/found.out"
rg -q '^candidate index=0 id=candidate_0000_n6_k5_m15 n=6 m=15 edge_path=.*/candidate_0000_n6_k5_m15.edge edge_sha256=[0-9a-f]{64} not_k_colourable_by_untrusted_search=1 geometry_claim=none cube_batch_path=.*/candidate_0000_n6_k5_m15.cubes cube_batch_sha256=[0-9a-f]{64} cube_count=25 cover_route=split_vertices_atleast_one_product$' "$WORK/found.out"
rg -q '^status=FINITE_GRAPH_CANDIDATE_EMITTED_UNPROMOTABLE$' "$WORK/found.out"

EDGE="$WORK/found/candidate_0000_n6_k5_m15.edge"
CUBES="$WORK/found/candidate_0000_n6_k5_m15.cubes"
[[ -s "$EDGE" ]]
[[ -s "$CUBES" ]]
rg -q '^p edge 6 15$' "$EDGE"
edge_rows="$(rg -c '^e ' "$EDGE")"
if [[ "$edge_rows" != "15" ]]; then
  echo "error: expected 15 edge rows, got $edge_rows" >&2
  exit 1
fi
cube_rows="$(rg -c '^[A-Za-z0-9_.-]+:' "$CUBES")"
if [[ "$cube_rows" != "25" ]]; then
  echo "error: expected 25 cube rows, got $cube_rows" >&2
  exit 1
fi
rg -q '^candidate_0000_n6_k5_m15_v0_c0_v1_c0: 0:0 1:0$' "$CUBES"
rg -q '^candidate_0000_n6_k5_m15_v0_c4_v1_c4: 0:4 1:4$' "$CUBES"

python3 "$BATCH" "$EDGE" 5 "$CUBES" "$WORK/found/batch" > "$WORK/found.batch.out"
rg -q '^cube_count=25$' "$WORK/found.batch.out"
rg -q '^conflict_count=5$' "$WORK/found.batch.out"
rg -q '^hard_count=20$' "$WORK/found.batch.out"

python3 "$SEARCH" "$WORK/external" --edge-file "$DEGREY_EDGE" --k 5 \
  --candidate-id degrey529_external --split-vertices 0 > "$WORK/external.out"
rg -q '^family=external_dimacs_edge$' "$WORK/external.out"
rg -q '^n=529$' "$WORK/external.out"
rg -q '^k=5$' "$WORK/external.out"
rg -q '^max_edges=2670$' "$WORK/external.out"
rg -q '^graphs_examined=1$' "$WORK/external.out"
rg -q '^candidate_count=1$' "$WORK/external.out"
rg -q '^split_vertices=0$' "$WORK/external.out"
rg -q '^finite_graph_search_claim=none_external_graph_packaging_only$' "$WORK/external.out"
rg -q '^candidate index=0 id=degrey529_external n=529 m=2670 edge_path=.*/degrey529_external.edge edge_sha256=[0-9a-f]{64} not_k_colourable_claim=none geometry_claim=none source_meta_path=.*/degrey529_external.meta.json source_meta_sha256=[0-9a-f]{64} cube_batch_path=.*/degrey529_external.cubes cube_batch_sha256=[0-9a-f]{64} cube_count=5 cover_route=split_vertices_atleast_one_product$' "$WORK/external.out"
rg -q '^status=EXTERNAL_GRAPH_PACKAGED_UNPROMOTABLE$' "$WORK/external.out"
cmp -s "$DEGREY_EDGE" "$WORK/external/degrey529_external.edge"
[[ -s "$WORK/external/degrey529_external.meta.json" ]]
python3 - "$WORK/external/degrey529_external.meta.json" "$DEGREY_EDGE" <<'PY'
import hashlib
import json
import sys

meta_path, edge_path = sys.argv[1], sys.argv[2]
meta = json.load(open(meta_path, encoding="ascii"))
assert meta["schema"] == "chi6_external_dimacs_edge_package.v1"
assert meta["candidate_id"] == "degrey529_external"
assert meta["source_edge_path"] == edge_path
assert meta["n"] == 529
assert meta["m"] == 2670
assert meta["k"] == 5
assert meta["split_vertices"] == [0]
assert meta["source_edge_sha256"] == hashlib.sha256(open(edge_path, "rb").read()).hexdigest()
assert meta["source_edge_sha256"] == meta["packaged_edge_sha256"]
assert meta["provenance_scope"] == "edge_packaging_only"
assert meta["promotion_gate"] == "requires_lrat_lean_and_exact_euclidean_geometry"
PY
external_file_count="$(find "$WORK/external" -maxdepth 1 -type f | wc -l | tr -d ' ')"
if [[ "$external_file_count" != "3" ]]; then
  echo "error: expected 3 external package files, got $external_file_count" >&2
  find "$WORK/external" -maxdepth 1 -type f -print >&2
  exit 1
fi
external_cube_rows="$(rg -c '^[A-Za-z0-9_.-]+:' "$WORK/external/degrey529_external.cubes")"
if [[ "$external_cube_rows" != "5" ]]; then
  echo "error: expected 5 external cube rows, got $external_cube_rows" >&2
  exit 1
fi
rg -q '^degrey529_external_v0_c0: 0:0$' "$WORK/external/degrey529_external.cubes"
rg -q '^degrey529_external_v0_c4: 0:4$' "$WORK/external/degrey529_external.cubes"

if python3 "$SEARCH" "$WORK/bad_n" --edge-file "$DEGREY_EDGE" --n 529 \
    > "$WORK/bad_n.out" 2>&1; then
  echo "error: external packaging accepted redundant --n" >&2
  exit 1
fi
rg -q '^error: --n must not be provided in external mode; vertex count is read from the DIMACS p edge header$' \
  "$WORK/bad_n.out"

if python3 "$SEARCH" "$WORK/bad_id" --edge-file "$DEGREY_EDGE" \
    --candidate-id 'bad/id' > "$WORK/bad_id.out" 2>&1; then
  echo "error: external packaging accepted invalid candidate id" >&2
  exit 1
fi
rg -q "^error: --candidate-id must use only letters, digits, '.', '_', or '-'$" \
  "$WORK/bad_id.out"

if python3 "$SEARCH" "$WORK/bad_cap" --edge-file "$DEGREY_EDGE" \
    --candidate-id degrey529_external --split-vertices 0,1,2 --max-cubes 100 \
    > "$WORK/bad_cap.out" 2>&1; then
  echo "error: external packaging ignored max-cubes cap" >&2
  exit 1
fi
rg -q 'split product would emit 125 cubes' "$WORK/bad_cap.out"

cat > "$WORK/bad_external.edge" <<'EOF'
p edge 3 2
e 1 2
e 2 2
EOF
if python3 "$SEARCH" "$WORK/bad_external" --edge-file "$WORK/bad_external.edge" \
    > "$WORK/bad_external.out" 2>&1; then
  echo "error: external packaging accepted malformed DIMACS edge file" >&2
  exit 1
fi
rg -q 'malformed edge 2 2' "$WORK/bad_external.out"

cat > "$WORK/empty_external.edge" <<'EOF'
p edge 3 0
EOF
if python3 "$SEARCH" "$WORK/empty_external" --edge-file "$WORK/empty_external.edge" \
    > "$WORK/empty_external.out" 2>&1; then
  echo "error: external packaging accepted empty edge file" >&2
  exit 1
fi
rg -q '^error: --edge-file must contain at least one edge$' "$WORK/empty_external.out"

if python3 "$SEARCH" "$WORK/bad" --n 6 --k 5 --split-vertices 0,6 \
    > "$WORK/bad.out" 2>&1; then
  echo "error: search manifest accepted an out-of-range split vertex" >&2
  exit 1
fi
rg -q 'split vertex out of range: 6' "$WORK/bad.out"

echo "chi6_candidate_search_manifest_gate: PASS"
