#!/usr/bin/env bash
# Gate for bounded colour-history sampling.
#
# This enumerates a deterministic set of proper 5-colourings for search
# steering only. It emits no SAT, chromatic, global UNSAT, or verified claim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

SAMPLER="$ROOT/examples/erdos/chi6_colouring_sampler.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$SAMPLER"
mkdir -p "$WORK"

cat > "$WORK/square.csv" <<'EOF'
id,x,y
0,0,0
1,1,0
2,0,1
3,1,1
EOF

echo "chi6_colouring_sampler_gate: workdir=$WORK"
python3 "$SAMPLER" "$WORK/sample" \
  --coords-csv "$WORK/square.csv" \
  --candidate-id square_sample \
  --max-colourings 4 \
  --node-limit 1000 \
  > "$WORK/sample.out"

rg -q '^chi6_colouring_sampler v1$' "$WORK/sample.out"
rg -q '^candidate_id=square_sample$' "$WORK/sample.out"
rg -q '^n=4$' "$WORK/sample.out"
rg -q '^m=4$' "$WORK/sample.out"
rg -q '^k=5$' "$WORK/sample.out"
rg -q '^colouring_count=4$' "$WORK/sample.out"
rg -q '^search_status=MAX_COLOURINGS_REACHED$' "$WORK/sample.out"
rg -q '^claim_scope=bounded_colouring_sampling_only$' "$WORK/sample.out"
rg -q '^sat_claim=none$' "$WORK/sample.out"
rg -q '^chromatic_claim=none$' "$WORK/sample.out"
rg -q '^global_unsat_claim=none$' "$WORK/sample.out"
rg -q '^verified_claim=none$' "$WORK/sample.out"
rg -q '^promotable=0$' "$WORK/sample.out"
rg -q '^status=COLOURING_SAMPLE_RECORDED$' "$WORK/sample.out"

SAMPLE_JSON="$(rg '^colouring_sampler_json=' "$WORK/sample.out" | cut -d= -f2-)"
[[ -s "$SAMPLE_JSON" ]]
python3 - "$SAMPLE_JSON" <<'PY'
import json
import sys
from pathlib import Path

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["schema"] == "chi6_colouring_sampler.v1"
assert meta["candidate_id"] == "square_sample"
assert meta["n"] == 4
assert meta["m"] == 4
assert meta["k"] == 5
assert meta["max_colourings"] == 4
assert meta["node_limit"] == 1000
assert meta["search_status"] == "MAX_COLOURINGS_REACHED"
assert meta["colouring_count"] == 4
assert meta["claim_scope"] == "bounded_colouring_sampling_only"
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["global_unsat_claim"] == "none"
assert meta["verified_claim"] == "none"
assert meta["promotable"] == 0
colourings_path = Path(meta["colourings_file"])
assert colourings_path.is_file()
lines = colourings_path.read_text(encoding="ascii").splitlines()
assert len(lines) == 4
assert len(set(lines)) == 4
edges = {(0, 1), (0, 2), (1, 3), (2, 3)}
for line in lines:
    colours = {}
    for token in line.split(","):
        v_raw, c_raw = token.split(":", 1)
        colours[int(v_raw)] = int(c_raw)
    assert sorted(colours) == [0, 1, 2, 3]
    assert all(0 <= c < 5 for c in colours.values())
    for u, v in edges:
        assert colours[u] != colours[v]
PY

if python3 "$SAMPLER" "$WORK/bad-count" \
    --coords-csv "$WORK/square.csv" \
    --max-colourings 0 \
    > "$WORK/bad-count.out" 2>&1; then
  echo "error: sampler accepted non-positive max-colourings" >&2
  exit 1
fi
rg -q -- '--max-colourings must be positive' "$WORK/bad-count.out"

if python3 "$SAMPLER" "$WORK/bad-nodes" \
    --coords-csv "$WORK/square.csv" \
    --node-limit 0 \
    > "$WORK/bad-nodes.out" 2>&1; then
  echo "error: sampler accepted non-positive node-limit" >&2
  exit 1
fi
rg -q -- '--node-limit must be positive' "$WORK/bad-nodes.out"

echo "chi6_colouring_sampler_gate: PASS"
