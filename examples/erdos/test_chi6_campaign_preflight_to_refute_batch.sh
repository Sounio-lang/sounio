#!/usr/bin/env bash
# Gate the single-preflight -> refute-attempt-batch adapter.
#
# K6 is used only as finite SAT plumbing calibration. The adapter emits a
# machine-safe refute batch and the refute attempt can emit leaf LRATs, but no
# cube-cover/global UNSAT/Euclidean chromatic claim is promoted here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

BRIDGE="$ROOT/examples/erdos/chi6_campaign_preflight_to_refute_batch.py"
ATTEMPT="$ROOT/examples/erdos/chi6_frontier_refute_attempt.py"
REFUTER="$ROOT/examples/erdos/cube_sieve_refute_batch.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$BRIDGE" "$ATTEMPT" "$REFUTER"
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

cat > "$WORK/k6.cubes" <<'EOF'
conflict: 0:0 1:1 2:2 3:3 4:4
small: 0:0
EOF

python3 - "$WORK/k6.preflight.json" "$WORK/k6.edge" "$WORK/k6.cubes" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

dst, edge, cubes = map(Path, sys.argv[1:])

def sha(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()

meta = {
    "schema": "chi6_frontier_campaign_preflight.v1",
    "candidate_id": "k6_ready",
    "campaign_id": "chi6camp_k6_ready",
    "campaign_mode": "split_product_frontier_preflight",
    "frontier_scout_path": "NONE",
    "frontier_scout_sha256": "NONE",
    "candidate_source_path": "NONE",
    "candidate_source_sha256": "NONE",
    "source_status": "PASS",
    "n": 6,
    "m": 15,
    "k": 5,
    "edge_path_abs": str(edge),
    "edge_sha256": sha(edge),
    "cube_count": 2,
    "cube_batch_path": str(cubes),
    "cube_batch_sha256": sha(cubes),
    "propagation_conflict_count": 0,
    "propagation_hard_count": 2,
    "recommended_next_action": "prepare_cube_refute_batch",
    "recommended_next_gate": "prepare_cube_refute_batch",
    "foundry_handoff_recommended": 1,
    "claim_scope": "deterministic_campaign_preflight_only",
    "geometry_claim": "none",
    "sat_claim": "none",
    "chromatic_claim": "none",
    "global_unsat_claim": "none",
    "verified_claim": "none",
    "promotable": 0,
}
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
print()
PY

echo "chi6_campaign_preflight_to_refute_batch_gate: workdir=$WORK"
python3 "$BRIDGE" "$WORK/k6.preflight.json" "$WORK/bridge" > "$WORK/bridge.out"

rg -q '^chi6_campaign_preflight_to_refute_batch v1$' "$WORK/bridge.out"
rg -q '^candidate_id=k6_ready$' "$WORK/bridge.out"
rg -q '^refute_ready_count=1$' "$WORK/bridge.out"
rg -q '^claim_scope=frontier_campaign_preflight_batch_only$' "$WORK/bridge.out"
rg -q '^sat_claim=none$' "$WORK/bridge.out"
rg -q '^chromatic_claim=none$' "$WORK/bridge.out"
rg -q '^global_unsat_claim=none$' "$WORK/bridge.out"
rg -q '^verified_claim=none$' "$WORK/bridge.out"
rg -q '^promotable=0$' "$WORK/bridge.out"
rg -q '^status=CAMPAIGN_PREFLIGHT_REFUTE_BATCH_READY$' "$WORK/bridge.out"

BATCH_JSON="$(rg '^preflight_batch_json=' "$WORK/bridge.out" | cut -d= -f2-)"
[[ -s "$BATCH_JSON" ]]
python3 - "$BATCH_JSON" "$REFUTER" <<'PY'
import json
import sys
from pathlib import Path

meta = json.load(open(sys.argv[1], encoding="ascii"))
refuter = str(Path(sys.argv[2]).resolve())
assert meta["schema"] == "chi6_frontier_campaign_preflight_batch.v1"
assert meta["claim_scope"] == "frontier_campaign_preflight_batch_only"
assert meta["refute_ready_count"] == 1
assert meta["first_refute_candidate"] == "k6_ready"
assert meta["first_refute_argv"][1] == refuter
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["global_unsat_claim"] == "none"
assert meta["verified_claim"] == "none"
assert meta["promotable"] == 0
row = meta["preflights"][0]
assert row["candidate_id"] == "k6_ready"
assert row["recommended_next_action"] == "prepare_cube_refute_batch"
assert row["refute_argv"][1] == refuter
assert row["cube_count"] == 2
assert row["propagation_hard_count"] == 2
PY

python3 "$ATTEMPT" "$BATCH_JSON" "$WORK/attempt" > "$WORK/attempt.out"
rg -q '^chi6_frontier_refute_attempt v1$' "$WORK/attempt.out"
rg -q '^attempt_count=1$' "$WORK/attempt.out"
rg -q '^refute_success_count=1$' "$WORK/attempt.out"
rg -q '^refute_failed_count=0$' "$WORK/attempt.out"
rg -q '^first_success_candidate=k6_ready$' "$WORK/attempt.out"
rg -q '^claim_scope=frontier_refute_attempt_only$' "$WORK/attempt.out"
rg -q '^promotable=0$' "$WORK/attempt.out"
rg -q '^status=FRONTIER_REFUTE_ATTEMPT_RECORDED$' "$WORK/attempt.out"

ATTEMPT_JSON="$(rg '^refute_attempt_json=' "$WORK/attempt.out" | cut -d= -f2-)"
python3 - "$ATTEMPT_JSON" <<'PY'
import json
import sys

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["status_counts"] == {"REFUTE_SUCCESS_UNPROMOTABLE": 1}
row = meta["attempts"][0]
assert row["classified_status"] == "REFUTE_SUCCESS_UNPROMOTABLE"
assert row["cube_count"] == 2
assert row["solver_unsat_count"] == 2
assert row["lrat_artifact_count_on_disk"] == 2
assert row["verified_claim"] == "none"
assert row["global_unsat_claim"] == "none"
assert row["promotable"] == "0"
PY

python3 - "$WORK/k6.preflight.json" "$WORK/not-ready.json" <<'PY'
import json
import sys

src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["recommended_next_action"] = "reject_or_mutate_frontier"
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
if python3 "$BRIDGE" "$WORK/not-ready.json" "$WORK/not-ready" \
    > "$WORK/not-ready.out" 2>&1; then
  echo "error: bridge accepted a non-refute-ready preflight" >&2
  exit 1
fi
rg -q 'recommended_next_action must be prepare_cube_refute_batch' "$WORK/not-ready.out"

echo "chi6_campaign_preflight_to_refute_batch_gate: PASS"
