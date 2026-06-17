#!/usr/bin/env bash
# Gate campaign.json -> per-frontier cube-campaign preflight queueing.
#
# This is still queue plumbing: it selects ranked exact-rational scout outputs,
# runs deterministic cube preflight, and exposes refute commands without running
# SAT/LRAT refutation or claiming a chromatic lower bound.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

CAMPAIGN="$ROOT/examples/erdos/chi6_rational_frontier_campaign.py"
BATCH="$ROOT/examples/erdos/chi6_frontier_campaign_preflight_batch.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$CAMPAIGN" "$BATCH"
mkdir -p "$WORK"

echo "chi6_frontier_campaign_preflight_batch_gate: workdir=$WORK"
python3 "$CAMPAIGN" "$WORK/campaign" \
  --candidate-prefix batch \
  --max-den-list 1,5 \
  --layers-list 1 \
  --max-points-list 16 \
  --split-depth 1 \
  --dsatur-node-limit 1 \
  > "$WORK/campaign.out"

rg -q '^status=RATIONAL_FRONTIER_CAMPAIGN_READY$' "$WORK/campaign.out"
rg -q '^candidate rank=0 .* dsatur_status=UNKNOWN_NODE_LIMIT ' "$WORK/campaign.out"

python3 "$BATCH" "$WORK/campaign/campaign.json" "$WORK/preflight-batch" --limit 1 \
  > "$WORK/preflight-batch.out"

rg -q '^chi6_frontier_campaign_preflight_batch v1$' "$WORK/preflight-batch.out"
rg -q '^selected_count=1$' "$WORK/preflight-batch.out"
rg -q '^skipped_count=1$' "$WORK/preflight-batch.out"
rg -q '^refute_ready_count=1$' "$WORK/preflight-batch.out"
rg -q '^first_refute_candidate=batch_d5_l1_p16_001$' "$WORK/preflight-batch.out"
rg -q '^first_refute_command=python3 .*/cube_sieve_refute_batch\.py .*$' "$WORK/preflight-batch.out"
rg -q '^first_refute_argv_json=\[".*/python3",".*/cube_sieve_refute_batch.py",.*\]$' "$WORK/preflight-batch.out"
rg -q '^claim_scope=frontier_campaign_preflight_batch_only$' "$WORK/preflight-batch.out"
rg -q '^sat_claim=none$' "$WORK/preflight-batch.out"
rg -q '^chromatic_claim=none$' "$WORK/preflight-batch.out"
rg -q '^global_unsat_claim=none$' "$WORK/preflight-batch.out"
rg -q '^verified_claim=none$' "$WORK/preflight-batch.out"
rg -q '^promotable=0$' "$WORK/preflight-batch.out"
rg -q '^status=FRONTIER_CAMPAIGN_PREFLIGHT_BATCH_READY$' "$WORK/preflight-batch.out"

MANIFEST="$(rg '^preflight_batch_json=' "$WORK/preflight-batch.out" | cut -d= -f2-)"
[[ -s "$MANIFEST" ]]
python3 - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["schema"] == "chi6_frontier_campaign_preflight_batch.v1"
assert meta["campaign_count"] == 2
assert meta["selected_count"] == 1
assert meta["skipped_count"] == 1
assert meta["skipped"] == [
    {"rank": 1, "candidate_id": "batch_d1_l1_p16_000", "reason": "limit_reached"}
]
assert meta["refute_ready_count"] == 1
assert meta["first_refute_candidate"] == "batch_d5_l1_p16_001"
assert "cube_sieve_refute_batch.py" in meta["first_refute_command"]
assert meta["first_refute_argv"][1].endswith("cube_sieve_refute_batch.py")
assert meta["action_counts"] == {"prepare_cube_refute_batch": 1}
assert meta["claim_scope"] == "frontier_campaign_preflight_batch_only"
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["global_unsat_claim"] == "none"
assert meta["verified_claim"] == "none"
assert meta["promotable"] == 0
rows = meta["preflights"]
assert len(rows) == 1
row = rows[0]
assert row["rank"] == 0
assert row["candidate_id"] == "batch_d5_l1_p16_001"
assert row["dsatur_status"] == "UNKNOWN_NODE_LIMIT"
assert row["recommended_next_action"] == "prepare_cube_refute_batch"
assert row["refute_argv"] == meta["first_refute_argv"]
assert row["cube_count"] == 5
assert row["propagation_hard_count"] == 5
assert row["propagation_conflict_count"] == 0
assert row["foundry_handoff_recommended"] == 1
assert Path(row["frontier_scout"]).is_file()
assert Path(row["preflight_stdout"]).is_file()
assert Path(row["preflight_stderr"]).is_file()
assert row["preflight_stderr_nonempty"] in (0, 1)
assert isinstance(row["preflight_stderr_excerpt"], str)
assert Path(row["campaign_preflight_json"]).is_file()
PY

python3 "$BATCH" "$WORK/campaign/campaign.json" "$WORK/preflight-batch-two" --limit 2 \
  > "$WORK/preflight-batch-two.out"
rg -q '^selected_count=2$' "$WORK/preflight-batch-two.out"
rg -q '^refute_ready_count=2$' "$WORK/preflight-batch-two.out"

if python3 "$BATCH" "$WORK/campaign/campaign.json" "$WORK/bad-limit" --limit 0 \
    > "$WORK/bad-limit.out" 2>&1; then
  echo "error: campaign preflight batch accepted zero limit" >&2
  exit 1
fi
rg -q -- '--limit must be positive' "$WORK/bad-limit.out"

if python3 "$BATCH" "$WORK/campaign/campaign.json" "$WORK/bad-sample" \
    --max-cubes 4 --sample-hard-cubes 5 > "$WORK/bad-sample.out" 2>&1; then
  echo "error: campaign preflight batch accepted sample-hard-cubes > max-cubes" >&2
  exit 1
fi
rg -q -- '--sample-hard-cubes cannot exceed --max-cubes' "$WORK/bad-sample.out"

python3 - "$WORK/campaign/campaign.json" "$WORK/bad-campaign.json" <<'PY'
import json
import sys
src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["promotable"] = 1
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
if python3 "$BATCH" "$WORK/bad-campaign.json" "$WORK/bad-campaign" \
    > "$WORK/bad-campaign.out" 2>&1; then
  echo "error: campaign preflight batch accepted promotable campaign input" >&2
  exit 1
fi
rg -q 'only accepts non-promotable campaigns' "$WORK/bad-campaign.out"

python3 - "$WORK/campaign/campaign.json" "$WORK/bad-count.json" <<'PY'
import json
import sys
src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["campaign_count"] = 99
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
if python3 "$BATCH" "$WORK/bad-count.json" "$WORK/bad-count" \
    > "$WORK/bad-count.out" 2>&1; then
  echo "error: campaign preflight batch accepted wrong campaign_count" >&2
  exit 1
fi
rg -q 'campaign_count does not match ranking length' "$WORK/bad-count.out"

python3 - "$WORK/campaign/campaign.json" "$WORK/bad-path.json" <<'PY'
import json
import sys
src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["ranking"][0]["frontier_scout"] = "/etc/passwd"
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
if python3 "$BATCH" "$WORK/bad-path.json" "$WORK/bad-path" \
    > "$WORK/bad-path.out" 2>&1; then
  echo "error: campaign preflight batch accepted frontier_scout outside campaign dir" >&2
  exit 1
fi
rg -q 'frontier_scout must stay inside campaign directory' "$WORK/bad-path.out"

python3 - "$WORK/campaign/campaign.json" "$WORK/bad-status.json" <<'PY'
import json
import sys
src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["ranking"][0]["dsatur_status"] = "K_COLORING_FOUND "
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
python3 "$BATCH" "$WORK/bad-status.json" "$WORK/bad-status" --skip-coloring-found \
  > "$WORK/bad-status.out"
rg -q '^selected_count=1$' "$WORK/bad-status.out"

python3 - "$WORK/campaign/campaign.json" "$WORK/bad-unknown-status.json" <<'PY'
import json
import sys
src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["ranking"][0]["dsatur_status"] = "MAYBE"
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
if python3 "$BATCH" "$WORK/bad-unknown-status.json" "$WORK/bad-unknown-status" \
    > "$WORK/bad-unknown-status.out" 2>&1; then
  echo "error: campaign preflight batch accepted unknown dsatur_status" >&2
  exit 1
fi
rg -q 'unknown dsatur_status' "$WORK/bad-unknown-status.out"

echo "chi6_frontier_campaign_preflight_batch_gate: PASS"
