#!/usr/bin/env bash
# Gate for executing refute argv entries from a frontier preflight batch.
#
# This records subproblem refutation attempts only. A successful run here means
# leaf LRAT artifacts were emitted by the repo-local refuter; it is still not a
# global no-5-colouring proof or a Euclidean chi>=6 witness.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

ATTEMPT="$ROOT/examples/erdos/chi6_frontier_refute_attempt.py"
REFUTER="$ROOT/examples/erdos/cube_sieve_refute_batch.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
python3 -m py_compile "$ATTEMPT" "$REFUTER"
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

python3 - "$WORK/k6.batch.json" "$REFUTER" "$WORK/k6.edge" "$WORK/k6.cubes" "$WORK/k6-refute" <<'PY'
import json
import sys

dst, refuter, edge, cubes, out_dir = sys.argv[1:]
argv = [sys.executable, refuter, edge, "5", cubes, out_dir]
meta = {
    "schema": "chi6_frontier_campaign_preflight_batch.v1",
    "claim_scope": "frontier_campaign_preflight_batch_only",
    "sat_claim": "none",
    "chromatic_claim": "none",
    "global_unsat_claim": "none",
    "verified_claim": "none",
    "promotable": 0,
    "refute_ready_count": 1,
    "first_refute_candidate": "k6_refute_success",
    "preflights": [
        {
            "rank": 0,
            "candidate_id": "k6_refute_success",
            "recommended_next_action": "prepare_cube_refute_batch",
            "refute_argv": argv,
        }
    ],
}
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
print()
PY

echo "chi6_frontier_refute_attempt_gate: workdir=$WORK"
python3 "$ATTEMPT" "$WORK/k6.batch.json" "$WORK/attempt-success" > "$WORK/attempt-success.out"

rg -q '^chi6_frontier_refute_attempt v1$' "$WORK/attempt-success.out"
rg -q '^attempt_count=1$' "$WORK/attempt-success.out"
rg -q '^refute_success_count=1$' "$WORK/attempt-success.out"
rg -q '^refute_failed_count=0$' "$WORK/attempt-success.out"
rg -q '^first_success_candidate=k6_refute_success$' "$WORK/attempt-success.out"
rg -q '^first_failed_candidate=NONE$' "$WORK/attempt-success.out"
rg -q '^claim_scope=frontier_refute_attempt_only$' "$WORK/attempt-success.out"
rg -q '^sat_claim=none$' "$WORK/attempt-success.out"
rg -q '^chromatic_claim=none$' "$WORK/attempt-success.out"
rg -q '^global_unsat_claim=none$' "$WORK/attempt-success.out"
rg -q '^verified_claim=none$' "$WORK/attempt-success.out"
rg -q '^promotable=0$' "$WORK/attempt-success.out"
rg -q '^status=FRONTIER_REFUTE_ATTEMPT_RECORDED$' "$WORK/attempt-success.out"

SUCCESS_MANIFEST="$(rg '^refute_attempt_json=' "$WORK/attempt-success.out" | cut -d= -f2-)"
[[ -s "$SUCCESS_MANIFEST" ]]
python3 - "$SUCCESS_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["schema"] == "chi6_frontier_refute_attempt.v1"
assert meta["attempt_count"] == 1
assert meta["refute_success_count"] == 1
assert meta["refute_failed_count"] == 0
assert meta["status_counts"] == {"REFUTE_SUCCESS_UNPROMOTABLE": 1}
assert meta["claim_scope"] == "frontier_refute_attempt_only"
assert meta["sat_claim"] == "none"
assert meta["chromatic_claim"] == "none"
assert meta["global_unsat_claim"] == "none"
assert meta["verified_claim"] == "none"
assert meta["promotable"] == 0
row = meta["attempts"][0]
assert row["candidate_id"] == "k6_refute_success"
assert row["returncode"] == 0
assert row["classified_status"] == "REFUTE_SUCCESS_UNPROMOTABLE"
assert row["refuter_status"] == "subproblem_lrat_artifacts_emitted_unpromotable"
assert row["cube_count"] == 2
assert row["solver_unsat_count"] == 2
assert row["refuter_lrat_artifact_count"] == 2
assert row["lrat_artifact_count_on_disk"] == 2
assert row["failed_count"] == 0
assert row["formal_proof_checker"] == "none"
assert row["verified_claim"] == "none"
assert row["global_unsat_claim"] == "none"
assert row["promotable"] == "0"
assert Path(row["stdout"]).is_file()
assert Path(row["stderr"]).is_file()
PY

cat > "$WORK/path3.edge" <<'EOF'
p edge 3 2
e 1 2
e 2 3
EOF

cat > "$WORK/path3.cubes" <<'EOF'
sat: 0:0
EOF

python3 - "$WORK/sat.batch.json" "$REFUTER" "$WORK/path3.edge" "$WORK/path3.cubes" "$WORK/path3-refute" <<'PY'
import json
import sys

dst, refuter, edge, cubes, out_dir = sys.argv[1:]
meta = {
    "schema": "chi6_frontier_campaign_preflight_batch.v1",
    "claim_scope": "frontier_campaign_preflight_batch_only",
    "sat_claim": "none",
    "chromatic_claim": "none",
    "global_unsat_claim": "none",
    "verified_claim": "none",
    "promotable": 0,
    "refute_ready_count": 1,
    "first_refute_candidate": "frontier_sat",
    "preflights": [
        {
            "rank": 0,
            "candidate_id": "frontier_sat",
            "recommended_next_action": "prepare_cube_refute_batch",
            "refute_argv": [sys.executable, refuter, edge, "3", cubes, out_dir],
        }
    ],
}
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
print()
PY

python3 "$ATTEMPT" "$WORK/sat.batch.json" "$WORK/attempt-sat" > "$WORK/attempt-sat.out"
rg -q '^attempt_count=1$' "$WORK/attempt-sat.out"
rg -q '^refute_success_count=0$' "$WORK/attempt-sat.out"
rg -q '^refute_failed_count=1$' "$WORK/attempt-sat.out"
rg -q '^first_success_candidate=NONE$' "$WORK/attempt-sat.out"
rg -q '^first_failed_candidate=frontier_sat$' "$WORK/attempt-sat.out"
rg -q '^promotable=0$' "$WORK/attempt-sat.out"

SAT_MANIFEST="$(rg '^refute_attempt_json=' "$WORK/attempt-sat.out" | cut -d= -f2-)"
python3 - "$SAT_MANIFEST" <<'PY'
import json
import sys

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["status_counts"] == {"REFUTE_SAT_MUTATE_FRONTIER": 1}
row = meta["attempts"][0]
assert row["candidate_id"] == "frontier_sat"
assert row["returncode"] == 1
assert row["classified_status"] == "REFUTE_SAT_MUTATE_FRONTIER"
assert row["stderr_nonempty"] == 1
assert "SAT colouring=" in row["stderr_excerpt"]
assert row["sat_colouring_vertex_count"] == 3
assert row["sat_colouring"].startswith("0:0,")
assert row["refuter_status"] == "NONE"
assert row["lrat_artifact_count_on_disk"] == 0
assert row["promotable"] == "NONE"
PY

python3 - "$WORK/k6.batch.json" "$WORK/promotable.batch.json" <<'PY'
import json
import sys

src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["promotable"] = 1
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
if python3 "$ATTEMPT" "$WORK/promotable.batch.json" "$WORK/bad-promotable" \
    > "$WORK/bad-promotable.out" 2>&1; then
  echo "error: refute attempt accepted promotable preflight batch" >&2
  exit 1
fi
rg -q 'only accept non-promotable preflight batches' "$WORK/bad-promotable.out"

python3 - "$WORK/k6.batch.json" "$WORK/bad-argv.batch.json" <<'PY'
import json
import sys

src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["preflights"][0]["refute_argv"] = "python3 refuter.py"
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
if python3 "$ATTEMPT" "$WORK/bad-argv.batch.json" "$WORK/bad-argv" \
    > "$WORK/bad-argv.out" 2>&1; then
  echo "error: refute attempt accepted shell-string refute_argv" >&2
  exit 1
fi
rg -q 'refute_argv must be a non-empty list' "$WORK/bad-argv.out"

python3 - "$WORK/k6.batch.json" "$WORK/bad-refuter.batch.json" <<'PY'
import json
import sys

src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["preflights"][0]["refute_argv"][1] = "/bin/echo"
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
if python3 "$ATTEMPT" "$WORK/bad-refuter.batch.json" "$WORK/bad-refuter" \
    > "$WORK/bad-refuter.out" 2>&1; then
  echo "error: refute attempt accepted non-canonical refuter" >&2
  exit 1
fi
rg -q 'must be the canonical cube_sieve_refute_batch.py' "$WORK/bad-refuter.out"

python3 - "$WORK/k6.batch.json" "$WORK/bad-python.batch.json" <<'PY'
import json
import sys

src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["preflights"][0]["refute_argv"][0] = "/bin/echo"
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
if python3 "$ATTEMPT" "$WORK/bad-python.batch.json" "$WORK/bad-python" \
    > "$WORK/bad-python.out" 2>&1; then
  echo "error: refute attempt accepted non-current Python executable" >&2
  exit 1
fi
rg -q 'refute_argv\[0\] must be a Python executable' "$WORK/bad-python.out"

python3 - "$WORK/k6.batch.json" "$WORK/bad-id.batch.json" <<'PY'
import json
import sys

src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["first_refute_candidate"] = "bad..id"
meta["preflights"][0]["candidate_id"] = "bad..id"
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
if python3 "$ATTEMPT" "$WORK/bad-id.batch.json" "$WORK/bad-id" \
    > "$WORK/bad-id.out" 2>&1; then
  echo "error: refute attempt accepted unsafe candidate_id" >&2
  exit 1
fi
rg -q 'unsafe candidate_id' "$WORK/bad-id.out"

python3 - "$WORK/k6.batch.json" "$WORK/bad-count.batch.json" <<'PY'
import json
import sys

src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["refute_ready_count"] = 99
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
if python3 "$ATTEMPT" "$WORK/bad-count.batch.json" "$WORK/bad-count" \
    > "$WORK/bad-count.out" 2>&1; then
  echo "error: refute attempt accepted wrong refute_ready_count" >&2
  exit 1
fi
rg -q 'refute_ready_count does not match ready rows' "$WORK/bad-count.out"

echo "chi6_frontier_refute_attempt_gate: PASS"
