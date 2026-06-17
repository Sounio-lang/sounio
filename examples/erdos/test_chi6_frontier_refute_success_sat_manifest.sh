#!/usr/bin/env bash
# Gate frontier REFUTE_SUCCESS_UNPROMOTABLE -> non-promotable SAT manifest.
#
# K6 is only a calibration graph. This gate proves that a successful frontier
# refute ledger can be packaged into the existing arbitrary cube-cover SAT
# manifest route with Lean-checked finite colourCNF UNSAT, while still making no
# Euclidean geometry or chi(R^2)>=6 claim.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

ATTEMPT="$ROOT/examples/erdos/chi6_frontier_refute_attempt.py"
PACKAGER="$ROOT/examples/erdos/make_chi6_frontier_refute_success_sat_manifest.py"
REFUTER="$ROOT/examples/erdos/cube_sieve_refute_batch.py"
VALIDATOR="$ROOT/examples/erdos/validate_chi6_candidate_manifest.sh"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }
python3 -m py_compile "$ATTEMPT" "$PACKAGER" "$REFUTER"
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

cat > "$WORK/k6_v0_cover.cubes" <<'EOF'
k6_frontier_success_sat_v0_c0: 0:0
k6_frontier_success_sat_v0_c1: 0:1
k6_frontier_success_sat_v0_c2: 0:2
k6_frontier_success_sat_v0_c3: 0:3
k6_frontier_success_sat_v0_c4: 0:4
EOF

python3 - "$WORK/k6.batch.json" "$REFUTER" "$WORK/k6.edge" \
    "$WORK/k6_v0_cover.cubes" "$WORK/k6-refute" <<'PY'
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
    "first_refute_candidate": "k6_frontier_success_sat",
    "preflights": [
        {
            "rank": 0,
            "candidate_id": "k6_frontier_success_sat",
            "recommended_next_action": "prepare_cube_refute_batch",
            "refute_argv": [sys.executable, refuter, edge, "5", cubes, out_dir],
        }
    ],
}
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
print()
PY

echo "chi6_frontier_refute_success_sat_manifest_gate: workdir=$WORK"
python3 "$ATTEMPT" "$WORK/k6.batch.json" "$WORK/attempt-success" \
  > "$WORK/attempt-success.out"
rg -q '^refute_success_count=1$' "$WORK/attempt-success.out"
rg -q '^first_success_candidate=k6_frontier_success_sat$' "$WORK/attempt-success.out"
rg -q '^promotable=0$' "$WORK/attempt-success.out"

ATTEMPT_JSON="$(rg '^refute_attempt_json=' "$WORK/attempt-success.out" | cut -d= -f2-)"
[[ -s "$ATTEMPT_JSON" ]]
python3 - "$ATTEMPT_JSON" <<'PY'
import json
import sys

meta = json.load(open(sys.argv[1], encoding="ascii"))
assert meta["schema"] == "chi6_frontier_refute_attempt.v1"
assert meta["refute_success_count"] == 1
row = meta["attempts"][0]
assert row["classified_status"] == "REFUTE_SUCCESS_UNPROMOTABLE"
assert row["cube_count"] == 5
assert row["solver_unsat_count"] == 5
assert row["refuter_lrat_artifact_count"] == 5
assert row["lrat_artifact_count_on_disk"] == 5
assert row["promotable"] == "0"
PY

# The five singleton blockers cover vertex 0 by the base at-least-one clause,
# so an empty-clause DRUP addition is RUP for the complement-cover CNF.
printf '0\n' > "$WORK/cover_complement.drup"

PACK_WORK="$WORK/frontier-success-sat"
python3 "$PACKAGER" "$ATTEMPT_JSON" "$WORK/cover_complement.drup" "$PACK_WORK" \
  > "$WORK/packager.out"

rg -q '^chi6_frontier_refute_success_sat_manifest v1$' "$WORK/packager.out"
rg -q '^input_schema=chi6_frontier_refute_attempt.v1$' "$WORK/packager.out"
rg -q '^frontier_candidate_id=k6_frontier_success_sat$' "$WORK/packager.out"
rg -q '^candidate_id=k6_frontier_success_sat$' "$WORK/packager.out"
rg -q '^cube_count=5$' "$WORK/packager.out"
rg -q '^lrat_artifact_count=5$' "$WORK/packager.out"
rg -q '^claim_scope=frontier_refute_success_sat_packaging_only$' "$WORK/packager.out"
rg -q '^sat_claim_scope=finite_colourCNF_edge_only$' "$WORK/packager.out"
rg -q '^chromatic_claim=none$' "$WORK/packager.out"
rg -q '^geometry_claim=none$' "$WORK/packager.out"
rg -q '^euclidean_claim=none$' "$WORK/packager.out"
rg -q '^promotable=0$' "$WORK/packager.out"
rg -q '^status=FRONTIER_REFUTE_SUCCESS_SAT_MANIFEST_PACKAGED$' "$WORK/packager.out"

MANIFEST="$PACK_WORK/candidate.manifest"
LINEAGE="$PACK_WORK/frontier_refute_success_sat_manifest.json"
[[ -s "$MANIFEST" ]]
[[ -s "$LINEAGE" ]]

rg -q '^candidate_id=k6_frontier_success_sat$' "$MANIFEST"
rg -q '^promotable=0$' "$MANIFEST"
rg -q '^geometry_proof_type=none$' "$MANIFEST"
rg -q '^sat_proof_route=cube_cover_generic$' "$MANIFEST"
rg -q '^triangle_sb=none$' "$MANIFEST"
rg -q '^cube_batch_path=package/k6_frontier_success_sat\.cubes$' "$MANIFEST"
rg -q '^cube_refutation_summary_path=cube_refute\.out$' "$MANIFEST"
rg -q '^cube_cover_certificate_path=NONE$' "$MANIFEST"
rg -q '^cube_cover_complement_cnf_path=cover_complement\.cnf$' "$MANIFEST"
rg -q '^cube_cover_complement_lrat_path=cover_complement\.lrat$' "$MANIFEST"
rg -q '^chromatic_claim=none$' "$MANIFEST"
rg -q '^geometry_claim=none$' "$MANIFEST"

rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=k6_frontier_success_sat$' \
  "$PACK_WORK/manifest_validator.log"
rg -q '^chi6_external_arbitrary_cube_cover_candidate: PASS$' \
  "$PACK_WORK/frontier_refute_success_sat_manifest.maker.out"
rg -q 'SounioSatCubeCover.cube_cover_of_complement_unsat' \
  "$PACK_WORK/SounioSatChi6ExternalArbitraryCoverReflect.lean"
rg -q 'Std\.Tactic\.BVDecide\.LRAT\.check_sound _ \(SounioSatCubeCover\.cubeCoverComplementCNF' \
  "$PACK_WORK/SounioSatChi6ExternalArbitraryCoverReflect.lean"

"$VALIDATOR" "$MANIFEST" > "$WORK/validator.out"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=k6_frontier_success_sat$' \
  "$WORK/validator.out"

python3 - "$LINEAGE" "$ATTEMPT_JSON" "$WORK/k6.edge" "$WORK/k6_v0_cover.cubes" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

lineage = json.load(open(sys.argv[1], encoding="ascii"))
attempt_json = Path(sys.argv[2])
edge = Path(sys.argv[3])
cubes = Path(sys.argv[4])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert lineage["schema"] == "chi6_frontier_refute_success_sat_manifest.v1"
assert lineage["status"] == "FRONTIER_REFUTE_SUCCESS_SAT_MANIFEST_PACKAGED"
assert lineage["claim_scope"] == "frontier_refute_success_sat_packaging_only"
assert lineage["refute_attempt_json_sha256"] == sha(attempt_json)
assert lineage["frontier_candidate_id"] == "k6_frontier_success_sat"
assert lineage["candidate_id"] == "k6_frontier_success_sat"
assert lineage["edge_sha256"] == sha(edge)
assert lineage["cube_batch_sha256"] == sha(cubes)
assert lineage["cube_count"] == 5
assert lineage["lrat_artifact_count"] == 5
assert lineage["refuter_cube_row_count"] == 5
assert lineage["sat_proof_route"] == "cube_cover_generic"
assert lineage["chromatic_claim"] == "none"
assert lineage["geometry_claim"] == "none"
assert lineage["euclidean_claim"] == "none"
assert lineage["promotable"] == 0
assert Path(lineage["candidate_manifest"]).is_file()
PY

ATTEMPT_SHA="$(sha256sum "$ATTEMPT_JSON" | awk '{print $1}')"
python3 - "$WORK/sweep-success.json" "$ATTEMPT_JSON" "$ATTEMPT_SHA" <<'PY'
import json
import sys

dst, attempt_json, attempt_sha = sys.argv[1:]
meta = {
    "schema": "chi6_frontier_refute_sweep.v1",
    "cell_count": 1,
    "refute_success_count": 1,
    "first_success_candidate": "k6_frontier_success_sat",
    "claim_scope": "frontier_refute_sweep_only",
    "sat_claim": "none",
    "chromatic_claim": "none",
    "global_unsat_claim": "none",
    "verified_claim": "none",
    "promotable": 0,
    "cells": [
        {
            "cell_index": 0,
            "cell_dir": "synthetic_sweep_cell",
            "refute_success_count": 1,
            "first_success_candidate": "k6_frontier_success_sat",
            "refute_attempt_json": attempt_json,
            "refute_attempt_sha256": attempt_sha,
        }
    ],
}
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
print()
PY

SWEEP_PACK_WORK="$WORK/frontier-success-sat-sweep"
python3 "$PACKAGER" "$WORK/sweep-success.json" "$WORK/cover_complement.drup" \
    "$SWEEP_PACK_WORK" --candidate-id k6_frontier_success_sat_from_sweep \
  > "$WORK/packager-sweep.out"
rg -q '^input_schema=chi6_frontier_refute_sweep.v1$' "$WORK/packager-sweep.out"
rg -q '^frontier_candidate_id=k6_frontier_success_sat$' "$WORK/packager-sweep.out"
rg -q '^candidate_id=k6_frontier_success_sat_from_sweep$' "$WORK/packager-sweep.out"
rg -q '^status=FRONTIER_REFUTE_SUCCESS_SAT_MANIFEST_PACKAGED$' "$WORK/packager-sweep.out"
rg -q '^candidate_id=k6_frontier_success_sat_from_sweep$' \
  "$SWEEP_PACK_WORK/candidate.manifest"
rg -q '^chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate=k6_frontier_success_sat_from_sweep$' \
  "$SWEEP_PACK_WORK/manifest_validator.log"
python3 - "$SWEEP_PACK_WORK/frontier_refute_success_sat_manifest.json" <<'PY'
import json
import sys

lineage = json.load(open(sys.argv[1], encoding="ascii"))
assert lineage["input_schema"] == "chi6_frontier_refute_sweep.v1"
assert lineage["sweep_cell_index"] == 0
assert lineage["frontier_candidate_id"] == "k6_frontier_success_sat"
assert lineage["candidate_id"] == "k6_frontier_success_sat_from_sweep"
assert lineage["promotable"] == 0
PY

if python3 "$PACKAGER" "$ATTEMPT_JSON" "$WORK/cover_complement.drup" \
    "$WORK/bad-candidate" --success-candidate absent \
    > "$WORK/bad-candidate.out" 2>&1; then
  echo "error: packager accepted absent success candidate" >&2
  exit 1
fi
rg -q 'no successful refute row for candidate_id=absent' "$WORK/bad-candidate.out"

if python3 "$PACKAGER" "$ATTEMPT_JSON" "$WORK/cover_complement.drup" \
    "$WORK/bad-output-id" --candidate-id . \
    > "$WORK/bad-output-id.out" 2>&1; then
  echo "error: packager accepted unsafe output candidate id" >&2
  exit 1
fi
rg -q 'unsafe output candidate_id' "$WORK/bad-output-id.out"

python3 - "$ATTEMPT_JSON" "$WORK/no-success.json" <<'PY'
import json
import sys

src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["refute_success_count"] = 0
meta["status_counts"] = {"REFUTE_NORESULT_MUTATE_FRONTIER": 1}
meta["first_success_candidate"] = "NONE"
meta["attempts"][0]["classified_status"] = "REFUTE_NORESULT_MUTATE_FRONTIER"
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
if python3 "$PACKAGER" "$WORK/no-success.json" "$WORK/cover_complement.drup" \
    "$WORK/no-success-out" > "$WORK/no-success.out" 2>&1; then
  echo "error: packager accepted a refute attempt with no success" >&2
  exit 1
fi
rg -q 'has no REFUTE_SUCCESS_UNPROMOTABLE row' "$WORK/no-success.out"

python3 - "$ATTEMPT_JSON" "$WORK/bad-promotable.json" <<'PY'
import json
import sys

src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["promotable"] = 1
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
PY
if python3 "$PACKAGER" "$WORK/bad-promotable.json" "$WORK/cover_complement.drup" \
    "$WORK/bad-promotable-out" > "$WORK/bad-promotable.out" 2>&1; then
  echo "error: packager accepted promotable refute-attempt input" >&2
  exit 1
fi
rg -q 'refute attempt must carry promotable=0' "$WORK/bad-promotable.out"

echo "manifest=$MANIFEST"
echo "lineage_json=$LINEAGE"
echo "chi6_frontier_refute_success_sat_manifest_gate: PASS"
