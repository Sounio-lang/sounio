#!/usr/bin/env bash
# Gate frontier refute success -> source-bound integrated promotion preflight.
#
# The fixture uses a rational unit square as the source package, then forges a
# search-ledger success row that the integrated preflight must still reject at
# the SAT half. This is intentional: the rung must be source-bound and
# fail-closed before any chi(R^2)>=6 promotion is possible.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ -z "${WORK:-}" ]]; then
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT INT TERM
fi

JOINER="$ROOT/examples/erdos/make_chi6_frontier_refute_success_promotion_preflight.py"
SAT_PACKAGER="$ROOT/examples/erdos/make_chi6_frontier_refute_success_sat_manifest.py"
REFUTER="$ROOT/examples/erdos/cube_sieve_refute_batch.py"

command -v rg >/dev/null 2>&1 || { echo "error: ripgrep (rg) required" >&2; exit 127; }
command -v sha256sum >/dev/null 2>&1 || { echo "error: sha256sum required" >&2; exit 127; }
python3 -m py_compile "$JOINER" "$SAT_PACKAGER" "$REFUTER"
mkdir -p "$WORK"

EDGE="$WORK/square.edge"
COORDS="$WORK/square.coords.csv"
SOURCE="$WORK/square.candidate-source.json"
CUBES="$WORK/square.cubes"
COVER="$WORK/cover_complement.drup"

cat > "$EDGE" <<'EOF'
p edge 4 4
e 1 2
e 2 3
e 3 4
e 4 1
EOF

cat > "$COORDS" <<'EOF'
id,x,y
0,0,0
1,1,0
2,1,1
3,0,1
EOF

cat > "$CUBES" <<'EOF'
square_fake_c0: 0:0
EOF

printf '0\n' > "$COVER"

python3 - "$WORK" "$REFUTER" "$EDGE" "$COORDS" "$SOURCE" "$CUBES" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

work, refuter, edge, coords, source, cubes = map(Path, sys.argv[1:])
candidate_id = "square_frontier_promotion_preflight"
refuter = refuter.resolve()
edge = edge.resolve()
coords = coords.resolve()
source = source.resolve()
cubes = cubes.resolve()
refute_dir = (work / "fake-refute").resolve()
cube_id = "square_fake_c0"
cube_dir = refute_dir / cube_id
cube_dir.mkdir(parents=True, exist_ok=True)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


source.write_text(
    json.dumps(
        {
            "schema": "chi6_solver_candidate_package.v1",
            "candidate_id": candidate_id,
            "edge_path": edge.name,
            "edge_sha256": sha(edge),
            "coords_path": coords.name,
            "coords_sha256": sha(coords),
            "coordinate_domain": "rational_xy",
            "n": 4,
            "m": 4,
            "k": 5,
            "split_vertices": [0, 1],
            "producer_command": "test fixture: rational unit square",
            "claim_scope": "solver_candidate_source_only",
            "promotion_gate": "requires_cube_cover_lrat_lean_exact_geometry_and_real_bridge",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="ascii",
)

campaign = {
    "schema": "chi6_frontier_campaign_preflight.v1",
    "candidate_id": candidate_id,
    "campaign_id": "synthetic_square_source_bound_join",
    "candidate_source_path": str(source),
    "candidate_source_sha256": sha(source),
    "source_status": "PASS",
    "n": 4,
    "m": 4,
    "k": 5,
    "edge_path_abs": str(edge),
    "edge_sha256": sha(edge),
    "coords_path_abs": str(coords),
    "coords_sha256": sha(coords),
    "split_vertices": [0, 1],
    "cube_count": 1,
    "cube_batch_path": str(cubes),
    "cube_batch_sha256": sha(cubes),
    "recommended_next_action": "prepare_cube_refute_batch",
    "recommended_next_gate": "prepare_cube_refute_batch",
    "claim_scope": "deterministic_campaign_preflight_only",
    "geometry_claim": "exact_rational_squared_distance_edges_only_from_source_validator",
    "sat_claim": "none",
    "chromatic_claim": "none",
    "global_unsat_claim": "none",
    "verified_claim": "none",
    "promotable": 0,
    "promotion_gate": "requires_leaf_lrat_cover_lrat_lean_exact_geometry_real_bridge",
}
campaign_path = (work / "campaign_preflight.json").resolve()
campaign_path.write_text(json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="ascii")

argv = [sys.executable, str(refuter), str(edge), "5", str(cubes), str(refute_dir)]
batch = {
    "schema": "chi6_frontier_campaign_preflight_batch.v1",
    "claim_scope": "frontier_campaign_preflight_batch_only",
    "sat_claim": "none",
    "chromatic_claim": "none",
    "global_unsat_claim": "none",
    "verified_claim": "none",
    "promotable": 0,
    "refute_ready_count": 1,
    "first_refute_candidate": candidate_id,
    "preflights": [
        {
            "rank": 0,
            "candidate_id": candidate_id,
            "recommended_next_action": "prepare_cube_refute_batch",
            "campaign_preflight_json": str(campaign_path),
            "campaign_preflight_sha256": sha(campaign_path),
            "refute_argv": argv,
        }
    ],
}
batch_path = (work / "preflight_batch.json").resolve()
batch_path.write_text(json.dumps(batch, indent=2, sort_keys=True) + "\n", encoding="ascii")

artifact_paths = {
    "cube": cube_dir / f"{cube_id}.cube",
    "cnf": cube_dir / f"{cube_id}.cnf",
    "drat": cube_dir / f"{cube_id}.drat",
    "lrat": cube_dir / f"{cube_id}.lrat",
}
artifact_paths["cube"].write_text("0:0\n", encoding="ascii")
artifact_paths["cnf"].write_text("p cnf 20 25\n0\n", encoding="ascii")
artifact_paths["drat"].write_text("0\n", encoding="ascii")
artifact_paths["lrat"].write_text("1 0\n", encoding="ascii")

stdout_path = (work / "fake-refute.out").resolve()
stderr_path = (work / "fake-refute.err").resolve()
stderr_path.write_text("", encoding="ascii")
stdout_path.write_text(
    "\n".join(
        [
            "cube_sieve_refute_batch v1",
            "formula_kind=colourCNF",
            f"edge_path={edge}",
            f"edge_sha256={sha(edge)}",
            "k=5",
            f"cube_batch_path={cubes}",
            f"cube_batch_sha256={sha(cubes)}",
            f"out_dir={refute_dir}",
            "cube_count=1",
            "solver_unsat_count=1",
            "failed_count=0",
            "lrat_artifact_count=1",
            "formal_proof_checker=none",
            "verified_claim=none",
            "global_unsat_claim=none",
            "geometry_claim=none",
            "promotable=0",
            "status=subproblem_lrat_artifacts_emitted_unpromotable",
            (
                f"cube index=0 id={cube_id} assignments=0:0 unit_lits=1 "
                f"cube={cube_id}/{cube_id}.cube cube_sha256={sha(artifact_paths['cube'])} "
                f"cnf={cube_id}/{cube_id}.cnf cnf_sha256={sha(artifact_paths['cnf'])} "
                f"drat={cube_id}/{cube_id}.drat drat_sha256={sha(artifact_paths['drat'])} "
                f"lrat={cube_id}/{cube_id}.lrat lrat_sha256={sha(artifact_paths['lrat'])} "
                "cnf_clauses=25 expected_cnf_clauses=25 drat_deletions=0"
            ),
            "",
        ]
    ),
    encoding="ascii",
)

attempt = {
    "schema": "chi6_frontier_refute_attempt.v1",
    "claim_scope": "frontier_refute_attempt_only",
    "sat_claim": "none",
    "chromatic_claim": "none",
    "global_unsat_claim": "none",
    "verified_claim": "none",
    "promotable": 0,
    "preflight_batch_json": str(batch_path),
    "preflight_batch_sha256": sha(batch_path),
    "refute_success_count": 1,
    "first_success_candidate": candidate_id,
    "status_counts": {"REFUTE_SUCCESS_UNPROMOTABLE": 1},
    "attempts": [
        {
            "candidate_id": candidate_id,
            "classified_status": "REFUTE_SUCCESS_UNPROMOTABLE",
            "returncode": 0,
            "classification_note": "leaf_lrat_artifacts_emitted_no_global_claim",
            "refuter_status": "subproblem_lrat_artifacts_emitted_unpromotable",
            "formal_proof_checker": "none",
            "verified_claim": "none",
            "global_unsat_claim": "none",
            "promotable": "0",
            "cube_count": 1,
            "solver_unsat_count": 1,
            "failed_count": 0,
            "refuter_lrat_artifact_count": 1,
            "lrat_artifact_count_on_disk": 1,
            "stdout": str(stdout_path),
            "stdout_sha256": sha(stdout_path),
            "stderr": str(stderr_path),
            "stderr_sha256": sha(stderr_path),
            "argv": argv,
        }
    ],
}
attempt_path = (work / "attempt-success.json").resolve()
attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY

ATTEMPT_JSON="$WORK/attempt-success.json"
[[ -s "$ATTEMPT_JSON" ]]

echo "chi6_frontier_refute_success_promotion_preflight_gate: workdir=$WORK"
OUT="$WORK/promotion-preflight"
python3 "$JOINER" "$ATTEMPT_JSON" "$COVER" "$OUT" > "$WORK/joiner.out"

rg -q '^chi6_frontier_refute_success_promotion_preflight v1$' "$WORK/joiner.out"
rg -q '^input_schema=chi6_frontier_refute_attempt.v1$' "$WORK/joiner.out"
rg -q '^frontier_candidate_id=square_frontier_promotion_preflight$' "$WORK/joiner.out"
rg -q '^candidate_id=square_frontier_promotion_preflight$' "$WORK/joiner.out"
rg -q '^source_status=PASS$' "$WORK/joiner.out"
rg -q '^geometry_status=PASS$' "$WORK/joiner.out"
rg -q '^sat_status=FAIL$' "$WORK/joiner.out"
rg -q '^integrated_status=INCOMPLETE$' "$WORK/joiner.out"
rg -q '^first_blocker=sat_arbitrary_cube_cover_refutation_absent$' "$WORK/joiner.out"
rg -q '^promotion_ready=0$' "$WORK/joiner.out"
rg -q '^claim_scope=frontier_refute_success_promotion_preflight_only$' "$WORK/joiner.out"
rg -q '^chromatic_claim=none$' "$WORK/joiner.out"
rg -q '^promotable=0$' "$WORK/joiner.out"
rg -q '^status=FRONTIER_REFUTE_SUCCESS_PROMOTION_PREFLIGHT_INCOMPLETE$' "$WORK/joiner.out"

LINEAGE="$OUT/frontier_refute_success_promotion_preflight.json"
[[ -s "$LINEAGE" ]]
python3 - "$LINEAGE" "$ATTEMPT_JSON" "$SOURCE" "$EDGE" "$COORDS" "$CUBES" "$COVER" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

lineage = json.load(open(sys.argv[1], encoding="ascii"))
attempt, source, edge, coords, cubes, cover = map(Path, sys.argv[2:])

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

assert lineage["schema"] == "chi6_frontier_refute_success_promotion_preflight.v1"
assert lineage["status"] == "FRONTIER_REFUTE_SUCCESS_PROMOTION_PREFLIGHT_INCOMPLETE"
assert lineage["claim_scope"] == "frontier_refute_success_promotion_preflight_only"
assert lineage["refute_attempt_json_sha256"] == sha(attempt)
assert lineage["candidate_source_sha256"] == sha(source)
assert lineage["edge_sha256"] == sha(edge)
assert lineage["coords_sha256"] == sha(coords)
assert lineage["cube_batch_sha256"] == sha(cubes)
assert lineage["cover_drup_or_rup_sha256"] == sha(cover)
assert lineage["source_status"] == "PASS"
assert lineage["geometry_status"] == "PASS"
assert lineage["sat_status"] == "FAIL"
assert lineage["integrated_status"] == "INCOMPLETE"
assert lineage["promotion_ready"] == 0
assert lineage["promotable"] == 0
assert lineage["chromatic_claim"] == "none"
PY

python3 - "$WORK/preflight_batch.json" "$WORK/bad-batch.json" <<'PY'
import json
import sys

src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["preflights"][0]["campaign_preflight_sha256"] = "0" * 64
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
print()
PY
python3 - "$ATTEMPT_JSON" "$WORK/bad-batch.json" "$WORK/bad-campaign-hash-attempt.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

src, batch, dst = map(Path, sys.argv[1:])
meta = json.load(open(src, encoding="ascii"))
meta["preflight_batch_json"] = str(batch)
meta["preflight_batch_sha256"] = hashlib.sha256(batch.read_bytes()).hexdigest()
dst.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY
if python3 "$JOINER" "$WORK/bad-campaign-hash-attempt.json" "$COVER" \
    "$WORK/bad-campaign-hash-out" > "$WORK/bad-campaign-hash.out" 2>&1; then
  echo "error: joiner accepted stale campaign_preflight_json hash" >&2
  exit 1
fi
rg -q 'campaign_preflight_json SHA256 mismatch' "$WORK/bad-campaign-hash.out"

python3 - "$WORK/campaign_preflight.json" "$WORK/bad-source-campaign.json" <<'PY'
import json
import sys

src, dst = sys.argv[1:]
meta = json.load(open(src, encoding="ascii"))
meta["candidate_source_sha256"] = "0" * 64
json.dump(meta, open(dst, "w", encoding="ascii"), indent=2, sort_keys=True)
print()
PY
python3 - "$WORK/preflight_batch.json" "$WORK/bad-source-campaign.json" "$WORK/bad-source-batch.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

src, campaign, dst = map(Path, sys.argv[1:])
meta = json.load(open(src, encoding="ascii"))
meta["preflights"][0]["campaign_preflight_json"] = str(campaign)
meta["preflights"][0]["campaign_preflight_sha256"] = hashlib.sha256(campaign.read_bytes()).hexdigest()
dst.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY
python3 - "$ATTEMPT_JSON" "$WORK/bad-source-batch.json" "$WORK/bad-source-attempt.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

src, batch, dst = map(Path, sys.argv[1:])
meta = json.load(open(src, encoding="ascii"))
meta["preflight_batch_json"] = str(batch)
meta["preflight_batch_sha256"] = hashlib.sha256(batch.read_bytes()).hexdigest()
dst.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="ascii")
PY
if python3 "$JOINER" "$WORK/bad-source-attempt.json" "$COVER" \
    "$WORK/bad-source-out" > "$WORK/bad-source.out" 2>&1; then
  echo "error: joiner accepted stale candidate_source_sha256" >&2
  exit 1
fi
rg -q 'candidate_source_path SHA256 mismatch' "$WORK/bad-source.out"

echo "lineage_json=$LINEAGE"
echo "chi6_frontier_refute_success_promotion_preflight_gate: PASS"
