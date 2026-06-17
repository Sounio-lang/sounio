#!/usr/bin/env python3
"""Convert one ready campaign preflight into a refute-attempt batch.

This is a machine-safe adapter from `chi6_frontier_campaign_preflight.v1` to the
`chi6_frontier_campaign_preflight_batch.v1` shape consumed by
`chi6_frontier_refute_attempt.py`.  It does not run the refuter and does not
promote any claim; it only preserves the exact argv needed for the existing
bounded refute executor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REFUTE = Path(__file__).with_name("cube_sieve_refute_batch.py").resolve()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as f:
        return json.load(f)


def require_no_claims(meta: dict[str, Any], label: str) -> None:
    for key in ("sat_claim", "chromatic_claim", "global_unsat_claim", "verified_claim"):
        if meta.get(key) != "none":
            raise ValueError(f"{label} must carry {key}=none")
    if meta.get("promotable") != 0:
        raise ValueError(f"{label} must carry promotable=0")


def require_file_with_hash(meta: dict[str, Any], path_key: str, hash_key: str) -> Path:
    path = Path(str(meta.get(path_key, "")))
    if not path.is_file():
        raise ValueError(f"missing {path_key}: {path}")
    expected = str(meta.get(hash_key, ""))
    if expected and expected != sha256_file(path):
        raise ValueError(f"{hash_key} does not match {path_key}")
    return path


def load_preflight(path: Path) -> dict[str, Any]:
    meta = load_json(path)
    if meta.get("schema") != "chi6_frontier_campaign_preflight.v1":
        raise ValueError("preflight schema must be chi6_frontier_campaign_preflight.v1")
    if meta.get("claim_scope") != "deterministic_campaign_preflight_only":
        raise ValueError("preflight claim_scope must be deterministic_campaign_preflight_only")
    require_no_claims(meta, "preflight")
    if meta.get("recommended_next_action") != "prepare_cube_refute_batch":
        raise ValueError("preflight recommended_next_action must be prepare_cube_refute_batch")
    if not REFUTE.is_file():
        raise ValueError(f"missing refuter: {REFUTE}")
    for key in ("candidate_id", "edge_path_abs", "k", "cube_batch_path", "cube_count"):
        if key not in meta:
            raise ValueError(f"preflight missing field: {key}")
    require_file_with_hash(meta, "edge_path_abs", "edge_sha256")
    require_file_with_hash(meta, "cube_batch_path", "cube_batch_sha256")
    if int(meta["k"]) != 5:
        raise ValueError("chi6 refute bridge only supports k=5")
    if int(meta["cube_count"]) < 1:
        raise ValueError("preflight cube_count must be positive")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_preflight_json", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()

    try:
        preflight = load_preflight(args.campaign_preflight_json)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        refute_out_dir = args.out_dir / "refute"
        candidate_id = str(preflight["candidate_id"])
        refute_argv = [
            sys.executable,
            str(REFUTE),
            str(preflight["edge_path_abs"]),
            str(preflight["k"]),
            str(preflight["cube_batch_path"]),
            str(refute_out_dir),
        ]
        row = {
            "rank": 0,
            "candidate_id": candidate_id,
            "dsatur_status": preflight.get("dsatur_status", "NONE"),
            "solver_heuristic_priority": preflight.get("solver_heuristic_priority"),
            "frontier_scout": preflight.get("frontier_scout_path", "NONE"),
            "frontier_scout_sha256": preflight.get("frontier_scout_sha256", "NONE"),
            "campaign_preflight_json": str(args.campaign_preflight_json),
            "campaign_preflight_sha256": sha256_file(args.campaign_preflight_json),
            "campaign_id": preflight.get("campaign_id", "NONE"),
            "cube_count": int(preflight["cube_count"]),
            "propagation_conflict_count": int(preflight.get("propagation_conflict_count", 0)),
            "propagation_hard_count": int(preflight.get("propagation_hard_count", 0)),
            "recommended_next_action": "prepare_cube_refute_batch",
            "refute_command": " ".join(refute_argv),
            "refute_argv": refute_argv,
            "foundry_handoff_recommended": int(preflight.get("foundry_handoff_recommended", 0)),
        }
        manifest = {
            "schema": "chi6_frontier_campaign_preflight_batch.v1",
            "campaign_json": str(args.campaign_preflight_json),
            "campaign_json_sha256": sha256_file(args.campaign_preflight_json),
            "campaign_count": 1,
            "selected_count": 1,
            "skipped_count": 0,
            "skipped": [],
            "action_counts": {"prepare_cube_refute_batch": 1},
            "refute_ready_count": 1,
            "first_refute_candidate": candidate_id,
            "first_refute_command": row["refute_command"],
            "first_refute_argv": refute_argv,
            "claim_scope": "frontier_campaign_preflight_batch_only",
            "sat_claim": "none",
            "chromatic_claim": "none",
            "global_unsat_claim": "none",
            "verified_claim": "none",
            "promotable": 0,
            "promotion_gate": "requires_refute_lrat_cover_lrat_lean_exact_geometry_real_bridge",
            "preflights": [row],
        }
        manifest_path = args.out_dir / "frontier_campaign_preflight_batch.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="ascii",
        )
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_campaign_preflight_to_refute_batch v1")
    print(f"preflight_batch_json={manifest_path}")
    print(f"preflight_batch_sha256={sha256_file(manifest_path)}")
    print(f"candidate_id={candidate_id}")
    print("refute_ready_count=1")
    print(f"first_refute_argv_json={json.dumps(refute_argv, separators=(',', ':'))}")
    print("claim_scope=frontier_campaign_preflight_batch_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("global_unsat_claim=none")
    print("verified_claim=none")
    print("promotable=0")
    print("status=CAMPAIGN_PREFLIGHT_REFUTE_BATCH_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
