#!/usr/bin/env python3
"""Omega Sprint 5 Track 1: Merkle inclusion proof hardening.

Validates that every artifact listed in omega_genesis.v1.0.json exists and matches
the pinned digest, then recomputes aggregate/merkle roots for replay integrity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "sounio.omega.merkle-inclusion-proof.v1"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify genesis manifest inclusion proof")
    parser.add_argument(
        "--manifest",
        default="artifacts/omega/omega_genesis.v1.0.json",
        help="Path to genesis manifest",
    )
    parser.add_argument(
        "--out",
        default="artifacts/omega/merkle_inclusion_proof.v1.json",
        help="Output proof report",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on any missing/mismatch/replay failure",
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"unable to read manifest {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid manifest JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"manifest {path} must be a JSON object")
    if payload.get("schema") != "sounio.omega.genesis.v1.0":
        raise SystemExit(
            f"manifest schema mismatch in {path}: expected sounio.omega.genesis.v1.0 got {payload.get('schema')!r}"
        )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise SystemExit(f"manifest {path} must include non-empty artifacts[]")
    return payload


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    manifest = load_manifest(manifest_path)

    artifacts = manifest["artifacts"]
    checked = 0
    missing: list[str] = []
    mismatched: list[dict[str, str]] = []
    digest_lines: list[str] = []
    digest_values: list[str] = []

    for item in sorted(artifacts, key=lambda row: str(row.get("path", ""))):
        if not isinstance(item, dict):
            missing.append("<invalid-artifact-record>")
            continue
        rel_path = str(item.get("path", "")).strip()
        expected_sha = str(item.get("sha256", "")).strip().lower()
        if not rel_path or len(expected_sha) != 64:
            missing.append(rel_path or "<missing-path>")
            continue

        path = Path(rel_path)
        if not path.exists():
            missing.append(rel_path)
            continue

        actual_sha = sha256_bytes(path.read_bytes())
        checked += 1
        digest_lines.append(f"{rel_path}:{actual_sha}")
        digest_values.append(actual_sha)

        if actual_sha != expected_sha:
            mismatched.append(
                {
                    "path": rel_path,
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                }
            )

    recomputed_aggregate = sha256_bytes("\n".join(digest_lines).encode("utf-8"))
    recomputed_merkle_root = sha256_bytes("|".join(digest_values).encode("utf-8"))

    aggregate_match = recomputed_aggregate == str(manifest.get("aggregate_sha256", ""))
    merkle_root_match = recomputed_merkle_root == str(manifest.get("merkle_root_sha256", ""))

    status = "pass"
    if missing or mismatched or not aggregate_match or not merkle_root_match:
        status = "fail"

    output = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "artifacts_total": len(artifacts),
        "artifacts_checked": checked,
        "missing_count": len(missing),
        "mismatch_count": len(mismatched),
        "included_ratio": (checked / len(artifacts)) if artifacts else 0.0,
        "aggregate_match": aggregate_match,
        "merkle_root_match": merkle_root_match,
        "recomputed_aggregate_sha256": recomputed_aggregate,
        "recomputed_merkle_root_sha256": recomputed_merkle_root,
        "missing_artifacts": missing,
        "mismatched_artifacts": mismatched,
        "status": status,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(
        "omega_merkle_inclusion_proof: "
        f"status={status} total={len(artifacts)} checked={checked} "
        f"missing={len(missing)} mismatched={len(mismatched)} "
        f"aggregate_match={str(aggregate_match).lower()} "
        f"merkle_root_match={str(merkle_root_match).lower()} out={out_path}"
    )

    if args.strict and status != "pass":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
