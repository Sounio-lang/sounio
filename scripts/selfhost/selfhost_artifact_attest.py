#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write provenance for the promoted self-hosted artifact.")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--authority-summary", required=True)
    parser.add_argument("--provenance-out", required=True)
    parser.add_argument("--bootstrap-summary")
    parser.add_argument("--bootstrap-sha256")
    return parser.parse_args()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, timeout=5).strip()
    except Exception:
        return "unknown"


def extract_fixed_point(authority: dict[str, object]) -> dict[str, object]:
    for check in authority["blocking_checks"]:
        if check["name"] == "fixed_point":
            return {
                "gen2_md5": check.get("gen2_md5", ""),
                "gen3_md5": check.get("gen3_md5", ""),
                "status": check.get("status", "unknown"),
            }
    return {"gen2_md5": "", "gen3_md5": "", "status": "missing"}


def main() -> int:
    args = parse_args()

    artifact_path = Path(args.artifact)
    source_path = Path(args.source)
    authority = json.loads(Path(args.authority_summary).read_text(encoding="utf-8"))
    bootstrap = {}
    if args.bootstrap_summary:
        bootstrap = json.loads(Path(args.bootstrap_summary).read_text(encoding="utf-8"))

    fixed_point = extract_fixed_point(authority)
    gate_status = []
    for check in authority["blocking_checks"]:
        gate_status.append({"name": check["name"], "status": check["status"]})
    for surface in authority["nonblocking_surfaces"]:
        gate_status.append({"name": surface["name"], "status": surface["status"], "blocking": False})

    report = {
        "schema": "sounio.selfhost_artifact_provenance.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "commit": git_output("rev-parse", "HEAD"),
            "branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
        },
        "artifact": {
            "path": str(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
            "sha256": sha256_file(str(artifact_path)),
        },
        "source": {
            "path": str(source_path),
            "sha256": sha256_file(str(source_path)),
        },
        "promotion": {
            "authority_summary": str(args.authority_summary),
            "overall_status": authority["overall_status"],
            "entrypoint": authority["entrypoint"],
            "fixed_point": fixed_point,
            "baseline_context": authority["baseline_context"],
            "gates_run": gate_status,
        },
        "bootstrap": {
            "authority_summary": str(args.bootstrap_summary or ""),
            "sha256_before_update": args.bootstrap_sha256 or "",
            "overall_status": bootstrap.get("overall_status", ""),
        },
    }

    Path(args.provenance_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
