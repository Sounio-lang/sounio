#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write provenance for the promoted self-hosted artifact.")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--authority-summary", required=True)
    parser.add_argument("--provenance-out", required=True)
    parser.add_argument("--bootstrap-summary")
    parser.add_argument("--bootstrap-sha256")
    parser.add_argument("--policy-file", default="scripts/selfhost/selfhost_promotion_policy.v1.json")
    parser.add_argument("--repro-summary")
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


def manifest_support_classes(manifest_key: str, path: str) -> list[str]:
    classes: set[str] = set()
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or not row[0] or row[0].startswith("#"):
                continue
            if manifest_key == "abi_parity":
                if len(row) >= 3:
                    classes.add(row[2])
            else:
                if len(row) >= 2:
                    classes.add(row[1])
    return sorted(classes)


def stable_sha256_payload(data: dict[str, object]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
    policy_path = Path(args.policy_file)
    authority = json.loads(Path(args.authority_summary).read_text(encoding="utf-8"))
    bootstrap = {}
    if args.bootstrap_summary:
        bootstrap = json.loads(Path(args.bootstrap_summary).read_text(encoding="utf-8"))
    reproducible = {}
    if args.repro_summary:
        reproducible = json.loads(Path(args.repro_summary).read_text(encoding="utf-8"))
    policy = json.loads(policy_path.read_text(encoding="utf-8"))

    fixed_point = extract_fixed_point(authority)
    gate_status = []
    for check in authority["blocking_checks"]:
        gate_status.append({"name": check["name"], "status": check["status"]})
    for surface in authority["nonblocking_surfaces"]:
        gate_status.append({"name": surface["name"], "status": surface["status"], "blocking": False})

    required_checks_manifest = policy["manifests"]["required_checks"]
    authority_model_doc = policy["reference_docs"]["authority_model"]
    release_train_doc = policy["reference_docs"]["release_train"]
    debt_register_doc = policy["reference_docs"]["debt_register"]
    taxonomy_manifests = []
    for manifest_key, manifest_path in policy["manifests"].items():
        if manifest_key == "required_checks":
            continue
        taxonomy_manifests.append(
            {
                "key": manifest_key,
                "path": manifest_path,
                "sha256": sha256_file(manifest_path),
                "support_classes": manifest_support_classes(manifest_key, manifest_path),
            }
        )

    report = {
        "schema": "sounio.selfhost_artifact_provenance.v4",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "attestation_commit": git_output("rev-parse", "HEAD"),
            "attestation_branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
            "artifact_source_commit": git_output("rev-parse", "HEAD"),
            "artifact_source_branch": git_output("rev-parse", "--abbrev-ref", "HEAD"),
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
        "policy": {
            "policy_file": str(policy_path),
            "policy_sha256": sha256_file(str(policy_path)),
            "required_checks_manifest": required_checks_manifest,
            "required_checks_manifest_sha256": sha256_file(required_checks_manifest),
            "authority_model_doc": authority_model_doc,
            "release_train_doc": release_train_doc,
            "debt_register_doc": debt_register_doc,
            "promotion_entrypoint": policy["entrypoints"]["promotion_entrypoint"],
            "verification_entrypoint": policy["entrypoints"]["provenance_gate"],
            "drift_entrypoint": policy["entrypoints"]["drift_gate"],
            "policy_entrypoint": policy["entrypoints"]["policy_gate"],
            "release_candidate_entrypoint": policy["entrypoints"]["release_candidate_gate"],
        },
        "target_taxonomy": {
            "allowed_support_classes": policy["allowed_support_classes"],
            "manifests": taxonomy_manifests,
        },
        "trust_planes": {
            "canonical": {
                "plane": "repo_local_provenance",
                "canonical": True,
                "status": "pass",
                "verification_entrypoint": policy["entrypoints"]["provenance_gate"],
            },
            "supplemental": {
                "plane": "repo_local_reproducible_bootstrap",
                "canonical": False,
                "status": reproducible.get("overall_status", "unknown"),
                "entrypoint": policy["entrypoints"].get("reproducible_bootstrap_gate", ""),
                "artifact_sha256": reproducible.get("artifact_sha256", ""),
                "gen1_sha256": reproducible.get("gen1_sha256", ""),
                "gen2_sha256": reproducible.get("gen2_sha256", ""),
                "gen3_sha256": reproducible.get("gen3_sha256", ""),
                "gen2_md5": reproducible.get("gen2_md5", ""),
                "gen3_md5": reproducible.get("gen3_md5", ""),
            },
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

    report["integrity"] = {
        "payload_sha256": stable_sha256_payload(report),
    }

    Path(args.provenance_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
