#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


REQUIRED_BLOCKING_GATES = [
    "fixed_point",
    "fallback_inventory",
    "abi_parity_regressions",
    "aarch64_compile_proof",
    "source_artifact_parity",
]

SUPPORTED_SCHEMAS = {
    "sounio.selfhost_artifact_provenance.v1",
    "sounio.selfhost_artifact_provenance.v2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify checked selfhost artifact provenance.")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    artifact_path = Path(args.artifact)
    provenance_path = Path(args.provenance)

    errors: list[str] = []
    warnings: list[str] = []

    if not artifact_path.is_file():
        raise SystemExit(f"missing artifact: {artifact_path}")
    if not provenance_path.is_file():
        raise SystemExit(f"missing provenance: {provenance_path}")

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    schema = provenance.get("schema", "")
    if schema not in SUPPORTED_SCHEMAS:
        errors.append(f"unsupported schema: {schema}")

    artifact = provenance.get("artifact", {})
    promotion = provenance.get("promotion", {})
    bootstrap = provenance.get("bootstrap", {})
    fixed_point = promotion.get("fixed_point", {})
    gates_run = promotion.get("gates_run", [])

    actual_sha256 = sha256_file(artifact_path)
    actual_size = artifact_path.stat().st_size
    recorded_sha256 = artifact.get("sha256", "")
    recorded_size = artifact.get("size_bytes", 0)

    if recorded_sha256 != actual_sha256:
      errors.append("artifact sha256 does not match provenance")
    if recorded_size != actual_size:
      errors.append("artifact size does not match provenance")

    recorded_path = artifact.get("path", "")
    if recorded_path and Path(recorded_path) != artifact_path:
        warnings.append("recorded artifact path differs from current repo path")

    if promotion.get("overall_status") != "pass":
        errors.append("promotion overall_status is not pass")

    gen2_md5 = fixed_point.get("gen2_md5", "")
    gen3_md5 = fixed_point.get("gen3_md5", "")
    if fixed_point.get("status") != "pass":
        errors.append("fixed_point status is not pass")
    if not gen2_md5 or not gen3_md5 or gen2_md5 != gen3_md5:
        errors.append("fixed_point md5 values are missing or do not match")

    gate_status = {gate.get("name", ""): gate.get("status", "") for gate in gates_run}
    for gate_name in REQUIRED_BLOCKING_GATES:
        if gate_status.get(gate_name) != "pass":
            errors.append(f"required gate missing or not pass: {gate_name}")

    bootstrap_status = bootstrap.get("overall_status", "")
    if bootstrap_status and bootstrap_status != "pass":
        errors.append("bootstrap overall_status is not pass")

    if schema == "sounio.selfhost_artifact_provenance.v2":
        policy = provenance.get("policy", {})
        for key in [
            "authority_model_doc",
            "release_train_doc",
            "required_checks_manifest",
            "promotion_entrypoint",
            "verification_entrypoint",
        ]:
            if not policy.get(key):
                errors.append(f"missing policy field: {key}")

    report = {
        "schema": "sounio.selfhost_artifact_provenance_verification.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact": {
            "path": str(artifact_path),
            "sha256": actual_sha256,
            "size_bytes": actual_size,
        },
        "provenance": {
            "path": str(provenance_path),
            "schema": schema,
        },
        "checks": {
            "artifact_sha256_match": recorded_sha256 == actual_sha256,
            "artifact_size_match": recorded_size == actual_size,
            "promotion_status_pass": promotion.get("overall_status") == "pass",
            "fixed_point_md5_match": bool(gen2_md5 and gen3_md5 and gen2_md5 == gen3_md5),
            "required_gates_present": all(gate_status.get(name) == "pass" for name in REQUIRED_BLOCKING_GATES),
            "bootstrap_status_pass": bootstrap_status in ("", "pass"),
        },
        "warnings": warnings,
        "errors": errors,
        "overall_status": "pass" if not errors else "fail",
    }

    Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Selfhost Artifact Provenance",
        "",
        f"- overall_status: {report['overall_status']}",
        f"- artifact_sha256: {actual_sha256}",
        f"- provenance_schema: {schema}",
        f"- fixed_point_md5: {gen2_md5}",
        "",
        "## Checks",
        "",
    ]
    for key, value in report["checks"].items():
        md_lines.append(f"- {key}: {'pass' if value else 'fail'}")
    if warnings:
        md_lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            md_lines.append(f"- {warning}")
    if errors:
        md_lines.extend(["", "## Errors", ""])
        for error in errors:
            md_lines.append(f"- {error}")
    Path(args.md_out).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
