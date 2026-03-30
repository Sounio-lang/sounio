#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import hashlib
import json
import time
from pathlib import Path


REQUIRED_BLOCKING_GATES = [
    "fixed_point",
    "fallback_inventory",
    "abi_parity_regressions",
    "aarch64_compile_proof",
    "aarch64_runtime",
    "source_artifact_parity",
]

SUPPORTED_SCHEMAS = {
    "sounio.selfhost_artifact_provenance.v1",
    "sounio.selfhost_artifact_provenance.v2",
    "sounio.selfhost_artifact_provenance.v3",
    "sounio.selfhost_artifact_provenance.v4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify checked selfhost artifact provenance.")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--policy", default="scripts/selfhost/selfhost_promotion_policy.v1.json")
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


def sha256_bytes(data: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(data)
    return digest.hexdigest()


def stable_sha256_payload(data: dict[str, object]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def git_show_blob(commit: str, rel_path: str) -> bytes | None:
    try:
        return subprocess.check_output(["git", "show", f"{commit}:{rel_path}"], timeout=10)
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    artifact_path = Path(args.artifact)
    provenance_path = Path(args.provenance)
    policy_path = Path(args.policy)

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

    if not policy_path.is_file():
        errors.append(f"missing policy file: {policy_path}")
        policy = {}
    else:
        policy = json.loads(policy_path.read_text(encoding="utf-8"))

    artifact = provenance.get("artifact", {})
    promotion = provenance.get("promotion", {})
    bootstrap = provenance.get("bootstrap", {})
    source = provenance.get("source", {})
    policy_meta = provenance.get("policy", {})
    taxonomy = provenance.get("target_taxonomy", {})
    trust_planes = provenance.get("trust_planes", {})
    integrity = provenance.get("integrity", {})
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

    source_path_value = source.get("path", "")
    if not source_path_value:
        errors.append("missing source.path in provenance")
        current_source_sha = ""
    else:
        source_path = Path(source_path_value)
        if not source_path.is_file():
            errors.append(f"missing source file from provenance: {source_path_value}")
            current_source_sha = ""
        else:
            current_source_sha = sha256_file(source_path)
            if current_source_sha != source.get("sha256", ""):
                errors.append("source sha256 does not match current source tree")

    if promotion.get("overall_status") != "pass":
        errors.append("promotion overall_status is not pass")

    gen2_md5 = fixed_point.get("gen2_md5", "")
    gen3_md5 = fixed_point.get("gen3_md5", "")
    if fixed_point.get("status") != "pass":
        errors.append("fixed_point status is not pass")
    if not gen2_md5 or not gen3_md5 or gen2_md5 != gen3_md5:
        errors.append("fixed_point md5 values are missing or do not match")

    gate_status = {gate.get("name", ""): gate.get("status", "") for gate in gates_run}
    required_gate_names = policy.get("required_gate_names_in_provenance", REQUIRED_BLOCKING_GATES)
    for gate_name in required_gate_names:
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
    if schema == "sounio.selfhost_artifact_provenance.v3":
        required_fields = policy.get("required_provenance_fields", [])
        for key in required_fields:
            if key not in provenance:
                errors.append(f"missing required provenance field: {key}")

        recorded_policy_file = policy_meta.get("policy_file", "")
        if recorded_policy_file != str(policy_path):
            errors.append("policy file path does not match current policy path")
        if policy_meta.get("policy_sha256", "") != sha256_file(policy_path):
            errors.append("policy sha256 does not match current policy file")

        required_checks_path = policy_meta.get("required_checks_manifest", "")
        if not required_checks_path:
            errors.append("missing required_checks_manifest in provenance")
        else:
            required_checks_file = Path(required_checks_path)
            if not required_checks_file.is_file():
                errors.append(f"missing required checks manifest: {required_checks_path}")
            elif policy_meta.get("required_checks_manifest_sha256", "") != sha256_file(required_checks_file):
                errors.append("required checks manifest sha256 does not match current file")

        allowed_support_classes = set(policy.get("allowed_support_classes", []))
        recorded_allowed_support_classes = set(taxonomy.get("allowed_support_classes", []))
        if recorded_allowed_support_classes != allowed_support_classes:
            errors.append("recorded allowed support classes do not match policy")

        recorded_manifest_paths: set[str] = set()
        expected_manifest_paths: set[str] = set()
        for manifest_key, manifest_path in policy.get("manifests", {}).items():
            if manifest_key == "required_checks":
                continue
            expected_manifest_paths.add(manifest_path)
        for manifest in taxonomy.get("manifests", []):
            manifest_path = Path(manifest.get("path", ""))
            if manifest.get("path", ""):
                recorded_manifest_paths.add(manifest.get("path", ""))
            if not manifest_path.is_file():
                errors.append(f"missing taxonomy manifest: {manifest.get('path', '')}")
                continue
            if manifest.get("sha256", "") != sha256_file(manifest_path):
                errors.append(f"taxonomy manifest sha256 does not match current file: {manifest_path}")
            for support_class in manifest.get("support_classes", []):
                if support_class not in allowed_support_classes:
                    errors.append(f"taxonomy manifest uses unsupported support_class: {support_class}")
        if recorded_manifest_paths != expected_manifest_paths:
            errors.append("taxonomy manifest set does not match promotion policy manifests")

        artifact_source_commit = provenance.get("git", {}).get("artifact_source_commit", "")
        if not artifact_source_commit or artifact_source_commit == "unknown":
            errors.append("missing artifact_source_commit in provenance")
        elif source_path_value:
            blob = git_show_blob(artifact_source_commit, source_path_value)
            if blob is None:
                errors.append("artifact_source_commit does not resolve the recorded source path")
            else:
                if sha256_bytes(blob) != source.get("sha256", ""):
                    errors.append("artifact_source_commit does not match recorded source sha256")

        integrity_payload = dict(provenance)
        integrity_payload.pop("integrity", None)
        expected_payload_sha = stable_sha256_payload(integrity_payload)
        if integrity.get("payload_sha256", "") != expected_payload_sha:
            errors.append("integrity payload sha256 does not match canonical provenance payload")

    if schema == "sounio.selfhost_artifact_provenance.v4":
        required_fields = policy.get("required_provenance_fields", [])
        for key in required_fields:
            if key not in provenance:
                errors.append(f"missing required provenance field: {key}")

        recorded_policy_file = policy_meta.get("policy_file", "")
        if recorded_policy_file != str(policy_path):
            errors.append("policy file path does not match current policy path")
        if policy_meta.get("policy_sha256", "") != sha256_file(policy_path):
            errors.append("policy sha256 does not match current policy file")

        required_checks_path = policy_meta.get("required_checks_manifest", "")
        if not required_checks_path:
            errors.append("missing required_checks_manifest in provenance")
        else:
            required_checks_file = Path(required_checks_path)
            if not required_checks_file.is_file():
                errors.append(f"missing required checks manifest: {required_checks_path}")
            elif policy_meta.get("required_checks_manifest_sha256", "") != sha256_file(required_checks_file):
                errors.append("required checks manifest sha256 does not match current file")

        allowed_support_classes = set(policy.get("allowed_support_classes", []))
        recorded_allowed_support_classes = set(taxonomy.get("allowed_support_classes", []))
        if recorded_allowed_support_classes != allowed_support_classes:
            errors.append("recorded allowed support classes do not match policy")

        recorded_manifest_paths: set[str] = set()
        expected_manifest_paths: set[str] = set()
        for manifest_key, manifest_path in policy.get("manifests", {}).items():
            if manifest_key == "required_checks":
                continue
            expected_manifest_paths.add(manifest_path)
        for manifest in taxonomy.get("manifests", []):
            manifest_path = Path(manifest.get("path", ""))
            if manifest.get("path", ""):
                recorded_manifest_paths.add(manifest.get("path", ""))
            if not manifest_path.is_file():
                errors.append(f"missing taxonomy manifest: {manifest.get('path', '')}")
                continue
            if manifest.get("sha256", "") != sha256_file(manifest_path):
                errors.append(f"taxonomy manifest sha256 does not match current file: {manifest_path}")
            for support_class in manifest.get("support_classes", []):
                if support_class not in allowed_support_classes:
                    errors.append(f"taxonomy manifest uses unsupported support_class: {support_class}")
        if recorded_manifest_paths != expected_manifest_paths:
            errors.append("taxonomy manifest set does not match promotion policy manifests")

        artifact_source_commit = provenance.get("git", {}).get("artifact_source_commit", "")
        if not artifact_source_commit or artifact_source_commit == "unknown":
            errors.append("missing artifact_source_commit in provenance")
        elif source_path_value:
            blob = git_show_blob(artifact_source_commit, source_path_value)
            if blob is None:
                errors.append("artifact_source_commit does not resolve the recorded source path")
            else:
                if sha256_bytes(blob) != source.get("sha256", ""):
                    errors.append("artifact_source_commit does not match recorded source sha256")

        canonical_plane = trust_planes.get("canonical", {})
        supplemental_plane = trust_planes.get("supplemental", {})
        if canonical_plane.get("plane", "") != "repo_local_provenance":
            errors.append("canonical trust plane is not repo_local_provenance")
        if canonical_plane.get("status", "") != "pass":
            errors.append("canonical trust plane is not pass")
        if canonical_plane.get("verification_entrypoint", "") != policy_meta.get("verification_entrypoint", ""):
            errors.append("canonical trust plane verification entrypoint does not match policy")
        if supplemental_plane.get("plane", "") != "repo_local_reproducible_bootstrap":
            errors.append("supplemental trust plane is not repo_local_reproducible_bootstrap")
        if supplemental_plane.get("status", "") != "pass":
            errors.append("supplemental trust plane is not pass")
        if supplemental_plane.get("entrypoint", "") != policy["entrypoints"].get("reproducible_bootstrap_gate", ""):
            errors.append("supplemental trust plane entrypoint does not match policy")
        if supplemental_plane.get("artifact_sha256", "") != actual_sha256:
            errors.append("supplemental trust plane artifact sha256 does not match checked artifact")
        if supplemental_plane.get("gen1_sha256", "") != actual_sha256:
            errors.append("supplemental trust plane gen1 sha256 does not match checked artifact")
        if supplemental_plane.get("gen2_sha256", "") != actual_sha256:
            errors.append("supplemental trust plane gen2 sha256 does not match checked artifact")
        if supplemental_plane.get("gen3_sha256", "") != actual_sha256:
            errors.append("supplemental trust plane gen3 sha256 does not match checked artifact")

        integrity_payload = dict(provenance)
        integrity_payload.pop("integrity", None)
        expected_payload_sha = stable_sha256_payload(integrity_payload)
        if integrity.get("payload_sha256", "") != expected_payload_sha:
            errors.append("integrity payload sha256 does not match canonical provenance payload")

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
            "source_sha256_match_current_tree": bool(current_source_sha and current_source_sha == source.get("sha256", "")),
            "promotion_status_pass": promotion.get("overall_status") == "pass",
            "fixed_point_md5_match": bool(gen2_md5 and gen3_md5 and gen2_md5 == gen3_md5),
            "required_gates_present": all(gate_status.get(name) == "pass" for name in required_gate_names),
            "bootstrap_status_pass": bootstrap_status in ("", "pass"),
            "policy_sha256_match": (
                bool(policy_meta.get("policy_sha256", "")) and policy_path.is_file() and policy_meta.get("policy_sha256", "") == sha256_file(policy_path)
            ),
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
