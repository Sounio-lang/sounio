#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify selfhost promotion policy structure and referenced files.")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    return parser.parse_args()


def load_support_classes(manifest_key: str, manifest_path: Path) -> set[str]:
    classes: set[str] = set()
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row or not row[0] or row[0].startswith("#"):
                continue
            if manifest_key == "abi_parity":
                if len(row) < 3:
                    raise SystemExit(f"invalid abi parity manifest row in {manifest_path}: {row}")
                classes.add(row[2])
            else:
                if len(row) < 2:
                    raise SystemExit(f"invalid aarch64 manifest row in {manifest_path}: {row}")
                classes.add(row[1])
    return classes


def main() -> int:
    args = parse_args()
    policy_path = Path(args.policy)
    if not policy_path.is_file():
        raise SystemExit(f"missing policy file: {policy_path}")

    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if policy.get("schema") != "sounio.selfhost_promotion_policy.v1":
        errors.append(f"unexpected schema: {policy.get('schema', '')}")

    allowed_support_classes = set(policy.get("allowed_support_classes", []))
    if not allowed_support_classes:
        errors.append("allowed_support_classes is empty")

    for group_name in ["entrypoints", "reference_docs", "manifests"]:
        group = policy.get(group_name, {})
        if not isinstance(group, dict) or not group:
            errors.append(f"missing or empty policy group: {group_name}")
            continue
        for key, rel_path in group.items():
            path = Path(rel_path)
            if not path.is_file():
                errors.append(f"missing referenced file for {group_name}.{key}: {rel_path}")

    if not policy.get("blocking_authority_checks"):
        errors.append("blocking_authority_checks is empty")

    if not policy.get("required_provenance_fields"):
        errors.append("required_provenance_fields is empty")

    manifests = policy.get("manifests", {})
    for manifest_key, rel_path in manifests.items():
        if manifest_key == "required_checks":
            continue
        if not rel_path:
            errors.append(f"missing manifest path for {manifest_key}")
            continue
        manifest_path = Path(rel_path)
        if not manifest_path.is_file():
            continue
        for support_class in load_support_classes(manifest_key, manifest_path):
            if support_class not in allowed_support_classes:
                errors.append(
                    f"manifest {rel_path} uses unsupported support_class {support_class}"
                )

    report = {
        "schema": "sounio.selfhost_promotion_policy_verification.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "policy_path": str(policy_path),
        "errors": errors,
        "overall_status": "pass" if not errors else "fail",
    }

    Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Selfhost Promotion Policy",
        "",
        f"- overall_status: {report['overall_status']}",
        f"- policy_path: {policy_path}",
    ]
    if errors:
        md_lines.extend(["", "## Errors", ""])
        for error in errors:
            md_lines.append(f"- {error}")
    Path(args.md_out).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
