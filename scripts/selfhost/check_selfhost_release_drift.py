#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


JOB_HEADER_RE = re.compile(r"^  ([A-Za-z0-9_-]+):\s*$")
JOB_NAME_RE = re.compile(r"^    name:\s*(.+?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect drift across selfhost policy, CI job names, and operator docs.")
    parser.add_argument("--policy", required=True)
    parser.add_argument("--required-checks", required=True)
    parser.add_argument("--ci-workflow", required=True)
    parser.add_argument("--release-workflow", required=True)
    parser.add_argument("--authority-doc", required=True)
    parser.add_argument("--release-doc", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    return parser.parse_args()


def extract_job_names(path: Path) -> list[str]:
    names: list[str] = []
    current_job = None
    for line in path.read_text(encoding="utf-8").splitlines():
        job_match = JOB_HEADER_RE.match(line)
        if job_match:
            current_job = job_match.group(1)
            continue
        if current_job is not None:
            name_match = JOB_NAME_RE.match(line)
            if name_match:
                names.append(name_match.group(1).strip().strip("'\""))
                current_job = None
    return names


def require_text(text: str, needle: str, label: str, errors: list[str]) -> None:
    if needle not in text:
        errors.append(f"missing {label}: {needle}")


def main() -> int:
    args = parse_args()

    policy = json.loads(Path(args.policy).read_text(encoding="utf-8"))
    checks = json.loads(Path(args.required_checks).read_text(encoding="utf-8"))
    authority_doc = Path(args.authority_doc).read_text(encoding="utf-8")
    release_doc = Path(args.release_doc).read_text(encoding="utf-8")
    ci_job_names = set(extract_job_names(Path(args.ci_workflow)))
    release_job_names = set(extract_job_names(Path(args.release_workflow)))

    errors: list[str] = []

    for entry in checks.get("required_checks", []):
      name = entry.get("name", "")
      if name not in ci_job_names:
          errors.append(f"required check missing from CI workflow: {name}")
      require_text(authority_doc, f"`{name}`", "authority doc required check", errors)
      require_text(release_doc, f"`{name}`", "release doc required check", errors)

    for entry in checks.get("informational_checks", []):
      name = entry.get("name", "")
      if name not in release_job_names:
          errors.append(f"informational check missing from release workflow: {name}")
      require_text(authority_doc, f"`{name}`", "authority doc informational check", errors)

    policy_entrypoints = policy.get("entrypoints", {})
    for label, rel_path in policy_entrypoints.items():
        if label in {"authority_gate", "focused_abi_gate", "promotion_entrypoint", "aarch64_runtime_gate", "reproducible_bootstrap_gate", "dual_trust_gate"}:
            require_text(release_doc, rel_path, f"release doc entrypoint {label}", errors)
        if label in {"authority_gate", "focused_abi_gate", "provenance_gate", "aarch64_runtime_gate", "dual_trust_gate"}:
            require_text(authority_doc, rel_path, f"authority doc entrypoint {label}", errors)

    require_text(authority_doc, "authority: repo_only", "authority doc metadata", errors)
    require_text(release_doc, "authority: repo_only", "release doc metadata", errors)

    required_checks_manifest = policy.get("manifests", {}).get("required_checks", "")
    require_text(authority_doc, required_checks_manifest, "authority doc required checks manifest", errors)
    require_text(release_doc, required_checks_manifest, "release doc required checks manifest", errors)

    report = {
        "schema": "sounio.selfhost_release_drift.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "errors": errors,
        "overall_status": "pass" if not errors else "fail",
    }

    Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Selfhost Release Drift",
        "",
        f"- overall_status: {report['overall_status']}",
    ]
    if errors:
        md_lines.extend(["", "## Errors", ""])
        for error in errors:
            md_lines.append(f"- {error}")
    Path(args.md_out).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
