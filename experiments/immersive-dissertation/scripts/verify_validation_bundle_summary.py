#!/usr/bin/env python3
"""Verify a persisted immersive validation bundle summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


FORBIDDEN_PROMOTION_MARKERS = [
    "WEBGPU_RUNTIME_NOT_AVAILABLE",
    "WEBGPU_RUNTIME_BROWSER_LAUNCH_FAIL",
    "WEBGPU_PBPK_KERNEL_RUNTIME_NOT_AVAILABLE",
    "WEBGPU_PBPK_KERNEL_RUNTIME_BROWSER_LAUNCH_FAIL",
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact_path(summary_path: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = summary_path.parent / candidate
    return candidate


def check_required_artifacts(summary_path: Path, summary: dict, errors: list[str]) -> None:
    artifacts = summary.get("artifacts", {})
    for key in ["screenshot", "webgpu_report", "webgpu_pbpk_kernel_report"]:
        value = artifacts.get(key)
        if not value:
            errors.append(f"missing artifact key {key}")
            continue
        path = artifact_path(summary_path, value)
        if not path.exists():
            errors.append(f"artifact does not exist: {key}={path}")


def check_hard_webgpu(summary_path: Path, summary: dict, errors: list[str]) -> None:
    if not summary.get("require_webgpu"):
        errors.append("hard proof summary must be produced with require_webgpu=true")
    if summary.get("webgpu_proof_required") is False:
        errors.append("hard proof summary cannot mark webgpu_proof_required=false")

    checks = {item.get("name"): item for item in summary.get("checks", [])}
    for name in ["webgpu_runtime", "webgpu_pbpk_kernel_runtime"]:
        check = checks.get(name)
        if not check:
            errors.append(f"missing hard check {name}")
            continue
        output = check.get("stdout", "") + "\n" + check.get("stderr", "")
        if check.get("status") != "pass":
            errors.append(f"{name} did not pass")
        for marker in FORBIDDEN_PROMOTION_MARKERS:
            if marker in output:
                errors.append(f"{name} contains fallback marker {marker}")

    webgpu_path = summary.get("artifacts", {}).get("webgpu_report")
    kernel_path = summary.get("artifacts", {}).get("webgpu_pbpk_kernel_report")
    if webgpu_path:
        webgpu = load_json(artifact_path(summary_path, webgpu_path))
        probe = webgpu.get("probe", {})
        for key in ["navigatorGpu", "adapterAvailable", "deviceAvailable"]:
            if not probe.get(key):
                errors.append(f"webgpu_report {key} not true")
        if webgpu.get("status") != "pass":
            errors.append("webgpu_report status not pass")
    if kernel_path:
        kernel = load_json(artifact_path(summary_path, kernel_path))
        probe = kernel.get("probe", {})
        for key in ["navigatorGpu", "adapterAvailable", "deviceAvailable"]:
            if not probe.get(key):
                errors.append(f"kernel_report {key} not true")
        if kernel.get("status") != "pass":
            errors.append("kernel_report status not pass")
        if len(kernel.get("kernel", {}).get("outputs", [])) < 4:
            errors.append("kernel_report missing compute outputs")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-webgpu-proof", action="store_true")
    parser.add_argument("summary")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    summary = load_json(summary_path)
    errors: list[str] = []
    if summary.get("schema") != "sounio.immersive_dissertation.validation_summary.v1":
        errors.append("unexpected summary schema")
    if "no new parameter estimation or clinical validation" not in summary.get("claim_boundary", ""):
        errors.append("summary missing illustrative replay clinical boundary")
    if "source_revision" not in summary:
        errors.append("summary missing source_revision")
    if not args.require_webgpu_proof and summary.get("webgpu_promotion_eligible") is not False:
        errors.append("fallback summary must have webgpu_promotion_eligible=false")
    if summary.get("status") != "pass" and not args.require_webgpu_proof:
        errors.append("summary status is not pass")
    if summary.get("failures") and not args.require_webgpu_proof:
        errors.append("summary has failures")
    check_required_artifacts(summary_path, summary, errors)
    if args.require_webgpu_proof:
        check_hard_webgpu(summary_path, summary, errors)

    if errors:
        print("VALIDATION_BUNDLE_SUMMARY_FAIL")
        print("\n".join(errors))
        return 1
    print("VALIDATION_BUNDLE_SUMMARY_PASS")
    print(f"checks={len(summary.get('checks', []))} artifacts={len(summary.get('artifacts', {}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
