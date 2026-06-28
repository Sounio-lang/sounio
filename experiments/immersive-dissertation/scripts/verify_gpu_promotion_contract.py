#!/usr/bin/env python3
"""Validate the hard-gate contract for promoting WebGPU claims."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REQUIRED_PASS_MARKERS = {
    "IMMERSIVE_VALIDATION_BUNDLE_PASS",
    "WEBGPU_RUNTIME_PASS",
    "WEBGPU_PBPK_KERNEL_CONTRACT_PASS",
    "WEBGPU_PBPK_KERNEL_RUNTIME_PASS",
    "VALIDATION_BUNDLE_SUMMARY_PASS",
}

FORBIDDEN_PROMOTION_MARKERS = {
    "WEBGPU_RUNTIME_NOT_AVAILABLE",
    "WEBGPU_RUNTIME_TIMEOUT",
    "WEBGPU_RUNTIME_BROWSER_LAUNCH_FAIL",
    "WEBGPU_PBPK_KERNEL_RUNTIME_NOT_AVAILABLE",
    "WEBGPU_PBPK_KERNEL_RUNTIME_BROWSER_LAUNCH_FAIL",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_gpu_promotion_contract.py <render-quality-contract.json>", file=sys.stderr)
        return 2

    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    errors: list[str] = []
    package = payload.get("gpu_promotion_package", {})
    if package.get("status") != "ready_for_gpu_host_execution":
        errors.append("gpu promotion package must be ready_for_gpu_host_execution")
    if "verify_webgpu_pbpk_kernel_contract.py" not in package.get("compute_kernel_verifier", ""):
        errors.append("gpu package missing static kernel verifier")
    runtime_verifier = package.get("compute_kernel_runtime_verifier", "")
    if "verify_webgpu_pbpk_kernel_runtime.mjs" not in runtime_verifier or "--require-webgpu" not in runtime_verifier:
        errors.append("gpu package missing required runtime kernel verifier")
    hard_gate = package.get("hard_gate_command", "")
    if "run_validation_bundle.py" not in hard_gate or "--require-webgpu" not in hard_gate:
        errors.append("hard gate command must require WebGPU")
    summary_gate = package.get("summary_verifier_command", "")
    if "verify_validation_bundle_summary.py" not in summary_gate or "--require-webgpu-proof" not in summary_gate:
        errors.append("summary verifier must require WebGPU proof")
    if REQUIRED_PASS_MARKERS - set(package.get("required_pass_markers", [])):
        errors.append("gpu promotion package missing pass markers")
    if FORBIDDEN_PROMOTION_MARKERS - set(package.get("fallback_markers_forbidden_for_promotion", [])):
        errors.append("gpu promotion package missing forbidden fallback markers")
    if "only the hard gate plus summary verifier" not in package.get("claim_boundary", ""):
        errors.append("gpu promotion claim boundary must require hard gate plus summary verifier")

    if errors:
        print("GPU_PROMOTION_CONTRACT_FAIL")
        for error in errors:
            print(error)
        return 1
    print("GPU_PROMOTION_CONTRACT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
