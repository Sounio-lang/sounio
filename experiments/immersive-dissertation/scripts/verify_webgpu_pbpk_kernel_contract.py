#!/usr/bin/env python3
"""Validate the WebGPU/WGSL PBPK kernel contract without executing GPU code."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]

REQUIRED_API = {
    "navigator.gpu must be present",
    "requestAdapter must return an adapter",
    "requestDevice must return a device",
    "device.lost must be handled or reported",
    "createShaderModule must accept the WGSL source",
    "createComputePipeline must build the PBPK compute pipeline",
    "compute pipeline must bind read-only input and read-write output storage buffers",
    "dispatchWorkgroups must cover the requested model horizon",
    "copyBufferToBuffer and mapAsync must read back finite output values",
}

REQUIRED_INPUTS = {
    "time_h",
    "ka_xr",
    "f_oral",
    "cl_cyp2d6",
    "cl_odv",
    "phenotype_scale",
    "parent_cmax_ng_ml",
    "odv_cmax_ng_ml",
}

REQUIRED_OUTPUTS = {
    "release_fraction",
    "parent_ng_ml",
    "odv_ng_ml",
    "odv_parent_ratio",
}


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_webgpu_pbpk_kernel_contract.py <webgpu-pbpk-kernel-contract.json>", file=sys.stderr)
        return 2

    contract_path = Path(sys.argv[1])
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if payload.get("schema") != "sounio.immersive_dissertation.webgpu_pbpk_kernel_contract.v1":
        errors.append("unexpected WebGPU PBPK kernel contract schema")
    if payload.get("status") != "ready_for_gpu_host_execution_not_promoted":
        errors.append("kernel contract must remain ready_for_gpu_host_execution_not_promoted")
    boundary = payload.get("claim_boundary", "")
    for needle in ["model replay only", "not observed C(t) calibration", "not a current-host WebGPU proof"]:
        if needle not in boundary:
            errors.append(f"claim boundary missing {needle!r}")

    if "ctx7 /gpuweb/gpuweb" not in json.dumps(payload.get("source_documents", [])):
        errors.append("contract must record ctx7 /gpuweb/gpuweb retrieval")

    missing_api = REQUIRED_API - set(payload.get("webgpu_api_requirements", []))
    if missing_api:
        errors.append(f"missing WebGPU API requirements: {', '.join(sorted(missing_api))}")

    runtime = payload.get("runtime_integration", {})
    if runtime.get("runtime_verifier") != "scripts/verify_webgpu_pbpk_kernel_runtime.mjs":
        errors.append("runtime integration must reference WebGPU PBPK runtime verifier")

    shader = payload.get("shader", {})
    shader_path = REPO / shader.get("path", "")
    wgsl = shader_path.read_text(encoding="utf-8") if shader_path.exists() else ""
    if not wgsl:
        errors.append(f"missing shader path {shader_path}")
    if shader.get("language") != "WGSL":
        errors.append("shader language must be WGSL")
    if shader.get("entry_point") != "main":
        errors.append("shader entry point must be main")
    if int(shader.get("workgroup_size", 0)) != 64:
        errors.append("shader workgroup size must be 64")
    if REQUIRED_INPUTS - set(shader.get("inputs", [])):
        errors.append("shader missing required inputs")
    if REQUIRED_OUTPUTS - set(shader.get("outputs", [])):
        errors.append("shader missing required outputs")

    for token in [
        "struct PbpkInput",
        "struct PbpkOutput",
        "@group(0) @binding(0) var<storage, read> input_state",
        "@group(0) @binding(1) var<storage, read_write> output_state",
        "@compute @workgroup_size(64)",
        "@builtin(global_invocation_id)",
        "arrayLength(&input_state)",
        "xr_release",
        "phenotype_scale",
        "odv_parent_ratio",
    ]:
        if token not in wgsl:
            errors.append(f"WGSL missing {token!r}")
    if re.search(r"observed|digitized|patient|dose_advice", wgsl, re.IGNORECASE):
        errors.append("WGSL must not contain observed-data or patient-advice terms")

    hard_gate = payload.get("hard_gate", {})
    runtime_verifier = hard_gate.get("runtime_compute_verifier", "")
    if "verify_webgpu_pbpk_kernel_runtime.mjs" not in runtime_verifier or "--require-webgpu" not in runtime_verifier:
        errors.append("hard gate must include runtime compute verifier")
    markers = set(hard_gate.get("required_markers", []))
    for marker in ["WEBGPU_RUNTIME_PASS", "WEBGPU_PBPK_KERNEL_RUNTIME_PASS", "WEBGPU_PBPK_KERNEL_CONTRACT_PASS"]:
        if marker not in markers:
            errors.append(f"hard gate missing marker {marker}")
    fallback_markers = set(hard_gate.get("fallback_markers_forbidden_for_promotion", []))
    for marker in ["WEBGPU_PBPK_KERNEL_RUNTIME_NOT_AVAILABLE", "WEBGPU_PBPK_KERNEL_RUNTIME_BROWSER_LAUNCH_FAIL"]:
        if marker not in fallback_markers:
            errors.append(f"hard gate missing forbidden fallback marker {marker}")

    if errors:
        print("WEBGPU_PBPK_KERNEL_CONTRACT_FAIL")
        for error in errors:
            print(error)
        return 1
    print("WEBGPU_PBPK_KERNEL_CONTRACT_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
