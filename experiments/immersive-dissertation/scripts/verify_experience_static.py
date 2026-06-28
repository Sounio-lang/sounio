#!/usr/bin/env python3
"""Static guardrails for the immersive dissertation recovery surface."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_FILES = [
    "index.html",
    "src/main.js",
    "src/style.css",
    "data/render-quality-contract.json",
    "data/webgpu-pbpk-kernel-contract.json",
    "shaders/pbpk_release_compute.wgsl",
    "scripts/verify_webgpu_runtime.mjs",
    "scripts/verify_webgpu_pbpk_kernel_runtime.mjs",
]


REQUIRED_TEXT = {
    "index.html": [
        "runtime-badge",
        "scene-canvas",
        "curve-canvas",
        "clinical-firewall",
        "no raw observed C(t) in browser",
        "quality-label",
    ],
    "src/main.js": [
        "Canvas depth fallback",
        "2DGX ready",
        "WGSL",
        "model curve only; observed C(t) blocked",
        "CYP2D6",
        "venlafaxine",
    ],
    "shaders/pbpk_release_compute.wgsl": [
        "@compute @workgroup_size(64)",
        "@group(0) @binding(0)",
        "@group(0) @binding(1)",
        "release_fraction",
        "odv_parent_ratio",
    ],
    "scripts/verify_webgpu_runtime.mjs": [
        "WEBGPU_RUNTIME_PASS",
        "WEBGPU_RUNTIME_NOT_AVAILABLE",
        "device.lost",
    ],
    "scripts/verify_webgpu_pbpk_kernel_runtime.mjs": [
        "WEBGPU_PBPK_KERNEL_RUNTIME_PASS",
        "WEBGPU_PBPK_KERNEL_RUNTIME_NOT_AVAILABLE",
        "copyBufferToBuffer",
        "mapAsync",
        "dispatchWorkgroups",
        "not observed C(t) calibration",
    ],
}


def fail(message: str) -> int:
    print("IMMERSIVE_EXPERIENCE_STATIC_FAIL")
    print(message)
    return 1


def main() -> int:
    missing = [path for path in REQUIRED_FILES if not (ROOT / path).exists()]
    if missing:
        return fail(f"missing files: {missing}")

    errors: list[str] = []
    for rel_path, needles in REQUIRED_TEXT.items():
        text = (ROOT / rel_path).read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                errors.append(f"{rel_path}: missing {needle!r}")

    render_contract = json.loads((ROOT / "data/render-quality-contract.json").read_text(encoding="utf-8"))
    kernel_contract = json.loads((ROOT / "data/webgpu-pbpk-kernel-contract.json").read_text(encoding="utf-8"))
    if render_contract.get("current_host_evidence", {}).get("evidence_level") != "fallback_visibility_verified":
        errors.append("render contract must preserve fallback evidence boundary")
    if "not observed C(t) calibration" not in kernel_contract.get("claim_boundary", ""):
        errors.append("kernel contract must block observed C(t) calibration claim")
    forbidden = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in ["index.html", "src/main.js", "src/style.css"]
    )
    for needle in ["patient-specific dosing advice", "bioequivalence decision", "WebGPU PASS"]:
        if needle in forbidden:
            errors.append(f"browser surface contains forbidden claim {needle!r}")

    if errors:
        return fail("\n".join(errors))

    print("IMMERSIVE_EXPERIENCE_STATIC_PASS")
    print(f"checked_files={len(REQUIRED_FILES)} required_text_groups={len(REQUIRED_TEXT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
