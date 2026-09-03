#!/usr/bin/env python3
"""Verify GPU Knowledge Vec4 DGX runtime receipt boundaries.

Modes:
- runtime-pass: require a real DGX/CUDA launch receipt.
- not-run: require a conservative package-only/not-launched receipt.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


PASS_MARKER = "PASS gpu_knowledge_vec4_aggregate_marker"


def fail(message: str) -> None:
    print(f"gpu_knowledge_vec4_runtime_receipt_verify: FAIL {message}", file=sys.stderr)
    raise SystemExit(1)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "receipt",
        nargs="?",
        default="artifacts/gpu/dgx_spark_public_gpu_gate.v1.json",
        help="DGX Spark public GPU gate JSON receipt",
    )
    parser.add_argument(
        "--mode",
        choices=("runtime-pass", "not-run"),
        default="runtime-pass",
        help="receipt state to require",
    )
    return parser.parse_args(argv[1:])


def require_common(receipt: dict) -> dict:
    if receipt.get("schema") != "sounio.dgx-spark-public-gpu-gate.v1":
        fail("bad schema")
    if receipt.get("status") != "pass":
        fail("receipt status is not pass")
    marker = receipt.get("gpu_knowledge_vec4_marker")
    if not isinstance(marker, dict):
        fail("missing marker block")
    if marker.get("enabled") is not True:
        fail("marker route was not enabled")
    if marker.get("kernel") != "gpu_knowledge_vec4_aggregate_marker":
        fail("bad marker kernel")
    if marker.get("copyback_offsets_bytes") != [0, 32, 64, 96]:
        fail("bad marker copyback offsets")
    if marker.get("expected_value_lanes") != [1.0, 2.0, 3.0, 4.0]:
        fail("bad marker expected lanes")
    boundaries = set(receipt.get("boundaries", []))
    for boundary in (
        "gpu_knowledge_vec4_marker_is_opt_in",
        "gpu_knowledge_vec4_marker_runtime_claim_requires_runtime_output",
        "does_not_claim_automatic_backend_pack_unpack",
        "does_not_claim_imported_runtime_fixture",
    ):
        if boundary not in boundaries:
            fail(f"missing boundary: {boundary}")
    return marker


def require_runtime_pass(receipt: dict, marker: dict) -> None:
    if receipt.get("package_only") is True:
        fail("runtime-pass receipt cannot be package-only")
    if marker.get("status") != "runtime_pass":
        fail("marker status is not runtime_pass")
    runtime_output = marker.get("runtime_output", "")
    if PASS_MARKER not in runtime_output:
        fail("marker runtime output missing PASS marker")
    remote = receipt.get("remote", {})
    for key in ("hostname", "uname_m", "ptxas_version", "nvcc_version"):
        if not remote.get(key):
            fail(f"missing remote {key}")
    boundaries = set(receipt.get("boundaries", []))
    if "dgx_spark_is_cuda_toolchain_and_runtime_authority" not in boundaries:
        fail("runtime receipt missing DGX authority boundary")
    if "package_only_does_not_claim_dgx_toolchain_or_runtime" in boundaries:
        fail("runtime receipt still carries package-only nonclaim")


def require_not_run(receipt: dict, marker: dict) -> None:
    if marker.get("status") in {"runtime_pass", "runtime_output_missing_pass_marker"}:
        fail("not-run receipt has runtime status")
    if marker.get("runtime_output"):
        fail("not-run receipt has marker runtime output")
    if receipt.get("package_only") is not True:
        fail("not-run receipt should be package-only in this lane")
    boundaries = set(receipt.get("boundaries", []))
    for boundary in (
        "package_only_no_remote_ssh",
        "package_only_does_not_claim_dgx_toolchain_or_runtime",
    ):
        if boundary not in boundaries:
            fail(f"missing package-only boundary: {boundary}")
    if "dgx_spark_is_cuda_toolchain_and_runtime_authority" in boundaries:
        fail("not-run receipt overclaims DGX authority")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    path = pathlib.Path(args.receipt)
    if not path.exists():
        fail(f"missing receipt: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    marker = require_common(receipt)
    if args.mode == "runtime-pass":
        require_runtime_pass(receipt, marker)
    else:
        require_not_run(receipt, marker)
    print(
        "gpu_knowledge_vec4_runtime_receipt_verify: PASS "
        f"mode={args.mode} receipt={path} marker_status={marker.get('status')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
