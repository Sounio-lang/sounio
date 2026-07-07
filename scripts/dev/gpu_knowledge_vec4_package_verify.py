#!/usr/bin/env python3
"""Verify a GPU Knowledge Vec4 DGX package manifest.

This verifier is intentionally local-only: it checks file integrity and launch
contract shape, but does not claim remote DGX toolchain or CUDA runtime proof.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys


REQUIRED_SCHEMA = "sounio.gpu-knowledge-vec4-dgx-package.v1"
REQUIRED_FILES = ("ptx", "runtime_harness", "local_ptxas_cubin")


def fail(message: str) -> None:
    print(f"gpu_knowledge_vec4_package_verify: FAIL {message}", file=sys.stderr)
    raise SystemExit(1)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_manifest_path(argv: list[str]) -> pathlib.Path:
    if len(argv) > 2:
        fail("usage: gpu_knowledge_vec4_package_verify.py [manifest.json]")
    if len(argv) == 2:
        return pathlib.Path(argv[1])
    return pathlib.Path("artifacts/gpu/dgx_spark_public_gpu_package/gpu_knowledge_vec4_package_manifest.v1.json")


def main(argv: list[str]) -> int:
    manifest_path = resolve_manifest_path(argv)
    if not manifest_path.exists():
        fail(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("schema") != REQUIRED_SCHEMA:
        fail("bad schema")
    if manifest.get("status") != "pass":
        fail("manifest status is not pass")

    contract = manifest.get("runtime_launch_contract", {})
    if contract.get("kernel") != "gpu_knowledge_vec4_aggregate_marker":
        fail("bad kernel")
    if contract.get("params") != ["out_ptr"]:
        fail("bad params")
    if contract.get("copyback_offsets_bytes") != [0, 32, 64, 96]:
        fail("bad copyback offsets")
    if contract.get("expected_value_lanes") != [1.0, 2.0, 3.0, 4.0]:
        fail("bad expected value lanes")
    if contract.get("status") != "local_package_only_not_remote_not_launched":
        fail("manifest overclaims runtime status")

    boundaries = set(manifest.get("boundaries", []))
    for boundary in (
        "local_ptxas_package_proof",
        "package_only_no_remote_ssh",
        "does_not_claim_dgx_toolchain_or_runtime",
        "does_not_claim_cuda_device_runtime_execution",
    ):
        if boundary not in boundaries:
            fail(f"missing boundary: {boundary}")

    files = manifest.get("files", {})
    manifest_dir = manifest_path.parent
    verified = []
    for key in REQUIRED_FILES:
        entry = files.get(key)
        if not isinstance(entry, dict):
            fail(f"missing file entry: {key}")
        path_text = entry.get("path")
        if not isinstance(path_text, str) or not path_text:
            fail(f"bad path for file entry: {key}")
        path = pathlib.Path(path_text)
        if not path.is_absolute():
            path = pathlib.Path.cwd() / path
        if not path.exists() and (manifest_dir / pathlib.Path(path_text).name).exists():
            path = manifest_dir / pathlib.Path(path_text).name
        if not path.exists():
            fail(f"missing package file for {key}: {entry.get('path')}")
        data_size = path.stat().st_size
        if data_size != entry.get("bytes"):
            fail(f"byte size mismatch for {key}")
        digest = sha256(path)
        if digest != entry.get("sha256"):
            fail(f"sha256 mismatch for {key}")
        verified.append(key)

    print(
        "gpu_knowledge_vec4_package_verify: PASS "
        f"manifest={manifest_path} files={','.join(verified)} "
        "status=local_package_only_not_remote_not_launched"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
