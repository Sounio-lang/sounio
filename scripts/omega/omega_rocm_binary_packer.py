#!/usr/bin/env python3
"""Deterministic bootstrap ROCm packer for Sounio parity gates.

This packer does NOT claim to emit vendor-native HSACO code objects.
It emits a Sounio-owned synthetic container with target/format metadata and
embedded PTX-oracle bytes so parity + attestation lanes can stay deterministic
while native ROCm ISA emission is being completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SCHEMA = "sounio.omega.rocm-packer.v1"


def _canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _build_container(fmt: str, target: str, ptx_bytes: bytes) -> tuple[bytes, dict]:
    ptx_sha = hashlib.sha256(ptx_bytes).hexdigest()
    entries = [
        {
            "kind": "ptx_oracle",
            "sha256": ptx_sha,
            "size_bytes": len(ptx_bytes),
        }
    ]
    if fmt == "hsaco":
        entries.append(
            {
                "kind": "hsaco_stub",
                "sha256": ptx_sha,
                "size_bytes": len(ptx_bytes),
            }
        )
    elif fmt == "multi":
        entries.append(
            {
                "kind": "hsaco_stub",
                "sha256": ptx_sha,
                "size_bytes": len(ptx_bytes),
            }
        )
        entries.append(
            {
                "kind": "bundle_stub",
                "sha256": ptx_sha,
                "size_bytes": len(ptx_bytes),
            }
        )

    manifest = {
        "schema": SCHEMA,
        "container_format": fmt,
        "target": target,
        "synthetic_container": True,
        "entries": entries,
    }
    header = {
        "magic": "SOUNIO_ROCM_PACK_V1",
        "target": target,
        "format": fmt,
        "manifest_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(),
    }

    header_bytes = _canonical(header)
    manifest_bytes = _canonical(manifest)
    payload = b"".join(
        [
            b"SOUNIO\x00ROCM\x00PACK\x00V1\x00",
            len(header_bytes).to_bytes(4, "little"),
            header_bytes,
            len(manifest_bytes).to_bytes(4, "little"),
            manifest_bytes,
            len(ptx_bytes).to_bytes(8, "little"),
            ptx_bytes,
        ]
    )

    meta = {
        "schema": SCHEMA,
        "target": target,
        "binary_format": fmt,
        "synthetic_container": True,
        "ptx_sha256": ptx_sha,
        "ptx_size_bytes": len(ptx_bytes),
        "container_sha256": hashlib.sha256(payload).hexdigest(),
        "container_size_bytes": len(payload),
        "entries": entries,
    }
    return payload, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Sounio deterministic ROCm binary packer")
    parser.add_argument("--ptx", required=True, help="Input PTX file path")
    parser.add_argument("--output", required=True, help="Output packed binary path")
    parser.add_argument("--format", required=True, choices=["hsaco", "multi"], help="Requested ROCm container format")
    parser.add_argument("--target", required=True, help="GPU target label (for example rocm-gfx942)")
    parser.add_argument("--meta-out", required=True, help="JSON metadata output path")
    args = parser.parse_args()

    ptx_path = Path(args.ptx)
    out_path = Path(args.output)
    meta_path = Path(args.meta_out)

    if not ptx_path.exists() or not ptx_path.is_file():
        raise SystemExit(f"missing PTX input: {ptx_path}")

    ptx_bytes = ptx_path.read_bytes()
    if not ptx_bytes:
        raise SystemExit(f"empty PTX input: {ptx_path}")

    payload, meta = _build_container(args.format, args.target, ptx_bytes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    out_path.write_bytes(payload)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
