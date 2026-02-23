#!/usr/bin/env python3
"""Generate governance artifact attestation for Sprint-1 Omega pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA = "sounio.omega.governance-attestation.v1"
DEFAULT_ARTIFACTS = (
    "artifacts/omega/shadow_audit.v1.json",
    "artifacts/omega/rl_readiness_bridge.v1.json",
    "artifacts/omega/rl_readiness_trend.v1.json",
    "artifacts/omega/rl_readiness_replay.v1.json",
    "artifacts/omega/policy_mode_guard.v1.json",
    "bootstrap/policies/rl_readiness.evidence.json",
    "artifacts/fpga/hardware_epistemic_power_live.v1.json",
    "artifacts/fpga/hardware_epistemic_power_live_trend.v1.json",
    "artifacts/ptx/omega/ptx_launch_report.json",
    "artifacts/sass/omega/sass_patch_report.json",
    "artifacts/quantum/omega/quantum_conformance.json",
    "artifacts/omega/performance_summary.v1.json",
    "artifacts/omega/external_baseline_collection.v1.json",
    "artifacts/omega/baseline_freeze.v1.json",
    "artifacts/omega/sprint1_release_readiness.v1.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Omega governance attestation generator"
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Artifact path to include (repeatable). Defaults to governance core set.",
    )
    parser.add_argument(
        "--private-key",
        default="",
        help="Optional Ed25519 private key PEM for signing aggregate digest",
    )
    parser.add_argument(
        "--key-id",
        default="",
        help="Optional key id attached to signature metadata",
    )
    parser.add_argument(
        "--out",
        default="artifacts/omega/governance_attestation.v1.json",
        help="Output attestation artifact path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when artifacts are missing or signature is required but unavailable",
    )
    parser.add_argument(
        "--require-signature",
        action="store_true",
        help="Require signature generation",
    )
    return parser.parse_args()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise SystemExit(f"unable to read artifact {path}: {exc}") from exc


def sign_digest(private_key: Path, aggregate_digest: str) -> str:
    if not private_key.exists():
        raise SystemExit(f"private key not found: {private_key}")
    with tempfile.NamedTemporaryFile(delete=False) as raw_file:
        raw_path = Path(raw_file.name)
        raw_file.write(aggregate_digest.encode("utf-8"))
    with tempfile.NamedTemporaryFile(delete=False) as sig_file:
        sig_path = Path(sig_file.name)
    try:
        proc = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-sign",
                "-inkey",
                str(private_key),
                "-rawin",
                "-in",
                str(raw_path),
                "-out",
                str(sig_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise SystemExit(
                "openssl signing failed for governance attestation:\n"
                f"{proc.stderr.strip()}"
            )
        return sig_path.read_bytes().hex()
    finally:
        try:
            os.unlink(raw_path)
        except OSError:
            pass
        try:
            os.unlink(sig_path)
        except OSError:
            pass


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    args = parse_args()
    artifacts = list(DEFAULT_ARTIFACTS)
    for raw in args.artifact:
        value = raw.strip()
        if value and value not in artifacts:
            artifacts.append(value)

    records = []
    missing = []
    for rel in artifacts:
        path = Path(rel)
        if not path.exists():
            missing.append(rel)
            continue
        digest = sha256_bytes(read_bytes(path))
        records.append({"path": rel, "sha256": digest})

    aggregate_lines = [f"{entry['path']}:{entry['sha256']}" for entry in sorted(records, key=lambda x: x["path"])]
    aggregate_digest = sha256_bytes("\n".join(aggregate_lines).encode("utf-8"))

    signature = ""
    key_id = args.key_id.strip()
    signed = False
    if args.private_key.strip():
        signature = sign_digest(Path(args.private_key.strip()), aggregate_digest)
        signed = True
        if not key_id:
            key_id = "manual-override"

    payload = {
        "schema": SCHEMA,
        "artifact_count": len(records),
        "artifacts": sorted(records, key=lambda x: x["path"]),
        "missing_artifacts": sorted(missing),
        "aggregate_sha256": aggregate_digest,
        "signed": signed,
        "key_id": key_id,
        "signature_hex": signature,
    }
    out_path = Path(args.out)
    write_json(out_path, payload)

    print(
        "omega_governance_attest: "
        f"artifacts={len(records)} "
        f"missing={len(missing)} "
        f"signed={str(signed).lower()} "
        f"report={out_path}"
    )

    if args.strict and missing:
        for item in missing:
            print(f"omega_governance_attest: missing artifact {item}", file=sys.stderr)
        return 2
    if args.require_signature and not signed:
        print("omega_governance_attest: signature required but not produced", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
