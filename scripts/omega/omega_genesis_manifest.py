#!/usr/bin/env python3
"""Build signed Omega Genesis manifest for cold boot reproducibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ed25519

SCHEMA = "sounio.omega.genesis.v1.0"
DEFAULT_ARTIFACTS = (
    "bootstrap/policies/policy.v2.json",
    "artifacts/omega/baseline_freeze.v1.json",
    "artifacts/omega/governance_attestation.v1.json",
    "artifacts/fpga/fpga_seed_report.json",
    "artifacts/fpga/hardware_epistemic_power_live.v1.json",
    "artifacts/fpga/hardware_resource_trend.v2.json",
    "artifacts/sass/omega/sass_patch_report.json",
    "artifacts/ptx/omega/ptx_launch_report.json",
    "artifacts/quantum/omega/quantum_conformance.json",
    "hardware/fpga/k_axi_merkle_root_lane.v",
    "hardware/rtl/qir/omega_genesis_emitter.sio",
    "hardware/rtl/kaxi/merkle_root_lane.sio",
)
DEFAULT_MARKERS = (
    "QIR_GENESIS_EMITTER_PASS",
    "MERKLE_ROOT_PASS",
    "RESOURCE_ZERO_DRIFT_PASS",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Omega genesis manifest builder")
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="additional artifact path (repeatable)",
    )
    parser.add_argument(
        "--gate-log",
        default="artifacts/omega_sprint1_gate.log",
        help="gate marker log path",
    )
    parser.add_argument(
        "--canonical-privkey",
        default="keys/bootstrap_ed25519",
        help="canonical Ed25519 private key path (hex)",
    )
    parser.add_argument(
        "--canonical-pubkey",
        default="keys/bootstrap_ed25519.pub",
        help="canonical Ed25519 public key path (hex)",
    )
    parser.add_argument(
        "--canonical-timestamp",
        default="keys/bootstrap_ed25519.created_at",
        help="canonical bootstrap timestamp path",
    )
    parser.add_argument(
        "--out",
        default="artifacts/omega/omega_genesis.v1.0.json",
        help="output manifest path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="require all artifacts and required markers",
    )
    return parser.parse_args()


def read_hex(path: Path, expected_len: int, label: str) -> str:
    try:
        text = path.read_text().strip()
    except OSError as exc:
        raise SystemExit(f"unable to read {label} {path}: {exc}") from exc
    if len(text) != expected_len:
        raise SystemExit(
            f"invalid {label} length in {path}: expected {expected_len} hex chars, got {len(text)}"
        )
    try:
        bytes.fromhex(text)
    except ValueError as exc:
        raise SystemExit(f"invalid {label} hex in {path}: {exc}") from exc
    return text


def canonical_pubkey_fingerprint(path: Path) -> str:
    pub_hex = read_hex(path, 64, "canonical public key")
    return sha256_bytes(bytes.fromhex(pub_hex))


def collect_artifacts(paths: list[str], strict: bool) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    missing: list[str] = []
    for rel in paths:
        path = Path(rel)
        if not path.exists():
            missing.append(rel)
            continue
        digest = sha256_bytes(path.read_bytes())
        records.append({"path": rel, "sha256": digest})
    if strict and missing:
        raise SystemExit(f"missing required genesis artifacts: {missing}")
    return sorted(records, key=lambda item: item["path"]), sorted(missing)


def read_gate_markers(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = [line.strip() for line in path.read_text().splitlines()]
    except OSError:
        return []
    return [line for line in lines if line]


def main() -> int:
    args = parse_args()
    artifacts = list(DEFAULT_ARTIFACTS)
    for item in args.artifact:
        value = item.strip()
        if value and value not in artifacts:
            artifacts.append(value)

    artifact_records, missing_artifacts = collect_artifacts(artifacts, args.strict)
    digest_lines = [f"{row['path']}:{row['sha256']}" for row in artifact_records]
    aggregate_sha256 = sha256_bytes("\n".join(digest_lines).encode("utf-8"))
    merkle_root_sha256 = sha256_bytes(
        "|".join(row["sha256"] for row in artifact_records).encode("utf-8")
    )

    canonical_priv_hex = read_hex(Path(args.canonical_privkey), 64, "canonical private key")
    canonical_pub_fingerprint = canonical_pubkey_fingerprint(Path(args.canonical_pubkey))

    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(canonical_priv_hex))
    signature_hex = private_key.sign(aggregate_sha256.encode("utf-8")).hex()

    gate_markers = read_gate_markers(Path(args.gate_log))
    marker_set = set(gate_markers)
    missing_markers = [marker for marker in DEFAULT_MARKERS if marker not in marker_set]
    if args.strict and missing_markers:
        raise SystemExit(f"missing required genesis markers in gate log: {missing_markers}")

    canonical_bootstrap_timestamp = ""
    try:
        canonical_bootstrap_timestamp = Path(args.canonical_timestamp).read_text().strip()
    except OSError:
        canonical_bootstrap_timestamp = ""

    payload = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "artifact_count": len(artifact_records),
        "artifacts": artifact_records,
        "missing_artifacts": missing_artifacts,
        "aggregate_sha256": aggregate_sha256,
        "merkle_root_sha256": merkle_root_sha256,
        "canonical_pubkey_fingerprint": canonical_pub_fingerprint,
        "canonical_bootstrap_timestamp": canonical_bootstrap_timestamp,
        "signature_hex": signature_hex,
        "signed": True,
        "gate_markers": gate_markers,
        "missing_required_markers": missing_markers,
        "cold_boot_ready": len(missing_artifacts) == 0 and len(missing_markers) == 0,
        "bootstrap_command": "PATH=target/debug:$PATH bash scripts/omega_sprint1_gate.sh --strict --report-full",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    print(
        "omega_genesis_manifest: "
        f"artifacts={len(artifact_records)} "
        f"missing={len(missing_artifacts)} "
        f"missing_markers={len(missing_markers)} "
        f"cold_boot_ready={str(payload['cold_boot_ready']).lower()} "
        f"out={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
