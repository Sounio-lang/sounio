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

from cryptography.hazmat.primitives.asymmetric import ed25519

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
    parser.add_argument(
        "--canonical-privkey",
        default=os.environ.get("OMEGA_CANONICAL_PRIVKEY", ""),
        help="Canonical bootstrap private key path (32-byte Ed25519 hex)",
    )
    parser.add_argument(
        "--canonical-pubkey",
        default=os.environ.get("OMEGA_CANONICAL_PUBKEY", "keys/bootstrap_ed25519.pub"),
        help="Canonical bootstrap public key path (32-byte Ed25519 hex)",
    )
    parser.add_argument(
        "--canonical-timestamp",
        default=os.environ.get(
            "OMEGA_CANONICAL_BOOTSTRAP_TIMESTAMP_PATH",
            "keys/bootstrap_ed25519.created_at",
        ),
        help="Canonical bootstrap timestamp path",
    )
    parser.add_argument(
        "--policy",
        default="bootstrap/policies/policy.v2.json",
        help="Optimization policy path used for canonical signing metadata",
    )
    parser.add_argument(
        "--baseline-freeze",
        default="artifacts/omega/baseline_freeze.v1.json",
        help="Baseline freeze artifact used for digest pin lineage",
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


def read_hex_key(path: Path, expected_len: int, key_label: str) -> str:
    try:
        raw = path.read_text().strip()
    except OSError as exc:
        raise SystemExit(f"unable to read {key_label} {path}: {exc}") from exc
    if len(raw) != expected_len:
        raise SystemExit(
            f"invalid {key_label} length in {path}: expected {expected_len} hex chars, got {len(raw)}"
        )
    try:
        bytes.fromhex(raw)
    except ValueError as exc:
        raise SystemExit(f"invalid {key_label} hex in {path}: {exc}") from exc
    return raw


def sign_digest_canonical(priv_hex: str, aggregate_digest: str) -> str:
    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(
        bytes.fromhex(priv_hex)
    )
    signature = private_key.sign(aggregate_digest.encode("utf-8"))
    return signature.hex()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def canonical_pubkey_fingerprint(path: Path) -> str:
    try:
        raw = path.read_text().strip()
    except OSError as exc:
        raise SystemExit(f"unable to read canonical pubkey {path}: {exc}") from exc
    if len(raw) != 64:
        raise SystemExit(f"invalid canonical pubkey length in {path}: expected 64 hex chars")
    try:
        pub_bytes = bytes.fromhex(raw)
    except ValueError as exc:
        raise SystemExit(f"invalid canonical pubkey hex in {path}: {exc}") from exc
    return sha256_bytes(pub_bytes)


def read_bootstrap_timestamp(path: Path) -> str:
    try:
        return path.read_text().strip()
    except OSError:
        return ""


def read_policy_sign_metadata(path: Path) -> tuple[str, str, str]:
    if not path.exists():
        return "", "", ""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "", "", ""
    if not isinstance(payload, dict):
        return "", "", ""
    fingerprint = str(payload.get("canonical_fingerprint", "")).strip()
    signed_at = str(payload.get("canonical_signed_at_utc", "")).strip()
    pinned_digest = str(payload.get("pinned_digest_sha256", "")).strip()
    return fingerprint, signed_at, pinned_digest


def read_baseline_freeze_lineage(path: Path) -> tuple[str, str]:
    if not path.exists():
        return "", ""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return "", ""
    if not isinstance(payload, dict):
        return "", ""
    digest = payload.get("freeze_digest_sha256", "")
    freeze_policy_pinned = payload.get("policy_pinned_digest_sha256", "")
    digest_text = str(digest).strip() if isinstance(digest, str) else ""
    freeze_policy_pinned_text = (
        str(freeze_policy_pinned).strip() if isinstance(freeze_policy_pinned, str) else ""
    )
    return digest_text, freeze_policy_pinned_text


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
    canonical_privkey_path = args.canonical_privkey.strip()
    if canonical_privkey_path and Path(canonical_privkey_path).exists():
        priv_hex = read_hex_key(
            Path(canonical_privkey_path), 64, "canonical private key"
        )
        signature = sign_digest_canonical(priv_hex, aggregate_digest)
        signed = True
        if not key_id:
            key_id = "canonical-bootstrap-ed25519"
    elif args.private_key.strip():
        signature = sign_digest(Path(args.private_key.strip()), aggregate_digest)
        signed = True
        if not key_id:
            key_id = "manual-override"

    canonical_pubkey_path = Path(args.canonical_pubkey)
    canonical_pubkey_fpr = canonical_pubkey_fingerprint(canonical_pubkey_path)
    canonical_bootstrap_ts = read_bootstrap_timestamp(Path(args.canonical_timestamp))
    policy_fingerprint, policy_signed_at, policy_pinned_digest = read_policy_sign_metadata(
        Path(args.policy)
    )
    baseline_freeze_digest, baseline_freeze_policy_pinned_digest = read_baseline_freeze_lineage(
        Path(args.baseline_freeze)
    )
    pinned_digest_match = (
        bool(policy_pinned_digest)
        and bool(baseline_freeze_policy_pinned_digest)
        and policy_pinned_digest == baseline_freeze_policy_pinned_digest
    )

    payload = {
        "schema": SCHEMA,
        "artifact_count": len(records),
        "artifacts": sorted(records, key=lambda x: x["path"]),
        "missing_artifacts": sorted(missing),
        "aggregate_sha256": aggregate_digest,
        "canonical_pubkey_path": str(canonical_pubkey_path),
        "canonical_pubkey_fingerprint": canonical_pubkey_fpr,
        "canonical_bootstrap_timestamp": canonical_bootstrap_ts,
        "policy_sign_fingerprint": policy_fingerprint,
        "policy_sign_timestamp": policy_signed_at,
        "policy_pinned_digest_sha256": policy_pinned_digest,
        "baseline_freeze_digest_sha256": baseline_freeze_digest,
        "baseline_freeze_policy_pinned_digest_sha256": baseline_freeze_policy_pinned_digest,
        "pinned_digest_match": pinned_digest_match,
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
        f"canonical_fpr={canonical_pubkey_fpr} "
        f"policy_sign_fpr={policy_fingerprint or 'missing'} "
        f"pinned_match={str(pinned_digest_match).lower()} "
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
    if args.strict and not canonical_bootstrap_ts:
        print(
            "omega_governance_attest: canonical bootstrap timestamp missing in strict mode",
            file=sys.stderr,
        )
        return 2
    if args.strict and (not policy_fingerprint or not policy_signed_at):
        print(
            "omega_governance_attest: policy sign metadata missing in strict mode "
            f"(policy={args.policy})",
            file=sys.stderr,
        )
        return 2
    if args.strict and (not policy_pinned_digest or not baseline_freeze_policy_pinned_digest):
        print(
            "omega_governance_attest: pinned digest lineage missing in strict mode "
            f"(policy={args.policy}, baseline_freeze={args.baseline_freeze})",
            file=sys.stderr,
        )
        return 2
    if args.strict and not pinned_digest_match:
        print(
            "omega_governance_attest: pinned digest mismatch in strict mode "
            f"(policy_pinned={policy_pinned_digest}, baseline_freeze_policy_pinned={baseline_freeze_policy_pinned_digest})",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
