#!/usr/bin/env python3
"""Freeze external baseline evidence into a deterministic Sprint-1 artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric import ed25519

SCHEMA = "sounio.omega.baseline-freeze.v1"
CONTRACT_SCHEMA = "sounio.independence.contract.v2"
MANIFEST_SCHEMA = "sounio.independence.baseline-set.v1"
EXTERNAL_SCHEMA = "sounio.omega.external-baseline-collection.v1"
PERFORMANCE_SCHEMA = "sounio.omega.performance-summary.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze baseline evidence for Sprint-1 governance"
    )
    parser.add_argument(
        "--contract",
        default="benchmarks/independence/contract.v2.json",
        help="Independence contract path",
    )
    parser.add_argument(
        "--baseline-manifest",
        default="benchmarks/independence/omega_sprint1_baselines.v1.json",
        help="Baseline manifest path",
    )
    parser.add_argument(
        "--external-collection",
        default="artifacts/omega/external_baseline_collection.v1.json",
        help="External baseline collection artifact path",
    )
    parser.add_argument(
        "--performance-summary",
        default="artifacts/omega/performance_summary.v1.json",
        help="Performance summary artifact path",
    )
    parser.add_argument(
        "--out",
        default="artifacts/omega/baseline_freeze.v1.json",
        help="Output baseline freeze artifact path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when freeze preconditions are not satisfied",
    )
    parser.add_argument(
        "--canonical-privkey",
        default=os.environ.get("OMEGA_CANONICAL_PRIVKEY", "keys/bootstrap_ed25519"),
        help="Canonical private key path (32-byte Ed25519 hex)",
    )
    parser.add_argument(
        "--canonical-pubkey",
        default=os.environ.get("OMEGA_CANONICAL_PUBKEY", "keys/bootstrap_ed25519.pub"),
        help="Canonical public key path (32-byte Ed25519 hex)",
    )
    parser.add_argument(
        "--canonical-timestamp",
        default=os.environ.get(
            "OMEGA_CANONICAL_BOOTSTRAP_TIMESTAMP_PATH",
            "keys/bootstrap_ed25519.created_at",
        ),
        help="Canonical key bootstrap timestamp file path",
    )
    parser.add_argument(
        "--policy",
        default="bootstrap/policies/policy.v2.json",
        help="Policy path used for pinned digest lineage metadata",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"unable to read JSON artifact {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid JSON payload in {path}: expected object")
    return payload


def read_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def parse_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


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


def canonical_pubkey_fingerprint(pub_hex: str) -> str:
    return hashlib.sha256(bytes.fromhex(pub_hex)).hexdigest()


def sign_digest_with_private_key(priv_hex: str, digest_hex: str) -> str:
    private_key = ed25519.Ed25519PrivateKey.from_private_bytes(bytes.fromhex(priv_hex))
    signature = private_key.sign(digest_hex.encode("utf-8"))
    return signature.hex()


def read_bootstrap_timestamp(path: Path) -> str:
    try:
        text = path.read_text().strip()
    except OSError:
        return ""
    return text


def read_policy_pinned_digest(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    value = payload.get("pinned_digest_sha256", "")
    return str(value).strip() if isinstance(value, str) else ""


def main() -> int:
    args = parse_args()
    contract_path = Path(args.contract)
    manifest_path = Path(args.baseline_manifest)
    external_path = Path(args.external_collection)
    performance_path = Path(args.performance_summary)

    contract = load_json(contract_path)
    manifest = load_json(manifest_path)
    external = load_json(external_path)
    performance = load_json(performance_path)

    errors: list[str] = []

    if contract.get("schema") != CONTRACT_SCHEMA:
        errors.append(
            f"{contract_path}: schema mismatch (expected {CONTRACT_SCHEMA}, got {contract.get('schema')})"
        )
    if manifest.get("schema") != MANIFEST_SCHEMA:
        errors.append(
            f"{manifest_path}: schema mismatch (expected {MANIFEST_SCHEMA}, got {manifest.get('schema')})"
        )
    if external.get("schema") != EXTERNAL_SCHEMA:
        errors.append(
            f"{external_path}: schema mismatch (expected {EXTERNAL_SCHEMA}, got {external.get('schema')})"
        )
    if performance.get("schema") != PERFORMANCE_SCHEMA:
        errors.append(
            f"{performance_path}: schema mismatch (expected {PERFORMANCE_SCHEMA}, got {performance.get('schema')})"
        )

    required_baselines = manifest.get("required_baselines", [])
    if not isinstance(required_baselines, list):
        required_baselines = []
        errors.append(f"{manifest_path}: required_baselines must be a list")

    collection_required = external.get("required_baselines", [])
    if not isinstance(collection_required, list):
        collection_required = []
        errors.append(f"{external_path}: required_baselines must be a list")

    baseline_count = parse_int(external.get("baseline_count"), default=len(required_baselines))
    available_count = parse_int(external.get("available_count"))
    missing_reports = external.get("missing_reports", [])
    if not isinstance(missing_reports, list):
        missing_reports = []
        errors.append(f"{external_path}: missing_reports must be a list")
    execution_failures = external.get("execution_failures", [])
    if not isinstance(execution_failures, list):
        execution_failures = []
        errors.append(f"{external_path}: execution_failures must be a list")

    rows = external.get("rows", [])
    if not isinstance(rows, list):
        rows = []
        errors.append(f"{external_path}: rows must be a list")

    minimum_samples = parse_int(performance.get("minimum_samples"), default=50)
    samples = parse_int(performance.get("samples"))
    threshold = parse_float(performance.get("threshold"), default=1.2)
    geomean = parse_float(performance.get("geomean_speedup"))
    measurement_mode = str(performance.get("measurement_mode", "unknown"))
    substitutions = performance.get("baseline_substitutions", [])
    if not isinstance(substitutions, list):
        substitutions = []
        errors.append(f"{performance_path}: baseline_substitutions must be a list")

    if baseline_count != len(required_baselines):
        errors.append(
            f"{external_path}: baseline_count={baseline_count} must match len(required_baselines)={len(required_baselines)}"
        )
    if set(required_baselines) != set(collection_required):
        errors.append(
            f"{external_path}: required_baselines mismatch against manifest"
        )
    if available_count > baseline_count:
        errors.append(
            f"{external_path}: available_count={available_count} cannot exceed baseline_count={baseline_count}"
        )
    if samples < 0:
        errors.append(f"{performance_path}: samples must be >= 0")

    if args.strict:
        if baseline_count <= 0:
            errors.append("baseline_count must be > 0 in --strict mode")
        if samples < minimum_samples:
            errors.append(
                f"{performance_path}: samples={samples} below minimum_samples={minimum_samples} in --strict mode"
            )
        if geomean < threshold:
            errors.append(
                f"{performance_path}: geomean_speedup={geomean:.6f} below threshold={threshold:.6f} in --strict mode"
            )
        if measurement_mode == "external_baseline_ingest":
            if available_count != baseline_count:
                errors.append(
                    f"{external_path}: external_baseline_ingest requires available_count==baseline_count ({available_count}!={baseline_count})"
                )
            if missing_reports:
                errors.append(
                    f"{external_path}: external_baseline_ingest requires missing_reports=[]"
                )
            if execution_failures:
                errors.append(
                    f"{external_path}: external_baseline_ingest requires execution_failures=[]"
                )
        elif available_count < baseline_count and not substitutions:
            errors.append(
                f"{performance_path}: substitutions required when available_count < baseline_count in non-ingest mode"
            )

    frozen_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        report_rel = str(row.get("report_path", ""))
        report_sha = ""
        if report_rel:
            report_path = Path(report_rel)
            if report_path.exists():
                report_sha = read_sha256(report_path)
        frozen_rows.append(
            {
                "baseline_name": str(row.get("baseline_name", "")),
                "report_path": report_rel,
                "report_sha256": report_sha,
                "available": bool(row.get("available", False)),
                "samples": parse_int(row.get("samples")),
                "reason": str(row.get("reason", "")),
                "entrypoint_return_code": parse_int(row.get("entrypoint_return_code")),
            }
        )

    digest_lines = [
        f"contract:{contract_path}:{read_sha256(contract_path)}",
        f"manifest:{manifest_path}:{read_sha256(manifest_path)}",
        f"external_collection:{external_path}:{read_sha256(external_path)}",
        f"performance_summary:{performance_path}:{read_sha256(performance_path)}",
    ]
    for row in sorted(frozen_rows, key=lambda item: item["baseline_name"]):
        digest_lines.append(
            "row:{name}:{available}:{samples}:{sha}:{reason}".format(
                name=row["baseline_name"],
                available=str(row["available"]).lower(),
                samples=row["samples"],
                sha=row["report_sha256"],
                reason=row["reason"],
            )
        )
    freeze_digest = hashlib.sha256("\n".join(digest_lines).encode("utf-8")).hexdigest()

    canonical_priv_path = Path(args.canonical_privkey)
    canonical_pub_path = Path(args.canonical_pubkey)
    canonical_ts_path = Path(args.canonical_timestamp)

    canonical_priv_hex = read_hex_key(canonical_priv_path, 64, "canonical private key")
    canonical_pub_hex = read_hex_key(canonical_pub_path, 64, "canonical public key")
    key_fingerprint = canonical_pubkey_fingerprint(canonical_pub_hex)
    freeze_signature = sign_digest_with_private_key(canonical_priv_hex, freeze_digest)
    bootstrap_timestamp = read_bootstrap_timestamp(canonical_ts_path)
    policy_pinned_digest = read_policy_pinned_digest(Path(args.policy))

    if args.strict:
        if len(key_fingerprint) != 64:
            errors.append("canonical key fingerprint must be 64-char hex in --strict mode")
        if len(freeze_signature) != 128:
            errors.append("baseline freeze signature must be 128-char hex in --strict mode")
        if not bootstrap_timestamp:
            errors.append("canonical bootstrap timestamp missing in --strict mode")

    payload = {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": str(contract_path),
        "baseline_manifest": str(manifest_path),
        "external_collection": str(external_path),
        "performance_summary": str(performance_path),
        "required_baselines": required_baselines,
        "baseline_count": baseline_count,
        "available_count": available_count,
        "measurement_mode": measurement_mode,
        "minimum_samples": minimum_samples,
        "samples": samples,
        "threshold": threshold,
        "geomean_speedup": geomean,
        "missing_reports": missing_reports,
        "execution_failures": execution_failures,
        "substitution_count": len(substitutions),
        "frozen_rows": frozen_rows,
        "freeze_digest_sha256": freeze_digest,
        "canonical_pubkey_path": str(canonical_pub_path),
        "canonical_pubkey_fingerprint": key_fingerprint,
        "canonical_bootstrap_timestamp": bootstrap_timestamp,
        "policy_pinned_digest_sha256": policy_pinned_digest,
        "signature_algo": "ed25519",
        "signature_scope": "freeze_digest_sha256",
        "signature_hex": freeze_signature,
        "signed": True,
        "status": "frozen" if not errors else "invalid",
    }

    out_path = Path(args.out)
    write_json(out_path, payload)

    print(
        "omega_baseline_freeze: "
        f"mode={measurement_mode} "
        f"baseline_count={baseline_count} "
        f"available_count={available_count} "
        f"signed=true "
        f"errors={len(errors)} "
        f"out={out_path}"
    )

    if errors:
        for err in errors:
            print(f"omega_baseline_freeze: {err}", file=sys.stderr)
        if args.strict:
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
