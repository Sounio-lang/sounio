#!/usr/bin/env python3
"""Verify pinned souc binary provenance (sha256 + ed25519 signature)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from cryptography.hazmat.primitives.asymmetric import ed25519
except Exception:  # pragma: no cover
    ed25519 = None

SCHEMA = "sounio.omega.souc-release-provenance.v1"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_first_sha_token(path: Path) -> str:
    text = path.read_text().strip()
    if not text:
        return ""
    return text.split()[0].strip().lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify pinned souc binary with checksum/signature and emit provenance report"
    )
    parser.add_argument("--binary", required=True, help="Path to souc binary")
    parser.add_argument("--out", required=True, help="Output provenance JSON path")
    parser.add_argument("--version", default="", help="Release version label")
    parser.add_argument("--platform", default="", help="Platform label")
    parser.add_argument("--asset-url", default="", help="Asset URL used for fetch")
    parser.add_argument("--sha256", default="", help="Expected sha256 hex")
    parser.add_argument("--sha256-file", default="", help="Path to .sha256 file")
    parser.add_argument("--signature-hex", default="", help="Ed25519 signature hex over sha256 hex string")
    parser.add_argument("--signature-file", default="", help="Path to signature hex file")
    parser.add_argument(
        "--public-key",
        default="keys/bootstrap_ed25519.pub",
        help="Ed25519 public key file (hex)",
    )
    parser.add_argument(
        "--require-signature",
        action="store_true",
        help="Fail when signature cannot be verified",
    )
    return parser.parse_args()


def load_hex(path: Path) -> str:
    return path.read_text().strip().lower()


def compute_pubkey_fingerprint(pub_hex: str) -> str:
    return hashlib.sha256(bytes.fromhex(pub_hex)).hexdigest()


def main() -> int:
    args = parse_args()

    binary_path = Path(args.binary)
    out_path = Path(args.out)

    errors: list[str] = []

    if not binary_path.exists():
        errors.append(f"binary missing: {binary_path}")
        actual_sha256 = ""
    else:
        actual_sha256 = sha256_file(binary_path)

    expected_sha256 = args.sha256.strip().lower()
    if not expected_sha256 and args.sha256_file:
        sha_file = Path(args.sha256_file)
        if sha_file.exists():
            expected_sha256 = read_first_sha_token(sha_file)
        else:
            errors.append(f"sha256 file missing: {sha_file}")

    if expected_sha256:
        if len(expected_sha256) != 64:
            errors.append("expected sha256 must be 64-char hex")
        elif expected_sha256 != actual_sha256:
            errors.append(
                f"sha256 mismatch expected={expected_sha256} actual={actual_sha256}"
            )

    signature_hex = args.signature_hex.strip().lower()
    if not signature_hex and args.signature_file:
        sig_file = Path(args.signature_file)
        if sig_file.exists():
            signature_hex = load_hex(sig_file)
        else:
            errors.append(f"signature file missing: {sig_file}")

    pubkey_path = Path(args.public_key)
    pubkey_hex = ""
    pubkey_fingerprint = ""
    signature_verified = False

    if pubkey_path.exists():
        pubkey_hex = load_hex(pubkey_path)
        try:
            pubkey_fingerprint = compute_pubkey_fingerprint(pubkey_hex)
        except Exception as exc:
            errors.append(f"invalid public key hex: {exc}")
    else:
        errors.append(f"public key missing: {pubkey_path}")

    if signature_hex and actual_sha256:
        if ed25519 is None:
            errors.append("cryptography package unavailable for signature verification")
        else:
            try:
                pubkey = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(pubkey_hex))
                pubkey.verify(bytes.fromhex(signature_hex), actual_sha256.encode("utf-8"))
                signature_verified = True
            except Exception as exc:
                errors.append(f"signature verification failed: {exc}")
    elif args.require_signature:
        errors.append("signature required but not provided")

    if args.require_signature and not signature_verified:
        errors.append("signature verification required but failed")

    status = "pass" if not errors else "fail"

    report = {
        "schema": SCHEMA,
        "generated_at_utc": now_utc_iso(),
        "status": status,
        "version": args.version,
        "platform": args.platform,
        "asset_url": args.asset_url,
        "binary_path": str(binary_path),
        "sha256": {
            "expected": expected_sha256,
            "actual": actual_sha256,
            "match": bool(expected_sha256) and expected_sha256 == actual_sha256,
        },
        "signature": {
            "required": bool(args.require_signature),
            "signature_hex": signature_hex,
            "verified": signature_verified,
            "public_key_path": str(pubkey_path),
            "canonical_pubkey_fingerprint": pubkey_fingerprint,
        },
        "errors": sorted(set(errors)),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")

    print(
        "omega_verify_release_souc: "
        f"status={status} sha_match={report['sha256']['match']} "
        f"signature_verified={signature_verified} report={out_path}"
    )

    if errors:
        for err in sorted(set(errors)):
            print(f"error: {err}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
