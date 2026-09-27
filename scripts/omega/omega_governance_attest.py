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
    "artifacts/fpga/fpga_seed_report.json",
    "artifacts/fpga/hardware_epistemic_power_live_trend.v1.json",
    "artifacts/fpga/hardware_resource_trend.v2.json",
    "artifacts/ptx/omega/ptx_launch_report.json",
    "artifacts/sass/omega/sass_patch_report.json",
    "artifacts/quantum/omega/quantum_conformance.json",
    "hardware/rtl/qir/omega_genesis_emitter.sio",
    "hardware/rtl/kaxi/merkle_root_lane.sio",
    "hardware/fpga/k_axi_merkle_root_lane.v",
    "artifacts/omega/performance_summary.v1.json",
    "artifacts/omega/external_baseline_collection.v1.json",
    "artifacts/omega/baseline_freeze.v1.json",
    "artifacts/omega/sprint1_release_readiness.v1.json",
    "artifacts/omega/omega_genesis.v1.0.json",
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
    parser.add_argument(
        "--hw-live",
        default="artifacts/fpga/hardware_epistemic_power_live.v1.json",
        help="Hardware live-read telemetry artifact path",
    )
    parser.add_argument(
        "--fpga-report",
        default="artifacts/fpga/fpga_seed_report.json",
        help="FPGA seed report artifact path",
    )
    parser.add_argument(
        "--resource-trend",
        default="artifacts/fpga/hardware_resource_trend.v2.json",
        help="Hardware resource trend artifact path",
    )
    parser.add_argument(
        "--qir-emitter",
        default="hardware/rtl/qir/omega_genesis_emitter.sio",
        help="Full QIR emitter module path",
    )
    parser.add_argument(
        "--merkle-sio",
        default="hardware/rtl/kaxi/merkle_root_lane.sio",
        help="Merkle lane SIO path",
    )
    parser.add_argument(
        "--merkle-rtl",
        default="hardware/fpga/k_axi_merkle_root_lane.v",
        help="Merkle lane RTL path",
    )
    parser.add_argument(
        "--genesis-manifest",
        default="artifacts/omega/omega_genesis.v1.0.json",
        help="Genesis manifest artifact path",
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


def read_sprint_hardware_lineage(
    hw_live_path: Path,
    fpga_path: Path,
    resource_trend_path: Path,
    qir_emitter_path: Path,
    merkle_sio_path: Path,
    merkle_rtl_path: Path,
) -> dict:
    result: dict = {
        "variance_q32_32": "",
        "poll_overhead_us": None,
        "bidir_kaxi_pass": None,
        "merkle_lane_pass": None,
        "qir_emitter_pass": None,
        "resource_trend_pass": None,
        "resource_trend_status": "",
        "resource_trend_max_relative_drift": None,
        "resource_trend_threshold": None,
    }

    if hw_live_path.exists():
        try:
            payload = json.loads(hw_live_path.read_text())
            if isinstance(payload, dict):
                raw_variance = payload.get("hardware_epistemic_power_variance_q32_32", "")
                if isinstance(raw_variance, int):
                    result["variance_q32_32"] = str(raw_variance)
                raw_overhead = payload.get("poll_overhead_us")
                if isinstance(raw_overhead, int):
                    result["poll_overhead_us"] = raw_overhead
        except (OSError, json.JSONDecodeError):
            pass

    if fpga_path.exists():
        try:
            payload = json.loads(fpga_path.read_text())
            if isinstance(payload, dict) and payload.get("stale"):
                # hardware/** has never been versioned in this repository --
                # the status fields below describe an environment this
                # checkout doesn't have. Record an explicit fail rather than
                # let a stale "pass" propagate. See fpga_seed_report.json's
                # stale_reason.
                result["bidir_kaxi_pass"] = False
                result["merkle_lane_pass"] = False
            elif isinstance(payload, dict):
                sim = payload.get("k_axi_return_sim_status", "")
                synth = payload.get("k_axi_return_synth_status", "")
                if isinstance(sim, str) and isinstance(synth, str):
                    result["bidir_kaxi_pass"] = sim == "pass" and synth == "pass"
                merkle_present = payload.get("merkle_lane_present")
                merkle_core = payload.get("merkle_lane_core_rtl_present")
                merkle_synth = payload.get("merkle_lane_synth_status")
                if (
                    isinstance(merkle_present, bool)
                    and isinstance(merkle_core, bool)
                    and isinstance(merkle_synth, str)
                ):
                    result["merkle_lane_pass"] = (
                        merkle_present and merkle_core and merkle_synth == "pass"
                    )
        except (OSError, json.JSONDecodeError):
            pass

    if resource_trend_path.exists():
        try:
            payload = json.loads(resource_trend_path.read_text())
            if isinstance(payload, dict):
                status = payload.get("last_status")
                max_drift = payload.get("last_max_relative_drift")
                threshold = payload.get("drift_threshold")
                if isinstance(status, str):
                    result["resource_trend_status"] = status
                if isinstance(max_drift, (int, float)):
                    result["resource_trend_max_relative_drift"] = float(max_drift)
                if isinstance(threshold, (int, float)):
                    result["resource_trend_threshold"] = float(threshold)
                if (
                    isinstance(status, str)
                    and isinstance(max_drift, (int, float))
                    and isinstance(threshold, (int, float))
                ):
                    result["resource_trend_pass"] = (
                        status in ("pass", "bootstrap")
                        and float(max_drift) <= float(threshold)
                    )
        except (OSError, json.JSONDecodeError):
            pass

    qir_pass = False
    if qir_emitter_path.exists():
        try:
            qir_text = qir_emitter_path.read_text()
            qir_pass = (
                "omega_qir_emit_shim" in qir_text
                and "omega_qir_emit_quantum_controller" in qir_text
                and "omega_qir_emit_bundle" in qir_text
                and "omega_qir_full_emitter_self_check" in qir_text
                and "selfhost-emitter" in qir_text
                and "template-direct" not in qir_text
            )
        except OSError:
            qir_pass = False
    result["qir_emitter_pass"] = qir_pass

    merkle_lane_sio_ok = False
    merkle_lane_rtl_ok = False
    if merkle_sio_path.exists():
        try:
            merkle_sio_text = merkle_sio_path.read_text()
            merkle_lane_sio_ok = (
                (
                    "merkle_lane_root_l64" in merkle_sio_text
                    and "merkle_lane_verify_root" in merkle_sio_text
                    and "merkle_lane_self_check" in merkle_sio_text
                )
                or (
                    "merkle_root_lane_l64" in merkle_sio_text
                    and "merkle_root_verify_l64" in merkle_sio_text
                    and "merkle_root_lane_self_check" in merkle_sio_text
                )
            )
        except OSError:
            merkle_lane_sio_ok = False
    if merkle_rtl_path.exists():
        try:
            merkle_rtl_text = merkle_rtl_path.read_text()
            merkle_lane_rtl_ok = (
                (
                    "module k_axi_merkle_lane_core" in merkle_rtl_text
                    and "merkle_lane_digest" in merkle_rtl_text
                    and "MERKLE_SALT" in merkle_rtl_text
                )
                or (
                    "module k_axi_merkle_root_lane_core" in merkle_rtl_text
                    and "merkle_root_l64" in merkle_rtl_text
                    and "merkle_root_valid" in merkle_rtl_text
                )
            )
        except OSError:
            merkle_lane_rtl_ok = False
    if result["merkle_lane_pass"] is None:
        result["merkle_lane_pass"] = merkle_lane_sio_ok and merkle_lane_rtl_ok
    else:
        result["merkle_lane_pass"] = (
            bool(result["merkle_lane_pass"]) and merkle_lane_sio_ok and merkle_lane_rtl_ok
        )

    return result


def read_genesis_manifest_lineage(path: Path) -> tuple[bool | None, bool | None, str]:
    if not path.exists():
        return None, None, ""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, None, ""
    if not isinstance(payload, dict):
        return None, None, ""
    signed = payload.get("signed")
    cold_boot_ready = payload.get("cold_boot_ready")
    aggregate_sha256 = payload.get("aggregate_sha256")
    signed_bool = signed if isinstance(signed, bool) else None
    cold_boot_ready_bool = cold_boot_ready if isinstance(cold_boot_ready, bool) else None
    aggregate = aggregate_sha256 if isinstance(aggregate_sha256, str) else ""
    return signed_bool, cold_boot_ready_bool, aggregate


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
    hardware_lineage = read_sprint_hardware_lineage(
        Path(args.hw_live),
        Path(args.fpga_report),
        Path(args.resource_trend),
        Path(args.qir_emitter),
        Path(args.merkle_sio),
        Path(args.merkle_rtl),
    )
    hw_variance_q32_32 = hardware_lineage["variance_q32_32"]
    hw_poll_overhead_us = hardware_lineage["poll_overhead_us"]
    bidir_kaxi_pass = hardware_lineage["bidir_kaxi_pass"]
    merkle_lane_pass = hardware_lineage["merkle_lane_pass"]
    qir_emitter_pass = hardware_lineage["qir_emitter_pass"]
    resource_trend_pass = hardware_lineage["resource_trend_pass"]
    resource_trend_status = hardware_lineage["resource_trend_status"]
    resource_trend_max_relative_drift = hardware_lineage["resource_trend_max_relative_drift"]
    resource_trend_threshold = hardware_lineage["resource_trend_threshold"]
    genesis_signed, genesis_cold_boot_ready, genesis_manifest_digest = read_genesis_manifest_lineage(
        Path(args.genesis_manifest)
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
        "hardware_epistemic_power_variance_q32_32": hw_variance_q32_32,
        "hardware_poll_overhead_us": hw_poll_overhead_us,
        "bidir_kaxi_pass": bidir_kaxi_pass,
        "merkle_lane_pass": merkle_lane_pass,
        "qir_emitter_pass": qir_emitter_pass,
        "resource_trend_pass": resource_trend_pass,
        "resource_trend_status": resource_trend_status,
        "resource_trend_max_relative_drift": resource_trend_max_relative_drift,
        "resource_trend_threshold": resource_trend_threshold,
        "genesis_manifest_path": args.genesis_manifest,
        "genesis_manifest_signed": genesis_signed,
        "genesis_cold_boot_ready": genesis_cold_boot_ready,
        "genesis_manifest_aggregate_sha256": genesis_manifest_digest,
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
        f"poll_overhead_us={hw_poll_overhead_us if hw_poll_overhead_us is not None else 'missing'} "
        f"bidir_kaxi_pass={str(bidir_kaxi_pass).lower() if bidir_kaxi_pass is not None else 'missing'} "
        f"merkle_lane_pass={str(merkle_lane_pass).lower() if merkle_lane_pass is not None else 'missing'} "
        f"qir_emitter_pass={str(qir_emitter_pass).lower() if qir_emitter_pass is not None else 'missing'} "
        f"resource_trend_pass={str(resource_trend_pass).lower() if resource_trend_pass is not None else 'missing'} "
        f"genesis_signed={str(genesis_signed).lower() if genesis_signed is not None else 'missing'} "
        f"genesis_cold_boot_ready={str(genesis_cold_boot_ready).lower() if genesis_cold_boot_ready is not None else 'missing'} "
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
    if args.strict and hw_poll_overhead_us is not None and hw_poll_overhead_us >= 1000:
        print(
            "omega_governance_attest: hardware poll overhead must be <1000us in strict mode "
            f"(got {hw_poll_overhead_us})",
            file=sys.stderr,
        )
        return 2
    if args.strict and bidir_kaxi_pass is False:
        print(
            "omega_governance_attest: bidir_kaxi_pass=false in strict mode",
            file=sys.stderr,
        )
        return 2
    if args.strict and merkle_lane_pass is not True:
        print(
            "omega_governance_attest: merkle_lane_pass!=true in strict mode",
            file=sys.stderr,
        )
        return 2
    if args.strict and qir_emitter_pass is not True:
        print(
            "omega_governance_attest: qir_emitter_pass!=true in strict mode",
            file=sys.stderr,
        )
        return 2
    if args.strict and resource_trend_pass is not True:
        print(
            "omega_governance_attest: resource_trend_pass!=true in strict mode",
            file=sys.stderr,
        )
        return 2
    if args.strict and genesis_signed is not True:
        print(
            "omega_governance_attest: genesis_manifest_signed!=true in strict mode",
            file=sys.stderr,
        )
        return 2
    if args.strict and genesis_cold_boot_ready is not True:
        print(
            "omega_governance_attest: genesis_cold_boot_ready!=true in strict mode",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
