#!/usr/bin/env python3
"""Build RL readiness evidence from Sprint-1 Omega artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BRIDGE_SCHEMA = "sounio.omega.rl-readiness-bridge.v1"
SHADOW_AUDIT_SCHEMA = "sounio.omega.shadow-audit.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RL readiness evidence from Omega hardware/software artifacts"
    )
    parser.add_argument(
        "--policy",
        default="bootstrap/policies/policy.v2.json",
        help="Optimization policy path",
    )
    parser.add_argument(
        "--ptx-report",
        default="artifacts/ptx/omega/ptx_launch_report.json",
        help="PTX launch telemetry report path",
    )
    parser.add_argument(
        "--sass-report",
        default="artifacts/sass/omega/sass_patch_report.json",
        help="SASS patch report path",
    )
    parser.add_argument(
        "--hw-live-report",
        default="artifacts/fpga/hardware_epistemic_power_live.v1.json",
        help="Hardware live-read report path",
    )
    parser.add_argument(
        "--quantum-report",
        default="artifacts/quantum/omega/quantum_conformance.json",
        help="Quantum conformance report path",
    )
    parser.add_argument(
        "--shadow-audit-report",
        default="artifacts/omega/shadow_audit.v1.json",
        help="Optional shadow-audit report path",
    )
    parser.add_argument(
        "--evidence-out",
        default="bootstrap/policies/rl_readiness.evidence.json",
        help="RL readiness evidence output path",
    )
    parser.add_argument(
        "--bridge-out",
        default="artifacts/omega/rl_readiness_bridge.v1.json",
        help="Detailed bridge artifact path",
    )
    parser.add_argument(
        "--compile-overhead",
        type=float,
        default=0.18,
        help="Compile overhead proxy to write into readiness evidence",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when synthesized evidence does not satisfy policy gate",
    )
    return parser.parse_args()


def load_json(path: Path, label: str) -> dict:
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"unable to read {label} ({path}): {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {label} ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid JSON payload in {label} ({path}): expected object")
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def saturate_nonnegative(value: float) -> float:
    if value != value or value < 0.0:  # NaN or negative
        return 0.0
    return value


def compute_epistemic_power_gain(ptx_report: dict) -> float:
    hw = float(ptx_report.get("hardware_epistemic_power_log_q32_32", 0))
    sw = float(ptx_report.get("software_epistemic_power_log_q32_32", 0))
    hybrid = float(ptx_report.get("hybrid_epistemic_power_log_q32_32", 0))
    denom = max(1.0, sw)
    return saturate_nonnegative((hybrid - hw) / denom)


def compute_bit_exact_mismatches(sass_report: dict) -> int:
    kernel_count = int(sass_report.get("kernel_count", 0))
    eq_pass = int(sass_report.get("equivalence_pass_count", 0))
    fallback_count = int(sass_report.get("fallback_count", 0))
    mismatches = max(0, kernel_count - eq_pass) + max(0, fallback_count)
    return mismatches


def compute_shadow_audit_clean(
    ptx_report: dict,
    hw_live_report: dict,
    quantum_report: dict,
    bit_exact_mismatches: int,
) -> bool:
    return (
        bool(ptx_report.get("pass_marker_seen", False))
        and bool(ptx_report.get("launch_accepted", False))
        and not bool(ptx_report.get("used_fallback", True))
        and bool(hw_live_report.get("live_read_conformant", False))
        and bool(quantum_report.get("conformant", False))
        and bit_exact_mismatches == 0
    )


def load_shadow_audit_clean(path: Path) -> bool | None:
    if not path.exists():
        return None
    payload = load_json(path, "shadow audit report")
    if payload.get("schema") != SHADOW_AUDIT_SCHEMA:
        raise SystemExit(
            f"invalid shadow audit schema in {path}: expected {SHADOW_AUDIT_SCHEMA}, got {payload.get('schema')}"
        )
    clean = payload.get("shadow_audit_clean")
    if not isinstance(clean, bool):
        raise SystemExit(f"invalid shadow_audit_clean in {path}: expected bool")
    return clean


def verify_against_policy(policy: dict, evidence: dict) -> tuple[bool, list[str]]:
    rl = policy.get("rl_readiness")
    if not isinstance(rl, dict):
        return False, ["policy missing rl_readiness block"]

    reasons: list[str] = []
    min_gain = float(rl.get("min_epistemic_power_gain", 0.0))
    max_overhead = float(rl.get("max_compile_overhead", 0.0))
    require_zero_mismatch = bool(rl.get("require_zero_bit_mismatch", False))
    require_shadow_clean = bool(rl.get("require_shadow_audit_clean", False))

    gain = float(evidence["epistemic_power_gain"])
    mismatches = int(evidence["bit_exact_mismatches"])
    overhead = float(evidence["compile_overhead"])
    shadow_clean = bool(evidence["shadow_audit_clean"])

    if gain < min_gain:
        reasons.append(
            f"epistemic_power_gain {gain:.6f} < min_epistemic_power_gain {min_gain:.6f}"
        )
    if require_zero_mismatch and mismatches != 0:
        reasons.append(f"bit_exact_mismatches {mismatches} != 0")
    if overhead > max_overhead:
        reasons.append(f"compile_overhead {overhead:.6f} > max_compile_overhead {max_overhead:.6f}")
    if require_shadow_clean and not shadow_clean:
        reasons.append("shadow_audit_clean=false")
    return len(reasons) == 0, reasons


def main() -> int:
    args = parse_args()
    if args.compile_overhead < 0.0:
        raise SystemExit("--compile-overhead must be >= 0")

    policy_path = Path(args.policy)
    ptx_path = Path(args.ptx_report)
    sass_path = Path(args.sass_report)
    hw_live_path = Path(args.hw_live_report)
    quantum_path = Path(args.quantum_report)
    shadow_audit_path = Path(args.shadow_audit_report)
    evidence_path = Path(args.evidence_out)
    bridge_path = Path(args.bridge_out)

    policy = load_json(policy_path, "policy")
    ptx = load_json(ptx_path, "ptx report")
    sass = load_json(sass_path, "sass report")
    hw_live = load_json(hw_live_path, "hardware live-read report")
    quantum = load_json(quantum_path, "quantum report")

    bit_exact_mismatches = compute_bit_exact_mismatches(sass)
    artifact_shadow_clean = compute_shadow_audit_clean(
        ptx,
        hw_live,
        quantum,
        bit_exact_mismatches,
    )
    reported_shadow_clean = load_shadow_audit_clean(shadow_audit_path)
    shadow_audit_clean = artifact_shadow_clean
    if reported_shadow_clean is not None:
        shadow_audit_clean = artifact_shadow_clean and reported_shadow_clean

    evidence = {
        "epistemic_power_gain": compute_epistemic_power_gain(ptx),
        "bit_exact_mismatches": bit_exact_mismatches,
        "compile_overhead": float(args.compile_overhead),
        "shadow_audit_clean": shadow_audit_clean,
    }

    policy_pass, reasons = verify_against_policy(policy, evidence)

    bridge_payload = {
        "schema": BRIDGE_SCHEMA,
        "policy": str(policy_path),
        "ptx_report": str(ptx_path),
        "sass_report": str(sass_path),
        "hw_live_report": str(hw_live_path),
        "quantum_report": str(quantum_path),
        "shadow_audit_report": str(shadow_audit_path),
        "shadow_audit_clean_reported": reported_shadow_clean,
        "shadow_audit_clean_artifacts": artifact_shadow_clean,
        "evidence_path": str(evidence_path),
        "evidence": evidence,
        "policy_rl_readiness_pass": policy_pass,
        "policy_rl_readiness_fail_reasons": reasons,
    }

    write_json(evidence_path, evidence)
    write_json(bridge_path, bridge_payload)

    print(
        "omega_rl_readiness_bridge: "
        f"gain={evidence['epistemic_power_gain']:.6f} "
        f"mismatches={evidence['bit_exact_mismatches']} "
        f"overhead={evidence['compile_overhead']:.6f} "
        f"audit_clean={str(evidence['shadow_audit_clean']).lower()} "
        f"policy_pass={str(policy_pass).lower()} "
        f"evidence={evidence_path} "
        f"bridge={bridge_path}"
    )

    if args.strict and not policy_pass:
        for reason in reasons:
            print(f"omega_rl_readiness_bridge: strict failed ({reason})", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
