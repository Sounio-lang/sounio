#!/usr/bin/env python3
"""Deterministic multi-run replay for RL readiness bridge/trend."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCHEMA = "sounio.omega.rl-readiness-replay.v1"
BRIDGE_SCHEMA = "sounio.omega.rl-readiness-bridge.v1"
TREND_SCHEMA = "sounio.omega.rl-readiness-trend.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Omega RL readiness deterministic replay"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Replay iterations to execute",
    )
    parser.add_argument(
        "--bridge-script",
        default="scripts/omega/omega_rl_readiness_bridge.py",
        help="Bridge generator script path",
    )
    parser.add_argument(
        "--trend-script",
        default="scripts/omega/omega_rl_readiness_trend.py",
        help="Trend updater script path",
    )
    parser.add_argument(
        "--bridge",
        default="artifacts/omega/rl_readiness_bridge.replay.v1.json",
        help="Replay bridge output path",
    )
    parser.add_argument(
        "--trend",
        default="artifacts/omega/rl_readiness_replay_trend.v1.json",
        help="Replay trend output path",
    )
    parser.add_argument(
        "--policy",
        default="bootstrap/policies/policy.v2.json",
        help="Policy path",
    )
    parser.add_argument(
        "--ptx-report",
        default="artifacts/ptx/omega/ptx_launch_report.json",
        help="PTX report path",
    )
    parser.add_argument(
        "--sass-report",
        default="artifacts/sass/omega/sass_patch_report.json",
        help="SASS report path",
    )
    parser.add_argument(
        "--hw-live-report",
        default="artifacts/fpga/hardware_epistemic_power_live.v1.json",
        help="Hardware live report path",
    )
    parser.add_argument(
        "--quantum-report",
        default="artifacts/quantum/omega/quantum_conformance.json",
        help="Quantum report path",
    )
    parser.add_argument(
        "--shadow-audit-report",
        default="artifacts/omega/shadow_audit.v1.json",
        help="Shadow audit report path",
    )
    parser.add_argument(
        "--compile-overhead",
        type=float,
        default=0.18,
        help="Compile overhead proxy for bridge generation",
    )
    parser.add_argument(
        "--out",
        default="artifacts/omega/rl_readiness_replay.v1.json",
        help="Replay artifact output path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when deterministic invariants do not hold",
    )
    return parser.parse_args()


def run_cmd(args: list[str]) -> None:
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"command failed (rc={proc.returncode}): {' '.join(args)}\n{proc.stdout}"
        )


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


def main() -> int:
    args = parse_args()
    if args.iterations <= 0:
        raise SystemExit("--iterations must be > 0")
    if args.compile_overhead < 0.0:
        raise SystemExit("--compile-overhead must be >= 0")

    bridge_script = Path(args.bridge_script)
    trend_script = Path(args.trend_script)
    if not bridge_script.exists():
        raise SystemExit(f"missing bridge script: {bridge_script}")
    if not trend_script.exists():
        raise SystemExit(f"missing trend script: {trend_script}")

    bridge_out = Path(args.bridge)
    trend_out = Path(args.trend)
    if trend_out.exists():
        trend_out.unlink()

    expected_evidence = None
    expected_trend_tuple = None
    evidence_stable = True
    trend_stable = True
    run_count_monotonic = True
    iterations_detail = []

    for idx in range(1, args.iterations + 1):
        run_cmd(
            [
                sys.executable,
                str(bridge_script),
                "--policy",
                args.policy,
                "--ptx-report",
                args.ptx_report,
                "--sass-report",
                args.sass_report,
                "--hw-live-report",
                args.hw_live_report,
                "--quantum-report",
                args.quantum_report,
                "--shadow-audit-report",
                args.shadow_audit_report,
                "--evidence-out",
                "bootstrap/policies/rl_readiness.evidence.json",
                "--bridge-out",
                str(bridge_out),
                "--compile-overhead",
                f"{args.compile_overhead}",
                "--strict",
            ]
        )
        run_cmd(
            [
                sys.executable,
                str(trend_script),
                "--bridge",
                str(bridge_out),
                "--trend",
                str(trend_out),
                "--strict",
            ]
        )

        bridge = load_json(bridge_out, "replay bridge")
        if bridge.get("schema") != BRIDGE_SCHEMA:
            raise SystemExit(
                f"invalid replay bridge schema: expected {BRIDGE_SCHEMA}, got {bridge.get('schema')}"
            )
        evidence = bridge.get("evidence")
        if not isinstance(evidence, dict):
            raise SystemExit("replay bridge missing evidence object")
        evidence_tuple = (
            float(evidence.get("epistemic_power_gain", 0.0)),
            int(evidence.get("bit_exact_mismatches", 0)),
            float(evidence.get("compile_overhead", 0.0)),
            bool(evidence.get("shadow_audit_clean", False)),
            bool(bridge.get("policy_rl_readiness_pass", False)),
        )
        if expected_evidence is None:
            expected_evidence = evidence_tuple
        elif evidence_tuple != expected_evidence:
            evidence_stable = False

        trend = load_json(trend_out, "replay trend")
        if trend.get("schema") != TREND_SCHEMA:
            raise SystemExit(
                f"invalid replay trend schema: expected {TREND_SCHEMA}, got {trend.get('schema')}"
            )
        runs = trend.get("runs")
        if not isinstance(runs, list) or not runs:
            raise SystemExit("replay trend missing runs")
        latest = runs[-1]
        if not isinstance(latest, dict):
            raise SystemExit("replay trend latest run is invalid")
        trend_tuple = (
            float(latest.get("epistemic_power_gain", 0.0)),
            int(latest.get("bit_exact_mismatches", 0)),
            float(latest.get("compile_overhead", 0.0)),
            bool(latest.get("shadow_audit_clean", False)),
            bool(latest.get("policy_rl_readiness_pass", False)),
            bool(latest.get("drift_violation", True)),
        )
        if expected_trend_tuple is None:
            expected_trend_tuple = trend_tuple
        elif trend_tuple != expected_trend_tuple:
            trend_stable = False

        run_count = int(trend.get("run_count", 0))
        if run_count != idx:
            run_count_monotonic = False

        iterations_detail.append(
            {
                "iteration": idx,
                "evidence_tuple": evidence_tuple,
                "trend_tuple": trend_tuple,
                "run_count": run_count,
            }
        )

    deterministic_replay_pass = evidence_stable and trend_stable and run_count_monotonic
    payload = {
        "schema": SCHEMA,
        "iterations": args.iterations,
        "evidence_stable": evidence_stable,
        "trend_stable": trend_stable,
        "run_count_monotonic": run_count_monotonic,
        "deterministic_replay_pass": deterministic_replay_pass,
        "details": iterations_detail,
    }
    out_path = Path(args.out)
    write_json(out_path, payload)

    print(
        "omega_rl_readiness_replay: "
        f"iterations={args.iterations} "
        f"evidence_stable={str(evidence_stable).lower()} "
        f"trend_stable={str(trend_stable).lower()} "
        f"run_count_monotonic={str(run_count_monotonic).lower()} "
        f"pass={str(deterministic_replay_pass).lower()} "
        f"report={out_path}"
    )

    if args.strict and not deterministic_replay_pass:
        print(
            "omega_rl_readiness_replay: strict failed (deterministic replay mismatch)",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
