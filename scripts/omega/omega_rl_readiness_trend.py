#!/usr/bin/env python3
"""Track RL readiness evidence over time and flag regressions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "sounio.omega.rl-readiness-trend.v1"
BRIDGE_SCHEMA = "sounio.omega.rl-readiness-bridge.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Omega RL readiness trend tracker"
    )
    parser.add_argument(
        "--bridge",
        default="artifacts/omega/rl_readiness_bridge.v1.json",
        help="RL readiness bridge artifact path",
    )
    parser.add_argument(
        "--trend",
        default="artifacts/omega/rl_readiness_trend.v1.json",
        help="RL readiness trend artifact path",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=120,
        help="Maximum historical runs to retain",
    )
    parser.add_argument(
        "--max-gain-drop",
        type=float,
        default=0.10,
        help="Maximum allowed absolute drop in epistemic gain before drift violation",
    )
    parser.add_argument(
        "--max-overhead-jump",
        type=float,
        default=0.05,
        help="Maximum allowed increase in compile overhead before drift violation",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when policy pass is false or drift violation is detected",
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


def load_trend(path: Path) -> dict:
    if not path.exists():
        return {"schema": SCHEMA, "runs": [], "last_status": "bootstrap"}
    trend = load_json(path, "rl readiness trend")
    if trend.get("schema") != SCHEMA:
        raise SystemExit(
            f"invalid trend schema in {path}: expected {SCHEMA}, got {trend.get('schema')}"
        )
    runs = trend.get("runs")
    if not isinstance(runs, list):
        raise SystemExit(f"invalid trend payload in {path}: runs must be a list")
    return trend


def clamp_nonnegative(value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        return 0.0
    return value


def main() -> int:
    args = parse_args()
    if args.max_runs <= 0:
        raise SystemExit("--max-runs must be > 0")
    if args.max_gain_drop < 0.0:
        raise SystemExit("--max-gain-drop must be >= 0")
    if args.max_overhead_jump < 0.0:
        raise SystemExit("--max-overhead-jump must be >= 0")

    bridge_path = Path(args.bridge)
    if not bridge_path.exists():
        raise SystemExit(f"missing RL readiness bridge artifact: {bridge_path}")

    bridge = load_json(bridge_path, "rl readiness bridge")
    if bridge.get("schema") != BRIDGE_SCHEMA:
        raise SystemExit(
            f"invalid bridge schema in {bridge_path}: expected {BRIDGE_SCHEMA}, got {bridge.get('schema')}"
        )
    evidence = bridge.get("evidence")
    if not isinstance(evidence, dict):
        raise SystemExit(f"invalid evidence block in {bridge_path}: expected object")

    gain = clamp_nonnegative(float(evidence.get("epistemic_power_gain", 0.0)))
    overhead = clamp_nonnegative(float(evidence.get("compile_overhead", 0.0)))
    mismatches = int(evidence.get("bit_exact_mismatches", 0))
    shadow_clean = bool(evidence.get("shadow_audit_clean", False))
    policy_pass = bool(bridge.get("policy_rl_readiness_pass", False))

    trend_path = Path(args.trend)
    trend = load_trend(trend_path)
    runs = trend["runs"]
    prev = runs[-1] if runs else None

    gain_drop_abs = 0.0
    overhead_jump = 0.0
    drift_violation = False
    if isinstance(prev, dict):
        prev_gain = float(prev.get("epistemic_power_gain", gain))
        prev_overhead = float(prev.get("compile_overhead", overhead))
        if math.isfinite(prev_gain):
            gain_drop_abs = max(0.0, prev_gain - gain)
        if math.isfinite(prev_overhead):
            overhead_jump = max(0.0, overhead - prev_overhead)
        drift_violation = (
            gain_drop_abs > args.max_gain_drop
            or overhead_jump > args.max_overhead_jump
        )

    entry = {
        "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "epistemic_power_gain": gain,
        "bit_exact_mismatches": mismatches,
        "compile_overhead": overhead,
        "shadow_audit_clean": shadow_clean,
        "policy_rl_readiness_pass": policy_pass,
        "gain_drop_abs": gain_drop_abs,
        "overhead_jump": overhead_jump,
        "drift_violation": drift_violation,
    }
    runs.append(entry)
    if len(runs) > args.max_runs:
        del runs[: len(runs) - args.max_runs]

    status = "pass" if policy_pass else "fail"
    if policy_pass and drift_violation:
        status = "warn"

    trend["last_status"] = status
    trend["latest_gain"] = gain
    trend["latest_overhead"] = overhead
    trend["latest_gain_drop_abs"] = gain_drop_abs
    trend["latest_overhead_jump"] = overhead_jump
    trend["max_gain_drop"] = args.max_gain_drop
    trend["max_overhead_jump"] = args.max_overhead_jump
    trend["run_count"] = len(runs)
    write_json(trend_path, trend)

    print(
        "omega_rl_readiness_trend: "
        f"status={status} "
        f"policy_pass={str(policy_pass).lower()} "
        f"gain={gain:.6f} "
        f"overhead={overhead:.6f} "
        f"gain_drop={gain_drop_abs:.6f} "
        f"overhead_jump={overhead_jump:.6f} "
        f"run_count={len(runs)} "
        f"trend={trend_path}"
    )

    if args.strict:
        if not policy_pass:
            print("omega_rl_readiness_trend: strict failed (policy readiness false)", file=sys.stderr)
            return 2
        if drift_violation:
            print("omega_rl_readiness_trend: strict failed (readiness drift violation)", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
