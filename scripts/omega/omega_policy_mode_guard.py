#!/usr/bin/env python3
"""Enforce policy-mode governance for Sprint-1 (shadow default)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCHEMA = "sounio.omega.policy-mode-guard.v1"
BRIDGE_SCHEMA = "sounio.omega.rl-readiness-bridge.v1"
TREND_SCHEMA = "sounio.omega.rl-readiness-trend.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Omega policy mode guard (shadow default)"
    )
    parser.add_argument(
        "--policy",
        default="bootstrap/policies/policy.v2.json",
        help="Policy JSON path",
    )
    parser.add_argument(
        "--bridge",
        default="artifacts/omega/rl_readiness_bridge.v1.json",
        help="RL readiness bridge artifact path",
    )
    parser.add_argument(
        "--rl-trend",
        default="artifacts/omega/rl_readiness_trend.v1.json",
        help="RL readiness trend artifact path",
    )
    parser.add_argument(
        "--min-stable-runs",
        type=int,
        default=5,
        help="Minimum trailing stable runs required for active mode",
    )
    parser.add_argument(
        "--allow-active",
        action="store_true",
        help="Allow policy_mode=active when readiness pass is present",
    )
    parser.add_argument(
        "--out",
        default="artifacts/omega/policy_mode_guard.v1.json",
        help="Output artifact path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when policy mode violates Sprint-1 shadow governance",
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


def stable_run_count(path: Path) -> tuple[int, bool]:
    if not path.exists():
        return 0, False
    trend = load_json(path, "rl readiness trend")
    if trend.get("schema") != TREND_SCHEMA:
        raise SystemExit(
            f"invalid trend schema in {path}: expected {TREND_SCHEMA}, got {trend.get('schema')}"
        )
    runs = trend.get("runs")
    if not isinstance(runs, list):
        raise SystemExit(f"invalid runs in trend {path}: expected list")

    stable = 0
    for entry in reversed(runs):
        if not isinstance(entry, dict):
            break
        if (
            bool(entry.get("policy_rl_readiness_pass", False))
            and bool(entry.get("shadow_audit_clean", False))
            and not bool(entry.get("drift_violation", True))
        ):
            stable += 1
        else:
            break
    return stable, True


def main() -> int:
    args = parse_args()
    if args.min_stable_runs <= 0:
        raise SystemExit("--min-stable-runs must be > 0")

    policy_path = Path(args.policy)
    bridge_path = Path(args.bridge)
    trend_path = Path(args.rl_trend)
    out_path = Path(args.out)

    policy = load_json(policy_path, "policy")
    policy_mode_raw = str(policy.get("policy_mode", "")).strip().lower()
    policy_mode = policy_mode_raw if policy_mode_raw else "unknown"

    bridge_pass = None
    if bridge_path.exists():
        bridge = load_json(bridge_path, "rl readiness bridge")
        if bridge.get("schema") != BRIDGE_SCHEMA:
            raise SystemExit(
                f"invalid bridge schema in {bridge_path}: expected {BRIDGE_SCHEMA}, got {bridge.get('schema')}"
            )
        bridge_pass = bool(bridge.get("policy_rl_readiness_pass", False))

    stable_count, trend_present = stable_run_count(trend_path)
    stable_runs_satisfied = stable_count >= args.min_stable_runs

    fail_reasons: list[str] = []
    guard_pass = True

    if policy_mode == "shadow":
        guard_pass = True
    elif policy_mode == "active":
        if not args.allow_active:
            guard_pass = False
            fail_reasons.append("policy_mode=active requires --allow-active")
        if bridge_pass is not True:
            guard_pass = False
            fail_reasons.append("policy_mode=active requires bridge policy_rl_readiness_pass=true")
        if not trend_present:
            guard_pass = False
            fail_reasons.append("policy_mode=active requires rl readiness trend artifact")
        elif not stable_runs_satisfied:
            guard_pass = False
            fail_reasons.append(
                f"policy_mode=active requires >= {args.min_stable_runs} stable trailing runs (got {stable_count})"
            )
    else:
        guard_pass = False
        fail_reasons.append(f"invalid policy_mode={policy_mode!r} (expected shadow|active)")

    payload = {
        "schema": SCHEMA,
        "policy_path": str(policy_path),
        "bridge_path": str(bridge_path),
        "trend_path": str(trend_path),
        "policy_mode": policy_mode,
        "allow_active": bool(args.allow_active),
        "min_stable_runs": int(args.min_stable_runs),
        "stable_run_count": int(stable_count),
        "stable_runs_satisfied": bool(stable_runs_satisfied),
        "bridge_policy_pass": bridge_pass,
        "guard_pass": guard_pass,
        "fail_reasons": fail_reasons,
    }
    write_json(out_path, payload)

    print(
        "omega_policy_mode_guard: "
        f"mode={policy_mode} "
        f"allow_active={str(bool(args.allow_active)).lower()} "
        f"bridge_pass={str(bridge_pass).lower() if bridge_pass is not None else 'none'} "
        f"guard_pass={str(guard_pass).lower()} "
        f"report={out_path}"
    )

    if args.strict and not guard_pass:
        for reason in fail_reasons:
            print(f"omega_policy_mode_guard: strict failed ({reason})", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
