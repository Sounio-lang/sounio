#!/usr/bin/env python3
"""Track historical hardware live-read telemetry and detect drift."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "sounio.omega.hardware-epistemic-power-live-trend.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Omega hardware epistemic power live-read trend tracker"
    )
    parser.add_argument(
        "--live-report",
        default="artifacts/fpga/hardware_epistemic_power_live.v1.json",
        help="Input hardware live-read telemetry artifact",
    )
    parser.add_argument(
        "--trend",
        default="artifacts/fpga/hardware_epistemic_power_live_trend.v1.json",
        help="Trend artifact path",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=120,
        help="Maximum historical runs to retain",
    )
    parser.add_argument(
        "--drift-threshold",
        type=float,
        default=0.20,
        help="Allowed absolute delta for epistemic gain ratio before drift violation",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when latest run is non-conformant or drift exceeds threshold",
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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def clamp_nonnegative(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    if value < 0.0:
        return 0.0
    return value


def compute_gain_ratio(live: dict) -> float:
    hw = int(live.get("hardware_epistemic_power_log_q32_32", 0))
    sw = int(live.get("software_epistemic_power_log_q32_32", 0))
    hybrid = int(live.get("hybrid_epistemic_power_log_q32_32", 0))
    denom = max(1, sw)
    return clamp_nonnegative((hybrid - hw) / float(denom))


def load_trend(path: Path) -> dict:
    if not path.exists():
        return {"schema": SCHEMA, "runs": [], "last_status": "bootstrap"}
    trend = load_json(path)
    if trend.get("schema") != SCHEMA:
        raise SystemExit(
            f"invalid trend schema in {path}: expected {SCHEMA}, got {trend.get('schema')}"
        )
    runs = trend.get("runs")
    if not isinstance(runs, list):
        raise SystemExit(f"invalid trend payload in {path}: runs must be a list")
    return trend


def main() -> int:
    args = parse_args()
    live_path = Path(args.live_report)
    trend_path = Path(args.trend)

    if args.max_runs <= 0:
        raise SystemExit("--max-runs must be > 0")
    if args.drift_threshold < 0.0:
        raise SystemExit("--drift-threshold must be >= 0")

    if not live_path.exists():
        raise SystemExit(f"missing live-read artifact: {live_path}")

    live = load_json(live_path)
    trend = load_trend(trend_path)
    runs = trend["runs"]

    gain_ratio = compute_gain_ratio(live)
    live_conformant = bool(live.get("live_read_conformant", False))
    launch_accepted = bool(live.get("launch_accepted", False))
    used_fallback = bool(live.get("used_fallback", True))

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    entry = {
        "ts_utc": now,
        "gain_ratio": gain_ratio,
        "live_read_conformant": live_conformant,
        "launch_accepted": launch_accepted,
        "used_fallback": used_fallback,
        "symbol_check_mode": str(live.get("symbol_check_mode", "")),
        "hardware_epistemic_power_log_q32_32": int(
            live.get("hardware_epistemic_power_log_q32_32", 0)
        ),
        "software_epistemic_power_log_q32_32": int(
            live.get("software_epistemic_power_log_q32_32", 0)
        ),
        "hybrid_epistemic_power_log_q32_32": int(
            live.get("hybrid_epistemic_power_log_q32_32", 0)
        ),
    }

    prev = runs[-1] if runs else None
    drift_abs = 0.0
    drift_violation = False
    if isinstance(prev, dict):
        prev_gain = float(prev.get("gain_ratio", 0.0))
        if math.isfinite(prev_gain):
            drift_abs = abs(gain_ratio - prev_gain)
            drift_violation = drift_abs > args.drift_threshold

    entry["drift_abs"] = drift_abs
    entry["drift_violation"] = drift_violation

    runs.append(entry)
    if len(runs) > args.max_runs:
        del runs[: len(runs) - args.max_runs]

    status = "pass"
    if not live_conformant or not launch_accepted or used_fallback:
        status = "fail"
    elif drift_violation:
        status = "warn"

    trend["last_status"] = status
    trend["latest_gain_ratio"] = gain_ratio
    trend["latest_drift_abs"] = drift_abs
    trend["drift_threshold"] = args.drift_threshold
    trend["run_count"] = len(runs)
    write_json(trend_path, trend)

    print(
        "omega_hw_epi_power_trend: "
        f"status={status} "
        f"gain_ratio={gain_ratio:.6f} "
        f"drift_abs={drift_abs:.6f} "
        f"drift_threshold={args.drift_threshold:.6f} "
        f"run_count={len(runs)} "
        f"trend={trend_path}"
    )

    if args.strict:
        if status == "fail":
            print(
                "omega_hw_epi_power_trend: strict failed (non-conformant live-read)",
                file=sys.stderr,
            )
            return 2
        if drift_violation:
            print(
                "omega_hw_epi_power_trend: strict failed (drift threshold exceeded)",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
