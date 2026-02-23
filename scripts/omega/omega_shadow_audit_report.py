#!/usr/bin/env python3
"""Emit a structured shadow-audit report from Sprint-1 gate markers."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "sounio.omega.shadow-audit.v1"
DEFAULT_MARKERS = (
    "CANONICAL_KEY_BOOTSTRAP_PASS",
    "QR_ALIAS_REGRESSION_PASS",
    "PURE_SOUNIO_KAXI_PASS",
    "KAXI_ADAPTER_SELF_CHECK_PASS",
    "HARDWARE_PUBLISH_SELF_CHECK_PASS",
    "PTX_LAUNCH_SELF_CHECK_PASS",
    "PTX_LAUNCH_TELEMETRY_PASS",
    "HW_EPI_POWER_LIVE_READ_PASS",
    "HW_EPI_POWER_TREND_PASS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Omega shadow-audit report generator"
    )
    parser.add_argument(
        "--gate-log",
        default="artifacts/omega_sprint1_gate.log",
        help="Gate log containing PASS markers",
    )
    parser.add_argument(
        "--out",
        default="artifacts/omega/shadow_audit.v1.json",
        help="Output shadow-audit JSON artifact",
    )
    parser.add_argument(
        "--marker",
        action="append",
        default=[],
        help="Additional required marker (can be repeated)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when any required marker is missing",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    args = parse_args()
    gate_log = Path(args.gate_log)
    if not gate_log.exists():
        raise SystemExit(f"missing gate log: {gate_log}")

    try:
        text = gate_log.read_text()
    except OSError as exc:
        raise SystemExit(f"unable to read gate log {gate_log}: {exc}") from exc

    required_markers = list(DEFAULT_MARKERS)
    for marker in args.marker:
        marker = marker.strip()
        if marker and marker not in required_markers:
            required_markers.append(marker)

    found = [marker for marker in required_markers if marker in text]
    missing = [marker for marker in required_markers if marker not in text]
    shadow_audit_clean = len(missing) == 0

    payload = {
        "schema": SCHEMA,
        "ts_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "gate_log": str(gate_log),
        "required_markers": required_markers,
        "found_markers": found,
        "missing_markers": missing,
        "shadow_audit_clean": shadow_audit_clean,
    }
    out_path = Path(args.out)
    write_json(out_path, payload)

    print(
        "omega_shadow_audit_report: "
        f"required={len(required_markers)} "
        f"found={len(found)} "
        f"missing={len(missing)} "
        f"clean={str(shadow_audit_clean).lower()} "
        f"report={out_path}"
    )

    if args.strict and not shadow_audit_clean:
        for marker in missing:
            print(f"omega_shadow_audit_report: missing marker {marker}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
