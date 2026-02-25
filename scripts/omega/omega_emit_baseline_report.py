#!/usr/bin/env python3
"""Emit external baseline report payloads for custom command hooks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPORT_SCHEMA = "sounio.omega.external-baseline-report.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit custom external baseline report")
    parser.add_argument("--baseline-name", required=True, help="Baseline name")
    parser.add_argument("--report", required=True, help="Output report path")
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Sample count",
    )
    parser.add_argument(
        "--speedup",
        action="append",
        default=[],
        help="Family speedup entry in family=value format; repeatable",
    )
    parser.add_argument(
        "--reason",
        default="custom_cmd",
        help="Reason label",
    )
    parser.add_argument(
        "--unavailable",
        action="store_true",
        help="Emit unavailable report (ignore speedups)",
    )
    return parser.parse_args()


def parse_speedups(entries: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for entry in entries:
        raw = entry.strip()
        if not raw:
            continue
        if "=" not in raw:
            raise SystemExit(f"invalid --speedup entry (expected family=value): {entry}")
        family, value = raw.split("=", 1)
        family = family.strip()
        if not family:
            raise SystemExit(f"invalid --speedup family: {entry}")
        try:
            parsed = float(value.strip())
        except ValueError as exc:
            raise SystemExit(f"invalid --speedup value: {entry}") from exc
        out[family] = parsed
    return out


def main() -> int:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be > 0")

    speedups = parse_speedups(args.speedup)
    available = not args.unavailable and bool(speedups)

    payload = {
        "schema": REPORT_SCHEMA,
        "baseline_name": args.baseline_name,
        "available": available,
        "samples": args.samples if available else 0,
        "family_speedups": speedups if available else {},
        "reason": args.reason,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2))

    print(
        "omega_emit_baseline_report: "
        f"baseline={args.baseline_name} "
        f"available={str(available).lower()} "
        f"report={report_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
