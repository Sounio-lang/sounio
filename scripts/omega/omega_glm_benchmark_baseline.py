#!/usr/bin/env python3
"""Generate GLM benchmark baseline artifact from benchmark_results JSON files."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

BASELINE_SCHEMA = "sounio.omega.glm-benchmark-baseline.v1"
RESULT_SCHEMA = "sounio.glm.benchmark.v2"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit GLM benchmark baseline artifact")
    parser.add_argument("--results-dir", default="benchmark_results")
    parser.add_argument("--out", default="artifacts/omega/glm_benchmark_baseline.v1.json")
    parser.add_argument("--notes", default="no-rust self-hosted benchmark baseline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_path = Path(args.out)

    if not results_dir.exists():
        print(f"error: results dir does not exist: {results_dir}")
        return 2

    cases: Dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "glm_threshold_gate.v1.json":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("schema") != RESULT_SCHEMA:
            continue

        benchmark = str(payload.get("benchmark", "unknown"))
        mode = str(payload.get("mode", "unknown"))
        key = f"{benchmark}_{mode}"

        cases[key] = {
            "benchmark": benchmark,
            "mode": mode,
            "duration_ms": float(payload.get("duration_ms", 0.0)),
            "status": str(payload.get("status", "unknown")),
            "exit_code": int(payload.get("exit_code", 0)),
            "generated_at_utc": str(payload.get("generated_at_utc", "")),
        }

    if not cases:
        print(f"error: no benchmark rows found in {results_dir}")
        return 2

    durations = [row["duration_ms"] for row in cases.values() if row["status"] == "pass"]
    payload = {
        "schema": BASELINE_SCHEMA,
        "generated_at_utc": now_utc(),
        "notes": args.notes,
        "source_results_dir": str(results_dir),
        "source_environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "machine": platform.machine(),
        },
        "summary": {
            "case_count": len(cases),
            "pass_count": sum(1 for row in cases.values() if row["status"] == "pass"),
            "max_duration_ms": max(durations) if durations else 0.0,
            "mean_duration_ms": (sum(durations) / len(durations)) if durations else 0.0,
        },
        "cases": dict(sorted(cases.items())),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(
        "omega_glm_benchmark_baseline: "
        f"cases={len(cases)} pass={payload['summary']['pass_count']} out={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
