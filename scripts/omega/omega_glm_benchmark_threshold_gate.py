#!/usr/bin/env python3
"""Validate self-hosted GLM benchmark outputs against explicit thresholds."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

SCHEMA = "sounio.omega.glm-benchmark-threshold-gate.v1"
BASELINE_SCHEMA = "sounio.omega.glm-benchmark-baseline.v1"
RESULT_SCHEMA = "sounio.glm.benchmark.v2"


@dataclass
class BenchRow:
    key: str
    benchmark: str
    mode: str
    duration_ms: float
    status: str
    exit_code: int


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enforce GLM benchmark threshold gate")
    parser.add_argument("--results-dir", default="benchmark_results")
    parser.add_argument("--out", default="benchmark_results/glm_threshold_gate.v1.json")
    parser.add_argument("--baseline", default="artifacts/omega/glm_benchmark_baseline.v1.json")
    parser.add_argument("--min-pass", type=int, default=4)
    parser.add_argument("--max-fail", type=int, default=0)
    parser.add_argument("--max-mean-ms", type=float, default=1500.0)
    parser.add_argument("--max-p95-ms", type=float, default=2500.0)
    parser.add_argument("--max-single-ms", type=float, default=5000.0)
    parser.add_argument("--max-regression-pct", type=float, default=25.0)
    parser.add_argument("--require-baseline", action="store_true")
    return parser.parse_args()


def percentile_95(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(len(s) - 1, math.ceil(0.95 * len(s)) - 1))
    return float(s[idx])


def load_rows(results_dir: Path) -> List[BenchRow]:
    rows: List[BenchRow] = []
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
        duration_ms = float(payload.get("duration_ms", 0.0))
        status = str(payload.get("status", "unknown"))
        exit_code = int(payload.get("exit_code", 0))
        rows.append(
            BenchRow(
                key=key,
                benchmark=benchmark,
                mode=mode,
                duration_ms=duration_ms,
                status=status,
                exit_code=exit_code,
            )
        )
    return rows


def load_baseline(path: Path) -> tuple[Dict[str, float], List[str]]:
    errors: List[str] = []
    if not path.exists():
        return {}, [f"missing baseline: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [f"invalid baseline json: {path}: {exc}"]

    schema = str(payload.get("schema", ""))
    if schema and schema != BASELINE_SCHEMA:
        errors.append(f"baseline schema mismatch: expected {BASELINE_SCHEMA} got {schema}")

    cases = payload.get("cases", {})
    if not isinstance(cases, dict):
        return {}, errors + ["baseline.cases must be object"]

    baseline: Dict[str, float] = {}
    for key, item in cases.items():
        try:
            if isinstance(item, dict):
                baseline[str(key)] = float(item.get("duration_ms", 0.0))
            else:
                baseline[str(key)] = float(item)
        except Exception:
            errors.append(f"baseline case has invalid duration: {key}")
    return baseline, errors


def main() -> int:
    args = parse_args()

    results_dir = Path(args.results_dir)
    out_path = Path(args.out)
    baseline_path = Path(args.baseline)

    rows = load_rows(results_dir)
    passed = [r for r in rows if r.status == "pass" and r.exit_code == 0]
    failed = [r for r in rows if r.status == "fail" or r.exit_code != 0]
    skipped = [r for r in rows if r.status == "skipped"]

    durations = [r.duration_ms for r in passed]
    mean_ms = (sum(durations) / len(durations)) if durations else 0.0
    p95_ms = percentile_95(durations)
    max_case_ms = max(durations) if durations else 0.0

    checks = []
    errors: List[str] = []

    def add_check(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "status": "pass" if ok else "fail", "detail": detail})
        if not ok:
            errors.append(f"{name}: {detail}")

    add_check(
        "min_pass_cases",
        len(passed) >= args.min_pass,
        f"pass={len(passed)} required>={args.min_pass}",
    )
    add_check(
        "max_fail_cases",
        len(failed) <= args.max_fail,
        f"fail={len(failed)} allowed<={args.max_fail}",
    )
    add_check(
        "max_mean_ms",
        mean_ms <= args.max_mean_ms,
        f"mean_ms={mean_ms:.3f} allowed<={args.max_mean_ms:.3f}",
    )
    add_check(
        "max_p95_ms",
        p95_ms <= args.max_p95_ms,
        f"p95_ms={p95_ms:.3f} allowed<={args.max_p95_ms:.3f}",
    )
    add_check(
        "max_single_case_ms",
        max_case_ms <= args.max_single_ms,
        f"max_case_ms={max_case_ms:.3f} allowed<={args.max_single_ms:.3f}",
    )

    baseline_durations: Dict[str, float] = {}
    baseline_errors: List[str] = []
    regressions: List[dict] = []

    if baseline_path.exists() or args.require_baseline:
        baseline_durations, baseline_errors = load_baseline(baseline_path)
        errors.extend(baseline_errors)
        for row in passed:
            baseline = baseline_durations.get(row.key)
            if baseline is None or baseline <= 0:
                continue
            limit = baseline * (1.0 + args.max_regression_pct / 100.0)
            if row.duration_ms > limit:
                regressions.append(
                    {
                        "case": row.key,
                        "baseline_ms": baseline,
                        "current_ms": row.duration_ms,
                        "allowed_ms": limit,
                        "regression_pct": ((row.duration_ms / baseline) - 1.0) * 100.0,
                    }
                )

        add_check(
            "baseline_regression",
            len(regressions) == 0 and len(baseline_errors) == 0,
            (
                f"regressions={len(regressions)} max_regression_pct={args.max_regression_pct:.2f} "
                f"baseline={baseline_path}"
            ),
        )
    else:
        add_check("baseline_regression", True, "baseline not required and not provided")

    status = "pass" if not errors and all(c["status"] == "pass" for c in checks) else "fail"

    out_payload = {
        "schema": SCHEMA,
        "generated_at_utc": now_utc(),
        "status": status,
        "inputs": {
            "results_dir": str(results_dir),
            "baseline": str(baseline_path),
        },
        "thresholds": {
            "min_pass": args.min_pass,
            "max_fail": args.max_fail,
            "max_mean_ms": args.max_mean_ms,
            "max_p95_ms": args.max_p95_ms,
            "max_single_ms": args.max_single_ms,
            "max_regression_pct": args.max_regression_pct,
            "require_baseline": bool(args.require_baseline),
        },
        "summary": {
            "total_cases": len(rows),
            "passed_cases": len(passed),
            "failed_cases": len(failed),
            "skipped_cases": len(skipped),
            "mean_ms": round(mean_ms, 6),
            "p95_ms": round(p95_ms, 6),
            "max_case_ms": round(max_case_ms, 6),
        },
        "checks": checks,
        "regressions": regressions,
        "errors": sorted(set(errors)),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload, indent=2) + "\n", encoding="utf-8")

    print(
        "omega_glm_benchmark_threshold_gate: "
        f"status={status} total={len(rows)} pass={len(passed)} fail={len(failed)} "
        f"mean_ms={mean_ms:.3f} p95_ms={p95_ms:.3f} out={out_path}"
    )

    if status != "pass":
        for err in out_payload["errors"]:
            print(f"error: {err}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
