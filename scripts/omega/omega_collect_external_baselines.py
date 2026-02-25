#!/usr/bin/env python3
"""Collect external baseline adapter reports for Omega Sprint-1 performance ingestion."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COLLECTION_SCHEMA = "sounio.omega.external-baseline-collection.v1"
REPORT_SCHEMA = "sounio.omega.external-baseline-report.v1"
CONTRACT_SCHEMA = "sounio.independence.contract.v2"
BASELINE_MANIFEST_SCHEMA = "sounio.independence.baseline-set.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect external baseline reports from contract adapters",
    )
    parser.add_argument(
        "--contract",
        default="benchmarks/independence/contract.v2.json",
        help="Independence contract path",
    )
    parser.add_argument(
        "--baseline-manifest",
        default="benchmarks/independence/omega_sprint1_baselines.v1.json",
        help="Baseline manifest path",
    )
    parser.add_argument(
        "--report-dir",
        default="artifacts/omega/external_baselines",
        help="Directory for per-baseline reports",
    )
    parser.add_argument(
        "--out",
        default="artifacts/omega/external_baseline_collection.v1.json",
        help="Collection summary output path",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute adapter entrypoints before reading reports",
    )
    parser.add_argument(
        "--timeout-secs",
        type=int,
        default=120,
        help="Per-adapter execution timeout in seconds",
    )
    parser.add_argument(
        "--baseline-python",
        default="python3",
        help="Python executable for adapter probes",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail when required baselines are missing reports",
    )
    return parser.parse_args()


def load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        raise SystemExit(f"unable to read {label} ({path}): {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {label} ({path}): {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid payload in {label} ({path}): expected object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def adapter_map(contract: dict[str, Any]) -> dict[str, str]:
    adapters = contract.get("baseline_adapters", [])
    if not isinstance(adapters, list):
        adapters = []
    mapping: dict[str, str] = {}
    for item in adapters:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        entry = item.get("entrypoint")
        if isinstance(name, str) and isinstance(entry, str) and name and entry:
            mapping[name] = entry
    return mapping


def normalize_family_list(contract: dict[str, Any]) -> list[str]:
    runtime_norm = contract.get("runtime_normalization_by_family", {})
    if not isinstance(runtime_norm, dict):
        return []
    out = [k for k in runtime_norm.keys() if isinstance(k, str)]
    return sorted(set(out))


def run_adapter(
    baseline_name: str,
    entrypoint: Path,
    report_path: Path,
    families: list[str],
    timeout_secs: int,
    baseline_python: str,
) -> tuple[int, str, str]:
    env = os.environ.copy()
    env["OMEGA_BASELINE_NAME"] = baseline_name
    env["OMEGA_BASELINE_REPORT"] = str(report_path)
    env["OMEGA_BASELINE_FAMILIES"] = ",".join(families)
    env["OMEGA_BASELINE_REPORT_SCHEMA"] = REPORT_SCHEMA
    env["OMEGA_BASELINE_PYTHON"] = baseline_python

    try:
        proc = subprocess.run(
            [str(entrypoint)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            timeout=max(1, timeout_secs),
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return 124, exc.stdout or "", exc.stderr or "timeout"
    except OSError as exc:
        return 127, "", str(exc)


def sanitize_text(value: str, max_chars: int = 4000) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "...(truncated)"


def validate_report(payload: dict[str, Any], baseline_name: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if payload.get("schema") != REPORT_SCHEMA:
        errors.append(f"schema mismatch (expected {REPORT_SCHEMA}, got {payload.get('schema')})")
    name = payload.get("baseline_name")
    if not isinstance(name, str) or not name:
        errors.append("baseline_name missing or invalid")
    elif name != baseline_name:
        errors.append(f"baseline_name mismatch (expected {baseline_name}, got {name})")

    available = payload.get("available")
    if not isinstance(available, bool):
        errors.append("available must be bool")

    if "samples" in payload and not isinstance(payload.get("samples"), int):
        errors.append("samples must be int")
    if "family_speedups" in payload and not isinstance(payload.get("family_speedups"), dict):
        errors.append("family_speedups must be object")
    return (len(errors) == 0), errors


def fallback_unavailable_report(
    baseline_name: str,
    report_path: Path,
    reason: str,
    stdout: str,
    stderr: str,
    rc: int,
) -> dict[str, Any]:
    payload = {
        "schema": REPORT_SCHEMA,
        "baseline_name": baseline_name,
        "available": False,
        "samples": 0,
        "family_speedups": {},
        "reason": reason,
        "entrypoint_status": {
            "return_code": rc,
            "stdout": sanitize_text(stdout),
            "stderr": sanitize_text(stderr),
        },
        "report_path": str(report_path),
    }
    return payload


def main() -> int:
    args = parse_args()
    if args.timeout_secs <= 0:
        raise SystemExit("--timeout-secs must be > 0")

    contract_path = Path(args.contract)
    baseline_manifest_path = Path(args.baseline_manifest)
    report_dir = Path(args.report_dir)
    out_path = Path(args.out)

    contract = load_json(contract_path, "contract")
    baseline_manifest = load_json(baseline_manifest_path, "baseline manifest")
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise SystemExit(
            f"invalid contract schema: expected {CONTRACT_SCHEMA}, got {contract.get('schema')}"
        )
    if baseline_manifest.get("schema") != BASELINE_MANIFEST_SCHEMA:
        raise SystemExit(
            f"invalid baseline manifest schema: expected {BASELINE_MANIFEST_SCHEMA}, got {baseline_manifest.get('schema')}"
        )

    required = baseline_manifest.get("required_baselines", [])
    if not isinstance(required, list):
        required = []
    required = [item for item in required if isinstance(item, str)]

    adapters = adapter_map(contract)
    families = normalize_family_list(contract)
    report_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    missing_reports: list[str] = []
    execution_failures: list[str] = []
    available_count = 0

    for baseline_name in required:
        entrypoint_raw = adapters.get(baseline_name, "")
        entrypoint = Path(entrypoint_raw) if entrypoint_raw else None
        report_path = report_dir / f"{baseline_name}.v1.json"
        rc = 0
        stdout = ""
        stderr = ""

        if args.execute:
            if entrypoint is None:
                rc = 127
                stderr = "adapter entrypoint missing in contract"
            else:
                rc, stdout, stderr = run_adapter(
                    baseline_name,
                    entrypoint,
                    report_path,
                    families,
                    args.timeout_secs,
                    args.baseline_python,
                )

        if report_path.exists():
            payload = load_json(report_path, f"baseline report ({baseline_name})")
            ok, report_errors = validate_report(payload, baseline_name)
            if not ok:
                payload = fallback_unavailable_report(
                    baseline_name=baseline_name,
                    report_path=report_path,
                    reason="invalid_report_schema_or_fields: " + "; ".join(report_errors),
                    stdout=stdout,
                    stderr=stderr,
                    rc=rc,
                )
                write_json(report_path, payload)
            else:
                if args.execute:
                    payload["entrypoint_status"] = {
                        "return_code": rc,
                        "stdout": sanitize_text(stdout),
                        "stderr": sanitize_text(stderr),
                    }
                    write_json(report_path, payload)
        else:
            reason = "adapter_did_not_emit_report"
            if not args.execute:
                reason = "report_missing_without_execution"
            payload = fallback_unavailable_report(
                baseline_name=baseline_name,
                report_path=report_path,
                reason=reason,
                stdout=stdout,
                stderr=stderr,
                rc=rc,
            )
            write_json(report_path, payload)
            missing_reports.append(baseline_name)

        if payload.get("available") is True:
            available_count += 1
        if args.execute and rc != 0:
            execution_failures.append(baseline_name)

        rows.append(
            {
                "baseline_name": baseline_name,
                "entrypoint": entrypoint_raw,
                "report_path": str(report_path),
                "available": bool(payload.get("available", False)),
                "samples": int(payload.get("samples", 0))
                if isinstance(payload.get("samples"), int)
                else 0,
                "reason": str(payload.get("reason", "")),
                "entrypoint_return_code": rc,
            }
        )

    summary = {
        "schema": COLLECTION_SCHEMA,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": str(contract_path),
        "baseline_manifest": str(baseline_manifest_path),
        "required_baselines": required,
        "external_baseline_dir": str(report_dir),
        "execute": bool(args.execute),
        "families": families,
        "baseline_python": args.baseline_python,
        "baseline_count": len(required),
        "available_count": available_count,
        "missing_reports": sorted(set(missing_reports)),
        "execution_failures": sorted(set(execution_failures)),
        "rows": rows,
    }
    write_json(out_path, summary)

    print(
        "omega_collect_external_baselines: "
        f"execute={str(args.execute).lower()} "
        f"baseline_count={summary['baseline_count']} "
        f"available_count={available_count} "
        f"missing_reports={len(summary['missing_reports'])} "
        f"execution_failures={len(summary['execution_failures'])} "
        f"out={out_path}"
    )

    if args.strict and summary["missing_reports"]:
        for item in summary["missing_reports"]:
            print(
                f"omega_collect_external_baselines: strict failed (missing report for {item})",
                file=sys.stderr,
            )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
