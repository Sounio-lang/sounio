#!/usr/bin/env python3
"""Generate reproducibility reports from selfhost zero-fallback gate artifacts.

This parser is intentionally strict. It exits non-zero when required files,
markers, or key/value fields are missing or malformed.

Example:
  scripts/selfhost_reproducibility_report.py \
    --gate-dir /tmp/sounio-selfhost-zero-gate-local

Outputs by default:
  <gate-dir>/artifacts/reproducibility_report.v1.json
  <gate-dir>/artifacts/reproducibility_report.v1.md
"""

from __future__ import annotations

import argparse
import json
import platform as py_platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "sounio.selfhost_zero_fallback_reproducibility_report"
SCHEMA_VERSION = "1.1.0"

SUMMARY_REQUIRED_KEYS = (
    "summary_pass",
    "summary_fail",
    "full_timeout_secs",
    "parse_timeout_secs",
    "parse_shard_count",
    "parse_shard_selector",
    "full_log",
    "report_log",
    "shards_log",
)

SUMMARY_INT_KEYS = (
    "summary_pass",
    "summary_fail",
    "full_timeout_secs",
    "parse_timeout_secs",
    "parse_shard_count",
)

STRICT_PATTERN = re.compile(r"\bstrict=([0-9]+)\b")
SUITE_FAIL_PATTERN = re.compile(r"SUITE FAIL:|Suites: suite fail")

ORACLE_MARKER = "SELFHOST=oracle backend=rust"

DRIVER_ORCH_FALLBACK_PATTERN = re.compile(
    r"SELFHOST=driver-first\s+schema=v1\s+event=driver_orchestration\b.*\bstatus=fallback\b"
)
DRIVER_ORCH_STRICT_ERROR_PATTERN = re.compile(
    r"SELFHOST=driver-first\s+schema=v1\s+event=driver_orchestration\b.*\bstatus=strict_error\b"
)
DRIVER_OUTPUT_PATTERN = re.compile(
    r"SELFHOST=driver-first\s+schema=v1\s+event=driver_output\b"
)


class ParseError(RuntimeError):
    """Raised when artifacts are missing or malformed."""


@dataclass(frozen=True)
class GatePaths:
    gate_dir: Path
    summary_file: Path
    full_log: Path
    report_log: Path
    shards_log: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate strict JSON and Markdown reproducibility reports from "
            "scripts/selfhost_zero_fallback_gate.sh artifacts."
        ),
        epilog=(
            "The command exits non-zero on missing files, malformed key/value "
            "lines, missing required markers, or inconsistent shard counts."
        ),
    )
    parser.add_argument(
        "--gate-dir",
        required=True,
        help="Gate artifact root directory that contains logs/ and artifacts/.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Output JSON path. Default: <gate-dir>/artifacts/reproducibility_report.v1.json",
    )
    parser.add_argument(
        "--md-out",
        default=None,
        help="Output Markdown path. Default: <gate-dir>/artifacts/reproducibility_report.v1.md",
    )
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise ParseError(f"missing required {label}: {path}")
    if not path.is_file():
        raise ParseError(f"expected file for {label}, found non-file path: {path}")


def parse_int(value: str, context: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ParseError(f"invalid integer for {context}: {value!r}") from exc
    if parsed < 0:
        raise ParseError(f"expected non-negative integer for {context}, got: {parsed}")
    return parsed


def read_lines(path: Path) -> list[str]:
    ensure_file(path, str(path))
    return path.read_text(encoding="utf-8").splitlines()


def parse_summary_file(summary_file: Path) -> dict[str, Any]:
    lines = read_lines(summary_file)
    data: dict[str, str] = {}
    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if "=" not in line:
            raise ParseError(f"malformed summary line {i} in {summary_file}: {raw!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ParseError(f"empty key in summary line {i} of {summary_file}")
        if key in data:
            raise ParseError(f"duplicate summary key {key!r} in {summary_file}")
        data[key] = value

    missing = [key for key in SUMMARY_REQUIRED_KEYS if key not in data]
    if missing:
        raise ParseError(f"missing summary keys in {summary_file}: {', '.join(missing)}")

    parsed: dict[str, Any] = dict(data)
    for key in SUMMARY_INT_KEYS:
        parsed[key] = parse_int(data[key], f"{summary_file}:{key}")
    if not data["parse_shard_selector"]:
        raise ParseError(f"empty parse_shard_selector in {summary_file}")
    return parsed


def resolve_log_path(gate_dir: Path, summary_value: str, expected_name: str) -> Path:
    from_summary = Path(summary_value)
    candidates: list[Path] = []
    candidates.append(gate_dir / "logs" / expected_name)
    if from_summary.is_absolute():
        candidates.append(from_summary)
    else:
        candidates.append(gate_dir / from_summary)

    seen: set[Path] = set()
    for candidate in candidates:
        normalized = candidate.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        if normalized.exists() and normalized.is_file():
            return normalized

    checked = ", ".join(str(c.resolve()) for c in candidates)
    raise ParseError(
        f"could not find log file for {expected_name}. checked: {checked}"
    )


def count_contains(lines: list[str], needle: str) -> int:
    return sum(1 for line in lines if needle in line)


def count_regex(lines: list[str], pattern: re.Pattern[str]) -> int:
    return sum(1 for line in lines if pattern.search(line))


def parse_marker_line(
    line: str,
    prefix: str,
    required_keys: tuple[str, ...],
    int_keys: tuple[str, ...],
    context: str,
) -> dict[str, Any]:
    if not line.startswith(prefix):
        raise ParseError(f"expected line with prefix {prefix!r} in {context}: {line!r}")
    rest = line[len(prefix) :].strip()
    if not rest:
        raise ParseError(f"marker missing key/value fields in {context}: {line!r}")
    fields: dict[str, str] = {}
    for token in rest.split():
        if "=" not in token:
            raise ParseError(f"malformed token {token!r} in {context}: {line!r}")
        key, value = token.split("=", 1)
        if not key:
            raise ParseError(f"empty key token in {context}: {line!r}")
        if key in fields:
            raise ParseError(f"duplicate key {key!r} in marker ({context}): {line!r}")
        fields[key] = value

    missing = [key for key in required_keys if key not in fields]
    if missing:
        raise ParseError(
            f"missing marker keys ({', '.join(missing)}) in {context}: {line!r}"
        )

    parsed: dict[str, Any] = dict(fields)
    for key in int_keys:
        parsed[key] = parse_int(fields[key], f"{context}:{key}")
    return parsed


def strict_value_from_log(lines: list[str], context: str) -> dict[str, Any]:
    values: list[int] = []
    occurrences = 0
    for line in lines:
        if "SELFHOST=driver-first" not in line:
            continue
        matches = STRICT_PATTERN.findall(line)
        for raw in matches:
            occurrences += 1
            values.append(parse_int(raw, f"{context}:strict"))

    if not values:
        raise ParseError(f"missing strict= marker in {context}")
    unique_values = sorted(set(values))
    if any(value not in (0, 1) for value in unique_values):
        raise ParseError(
            f"strict marker has unsupported value(s) in {context}: {unique_values}"
        )
    if len(unique_values) != 1:
        raise ParseError(
            f"inconsistent strict marker values in {context}: {unique_values}"
        )

    return {"strict_value": unique_values[0], "occurrences": occurrences}


def parse_full_selfhost_markers(lines: list[str]) -> dict[str, Any]:
    parse_all_header_count = count_contains(lines, "=== Parse All ===")
    parse_all_summary_count = count_contains(lines, "Parse All:")
    suites_all_count = count_contains(lines, "Suites: all passed")
    suite_fail_count = count_regex(lines, SUITE_FAIL_PATTERN)
    fallback_count = count_regex(lines, DRIVER_ORCH_FALLBACK_PATTERN)
    strict_error_count = count_regex(lines, DRIVER_ORCH_STRICT_ERROR_PATTERN)
    driver_output_count = count_regex(lines, DRIVER_OUTPUT_PATTERN)
    oracle_count = count_contains(lines, ORACLE_MARKER)

    markers_pass = (
        parse_all_header_count > 0
        and parse_all_summary_count > 0
        and suites_all_count > 0
        and suite_fail_count == 0
        and fallback_count == 0
        and strict_error_count == 0
        and oracle_count == 0
    )

    return {
        "parse_all_header_count": parse_all_header_count,
        "parse_all_summary_count": parse_all_summary_count,
        "suites_all_count": suites_all_count,
        "suite_fail_count": suite_fail_count,
        "fallback_marker_count": fallback_count,
        "strict_error_marker_count": strict_error_count,
        "driver_output_marker_count": driver_output_count,
        "oracle_marker_count": oracle_count,
        "markers_pass": markers_pass,
    }


def parse_parse_all_report(lines: list[str], expected_shard_count: int) -> dict[str, Any]:
    summary_lines = [line for line in lines if line.startswith("PARSE_ALL_REPORT_SUMMARY ")]
    if len(summary_lines) != 1:
        raise ParseError(
            "expected exactly one PARSE_ALL_REPORT_SUMMARY line "
            f"but found {len(summary_lines)}"
        )
    summary = parse_marker_line(
        summary_lines[0],
        "PARSE_ALL_REPORT_SUMMARY ",
        required_keys=("corpus", "shard_count", "total_files"),
        int_keys=("shard_count", "total_files"),
        context="PARSE_ALL_REPORT_SUMMARY",
    )

    shard_lines = [line for line in lines if line.startswith("PARSE_ALL_REPORT_SHARD ")]
    if not shard_lines:
        raise ParseError("missing PARSE_ALL_REPORT_SHARD lines")

    shards: list[dict[str, Any]] = []
    for idx, line in enumerate(shard_lines):
        shard = parse_marker_line(
            line,
            "PARSE_ALL_REPORT_SHARD ",
            required_keys=("shard_index", "shard_count", "files"),
            int_keys=("shard_index", "shard_count", "files"),
            context=f"PARSE_ALL_REPORT_SHARD[{idx}]",
        )
        if shard["shard_count"] != summary["shard_count"]:
            raise ParseError(
                "PARSE_ALL_REPORT_SHARD shard_count mismatch: "
                f"expected {summary['shard_count']}, got {shard['shard_count']}"
            )
        shards.append(shard)

    shard_indexes = sorted(shard["shard_index"] for shard in shards)
    if len(shard_indexes) != len(set(shard_indexes)):
        raise ParseError("duplicate shard_index entries in PARSE_ALL_REPORT_SHARD lines")

    expected_indexes = list(range(summary["shard_count"]))
    if shard_indexes != expected_indexes:
        raise ParseError(
            "PARSE_ALL_REPORT_SHARD indexes are not contiguous from 0 to shard_count-1"
        )

    if summary["shard_count"] != expected_shard_count:
        raise ParseError(
            "PARSE_ALL_REPORT_SUMMARY shard_count mismatch with summary.txt: "
            f"{summary['shard_count']} vs {expected_shard_count}"
        )

    if len(shards) != summary["shard_count"]:
        raise ParseError(
            "PARSE_ALL_REPORT_SHARD line count mismatch: "
            f"{len(shards)} vs declared {summary['shard_count']}"
        )

    shard_file_total = sum(shard["files"] for shard in shards)
    if shard_file_total != summary["total_files"]:
        raise ParseError(
            "PARSE_ALL_REPORT_SHARD file total mismatch: "
            f"{shard_file_total} vs declared total_files={summary['total_files']}"
        )

    return {
        "summary": summary,
        "shards": sorted(shards, key=lambda shard: shard["shard_index"]),
        "shard_lines_observed": len(shards),
        "markers_pass": (
            len(summary_lines) == 1
            and len(shards) == expected_shard_count
            and summary["shard_count"] == expected_shard_count
        ),
    }


def parse_parse_all_shards(lines: list[str], expected_shard_count: int) -> dict[str, Any]:
    multi_lines = [line for line in lines if line.startswith("PARSE_ALL_MULTI_SUMMARY ")]
    if len(multi_lines) != 1:
        raise ParseError(
            "expected exactly one PARSE_ALL_MULTI_SUMMARY line "
            f"but found {len(multi_lines)}"
        )

    multi_summary = parse_marker_line(
        multi_lines[0],
        "PARSE_ALL_MULTI_SUMMARY ",
        required_keys=("shard_count", "selector", "executed", "passed", "failed", "invalid"),
        int_keys=("shard_count", "executed", "passed", "failed", "invalid"),
        context="PARSE_ALL_MULTI_SUMMARY",
    )

    if multi_summary["shard_count"] != expected_shard_count:
        raise ParseError(
            "PARSE_ALL_MULTI_SUMMARY shard_count mismatch with summary.txt: "
            f"{multi_summary['shard_count']} vs {expected_shard_count}"
        )

    shard_summary_lines = [line for line in lines if line.startswith("PARSE_ALL_SUMMARY ")]
    if not shard_summary_lines:
        raise ParseError("missing PARSE_ALL_SUMMARY lines in parse_all_shards.log")

    shard_outcomes: list[dict[str, Any]] = []
    for idx, line in enumerate(shard_summary_lines):
        outcome = parse_marker_line(
            line,
            "PARSE_ALL_SUMMARY ",
            required_keys=(
                "corpus",
                "shard_index",
                "shard_count",
                "selected",
                "passed",
                "failed",
            ),
            int_keys=("shard_index", "shard_count", "selected", "passed", "failed"),
            context=f"PARSE_ALL_SUMMARY[{idx}]",
        )
        if outcome["shard_count"] != expected_shard_count:
            raise ParseError(
                "PARSE_ALL_SUMMARY shard_count mismatch: "
                f"{outcome['shard_count']} vs expected {expected_shard_count}"
            )
        shard_outcomes.append(outcome)

    shard_indexes = [entry["shard_index"] for entry in shard_outcomes]
    if len(shard_indexes) != len(set(shard_indexes)):
        raise ParseError("duplicate shard_index entries in PARSE_ALL_SUMMARY lines")

    if len(shard_outcomes) != multi_summary["executed"]:
        raise ParseError(
            "PARSE_ALL_SUMMARY line count mismatch with executed field: "
            f"{len(shard_outcomes)} vs {multi_summary['executed']}"
        )

    aggregate_passed = sum(entry["passed"] for entry in shard_outcomes)
    aggregate_failed = sum(entry["failed"] for entry in shard_outcomes)
    aggregate_selected = sum(entry["selected"] for entry in shard_outcomes)

    markers_pass = (
        len(shard_outcomes) > 0
        and multi_summary["executed"] > 0
        and multi_summary["executed"] == multi_summary["passed"]
        and multi_summary["failed"] == 0
        and multi_summary["invalid"] == 0
    )

    return {
        "multi_summary": multi_summary,
        "shard_outcomes": sorted(shard_outcomes, key=lambda entry: entry["shard_index"]),
        "summary_lines_observed": len(shard_outcomes),
        "aggregate_selected": aggregate_selected,
        "aggregate_passed": aggregate_passed,
        "aggregate_failed": aggregate_failed,
        "markers_pass": markers_pass,
    }


def isoformat_utc_from_epoch(epoch_seconds: float) -> str:
    return (
        datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def platform_info() -> dict[str, str]:
    uname = py_platform.uname()
    return {
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "python_version": py_platform.python_version(),
        "python_implementation": py_platform.python_implementation(),
    }


def status_word(value: bool) -> str:
    return "PASS" if value else "FAIL"


def markdown_table(rows: list[tuple[str, Any]]) -> list[str]:
    lines = ["| Field | Value |", "|---|---|"]
    for key, value in rows:
        lines.append(f"| {key} | `{value}` |")
    return lines


def render_markdown(report: dict[str, Any]) -> str:
    gate_config = report["gate_config"]
    gate_outcome = report["gate_outcome"]
    markers = report["markers"]
    parse_all_report = report["parse_all_report"]
    parse_all_shards = report["parse_all_shards"]
    fallback = report["fallback_oracle_markers"]

    lines: list[str] = []
    lines.append("# Selfhost Zero-Fallback Reproducibility Report")
    lines.append("")
    lines.append(f"- Schema: `{report['schema']}`")
    lines.append(f"- Schema version: `{report['schema_version']}`")
    lines.append(f"- Gate artifact directory: `{report['gate_artifact_dir']}`")
    lines.append(f"- Artifact timestamp (UTC): `{report['artifact_timestamp_utc']}`")
    lines.append("")

    lines.append("## Platform")
    lines.extend(markdown_table(list(report["platform"].items())))
    lines.append("")

    lines.append("## Gate Configuration")
    lines.extend(
        markdown_table(
            [
                ("full_timeout_secs", gate_config["full_timeout_secs"]),
                ("parse_timeout_secs", gate_config["parse_timeout_secs"]),
                ("build_timeout_secs", gate_config["build_timeout_secs"]),
                ("parse_shard_count", gate_config["parse_shard_count"]),
                ("parse_shard_selector", gate_config["parse_shard_selector"]),
                ("strict_mode", gate_config["strict_mode"]),
                ("no_rust_fallback_requested", gate_config["no_rust_fallback_requested"]),
                ("no_rust_fallback_observed", gate_config["no_rust_fallback_observed"]),
            ]
        )
    )
    lines.append("")

    lines.append("## Gate Outcome")
    lines.extend(
        markdown_table(
            [
                ("summary_pass", gate_outcome["summary_pass"]),
                ("summary_fail", gate_outcome["summary_fail"]),
                ("summary_status", gate_outcome["summary_status"]),
                ("full_selfhost_markers_pass", gate_outcome["full_selfhost_markers_pass"]),
                ("parse_all_report_markers_pass", gate_outcome["parse_all_report_markers_pass"]),
                ("parse_all_shards_markers_pass", gate_outcome["parse_all_shards_markers_pass"]),
                ("publication_ready", gate_outcome["publication_ready"]),
            ]
        )
    )
    lines.append("")

    lines.append("## Full Selfhost Marker Counts")
    full_markers = markers["full_selfhost"]
    lines.extend(
        markdown_table(
            [
                ("parse_all_header_count", full_markers["parse_all_header_count"]),
                ("parse_all_summary_count", full_markers["parse_all_summary_count"]),
                ("suites_all_count", full_markers["suites_all_count"]),
                ("suite_fail_count", full_markers["suite_fail_count"]),
                ("fallback_marker_count", full_markers["fallback_marker_count"]),
                ("strict_error_marker_count", full_markers["strict_error_marker_count"]),
                ("driver_output_marker_count", full_markers["driver_output_marker_count"]),
                ("oracle_marker_count", full_markers["oracle_marker_count"]),
                ("strict_marker_occurrences", full_markers["strict_marker_occurrences"]),
                ("markers_pass", full_markers["markers_pass"]),
            ]
        )
    )
    lines.append("")

    lines.append("## Parse-All Report Outcome")
    report_summary = parse_all_report["summary"]
    lines.extend(
        markdown_table(
            [
                ("corpus", report_summary["corpus"]),
                ("shard_count", report_summary["shard_count"]),
                ("total_files", report_summary["total_files"]),
                ("shard_lines_observed", parse_all_report["shard_lines_observed"]),
                ("markers_pass", parse_all_report["markers_pass"]),
            ]
        )
    )
    lines.append("")

    lines.append("## Parse-All Shards Outcome")
    shards_multi = parse_all_shards["multi_summary"]
    lines.extend(
        markdown_table(
            [
                ("selector", shards_multi["selector"]),
                ("executed", shards_multi["executed"]),
                ("passed", shards_multi["passed"]),
                ("failed", shards_multi["failed"]),
                ("invalid", shards_multi["invalid"]),
                ("summary_lines_observed", parse_all_shards["summary_lines_observed"]),
                ("aggregate_selected", parse_all_shards["aggregate_selected"]),
                ("aggregate_passed", parse_all_shards["aggregate_passed"]),
                ("aggregate_failed", parse_all_shards["aggregate_failed"]),
                ("markers_pass", parse_all_shards["markers_pass"]),
            ]
        )
    )
    lines.append("")

    lines.append("## Parse-All Shard Line Items")
    lines.append("| shard_index | selected | passed | failed |")
    lines.append("|---:|---:|---:|---:|")
    for outcome in parse_all_shards["shard_outcomes"]:
        lines.append(
            f"| {outcome['shard_index']} | {outcome['selected']} | "
            f"{outcome['passed']} | {outcome['failed']} |"
        )
    lines.append("")

    lines.append("## Driver/Oracle Marker Presence")
    lines.append("| Log | driver_fallback | driver_strict_error | driver_output | oracle | status |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for log_name in ("full_log", "report_log", "shards_log"):
        info = fallback[log_name]
        status = (
            "clean"
            if info["driver_fallback"] == 0
            and info["driver_strict_error"] == 0
            and info["oracle"] == 0
            else "present"
        )
        lines.append(
            f"| {log_name} | {info['driver_fallback']} | {info['driver_strict_error']} | "
            f"{info['driver_output']} | {info['oracle']} | {status} |"
        )
    lines.append(
        f"| total | {fallback['total_driver_fallback_markers']} | "
        f"{fallback['total_driver_strict_error_markers']} | "
        f"{fallback['total_driver_output_markers']} | "
        f"{fallback['total_oracle_markers']} | "
        f"{status_word(fallback['no_rust_fallback_observed'])} |"
    )
    lines.append("")

    lines.append("## Inputs")
    lines.extend(
        markdown_table(
            [
                ("summary_file", report["inputs"]["summary_file"]),
                ("full_log", report["inputs"]["full_log"]),
                ("report_log", report["inputs"]["report_log"]),
                ("shards_log", report["inputs"]["shards_log"]),
            ]
        )
    )
    lines.append("")
    return "\n".join(lines)


def build_report(paths: GatePaths) -> dict[str, Any]:
    summary = parse_summary_file(paths.summary_file)
    full_lines = read_lines(paths.full_log)
    report_lines = read_lines(paths.report_log)
    shards_lines = read_lines(paths.shards_log)

    full_strict = strict_value_from_log(full_lines, str(paths.full_log))
    report_strict = strict_value_from_log(report_lines, str(paths.report_log))
    shards_strict = strict_value_from_log(shards_lines, str(paths.shards_log))

    strict_values = {
        full_strict["strict_value"],
        report_strict["strict_value"],
        shards_strict["strict_value"],
    }
    if len(strict_values) != 1:
        raise ParseError(f"inconsistent strict values across logs: {sorted(strict_values)}")
    strict_mode = strict_values.pop()

    full_markers = parse_full_selfhost_markers(full_lines)
    parse_all_report = parse_parse_all_report(report_lines, summary["parse_shard_count"])
    parse_all_shards = parse_parse_all_shards(shards_lines, summary["parse_shard_count"])

    fallback_oracle = {
        "full_log": {
            "driver_fallback": count_regex(full_lines, DRIVER_ORCH_FALLBACK_PATTERN),
            "driver_strict_error": count_regex(full_lines, DRIVER_ORCH_STRICT_ERROR_PATTERN),
            "driver_output": count_regex(full_lines, DRIVER_OUTPUT_PATTERN),
            "oracle": count_contains(full_lines, ORACLE_MARKER),
        },
        "report_log": {
            "driver_fallback": count_regex(report_lines, DRIVER_ORCH_FALLBACK_PATTERN),
            "driver_strict_error": count_regex(report_lines, DRIVER_ORCH_STRICT_ERROR_PATTERN),
            "driver_output": count_regex(report_lines, DRIVER_OUTPUT_PATTERN),
            "oracle": count_contains(report_lines, ORACLE_MARKER),
        },
        "shards_log": {
            "driver_fallback": count_regex(shards_lines, DRIVER_ORCH_FALLBACK_PATTERN),
            "driver_strict_error": count_regex(shards_lines, DRIVER_ORCH_STRICT_ERROR_PATTERN),
            "driver_output": count_regex(shards_lines, DRIVER_OUTPUT_PATTERN),
            "oracle": count_contains(shards_lines, ORACLE_MARKER),
        },
    }
    fallback_oracle["total_driver_fallback_markers"] = (
        fallback_oracle["full_log"]["driver_fallback"]
        + fallback_oracle["report_log"]["driver_fallback"]
        + fallback_oracle["shards_log"]["driver_fallback"]
    )
    fallback_oracle["total_driver_strict_error_markers"] = (
        fallback_oracle["full_log"]["driver_strict_error"]
        + fallback_oracle["report_log"]["driver_strict_error"]
        + fallback_oracle["shards_log"]["driver_strict_error"]
    )
    fallback_oracle["total_driver_output_markers"] = (
        fallback_oracle["full_log"]["driver_output"]
        + fallback_oracle["report_log"]["driver_output"]
        + fallback_oracle["shards_log"]["driver_output"]
    )
    fallback_oracle["total_oracle_markers"] = (
        fallback_oracle["full_log"]["oracle"]
        + fallback_oracle["report_log"]["oracle"]
        + fallback_oracle["shards_log"]["oracle"]
    )
    fallback_oracle["no_rust_fallback_observed"] = (
        fallback_oracle["total_driver_fallback_markers"] == 0
        and fallback_oracle["total_driver_strict_error_markers"] == 0
        and fallback_oracle["total_oracle_markers"] == 0
    )

    gate_outcome = {
        "summary_pass": summary["summary_pass"],
        "summary_fail": summary["summary_fail"],
        "summary_status": "pass" if summary["summary_fail"] == 0 else "fail",
        "full_selfhost_markers_pass": full_markers["markers_pass"],
        "parse_all_report_markers_pass": parse_all_report["markers_pass"],
        "parse_all_shards_markers_pass": parse_all_shards["markers_pass"],
    }
    gate_outcome["publication_ready"] = (
        gate_outcome["summary_fail"] == 0
        and gate_outcome["full_selfhost_markers_pass"]
        and gate_outcome["parse_all_report_markers_pass"]
        and gate_outcome["parse_all_shards_markers_pass"]
        and fallback_oracle["no_rust_fallback_observed"]
        and strict_mode == 1
    )

    timestamp_inputs = [
        paths.summary_file.stat().st_mtime,
        paths.full_log.stat().st_mtime,
        paths.report_log.stat().st_mtime,
        paths.shards_log.stat().st_mtime,
    ]

    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "gate_artifact_dir": str(paths.gate_dir.resolve()),
        "artifact_timestamp_utc": isoformat_utc_from_epoch(max(timestamp_inputs)),
        "platform": platform_info(),
        "gate_config": {
            "full_timeout_secs": summary["full_timeout_secs"],
            "parse_timeout_secs": summary["parse_timeout_secs"],
            "build_timeout_secs": None,
            "parse_shard_count": summary["parse_shard_count"],
            "parse_shard_selector": summary["parse_shard_selector"],
            "strict_mode": bool(strict_mode),
            "no_rust_fallback_requested": True,
            "no_rust_fallback_observed": fallback_oracle["no_rust_fallback_observed"],
        },
        "gate_outcome": gate_outcome,
        "markers": {
            "full_selfhost": {
                **full_markers,
                "strict_marker_occurrences": full_strict["occurrences"],
            },
            "parse_all_report": {
                "summary_lines": 1,
                "shard_lines": parse_all_report["shard_lines_observed"],
                "strict_marker_occurrences": report_strict["occurrences"],
            },
            "parse_all_shards": {
                "multi_summary_lines": 1,
                "summary_lines": parse_all_shards["summary_lines_observed"],
                "strict_marker_occurrences": shards_strict["occurrences"],
            },
        },
        "parse_all_report": parse_all_report,
        "parse_all_shards": parse_all_shards,
        "fallback_oracle_markers": fallback_oracle,
        "inputs": {
            "summary_file": str(paths.summary_file),
            "full_log": str(paths.full_log),
            "report_log": str(paths.report_log),
            "shards_log": str(paths.shards_log),
        },
    }


def main() -> int:
    args = parse_args()
    gate_dir = Path(args.gate_dir).resolve()
    summary_file = gate_dir / "artifacts" / "summary.txt"

    try:
        ensure_file(summary_file, "summary file")
        summary = parse_summary_file(summary_file)

        paths = GatePaths(
            gate_dir=gate_dir,
            summary_file=summary_file,
            full_log=resolve_log_path(gate_dir, summary["full_log"], "full_selfhost.log"),
            report_log=resolve_log_path(gate_dir, summary["report_log"], "parse_all_report.log"),
            shards_log=resolve_log_path(gate_dir, summary["shards_log"], "parse_all_shards.log"),
        )

        report = build_report(paths)
    except ParseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    json_out = (
        Path(args.json_out).resolve()
        if args.json_out
        else gate_dir / "artifacts" / "reproducibility_report.v1.json"
    )
    md_out = (
        Path(args.md_out).resolve()
        if args.md_out
        else gate_dir / "artifacts" / "reproducibility_report.v1.md"
    )

    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)

    json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_out.write_text(render_markdown(report) + "\n", encoding="utf-8")

    print(f"wrote_json={json_out}")
    print(f"wrote_markdown={md_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
