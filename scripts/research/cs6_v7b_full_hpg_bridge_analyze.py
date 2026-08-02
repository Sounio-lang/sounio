#!/usr/bin/env python3
"""Analyze V7-B bridge execution failure anatomy."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


EXECUTION_SCHEMA = "sounio.cs6.v7b-full-hpg-bridge-execution-summary.v1"
ANATOMY_SCHEMA = "sounio.cs6.v7b-full-hpg-bridge-anatomy.v1"
EXPECTED_ATTEMPTS = 6
SUMMARY_KEYS = (
    "ALL_FINITE",
    "STRUCTURAL_PASS",
    "C1_ORIENTATION_UNRESOLVED",
    "C2_HULL_ORIENTATION_UNRESOLVED",
    "AFFINE_ORIENTATION_CERTIFIED",
    "LIOUVILLE_ORIENTATION_CERTIFIED",
    "ANY_PROJECTIVE_ORIENTATION_CERTIFIED",
    "EVENT1_CHARTS_CERTIFIED",
    "EVENT2_CHARTS_CERTIFIED",
    "HOMOGENEOUS_ORIENTATION_CERTIFIED",
    "HOMOGENEOUS_COMPUTATION_VALID",
    "HOMOGENEOUS_CERTIFICATE_PASS",
    "CERTIFICATE_PASS",
    "PROBE_PASS",
)


def fail(message: str) -> None:
    raise SystemExit(f"V7-B anatomy error: {message}")


def canonical(path: Path) -> str:
    raw = path.read_bytes()
    if raw and (not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw):
        fail(f"noncanonical text: {path}")
    try:
        return raw.decode("ascii")
    except UnicodeError as error:
        raise SystemExit(f"non-ASCII text: {path}") from error


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_kv_lines(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in canonical(path).splitlines():
        if line.count("=") != 1:
            fail(f"malformed key/value line in {path}: {line}")
        key, value = line.split("=", 1)
        if not key or not value or key in fields:
            fail(f"bad key/value line in {path}: {line}")
        fields[key] = value
    return fields


def parse_results(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(canonical(path).splitlines(), delimiter="\t"))


def parse_worker_summary(stdout_path: Path) -> dict[str, str]:
    summaries = [line for line in canonical(stdout_path).splitlines() if line.startswith("SUMMARY ")]
    if not summaries:
        return {}
    if len(summaries) != 1:
        fail(f"expected one worker SUMMARY in {stdout_path}")
    fields: dict[str, str] = {}
    for token in summaries[0].split()[1:]:
        if token.count("=") != 1:
            fail(f"malformed worker summary token in {stdout_path}: {token}")
        key, value = token.split("=", 1)
        if key in fields:
            fail(f"duplicate worker summary key in {stdout_path}: {key}")
        fields[key] = value
    return fields


def attempt_dir(receipt_dir: Path, row: dict[str, str]) -> Path:
    index = int(row["ATTEMPT_INDEX"])
    return receipt_dir / "attempts" / f"A{index:04d}_{row['PARENT_V7_ORDINAL']}_{row['CARRIER']}"


def classify(row: dict[str, str], worker: dict[str, str]) -> tuple[str, str]:
    status = row["STATUS"]
    if status == "SECTION_RESIDENT_CROSSING_UNAVAILABLE":
        return "section_resident_crossing_unavailable", "crossing_missing_before_worker_summary"
    if status == "FULL_BRIDGE_PROBE_REJECTED" and worker.get("PROBE_PASS") == "false":
        if worker.get("STRUCTURAL_PASS") == "true" and worker.get("LIOUVILLE_ORIENTATION_CERTIFIED") == "true":
            if worker.get("C1_ORIENTATION_UNRESOLVED") == "true" or worker.get("C2_HULL_ORIENTATION_UNRESOLVED") == "true":
                return "c1_c2_orientation_unresolved", "structural_and_liouville_ok_c1_or_c2_orientation_unresolved"
            if worker.get("ANY_PROJECTIVE_ORIENTATION_CERTIFIED") == "false":
                return "orientation_certificate_rejected", "structural_and_liouville_ok_projective_none"
            return "certificate_rejected", "structural_and_liouville_ok_certificate_false"
        return "worker_probe_rejected", "summary_probe_false"
    if status == "FULL_BRIDGE_PROBE_PASS":
        return "unexpected_full_bridge_pass", "not_allowed_in_anatomy_checkpoint"
    return "unknown", f"status={status}"


def write_tsv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-dir", type=Path, default=Path("scripts/research/receipts/cs6_v7b_full_hpg_bridge_execution_v1"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    execution_dir = args.execution_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    execution_summary = parse_kv_lines(execution_dir / "summary.txt")
    if execution_summary.get("SCHEMA") != EXECUTION_SCHEMA:
        fail("execution summary schema mismatch")
    if execution_summary.get("RUN_VALID") != "true" or execution_summary.get("ATTEMPTS_COMPLETED") != "6":
        fail("execution receipt is not the six-attempt valid run")
    if execution_summary.get("UNKNOWN_FAILURE") != "0":
        fail("execution receipt has unknown failures")

    results = parse_results(execution_dir / "results.tsv")
    if len(results) != EXPECTED_ATTEMPTS:
        fail("expected six result rows")

    anatomy_rows: list[dict[str, str]] = []
    for row in results:
        directory = attempt_dir(execution_dir, row)
        stdout_path = directory / "stdout.txt"
        stderr_path = directory / "stderr.txt"
        if not stdout_path.exists() or not stderr_path.exists():
            fail(f"missing attempt stdout/stderr for attempt {row['ATTEMPT_INDEX']}")
        worker = parse_worker_summary(stdout_path)
        failure_class, failure_reason = classify(row, worker)
        anatomy = {
            "ATTEMPT_INDEX": row["ATTEMPT_INDEX"],
            "PARENT_V7_ORDINAL": row["PARENT_V7_ORDINAL"],
            "CHECKPOINT_ROLE": row["CHECKPOINT_ROLE"],
            "CARRIER": row["CARRIER"],
            "STATUS": row["STATUS"],
            "FAILURE_CLASS": failure_class,
            "FAILURE_REASON": failure_reason,
            "WORKER_SUMMARY_EMITTED": str(bool(worker)).lower(),
            "STDOUT_SHA256": sha256_path(stdout_path),
            "STDERR_SHA256": sha256_path(stderr_path),
        }
        for key in SUMMARY_KEYS:
            anatomy[key] = worker.get(key, "UNEMITTED")
        anatomy_rows.append(anatomy)

    fields = (
        "ATTEMPT_INDEX",
        "PARENT_V7_ORDINAL",
        "CHECKPOINT_ROLE",
        "CARRIER",
        "STATUS",
        "FAILURE_CLASS",
        "FAILURE_REASON",
        "WORKER_SUMMARY_EMITTED",
        "STDOUT_SHA256",
        "STDERR_SHA256",
        *SUMMARY_KEYS,
    )
    write_tsv(out_dir / "anatomy.tsv", anatomy_rows, fields)

    counts: dict[str, int] = {}
    for row in anatomy_rows:
        counts[row["FAILURE_CLASS"]] = counts.get(row["FAILURE_CLASS"], 0) + 1
    full_passes = counts.get("unexpected_full_bridge_pass", 0)
    if full_passes:
        fail("unexpected full bridge pass in anatomy checkpoint")
    if counts.get("unknown", 0):
        fail("unknown anatomy class")

    controls = [row for row in anatomy_rows if row["CHECKPOINT_ROLE"].startswith("POSITIVE_CONTROL")]
    targets = [row for row in anatomy_rows if row["CHECKPOINT_ROLE"] == "MASKED_TARGET"]
    control_c1_c2_unresolved = sum(row["FAILURE_CLASS"] == "c1_c2_orientation_unresolved" for row in controls)
    control_orientation_rejections = sum(row["FAILURE_CLASS"] == "orientation_certificate_rejected" for row in controls)
    target_crossing_missing = sum(row["FAILURE_CLASS"] == "section_resident_crossing_unavailable" for row in targets)

    summary_lines = [
        f"SCHEMA={ANATOMY_SCHEMA}",
        "RUN_VALID=true",
        f"ATTEMPTS_ANALYZED={len(anatomy_rows)}",
        f"WORKER_SUMMARIES_EMITTED={sum(row['WORKER_SUMMARY_EMITTED'] == 'true' for row in anatomy_rows)}",
        f"SECTION_RESIDENT_CROSSING_UNAVAILABLE={counts.get('section_resident_crossing_unavailable', 0)}",
        f"C1_C2_ORIENTATION_UNRESOLVED={counts.get('c1_c2_orientation_unresolved', 0)}",
        f"ORIENTATION_CERTIFICATE_REJECTED={counts.get('orientation_certificate_rejected', 0)}",
        f"CERTIFICATE_REJECTED={counts.get('certificate_rejected', 0)}",
        f"UNKNOWN_ANATOMY={counts.get('unknown', 0)}",
        f"CONTROL_C1_C2_ORIENTATION_UNRESOLVED={control_c1_c2_unresolved}",
        f"CONTROL_ORIENTATION_REJECTIONS={control_orientation_rejections}",
        f"TARGET_CROSSING_MISSING={target_crossing_missing}",
        "NEXT_EXPERIMENT_CLASS=c1_c2_orientation_and_section_crossing_reparameterization",
        "V7_B_ELIGIBILITY=false",
        "PROMOTION_ELIGIBLE=false",
        "OPEN_PROBLEM_SOLVED=false",
        "FPGA_EXECUTION=false",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="ascii")
    (out_dir / "files.sha256").write_text(
        "\n".join(
            [
                f"{sha256_path(execution_dir / 'results.tsv')}  execution/results.tsv",
                f"{sha256_path(execution_dir / 'summary.txt')}  execution/summary.txt",
                f"{sha256_path(out_dir / 'anatomy.tsv')}  anatomy.tsv",
                f"{sha256_path(out_dir / 'summary.txt')}  summary.txt",
            ]
        )
        + "\n",
        encoding="ascii",
    )
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
