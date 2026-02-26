#!/usr/bin/env python3
"""Emit parallel cutover status artifacts from gate markers and provenance reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

CUTOVER_SCHEMA = "sounio.omega.parallel-cutover-status.v1"
PROGRESS_SCHEMA = "sounio.omega.selfhost-compiler-progress.v1"

TRACK_B_STEPS = [
    ("data_structures", "DATA_STRUCTURES_PASS"),
    ("gpu_ir_expansion", "GPU_IR_EXPANSION_PASS"),
    ("hlir_gpu_cross_coverage", "HLIR_GPU_CROSS_COVERAGE_PASS"),
    ("hlir_lowering", "HLIR_LOWERING_PASS"),
    ("metal_msl_codegen", "METAL_MSL_CODEGEN_PASS"),
    ("ptx_regalloc_expansion", "PTX_REGALLOC_EXPANSION_PASS"),
    ("gpu_opcode_smoke", "GPU_OPCODE_SMOKE_PASS"),
]

TRACK_B_ORDERED_PLAN_MARKERS = [
    "DATA_STRUCTURES_PASS",
    "GPU_IR_EXPANSION_PASS",
    "HLIR_LOWERING_PASS",
    "METAL_MSL_CODEGEN_PASS",
    "PTX_REGALLOC_EXPANSION_PASS",
]


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit parallel no-rust + selfhost compiler status")
    parser.add_argument("--gate-log", default="artifacts/omega_sprint1_gate.log")
    parser.add_argument(
        "--provenance",
        default="artifacts/omega/souc_release_provenance.v1.json",
    )
    parser.add_argument(
        "--no-rust-surface",
        default="artifacts/omega/no_rust_execution_surface.v1.json",
    )
    parser.add_argument(
        "--out-cutover",
        default="artifacts/omega/parallel_cutover_status.v1.json",
    )
    parser.add_argument(
        "--out-progress",
        default="artifacts/omega/selfhost_compiler_progress.v1.json",
    )
    return parser.parse_args()


def marker_offset(log_text: str, marker: str) -> int:
    return log_text.find(marker)


def read_json_status(
    path: Path, field: str = "status"
) -> tuple[str, bool, list[str], dict | None]:
    errors: list[str] = []
    if not path.exists():
        return ("missing", False, [f"missing report: {path}"], None)
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return ("invalid", False, [f"invalid report: {path}: {exc}"], None)
    status = str(payload.get(field, "unknown"))
    ok = status == "pass"
    return (status, ok, errors, payload)


def main() -> int:
    args = parse_args()

    gate_log_path = Path(args.gate_log)
    provenance_path = Path(args.provenance)
    no_rust_surface_path = Path(args.no_rust_surface)
    out_cutover_path = Path(args.out_cutover)
    out_progress_path = Path(args.out_progress)
    crates_path = Path("crates")

    errors: list[str] = []

    if not gate_log_path.exists():
        errors.append(f"missing gate log: {gate_log_path}")
        gate_text = ""
    else:
        gate_text = gate_log_path.read_text()

    provenance_status, provenance_ok, provenance_errors, _ = read_json_status(provenance_path)
    errors.extend(provenance_errors)

    no_rust_status, no_rust_ok, no_rust_errors, no_rust_payload = read_json_status(
        no_rust_surface_path
    )
    errors.extend(no_rust_errors)
    no_rust_runtime_ok = False
    if isinstance(no_rust_payload, dict):
        runtime_violations = no_rust_payload.get("runtime_violation_count")
        no_rust_runtime_ok = (
            no_rust_payload.get("require_runtime_absence") is True
            and runtime_violations in (0, 0.0)
        )

    track_a_requirements = ["REPO_HARD_NO_RUST_PASS", "SOUC_RELEASE_PROVENANCE_PASS"]
    track_a_missing = [
        marker for marker in track_a_requirements if marker_offset(gate_text, marker) < 0
    ]
    if not provenance_ok:
        track_a_missing.append("SOUC_RELEASE_PROVENANCE_REPORT_PASS")
    if not no_rust_ok:
        track_a_missing.append("NO_RUST_EXECUTION_SURFACE_REPORT_PASS")
    if not no_rust_runtime_ok:
        track_a_missing.append("NO_RUST_RUNTIME_ABSENCE_PASS")
    crates_removed = not crates_path.exists()
    if not crates_removed:
        track_a_missing.append("CRATES_MAINLINE_REMOVED_PASS")
    track_a_status = "pass" if not track_a_missing else "fail"

    step_status = []
    track_b_missing = []
    order_violations = []
    previous_plan_offset = -1
    for stage, marker in TRACK_B_STEPS:
        offset = marker_offset(gate_text, marker)
        ok = offset >= 0
        if not ok:
            track_b_missing.append(marker)
        if (
            ok
            and marker in TRACK_B_ORDERED_PLAN_MARKERS
            and offset < previous_plan_offset
        ):
            order_violations.append(
                f"{marker} appears out of canonical plan order "
                f"(offset={offset}, previous_offset={previous_plan_offset})"
            )
        if ok and marker in TRACK_B_ORDERED_PLAN_MARKERS:
            previous_plan_offset = offset
        step_status.append(
            {
                "stage": stage,
                "marker": marker,
                "status": "pass" if ok else "fail",
                "gate_log_offset": offset,
            }
        )

    track_b_status = "pass" if (not track_b_missing and not order_violations) else "fail"
    track_b_order_status = "pass" if not order_violations else "fail"

    blocking_failures = sorted(
        set(
            errors
            + [f"track_a:{item}" for item in track_a_missing]
            + [f"track_b:{item}" for item in track_b_missing]
            + [f"track_b_order:{item}" for item in order_violations]
        )
    )
    parallel_status = "pass" if not blocking_failures else "fail"

    progress = {
        "schema": PROGRESS_SCHEMA,
        "generated_at_utc": now_utc_iso(),
        "status": track_b_status,
        "order_status": track_b_order_status,
        "steps": step_status,
    }

    cutover = {
        "schema": CUTOVER_SCHEMA,
        "generated_at_utc": now_utc_iso(),
        "status": parallel_status,
        "track_a_status": track_a_status,
        "track_b_status": track_b_status,
        "track_b_order_status": track_b_order_status,
        "provenance_status": provenance_status,
        "no_rust_surface_status": no_rust_status,
        "no_rust_runtime_absence_status": "pass" if no_rust_runtime_ok else "fail",
        "crates_mainline_removed_status": "pass" if crates_removed else "fail",
        "blocking_failures": blocking_failures,
    }

    out_progress_path.parent.mkdir(parents=True, exist_ok=True)
    out_progress_path.write_text(json.dumps(progress, indent=2) + "\n")

    out_cutover_path.parent.mkdir(parents=True, exist_ok=True)
    out_cutover_path.write_text(json.dumps(cutover, indent=2) + "\n")

    print(
        "omega_parallel_cutover_status: "
        f"status={parallel_status} track_a={track_a_status} track_b={track_b_status} "
        f"order={track_b_order_status} out={out_cutover_path} progress={out_progress_path}"
    )

    return 0 if parallel_status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
