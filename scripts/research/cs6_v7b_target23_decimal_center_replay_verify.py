#!/usr/bin/env python3
"""Verify raw evidence from the independent Decimal center-orbit replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from decimal import Decimal, localcontext
from pathlib import Path

from cs6_v7b_target23_decimal_center_replay_run import (
    CARRIERS, EXPECTED_LEAVES, RESULT_COLUMNS, WORKER_REL, binary64_decimal,
    build_plan, canonical, parse_fields,
)


SUMMARY_KEYS = (
    "SCHEMA", "PRE_EXECUTION_GIT_COMMIT", "RUN_COMPLETE", "POINTWISE_ORBITS",
    "POINTWISE_LEAVES_PASS", "ALL_POINTWISE_LEAVES_PASS",
    "INDEPENDENT_POINTWISE_SCOUT_COMPLETED", "RIGOROUS_INTERVAL_CERTIFICATE",
    "INDEPENDENT_INTERVAL_ENGINE", "GLOBAL_HPG_CERTIFICATE", "V7_B_ELIGIBILITY",
    "PROMOTION_ELIGIBLE", "OPEN_PROBLEM_SOLVED", "NOVELTY_OR_PRIORITY_CLAIMED",
    "FPGA_EXECUTION",
)
KEY_RE = re.compile(r"[A-Z][A-Z0-9_]*")
ZS = "22.3274637391"
ORIGIN_X = "15.186446520640786"
ORIGIN_Y = "10.908543194765466"
UNSTABLE_X = "-0.67430316214199759"
UNSTABLE_Y = "-0.73845463335624273"
STABLE_X = "-0.94170446778164518"
STABLE_Y = "0.33644122125579123"
RADIUS_U = "0.004"
RADIUS_S = "0.3"
WORKER_KEYS = (
    "SCHEMA", "WORKER_SOURCE_SHA256", "PYTHON_VERSION", "PYTHON_IMPLEMENTATION",
    "DECIMAL_IMPLEMENTATION", "RUN_CHALLENGE", "ATTEMPT_BINDING", "U_DEPTH",
    "U_INDEX", "S_DEPTH", "S_INDEX",
    *(f"{prefix}_{key}" for prefix in ("COARSE", "FINE") for key in (
        "STEPS", "RETURN1_TIME", "RETURN2_TIME", "ELL", "INITIAL_NORMAL",
        "FINAL_NORMAL", "Q0_AREA", "DETERMINANT",
    )),
    "ABSOLUTE_DETERMINANT_DELTA", "CENTER_REPLAY_SELF_CONSISTENT",
    "CAPD_USED_BY_INTEGRATOR", "RIGOROUS_INTERVAL_CERTIFICATE",
    "POINTWISE_FALSIFICATION_ONLY",
)


def fail(message: str) -> None:
    raise SystemExit(f"decimal center replay verify error: {message}")


def kv_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in canonical(path).splitlines():
        if line.count("=") != 1:
            fail(f"malformed key-value line: {path}")
        key, value = line.split("=", 1)
        if not KEY_RE.fullmatch(key) or not value or key in result:
            fail(f"invalid key-value field: {path}:{key}")
        result[key] = value
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recompute_determinant(fields: dict[str, str], prefix: str, precision: int) -> Decimal:
    with localcontext() as context:
        context.prec = precision
        ell = Decimal(fields[f"{prefix}_ELL"])
        initial = Decimal(fields[f"{prefix}_INITIAL_NORMAL"])
        final = Decimal(fields[f"{prefix}_FINAL_NORMAL"])
        area = Decimal(fields[f"{prefix}_Q0_AREA"])
        return ell.exp() * initial / final * area


def recompute_source_geometry(leaf: object, precision: int) -> tuple[Decimal, Decimal]:
    with localcontext() as context:
        context.prec = precision
        ru, rs = Decimal(RADIUS_U), Decimal(RADIUS_S)
        u = -ru + (Decimal(leaf.u_index) + Decimal("0.5")) * (2 * ru) / Decimal(2**leaf.u_depth)
        s = -rs + (Decimal(leaf.s_index) + Decimal("0.5")) * (2 * rs) / Decimal(2**leaf.s_depth)
        x = Decimal(ORIGIN_X) + Decimal(UNSTABLE_X) * u + Decimal(STABLE_X) * s
        y = Decimal(ORIGIN_Y) + Decimal(UNSTABLE_Y) * u + Decimal(STABLE_Y) * s
        initial_normal = x * y - Decimal(ZS)
        q0_area = (
            Decimal(UNSTABLE_X) * Decimal(STABLE_Y)
            - Decimal(STABLE_X) * Decimal(UNSTABLE_Y)
        ) * ru * rs
        return initial_normal, q0_area


def verify(receipt: Path, source_commit: str) -> dict[str, str]:
    root = Path.cwd()
    summary = kv_file(receipt / "summary.txt")
    if tuple(summary) != SUMMARY_KEYS:
        fail("summary key order or population drifted")
    if summary["SCHEMA"] != "sounio.cs6.v7b-target23-decimal-center-replay-summary.v1":
        fail("summary schema drifted")
    if summary["PRE_EXECUTION_GIT_COMMIT"] != source_commit:
        fail("pre-execution commit mismatch")
    rows = list(csv.DictReader(canonical(receipt / "results.tsv").splitlines(), delimiter="\t"))
    if len(rows) != EXPECTED_LEAVES or not rows or tuple(rows[0]) != RESULT_COLUMNS:
        fail("result shape drifted")
    try:
        plans = build_plan(root, source_commit)
    except (SystemExit, ValueError) as error:
        fail(f"could not reconstruct frozen plan: {error}")
    worker_sha = sha256(root / WORKER_REL)
    passed = 0
    maximum_delta = Decimal(0)
    minimum_margin: Decimal | None = None
    runtime_versions: set[str] = set()
    command_executables: set[str] = set()
    for index, (row, plan) in enumerate(zip(rows, plans, strict=True), 1):
        leaf = plan.leaf
        expected = {
            "LEAF_INDEX": str(index), "LAYER": leaf.layer, "LEAF_ID": leaf.leaf_id,
            "U_DEPTH": str(leaf.u_depth), "U_INDEX": str(leaf.u_index),
            "S_DEPTH": str(leaf.s_depth), "S_INDEX": str(leaf.s_index),
            "INPUT_SHA256": leaf.input_sha256, "RUN_CHALLENGE": plan.challenge,
            "ATTEMPT_BINDING": plan.binding,
            "C0HORECT2_LOWER": plan.capd[CARRIERS[0]][0],
            "C0HORECT2_UPPER": plan.capd[CARRIERS[0]][1],
            "C0RECT2_LOWER": plan.capd[CARRIERS[1]][0],
            "C0RECT2_UPPER": plan.capd[CARRIERS[1]][1],
        }
        for key, value in expected.items():
            if row[key] != value:
                fail(f"leaf {index} field mismatch: {key}")
        attempt_dir = receipt / "attempts" / plan.identity
        stdout_path, stderr_path = attempt_dir / "stdout.txt", attempt_dir / "stderr.txt"
        command_path = attempt_dir / "command.txt"
        if not stdout_path.is_file() or not stderr_path.is_file() or not command_path.is_file():
            fail(f"leaf evidence missing: {index}")
        stdout, stderr = stdout_path.read_bytes(), stderr_path.read_bytes()
        if row["STDOUT_SHA256"] != hashlib.sha256(stdout).hexdigest():
            fail(f"stdout hash mismatch: {index}")
        if row["STDERR_SHA256"] != hashlib.sha256(stderr).hexdigest() or stderr:
            fail(f"stderr mismatch or nonempty: {index}")
        if row["WORKER_RC"] != "0":
            fail(f"worker failed: {index}")
        fields = parse_fields(stdout, plan.identity)
        if tuple(fields) != WORKER_KEYS:
            fail(f"worker field population or order drifted: leaf {index}")
        runtime_versions.add(fields["PYTHON_VERSION"])
        required_worker = {
            "SCHEMA": "sounio.cs6.v7b-target23-decimal-center-worker.v1",
            "WORKER_SOURCE_SHA256": worker_sha, "RUN_CHALLENGE": plan.challenge,
            "ATTEMPT_BINDING": plan.binding, "U_DEPTH": str(leaf.u_depth),
            "U_INDEX": str(leaf.u_index), "S_DEPTH": str(leaf.s_depth),
            "S_INDEX": str(leaf.s_index), "CENTER_REPLAY_SELF_CONSISTENT": "true",
            "CAPD_USED_BY_INTEGRATOR": "false", "RIGOROUS_INTERVAL_CERTIFICATE": "false",
            "POINTWISE_FALSIFICATION_ONLY": "true",
            "PYTHON_IMPLEMENTATION": "CPython", "DECIMAL_IMPLEMENTATION": "stdlib-decimal",
        }
        for key, value in required_worker.items():
            if fields.get(key) != value:
                fail(f"worker field mismatch: leaf {index}:{key}")
        mapping = {
            "COARSE_RETURN1_TIME": "COARSE_RETURN1_TIME", "COARSE_RETURN2_TIME": "COARSE_RETURN2_TIME",
            "FINE_RETURN1_TIME": "FINE_RETURN1_TIME", "FINE_RETURN2_TIME": "FINE_RETURN2_TIME",
            "COARSE_ELL": "COARSE_ELL", "FINE_ELL": "FINE_ELL",
            "COARSE_INITIAL_NORMAL": "COARSE_INITIAL_NORMAL", "COARSE_FINAL_NORMAL": "COARSE_FINAL_NORMAL",
            "FINE_INITIAL_NORMAL": "FINE_INITIAL_NORMAL", "FINE_FINAL_NORMAL": "FINE_FINAL_NORMAL",
            "Q0_AREA": "FINE_Q0_AREA", "COARSE_DETERMINANT": "COARSE_DETERMINANT",
            "FINE_DETERMINANT": "FINE_DETERMINANT",
            "ABSOLUTE_DETERMINANT_DELTA": "ABSOLUTE_DETERMINANT_DELTA",
        }
        for result_key, worker_key in mapping.items():
            if row[result_key] != fields.get(worker_key):
                fail(f"raw/result mismatch: leaf {index}:{result_key}")
        if Decimal(fields["COARSE_Q0_AREA"]) != Decimal(fields["FINE_Q0_AREA"]):
            fail(f"Q0 area changed between resolutions: {index}")
        for prefix, precision in (("COARSE", 50), ("FINE", 80)):
            determinant = Decimal(fields[f"{prefix}_DETERMINANT"])
            expected_initial, expected_area = recompute_source_geometry(leaf, precision)
            if Decimal(fields[f"{prefix}_INITIAL_NORMAL"]) != expected_initial:
                fail(f"source center or initial normal mismatch: leaf {index}:{prefix}")
            if Decimal(fields[f"{prefix}_Q0_AREA"]) != expected_area:
                fail(f"source frame area mismatch: leaf {index}:{prefix}")
            if determinant != recompute_determinant(fields, prefix, precision):
                fail(f"Liouville formula mismatch: leaf {index}:{prefix}")
            if not 0 < int(fields[f"{prefix}_STEPS"]) <= 20000:
                fail(f"step count mismatch: leaf {index}:{prefix}")
            if not (Decimal(fields[f"{prefix}_RETURN1_TIME"]) < Decimal(fields[f"{prefix}_RETURN2_TIME"])):
                fail(f"return order mismatch: leaf {index}:{prefix}")
            if Decimal(fields[f"{prefix}_INITIAL_NORMAL"]) <= 0 or Decimal(fields[f"{prefix}_FINAL_NORMAL"]) <= 0:
                fail(f"nonpositive normal velocity: leaf {index}:{prefix}")
            if determinant >= 0:
                fail(f"nonnegative determinant: leaf {index}:{prefix}")
        coarse, fine = Decimal(fields["COARSE_DETERMINANT"]), Decimal(fields["FINE_DETERMINANT"])
        with localcontext() as context:
            context.prec = 100
            delta = abs(coarse - fine)
        if delta != Decimal(fields["ABSOLUTE_DETERMINANT_DELTA"]) or delta > Decimal("1E-16"):
            fail(f"resolution disagreement: leaf {index}")
        intervals = [tuple(binary64_decimal(x) for x in plan.capd[c]) for c in CARRIERS]
        coarse_inside = all(lower < coarse < upper for lower, upper in intervals)
        fine_inside = all(lower < fine < upper for lower, upper in intervals)
        if row["COARSE_INSIDE_BOTH_CAPD"] != str(coarse_inside).lower():
            fail(f"coarse containment mismatch: leaf {index}")
        if row["FINE_INSIDE_BOTH_CAPD"] != str(fine_inside).lower():
            fail(f"fine containment mismatch: leaf {index}")
        leaf_pass = coarse_inside and fine_inside and row["CENTER_REPLAY_SELF_CONSISTENT"] == "true"
        if row["POINTWISE_LEAF_PASS"] != str(leaf_pass).lower() or not leaf_pass:
            fail(f"pointwise leaf failed: {index}")
        maximum_delta = max(maximum_delta, delta)
        for lower, upper in intervals:
            margin = min(fine - lower, upper - fine)
            minimum_margin = margin if minimum_margin is None else min(minimum_margin, margin)
        command = canonical(command_path).split()
        expected_tail = (str(leaf.u_depth), str(leaf.u_index), str(leaf.s_depth), str(leaf.s_index), plan.challenge, plan.binding)
        if (len(command) != 9 or command[1] != "-B"
                or Path(command[2]).resolve() != (root / WORKER_REL).resolve()
                or tuple(command[3:]) != expected_tail):
            fail(f"worker command mismatch: leaf {index}")
        command_executables.add(command[0])
        passed += 1
    expected_summary = {
        "RUN_COMPLETE": "true", "POINTWISE_ORBITS": str(EXPECTED_LEAVES),
        "POINTWISE_LEAVES_PASS": str(passed), "ALL_POINTWISE_LEAVES_PASS": "true",
        "INDEPENDENT_POINTWISE_SCOUT_COMPLETED": "false",
        "RIGOROUS_INTERVAL_CERTIFICATE": "false", "INDEPENDENT_INTERVAL_ENGINE": "false",
        "GLOBAL_HPG_CERTIFICATE": "false", "V7_B_ELIGIBILITY": "false",
        "PROMOTION_ELIGIBLE": "false", "OPEN_PROBLEM_SOLVED": "false",
        "NOVELTY_OR_PRIORITY_CLAIMED": "false", "FPGA_EXECUTION": "false",
    }
    for key, value in expected_summary.items():
        if summary[key] != value:
            fail(f"summary mismatch: {key}")
    provenance = receipt / "provenance"
    if canonical(provenance / "git-head.txt").strip() != source_commit:
        fail("Slurm payload Git identity mismatch")
    slurm = kv_file(provenance / "slurm-context.txt")
    if not slurm.get("SLURM_JOB_ID", "").isdigit() or slurm.get("SLURM_CPUS_PER_TASK") != "32":
        fail("Slurm execution attestation invalid")
    runtime = kv_file(provenance / "python-runtime.txt")
    if runtime.get("PYTHON_DECIMAL_IMPLEMENTATION") != "stdlib-decimal" or runtime.get("CAPD_IMPORTED") != "false":
        fail("independent runtime attestation invalid")
    if runtime_versions != {runtime.get("PYTHON_VERSION")}:
        fail("runtime version attestation mismatch")
    if command_executables != {runtime.get("PYTHON_EXECUTABLE")}:
        fail("runtime executable attestation mismatch")
    return {
        "SLURM_JOB_ID": slurm["SLURM_JOB_ID"], "LEAVES_VERIFIED": str(passed),
        "MAXIMUM_ABSOLUTE_DELTA": str(maximum_delta),
        "MINIMUM_FINE_CAPD_MARGIN": str(minimum_margin),
        "INDEPENDENT_POINTWISE_SCOUT_COMPLETED": "true",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    result = verify(args.receipt, args.source_commit)
    print("VERIFY_SCHEMA=sounio.cs6.v7b-target23-decimal-center-replay-verification.v1")
    for key in ("SLURM_JOB_ID", "LEAVES_VERIFIED", "MAXIMUM_ABSOLUTE_DELTA", "MINIMUM_FINE_CAPD_MARGIN", "INDEPENDENT_POINTWISE_SCOUT_COMPLETED"):
        print(f"{key}={result[key]}")
    print("RIGOROUS_INTERVAL_CERTIFICATE=false")
    print("INDEPENDENT_INTERVAL_ENGINE=false")
    print("GLOBAL_HPG_CERTIFICATE=false")
    print("V7_B_ELIGIBILITY=false")
    print("OPEN_PROBLEM_SOLVED=false")
    print("FPGA_EXECUTION=false")


if __name__ == "__main__":
    main()
