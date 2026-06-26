#!/usr/bin/env python3
"""Tiny SMT-LIB external-baseline scaffold for solver novelty readiness."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


LCG_A = 1103515245
LCG_C = 12345
LCG_M = 2147483648


@dataclass(frozen=True)
class SmtLibInstance:
    instance_id: int
    name: str
    seed: int
    expected: str
    text: str


def lcg_next(state: int) -> int:
    return (state * LCG_A + LCG_C) % LCG_M


def solver_version(path: str | None) -> str:
    if not path:
        return ""
    for args in ([path, "--version"], [path, "-version"]):
        try:
            completed = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        lines = (completed.stdout + completed.stderr).strip().splitlines()
        if lines:
            return lines[0][:200]
    return "version_unavailable"


def solver_command(name: str, path: str, smt2: Path) -> list[str]:
    if name == "cvc5":
        return [path, "--lang", "smt2", str(smt2)]
    return [path, str(smt2)]


def parse_result(stdout: str, stderr: str) -> str:
    for line in (stdout + "\n" + stderr).splitlines():
        word = line.strip().lower()
        if word == "sat":
            return "SAT"
        if word == "unsat":
            return "UNSAT"
        if word == "unknown":
            return "UNKNOWN"
    return "UNPARSED"


def write_instance(inst: SmtLibInstance, path: Path) -> str:
    path.write_text(inst.text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixed_instances() -> list[SmtLibInstance]:
    instances: list[SmtLibInstance] = [
        SmtLibInstance(
            1,
            "qflia_box_sat",
            0,
            "SAT",
            "\n".join(
                [
                    "(set-logic QF_LIA)",
                    "(declare-const x Int)",
                    "(declare-const y Int)",
                    "(assert (>= x 0))",
                    "(assert (<= x 5))",
                    "(assert (= y (+ x 1)))",
                    "(check-sat)",
                    "",
                ]
            ),
        ),
        SmtLibInstance(
            2,
            "qflia_box_unsat",
            0,
            "UNSAT",
            "\n".join(
                [
                    "(set-logic QF_LIA)",
                    "(declare-const x Int)",
                    "(assert (>= x 3))",
                    "(assert (<= x 2))",
                    "(check-sat)",
                    "",
                ]
            ),
        ),
    ]
    state = 9000
    for idx in range(4):
        state = lcg_next(state)
        lo = int(state % 8)
        state = lcg_next(state)
        gap = int(state % 3) + 1
        hi = lo - gap
        instances.append(
            SmtLibInstance(
                idx + 3,
                "qflia_seeded_bound_unsat",
                9000 + idx,
                "UNSAT",
                "\n".join(
                    [
                        "(set-logic QF_LIA)",
                        f"; seed={9000 + idx}",
                        "(declare-const x Int)",
                        f"(assert (>= x {lo}))",
                        f"(assert (<= x {hi}))",
                        "(check-sat)",
                        "",
                    ]
                ),
            )
        )
    return instances


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(os.environ.get("SOLVER_EXTERNAL_SMTLIB_OUT_DIR", f"/tmp/sounio-solver-external-smtlib-tiny-{stamp}"))
    smt_dir = out_dir / "smt2"
    smt_dir.mkdir(parents=True, exist_ok=True)

    solver_names = ["z3", "cvc5", "yices-smt2"]
    solvers = [(name, shutil.which(name)) for name in solver_names]
    available = [(name, path) for name, path in solvers if path]

    availability_path = out_dir / "solver_availability.csv"
    with availability_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["solver", "path", "available", "version"])
        writer.writeheader()
        for name, path in solvers:
            writer.writerow(
                {
                    "solver": name,
                    "path": path or "",
                    "available": 1 if path else 0,
                    "version": solver_version(path),
                }
            )

    manifest_rows: list[dict[str, object]] = []
    for inst in fixed_instances():
        smt_path = smt_dir / f"{inst.instance_id}_{inst.name}_{inst.seed}.smt2"
        digest = write_instance(inst, smt_path)
        manifest_rows.append(
            {
                "instance_id": inst.instance_id,
                "name": inst.name,
                "seed": inst.seed,
                "expected": inst.expected,
                "sha256": digest,
                "smt2": str(smt_path),
            }
        )

    instance_manifest = out_dir / "smtlib_manifest.csv"
    with instance_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    result_path = out_dir / "external_results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "instance_id",
            "name",
            "seed",
            "expected",
            "solver",
            "solver_path",
            "exit_code",
            "result",
            "matches_expected",
            "elapsed_ms",
            "smt2_sha256",
            "smt2",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in manifest_rows:
            for solver_name, solver_path in available:
                cmd = solver_command(solver_name, solver_path, Path(str(row["smt2"])))
                start = time.monotonic_ns()
                try:
                    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
                    exit_code = completed.returncode
                    result = parse_result(completed.stdout, completed.stderr)
                except subprocess.TimeoutExpired:
                    exit_code = 124
                    result = "TIMEOUT"
                elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
                writer.writerow(
                    {
                        "instance_id": row["instance_id"],
                        "name": row["name"],
                        "seed": row["seed"],
                        "expected": row["expected"],
                        "solver": solver_name,
                        "solver_path": solver_path,
                        "exit_code": exit_code,
                        "result": result,
                        "matches_expected": 1 if result == row["expected"] else 0,
                        "elapsed_ms": elapsed_ms,
                        "smt2_sha256": row["sha256"],
                        "smt2": row["smt2"],
                    }
                )

    manifest_path = out_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("schema=sounio.solver.external_smtlib_tiny.run.v1\n")
        f.write(f"timestamp_utc={stamp}\n")
        f.write(f"repo={repo}\n")
        f.write(f"instances={len(manifest_rows)}\n")
        f.write(f"available_solvers={','.join(name for name, _ in available) if available else 'none'}\n")
        f.write(f"smtlib_manifest={instance_manifest}\n")
        f.write(f"solver_availability={availability_path}\n")
        f.write(f"external_results={result_path}\n")
        f.write("note=Tiny SMT-LIB QF_LIA external-baseline scaffold; not SMT-COMP-scale evidence or public novelty evidence.\n")

    print(manifest_path.read_text(encoding="utf-8"), end="")
    if not available:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
