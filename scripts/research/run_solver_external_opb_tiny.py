#!/usr/bin/env python3
"""Tiny OPB external-baseline scaffold for solver novelty readiness.

The current workspace usually lacks OPB/PB solvers. This script still generates
deterministic OPB fixtures and records solver availability so absence is an
auditable blocker rather than an implicit gap.
"""

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


@dataclass(frozen=True)
class OpbInstance:
    instance_id: int
    name: str
    expected: str
    text: str


def solver_version(path: str | None) -> str:
    if not path:
        return ""
    for args in ([path, "--version"], [path, "-version"], [path, "-h"]):
        try:
            completed = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        lines = (completed.stdout + completed.stderr).strip().splitlines()
        if lines:
            return lines[0][:200]
    return "version_unavailable"


def parse_result(stdout: str, stderr: str) -> str:
    text = (stdout + "\n" + stderr).upper()
    if "UNSAT" in text or "INFEASIBLE" in text:
        return "UNSAT"
    if "SAT" in text or "OPTIMUM FOUND" in text or "FEASIBLE" in text:
        return "SAT"
    if "UNKNOWN" in text:
        return "UNKNOWN"
    return "UNPARSED"


def instances() -> list[OpbInstance]:
    return [
        OpbInstance(
            1,
            "pb_atleast_one_sat",
            "SAT",
            "\n".join(
                [
                    "* #variable= 3 #constraint= 1",
                    "1 x1 1 x2 1 x3 >= 1 ;",
                    "",
                ]
            ),
        ),
        OpbInstance(
            2,
            "pb_contradictory_bound_unsat",
            "UNSAT",
            "\n".join(
                [
                    "* #variable= 2 #constraint= 2",
                    "1 x1 1 x2 >= 2 ;",
                    "-1 x1 -1 x2 >= -1 ;",
                    "",
                ]
            ),
        ),
        OpbInstance(
            3,
            "pb_cardinality_window_sat",
            "SAT",
            "\n".join(
                [
                    "* #variable= 4 #constraint= 2",
                    "1 x1 1 x2 1 x3 1 x4 >= 2 ;",
                    "-1 x1 -1 x2 -1 x3 -1 x4 >= -3 ;",
                    "",
                ]
            ),
        ),
        OpbInstance(
            4,
            "pb_cardinality_window_unsat",
            "UNSAT",
            "\n".join(
                [
                    "* #variable= 4 #constraint= 2",
                    "1 x1 1 x2 1 x3 1 x4 >= 4 ;",
                    "-1 x1 -1 x2 -1 x3 -1 x4 >= -3 ;",
                    "",
                ]
            ),
        ),
    ]


def write_opb(inst: OpbInstance, path: Path) -> str:
    path.write_text(inst.text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def solver_command(name: str, path: str, opb: Path) -> list[str]:
    if name == "scip":
        return [path, "-c", f"read {opb}", "-c", "optimize", "-c", "quit"]
    return [path, str(opb)]


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(os.environ.get("SOLVER_EXTERNAL_OPB_OUT_DIR", f"/tmp/sounio-solver-external-opb-tiny-{stamp}"))
    opb_dir = out_dir / "opb"
    opb_dir.mkdir(parents=True, exist_ok=True)

    solver_names = [
        "roundingsat",
        "open-wbo",
        "open-wbo_static",
        "scip",
        "sat4j-pb",
        "minisat+",
        "minisatp",
        "pbsolver",
    ]
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

    rows: list[dict[str, object]] = []
    for inst in instances():
        opb_path = opb_dir / f"{inst.instance_id}_{inst.name}.opb"
        digest = write_opb(inst, opb_path)
        rows.append(
            {
                "instance_id": inst.instance_id,
                "name": inst.name,
                "expected": inst.expected,
                "sha256": digest,
                "opb": str(opb_path),
            }
        )

    instance_manifest = out_dir / "opb_manifest.csv"
    with instance_manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result_path = out_dir / "external_results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "instance_id",
            "name",
            "expected",
            "solver",
            "solver_path",
            "exit_code",
            "result",
            "matches_expected",
            "elapsed_ms",
            "opb_sha256",
            "opb",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for solver_name, solver_path in available:
                start = time.monotonic_ns()
                try:
                    completed = subprocess.run(
                        solver_command(solver_name, solver_path, Path(str(row["opb"]))),
                        text=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        timeout=15,
                    )
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
                        "expected": row["expected"],
                        "solver": solver_name,
                        "solver_path": solver_path,
                        "exit_code": exit_code,
                        "result": result,
                        "matches_expected": 1 if result == row["expected"] else 0,
                        "elapsed_ms": elapsed_ms,
                        "opb_sha256": row["sha256"],
                        "opb": row["opb"],
                    }
                )

    manifest_path = out_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("schema=sounio.solver.external_opb_tiny.run.v1\n")
        f.write(f"timestamp_utc={stamp}\n")
        f.write(f"repo={repo}\n")
        f.write(f"instances={len(rows)}\n")
        f.write(f"available_solvers={','.join(name for name, _ in available) if available else 'none'}\n")
        f.write(f"opb_manifest={instance_manifest}\n")
        f.write(f"solver_availability={availability_path}\n")
        f.write(f"external_results={result_path}\n")
        f.write("note=Tiny OPB scaffold. If available_solvers=none, this is blocker evidence, not solver-baseline evidence.\n")

    print(manifest_path.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
