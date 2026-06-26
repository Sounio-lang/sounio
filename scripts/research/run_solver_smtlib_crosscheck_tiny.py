#!/usr/bin/env python3
"""Crosscheck Sounio QF_LIA fixture results against external SMT-LIB solvers."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def read_manifest(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" in raw:
            key, value = raw.split("=", 1)
            data[key] = value
    return data


def result_label(code: str) -> str:
    if code == "1":
        return "SAT"
    if code == "0":
        return "UNSAT"
    if code == "-1":
        return "LIA_ERROR"
    if code == "-2":
        return "BUDGET_EXCEEDED"
    return f"OTHER_{code}"


def run_checked(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=True)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(os.environ.get("SOLVER_SMTLIB_CROSSCHECK_OUT_DIR", f"/tmp/sounio-solver-smtlib-crosscheck-tiny-{stamp}"))
    sounio_dir = out_dir / "sounio_qflia"
    external_dir = out_dir / "external_smtlib"
    sounio_dir.mkdir(parents=True, exist_ok=True)
    external_dir.mkdir(parents=True, exist_ok=True)

    souc_bin = os.environ.get("SOUC_BIN", "./bin/souc")
    source = Path("benchmarks/solver/smtlib_qflia_crosscheck_tiny.sio")
    binary = sounio_dir / "smtlib_qflia_crosscheck_tiny"
    sounio_csv = sounio_dir / "sounio_qflia_results.csv"
    compile_log = sounio_dir / "compile.log"

    compile_start = time.monotonic_ns()
    compiled = subprocess.run([souc_bin, "compile", str(source), "-o", str(binary)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    compile_ms = (time.monotonic_ns() - compile_start) // 1_000_000
    compile_log.write_text(compiled.stdout + compiled.stderr, encoding="utf-8")
    if compiled.returncode != 0:
        print(compile_log.read_text(encoding="utf-8"), end="")
        return compiled.returncode

    run_start = time.monotonic_ns()
    ran = subprocess.run([str(binary)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    sounio_elapsed_ms = (time.monotonic_ns() - run_start) // 1_000_000
    sounio_csv.write_text(ran.stdout, encoding="utf-8")
    (sounio_dir / "stderr.txt").write_text(ran.stderr, encoding="utf-8")
    if ran.returncode != 0:
        print(ran.stdout + ran.stderr, end="")
        return ran.returncode

    env = os.environ.copy()
    env["SOLVER_EXTERNAL_SMTLIB_OUT_DIR"] = str(external_dir)
    external_run = run_checked(["scripts/research/run_solver_external_smtlib_tiny.py"], env)
    external_manifest = read_manifest(external_dir / "manifest.txt")
    external_csv = Path(external_manifest["external_results"])

    external_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    with external_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["instance_id"], row["seed"])
            external_by_key.setdefault(key, []).append(row)

    result_path = out_dir / "crosscheck_results.csv"
    summary = Counter()
    rows_written = 0
    with sounio_csv.open(newline="", encoding="utf-8") as fin, result_path.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(row for row in fin if not row.startswith("#"))
        fields = [
            "instance_id",
            "seed",
            "sounio_expected",
            "sounio_result_code",
            "sounio_result",
            "external_solver",
            "external_expected",
            "external_result",
            "matches_external",
            "external_elapsed_ms",
            "smt2_sha256",
        ]
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            key = (row["instance_id"], row["seed"])
            external_rows = external_by_key.get(key, [])
            if not external_rows:
                summary["missing_external"] += 1
                continue
            sounio_label = result_label(row["sounio_result_code"])
            for external in external_rows:
                matches = 1 if sounio_label == external["result"] else 0
                summary["rows"] += 1
                summary[f"matches_{matches}"] += 1
                summary[f"solver_{external['solver']}"] += 1
                if matches == 0:
                    summary[f"mismatch_{external['solver']}"] += 1
                writer.writerow(
                    {
                        "instance_id": row["instance_id"],
                        "seed": row["seed"],
                        "sounio_expected": result_label(row["expected_code"]),
                        "sounio_result_code": row["sounio_result_code"],
                        "sounio_result": sounio_label,
                        "external_solver": external["solver"],
                        "external_expected": external["expected"],
                        "external_result": external["result"],
                        "matches_external": matches,
                        "external_elapsed_ms": external["elapsed_ms"],
                        "smt2_sha256": external["smt2_sha256"],
                    }
                )
                rows_written += 1

    summary_path = out_dir / "crosscheck_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in sorted(summary):
            writer.writerow({"metric": key, "value": summary[key]})

    all_matched = rows_written > 0 and summary["matches_0"] == 0
    manifest_path = out_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("schema=sounio.solver.smtlib_crosscheck_tiny.run.v1\n")
        f.write(f"timestamp_utc={stamp}\n")
        f.write(f"repo={repo}\n")
        f.write(f"source={source}\n")
        f.write(f"sounio_results={sounio_csv}\n")
        f.write(f"sounio_compile_log={compile_log}\n")
        f.write(f"sounio_compile_ms={compile_ms}\n")
        f.write(f"sounio_elapsed_ms={sounio_elapsed_ms}\n")
        f.write(f"external_smtlib_manifest={external_dir / 'manifest.txt'}\n")
        f.write(f"crosscheck_results={result_path}\n")
        f.write(f"crosscheck_summary={summary_path}\n")
        f.write(f"available_solvers={external_manifest.get('available_solvers', 'unknown')}\n")
        f.write(f"rows={rows_written}\n")
        f.write(f"mismatches={summary['matches_0']}\n")
        f.write(f"all_matched={1 if all_matched else 0}\n")
        f.write("note=Tiny Sounio-vs-external SMT-LIB QF_LIA result crosscheck. Not SMT-COMP-scale performance evidence.\n")

    (out_dir / "external_stdout.txt").write_text(external_run.stdout + external_run.stderr, encoding="utf-8")
    print(manifest_path.read_text(encoding="utf-8"), end="")
    return 0 if all_matched else 1


if __name__ == "__main__":
    sys.exit(main())
