#!/usr/bin/env python3
"""Crosscheck Sounio tiny SMT ablation results against external DIMACS solvers.

This script joins:

- `scripts/research/run_solver_ablation_tiny.sh`
- `scripts/research/run_solver_external_dimacs_tiny.py`

on `(instance_id, seed)` and compares the Sounio SAT/UNSAT result for every
configuration row against each available external DIMACS solver result. It is a
tiny smoke crosscheck, not a SAT competition benchmark.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
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


def sounio_result_label(value: str) -> str:
    if value == "0":
        return "UNSAT"
    if value == "1":
        return "SAT"
    if value == "-1":
        return "LIA_ERROR"
    if value == "-2":
        return "BUDGET_EXCEEDED"
    return f"OTHER_{value}"


def run_checked(cmd: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, check=True)


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(os.environ.get("SOLVER_DIMACS_CROSSCHECK_OUT_DIR", f"/tmp/sounio-solver-dimacs-crosscheck-tiny-{stamp}"))
    ablation_dir = out_dir / "sounio_ablation"
    external_dir = out_dir / "external_dimacs"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SOLVER_ABLATION_OUT_DIR"] = str(ablation_dir)
    env["SOLVER_EXTERNAL_DIMACS_OUT_DIR"] = str(external_dir)

    ablation_run = run_checked(["bash", "scripts/research/run_solver_ablation_tiny.sh"], env)
    external_run = run_checked(["scripts/research/run_solver_external_dimacs_tiny.py"], env)

    ablation_manifest = read_manifest(ablation_dir / "manifest.txt")
    external_manifest = read_manifest(external_dir / "manifest.txt")
    ablation_csv = Path(ablation_manifest["csv"])
    external_csv = Path(external_manifest["external_results"])

    external_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    with external_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["instance_id"], row["seed"])
            external_by_key.setdefault(key, []).append(row)

    result_path = out_dir / "crosscheck_results.csv"
    summary = Counter()
    rows_written = 0
    with ablation_csv.open(newline="", encoding="utf-8") as fin, result_path.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(row for row in fin if not row.startswith("#"))
        fieldnames = [
            "instance_id",
            "seed",
            "rep",
            "score_mode",
            "phase_mode",
            "restart_budget",
            "sounio_result_code",
            "sounio_result",
            "external_solver",
            "external_result",
            "matches_external",
            "external_elapsed_ms",
            "cnf_sha256",
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            key = (row["instance_id"], row["seed"])
            external_rows = external_by_key.get(key, [])
            if not external_rows:
                summary["missing_external"] += 1
                continue
            sounio_label = sounio_result_label(row["result"])
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
                        "rep": row["rep"],
                        "score_mode": row["score_mode"],
                        "phase_mode": row["phase_mode"],
                        "restart_budget": row["restart_budget"],
                        "sounio_result_code": row["result"],
                        "sounio_result": sounio_label,
                        "external_solver": external["solver"],
                        "external_result": external["result"],
                        "matches_external": matches,
                        "external_elapsed_ms": external["elapsed_ms"],
                        "cnf_sha256": external["cnf_sha256"],
                    }
                )
                rows_written += 1

    summary_path = out_dir / "crosscheck_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key in sorted(summary):
            writer.writerow({"metric": key, "value": summary[key]})

    manifest_path = out_dir / "manifest.txt"
    all_matched = rows_written > 0 and summary["matches_0"] == 0
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("schema=sounio.solver.dimacs_crosscheck_tiny.run.v1\n")
        f.write(f"timestamp_utc={stamp}\n")
        f.write(f"repo={repo}\n")
        f.write(f"sounio_ablation_manifest={ablation_dir / 'manifest.txt'}\n")
        f.write(f"external_dimacs_manifest={external_dir / 'manifest.txt'}\n")
        f.write(f"crosscheck_results={result_path}\n")
        f.write(f"crosscheck_summary={summary_path}\n")
        f.write(f"available_solvers={external_manifest.get('available_solvers', 'unknown')}\n")
        f.write(f"rows={rows_written}\n")
        f.write(f"mismatches={summary['matches_0']}\n")
        f.write(f"all_matched={1 if all_matched else 0}\n")
        f.write("note=Tiny Sounio-vs-external DIMACS result crosscheck over generated fixtures. Not SATComp-scale performance evidence.\n")

    (out_dir / "ablation_stdout.txt").write_text(ablation_run.stdout + ablation_run.stderr, encoding="utf-8")
    (out_dir / "external_stdout.txt").write_text(external_run.stdout + external_run.stderr, encoding="utf-8")
    print(manifest_path.read_text(encoding="utf-8"), end="")
    return 0 if all_matched else 1


if __name__ == "__main__":
    sys.exit(main())
