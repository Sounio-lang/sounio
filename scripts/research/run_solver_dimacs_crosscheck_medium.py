#!/usr/bin/env python3
"""Crosscheck the medium Sounio SMT ablation against external DIMACS solvers."""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


LCG_A = 1103515245
LCG_C = 12345
LCG_M = 2147483648


@dataclass(frozen=True)
class Instance:
    instance_id: int
    name: str
    seed: int
    n_vars: int
    clauses: tuple[tuple[int, ...], ...]


def lcg_next(state: int) -> int:
    return (state * LCG_A + LCG_C) % LCG_M


def dimacs_lit(sounio_lit: int) -> int:
    var_idx = sounio_lit // 2
    is_neg = sounio_lit % 2
    dimacs_var = var_idx + 1
    return -dimacs_var if is_neg else dimacs_var


def mixed_2_5(instance_id: int, name: str, seed: int, n_vars: int, binary_count: int, wide_count: int) -> Instance:
    state = seed
    clauses: list[tuple[int, ...]] = []
    for _ in range(binary_count):
        lits: list[int] = []
        for _ in range(2):
            raw = lcg_next(state)
            state = raw
            lits.append(dimacs_lit((raw % n_vars) * 2 + ((raw // 65536) % 2)))
        clauses.append(tuple(lits))
    for _ in range(wide_count):
        lits = []
        for _ in range(5):
            raw = lcg_next(state)
            state = raw
            lits.append(dimacs_lit((raw % n_vars) * 2 + ((raw // 65536) % 2)))
        clauses.append(tuple(lits))
    return Instance(instance_id, name, seed, n_vars, tuple(clauses))


def php(instance_id: int, name: str, pigeons: int, holes: int) -> Instance:
    n_vars = pigeons * holes
    clauses: list[tuple[int, ...]] = []
    for i in range(pigeons):
        clauses.append(tuple((i * holes + j) + 1 for j in range(holes)))
    for j in range(holes):
        for i1 in range(pigeons):
            for i2 in range(i1 + 1, pigeons):
                clauses.append((-(i1 * holes + j + 1), -(i2 * holes + j + 1)))
    return Instance(instance_id, name, 0, n_vars, tuple(clauses))


def medium_instances() -> list[Instance]:
    instances: list[Instance] = []
    for seed in range(5000, 5012):
        instances.append(mixed_2_5(1, "mixed_20v_60c_random", seed, 20, 40, 20))
    for seed in range(7000, 7012):
        instances.append(mixed_2_5(3, "mixed_28v_100c_random", seed, 28, 70, 30))
    instances.append(php(2, "php5_4_unsat", 5, 4))
    instances.append(php(4, "php6_5_unsat", 6, 5))
    return instances


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_dimacs(inst: Instance, path: Path) -> str:
    with path.open("w", encoding="utf-8") as f:
        f.write("c schema=sounio.solver.external_dimacs_medium.v1\n")
        f.write(f"c instance_id={inst.instance_id}\n")
        f.write(f"c name={inst.name}\n")
        f.write(f"c seed={inst.seed}\n")
        f.write(f"p cnf {inst.n_vars} {len(inst.clauses)}\n")
        for clause in inst.clauses:
            f.write(" ".join(str(lit) for lit in clause))
            f.write(" 0\n")
    return sha256(path)


def parse_solver_result(stdout: str, stderr: str) -> str:
    text = (stdout + "\n" + stderr).upper()
    if "UNSATISFIABLE" in text or "\nUNSAT\n" in text or text.strip() == "UNSAT":
        return "UNSAT"
    if "SATISFIABLE" in text or "\nSAT\n" in text or text.strip() == "SAT":
        return "SAT"
    if "UNKNOWN" in text:
        return "UNKNOWN"
    return "UNPARSED"


def solver_version(path: str | None) -> str:
    if not path:
        return ""
    for probe in ([path, "--version"], [path, "-version"]):
        try:
            completed = subprocess.run(probe, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        lines = (completed.stdout + completed.stderr).strip().splitlines()
        if lines:
            return lines[0][:200]
    return "version_unavailable"


def read_manifest(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
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


def run_external(instances: list[Instance], out_dir: Path) -> tuple[Path, Path, Path, str]:
    cnf_dir = out_dir / "cnf"
    cnf_dir.mkdir(parents=True, exist_ok=True)
    solver_names = ["z3", "kissat", "cadical", "minisat", "glucose", "cryptominisat5"]
    solvers = [(name, shutil.which(name)) for name in solver_names]
    available = [(name, path) for name, path in solvers if path]

    dimacs_rows: list[dict[str, object]] = []
    for inst in instances:
        cnf_path = cnf_dir / f"{inst.instance_id}_{inst.name}_{inst.seed}.cnf"
        digest = write_dimacs(inst, cnf_path)
        dimacs_rows.append(
            {
                "instance_id": inst.instance_id,
                "name": inst.name,
                "seed": inst.seed,
                "n_vars": inst.n_vars,
                "n_clauses": len(inst.clauses),
                "sha256": digest,
                "cnf": str(cnf_path),
            }
        )

    availability_path = out_dir / "solver_availability.csv"
    with availability_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["solver", "path", "available", "version"])
        writer.writeheader()
        for name, path in solvers:
            writer.writerow({"solver": name, "path": path or "", "available": 1 if path else 0, "version": solver_version(path)})

    dimacs_path = out_dir / "dimacs_manifest.csv"
    with dimacs_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dimacs_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dimacs_rows)

    result_path = out_dir / "external_results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "instance_id",
            "name",
            "seed",
            "n_vars",
            "n_clauses",
            "solver",
            "solver_path",
            "exit_code",
            "result",
            "elapsed_ms",
            "cnf_sha256",
            "cnf",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in dimacs_rows:
            for solver_name, solver_path in available:
                started = time.monotonic_ns()
                try:
                    completed = subprocess.run([solver_path, str(row["cnf"])], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
                    exit_code = completed.returncode
                    result = parse_solver_result(completed.stdout, completed.stderr)
                except subprocess.TimeoutExpired:
                    exit_code = 124
                    result = "TIMEOUT"
                elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
                writer.writerow(
                    {
                        "instance_id": row["instance_id"],
                        "name": row["name"],
                        "seed": row["seed"],
                        "n_vars": row["n_vars"],
                        "n_clauses": row["n_clauses"],
                        "solver": solver_name,
                        "solver_path": solver_path,
                        "exit_code": exit_code,
                        "result": result,
                        "elapsed_ms": elapsed_ms,
                        "cnf_sha256": row["sha256"],
                        "cnf": row["cnf"],
                    }
                )
    return dimacs_path, availability_path, result_path, ",".join(name for name, _ in available) if available else "none"


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(os.environ.get("SOLVER_DIMACS_CROSSCHECK_MEDIUM_OUT_DIR", f"/tmp/sounio-solver-dimacs-crosscheck-medium-{stamp}"))
    ablation_dir = out_dir / "sounio_ablation_medium"
    external_dir = out_dir / "external_dimacs_medium"
    out_dir.mkdir(parents=True, exist_ok=True)
    external_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SOLVER_ABLATION_MEDIUM_OUT_DIR"] = str(ablation_dir)
    ablation_run = subprocess.run(["scripts/research/run_solver_ablation_medium.py"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    (out_dir / "ablation_stdout.txt").write_text(ablation_run.stdout + ablation_run.stderr, encoding="utf-8")
    if ablation_run.returncode != 0:
        print(ablation_run.stdout + ablation_run.stderr, end="")
        return ablation_run.returncode

    ablation_manifest = read_manifest(ablation_dir / "manifest.txt")
    timed_csv = Path(ablation_manifest["timed_csv"])
    instances = medium_instances()
    dimacs_path, availability_path, external_csv, available_solvers = run_external(instances, external_dir)

    external_by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    with external_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            external_by_key.setdefault((row["instance_id"], row["seed"]), []).append(row)

    result_path = out_dir / "crosscheck_results.csv"
    summary = Counter()
    rows_written = 0
    with timed_csv.open(newline="", encoding="utf-8") as fin, result_path.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin)
        fields = [
            "instance_id",
            "seed",
            "rep",
            "score_mode",
            "phase_mode",
            "restart_budget",
            "sounio_result_code",
            "sounio_result",
            "sounio_elapsed_ms_since_prev_row",
            "external_solver",
            "external_result",
            "matches_external",
            "external_elapsed_ms",
            "cnf_sha256",
        ]
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            external_rows = external_by_key.get((row["instance_id"], row["seed"]), [])
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
                        "sounio_elapsed_ms_since_prev_row": row["elapsed_ms_since_prev_row"],
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

    all_matched = rows_written > 0 and summary["matches_0"] == 0
    manifest_path = out_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("schema=sounio.solver.dimacs_crosscheck_medium.run.v1\n")
        f.write(f"timestamp_utc={stamp}\n")
        f.write(f"repo={repo}\n")
        f.write(f"sounio_ablation_manifest={ablation_dir / 'manifest.txt'}\n")
        f.write(f"dimacs_manifest={dimacs_path}\n")
        f.write(f"solver_availability={availability_path}\n")
        f.write(f"external_results={external_csv}\n")
        f.write(f"crosscheck_results={result_path}\n")
        f.write(f"crosscheck_summary={summary_path}\n")
        f.write(f"available_solvers={available_solvers}\n")
        f.write(f"instances={len(instances)}\n")
        f.write(f"rows={rows_written}\n")
        f.write(f"mismatches={summary['matches_0']}\n")
        f.write(f"all_matched={1 if all_matched else 0}\n")
        f.write("note=Medium Sounio-vs-external DIMACS result crosscheck over generated fixtures. More substantial than tiny smoke, still not SATComp-scale performance evidence.\n")

    print(manifest_path.read_text(encoding="utf-8"), end="")
    return 0 if all_matched else 1


if __name__ == "__main__":
    sys.exit(main())
