#!/usr/bin/env python3
"""Tiny DIMACS external-baseline scaffold for Sounio SMT/SAT readiness.

This is intentionally small: it proves that we can generate reproducible
DIMACS slices from the same fixture family as the internal ablation, run real
external solvers when installed, and record absence explicitly when not.
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
    if is_neg:
        return -dimacs_var
    return dimacs_var


def mixed_2_5(seed: int) -> Instance:
    n_vars = 20
    state = seed
    clauses: list[tuple[int, ...]] = []
    for _ in range(40):
        lits: list[int] = []
        for _ in range(2):
            raw = lcg_next(state)
            state = raw
            sounio_lit = (raw % n_vars) * 2 + ((raw // 65536) % 2)
            lits.append(dimacs_lit(sounio_lit))
        clauses.append(tuple(lits))
    for _ in range(20):
        lits = []
        for _ in range(5):
            raw = lcg_next(state)
            state = raw
            sounio_lit = (raw % n_vars) * 2 + ((raw // 65536) % 2)
            lits.append(dimacs_lit(sounio_lit))
        clauses.append(tuple(lits))
    return Instance(1, "mixed_2_5_random", seed, n_vars, tuple(clauses))


def php_5_4() -> Instance:
    n_vars = 20
    clauses: list[tuple[int, ...]] = []
    for i in range(5):
        clauses.append(tuple((i * 4 + j) + 1 for j in range(4)))
    for j in range(4):
        for i1 in range(5):
            for i2 in range(i1 + 1, 5):
                clauses.append((-(i1 * 4 + j + 1), -(i2 * 4 + j + 1)))
    return Instance(2, "php5_4_unsat", 0, n_vars, tuple(clauses))


def write_dimacs(inst: Instance, path: Path) -> str:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"c schema=sounio.solver.external_dimacs_tiny.v1\n")
        f.write(f"c instance_id={inst.instance_id}\n")
        f.write(f"c name={inst.name}\n")
        f.write(f"c seed={inst.seed}\n")
        f.write(f"p cnf {inst.n_vars} {len(inst.clauses)}\n")
        for clause in inst.clauses:
            f.write(" ".join(str(lit) for lit in clause))
            f.write(" 0\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_solver_result(stdout: str, stderr: str) -> str:
    text = (stdout + "\n" + stderr).upper()
    if "UNSATISFIABLE" in text or "\nUNSAT\n" in text or text.strip() == "UNSAT":
        return "UNSAT"
    if "SATISFIABLE" in text or "\nSAT\n" in text or text.strip() == "SAT":
        return "SAT"
    if "UNKNOWN" in text:
        return "UNKNOWN"
    return "UNPARSED"


def solver_command(name: str, path: str, cnf: Path) -> list[str]:
    if name == "z3":
        return [path, str(cnf)]
    return [path, str(cnf)]


def solver_version(name: str, path: str | None) -> str:
    if not path:
        return ""
    probes = [[path, "--version"], [path, "-version"]]
    for probe in probes:
        try:
            completed = subprocess.run(probe, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        text = (completed.stdout + completed.stderr).strip().splitlines()
        if text:
            return text[0][:200]
    return "version_unavailable"


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(os.environ.get("SOLVER_EXTERNAL_DIMACS_OUT_DIR", f"/tmp/sounio-solver-external-dimacs-tiny-{stamp}"))
    cnf_dir = out_dir / "cnf"
    cnf_dir.mkdir(parents=True, exist_ok=True)

    solver_names = ["z3", "kissat", "cadical", "minisat", "glucose", "cryptominisat5"]
    solvers = [(name, shutil.which(name)) for name in solver_names]
    available = [(name, path) for name, path in solvers if path]

    instances = [mixed_2_5(seed) for seed in range(5000, 5005)]
    instances.append(php_5_4())

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
            writer.writerow(
                {
                    "solver": name,
                    "path": path or "",
                    "available": 1 if path else 0,
                    "version": solver_version(name, path),
                }
            )

    dimacs_path = out_dir / "dimacs_manifest.csv"
    with dimacs_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dimacs_rows[0].keys()))
        writer.writeheader()
        writer.writerows(dimacs_rows)

    result_path = out_dir / "external_results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
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
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in dimacs_rows:
            for solver_name, solver_path in available:
                cmd = solver_command(solver_name, solver_path, Path(str(row["cnf"])))
                start = time.monotonic_ns()
                try:
                    completed = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
                    exit_code = completed.returncode
                    result = parse_solver_result(completed.stdout, completed.stderr)
                except subprocess.TimeoutExpired as exc:
                    exit_code = 124
                    result = "TIMEOUT"
                elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
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

    manifest_path = out_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("schema=sounio.solver.external_dimacs_tiny.run.v1\n")
        f.write(f"timestamp_utc={stamp}\n")
        f.write(f"repo={repo}\n")
        f.write(f"instances={len(instances)}\n")
        f.write(f"available_solvers={','.join(name for name, _ in available) if available else 'none'}\n")
        f.write(f"dimacs_manifest={dimacs_path}\n")
        f.write(f"solver_availability={availability_path}\n")
        f.write(f"external_results={result_path}\n")
        f.write("note=Tiny DIMACS external-baseline scaffold; not SATComp-scale evidence or public novelty evidence.\n")

    print(manifest_path.read_text(encoding="utf-8"), end="")
    if not available:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
