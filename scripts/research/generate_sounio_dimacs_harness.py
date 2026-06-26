#!/usr/bin/env python3
"""Generate Sounio SMT harnesses from small DIMACS CNF files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cnf:
    n_vars: int
    clauses: tuple[tuple[int, ...], ...]


def parse_dimacs(path: Path) -> Cnf:
    n_vars = 0
    clauses: list[tuple[int, ...]] = []
    pending: list[int] = []
    with path.open("rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p "):
                parts = line.split()
                if len(parts) >= 4 and parts[1] == "cnf":
                    n_vars = int(parts[2])
                continue
            for token in line.split():
                lit = int(token)
                if lit == 0:
                    if pending:
                        clauses.append(tuple(pending))
                        pending = []
                    continue
                pending.append(lit)
        if pending:
            clauses.append(tuple(pending))
    return Cnf(n_vars=n_vars, clauses=tuple(clauses))


def sounio_lit(dimacs_lit: int) -> int:
    var = abs(dimacs_lit) - 1
    return var * 2 + (1 if dimacs_lit < 0 else 0)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(cnf_path: Path, out_path: Path, expected: str, instance_id: str) -> None:
    cnf = parse_dimacs(cnf_path)
    if cnf.n_vars > 1024:
        raise ValueError(f"{cnf_path}: n_vars {cnf.n_vars} exceeds Sounio SMT limit")
    if len(cnf.clauses) > 4096:
        raise ValueError(f"{cnf_path}: clauses {len(cnf.clauses)} exceeds Sounio SMT limit")
    if any(len(clause) > 8 for clause in cnf.clauses):
        raise ValueError(f"{cnf_path}: clause width exceeds smt_add_clause width 8")
    expected_code = {"SAT": 1, "UNSAT": 0}.get(expected.upper(), -9)
    lines = [
        "//@ run-pass",
        f"//@ generated_from: {cnf_path}",
        f"//@ instance_id: {instance_id}",
        "use theorem::smt::*",
        "",
    ]
    block_size = 96
    for block_start in range(0, len(cnf.clauses), block_size):
        block_idx = block_start // block_size
        lines.append(f"fn add_block_{block_idx}(ctx: &!SmtContext) with Mut, Div {{")
        for idx, clause in enumerate(cnf.clauses[block_start : block_start + block_size], start=block_start):
            lines.append(f"    var c{idx}: [i32; 8] = [0 as i32; 8]")
            for lit_idx, lit in enumerate(clause):
                lines.append(f"    c{idx}[{lit_idx}] = {sounio_lit(lit)}")
            lines.append(f"    smt_add_clause(ctx, &c{idx}, {len(clause)})")
        lines.extend(["}", ""])
    lines.extend(
        [
            "fn main() -> i32 with IO, Mut, Div {",
            "    var ctx = smt_new()",
        ]
    )
    for block_start in range(0, len(cnf.clauses), block_size):
        block_idx = block_start // block_size
        lines.append(f"    add_block_{block_idx}(&! ctx)")
    lines.extend(
        [
            "    let result = smt_solve(&! ctx)",
            f"    println(\"SOUNIO_DIMACS_RESULT {instance_id}\")",
            "    println(result)",
        ]
    )
    if expected_code >= 0:
        lines.extend(
            [
                f"    if result != {expected_code} {{",
                "        return 10 + result",
                "    }",
            ]
        )
    lines.extend(["    return 0", "}", ""])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="corpus_manifest.csv from fetcher")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    with args.manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["domain"] != "sat" or row["status"] != "OK":
                continue
            src = Path(row["local_path"])
            out = args.out_dir / f"{row['instance_id']}.sio"
            try:
                generate(src, out, row["expected"], row["instance_id"])
                status = "OK"
                err = ""
            except Exception as exc:
                status = "ERROR"
                err = str(exc)
            rows.append(
                {
                    "instance_id": row["instance_id"],
                    "expected": row["expected"],
                    "cnf_sha256": row["sha256"],
                    "sio_path": str(out),
                    "sio_sha256": sha256_file(out) if out.exists() else "",
                    "status": status,
                    "error": err,
                }
            )

    result = args.out_dir / "sounio_dimacs_harness_manifest.csv"
    fieldnames = ["instance_id", "expected", "cnf_sha256", "sio_path", "sio_sha256", "status", "error"]
    with result.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(result)
    return 0 if rows and all(row["status"] == "OK" for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
