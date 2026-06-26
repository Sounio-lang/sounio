#!/usr/bin/env python3
"""Generate and replay a tiny SAT RUP/DRAT-addition proof log.

This is a file-level certificate replay smoke test for the solver novelty lane.
It checks a narrow RUP subset over DIMACS CNF plus DRAT-style addition lines.
It is not a full DRAT/LRAT/FRAT checker and is not a replacement for an
independent verified checker.
"""

from __future__ import annotations

import csv
import hashlib
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class ReplayResult:
    name: str
    expected_accept: int
    accepted: int
    steps: int
    reason: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_dimacs(path: Path) -> tuple[int, list[list[int]]]:
    n_vars = 0
    clauses: list[list[int]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p "):
            parts = line.split()
            if len(parts) != 4 or parts[1] != "cnf":
                raise ValueError(f"{path}:{line_no}: expected 'p cnf <vars> <clauses>'")
            n_vars = int(parts[2])
            continue
        lits = [int(x) for x in line.split()]
        if not lits or lits[-1] != 0:
            raise ValueError(f"{path}:{line_no}: DIMACS clause must end in 0")
        clause = lits[:-1]
        for lit in clause:
            if lit == 0 or abs(lit) > n_vars:
                raise ValueError(f"{path}:{line_no}: literal out of range: {lit}")
        clauses.append(clause)
    if n_vars <= 0:
        raise ValueError(f"{path}: missing DIMACS header")
    return n_vars, clauses


def parse_drat_additions(path: Path, n_vars: int) -> list[list[int]]:
    additions: list[list[int]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("d "):
            raise ValueError(f"{path}:{line_no}: deletion lines are outside this tiny replay subset")
        lits = [int(x) for x in line.split()]
        if not lits or lits[-1] != 0:
            raise ValueError(f"{path}:{line_no}: proof addition must end in 0")
        clause = lits[:-1]
        for lit in clause:
            if lit == 0 or abs(lit) > n_vars:
                raise ValueError(f"{path}:{line_no}: literal out of range: {lit}")
        additions.append(clause)
    return additions


def lit_value(lit: int, assignment: dict[int, bool]) -> bool | None:
    val = assignment.get(abs(lit))
    if val is None:
        return None
    return val if lit > 0 else not val


def assign_lit_true(lit: int, assignment: dict[int, bool]) -> bool:
    var = abs(lit)
    val = lit > 0
    old = assignment.get(var)
    if old is None:
        assignment[var] = val
        return True
    return old == val


def clause_unit_or_conflict(clause: list[int], assignment: dict[int, bool]) -> tuple[str, int | None]:
    unassigned: list[int] = []
    for lit in clause:
        val = lit_value(lit, assignment)
        if val is True:
            return "satisfied", None
        if val is None:
            unassigned.append(lit)
    if not unassigned:
        return "conflict", None
    if len(unassigned) == 1:
        return "unit", unassigned[0]
    return "open", None


def is_rup(existing: list[list[int]], candidate: list[int]) -> bool:
    assignment: dict[int, bool] = {}
    for lit in candidate:
        if not assign_lit_true(-lit, assignment):
            return True

    changed = True
    while changed:
        changed = False
        for clause in existing:
            kind, unit = clause_unit_or_conflict(clause, assignment)
            if kind == "conflict":
                return True
            if kind == "unit":
                assert unit is not None
                if not assign_lit_true(unit, assignment):
                    return True
                changed = True
    return False


def replay_proof(cnf_path: Path, proof_path: Path, expected_accept: int) -> ReplayResult:
    n_vars, clauses = parse_dimacs(cnf_path)
    additions = parse_drat_additions(proof_path, n_vars)
    existing = [list(c) for c in clauses]
    for idx, candidate in enumerate(additions, start=1):
        if not is_rup(existing, candidate):
            return ReplayResult(proof_path.stem, expected_accept, 0, idx - 1, f"non_rup_step_{idx}")
        existing.append(candidate)
    return ReplayResult(proof_path.stem, expected_accept, 1, len(additions), "accepted")


def write_fixtures(out_dir: Path) -> tuple[Path, Path, Path, Path]:
    fixtures = out_dir / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)

    cnf = fixtures / "rup_xor_unsat.cnf"
    cnf.write_text(
        "\n".join(
            [
                "c Tiny XOR-style UNSAT fixture for RUP replay",
                "p cnf 2 4",
                "1 2 0",
                "-1 2 0",
                "1 -2 0",
                "-1 -2 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    good = fixtures / "rup_xor_unsat.drat"
    good.write_text(
        "\n".join(
            [
                "c derive unit 1, then derive empty by RUP",
                "1 0",
                "0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    bad_cnf = fixtures / "rup_sat_base.cnf"
    bad_cnf.write_text(
        "\n".join(
            [
                "c SAT fixture used to check that non-RUP additions are rejected",
                "p cnf 2 1",
                "1 2 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    bad = fixtures / "rup_sat_wrong.drat"
    bad.write_text(
        "\n".join(
            [
                "c 1 is not RUP for the SAT base clause (1 or 2)",
                "1 0",
                "0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return cnf, good, bad_cnf, bad


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(os.environ.get("SOLVER_SAT_RUP_REPLAY_OUT_DIR", f"/tmp/sounio-solver-sat-rup-replay-tiny-{stamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cnf, good, bad_cnf, bad = write_fixtures(out_dir)
    results = [
        replay_proof(cnf, good, 1),
        replay_proof(bad_cnf, bad, 0),
    ]

    csv_path = out_dir / "replay_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "expected_accept", "accepted", "steps", "reason"])
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    all_expected = all(result.expected_accept == result.accepted for result in results)
    manifest_path = out_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("schema=sounio.solver.sat_rup_replay_tiny.run.v1\n")
        f.write(f"timestamp_utc={stamp}\n")
        f.write(f"repo={repo}\n")
        f.write("format=drat_addition_rup_subset\n")
        f.write(f"cnf={cnf}\n")
        f.write(f"rejected_cnf={bad_cnf}\n")
        f.write(f"accepted_proof={good}\n")
        f.write(f"rejected_proof={bad}\n")
        f.write(f"replay_results={csv_path}\n")
        f.write(f"cnf_sha256={sha256(cnf)}\n")
        f.write(f"rejected_cnf_sha256={sha256(bad_cnf)}\n")
        f.write(f"accepted_proof_sha256={sha256(good)}\n")
        f.write(f"rejected_proof_sha256={sha256(bad)}\n")
        f.write(f"accepted_proof_steps={results[0].steps}\n")
        f.write(f"rejected_wrong_proof={1 if results[1].accepted == 0 else 0}\n")
        f.write(f"all_expected={1 if all_expected else 0}\n")
        f.write("note=Tiny file-level RUP replay over DIMACS plus DRAT-style additions. Not full DRAT, LRAT, FRAT, or an independent verified checker.\n")

    print(manifest_path.read_text(encoding="utf-8"), end="")
    return 0 if all_expected else 1


if __name__ == "__main__":
    sys.exit(main())
