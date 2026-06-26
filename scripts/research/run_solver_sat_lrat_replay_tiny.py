#!/usr/bin/env python3
"""Generate and replay a tiny LRAT-style proof log.

This is a file-level certificate replay smoke test for the solver novelty lane.
It checks a narrow LRAT/RUP subset over DIMACS CNF plus text proof lines:

    <new-id> <lemma-lits...> 0 <hint-ids...> 0
    d <deleted-ids...> 0

Initial DIMACS clauses receive IDs 1..n in file order. Addition lines must use
fresh IDs, active hints, and RUP over the hinted clauses. This is intentionally
small: not full LRAT, not FRAT, and not an independent verified checker.
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
class Clause:
    lits: list[int]
    active: bool = True


@dataclass
class ReplayResult:
    name: str
    expected_accept: int
    accepted: int
    additions: int
    deletions: int
    reason: str


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_dimacs(path: Path) -> tuple[int, dict[int, Clause]]:
    n_vars = 0
    clauses: dict[int, Clause] = {}
    next_id = 1
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
        clauses[next_id] = Clause(clause)
        next_id += 1
    if n_vars <= 0:
        raise ValueError(f"{path}: missing DIMACS header")
    return n_vars, clauses


def split_lrat_addition(tokens: list[int]) -> tuple[list[int], list[int]]:
    try:
        sep = tokens.index(0)
    except ValueError as exc:
        raise ValueError("LRAT addition line must contain lemma terminator 0") from exc
    lemma = tokens[:sep]
    rest = tokens[sep + 1 :]
    if not rest or rest[-1] != 0:
        raise ValueError("LRAT addition line must end hints with 0")
    hints = rest[:-1]
    return lemma, hints


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


def replay_hint_clause(clause: list[int], assignment: dict[int, bool]) -> str:
    unassigned: list[int] = []
    for lit in clause:
        val = lit_value(lit, assignment)
        if val is True:
            return "satisfied"
        if val is None:
            unassigned.append(lit)
    if not unassigned:
        return "conflict"
    if len(unassigned) == 1:
        if not assign_lit_true(unassigned[0], assignment):
            return "conflict"
        return "unit"
    return "open"


def is_lrat_rup(clauses: dict[int, Clause], lemma: list[int], hints: list[int]) -> tuple[bool, str]:
    assignment: dict[int, bool] = {}
    for lit in lemma:
        if not assign_lit_true(-lit, assignment):
            return True, "tautological_lemma"

    for hint_id in hints:
        clause = clauses.get(hint_id)
        if clause is None:
            return False, f"missing_hint_{hint_id}"
        if not clause.active:
            return False, f"deleted_hint_{hint_id}"
        status = replay_hint_clause(clause.lits, assignment)
        if status == "conflict":
            return True, "rup_conflict"
    return False, "no_conflict_from_hints"


def replay_lrat(cnf_path: Path, proof_path: Path, expected_accept: int) -> ReplayResult:
    n_vars, clauses = parse_dimacs(cnf_path)
    additions = 0
    deletions = 0

    for line_no, raw in enumerate(proof_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("d "):
            ids = [int(x) for x in line.split()[1:]]
            if not ids or ids[-1] != 0:
                return ReplayResult(proof_path.stem, expected_accept, 0, additions, deletions, f"bad_delete_line_{line_no}")
            for clause_id in ids[:-1]:
                clause = clauses.get(clause_id)
                if clause is None:
                    return ReplayResult(proof_path.stem, expected_accept, 0, additions, deletions, f"delete_missing_{clause_id}")
                clause.active = False
                deletions += 1
            continue

        parts = [int(x) for x in line.split()]
        if len(parts) < 3:
            return ReplayResult(proof_path.stem, expected_accept, 0, additions, deletions, f"short_add_line_{line_no}")
        new_id = parts[0]
        if new_id <= 0:
            return ReplayResult(proof_path.stem, expected_accept, 0, additions, deletions, f"bad_id_{line_no}")
        if new_id in clauses:
            return ReplayResult(proof_path.stem, expected_accept, 0, additions, deletions, f"reused_id_{new_id}")
        try:
            lemma, hints = split_lrat_addition(parts[1:])
        except ValueError as exc:
            return ReplayResult(proof_path.stem, expected_accept, 0, additions, deletions, f"bad_add_line_{line_no}:{exc}")
        for lit in lemma:
            if lit == 0 or abs(lit) > n_vars:
                return ReplayResult(proof_path.stem, expected_accept, 0, additions, deletions, f"literal_out_of_range_{line_no}")
        ok, reason = is_lrat_rup(clauses, lemma, hints)
        if not ok:
            return ReplayResult(proof_path.stem, expected_accept, 0, additions, deletions, f"non_lrat_rup_step_{line_no}:{reason}")
        clauses[new_id] = Clause(lemma)
        additions += 1

    return ReplayResult(proof_path.stem, expected_accept, 1, additions, deletions, "accepted")


def write_fixtures(out_dir: Path) -> tuple[Path, dict[str, Path]]:
    fixtures = out_dir / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)

    cnf = fixtures / "lrat_xor_unsat.cnf"
    cnf.write_text(
        "\n".join(
            [
                "c Tiny XOR-style UNSAT fixture for LRAT/RUP replay",
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

    proofs: dict[str, Path] = {}
    proofs["lrat_xor_unsat"] = fixtures / "lrat_xor_unsat.lrat"
    proofs["lrat_xor_unsat"].write_text(
        "\n".join(
            [
                "c id 5 derives unit 1 using hints 1 and 3; id 6 derives empty using hints 5,2,4",
                "5 1 0 1 3 0",
                "6 0 5 2 4 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    proofs["lrat_bad_hint"] = fixtures / "lrat_bad_hint.lrat"
    proofs["lrat_bad_hint"].write_text(
        "\n".join(
            [
                "c hints 2 and 4 are both satisfied under not-1, so no conflict follows",
                "5 1 0 2 4 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    proofs["lrat_reused_id"] = fixtures / "lrat_reused_id.lrat"
    proofs["lrat_reused_id"].write_text("1 1 0 1 3 0\n", encoding="utf-8")

    proofs["lrat_deleted_hint"] = fixtures / "lrat_deleted_hint.lrat"
    proofs["lrat_deleted_hint"].write_text(
        "\n".join(
            [
                "d 3 0",
                "5 1 0 1 3 0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    return cnf, proofs


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(os.environ.get("SOLVER_SAT_LRAT_REPLAY_OUT_DIR", f"/tmp/sounio-solver-sat-lrat-replay-tiny-{stamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    cnf, proofs = write_fixtures(out_dir)
    expected = {
        "lrat_xor_unsat": 1,
        "lrat_bad_hint": 0,
        "lrat_reused_id": 0,
        "lrat_deleted_hint": 0,
    }
    results = [replay_lrat(cnf, proofs[name], expected[name]) for name in sorted(proofs)]

    csv_path = out_dir / "replay_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "expected_accept", "accepted", "additions", "deletions", "reason"])
        writer.writeheader()
        for result in results:
            writer.writerow(result.__dict__)

    all_expected = all(result.expected_accept == result.accepted for result in results)
    manifest_path = out_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("schema=sounio.solver.sat_lrat_replay_tiny.run.v1\n")
        f.write(f"timestamp_utc={stamp}\n")
        f.write(f"repo={repo}\n")
        f.write("format=lrat_rup_hint_subset\n")
        f.write(f"cnf={cnf}\n")
        f.write(f"cnf_sha256={sha256(cnf)}\n")
        f.write(f"accepted_proof={proofs['lrat_xor_unsat']}\n")
        f.write(f"accepted_proof_sha256={sha256(proofs['lrat_xor_unsat'])}\n")
        f.write(f"replay_results={csv_path}\n")
        f.write(f"accepted_proof_additions={next(r.additions for r in results if r.name == 'lrat_xor_unsat')}\n")
        f.write(f"negative_controls={sum(1 for r in results if r.expected_accept == 0)}\n")
        f.write(f"all_expected={1 if all_expected else 0}\n")
        f.write("note=Tiny LRAT/RUP hint replay over DIMACS plus text LRAT-style additions/deletions. Not full LRAT/FRAT or an independent verified checker.\n")

    print(manifest_path.read_text(encoding="utf-8"), end="")
    return 0 if all_expected else 1


if __name__ == "__main__":
    sys.exit(main())
