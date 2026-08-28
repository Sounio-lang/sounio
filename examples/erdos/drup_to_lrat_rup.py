#!/usr/bin/env python3
"""Convert a RUP-addition DRUP/DRAT proof into contiguous LRAT hints.

This is deliberately **untrusted tooling**. It performs unit propagation to
find LRAT hint lists for each added clause, then the generated LRAT must still
be checked by Lean's verified LRAT checker via `SounioSatReflect`.

Scope:
  * DIMACS CNF input.
  * ASCII DRUP/DRAT-style input where every addition is RUP. Arbitrary RAT
    additions are intentionally unsupported.
  * Deletion records are rejected. The first producer target is the streamed
    K6/no-5 smoke proof, whose certificate has no deletions. Larger proofs
    should use `drat-trim -L` or a deletion-aware converter.
  * The scan is conservative and untrusted: it can fail on input outside this
    narrow RUP-addition subset, and the final arbiter remains the Lean LRAT
    checker.

If any added clause is not RUP under this conservative scan, the tool fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


Clause = tuple[int, ...]


def parse_dimacs(path: Path) -> tuple[int, list[Clause]]:
    nvars = 0
    saw_header = False
    clauses: list[Clause] = []
    pending: list[int] = []
    with path.open("r", encoding="ascii") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            toks = line.split()
            if toks[0] == "p":
                if len(toks) < 4 or toks[1] != "cnf":
                    raise ValueError(f"unsupported DIMACS header in {path}: {line}")
                nvars = int(toks[2])
                saw_header = True
                continue
            for tok in toks:
                try:
                    lit = int(tok)
                except ValueError as exc:
                    raise ValueError(f"non-integer DIMACS token at {path}:{lineno}: {tok}") from exc
                if lit == 0:
                    clauses.append(tuple(pending))
                    pending = []
                else:
                    if not saw_header:
                        raise ValueError(f"DIMACS literal before p cnf header at {path}:{lineno}")
                    if abs(lit) > nvars:
                        raise ValueError(
                            f"DIMACS literal outside declared range at {path}:{lineno}: {lit}"
                        )
                    pending.append(lit)
    if pending:
        raise ValueError(f"DIMACS clause in {path} is missing terminating 0")
    if not saw_header:
        raise ValueError(f"DIMACS file {path} is missing a p cnf header")
    return nvars, clauses


def checked_lit(tok: str, path: Path, lineno: int, nvars: int, context: str) -> int:
    try:
        lit = int(tok)
    except ValueError as exc:
        raise ValueError(f"non-integer {context} token at {path}:{lineno}: {tok}") from exc
    if lit != 0 and abs(lit) > nvars:
        raise ValueError(f"{context} literal outside declared range at {path}:{lineno}: {lit}")
    return lit


def iter_drup(path: Path, nvars: int):
    with path.open("r", encoding="ascii") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            toks = line.split()
            if toks[0] == "d":
                raise ValueError(
                    f"deletion line {path}:{lineno} is unsupported by this narrow converter"
                )
            lits: list[int] = []
            for tok in toks:
                lit = checked_lit(tok, path, lineno, nvars, "proof")
                if lit == 0:
                    break
                lits.append(lit)
            else:
                raise ValueError(f"DRUP/DRAT line {path}:{lineno} is missing terminating 0")
            yield lineno, "add", tuple(lits)


def lit_value(lit: int, assignment: dict[int, bool]) -> bool | None:
    if lit == 0:
        raise ValueError("0 is not a valid literal")
    val = assignment.get(abs(lit))
    if val is None:
        return None
    return val if lit > 0 else not val


def assign_lit(lit: int, assignment: dict[int, bool]) -> bool:
    var = abs(lit)
    val = lit > 0
    old = assignment.get(var)
    if old is None:
        assignment[var] = val
        return True
    return old == val


def rup_hints(active: list[tuple[int, Clause]], clause: Clause, lineno: int) -> list[int] | None:
    assignment: dict[int, bool] = {}
    for lit in clause:
        if not assign_lit(-lit, assignment):
            raise ValueError(f"tautological added clause at proof line {lineno} is unsupported")

    hints: list[int] = []
    changed = True
    while changed:
        changed = False
        for cid, db_clause in active:
            unassigned = 0
            unit_lit = 0
            satisfied = False
            for lit in db_clause:
                val = lit_value(lit, assignment)
                if val is True:
                    satisfied = True
                    break
                if val is None:
                    unassigned += 1
                    unit_lit = lit
            if satisfied:
                continue
            if unassigned == 0:
                hints.append(cid)
                return hints
            if unassigned == 1:
                if assign_lit(unit_lit, assignment):
                    hints.append(cid)
                    changed = True
                else:
                    hints.append(cid)
                    return hints
    return None


def clause_to_lrat(clause: Clause, hints: list[int], cid: int) -> str:
    left = " ".join(str(l) for l in clause)
    right = " ".join(str(h) for h in hints)
    if left:
        return f"{cid} {left} 0 {right} 0"
    return f"{cid} 0 {right} 0"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cnf", type=Path)
    ap.add_argument("drup", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    try:
        nvars, original = parse_dimacs(args.cnf)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    active: list[tuple[int, Clause]] = [(i + 1, c) for i, c in enumerate(original)]
    next_id = len(original) + 1
    emitted: list[str] = []
    saw_empty = False
    last_add_empty = False

    try:
        proof_rows = list(iter_drup(args.drup, nvars))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for lineno, kind, clause in proof_rows:
        if kind == "del":
            raise AssertionError("iter_drup no longer emits deletion rows")
        if saw_empty:
            print(
                f"error: proof line {lineno} appears after the final empty-clause addition",
                file=sys.stderr,
            )
            return 2
        try:
            hints = rup_hints(active, clause, lineno)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if hints is None:
            print(
                f"error: line {lineno} is not RUP under conservative scan: {clause}",
                file=sys.stderr,
            )
            return 2
        emitted.append(clause_to_lrat(clause, hints, next_id))
        active.append((next_id, clause))
        next_id += 1
        if len(clause) == 0:
            saw_empty = True
            last_add_empty = True
        else:
            last_add_empty = False

    args.out.write_text("\n".join(emitted) + ("\n" if emitted else ""), encoding="ascii")
    print(
        "wrote "
        f"{args.out} (vars={nvars}, original={len(original)}, "
        f"additions={len(emitted)}, deletions=unsupported, empty={int(saw_empty)})",
        file=sys.stderr,
    )
    if not saw_empty or not last_add_empty:
        print("error: no empty-clause addition was seen", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
