#!/usr/bin/env python3
"""Embed a DIMACS CNF file into a runnable Sounio (.sio) program that uses the CDCL solver.

The generated .sio program:
  - Imports theorem::cdcl::* (CdclCtx, cdcl_new, cdcl_add_clause, cdcl_solve,
    cdcl_nclauses, cdcl_verify)
  - Loads clauses via cdcl_add_clause(&!ctx, &arr, n) exactly as in
    tests/stdlib/theorem/test_cdcl_core.sio
  - Prints the result with print_int (never println(<raw int>) — compiler quirk #3)
  - For known-UNSAT instances (expected=UNSAT in comments or --expect-unsat flag),
    also calls cdcl_verify and prints the proof-checker result

ALL FOUR madaros-relocgate codegen quirks are respected:
  1. No module-global arrays — all state in CdclCtx struct, threaded by &!CdclCtx
  2. Value ctx passed as &!ctx from main; inside helper fns (ctx: &!CdclCtx) passed plain
  3. Integers printed via print_int(x as i64), never println(<int>)
  4. Negative literals written as (0 - N), not as bare -N

Literals: CDCL uses DIMACS-signed 1-indexed literals directly — NO even/odd re-encoding
(that is the SMT convention; CDCL takes signed i64 matching the DIMACS file verbatim).

One [i64; 1024] buffer is declared per block function and reused across clauses in that
block (the 8 KB array is stack-allocated once per call, not once per clause).

Usage:
  python embed_dimacs_cdcl.py input.cnf -o /tmp/output.sio [--expect-unsat]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

# CDCL capacity limits (from stdlib/theorem/cdcl.sio header)
CDCL_MAX_VARS = 1024
CDCL_MAX_CLAUSES = 16384
CDCL_CLAUSE_BUFFER = 1024  # [i64; 1024] — max lits per clause
CDCL_INPUT_MAX_CLAUSES = 8192
CDCL_INPUT_MAX_LITERALS = 32768
CDCL_INPUT_MAX_BYTES = 8 * 1024 * 1024

BLOCK_SIZE = 32  # clauses per helper function (keeps stack-frame size manageable)


# ---------------------------------------------------------------------------
# DIMACS parser (standalone; no dependency on generate_cdcl_dimacs)
# ---------------------------------------------------------------------------


def parse_dimacs(path: Path) -> tuple[int, list[tuple[int, ...]], str]:
    """Parse a DIMACS CNF file.

    Returns (n_vars, clauses, expected) where expected is "" if no
    'c expected=...' comment is present.
    """
    if path.stat().st_size > CDCL_INPUT_MAX_BYTES:
        raise ValueError(f"DIMACS input exceeds {CDCL_INPUT_MAX_BYTES} bytes")

    n_vars = 0
    declared_clauses: int | None = None
    saw_header = False
    clauses: list[tuple[int, ...]] = []
    pending: list[int] = []
    expected = ""
    total_literals = 0
    with path.open("rt", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line == "c" or (
                line.startswith("c") and len(line) > 1 and line[1].isspace()
            ):
                match = re.fullmatch(
                    r"c\s+expected\s*=\s*(SAT|UNSAT)", line, re.IGNORECASE
                )
                if match:
                    if expected:
                        raise ValueError(
                            "DIMACS contains more than one expected-result annotation"
                        )
                    expected = match.group(1).upper()
                elif re.match(r"c\s+expected", line, re.IGNORECASE):
                    raise ValueError(
                        f"invalid DIMACS expected-result annotation: {line}"
                    )
                continue
            if line.startswith("%"):
                raise ValueError(
                    "DIMACS '%' terminator is not supported; use an explicit clause count"
                )
            if line.startswith("p "):
                parts = line.split()
                if saw_header:
                    raise ValueError("DIMACS contains more than one problem header")
                if len(parts) != 4 or parts[1] != "cnf":
                    raise ValueError(f"invalid DIMACS problem header: {line}")
                n_vars = int(parts[2])
                declared_clauses = int(parts[3])
                if n_vars < 0 or declared_clauses < 0:
                    raise ValueError("DIMACS header counts must be non-negative")
                if n_vars >= CDCL_MAX_VARS:
                    raise ValueError(f"n_vars={n_vars} exceeds CDCL maximum 1023")
                if declared_clauses > CDCL_INPUT_MAX_CLAUSES:
                    raise ValueError(
                        f"{declared_clauses} input clauses exceeds conservative limit "
                        f"{CDCL_INPUT_MAX_CLAUSES}"
                    )
                saw_header = True
                continue
            if not saw_header:
                raise ValueError("DIMACS clause data appears before the problem header")
            for token in line.split():
                lit = int(token)
                if lit == 0:
                    clauses.append(tuple(pending))
                    pending = []
                    if len(clauses) > CDCL_INPUT_MAX_CLAUSES:
                        raise ValueError(
                            f"input clauses exceeds conservative limit {CDCL_INPUT_MAX_CLAUSES}"
                        )
                else:
                    if abs(lit) > n_vars:
                        raise ValueError(
                            f"literal {lit} exceeds declared variable count {n_vars}"
                        )
                    pending.append(lit)
                    total_literals += 1
                    if len(pending) > CDCL_CLAUSE_BUFFER:
                        raise ValueError(
                            f"clause exceeds {CDCL_CLAUSE_BUFFER} literal buffer"
                        )
                    if total_literals > CDCL_INPUT_MAX_LITERALS:
                        raise ValueError(
                            f"input literals exceeds conservative limit {CDCL_INPUT_MAX_LITERALS}"
                        )
    if pending:
        raise ValueError("DIMACS final clause is not terminated by 0")
    if not saw_header:
        raise ValueError("DIMACS problem header is missing")
    if declared_clauses != len(clauses):
        raise ValueError(
            f"DIMACS declares {declared_clauses} clauses but contains {len(clauses)}"
        )
    return n_vars, clauses, expected


# ---------------------------------------------------------------------------
# Sounio literal encoding
# ---------------------------------------------------------------------------


def sounio_literal(dimacs_lit: int) -> str:
    """Return the .sio source text for a DIMACS-signed literal.

    CDCL uses DIMACS-verbatim signed i64 literals (1-indexed, positive=true,
    negative=false).  Negative literals must be written as (0 - N) to satisfy
    the Sounio type-checker (bare negative integer literals error with E008/E001).
    """
    if dimacs_lit >= 0:
        return str(dimacs_lit)
    return f"0 - {-dimacs_lit}"


# ---------------------------------------------------------------------------
# .sio code generator
# ---------------------------------------------------------------------------


def generate_sio(
    n_vars: int,
    clauses: list[tuple[int, ...]],
    *,
    instance_id: str,
    cnf_path: Path,
    expected: str,
    emit_verify: bool,
    phase_mode: int | None = None,
) -> str:
    """Return the full text of a runnable .sio CDCL harness for the given CNF.

    *phase_mode* selects the polarity arm via cdcl_set_phase_mode before solving:
      0 = saved-phase, 1 = target-phasing (SOTA baseline), 2 = Beta-TS (novel).
    None leaves the solver default (0). The harness always prints the conflict
    count (SOUNIO_CDCL_CONFLICTS), the ablation metric.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", instance_id):
        raise ValueError(
            "instance_id must contain only letters, digits, '.', '_' or '-'"
        )
    if n_vars >= CDCL_MAX_VARS:
        raise ValueError(f"n_vars={n_vars} exceeds CDCL MAXV={CDCL_MAX_VARS}")
    if len(clauses) > CDCL_INPUT_MAX_CLAUSES:
        raise ValueError(
            f"{len(clauses)} input clauses exceeds conservative limit {CDCL_INPUT_MAX_CLAUSES}"
        )
    total_literals = sum(len(clause) for clause in clauses)
    if total_literals > CDCL_INPUT_MAX_LITERALS:
        raise ValueError(
            f"{total_literals} input literals exceeds conservative limit {CDCL_INPUT_MAX_LITERALS}"
        )
    for i, clause in enumerate(clauses):
        if len(clause) > CDCL_CLAUSE_BUFFER:
            raise ValueError(
                f"Clause {i} has {len(clause)} lits > CDCL buffer {CDCL_CLAUSE_BUFFER}"
            )

    n_blocks = max(1, (len(clauses) + BLOCK_SIZE - 1) // BLOCK_SIZE)

    lines: list[str] = []

    # Header
    lines += [
        "//@ run-pass",
        f"//@ generated_from_sha256: {hashlib.sha256(cnf_path.read_bytes()).hexdigest()}",
        f"//@ instance_id: {instance_id}",
        f"//@ n_vars: {n_vars}",
        f"//@ n_clauses: {len(clauses)}",
        f"//@ expected: {expected or 'unknown'}",
        "// Generated by scripts/research/embed_dimacs_cdcl.py",
        "// Follows idioms from tests/stdlib/theorem/test_cdcl_core.sio",
        "",
        "use theorem::cdcl::*",
        "",
        "// Quirk #3: never println(<raw int>); use this helper.",
        "fn print_int(n: i64) with IO { println(n as i32 as string) }",
        "",
    ]

    # Block functions — one per BLOCK_SIZE clauses
    # Each block takes ctx: &!CdclCtx and passes it plain to cdcl_add_clause
    # (ctx is already a ref inside the function).
    # One [i64; 1024] buffer is reused across all clauses in the block.
    for block_idx in range(n_blocks):
        start = block_idx * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, len(clauses))
        block_clauses = clauses[start:end]

        lines.append(f"// Clauses {start}..{end - 1}")
        # Quirk #2: ctx param is already &!CdclCtx — no need to re-borrow inside.
        # Only Mut effect needed (no IO, no Div).
        lines.append(f"fn add_block_{block_idx}(ctx: &!CdclCtx) -> i32 with Mut {{")
        # One shared buffer per block (8KB stack-allocated once, not per clause)
        lines.append("    var arr: [i64; 1024] = [0 as i64; 1024]")

        for clause in block_clauses:
            # Overwrite arr[0..len]; cdcl_add_clause reads only the first `len` slots.
            for slot, lit in enumerate(clause):
                lines.append(f"    arr[{slot}] = {sounio_literal(lit)}")
            # Quirk #3: capture return value with let _ to avoid tail-expression issues.
            lines.append(
                f"    if cdcl_add_clause(ctx, &arr, {len(clause)}) < 0 {{ return 0 - 1 }}"
            )

        lines.append("    return 0")
        lines.append("}")
        lines.append("")

    # main
    # Quirk #5: cdcl_solve carries the Div effect (Beta-TS arm uses sqrt/ln/hash),
    # so main must declare Div as well — otherwise E035 "effect not declared".
    lines.append("fn main() -> i32 with IO, Mut, Div {")
    # Quirk #1 (& #2): ctx is a value — always pass &!ctx to any fn that mutates it.
    lines.append("    var ctx = cdcl_new()")

    for block_idx in range(n_blocks):
        lines.append(f"    if add_block_{block_idx}(&!ctx) != 0 {{ return 2 }}")

    # Select the polarity arm (ablation knob) before solving.
    if phase_mode is not None:
        lines.append(f"    cdcl_set_phase_mode(&!ctx, {phase_mode})")

    lines.append("    let orig_nc = cdcl_nclauses(&!ctx)")
    lines.append("    let result = cdcl_solve(&!ctx)")
    lines.append(f'    print("SOUNIO_CDCL_RESULT {instance_id} ")')
    lines.append("    print_int(result as i64)")
    # Ablation metric: conflicts (read before the destructive cdcl_verify call).
    lines.append(f'    print("SOUNIO_CDCL_CONFLICTS {instance_id} ")')
    lines.append("    print_int(cdcl_n_conflicts(&!ctx))")
    lines.append(f'    print("SOUNIO_CDCL_DECISIONS {instance_id} ")')
    lines.append("    print_int(cdcl_n_decisions(&!ctx))")
    lines.append(f'    print("SOUNIO_CDCL_RESTARTS {instance_id} ")')
    lines.append("    print_int(cdcl_n_restarts(&!ctx))")

    lines.append("    if result == 1 {")
    lines.append("        var model_var: i32 = 1")
    lines.append(f"        while model_var <= {n_vars} {{")
    lines.append(f'            print("SOUNIO_CDCL_MODEL_VAR {instance_id} ")')
    lines.append("            print_int(model_var as i64)")
    lines.append(f'            print("SOUNIO_CDCL_MODEL_VALUE {instance_id} ")')
    lines.append("            print_int(cdcl_assignment(&!ctx, model_var) as i64)")
    lines.append("            model_var = model_var + 1")
    lines.append("        }")
    lines.append("    }")

    if emit_verify or expected == "UNSAT":
        # Call the independent RUP proof checker when UNSAT is expected or requested.
        # cdcl_verify returns: 1=VERIFIED, 0=did-not-reach-empty-clause, -1=lemma-failed.
        lines.append("    if result == 0 {")
        lines.append("        let vresult = cdcl_verify(&!ctx, orig_nc)")
        lines.append(f'        print("SOUNIO_CDCL_RUP_VERIFY {instance_id} ")')
        lines.append("        print_int(vresult as i64)")
        lines.append("        var proof_idx: i32 = 0")
        lines.append("        while proof_idx < cdcl_proof_len(&!ctx) {")
        lines.append("            var lit_idx: i32 = 0")
        lines.append(
            "            let proof_clause_len = cdcl_proof_clause_len(&!ctx, proof_idx)"
        )
        lines.append(f'            print("SOUNIO_CDCL_PROOF_CLAUSE {instance_id} ")')
        lines.append("            print_int(proof_clause_len as i64)")
        lines.append("            while lit_idx < proof_clause_len {")
        lines.append(f'                print("SOUNIO_CDCL_PROOF_LIT {instance_id} ")')
        lines.append(
            "                print_int(cdcl_proof_clause_lit(&!ctx, proof_idx, lit_idx))"
        )
        lines.append("                lit_idx = lit_idx + 1")
        lines.append("            }")
        lines.append("            proof_idx = proof_idx + 1")
        lines.append("        }")
        lines.append("    }")

    lines.append("    return 0")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level embed function (library API)
# ---------------------------------------------------------------------------


def embed(
    cnf_path: Path,
    out_path: Path,
    *,
    instance_id: str | None = None,
    expect_unsat: bool = False,
    emit_unsat_proof: bool = False,
    phase_mode: int | None = None,
) -> dict[str, object]:
    """Embed *cnf_path* as a .sio CDCL harness at *out_path*.

    *phase_mode* selects the polarity arm (0/1/2); None uses the solver default.

    Returns a metadata dict with keys: instance_id, n_vars, n_clauses, expected,
    phase_mode, cnf_sha256, sio_sha256, sio_path.
    """
    if instance_id is None:
        instance_id = cnf_path.stem

    n_vars, clauses, expected = parse_dimacs(cnf_path)
    if expect_unsat:
        expected = "UNSAT"

    sio_text = generate_sio(
        n_vars,
        clauses,
        instance_id=instance_id,
        cnf_path=cnf_path,
        expected=expected,
        emit_verify=(emit_unsat_proof or expected == "UNSAT"),
        phase_mode=phase_mode,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(sio_text, encoding="utf-8")

    cnf_sha256 = hashlib.sha256(cnf_path.read_bytes()).hexdigest()
    sio_sha256 = hashlib.sha256(out_path.read_bytes()).hexdigest()

    return {
        "instance_id": instance_id,
        "n_vars": n_vars,
        "n_clauses": len(clauses),
        "expected": expected or "unknown",
        "unsat_proof_instrumented": emit_unsat_proof or expected == "UNSAT",
        "phase_mode": phase_mode if phase_mode is not None else "default",
        "cnf_sha256": cnf_sha256,
        "sio_sha256": sio_sha256,
        "sio_path": str(out_path),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("cnf", type=Path, help="Input DIMACS CNF file")
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        help="Output .sio path (default: <stem>.sio in same dir)",
    )
    p.add_argument("--instance-id", help="Instance identifier (default: CNF stem)")
    p.add_argument(
        "--expect-unsat",
        action="store_true",
        help="Emit cdcl_verify call even if CNF has no expected= comment",
    )
    p.add_argument(
        "--phase-mode",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="Polarity arm: 0=saved-phase, 1=target-phasing (SOTA), 2=Beta-TS (novel)",
    )
    args = p.parse_args()

    out = args.out or args.cnf.with_suffix(".sio")
    try:
        meta = embed(
            args.cnf,
            out,
            instance_id=args.instance_id,
            expect_unsat=args.expect_unsat,
            phase_mode=args.phase_mode,
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    for k, v in meta.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
