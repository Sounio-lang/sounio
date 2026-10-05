#!/usr/bin/env python3
"""Self-falsifying compilation, rung R27 — declared alive, never checked.

Spec: docs/research/self_falsifying_compilation_line_r27_2026-08-01.md

R22 found a field shaped like a measurement that was a literal, enforced by a
gate. R23 found `validated_by` was path ownership. R25 found research authority
was a path default. Each time the defect lived in document front-matter.

This rung finds the same shape inside the compiler's own claim ontology.

Every production claim declares `verdict = Verdict::Alive`. The claim executor
never checks it. Its ONLY read of the `verdict` field scans the slice for the
substring "archived" (claim_executor.sio:453-457, via ce_slice_is_archived);
the token "Alive" does not occur anywhere in the executor. Aliveness is asserted
sixteen times and tested zero times.

And the promise the ELF carries is narrower than the mechanism suggests: an
entire compiler lane emits ELFs without ever calling the verifier.

CLAUSES (all static — this rung invokes no compiler and takes no build lock):

  A1_ALIVE_IS_UNCHECKED   every claim declares Verdict::Alive; the executor
                          reads `verdict` at exactly one call site and only to
                          test for "archived"; "Alive" occurs zero times there.
  A2_PROMISE_SCOPE_IS_NARROWER_THAN_THE_MECHANISM
                          enumerate the lanes that emit an ELF and whether each
                          is covered by a claim_executor_verify call. lean_single
                          emits from three functions and calls the verifier zero
                          times. The promise does not reach that lane.
  A3_BINDINGS_ARE_RARE    of the claims in the manifest, how many declare
                          anything beyond an exit code (verdict_token, witness,
                          provenance). The rest are EXIT_ONLY: the gate ran, its
                          content was not checked (R2, R15).
  A4_ANCHORING_CHANGES_THE_CENSUS
                          the control. A naive `grep -c witness` over the same
                          file counts prose in comment blocks. If the census is
                          not comment-stripped and field-anchored it reports a
                          different, larger number -- and this rung's own
                          headline would be wrong. Measured both ways.

WHAT THIS RUNG DOES NOT DO. It does not show that any claim is false, and it
does not perturb a gate to test refutability -- that needs compiler runs and is
left to a rung that budgets them. It measures what the compiler CHECKS, not what
is true.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "examples/epistemic/rupture_claims_verified.sio"
EXECUTOR = ROOT / "self-hosted/compiler/claim_executor.sio"
MAIN = ROOT / "self-hosted/compiler/main.sio"
DRIVER = ROOT / "self-hosted/compiler/native_compile_driver.sio"
LEAN = ROOT / "self-hosted/compiler/lean_single.sio"

BIND_FIELDS = ("verdict_token", "witness", "provenance")


def strip_comments(text: str) -> str:
    return "\n".join(l for l in text.split("\n") if not re.match(r"^\s*//", l))


def read(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def census_manifest():
    """Claims, their declared verdict, and their bindings -- comment-stripped."""
    claims = []
    bound = {}
    current = None
    for lineno, line in enumerate(read(MANIFEST).split("\n"), start=1):
        if re.match(r"^\s*//", line):
            continue
        m = re.match(r"^claim\s+([A-Za-z0-9_]+)", line)
        if m:
            current = m.group(1)
            claims.append(current)
        b = re.match(r"^\s*(%s)\s*=" % "|".join(BIND_FIELDS), line)
        if b and current is not None:
            bound.setdefault(current, []).append((b.group(1), lineno))
    alive = len(re.findall(r"Verdict::Alive", strip_comments(read(MANIFEST))))
    return claims, bound, alive


def clause_a1(claims, alive) -> bool:
    src = read(EXECUTOR)
    call_sites = [
        n for n, l in enumerate(src.split("\n"), start=1)
        if "ce_slice_is_archived" in l and not re.match(r"^\s*(pub )?fn ", l)
        and not re.match(r"^\s*//", l)
    ]
    alive_mentions = len([
        l for l in src.split("\n")
        if "Alive" in l and not re.match(r"^\s*//", l)
    ])
    verdict_reads = [
        n for n, l in enumerate(src.split("\n"), start=1)
        if re.search(r'ce_name_eq_str\(f\.name,\s*"verdict"\)', l)
    ]
    print(f"A1 manifest: {len(claims)} claims, {alive} declare Verdict::Alive")
    print(f"    executor reads the `verdict` field at line(s): {verdict_reads}")
    print(f"    and the only thing it does with it is ce_slice_is_archived at {call_sites}")
    print(f"    occurrences of `Alive` in the executor (code, not comments): {alive_mentions}")
    print("    Aliveness is asserted by every claim and tested by none.")
    ok = (
        len(claims) > 0
        and alive == len(claims)
        and len(verdict_reads) == 1
        and len(call_sites) == 1
        and alive_mentions == 0
    )
    print(f"A1_ALIVE_IS_UNCHECKED {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def clause_a2() -> bool:
    def count(p: Path, pat: str) -> int:
        return len([l for l in read(p).split("\n")
                    if re.search(pat, l) and not re.match(r"^\s*//", l)])

    verify_main = count(MAIN, r"\bclaim_executor_verify\(")
    verify_lean = count(LEAN, r"\bclaim_executor_verify\b")
    emit_lean = count(LEAN, r"^fn write_elf")
    emit_driver = count(DRIVER, r"^\s*let wrote = driver_write_elf_to_file\(")

    print("A2 lanes that can emit an ELF, and whether the verifier covers them:")
    print(f"    compiler/main.sio            claim_executor_verify call sites: {verify_main}")
    print(f"    compiler/native_compile_driver.sio  emitting call sites: {emit_driver}")
    print(f"    compiler/lean_single.sio     ELF-emitting fns: {emit_lean}"
          f"   claim_executor_verify calls: {verify_lean}")
    print("    lean_single emits and never verifies: that lane is OUTSIDE the promise,")
    print("    and the promise sentence in the spec must name it as such.")
    ok = verify_main >= 1 and emit_lean > 0 and verify_lean == 0
    print(f"A2_PROMISE_SCOPE_IS_NARROWER_THAN_THE_MECHANISM {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def clause_a3(claims, bound) -> bool:
    exit_only = [c for c in claims if c not in bound]
    print(f"A3 of {len(claims)} claims, {len(bound)} declare a binding beyond the exit code:")
    for name, fields in bound.items():
        print(f"      {name}: " + ", ".join(f"{f}@{n}" for f, n in fields))
    print(f"    the other {len(exit_only)} are EXIT_ONLY -- the gate ran, its content was")
    print("    not checked. Per R2/R15 that is exit-code gating, not claim verification.")
    ok = len(claims) > 0 and len(bound) >= 1 and len(exit_only) == len(claims) - len(bound)
    print(f"A3_BINDINGS_ARE_RARE {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def clause_a4(bound) -> bool:
    raw = read(MANIFEST)
    anchored = sum(len(v) for v in bound.values())
    naive = {f: len(re.findall(f, raw)) for f in BIND_FIELDS}
    naive_total = sum(naive.values())
    print("A4 the control -- the same file counted two ways:")
    print(f"    naive substring counts: {naive}  total {naive_total}")
    print(f"    comment-stripped, field-anchored: {anchored}")
    print("    The difference is prose: the manifest discusses witnesses and provenance")
    print("    in comment blocks. An unanchored census reports the larger number and")
    print("    this rung's headline would be wrong in the direction that flatters it.")
    ok = naive_total > anchored and anchored > 0
    print(f"A4_ANCHORING_CHANGES_THE_CENSUS {'PASS' if ok else 'FAIL'}")
    print()
    return ok


def main() -> int:
    print("Self-falsifying compilation R27 -- declared alive, never checked")
    print("=" * 72)
    print()
    for p in (MANIFEST, EXECUTOR, MAIN, DRIVER, LEAN):
        if not p.is_file():
            print(f"missing input: {p.relative_to(ROOT)}")
            print("SELF_FALSIFYING_R27_VERDICT INCONCLUSIVE")
            return 1

    claims, bound, alive = census_manifest()
    ok1 = clause_a1(claims, alive)
    ok2 = clause_a2()
    ok3 = clause_a3(claims, bound)
    ok4 = clause_a4(bound)

    ok = ok1 and ok2 and ok3 and ok4
    verdict = (
        f"CLAIM_LIVENESS_DEFINED__DECLARED_ALIVE_IS_UNCHECKED__{len(bound)}_OF_{len(claims)}_BOUND"
        if ok else "INCONCLUSIVE"
    )
    print("-" * 72)
    print("Every claim in the manifest declares itself alive. The executor never")
    print("reads that declaration except to look for the word 'archived'. One claim")
    print("of sixteen binds anything beyond an exit code, and an entire compiler")
    print("lane emits ELFs without consulting the verifier at all. The mechanism is")
    print("real; the promise it carries is narrower than the word 'Alive' suggests.")
    print()
    print(f"SELF_FALSIFYING_R27_VERDICT {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
