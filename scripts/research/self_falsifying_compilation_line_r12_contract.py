#!/usr/bin/env python3
"""Self-falsifying compilation, rung R12 — the N-version search.

Spec: docs/research/self_falsifying_compilation_line_r12_2026-07-27.md

R12 was planned in five phases (outside search, pre-registration, a hand-built
corpus of 12 implementation pairs, measurement, retrospective). Phase 0 returned
a pre-registered TERMINATING outcome and the remaining phases were not run. This
contract is therefore not a measurement harness: it is a *pin* on the external
facts that terminated the branch, plus a check that the stop was actually
honoured.

WHY PIN RATHER THAN MEASURE. The narrowing rests on figures read out of someone
else's paper. Prose citing a paper can be softened one adjective at a time with
every gate staying green — which is the sub-token failure this line has now hit
five times, once inside this very clause (see spec §5.1). Pinning the figures means re-widening the claim requires editing a
number the gate checks.

CLAUSES:

  C1_PRIOR_ART_PINNED
      Every load-bearing external figure appears verbatim in the spec: the
      arXiv identifiers, CodeBLEU's component weights, the study's scale, and
      the reliability-gain figures. Delete or soften one and this fails.

  C2_R6_MEASURE_IS_POORER
      §2's load-bearing claim — that R6's measure uses strictly less
      information than the CodeBLEU-based measure already shown insufficient —
      checked against R6's ACTUAL SOURCE, not against a reading of it. R6 must
      be shown to use canonicalised syntax only: no dataflow analysis, no
      lexical n-grams, and no execution of the code it compares.

  C3_STOP_WAS_HONOURED
      The pre-registered stop is falsifiable: the planned corpus directory must
      NOT exist. If a later rung builds it, this clause fails and says the
      spec's §0 has gone stale — rather than letting "we stopped at Phase 0"
      quietly become untrue.

VERDICT OPTIONS, FIXED BEFORE RUNNING (pre-registered in the plan, 2026-07-27):
  PRIOR_ART_HAS_ARTEFACT_MEASURE__CLAIM_NARROWS_FOURTH
      prior art has a mechanical artifact-level independence measure  [FIRED]
  MEASURE_BLIND__STRUCTURAL_DISTANCE_DOES_NOT_BOUND_SEMANTIC_INDEPENDENCE
      would have required Phases 2-4: constructed counterexample
  MEASURE_DISCRIMINATES__MISREADING_CONSTRAINS_STRUCTURE
      would have required Phases 2-4: misreadings converge structurally

WHAT THIS IS NOT. Not a replication of arXiv:2607.02808 — this rung explicitly
declines to run an n=6 replication of a 224-problem result. Not an exhaustive
search: two targeted questions, answered.

Pure Python 3 (ast, re).
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SPEC = "docs/research/self_falsifying_compilation_line_r12_2026-07-27.md"
R6_CONTRACT = "scripts/research/self_falsifying_compilation_line_r6_contract.py"
PLANNED_CORPUS = "scripts/research/r12_diversity_corpus"

VERDICT = "PRIOR_ART_HAS_ARTEFACT_MEASURE__CLAIM_NARROWS_FOURTH"

# Every one of these is load-bearing for the narrowing in spec §1.1/§2. Each was
# read from the paper itself (arXiv abstract page + decompressed PDF content
# streams, which agree verbatim) rather than from a search summary — see spec §6
# for why that mattered here.
#
# ! PINS CARRY THEIR CONTEXT, AND THE REASON IS A LIVE CATCH.
# The first version of this list pinned the bare string "0.43". Negative test N2
# softened the §1.1 headline figure to "roughly half" and THE GATE STAYED GREEN,
# because "0.43" still occurred in §3. A bare substring that appears more than
# once does not pin the sentence that carries the claim — it pins the corpus.
# That is the same sub-token failure this line has now hit five times, committed
# inside the guard written to prevent it. Each needle below is therefore long
# enough to be unique to the claim it guards, and `_pin_is_unique` enforces that
# no pin silently degrades into a multi-match again.
# Mode "unique" = a figure that CARRIES the claim: it must occur exactly once,
# so it cannot be softened in the sentence that matters while another occurrence
# keeps the gate green. Mode "present" = an identifier that SHOULD recur wherever
# the source is cited; demanding uniqueness there would penalise citing properly.
PINNED: list[tuple[str, str, str]] = [
    ("primary paper id",            "arXiv:2607.02808", "present"),
    ("primary paper date",          "submitted 2 July 2026", "unique"),
    ("CodeBLEU AST weight",         "**AST similarity (0.4)**", "unique"),
    ("CodeBLEU dataflow weight",    "**dataflow similarity (0.4)**", "unique"),
    ("study scale",                 "**224 problems × 12 models × 5 languages × 3 prompting strategies**", "unique"),
    ("reliability gain, 3 and 5",   "realise only **0.43** and **0.44** of the\n  reliability gain achievable under independence", "unique"),
    ("same-model gain",             "**below 0.3** when drawn from", "unique"),
    ("shared root causes",          "even different failure patterns often\n  share root causes", "unique"),
    ("gain restated in §3",         "only\nreaches 0.43 of the independent-case gain", "unique"),
    ("Knight & Leveson",            "Knight & Leveson (IEEE TSE 12(1), 1986", "unique"),
    ("ensemble-selection prior art", "EnsLLM, arXiv:2503.15838", "unique"),
    ("Type-4 definition",           "functionally similar without being textually similar", "unique"),
    ("R6 information claim",        "consults strictly less information than CodeBLEU does", "unique"),
    ("R6 non-containment caveat",   "neither contains the other", "unique"),
]

# R6 may import these and still be a syntax-only measure. Anything else — a
# dataflow library, an execution facility — would falsify §2's claim that R6
# uses strictly less information than CodeBLEU.
R6_ALLOWED_IMPORTS = {"__future__", "ast", "difflib", "re", "sys", "pathlib"}
R6_FORBIDDEN_NAMES = {"exec", "eval", "compile", "subprocess", "importlib", "runpy"}


def read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def clause_c1() -> tuple[bool, list[str]]:
    """Every load-bearing external figure appears in the spec, exactly once.

    Exactly once, not at least once: a needle matching in two places can be
    softened in the one that carries the claim while the gate stays green. That
    is not hypothetical — it is what the first version of this clause did.
    """
    try:
        spec = read(SPEC)
    except OSError as exc:
        print(f"C1_PRIOR_ART_PINNED FAIL  spec unreadable: {exc}")
        return False, []

    bad: list[str] = []
    for label, needle, mode in PINNED:
        n = spec.count(needle)
        flat = needle.replace("\n", " ")
        short = flat if len(flat) <= 52 else flat[:49] + "..."
        if n == 0:
            print(f"  [MISSING] {label}: {short}")
            bad.append(f"{label}: absent — {flat!r}")
        elif mode == "unique" and n > 1:
            print(f"  [AMBIGUOUS x{n}] {label}: {short}")
            bad.append(f"{label}: matches {n} places, so it pins nothing — {flat!r}")
        else:
            print(f"  [OK x{n}] {label}: {short}")

    ok = not bad
    print(f"C1_PRIOR_ART_PINNED {'PASS' if ok else 'FAIL'}  "
          f"{len(PINNED) - len(bad)}/{len(PINNED)} external figures uniquely pinned")
    return ok, bad


def clause_c2() -> bool:
    """§2's claim, checked against R6's source rather than against a reading of it."""
    try:
        src = read(R6_CONTRACT)
    except OSError as exc:
        print(f"C2_R6_MEASURE_IS_POORER FAIL  R6 contract unreadable: {exc}")
        return False

    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])

    unexpected = imported - R6_ALLOWED_IMPORTS
    used_forbidden = sorted(
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id in R6_FORBIDDEN_NAMES
    )

    print(f"  R6 imports: {sorted(imported)}")
    print(f"  [{'OK' if not unexpected else 'FAIL'}] no analysis beyond syntax"
          f"{'' if not unexpected else f' — unexpected: {sorted(unexpected)}'}")
    print(f"  [{'OK' if not used_forbidden else 'FAIL'}] never executes the code it compares"
          f"{'' if not used_forbidden else f' — found: {used_forbidden}'}")
    print("  => R6 consults canonicalised syntax ONLY. CodeBLEU also consults dataflow")
    print("     (0.4) and lexical n-grams (0.2) — 60% of its weight. Less information,")
    print("     not a sub-computation: the two syntactic measures differ. The richer")
    print("     one already failed at 224x12, so the poorer one cannot beat it.")

    ok = not unexpected and not used_forbidden
    print(f"C2_R6_MEASURE_IS_POORER {'PASS' if ok else 'FAIL'}")
    return ok


def clause_c3() -> bool:
    """The pre-registered stop is falsifiable: the planned corpus must not exist."""
    built = (REPO / PLANNED_CORPUS).exists()
    if built:
        print(f"  [FAIL] {PLANNED_CORPUS}/ exists — Phases 2-4 were run after all")
        print("         spec §0 claims the rung stopped at Phase 0 and is now STALE")
    else:
        print(f"  [OK] {PLANNED_CORPUS}/ absent — the n=6 replication was declined")
    ok = not built
    print(f"C3_STOP_WAS_HONOURED {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("R12 — the N-version search: prior art on artifact-level independence")
    print("=" * 72)
    print()
    print("Phase 0 of 5 ran. It returned a pre-registered terminating outcome and")
    print("Phases 1-4 were not executed. See spec §0.")
    print()

    c1_ok, missing = clause_c1()
    print()
    c2_ok = clause_c2()
    print()
    c3_ok = clause_c3()
    print()

    if missing:
        print("Missing pins mean the spec's claim has drifted from the evidence:")
        for m in missing:
            print(f"  - {m}")
        print()

    all_ok = c1_ok and c2_ok and c3_ok
    print("-" * 72)
    print("The measure R6 proposed is not new, and its central assumption was")
    print("refuted at 224 problems x 12 models three weeks before this rung ran.")
    print("R6's measure uses strictly less information than the one that failed.")
    print()
    print(f"SELF_FALSIFYING_R12_VERDICT {VERDICT}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
