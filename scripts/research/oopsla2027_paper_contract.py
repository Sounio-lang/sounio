#!/usr/bin/env python3
"""OOPSLA 2027 paper draft — bound to the rung evidence it cites.

Spec/artefact: docs/papers/oopsla2027/paper.md

The R5 contract binds the SKELETON (outline.md) to the rung specs. This
contract does the same for the full DRAFT: a paper is where claims get
restated far from the harness that measured them, and prose drifts. So every
verdict token cited in the draft's contribution tables must match the token
its rung's spec declares, every load-bearing figure must be present, and the
honesty concessions the related-work searches bought must survive prose edits.

Clauses:
  P1_TOKENS_BOUND    every rung/token citation in the draft matches its spec
  P2_ALL_RUNGS_CITED every rung R0..R15 that exists on disk is cited
  P3_FIGURES_PINNED  the load-bearing figures are present (drift guard)
  P4_HONESTY_MARKERS the narrowings and the one-sided concession are stated

Pure Python 3.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = "docs/papers/oopsla2027/paper.md"

SPEC_TOKEN_RE = re.compile(
    r"^\*\*Status:\*\*\s*`[^`]*`\s*[—-]+\s*`([A-Za-z0-9_]+)`", re.MULTILINE)
CITE_RE = re.compile(r"\|\s*(R[0-9]+)\s*\|\s*`([A-Za-z0-9_]+)`\s*\|")

# A contribution can rest on more than one rung — C2 is measured at R1 and
# closed at R29 — so a row may carry a rung LIST against a token LIST:
#
#   | C2 | ... | R1, R29 | `BOUND_16__...`; `CLOSURE_WALKED__...` |
#
# The two lists are zipped positionally, and a length mismatch is a failure
# rather than a truncation: silently dropping the tail is exactly how a paper
# ends up citing a rung it no longer supports.
MULTI_CITE_RE = re.compile(
    r"\|\s*(R[0-9]+(?:\s*,\s*R[0-9]+)+)\s*\|\s*((?:`[A-Za-z0-9_]+`\s*;?\s*)+)\|")


def multi_citations(paper: str) -> tuple[dict[str, str], bool]:
    """Rung->token pairs from multi-rung contribution rows."""
    out: dict[str, str] = {}
    ok = True
    for rung_cell, token_cell in MULTI_CITE_RE.findall(paper):
        rungs = [r.strip() for r in rung_cell.split(",")]
        tokens = re.findall(r"`([A-Za-z0-9_]+)`", token_cell)
        if len(rungs) != len(tokens):
            print(f"  row cites {len(rungs)} rungs ({', '.join(rungs)}) "
                  f"against {len(tokens)} tokens — cannot be paired")
            ok = False
            continue
        out.update(zip(rungs, tokens))
    return out, ok


def read(rel: str) -> str:
    try:
        return (REPO / rel).read_text(errors="replace")
    except OSError:
        return ""


def discovered_rungs() -> dict[str, str]:
    found: dict[str, str] = {}
    for p in sorted((REPO / "docs/research").glob(
            "self_falsifying_compilation_line*.md")):
        rel = str(p.relative_to(REPO))
        m = re.search(r"_line_r(\d+)_", rel)
        if m:
            found[f"R{int(m.group(1))}"] = rel
        elif re.search(r"_line_2026", rel):
            found["R0"] = rel
    return found


def spec_token(rel: str) -> str | None:
    m = SPEC_TOKEN_RE.search(read(rel))
    return m.group(1) if m else None


def clause_p1(paper: str, rungs: dict[str, str]) -> tuple[bool, dict]:
    cited = dict(CITE_RE.findall(paper))
    multi, multi_ok = multi_citations(paper)
    cited.update(multi)
    ok = bool(cited) and multi_ok
    if not cited:
        print("  no rung/token citations found in the draft")
    for rung, token in sorted(cited.items(), key=lambda kv: int(kv[0][1:])):
        declared = spec_token(rungs.get(rung, ""))
        if declared is None:
            print(f"  {rung}: spec declares no token (spec missing?)")
            ok = False
        elif declared != token:
            print(f"  {rung}: draft cites {token}")
            print(f"      spec declares  {declared}  <- DRAFT HAS DRIFTED")
            ok = False
        else:
            print(f"  {rung}: {token}")
    print(f"P1_TOKENS_BOUND {'PASS' if ok else 'FAIL'} — "
          f"{len(cited)} cited tokens checked against their specs")
    return ok, cited


def clause_p2(cited: dict, rungs: dict[str, str]) -> bool:
    missing = sorted(set(rungs) - set(cited), key=lambda r: int(r[1:]))
    ok = not missing
    if missing:
        print(f"  rungs on disk with no citation in the draft: {missing}")
    print(f"P2_ALL_RUNGS_CITED {'PASS' if ok else 'FAIL'} — "
          f"{len(cited)}/{len(rungs)} rungs cited")
    return ok


def clause_p3(paper: str) -> bool:
    """Figures whose silent drift would change the paper's meaning.

    Each is a figure a rung measured; if prose edits soften or drop one, the
    draft has drifted from its evidence. Presence pins (not cardinality pins —
    several figures legitimately recur in table and prose).
    """
    required = [
        ("343", "R6 shared-derivation pair count"),
        ("1 081", "R6 pair denominator"),
        ("31.7", "R6 percentage"),
        ("5 440", "R7/R8 audited basis products"),
        ("23 shared clusters", "R8 cluster count"),
        ("12 irreducible kernels", "R8 kernel count"),
        ("0.929", "R8 four-vs-three catch"),
        ("0.151", "R6/R7 falsifier-oracle similarity"),
        ("0.43", "R12 three-version reliability fraction"),
        ("0.44", "R12 five-version reliability fraction"),
        ("224 problems", "R12 study scale"),
        ("21 pairs", "R13 identical-fate pairs"),
        ("0.479", "R13 similarity lower bound"),
        ("0.594", "R13 similarity upper bound"),
        ("0.565", "R13 independent-pair kill agreement"),
        ("0.513", "R13 shared-pair kill agreement"),
        ("536", "R14 perturbation cells"),
        ("407", "R14 verdict changes"),
        ("117", "R14 crashes"),
        ("126 of 128", "R15 fiber graphs changed"),
        ("3.55e-15", "R3 E6 falsifier residue"),
        ("9.3", "R2 token-declaring spec percentage"),
    ]
    ok = True
    for needle, why in required:
        if needle not in paper:
            print(f"  MISSING figure {needle!r} — {why}")
            ok = False
    print(f"P3_FIGURES_PINNED {'PASS' if ok else 'FAIL'} — "
          f"{len(required)} load-bearing figures checked")
    return ok


def clause_p4(paper: str) -> bool:
    """The concessions the four searches bought must survive prose edits."""
    required = [
        ("is not new", "the mechanism novelty concession (first narrowing)"),
        ("build.rs", "the prior art making that concession concrete"),
        ("clone detection", "the technique novelty concession (second narrowing)"),
        ("Residual risk", "the remaining unsearched risk is still stated"),
        ("one-sided", "the fourth narrowing: the independence measure is one-sided"),
        ("withdrawn", "the corroborator compiler rule is withdrawn, not deferred"),
        ("case-study", "the single-corpus scope is stated in the threats"),
        ("not self-starting", "the falsifier limit is still stated"),
        ("Leveson", "the N-version lineage is cited"),
    ]
    ok = True
    for needle, why in required:
        if needle not in paper:
            print(f"  MISSING {needle!r} — {why}")
            ok = False
    print(f"P4_HONESTY_MARKERS {'PASS' if ok else 'FAIL'} — "
          f"the narrowings and scope concessions are still stated")
    return ok


def main() -> int:
    print("OOPSLA 2027 PAPER DRAFT — bound to the rung evidence it cites")
    print("=" * 74)

    paper = read(PAPER)
    if not paper:
        print(f"P1_TOKENS_BOUND FAIL — draft missing: {PAPER}")
        print("OOPSLA2027_PAPER_VERDICT INCOMPLETE")
        return 1

    rungs = discovered_rungs()
    p1, cited = clause_p1(paper, rungs)
    print()
    p2 = clause_p2(cited, rungs)
    print()
    p3 = clause_p3(paper)
    print()
    p4 = clause_p4(paper)
    print()

    print("=" * 74)
    if not (p1 and p2 and p3 and p4):
        print("OOPSLA2027_PAPER_VERDICT INCOMPLETE")
        return 1
    print(f"  rungs cited        : {len(cited)}/{len(rungs)}")
    print("OOPSLA2027_PAPER_VERDICT DRAFT_TOKEN_BOUND__FIGURES_PINNED__NARROWINGS_STATED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
