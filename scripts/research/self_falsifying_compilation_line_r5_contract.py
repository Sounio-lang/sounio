#!/usr/bin/env python3
"""Self-falsifying compilation, rung R5 — the paper skeleton, bound to its evidence.

Spec/artefact: docs/papers/oopsla2027/outline.md

A paper is where a research line's claims get restated in prose, far from the
harness that measured them. That is the sub-token failure mode with a wider
blast radius: the headline stays plausible while the number underneath it goes
stale, and no gate looks at prose.

So the paper skeleton cites, for every contribution, the verdict token of the
rung that measured it — and this contract fails if any cited token disagrees
with what that rung's spec declares.

Chain of custody:

    paper  ->  spec      checked here
    spec   ->  contract  checked by each rung's own gate

which is why this contract does not re-run the rung contracts (R4's scans the
whole history and takes minutes): that link is already guarded, and duplicating
it would make the paper gate slow enough that nobody runs it.

Clauses:
  W1_TOKENS_BOUND    every token cited in the paper matches its spec's Status
  W2_ALL_RUNGS_CITED every rung of the line is represented
  W3_HONESTY_MARKERS the unverified related-work section is still marked

Pure Python 3.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = "docs/papers/oopsla2027/outline.md"

# rung label -> spec that declares its verdict token
RUNGS = {
    "R0": "docs/research/self_falsifying_compilation_line_2026-07-26.md",
    "R1": "docs/research/self_falsifying_compilation_line_r1_2026-07-26.md",
    "R2": "docs/research/self_falsifying_compilation_line_r2_2026-07-26.md",
    "R3": "docs/research/self_falsifying_compilation_line_r3_2026-07-26.md",
    "R4": "docs/research/self_falsifying_compilation_line_r4_2026-07-26.md",
}

SPEC_TOKEN_RE = re.compile(
    r"^\*\*Status:\*\*\s*`[^`]*`\s*[—-]+\s*`([A-Za-z0-9_]+)`", re.MULTILINE)
# Tokens cited in the paper's contribution table: | R1 | ... | `TOKEN` |
CITE_RE = re.compile(r"\|\s*(R[0-4])\s*\|\s*`([A-Za-z0-9_]+)`\s*\|")


def read(rel: str) -> str:
    try:
        return (REPO / rel).read_text(errors="replace")
    except OSError:
        return ""


def spec_token(rel: str) -> str | None:
    m = SPEC_TOKEN_RE.search(read(rel))
    return m.group(1) if m else None


def clause_w1(paper: str) -> tuple[bool, dict]:
    cited = dict(CITE_RE.findall(paper))
    ok = True
    if not cited:
        print("  no rung/token citations found in the paper's contribution table")
        ok = False
    for rung, token in sorted(cited.items()):
        declared = spec_token(RUNGS.get(rung, ""))
        if declared is None:
            print(f"  {rung}: spec declares no token (spec missing?)")
            ok = False
        elif declared != token:
            print(f"  {rung}: paper cites {token}")
            print(f"      spec declares  {declared}  <- PAPER HAS DRIFTED")
            ok = False
        else:
            print(f"  {rung}: {token}")
    print(f"W1_TOKENS_BOUND {'PASS' if ok else 'FAIL'} — "
          f"{len(cited)} cited tokens checked against their specs")
    return ok, cited


def clause_w2(cited: dict) -> bool:
    missing = sorted(set(RUNGS) - set(cited))
    ok = not missing
    if missing:
        print(f"  rungs with no contribution cited: {missing}")
    print(f"W2_ALL_RUNGS_CITED {'PASS' if ok else 'FAIL'} — "
          f"{len(cited)}/{len(RUNGS)} rungs represented")
    return ok


def clause_w3(paper: str) -> bool:
    """The related-work honesty markers must survive prose edits.

    §7.2 is conjecture. If someone deletes the marker while leaving the list,
    the paper starts asserting a novelty claim nobody verified — which is the
    overclaim this whole line exists to study, committed in the venue where it
    would do the most damage.
    """
    required = [
        ("PARTIALLY VERIFIED", "the related-work heading still flags its status"),
        ("not yet checked", "§7.2 is still marked as unchecked"),
        ("SKELETON", "the artefact still declares itself a skeleton, not a draft"),
    ]
    ok = True
    for needle, why in required:
        if needle not in paper:
            print(f"  MISSING {needle!r} — {why}")
            ok = False
    print(f"W3_HONESTY_MARKERS {'PASS' if ok else 'FAIL'} — "
          f"unverified sections still marked as unverified")
    return ok


def main() -> int:
    print("SELF-FALSIFYING COMPILATION R5 — paper skeleton bound to its evidence")
    print("=" * 74)

    paper = read(PAPER)
    if not paper:
        print(f"W1_TOKENS_BOUND FAIL — paper missing: {PAPER}")
        print("SELF_FALSIFYING_R5_VERDICT INCOMPLETE")
        return 1

    w1, cited = clause_w1(paper)
    print()
    w2 = clause_w2(cited)
    print()
    w3 = clause_w3(paper)
    print()

    print("=" * 74)
    if not (w1 and w2 and w3):
        print("SELF_FALSIFYING_R5_VERDICT INCOMPLETE")
        return 1

    token = "PAPER_SKELETON_TOKEN_BOUND__RELATED_WORK_PARTIALLY_VERIFIED"
    print(f"  rungs cited        : {len(cited)}/{len(RUNGS)}")
    print(f"  tokens agreeing    : {len(cited)}/{len(cited)}")
    print(f"  related work       : 2 neighbours checked, §7.2 unverified")
    print(f"SELF_FALSIFYING_R5_VERDICT {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
