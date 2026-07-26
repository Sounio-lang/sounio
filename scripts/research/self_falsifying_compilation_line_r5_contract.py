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

BUT that reasoning only holds if the rung gates actually RUN. W4 measures
whether they are wired into a CI workflow; if they are not, the spec->contract
link is guarded by a check nobody executes, and the chain of custody is
aspirational rather than closed. The verdict token carries that state, so the
paper cannot assert a guarantee the repository does not provide.

Clauses:
  W1_TOKENS_BOUND    every token cited in the paper matches its spec's Status
  W2_ALL_RUNGS_CITED every rung that EXISTS is represented (globbed, not listed)
  W3_HONESTY_MARKERS the unverified related-work section is still marked
  W4_CI_WIRING       are the line's gates invoked by any CI workflow?

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


def discovered_rungs() -> dict[str, str]:
    """Rung specs actually on disk, so a rung added later cannot be omitted.

    A hardcoded list would pass forever while silently ignoring R6 — the same
    blind spot as a bucket that grows unnoticed.
    """
    found = dict(RUNGS)
    for p in sorted((REPO / "docs/research").glob(
            "self_falsifying_compilation_line*.md")):
        rel = str(p.relative_to(REPO))
        if rel in found.values():
            continue
        m = re.search(r"_line_r(\d+)_", rel)
        found[f"R{m.group(1)}" if m else "R?"] = rel
    return found


def clause_w2(cited: dict) -> bool:
    rungs = discovered_rungs()
    missing = sorted(set(rungs) - set(cited))
    ok = not missing
    if missing:
        print(f"  rung specs on disk with no contribution cited: "
              f"{[(r, rungs[r]) for r in missing]}")
    print(f"W2_ALL_RUNGS_CITED {'PASS' if ok else 'FAIL'} — "
          f"{len(cited)}/{len(rungs)} rungs represented (set discovered on disk)")
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


def clause_w4() -> tuple[bool, int, int]:
    """Are the line's gates invoked by any CI workflow?"""
    gates = sorted(p.name for p in (REPO / "scripts/ci").glob(
        "self_falsifying_compilation_line*_gate.sh"))
    wf_text = ""
    wf_dir = REPO / ".github/workflows"
    if wf_dir.is_dir():
        for wf in wf_dir.glob("*.yml"):
            wf_text += wf.read_text(errors="replace")
    wired = [g for g in gates if g in wf_text]
    unwired = [g for g in gates if g not in wired]
    if unwired:
        print(f"  gates NOT invoked by any workflow: {unwired}")
    print(f"W4_CI_WIRING {len(wired)}/{len(gates)} of the line's gates are wired "
          f"into CI")
    # Reported, not required: whether to spend CI minutes on these is a
    # repository decision, and .github/workflows/ci.yml is presently being
    # edited by another agent. The verdict token below carries the state so it
    # cannot be quietly forgotten.
    print(f"W4_CI_WIRING PASS — measured")
    return True, len(wired), len(gates)


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
    w4, wired, total_gates = clause_w4()
    print()

    print("=" * 74)
    if not (w1 and w2 and w3 and w4):
        print("SELF_FALSIFYING_R5_VERDICT INCOMPLETE")
        return 1

    suffix = "CI_WIRED" if wired == total_gates and total_gates else "CI_UNWIRED"
    token = ("PAPER_SKELETON_TOKEN_BOUND__RELATED_WORK_PARTIALLY_VERIFIED"
             f"__{suffix}")
    print(f"  rungs cited        : {len(cited)}/{len(discovered_rungs())}")
    print(f"  tokens agreeing    : {len(cited)}/{len(cited)}")
    print(f"  related work       : 2 neighbours checked, §7.2 unverified")
    print(f"  gates wired into CI: {wired}/{total_gates} "
          f"— chain of custody is "
          f"{'closed' if suffix == 'CI_WIRED' else 'ASPIRATIONAL until wired'}")
    print(f"SELF_FALSIFYING_R5_VERDICT {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
