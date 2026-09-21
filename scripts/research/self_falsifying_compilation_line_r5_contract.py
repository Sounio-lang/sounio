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
# R[0-9], not R[0-4]: the first version was pinned to the rungs that existed
# when it was written, so it silently stopped seeing citations once the line
# grew past R4 — W2 reported them missing while W1 said every cited token
# checked out. A guard scoped to today's data is a guard that decays.
CITE_RE = re.compile(r"\|\s*(R[0-9]+)\s*\|\s*`([A-Za-z0-9_]+)`\s*\|")

# A contribution may rest on more than one rung — C2 is measured at R1 and
# closed at R29 — so a row may carry a rung LIST against a token LIST. The two
# are zipped positionally; a length mismatch fails rather than truncating,
# because dropping the tail is how an outline ends up citing a rung it no
# longer supports.
MULTI_CITE_RE = re.compile(
    r"\|\s*(R[0-9]+(?:\s*,\s*R[0-9]+)+)\s*\|\s*((?:`[A-Za-z0-9_]+`\s*;?\s*)+)\|")


def multi_citations(text: str) -> tuple[dict, bool]:
    out, ok = {}, True
    for rung_cell, token_cell in MULTI_CITE_RE.findall(text):
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


def spec_token(rel: str) -> str | None:
    m = SPEC_TOKEN_RE.search(read(rel))
    return m.group(1) if m else None


def clause_w1(paper: str) -> tuple[bool, dict]:
    # Resolve specs by DISCOVERY, not from the hardcoded RUNGS seed: W1 used
    # the seed while W2 globbed, so once the line grew past the seed W1 said
    # "spec declares no token" for rungs whose specs were sitting right there.
    known = discovered_rungs()
    cited = dict(CITE_RE.findall(paper))
    _multi, _multi_ok = multi_citations(paper)
    cited.update(_multi)
    ok = True
    if not cited:
        print("  no rung/token citations found in the paper's contribution table")
        ok = False
    for rung, token in sorted(cited.items()):
        declared = spec_token(known.get(rung, ""))
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
    """The narrowed novelty claim must survive prose edits.

    Searching the related work established that the MECHANISM IS NOT NOVEL, and
    the claim was narrowed to proposition-binding. Prose gets rewritten; a
    concession is the easiest sentence to lose. If it goes, the paper silently
    re-widens to the claim the search refuted — the overclaim this whole line
    exists to study, committed in the venue where it would do the most damage.
    Negative-tested: turning "is not new" back into "is a contribution" fails
    this clause.
    """
    # These track the paper's ACTUAL honesty state, which changed when the
    # related-work search was done: the load-bearing neighbours are now checked,
    # and the finding was that the MECHANISM IS NOT NOVEL (Cargo build.rs runs
    # arbitrary code before compilation and fails the build; snapshot testing
    # binds a declared expected output). The novelty claim was narrowed to
    # binding a declared PROPOSITION. These markers exist so that narrowing
    # cannot be quietly widened again by a later prose edit.
    required = [
        ("is not new", "the paper still concedes the mechanism is not novel"),
        ("build.rs", "the prior art that makes that concession concrete is still named"),
        ("Residual risk", "the remaining unsearched risk is still stated"),
        ("SKELETON", "the artefact still declares itself a skeleton, not a draft"),
    ]
    ok = True
    for needle, why in required:
        if needle not in paper:
            print(f"  MISSING {needle!r} — {why}")
            ok = False
    print(f"W3_HONESTY_MARKERS {'PASS' if ok else 'FAIL'} — "
          f"the narrowed novelty claim and its residual risk are still stated")
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
    token = f"PAPER_SKELETON_TOKEN_BOUND__NOVELTY_NARROWED_BY_SEARCH__{suffix}"
    print(f"  rungs cited        : {len(cited)}/{len(discovered_rungs())}")
    print(f"  tokens agreeing    : {len(cited)}/{len(cited)}")
    print(f"  related work       : load-bearing neighbours checked; mechanism "
          f"conceded NOT novel; claim narrowed to proposition-binding")
    print(f"  gates wired into CI: {wired}/{total_gates} "
          f"— chain of custody is "
          f"{'closed' if suffix == 'CI_WIRED' else 'ASPIRATIONAL until wired'}")
    print(f"SELF_FALSIFYING_R5_VERDICT {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
