#!/usr/bin/env python3
"""Do the artifacts this corpus cites actually exist?

Every mechanism this line has built checks what a gate COMPUTES AND EMITS: the
exit status, the proposition (R2), the evidence fingerprint (R17). None checks
what a claim CITES. A contract can be green, its token correct and its witness
matching, while the derivation it says it rests on is not in the tree.

Found by looking: `cd_tower_collapse_isomorphism.py`, cited as the source of the
"VERIFIED n<=8" parity-collapse map Phi -- the UPPER BOUND of the completeness
pincer for ZD_FIBER_SPECTRUM_COMPLETE_INVARIANT_N_LE_8 -- is not on this branch.
It was committed to `lean/cd-seamflip-forall-n` and never merged here.

This scans the whole research corpus for cited repository artifacts and reports
which are absent, with what cites them and whether the citing file is itself
bound to a claim.

Conservative about prose, but NOT in the way the first version was.

INSTRUMENT FAILURE, recorded because it is the better half of this rung. The
first version accepted a bare basename as a citation only if that basename was
`git ls-files`-tracked -- i.e. only if the file EXISTED. So a file cited by bare
name and absent from the tree could never be reported. **The filter's
precondition was the negation of the thing being detected**, and the audit
duly returned 65 missing artifacts while dropping
`cd_tower_collapse_isomorphism.py`, the one whose absence motivated the rung.

The fix does not hardcode a whitelist. It derives the corpus's own naming
families from the tracked artifacts (the token before the first underscore of
every tracked research script), and accepts a bare basename that belongs to one
of them whether or not it exists. Prose filenames outside those families are
still ignored.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SCAN_GLOBS = ["scripts/research/*.py", "docs/research/*.md", "scripts/ci/*.sh"]
# A citation is a path under a known artifact directory, with a research
# extension. Bare basenames are resolved against scripts/research/.
DIR_RE = re.compile(
    r"(?:scripts/(?:research|ci)|docs/research|self-hosted/[\w/]+|stdlib/[\w/]+)"
    r"/[\w./-]+\.(?:py|sh|sio|md|lean)")
BARE_RE = re.compile(r"\b([a-z0-9_]{6,}\.(?:py|sh|sio))\b")


def tracked() -> set[str]:
    out = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True,
                         text=True).stdout.split()
    return set(out)


def families(known: set[str]) -> set[str]:
    """Naming families the corpus actually uses, derived from tracked artifacts.

    Self-calibrating on purpose: a hardcoded prefix list would be a second place
    to keep in sync, and would quietly stop covering new families.
    """
    fam = set()
    for k in known:
        if k.startswith(("scripts/research/", "scripts/ci/", "docs/research/")):
            b = Path(k).stem
            if "_" in b:
                fam.add(b.split("_", 1)[0])
    return {f for f in fam if len(f) >= 2}


def main() -> int:
    files = [p for g in SCAN_GLOBS for p in sorted(REPO.glob(g))]
    known = tracked()
    known_base = {Path(k).name: k for k in known}
    fams = families(known)

    # A citation in prose and a dependency at runtime are not the same thing.
    # Dismissing the never-committed bucket as "planned names" was too quick:
    # cd_tower_automorphism_oracle.py is loaded by exec_module() with no
    # fallback, was NEVER committed anywhere, and its absence means the orbit
    # theorem's verification script cannot run in any checkout of this repo.
    EXEC_CTX = re.compile(
        r"(?:spec_from_file_location|exec_module|open\s*\(|runpy|subprocess|"
        r"execfile|source_path|SourceFileLoader|bash\s+|python3?\s+)")
    cites: dict[str, set[str]] = {}
    hard: set[str] = set()
    for p in files:
        rel = str(p.relative_to(REPO))
        try:
            s = p.read_text(encoding="utf8", errors="replace")
        except OSError:
            continue
        found = set(DIR_RE.findall(s))
        for b in BARE_RE.findall(s):
            # A bare basename counts when it belongs to one of the corpus's own
            # naming families -- NOT when it happens to exist. Requiring
            # existence here is what made the first version unable to report an
            # absent file at all.
            if b in known_base:
                found.add(known_base[b])
            elif b.split("_", 1)[0] in fams:
                found.add(f"scripts/research/{b}")
        for c in found:
            if c == rel:
                continue
            cites.setdefault(c, set()).add(rel)
            # hard dependency if the citation sits on a line with an
            # execution/loading construct
            base = Path(c).name
            for line in s.splitlines():
                if base in line and EXEC_CTX.search(line):
                    hard.add(c)
                    break

    missing = {c: v for c, v in cites.items() if not (REPO / c).exists()}
    missing_hard = {c for c in missing if c in hard}

    print("R20 — do the artifacts this corpus cites exist?")
    print("=" * 72)
    print(f"scanned {len(files)} files; {len(cites)} distinct artifacts cited; "
          f"{len(missing)} MISSING from the tree")
    print(f"  of the missing, {len(missing_hard)} are HARD DEPENDENCIES "
          f"(loaded/executed, not merely mentioned)")
    for c in sorted(missing_hard):
        print(f"      {c}")
    print()

    rows = []
    for c, citers in sorted(missing.items()):
        # where did it go? (any branch)
        log = subprocess.run(
            ["git", "log", "--all", "--oneline", "-1", "--", c],
            cwd=REPO, capture_output=True, text=True).stdout.strip()
        branches = ""
        if log:
            sha = log.split()[0]
            b = subprocess.run(["git", "branch", "-a", "--contains", sha],
                               cwd=REPO, capture_output=True, text=True).stdout
            branches = ", ".join(sorted(
                x.strip("* +").strip() for x in b.splitlines()
                if "origin/" not in x))[:60]
        rows.append({"artifact": c, "hard_dependency": c in hard,
                     "cited_by": sorted(citers),
                     "last_commit": log, "on_branches": branches})
        print(f"  MISSING  {c}")
        print(f"      cited by: {', '.join(sorted(citers))}")
        if log:
            print(f"      last seen: {log}")
            print(f"      lives on: {branches or '(no local branch)'}")
        else:
            print("      never committed anywhere")
        print()

    out = REPO / "scripts/research/r20/audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(
        {"scanned": len(files), "cited": len(cites), "missing": rows}, indent=1))
    print(f"-> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
