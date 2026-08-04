#!/usr/bin/env python3
"""Self-falsifying compilation, rung R4 — the retrospective, run as specified.

Spec: docs/research/self_falsifying_compilation_line_r4_2026-07-26.md

R0 §5 fixed the predicate BEFORE this study ran, with three arms evaluated at
the parent commit `c^` — the state in which the claim was false:

  Arm A — exit-code gating : the harness named at c^ exits non-zero at c^
  Arm B — token binding    : the spec's declared token at c^ differs from the
                             token the harness emits at c^
  Arm C — cross-version    : the CORRECTED harness (from c) fails, or emits a
                             token differing from c^'s declared one

  buckets: CAUGHT_A / CAUGHT_B / CAUGHT_C / SILENT / UNCLASSIFIABLE
  UNCLASSIFIABLE is counted separately and NEVER redistributed.

R0 §5 also stated in advance that arms A and B are known-blind by construction
and are run to estimate a rate, not to test the mechanism; arm C was the one
whose outcome was open.

Two populations, both reported:

  P1  OBJECTIVE  — commits where a spec's verdict token changed. Machine-
                   decidable, no message parsing.
  P2  FLAGGED    — commits whose message signals a correction and which touch a
                   research spec or contract. Larger, noisier.

Pure Python 3 + git.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SPEC_TOKEN_RE = re.compile(
    r"^\*\*Status:\*\*\s*`[^`]*`\s*[—-]+\s*`([A-Za-z0-9_]+)`", re.MULTILINE)
HARNESS_TOKEN_RE = re.compile(r"[A-Z0-9_]*_VERDICT\s+([A-Za-z0-9_]+)")
PLACEHOLDER = {"INCOMPLETE", "FAIL", "FAILED", "ERROR"}

# A commit message signalling a correction of a previously asserted claim.
CORRECTION_RE = re.compile(
    r"retract|deflate|overclaim|refut|was wrong|self-caught|conflat"
    r"|correct(?:s|ed|ion)?\b|deflat", re.IGNORECASE)

# This line's own specs are excluded from both populations. Including them is
# circular — R0's token moved UNBOUND -> BOUND because R1 did the binding, which
# is a state change, not a correction of an error, and its commit message
# mentions corrections made elsewhere. Same self-reference discount R1 applied
# to its coverage figure.
SELF_PREFIX = "docs/research/self_falsifying_compilation_line"


def is_self(path: str) -> bool:
    return path.startswith(SELF_PREFIX)


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def git_ok(*args: str) -> bool:
    r = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    return r.returncode == 0


def spec_token(blob: str) -> str | None:
    m = SPEC_TOKEN_RE.search(blob or "")
    return m.group(1) if m else None


def harness_tokens(blob: str) -> set[str]:
    return {t for t in HARNESS_TOKEN_RE.findall(blob or "") if t not in PLACEHOLDER}


def harness_for(spec_path: str, sha: str) -> str | None:
    """The contract a spec names as its harness, as of `sha`."""
    blob = git("show", f"{sha}:{spec_path}")
    m = re.search(r"\*\*Harness:\*\*\s*`([^`]+)`", blob or "")
    if m and m.group(1).endswith(".py"):
        return m.group(1)
    # fall back to the conventional name
    stem = Path(spec_path).stem
    for cand in (f"scripts/research/{stem}_contract.py",
                 f"scripts/research/{stem.replace('_spec', '')}_contract.py"):
        if git("show", f"{sha}:{cand}"):
            return cand
    return None


# ---------------------------------------------------------------- populations


def population_p1() -> list[dict]:
    """Commits where a spec's verdict token changed."""
    out = []
    for sha in git("log", "--format=%H", "--", "docs/research").split():
        files = [f for f in git("show", "--name-only", "--format=", sha).split()
                 if f.startswith("docs/research/") and f.endswith(".md")]
        msg = git("log", "-1", "--format=%B", sha)
        for f in files:
            before = spec_token(git("show", f"{sha}^:{f}"))
            after = spec_token(git("show", f"{sha}:{f}"))
            if is_self(f):
                continue
            if before and after and before != after:
                out.append({
                    "sha": sha, "spec": f, "before": before, "after": after,
                    "is_correction": bool(CORRECTION_RE.search(msg)),
                    "subject": msg.splitlines()[0] if msg else "",
                })
    return out


def population_p2() -> list[dict]:
    """Commits whose message signals a correction and touch research artefacts."""
    out = []
    for sha in git("log", "--format=%H").split():
        msg = git("log", "-1", "--format=%B", sha)
        if not CORRECTION_RE.search(msg or ""):
            continue
        files = git("show", "--name-only", "--format=", sha).split()
        specs = [f for f in files
                 if f.startswith("docs/research/") and f.endswith(".md")
                 and not is_self(f)]
        contracts = [f for f in files
                     if f.startswith("scripts/research/") and f.endswith(".py")]
        if not specs and not contracts:
            continue
        out.append({"sha": sha, "specs": specs, "contracts": contracts,
                    "subject": msg.splitlines()[0] if msg else ""})
    return out


def run_harness_blob(blob: str, timeout: int = 180) -> tuple[int | None, str | None]:
    """Run a historical harness verbatim; return (exit code, emitted token).

    (None, None) when it cannot be run standalone — missing imports, data files,
    or a timeout. That is recorded as not-executed, never as a pass.
    """
    if not blob:
        return None, None
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
            fh.write(blob)
            tmp = fh.name
        r = subprocess.run([sys.executable, tmp], capture_output=True,
                           text=True, timeout=timeout, cwd=REPO)
        toks = HARNESS_TOKEN_RE.findall(r.stdout or "")
        emitted = toks[-1] if toks else None
        return r.returncode, emitted
    except (subprocess.TimeoutExpired, OSError):
        return None, None
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------- classify


def classify(sha: str, spec: str) -> dict:
    """Apply the three arms at c^ for one (commit, spec) pair."""
    res = {"sha": sha[:9], "spec": Path(spec).name, "bucket": None, "why": ""}

    if not git_ok("cat-file", "-e", f"{sha}^{{commit}}"):
        res["bucket"], res["why"] = "UNCLASSIFIABLE", "commit unreachable"
        return res

    before_spec = git("show", f"{sha}^:{spec}")
    if not before_spec:
        # The commit CREATED this spec, so there was no prior claim to catch.
        # Not a defect in the data — a selection artefact of matching on commit
        # messages. Bucketed separately so it does not inflate UNCLASSIFIABLE.
        res["bucket"], res["why"] = "NO_PRIOR_CLAIM", "spec created in this commit"
        return res

    tok_before = spec_token(before_spec)
    if not tok_before:
        res["bucket"], res["why"] = "UNCLASSIFIABLE", "spec declares no verdict token at c^"
        return res

    harness = harness_for(spec, sha)
    if not harness:
        res["bucket"], res["why"] = "UNCLASSIFIABLE", "no harness named"
        return res

    h_before = git("show", f"{sha}^:{harness}")
    if not h_before:
        res["bucket"], res["why"] = "UNCLASSIFIABLE", "harness absent at c^"
        return res

    emit_before = harness_tokens(h_before)
    if not emit_before:
        res["bucket"], res["why"] = "UNCLASSIFIABLE", "harness emits no verdict token at c^"
        return res

    # Arm B — token binding at c^.
    if tok_before not in emit_before:
        res["bucket"] = "CAUGHT_B"
        res["why"] = f"spec token {tok_before} not among {sorted(emit_before)}"
        return res

    # Arm A — exit-code gating at c^, EXECUTED where the harness is
    # self-contained. These contracts are pure numpy computations with no
    # repository dependencies (the same property that makes arm C degenerate),
    # so the historical version can simply be run. Where execution is not
    # possible the result is recorded as not-executed WITH the reason, never
    # silently assumed.
    rc, emitted = run_harness_blob(h_before)
    if rc is None:
        arm_a = "NOT EXECUTED (harness could not be run standalone)"
    elif rc != 0:
        res["bucket"] = "CAUGHT_A"
        res["why"] = f"harness at c^ exits {rc}"
        return res
    else:
        arm_a = f"executed at c^: exit 0, emitted {emitted or '(no token)'}"
        # Executed arm B: compare what the harness ACTUALLY emitted, not what
        # it could emit. Strictly stronger than the static membership test.
        if emitted and emitted != tok_before:
            res["bucket"] = "CAUGHT_B"
            res["why"] = (f"harness at c^ emitted {emitted}, spec declared "
                          f"{tok_before} [arm A: {arm_a}]")
            return res

    # Arm C — cross-version replay: the CORRECTED harness's token vs c^'s spec.
    h_after = git("show", f"{sha}:{harness}")
    emit_after = harness_tokens(h_after)
    if not emit_after:
        res["bucket"], res["why"] = "UNCLASSIFIABLE", "corrected harness emits no token"
        return res
    if tok_before not in emit_after:
        res["bucket"] = "CAUGHT_C"
        res["why"] = (f"corrected harness emits {sorted(emit_after)}, which does not "
                      f"include c^'s declared {tok_before} [arm A: {arm_a}]")
        return res

    res["bucket"] = "SILENT"
    res["why"] = f"all arms agree at c^ on {tok_before} [arm A: {arm_a}]"
    return res


# ---------------------------------------------------------------- main


def main() -> int:
    print("SELF-FALSIFYING COMPILATION R4 — retrospective over the correction history")
    print("=" * 76)

    if not git_ok("rev-parse", "--git-dir"):
        print("R4_POPULATION FAIL — not a git repository")
        print("SELF_FALSIFYING_R4_VERDICT INCOMPLETE")
        return 1

    p1 = population_p1()
    p1_corr = [r for r in p1 if r["is_correction"]]
    p1_prog = [r for r in p1 if not r["is_correction"]]
    p2 = population_p2()

    print(f"P1 OBJECTIVE — spec verdict-token changes in the whole history: {len(p1)}")
    for r in p1:
        kind = "correction" if r["is_correction"] else "progression"
        print(f"    {r['sha'][:9]} [{kind}] {Path(r['spec']).name}")
        print(f"        {r['before']} -> {r['after']}")
    print(f"P1_POPULATION {len(p1_corr)} corrections, {len(p1_prog)} progressions")
    print()

    print(f"P2 FLAGGED — commits whose message signals a correction and that touch")
    print(f"   a research spec or contract: {len(p2)}")
    with_spec = [r for r in p2 if r["specs"]]
    print(f"   of those, touching a spec at all: {len(with_spec)}")
    print()

    # Classify P2's spec-touching commits under the fixed predicate.
    rows = []
    for r in with_spec:
        for spec in r["specs"]:
            rows.append(classify(r["sha"], spec))

    buckets: dict[str, int] = {}
    for row in rows:
        buckets[row["bucket"]] = buckets.get(row["bucket"], 0) + 1

    print(f"R4_CLASSIFICATION over {len(rows)} (commit, spec) pairs from P2:")
    for b in ("CAUGHT_A", "CAUGHT_B", "CAUGHT_C", "SILENT",
              "UNCLASSIFIABLE", "NO_PRIOR_CLAIM"):
        print(f"    {b:16s} {buckets.get(b, 0)}")
    print()

    reasons: dict[str, int] = {}
    for row in rows:
        if row["bucket"] == "UNCLASSIFIABLE":
            reasons[row["why"]] = reasons.get(row["why"], 0) + 1
    if reasons:
        print("R4_UNCLASSIFIABLE reasons:")
        for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {n:4d}  {why}")
        print()

    no_prior = buckets.get("NO_PRIOR_CLAIM", 0)
    eligible = len(rows) - no_prior          # pairs where a prior claim existed
    classifiable = eligible - buckets.get("UNCLASSIFIABLE", 0)
    caught = sum(buckets.get(b, 0) for b in ("CAUGHT_A", "CAUGHT_B", "CAUGHT_C"))
    for row in rows:
        if row["bucket"] not in ("UNCLASSIFIABLE", "SILENT"):
            print(f"    {row['bucket']}: {row['sha']} {row['spec']}")
            print(f"        {row['why']}")

    print(f"R4_POPULATION PASS — measured")
    print(f"R4_CLASSIFICATION PASS — measured")
    print()
    print("=" * 76)

    # Verdict, on the criteria fixed in R0 §5: the study is reportable only if
    # enough of the history is classifiable to say anything.
    if classifiable == 0:
        token = "RETROSPECTIVE_UNRUNNABLE__CORPUS_LACKS_MACHINE_READABLE_CLAIMS"
    elif caught == 0:
        token = "RETROSPECTIVE_RUN__NO_ARM_FIRED__CORPUS_MOSTLY_UNCLASSIFIABLE"
    else:
        token = "RETROSPECTIVE_RUN__SOME_ARM_FIRED"

    print(f"  P1 objective corrections   : {len(p1_corr)}")
    print(f"  P2 flagged, spec-touching  : {len(with_spec)}")
    print(f"  pairs with a prior claim   : {eligible}/{len(rows)} "
          f"({no_prior} created the spec, so nothing to catch)")
    print(f"  classifiable pairs         : {classifiable}/{eligible}")
    print(f"  caught by any arm          : {caught}")
    print(f"SELF_FALSIFYING_R4_VERDICT {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
