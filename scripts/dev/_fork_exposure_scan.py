#!/usr/bin/env python3
"""Text-based scanner backing scripts/dev/check_self_hosted_runner_fork_exposure.sh.

Deliberately NOT a YAML parser -- a line/indent scan of the workflow source,
the same idiom scripts/ci/impact_ci_selftest.sh already uses to check
ci-decision's needs list against evaluate_ci_decision.py's required map.

Usage:
    _fork_exposure_scan.py <workflow.yml> [<workflow.yml> ...]
        Print one "<file>:<job>: <reason>" line per violation.

    _fork_exposure_scan.py --count-jobs <workflow.yml> [<workflow.yml> ...]
        Print the total number of jobs scanned (anti-vacuity control for the
        selftest -- a scanner that finds nothing is not a passing scanner).
"""
from __future__ import annotations

import re
import sys

SELF_HOSTED_RE = re.compile(r"self-hosted")
RUNNER_VAR_RE = re.compile(r"vars\.[A-Za-z0-9_]*RUNNER[A-Za-z0-9_]*", re.IGNORECASE)
GUARD_SUBSTRING = "head.repo.full_name == github.repository"
# Indent-width-agnostic (some leading whitespace, not a specific count): a
# `pull_request:` key nested under `on:` at any indentation, bare or as an
# explicit empty mapping (`pull_request: {}` is YAML-equivalent to bare
# `pull_request:` -- both mean "trigger on all default activity types").
BARE_PULL_REQUEST_BLOCK_RE = re.compile(r"^\s+pull_request:\s*(\{\s*\})?\s*$")
BARE_PULL_REQUEST_INLINE_RE = re.compile(r"(?<!_target)\bpull_request\b")


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


_ON_LINE_RE = re.compile(r"^on:\s*(.*)$")


def has_bare_pull_request_trigger(lines: list[str]) -> bool:
    """True if the workflow's `on:` section names `pull_request` (fork-reachable),
    as opposed to only `pull_request_target` (a separate, out-of-scope risk class --
    see docs/ops/fork_pr_self_hosted_runner_policy.md)."""
    in_on = False
    for line in lines:
        m = _ON_LINE_RE.match(line)
        if m:
            rest = m.group(1).strip()
            # A comment-only remainder (`on:  # triggers`) is YAML-equivalent
            # to a bare `on:` -- the real trigger value is in the following
            # block, not on this line. Without this check `rest` was truthy
            # (non-empty string), so the branch below searched "# triggers"
            # for a pull_request token, found none, and set in_on = False --
            # never scanning the block that actually names pull_request.
            if rest and not rest.startswith("#"):
                # Everything GitHub Actions allows on the `on:` line itself
                # without a following block: a bracketed list
                # (`on: [push, pull_request]`) or a single bare scalar
                # (`on: pull_request`). Both are just text at this point --
                # scan it directly for a standalone `pull_request` token.
                # BARE_PULL_REQUEST_INLINE_RE's trailing \b already refuses
                # to match inside `pull_request_target` (no word boundary
                # between "t" and the following "_"), so a combined list
                # like `[pull_request, pull_request_target]` is still
                # correctly flagged for the `pull_request` it also names --
                # unlike a bare `"pull_request_target" not in line` check,
                # which would wrongly disqualify the whole line.
                if BARE_PULL_REQUEST_INLINE_RE.search(rest):
                    return True
                # Inline form has no continuation block to scan.
                in_on = False
                continue
            in_on = True
            continue
        if in_on:
            if line and not line.startswith(" ") and not line.startswith("#"):
                in_on = False
                continue
            if BARE_PULL_REQUEST_BLOCK_RE.match(line):
                return True
            # `- pull_request` list-item form under `on:`.
            if re.match(r"^\s*-\s*pull_request\s*$", line):
                return True
    return False


# YAML allows a mapping key to be quoted (`"danger":` / `'danger':`), not
# just bare (`danger:`). The unquoted-only pattern let a quoted job key sail
# past unrecognized -- iter_job_blocks never opened a block for it, so
# job_is_self_hosted_ish never saw its runs-on:/if: lines at all, regardless
# of how dangerous they were.
_JOB_KEY_RE = re.compile(r"""^(\s+)(?:"([^"]+)"|'([^']+)'|([A-Za-z0-9_.-]+)):\s*$""")


def iter_job_blocks(lines: list[str]) -> list[tuple[str, int, list[str]]]:
    """Return (job_name, job_body_indent, block_lines) for each job under `jobs:`.

    The job-key indent width is detected from the FIRST job key seen under
    `jobs:`, not hardcoded to 2 spaces -- a workflow whose jobs map is
    consistently indented some other (nonzero) width is still scanned
    correctly, not silently skipped. Job body fields (runs-on:, if:, ...) are
    one further indent step inward from the job key -- but that step's WIDTH
    is not assumed to be 2 either (a file consistently indented some other
    width throughout would put them at job_indent + 4, not + 2). Each job's
    own body indent is instead captured from the first non-blank line inside
    its block, so detection tracks whatever width that file actually uses.
    """
    in_jobs = False
    job_indent: int | None = None
    job_name = None
    job_body_indent: int | None = None
    block: list[str] = []
    yield_blocks: list[tuple[str, int, list[str]]] = []

    for line in lines:
        if re.match(r"^jobs:\s*$", line):
            in_jobs = True
            continue
        if not in_jobs:
            continue
        if line and not line.startswith(" "):
            # A column-0 COMMENT is not a dedent -- YAML comments carry no
            # indentation semantics, so `# separator` between two jobs must
            # not end the scan and hide every job after it. Only a real
            # column-0 KEY (a new top-level section after the jobs map) ends
            # the scan.
            if line.lstrip().startswith("#"):
                continue
            break

        m = _JOB_KEY_RE.match(line)
        if m and (job_indent is None or len(m.group(1)) == job_indent):
            if job_indent is None:
                job_indent = len(m.group(1))
            if job_name is not None:
                yield_blocks.append((job_name, job_body_indent, block[:]))
            job_name = m.group(2) or m.group(3) or m.group(4)
            job_body_indent = None
            block = []
            continue
        if job_name is not None:
            if job_body_indent is None and line.strip():
                job_body_indent = indent_of(line)
            block.append(line)
    if job_name is not None:
        yield_blocks.append((job_name, job_body_indent, block[:]))
    return yield_blocks


def job_is_self_hosted_ish(block: list[str], body_indent: int) -> bool:
    collecting = False
    for line in block:
        ind = indent_of(line)
        if ind == body_indent and re.match(r"^\s*runs-on:\s*(.*)$", line):
            value = re.match(r"^\s*runs-on:\s*(.*)$", line).group(1)
            if SELF_HOSTED_RE.search(value) or RUNNER_VAR_RE.search(value):
                return True
            # Same comment-only-remainder case as has_bare_pull_request_trigger
            # above: `runs-on:  # see below` is YAML-equivalent to a bare
            # `runs-on:` followed by a block list, not a value of "# see below".
            if value == "" or value.strip().startswith("#"):
                collecting = True
                continue
            return False
        if collecting:
            if ind > body_indent and line.strip().startswith("-"):
                if SELF_HOSTED_RE.search(line) or RUNNER_VAR_RE.search(line):
                    return True
                continue
            collecting = False
    return False


_CANONICAL_GUARD_CLAUSES = {
    "github.event_name != 'pull_request'",
    'github.event_name != "pull_request"',
    GUARD_SUBSTRING.replace("head.repo.full_name", "github.event.pull_request.head.repo.full_name"),
}


def _split_top_level(expr: str, sep: str) -> list[str]:
    """Split expr on sep, but only where paren/bracket depth is 0 -- so a
    `sep` hiding inside a parenthesized sub-expression does not split it."""
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    i = 0
    n = len(expr)
    while i < n:
        c = expr[i]
        if c in "([":
            depth += 1
            buf.append(c)
            i += 1
        elif c in ")]":
            depth -= 1
            buf.append(c)
            i += 1
        elif depth == 0 and expr[i:i + len(sep)] == sep:
            parts.append("".join(buf))
            buf = []
            i += len(sep)
        else:
            buf.append(c)
            i += 1
    parts.append("".join(buf))
    return parts


def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        spans_whole = True
        for i, c in enumerate(s):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and i != len(s) - 1:
                    spans_whole = False
                    break
        if not spans_whole:
            break
        s = s[1:-1].strip()
    return s


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _is_restricting_clause(clause: str) -> bool:
    """A single &&-level clause actually restricts execution to same-repo
    pull_request events -- not merely a clause that MENTIONS the guard text
    somewhere it could be bypassed by an ||. Accepts: the bare
    `github.event_name != 'pull_request'` check; the bare same-repo compare;
    or an OR of exactly those two (the documented canonical guard) -- and
    ONLY when that OR is the clause's entire content, so
    `guard || vars.ENABLE == '1'` (a real bypass: true whenever the OTHER
    side is true, regardless of repo origin) is correctly rejected."""
    normalized = _normalize(_strip_outer_parens(clause))
    if normalized in _CANONICAL_GUARD_CLAUSES:
        return True
    or_parts = [_normalize(_strip_outer_parens(p)) for p in _split_top_level(normalized, "||")]
    return len(or_parts) == 2 and all(p in _CANONICAL_GUARD_CLAUSES for p in or_parts)


def job_has_guard(block: list[str], body_indent: int) -> bool:
    for line in block:
        ind = indent_of(line)
        m = ind == body_indent and re.match(r"^\s*if:\s*(.*)$", line)
        if not m:
            continue
        value = m.group(1)
        # The guard must be its own top-level &&-clause: `guard && rest` (or
        # `rest && guard`) restricts every path through the condition. A
        # guard merely present somewhere inside an ||-clause with unrelated
        # terms does not -- `A || vars.ENABLE == '1'` is true whenever EITHER
        # side is true, so embedding the guard there is not a restriction at
        # all. Splitting on top-level && and checking each clause in
        # isolation is what tells the two apart.
        for clause in _split_top_level(value, "&&"):
            if _is_restricting_clause(clause):
                return True
    return False


def scan(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    violations: list[str] = []
    if not has_bare_pull_request_trigger(lines):
        return violations

    for job_name, body_indent, block in iter_job_blocks(lines):
        if not job_is_self_hosted_ish(block, body_indent):
            continue
        if job_has_guard(block, body_indent):
            continue
        violations.append(f"{path}:{job_name}: self-hosted runner reachable via pull_request with no same-repo guard")
    return violations


def count_jobs(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    return len(iter_job_blocks(lines))


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: _fork_exposure_scan.py [--count-jobs] <workflow.yml> ...", file=sys.stderr)
        return 2

    if argv[0] == "--count-jobs":
        total = sum(count_jobs(p) for p in argv[1:])
        print(total)
        return 0

    all_violations: list[str] = []
    for path in argv:
        all_violations.extend(scan(path))

    for v in all_violations:
        print(v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
