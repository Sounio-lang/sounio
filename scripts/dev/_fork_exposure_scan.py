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


def has_bare_pull_request_trigger(lines: list[str]) -> bool:
    """True if the workflow's `on:` section names `pull_request` (fork-reachable),
    as opposed to only `pull_request_target` (a separate, out-of-scope risk class --
    see docs/ops/fork_pr_self_hosted_runner_policy.md)."""
    in_on = False
    for line in lines:
        if re.match(r"^on:\s*(\[.*\])?\s*$", line):
            in_on = True
            # Bracket/inline form: `on: [push, pull_request]`.
            if BARE_PULL_REQUEST_INLINE_RE.search(line) and "pull_request_target" not in line:
                return True
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


_JOB_KEY_RE = re.compile(r"^(\s+)([A-Za-z0-9_.-]+):\s*$")


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
            # Dedent to column 0 outside `jobs:` -- top-level key, e.g. a new
            # `concurrency:`/`env:` section after the jobs map (not expected in
            # this repo's layout, but end the scan defensively either way).
            break

        m = _JOB_KEY_RE.match(line)
        if m and (job_indent is None or len(m.group(1)) == job_indent):
            if job_indent is None:
                job_indent = len(m.group(1))
            if job_name is not None:
                yield_blocks.append((job_name, job_body_indent, block[:]))
            job_name = m.group(2)
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
            if value == "":
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


def job_has_guard(block: list[str], body_indent: int) -> bool:
    for line in block:
        ind = indent_of(line)
        if ind == body_indent and re.match(r"^\s*if:\s*(.*)$", line):
            if GUARD_SUBSTRING in line:
                return True
            if "github.event_name != 'pull_request'" in line or 'github.event_name != "pull_request"' in line:
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
