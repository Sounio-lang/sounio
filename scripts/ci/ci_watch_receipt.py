#!/usr/bin/env python3
"""Classify a completed GitHub Actions run and emit a short CI receipt."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


BLOCKED_PATTERNS = (
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"no space left on device",
        r"hosted runner.*(lost|offline|unavailable)",
        r"runner.*communication.*lost",
        r"resource not accessible by integration",
        r"bad credentials",
        r"authentication failed",
        r"rate limit exceeded",
        r"quota.*exceeded",
        r"secret.*(not set|unavailable|missing)",
    )
)
BLOCKED_PATTERNS = tuple(BLOCKED_PATTERNS)


def classify(run: dict, jobs: list[dict], logs: str) -> tuple[str, str]:
    conclusion = run.get("conclusion", "")
    job_conclusions = {job.get("conclusion", "") for job in jobs}
    if conclusion == "success":
        return "GREEN", "All selected checks passed; ready for merge subject to repository policy."
    if conclusion == "cancelled":
        return "SUPERSEDED", "Cancelled runs are silent because a newer run owns the evidence."
    if conclusion == "timed_out" or "timed_out" in job_conclusions:
        return "TIMEOUT", "Inspect the last completed step and reduce or split the bounded gate."
    if conclusion in {"action_required", "startup_failure", "stale"}:
        return "BLOCKED", "The run did not reach a trustworthy code verdict; inspect runner or repository infrastructure."
    if any(pattern.search(logs) for pattern in BLOCKED_PATTERNS):
        return "BLOCKED", "Retry only after the runner, credential, quota, or infrastructure condition changes."
    return "FIXABLE", "Open the failed job receipt, reproduce its focused gate, and patch the owning lane."


def evidence_lines(logs: str) -> list[str]:
    selected: list[str] = []
    for raw in logs.splitlines():
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw).strip()
        if not line or not re.search(r"(^|\W)(error|failed|failure|timeout|timed out)(\W|$)", line, re.IGNORECASE):
            continue
        line = line.replace("`", "'")
        selected.append(line[-240:])
    return selected[-3:]


def render(run: dict, jobs: list[dict], logs: str) -> tuple[dict, str]:
    status, next_action = classify(run, jobs, logs)
    failed = [
        job.get("name", "unnamed")
        for job in jobs
        if job.get("conclusion") not in {"success", "skipped", "neutral", None, ""}
    ]
    receipt = {
        "schema": "sounio.ci-watch-receipt.v1",
        "status": status,
        "run_id": run.get("id"),
        "head_sha": run.get("head_sha"),
        "url": run.get("html_url"),
        "failed_jobs": failed,
        "next_action": next_action,
    }
    lines = [
        "<!-- sounio-ci-watch -->",
        f"**CI Watch: {status}**",
        "",
        f"Run: [{run.get('id', 'unknown')}]({run.get('html_url', '#')})  ",
        f"Head: `{str(run.get('head_sha', 'unknown'))[:12]}`  ",
        f"Failed jobs: `{', '.join(failed) if failed else 'none'}`",
        "",
        f"Next action: {next_action}",
    ]
    evidence = evidence_lines(logs)
    if evidence:
        lines.extend(("", "Evidence:"))
        lines.extend(f"- `{line}`" for line in evidence)
    lines.extend(("", "This watcher observes CI only. It cannot merge, edit code, or decide scientific semantics."))
    return receipt, "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--jobs", required=True, type=Path)
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--receipt-json", required=True, type=Path)
    parser.add_argument("--receipt-md", required=True, type=Path)
    args = parser.parse_args()

    run = json.loads(args.run.read_text())
    jobs_payload = json.loads(args.jobs.read_text())
    jobs = jobs_payload.get("jobs", jobs_payload if isinstance(jobs_payload, list) else [])
    logs = "\n".join(path.read_text(errors="replace") for path in sorted(args.logs.glob("*.log")))
    receipt, markdown = render(run, jobs, logs)
    args.receipt_json.write_text(json.dumps(receipt, indent=2) + "\n")
    args.receipt_md.write_text(markdown)

    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as handle:
            handle.write(f"status={receipt['status']}\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
