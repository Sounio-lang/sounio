#!/usr/bin/env python3
"""Fail unless every CI job selected by impact classification succeeded."""

from __future__ import annotations

import json
import os
import sys


def truthy(value: str | None) -> bool:
    return value == "true"


def is_nightly() -> bool:
    """True when this run is one that must actually execute the nightly jobs.

    A scheduled run always is. A workflow_dispatch is only when the operator
    ticked the `nightly` input, matching the `if:` on the r6-corpus-sweep job.
    """
    if os.environ.get("GITHUB_EVENT_NAME") == "schedule":
        return True
    return os.environ.get("NIGHTLY_INPUT", "").lower() == "true"


def main() -> int:
    needs = json.loads(os.environ.get("NEEDS_JSON", "{}"))
    impact = needs.get("impact", {}).get("outputs", {})
    failures: list[str] = []

    required = {
        "contracts": True,
        "native-selfhost-linux-x86_64": any(
            truthy(impact.get(key)) for key in ("compiler", "runtime", "stdlib", "tests", "full")
        ),
        "source-bootstrap-selfhost-linux-x86_64": truthy(impact.get("compiler")) or truthy(impact.get("full")),
        "madaros-current-source-deref-f64": any(
            truthy(impact.get(key)) for key in ("compiler", "tests", "full")
        ),
        "native-selfhost-macos-arm64": truthy(impact.get("compiler")) or truthy(impact.get("full")),
        "full-test-suite": any(truthy(impact.get(key)) for key in ("compiler", "runtime", "stdlib", "tests", "full")),
        "madaros-witness-gate": any(
            truthy(impact.get(key)) for key in ("compiler", "runtime", "stdlib", "tests", "full")
        ),
        "gate-wave-0": any(
            truthy(impact.get(key)) for key in ("compiler", "runtime", "stdlib", "tests", "full")
        ),
        "sounio-lint": any(truthy(impact.get(key)) for key in ("compiler", "stdlib", "tests", "sio", "full")),
        "lean-proofs": truthy(impact.get("lean")) or truthy(impact.get("full")),
        "website": truthy(impact.get("website")) or truthy(impact.get("full")),
        # #2392 follow-up. R6 is selected by the EVENT, not by impact
        # classification: it is nightly-only, so `skipped` is the correct
        # outcome on a pull request and a non-evaluation on a scheduled run.
        # Selecting it unconditionally would fail every PR; leaving it out of
        # this dict entirely — the state this fixes — means a red R6 never
        # reaches the verdict at all.
        "r6-corpus-sweep": is_nightly(),
    }

    for job, selected in required.items():
        result = needs.get(job, {}).get("result", "missing")
        if selected and result != "success":
            failures.append(f"selected job {job} ended as {result}")
        elif not selected and result not in {"success", "skipped"}:
            failures.append(f"unselected job {job} ended as {result}")

    if failures:
        for failure in failures:
            print(f"CI_DECISION_FAIL: {failure}", file=sys.stderr)
        return 1

    selected_jobs = ",".join(job for job, selected in required.items() if selected)
    print(f"CI_DECISION_PASS selected={selected_jobs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
