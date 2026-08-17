#!/usr/bin/env python3
"""Fail unless every CI job selected by impact classification succeeded."""

from __future__ import annotations

import json
import os
import sys


def truthy(value: str | None) -> bool:
    return value == "true"


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
        "sounio-lint": any(truthy(impact.get(key)) for key in ("compiler", "stdlib", "tests", "sio", "full")),
        "lean-proofs": truthy(impact.get("lean")) or truthy(impact.get("full")),
        "website": truthy(impact.get("website")) or truthy(impact.get("full")),
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
