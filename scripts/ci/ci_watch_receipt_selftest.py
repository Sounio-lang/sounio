#!/usr/bin/env python3

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("ci_watch_receipt.py")
SPEC = importlib.util.spec_from_file_location("ci_watch_receipt", MODULE_PATH)
assert SPEC and SPEC.loader
WATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WATCH)


def check(expected: str, conclusion: str, jobs=None, logs: str = "") -> None:
    status, _ = WATCH.classify(
        {"conclusion": conclusion}, jobs or [], logs
    )
    assert status == expected, (expected, status)


check("GREEN", "success")
check("SUPERSEDED", "cancelled")
check("TIMEOUT", "failure", [{"conclusion": "timed_out"}])
check("BLOCKED", "failure", logs="The hosted runner is unavailable")
check("BLOCKED", "startup_failure")
check("FIXABLE", "failure", logs="assertion failed: expected 3, got 2")

receipt, markdown = WATCH.render(
    {"conclusion": "failure", "id": 42, "head_sha": "abcdef123456", "html_url": "https://example.test/run/42"},
    [{"name": "Contracts", "conclusion": "failure"}],
    "error: registry is stale",
)
assert receipt["status"] == "FIXABLE"
assert "CI Watch: FIXABLE" in markdown
assert "cannot merge, edit code" in markdown
print("CI_WATCH_RECEIPT_SELFTEST_PASS")
