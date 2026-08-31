"""Implementation of the `sounio_test` MCP tool."""

from __future__ import annotations

import fnmatch
import time
from pathlib import Path
from typing import Any

from .common import REPO_ROOT, SounioToolError, resolve_souc, run_command, workspace_path
from .diagnostics import diagnostics_from_payload


def _annotations(path: Path) -> dict[str, list[str]]:
    annotations: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines()[:20]:
        if not line.startswith("//@"):
            continue
        body = line[3:].strip()
        if ":" in body:
            key, value = body.split(":", 1)
            annotations.setdefault(key.strip(), []).append(value.strip())
        else:
            annotations.setdefault(body, []).append("")
    return annotations


def _discover(pattern: str, test_filter: str | None) -> list[Path]:
    root = workspace_path(pattern)
    if root.is_file():
        candidates = [root]
    elif root.is_dir():
        candidates = sorted(root.rglob("*.sio"))
    else:
        raise SounioToolError(f"test pattern is not a file or directory: {pattern}")
    if not test_filter:
        return candidates
    selected: list[Path] = []
    for path in candidates:
        rel = str(path.relative_to(REPO_ROOT))
        if test_filter in path.name or test_filter in rel or fnmatch.fnmatch(rel, test_filter):
            selected.append(path)
    return selected


async def _run_one(souc: Path, path: Path) -> tuple[str, dict[str, Any] | None]:
    ann = _annotations(path)
    name = str(path.relative_to(REPO_ROOT))
    if "ignore" in ann:
        return "skipped", None
    is_compile_fail = (
        "compile-fail" in ann
        or "error-pattern" in ann
        or "compile-fail" in path.parts
    )
    check_only = "check-only" in ann
    if is_compile_fail or check_only:
        result = await run_command([souc, "check", path, "--json"], timeout_sec=30)
        diagnostics = diagnostics_from_payload(result.stdout, result.stderr)
        if is_compile_fail:
            expected = ann.get("error-pattern", [""])[0]
            haystack = result.stdout + "\n" + result.stderr + "\n" + repr(diagnostics)
            if diagnostics and (not expected or expected in haystack):
                return "passed", None
            return "failed", {
                "name": name,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "expected": expected or "diagnostics",
                "got": haystack[-2000:],
            }
        if not diagnostics:
            return "passed", None
        return "failed", {
            "name": name,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "expected": "no diagnostics",
            "got": repr(diagnostics),
        }

    result = await run_command([souc, "run", path], timeout_sec=30)
    expected_stdout = ann.get("expect-stdout", [""])[0]
    expected_contains = [m for m in ann.get("expect-stdout-contains", []) if m]
    if result.returncode != 0:
        return "failed", {
            "name": name,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "expected": expected_stdout or (expected_contains[0] if expected_contains else "exit 0"),
            "got": f"exit {result.returncode}",
        }
    if expected_stdout and expected_stdout not in result.stdout:
        return "failed", {
            "name": name,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "expected": expected_stdout,
            "got": f"exit {result.returncode}",
        }
    for marker in expected_contains:
        if marker not in result.stdout:
            return "failed", {
                "name": name,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "expected": marker,
                "got": f"exit {result.returncode}",
            }
    return "passed", None


async def sounio_test(
    test_pattern: str = "tests/run-pass",
    test_filter: str | None = None,
) -> dict[str, Any]:
    """Run Sounio tests programmatically through the checked compiler launcher."""

    started = time.perf_counter()
    try:
        tests = _discover(test_pattern, test_filter)
        souc = resolve_souc()
    except SounioToolError as exc:
        return {
            "total": 0,
            "passed": 0,
            "failed": 1,
            "skipped": 0,
            "duration_sec": 0.0,
            "failures": [
                {"name": test_pattern, "stdout": "", "stderr": str(exc), "expected": "", "got": ""}
            ],
        }

    passed = 0
    failed = 0
    skipped = 0
    failures: list[dict[str, Any]] = []
    for path in tests:
        status, failure = await _run_one(souc, path)
        if status == "passed":
            passed += 1
        elif status == "skipped":
            skipped += 1
        else:
            failed += 1
            if failure is not None:
                failures.append(failure)
    return {
        "total": len(tests),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "duration_sec": round(time.perf_counter() - started, 3),
        "failures": failures,
    }
