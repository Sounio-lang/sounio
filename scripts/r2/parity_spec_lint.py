#!/usr/bin/env python3
"""Static validator for scripts/r2/parity-spec.toml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"error: tomllib unavailable: {exc}") from exc


ALLOWED_COMMANDS = {
    "analyze",
    "bench",
    "build",
    "check",
    "compile",
    "diagnostics",
    "fmt",
    "help",
    "info",
    "lint",
    "repl",
    "run",
    "test",
    "version",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate R2 parity spec TOML shape.")
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("scripts/r2/parity-spec.toml"),
        help="Path to parity spec TOML.",
    )
    return parser.parse_args()


def ensure(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def validate(spec: dict, spec_path: Path) -> list[str]:
    errors: list[str] = []

    ensure(isinstance(spec.get("version"), int), errors, "missing/invalid `version` (int)")
    ensure(spec.get("name"), errors, "missing `name`")
    ensure(spec.get("artifact_root"), errors, "missing `artifact_root`")

    suite = spec.get("suite")
    ensure(isinstance(suite, dict), errors, "missing `[suite]` table")
    if isinstance(suite, dict):
        ensure(bool(suite.get("default_backend")), errors, "missing suite.default_backend")
        ensure(bool(suite.get("oracle_backend")), errors, "missing suite.oracle_backend")
        ensure(
            isinstance(suite.get("timeout_seconds"), int),
            errors,
            "missing/invalid suite.timeout_seconds (int)",
        )

    fidelity = spec.get("cultural_fidelity")
    ensure(isinstance(fidelity, dict), errors, "missing `[cultural_fidelity]` table")
    if isinstance(fidelity, dict):
        terms = fidelity.get("forbidden_terms")
        ensure(isinstance(terms, list) and len(terms) > 0, errors, "cultural_fidelity.forbidden_terms must be a non-empty list")

    cases = spec.get("case")
    ensure(isinstance(cases, list) and len(cases) > 0, errors, "expected at least one `[[case]]` entry")
    seen_ids: set[str] = set()
    if isinstance(cases, list):
        for idx, case in enumerate(cases, start=1):
            prefix = f"case[{idx}]"
            if not isinstance(case, dict):
                errors.append(f"{prefix} must be a table")
                continue
            case_id = case.get("id")
            ensure(isinstance(case_id, str) and case_id.strip(), errors, f"{prefix}.id must be a non-empty string")
            if isinstance(case_id, str):
                if case_id in seen_ids:
                    errors.append(f"duplicate case id: {case_id}")
                seen_ids.add(case_id)
            command = case.get("command")
            ensure(command in ALLOWED_COMMANDS, errors, f"{prefix}.command must be one of {sorted(ALLOWED_COMMANDS)}")
            args = case.get("args")
            ensure(isinstance(args, list), errors, f"{prefix}.args must be a list")
            exit_code = case.get("expected_exit_code")
            ensure(isinstance(exit_code, int), errors, f"{prefix}.expected_exit_code must be an int")

    if errors:
        errors.insert(0, f"invalid parity spec: {spec_path}")
    return errors


def main() -> int:
    args = parse_args()
    spec_path = args.spec.resolve()
    if not spec_path.exists():
        print(f"PARITY_SPEC_LINT FAIL path_missing={spec_path}")
        return 1
    try:
        payload = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        print(f"PARITY_SPEC_LINT FAIL parse_error={exc}")
        return 1

    errors = validate(payload, spec_path)
    if errors:
        print(f"PARITY_SPEC_LINT FAIL errors={len(errors) - 1}")
        for error in errors:
            print(f" - {error}")
        return 1

    case_count = len(payload.get("case", []))
    print(f"PARITY_SPEC_LINT PASS cases={case_count} spec={spec_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
