#!/usr/bin/env python3
"""Execute scripts/r2/parity-spec.toml against souc with minimal contracts."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(f"error: tomllib unavailable: {exc}") from exc

from parity_spec_lint import validate


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = REPO_ROOT / "scripts" / "r2" / "parity-spec.toml"


def default_souc_bin() -> Path:
    target_dir = os.environ.get("CARGO_TARGET_DIR")
    if target_dir:
        return Path(target_dir).resolve() / "debug" / "souc"
    return REPO_ROOT / "target" / "debug" / "souc"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run R2 parity spec cases against souc and enforce stdout/stderr/exit contracts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root used for resolving relative paths.",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC_PATH,
        help="Path to parity spec TOML.",
    )
    parser.add_argument(
        "--souc-bin",
        type=Path,
        default=default_souc_bin(),
        help="Path to souc executable.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic smoke self-test for invocation and mismatch paths.",
    )
    return parser.parse_args(argv)


def coerce_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def sanitize_case_id(case_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
    return cleaned or "case"


def parse_string_list(case: dict[str, Any], key: str, errors: list[str], case_id: str) -> list[str]:
    value = case.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        errors.append(f"case[{case_id}] {key} must be a list of strings")
        return []
    return value


def build_invocation(souc_bin: Path, command: str, args: list[str]) -> list[str]:
    if command == "help":
        if args:
            return [str(souc_bin), *args]
        return [str(souc_bin), "--help"]
    if command == "version":
        if args:
            return [str(souc_bin), *args]
        return [str(souc_bin), "--version"]
    return [str(souc_bin), command, *args]


def evaluate_stream_contracts(
    *,
    label: str,
    actual: str,
    expected_exact: str | None,
    must_contain: list[str],
    must_exclude: list[str],
    failures: list[str],
) -> None:
    if expected_exact is not None and actual != expected_exact:
        failures.append(
            f"{label} exact mismatch (expected={expected_exact!r} actual={actual!r})"
        )
    for token in must_contain:
        if token not in actual:
            failures.append(f"{label} missing expected substring: {token!r}")
    for token in must_exclude:
        if token in actual:
            failures.append(f"{label} contains forbidden substring: {token!r}")


def run_case(
    *,
    case: dict[str, Any],
    case_index: int,
    suite: dict[str, Any],
    root: Path,
    souc_bin: Path,
    artifact_root: Path,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    case_id = case.get("id", f"case_{case_index}")
    command = case.get("command")
    args = case.get("args")
    expected_exit = case.get("expected_exit_code")

    if not isinstance(case_id, str) or not case_id.strip():
        return False, [f"case[{case_index}] invalid id"]
    if not isinstance(command, str):
        return False, [f"case[{case_id}] missing/invalid command"]
    if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
        return False, [f"case[{case_id}] args must be a list of strings"]
    if not isinstance(expected_exit, int):
        return False, [f"case[{case_id}] missing/invalid expected_exit_code"]

    parse_errors: list[str] = []
    stdout_contains = parse_string_list(case, "expected_stdout_contains", parse_errors, case_id)
    stdout_excludes = parse_string_list(case, "expected_stdout_excludes", parse_errors, case_id)
    stderr_contains = parse_string_list(case, "expected_stderr_contains", parse_errors, case_id)
    stderr_excludes = parse_string_list(case, "expected_stderr_excludes", parse_errors, case_id)
    if parse_errors:
        return False, parse_errors

    expected_stdout = case.get("expected_stdout")
    expected_stderr = case.get("expected_stderr")
    if expected_stdout is not None and not isinstance(expected_stdout, str):
        return False, [f"case[{case_id}] expected_stdout must be a string when present"]
    if expected_stderr is not None and not isinstance(expected_stderr, str):
        return False, [f"case[{case_id}] expected_stderr must be a string when present"]

    timeout_seconds = int(suite.get("timeout_seconds", 120))
    backend = suite.get("default_backend")

    env = os.environ.copy()
    if isinstance(backend, str) and backend:
        env["SOUNIO_SELFHOST_PIPELINE"] = backend

    invocation = build_invocation(souc_bin, command, args)
    print(f"PARITY_CASE RUN id={case_id} argv={shlex.join(invocation)}")

    timed_out = False
    try:
        completed = subprocess.run(
            invocation,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        code = int(completed.returncode)
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        code = 124
        stdout = coerce_text(exc.stdout)
        stderr = coerce_text(exc.stderr)
        failures.append(f"timed out after {timeout_seconds}s")

    case_file_stem = sanitize_case_id(case_id)
    (artifact_root / f"{case_file_stem}.stdout").write_text(stdout, encoding="utf-8")
    (artifact_root / f"{case_file_stem}.stderr").write_text(stderr, encoding="utf-8")
    (artifact_root / f"{case_file_stem}.exit").write_text(f"{code}\n", encoding="utf-8")

    if code != expected_exit:
        failures.append(f"exit mismatch (expected={expected_exit} actual={code})")

    evaluate_stream_contracts(
        label="stdout",
        actual=stdout,
        expected_exact=expected_stdout,
        must_contain=stdout_contains,
        must_exclude=stdout_excludes,
        failures=failures,
    )
    evaluate_stream_contracts(
        label="stderr",
        actual=stderr,
        expected_exact=expected_stderr,
        must_contain=stderr_contains,
        must_exclude=stderr_excludes,
        failures=failures,
    )

    if failures:
        print(f"PARITY_CASE FAIL id={case_id} rc={code} timeout={int(timed_out)}")
        for failure in failures:
            print(f" - {failure}")
        return False, failures

    print(f"PARITY_CASE PASS id={case_id} rc={code}")
    return True, []


def execute_spec(spec: dict[str, Any], *, spec_path: Path, root: Path, souc_bin: Path) -> int:
    suite = spec.get("suite", {})
    cases = spec.get("case", [])
    artifact_root_value = spec.get("artifact_root", "artifacts/r2")
    artifact_root = artifact_root_value if isinstance(artifact_root_value, str) else "artifacts/r2"
    artifact_dir = (root / artifact_root).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    backend = suite.get("default_backend", "inherit")
    timeout_seconds = suite.get("timeout_seconds", 120)
    print(
        "PARITY_SPEC_EXEC START "
        f"spec={spec_path} cases={len(cases)} souc={souc_bin} "
        f"backend={backend} timeout_seconds={timeout_seconds} artifacts={artifact_dir}"
    )

    pass_count = 0
    fail_count = 0
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            fail_count += 1
            print(f"PARITY_CASE FAIL id=case_{index} rc=0 timeout=0")
            print(f" - case[{index}] must be a table")
            continue
        ok, _ = run_case(
            case=case,
            case_index=index,
            suite=suite if isinstance(suite, dict) else {},
            root=root,
            souc_bin=souc_bin,
            artifact_root=artifact_dir,
        )
        if ok:
            pass_count += 1
        else:
            fail_count += 1

    if fail_count:
        print(
            f"PARITY_SPEC_EXEC RESULT FAIL pass={pass_count} fail={fail_count} artifacts={artifact_dir}"
        )
        return 1

    print(f"PARITY_SPEC_EXEC RESULT PASS pass={pass_count} fail=0 artifacts={artifact_dir}")
    return 0


def run_self_test() -> int:
    script_path = Path(__file__).resolve()
    with tempfile.TemporaryDirectory(prefix="sounio-parity-selftest-") as tmp:
        temp_root = Path(tmp)
        fake_souc = temp_root / "fake_souc.py"
        fake_souc.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env python3",
                    "import sys",
                    "",
                    "def main() -> int:",
                    "    args = sys.argv[1:]",
                    "    if args == ['--help']:",
                    "        sys.stdout.write('Sounio compiler\\nUSAGE\\nCOMMANDS\\n')",
                    "        return 0",
                    "    if args == ['--version']:",
                    "        sys.stdout.write('souc 0.0.0-selftest\\n')",
                    "        return 0",
                    "    if len(args) >= 2 and args[0] == 'run':",
                    "        sys.stdout.write('42\\n')",
                    "        sys.stderr.write('SELFHOST=driver_output\\n')",
                    "        return 0",
                    "    if len(args) >= 2 and args[0] == 'check':",
                    "        return 0",
                    "    sys.stderr.write('unexpected args: ' + ' '.join(args) + '\\n')",
                    "    return 2",
                    "",
                    "if __name__ == '__main__':",
                    "    raise SystemExit(main())",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        fake_souc.chmod(0o755)

        (temp_root / "dummy.sio").write_text("fn main() -> i64 { 42 }\n", encoding="utf-8")

        pass_spec = temp_root / "pass-spec.toml"
        pass_spec.write_text(
            "\n".join(
                [
                    'version = 1',
                    'name = "self-test-pass"',
                    'artifact_root = "artifacts/pass"',
                    "",
                    "[suite]",
                    'default_backend = "driver"',
                    'oracle_backend = "rust"',
                    "timeout_seconds = 5",
                    "",
                    "[cultural_fidelity]",
                    "enabled = true",
                    'forbidden_terms = ["cargo"]',
                    "scan_logs = true",
                    "scan_help = true",
                    "scan_errors = true",
                    "",
                    "[[case]]",
                    'id = "run_smoke"',
                    'command = "run"',
                    'args = ["dummy.sio"]',
                    "expected_exit_code = 0",
                    'expected_stdout = "42\\n"',
                    'expected_stderr_contains = ["SELFHOST=driver_output"]',
                    "",
                    "[[case]]",
                    'id = "help_smoke"',
                    'command = "help"',
                    'args = ["--help"]',
                    "expected_exit_code = 0",
                    'expected_stdout_contains = ["Sounio compiler", "USAGE", "COMMANDS"]',
                    "",
                ]
            ),
            encoding="utf-8",
        )

        fail_spec = temp_root / "fail-spec.toml"
        fail_spec.write_text(
            "\n".join(
                [
                    'version = 1',
                    'name = "self-test-fail"',
                    'artifact_root = "artifacts/fail"',
                    "",
                    "[suite]",
                    'default_backend = "driver"',
                    'oracle_backend = "rust"',
                    "timeout_seconds = 5",
                    "",
                    "[cultural_fidelity]",
                    "enabled = true",
                    'forbidden_terms = ["cargo"]',
                    "scan_logs = true",
                    "scan_help = true",
                    "scan_errors = true",
                    "",
                    "[[case]]",
                    'id = "mismatch_smoke"',
                    'command = "run"',
                    'args = ["dummy.sio"]',
                    "expected_exit_code = 0",
                    'expected_stdout = "41\\n"',
                    "",
                ]
            ),
            encoding="utf-8",
        )

        base_cmd = [sys.executable, str(script_path), "--root", str(temp_root), "--souc-bin", str(fake_souc)]

        pass_run = subprocess.run(
            [*base_cmd, "--spec", str(pass_spec)],
            cwd=str(temp_root),
            capture_output=True,
            text=True,
            check=False,
        )
        fail_run = subprocess.run(
            [*base_cmd, "--spec", str(fail_spec)],
            cwd=str(temp_root),
            capture_output=True,
            text=True,
            check=False,
        )

        if pass_run.returncode != 0:
            print("SELF_TEST FAIL expected pass spec to return 0", file=sys.stderr)
            print(pass_run.stdout, file=sys.stderr)
            print(pass_run.stderr, file=sys.stderr)
            return 1
        if fail_run.returncode != 1:
            print("SELF_TEST FAIL expected fail spec to return 1", file=sys.stderr)
            print(fail_run.stdout, file=sys.stderr)
            print(fail_run.stderr, file=sys.stderr)
            return 1

        print("SELF_TEST PASS parity_spec_exec")
        return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.self_test:
        return run_self_test()

    root = args.root.resolve()
    spec_path = args.spec.resolve()
    souc_bin = args.souc_bin.resolve()

    if not spec_path.exists():
        print(f"PARITY_SPEC_EXEC FAIL path_missing={spec_path}")
        return 2
    if not souc_bin.exists():
        print(f"PARITY_SPEC_EXEC FAIL souc_missing={souc_bin}")
        return 2
    if not os.access(souc_bin, os.X_OK):
        print(f"PARITY_SPEC_EXEC FAIL souc_not_executable={souc_bin}")
        return 2

    try:
        payload = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover
        print(f"PARITY_SPEC_EXEC FAIL parse_error={exc}")
        return 2

    errors = validate(payload, spec_path)
    if errors:
        print(f"PARITY_SPEC_EXEC FAIL lint_errors={len(errors) - 1}")
        for error in errors:
            print(f" - {error}")
        return 2

    return execute_spec(payload, spec_path=spec_path, root=root, souc_bin=souc_bin)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
