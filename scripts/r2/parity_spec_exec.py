#!/usr/bin/env python3
"""Execute scripts/r2/parity-spec.toml against souc with minimal contracts."""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
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


@dataclass(frozen=True)
class StreamExpectation:
    expected_text: str | None
    source_label: str | None


@dataclass(frozen=True)
class InvocationResult:
    code: int
    stdout: str
    stderr: str
    timed_out: bool


def normalize_oracle_compare_tokens(suite: dict[str, Any]) -> list[str]:
    raw = suite.get("oracle_compare")
    if not isinstance(raw, list):
        return []
    normalized: list[str] = []
    for token in raw:
        if not isinstance(token, str):
            continue
        mapped = "exit_code" if token == "exit" else token
        if mapped in {"exit_code", "stdout", "stderr"} and mapped not in normalized:
            normalized.append(mapped)
    return normalized


def resolve_contract_path(root: Path, spec_path: Path, raw_path: str) -> Path:
    contract_path = Path(raw_path)
    if contract_path.is_absolute():
        return contract_path.resolve()
    from_root = (root / contract_path).resolve()
    from_spec_dir = (spec_path.parent / contract_path).resolve()
    if from_root.exists() or not from_spec_dir.exists():
        return from_root
    return from_spec_dir


def load_stream_expectation(
    *,
    case: dict[str, Any],
    case_id: str,
    stream: str,
    root: Path,
    spec_path: Path,
    parse_errors: list[str],
) -> StreamExpectation:
    inline_key = f"expected_{stream}"
    file_key = f"expected_{stream}_file"
    inline_value = case.get(inline_key)
    file_value = case.get(file_key)

    if inline_value is not None and not isinstance(inline_value, str):
        parse_errors.append(f"case[{case_id}] {inline_key} must be a string when present")
    if file_value is not None and not isinstance(file_value, str):
        parse_errors.append(f"case[{case_id}] {file_key} must be a string path when present")
    if inline_value is not None and file_value is not None:
        parse_errors.append(f"case[{case_id}] cannot set both {inline_key} and {file_key}")

    if parse_errors:
        return StreamExpectation(expected_text=None, source_label=None)
    if isinstance(inline_value, str):
        return StreamExpectation(
            expected_text=inline_value,
            source_label=f"case[{case_id}].{inline_key}",
        )
    if not isinstance(file_value, str):
        return StreamExpectation(expected_text=None, source_label=None)

    contract_path = resolve_contract_path(root, spec_path, file_value)
    if not contract_path.exists():
        parse_errors.append(
            f"case[{case_id}] missing {file_key} file: {contract_path}"
        )
        return StreamExpectation(expected_text=None, source_label=None)
    if contract_path.is_dir():
        parse_errors.append(
            f"case[{case_id}] {file_key} points to a directory: {contract_path}"
        )
        return StreamExpectation(expected_text=None, source_label=None)
    try:
        expected_text = contract_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover
        parse_errors.append(
            f"case[{case_id}] failed to read {file_key} {contract_path}: {exc}"
        )
        return StreamExpectation(expected_text=None, source_label=None)
    return StreamExpectation(
        expected_text=expected_text,
        source_label=f"{file_key}:{contract_path}",
    )


def run_invocation(
    *,
    invocation: list[str],
    root: Path,
    backend: str | None,
    timeout_seconds: int,
) -> InvocationResult:
    env = os.environ.copy()
    if isinstance(backend, str) and backend:
        env["SOUNIO_SELFHOST_PIPELINE"] = backend
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
        return InvocationResult(
            code=int(completed.returncode),
            stdout=completed.stdout,
            stderr=completed.stderr,
            timed_out=timed_out,
        )
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        return InvocationResult(
            code=124,
            stdout=coerce_text(exc.stdout),
            stderr=coerce_text(exc.stderr),
            timed_out=timed_out,
        )


def write_result_artifacts(
    *,
    artifact_root: Path,
    case_file_stem: str,
    suffix: str,
    result: InvocationResult,
) -> dict[str, Path]:
    file_stem = f"{case_file_stem}{suffix}"
    stdout_path = artifact_root / f"{file_stem}.stdout"
    stderr_path = artifact_root / f"{file_stem}.stderr"
    exit_path = artifact_root / f"{file_stem}.exit"
    timeout_path = artifact_root / f"{file_stem}.timeout"

    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    exit_path.write_text(f"{result.code}\n", encoding="utf-8")
    timeout_path.write_text(f"{int(result.timed_out)}\n", encoding="utf-8")

    return {
        "stdout": stdout_path,
        "stderr": stderr_path,
        "exit": exit_path,
        "timeout": timeout_path,
    }


def write_unified_diff(
    *,
    expected_text: str,
    actual_text: str,
    expected_label: str,
    actual_label: str,
    diff_path: Path,
) -> None:
    diff_text = "".join(
        difflib.unified_diff(
            expected_text.splitlines(keepends=True),
            actual_text.splitlines(keepends=True),
            fromfile=expected_label,
            tofile=actual_label,
            n=3,
        )
    )
    if not diff_text:
        diff_text = "(no diff generated)\n"
    elif not diff_text.endswith("\n"):
        diff_text += "\n"
    diff_path.write_text(diff_text, encoding="utf-8")


def evaluate_stream_contracts(
    *,
    case_id: str,
    case_file_stem: str,
    artifact_root: Path,
    label: str,
    actual: str,
    actual_artifact_path: Path,
    expected_exact: StreamExpectation,
    must_contain: list[str],
    must_exclude: list[str],
    failures: list[str],
) -> None:
    if expected_exact.expected_text is not None and actual != expected_exact.expected_text:
        diff_path = artifact_root / f"{case_file_stem}.{label}.expected.diff"
        expected_label = expected_exact.source_label or f"case[{case_id}].expected_{label}"
        write_unified_diff(
            expected_text=expected_exact.expected_text,
            actual_text=actual,
            expected_label=expected_label,
            actual_label=str(actual_artifact_path),
            diff_path=diff_path,
        )
        failures.append(
            f"{label} exact mismatch ({expected_label}) diff={diff_path} actual={actual_artifact_path}"
        )
    for token in must_contain:
        if token not in actual:
            failures.append(f"{label} missing expected substring: {token!r}")
    for token in must_exclude:
        if token in actual:
            failures.append(f"{label} contains forbidden substring: {token!r}")


def evaluate_oracle_contracts(
    *,
    case_file_stem: str,
    artifact_root: Path,
    oracle_backend: str,
    oracle_compare: list[str],
    primary: InvocationResult,
    primary_artifacts: dict[str, Path],
    oracle: InvocationResult,
    oracle_artifacts: dict[str, Path],
    failures: list[str],
) -> None:
    for token in oracle_compare:
        if token == "exit_code":
            if primary.code != oracle.code:
                failures.append(
                    "oracle exit_code mismatch "
                    f"(backend={oracle_backend} primary={primary.code} oracle={oracle.code} "
                    f"artifacts={primary_artifacts['exit']} vs {oracle_artifacts['exit']})"
                )
            continue

        primary_text = primary.stdout if token == "stdout" else primary.stderr
        oracle_text = oracle.stdout if token == "stdout" else oracle.stderr
        if primary_text == oracle_text:
            continue

        diff_path = artifact_root / f"{case_file_stem}.oracle.{token}.diff"
        write_unified_diff(
            expected_text=primary_text,
            actual_text=oracle_text,
            expected_label=str(primary_artifacts[token]),
            actual_label=str(oracle_artifacts[token]),
            diff_path=diff_path,
        )
        failures.append(
            f"oracle {token} mismatch (backend={oracle_backend}) diff={diff_path} "
            f"primary={primary_artifacts[token]} oracle={oracle_artifacts[token]}"
        )


def run_case(
    *,
    case: dict[str, Any],
    case_index: int,
    suite: dict[str, Any],
    spec_path: Path,
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
    expected_stdout = load_stream_expectation(
        case=case,
        case_id=case_id,
        stream="stdout",
        root=root,
        spec_path=spec_path,
        parse_errors=parse_errors,
    )
    expected_stderr = load_stream_expectation(
        case=case,
        case_id=case_id,
        stream="stderr",
        root=root,
        spec_path=spec_path,
        parse_errors=parse_errors,
    )
    if parse_errors:
        return False, parse_errors

    timeout_seconds = int(suite.get("timeout_seconds", 120))
    default_backend = suite.get("default_backend")
    oracle_backend = suite.get("oracle_backend")
    oracle_compare = normalize_oracle_compare_tokens(suite)

    invocation = build_invocation(souc_bin, command, args)
    print(f"PARITY_CASE RUN id={case_id} argv={shlex.join(invocation)}")
    primary = run_invocation(
        invocation=invocation,
        root=root,
        backend=default_backend if isinstance(default_backend, str) else None,
        timeout_seconds=timeout_seconds,
    )
    if primary.timed_out:
        failures.append(f"timed out after {timeout_seconds}s")

    case_file_stem = sanitize_case_id(case_id)
    primary_artifacts = write_result_artifacts(
        artifact_root=artifact_root,
        case_file_stem=case_file_stem,
        suffix="",
        result=primary,
    )

    if primary.code != expected_exit:
        failures.append(f"exit mismatch (expected={expected_exit} actual={primary.code})")

    evaluate_stream_contracts(
        case_id=case_id,
        case_file_stem=case_file_stem,
        artifact_root=artifact_root,
        label="stdout",
        actual=primary.stdout,
        actual_artifact_path=primary_artifacts["stdout"],
        expected_exact=expected_stdout,
        must_contain=stdout_contains,
        must_exclude=stdout_excludes,
        failures=failures,
    )
    evaluate_stream_contracts(
        case_id=case_id,
        case_file_stem=case_file_stem,
        artifact_root=artifact_root,
        label="stderr",
        actual=primary.stderr,
        actual_artifact_path=primary_artifacts["stderr"],
        expected_exact=expected_stderr,
        must_contain=stderr_contains,
        must_exclude=stderr_excludes,
        failures=failures,
    )

    if oracle_compare:
        if not isinstance(oracle_backend, str) or not oracle_backend:
            failures.append("oracle parity configured but suite.oracle_backend is missing/invalid")
        else:
            print(
                "PARITY_CASE ORACLE "
                f"id={case_id} backend={oracle_backend} compare={','.join(oracle_compare)}"
            )
            oracle = run_invocation(
                invocation=invocation,
                root=root,
                backend=oracle_backend,
                timeout_seconds=timeout_seconds,
            )
            oracle_artifacts = write_result_artifacts(
                artifact_root=artifact_root,
                case_file_stem=case_file_stem,
                suffix=".oracle",
                result=oracle,
            )
            if oracle.timed_out:
                failures.append(
                    f"oracle run timed out after {timeout_seconds}s (backend={oracle_backend})"
                )
            evaluate_oracle_contracts(
                case_file_stem=case_file_stem,
                artifact_root=artifact_root,
                oracle_backend=oracle_backend,
                oracle_compare=oracle_compare,
                primary=primary,
                primary_artifacts=primary_artifacts,
                oracle=oracle,
                oracle_artifacts=oracle_artifacts,
                failures=failures,
            )

    if failures:
        print(f"PARITY_CASE FAIL id={case_id} rc={primary.code} timeout={int(primary.timed_out)}")
        for failure in failures:
            print(f" - {failure}")
        return False, failures

    print(f"PARITY_CASE PASS id={case_id} rc={primary.code}")
    return True, []


def execute_spec(spec: dict[str, Any], *, spec_path: Path, root: Path, souc_bin: Path) -> int:
    suite = spec.get("suite", {})
    cases = spec.get("case", [])
    artifact_root_value = spec.get("artifact_root", "artifacts/r2")
    artifact_root = artifact_root_value if isinstance(artifact_root_value, str) else "artifacts/r2"
    artifact_dir = (root / artifact_root).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    suite_dict = suite if isinstance(suite, dict) else {}
    backend = suite_dict.get("default_backend", "inherit")
    oracle_backend = suite_dict.get("oracle_backend", "inherit")
    oracle_compare = normalize_oracle_compare_tokens(suite_dict)
    timeout_seconds = suite_dict.get("timeout_seconds", 120)
    oracle_compare_label = ",".join(oracle_compare) if oracle_compare else "disabled"
    print(
        "PARITY_SPEC_EXEC START "
        f"spec={spec_path} cases={len(cases)} souc={souc_bin} "
        f"backend={backend} oracle_backend={oracle_backend} "
        f"oracle_compare={oracle_compare_label} timeout_seconds={timeout_seconds} "
        f"artifacts={artifact_dir}"
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
            suite=suite_dict,
            spec_path=spec_path,
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
                    "import os",
                    "import sys",
                    "",
                    "def main() -> int:",
                    "    args = sys.argv[1:]",
                    "    backend = os.environ.get('SOUNIO_SELFHOST_PIPELINE', 'inherit')",
                    "    if args == ['--help']:",
                    "        sys.stdout.write('Sounio compiler\\nUSAGE\\nCOMMANDS\\n')",
                    "        return 0",
                    "    if args == ['--version']:",
                    "        sys.stdout.write('souc 0.0.0-selftest\\n')",
                    "        return 0",
                    "    if len(args) >= 2 and args[0] == 'run':",
                    "        program = args[1]",
                    "        if program.endswith('oracle_mismatch.sio') and backend == 'rust':",
                    "            sys.stdout.write('99\\n')",
                    "        else:",
                    "            sys.stdout.write('42\\n')",
                    "        if backend == 'rust':",
                    "            sys.stderr.write('SELFHOST=oracle backend=rust\\n')",
                    "        else:",
                    "            sys.stderr.write('SELFHOST=driver_output\\n')",
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
        (temp_root / "oracle_mismatch.sio").write_text("fn main() -> i64 { 42 }\n", encoding="utf-8")
        golden_dir = temp_root / "golden"
        golden_dir.mkdir(parents=True, exist_ok=True)
        (golden_dir / "run_smoke.stdout").write_text("42\n", encoding="utf-8")
        (golden_dir / "run_smoke.stderr").write_text("SELFHOST=driver_output\n", encoding="utf-8")

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
                    'oracle_compare = ["exit_code", "stdout"]',
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
                    'expected_stdout_file = "golden/run_smoke.stdout"',
                    'expected_stderr_file = "golden/run_smoke.stderr"',
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
                    'oracle_compare = ["exit_code", "stdout"]',
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
                    'id = "oracle_mismatch_smoke"',
                    'command = "run"',
                    'args = ["oracle_mismatch.sio"]',
                    "expected_exit_code = 0",
                    'expected_stdout = "42\\n"',
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
        if "oracle stdout mismatch" not in fail_run.stdout:
            print("SELF_TEST FAIL expected oracle mismatch failure text", file=sys.stderr)
            print(fail_run.stdout, file=sys.stderr)
            print(fail_run.stderr, file=sys.stderr)
            return 1

        pass_oracle_stdout = temp_root / "artifacts" / "pass" / "run_smoke.oracle.stdout"
        pass_oracle_exit = temp_root / "artifacts" / "pass" / "run_smoke.oracle.exit"
        fail_oracle_diff = (
            temp_root
            / "artifacts"
            / "fail"
            / "oracle_mismatch_smoke.oracle.stdout.diff"
        )
        if not pass_oracle_stdout.exists() or not pass_oracle_exit.exists():
            print("SELF_TEST FAIL expected oracle artifacts for pass case", file=sys.stderr)
            return 1
        if not fail_oracle_diff.exists():
            print("SELF_TEST FAIL expected oracle stdout diff artifact for fail case", file=sys.stderr)
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
