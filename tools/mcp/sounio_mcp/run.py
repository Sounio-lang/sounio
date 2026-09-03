"""Implementation of the `sounio_run` MCP tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .check import sounio_check
from .common import SounioToolError, build_env, resolve_souc, run_command, sio_source_path


async def sounio_run(
    source_path: str,
    args: list[str] | None = None,
    timeout_sec: int = 30,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Compile and run a Sounio source file, returning captured process output."""

    checked = await sounio_check(source_path)
    if not checked.get("valid"):
        return {
            "status": "compile_error",
            "stdout": str(checked.get("stdout", "")),
            "stderr": str(checked.get("stderr", "")),
            "exit_code": None,
            "duration_ms": checked.get("duration_ms", 0),
            "diagnostics": checked.get("diagnostics", []),
        }

    try:
        source = sio_source_path(source_path)
        souc = resolve_souc()
        proc_env = build_env(env)
    except SounioToolError as exc:
        return {
            "status": "runtime_error",
            "stdout": "",
            "stderr": str(exc),
            "exit_code": None,
            "duration_ms": checked.get("duration_ms", 0),
            "diagnostics": [],
        }

    bounded_timeout = max(1, min(int(timeout_sec), 300))
    program_args = [str(arg) for arg in (args or [])]
    command: list[str | Path] = [souc, "run", source]
    if program_args:
        command.extend(["--", *program_args])
    result = await run_command(command, timeout_sec=bounded_timeout, env=proc_env)
    if result.timed_out:
        status = "timeout"
    elif result.returncode == 0:
        status = "ok"
    else:
        status = "runtime_error"
    return {
        "status": status,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.returncode,
        "duration_ms": int(checked.get("duration_ms", 0)) + result.duration_ms,
        "diagnostics": [],
    }
