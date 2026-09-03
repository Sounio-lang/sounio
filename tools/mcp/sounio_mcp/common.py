"""Shared subprocess, path, and environment helpers for the Sounio MCP tools."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SOUC_RESOLVER = REPO_ROOT / "scripts" / "lib" / "resolve_souc.sh"
SENSITIVE_ENV_PATTERN = re.compile(
    r"(TOKEN|KEY|SECRET|PASSWORD|PASSWD|AUTH|CREDENTIAL|COOKIE|SESSION|SSH|"
    r"GITHUB|OPENAI|ANTHROPIC|AWS_|GOOGLE_|AZURE_|PROXY)",
    re.IGNORECASE,
)


class SounioToolError(ValueError):
    """Raised when an MCP tool receives an unsafe or unsupported request."""


@dataclass(frozen=True)
class CommandResult:
    """Completed subprocess output."""

    args: tuple[str, ...]
    returncode: int | None
    stdout: str
    stderr: str
    duration_ms: int
    timed_out: bool = False


def _within_repo(path: Path) -> bool:
    try:
        path.relative_to(REPO_ROOT)
    except ValueError:
        return False
    return True


def workspace_path(value: str, *, must_exist: bool = True) -> Path:
    """Resolve a user-supplied path inside the checked-out Sounio workspace."""

    if not value or "\x00" in value:
        raise SounioToolError("path must be a non-empty workspace path")
    if len(value) > 4096:
        raise SounioToolError("path is too long")
    raw = Path(value)
    candidate = raw if raw.is_absolute() else REPO_ROOT / raw
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise SounioToolError(f"could not resolve path: {exc}") from exc
    if not _within_repo(resolved):
        raise SounioToolError(f"path escapes the Sounio workspace: {value}")
    if must_exist and not resolved.exists():
        raise SounioToolError(f"path does not exist: {value}")
    return resolved


def sio_source_path(value: str) -> Path:
    """Resolve a Sounio source path and require a `.sio` suffix."""

    path = workspace_path(value)
    if path.suffix != ".sio":
        raise SounioToolError(f"expected a .sio source file, got: {value}")
    if not path.is_file():
        raise SounioToolError(f"source path is not a file: {value}")
    return path


def display_path(path: Path) -> str:
    """Return a compact repo-relative path when possible."""

    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def resolve_souc() -> Path:
    """Resolve the compiler through the repo's canonical `resolve_souc.sh` helper."""

    if not SOUC_RESOLVER.exists():
        fallback = REPO_ROOT / "bin" / "souc"
        if fallback.exists():
            return fallback
        raise SounioToolError("scripts/lib/resolve_souc.sh and bin/souc are unavailable")

    script = (
        f"cd {shlex.quote(str(REPO_ROOT))} "
        "&& source scripts/lib/resolve_souc.sh "
        "&& sounio_require_souc "
        '&& printf "%s\\n" "$SOUC_BIN"'
    )
    proc = subprocess.run(
        ["bash", "-lc", script],
        cwd=str(REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise SounioToolError(
            "failed to resolve souc through scripts/lib/resolve_souc.sh: "
            f"{proc.stderr.strip()}"
        )
    output = proc.stdout.strip()
    path = Path(output).resolve(strict=False)
    if not path.exists():
        raise SounioToolError(f"resolved souc path does not exist: {path}")
    return path


def build_env(overrides: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build a subprocess environment without propagating secrets or network proxies."""

    env = {k: v for k, v in os.environ.items() if not SENSITIVE_ENV_PATTERN.search(k)}
    env["SOUNIO_STDLIB_PATH"] = str(REPO_ROOT / "stdlib")
    env["SOUNIO_REPO_HARD_NO_RUST"] = "1"
    if overrides:
        for key, value in overrides.items():
            if SENSITIVE_ENV_PATTERN.search(key):
                raise SounioToolError(f"refusing to pass sensitive environment key: {key}")
            env[key] = str(value)
    return env


async def run_command(
    args: Sequence[str | Path],
    *,
    timeout_sec: int = 30,
    env: Mapping[str, str] | None = None,
    cwd: Path = REPO_ROOT,
) -> CommandResult:
    """Run a compiler command with bounded time and captured output."""

    command = tuple(str(arg) for arg in args)
    start = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        env=dict(env) if env is not None else build_env(),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
        timed_out = False
    except TimeoutError:
        proc.kill()
        stdout_b, stderr_b = await proc.communicate()
        timed_out = True
    duration_ms = int((time.perf_counter() - start) * 1000)
    return CommandResult(
        args=command,
        returncode=proc.returncode,
        stdout=stdout_b.decode("utf-8", errors="replace"),
        stderr=stderr_b.decode("utf-8", errors="replace"),
        duration_ms=duration_ms,
        timed_out=timed_out,
    )
