#!/usr/bin/env python3
"""Stable project hook launcher for the shared Sounio coordination runtime."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


CLIENT_PROTOCOL = "3"


def git_path(cwd: Path, *args: str) -> Path | None:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    path = Path(result.stdout.strip())
    if not path.is_absolute():
        path = cwd / path
    return path.resolve()


def manifest_protocol(manifest: Path) -> str | None:
    try:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.startswith("protocol_version="):
                return line.partition("=")[2]
    except OSError:
        return None
    return None


def main() -> int:
    source_root = git_path(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel")
    worktree = git_path(Path.cwd(), "rev-parse", "--show-toplevel") or source_root
    if source_root is None or worktree is None:
        return 0
    common = git_path(worktree, "rev-parse", "--git-common-dir")
    if common is None:
        return 0

    runtime_root = Path(
        os.environ.get("SOUNIO_COORD_RUNTIME_DIR", str(common / "sounio-coord-runtime"))
    )
    local = source_root / "scripts" / "dev" / "sounio_coord_agent_hook_runtime.py"
    selected = local
    current = runtime_root / "current"
    if (
        os.environ.get("SOUNIO_COORD_RUNTIME_MODE", "shared") != "local"
        and (current.exists() or current.is_symlink())
    ):
        try:
            resolved = current.resolve(strict=True)
        except OSError:
            sys.stderr.write(
                f"sounio coordination hook found a broken runtime link: {current}\n"
            )
            return 2
        selected = resolved / "hooks" / "sounio_coord_agent_hook_runtime.py"
        protocol = manifest_protocol(resolved / "manifest")
        if protocol != CLIENT_PROTOCOL:
            sys.stderr.write(
                "sounio coordination hook refused incompatible shared runtime: "
                f"client_protocol={CLIENT_PROTOCOL} runtime_protocol={protocol or 'missing'}\n"
            )
            return 2
    if not selected.is_file():
        sys.stderr.write(f"sounio coordination hook runtime missing: {selected}\n")
        return 2
    os.execv(sys.executable, [sys.executable, str(selected), *sys.argv[1:]])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
