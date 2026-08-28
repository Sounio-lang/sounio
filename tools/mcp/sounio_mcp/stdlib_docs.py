"""MCP resource support for Sounio standard-library documentation."""

from __future__ import annotations

import re
from pathlib import Path

from .common import REPO_ROOT, display_path

MODULE_RE = re.compile(r"^[A-Za-z0-9_./:-]+$")


def _candidate_paths(module: str) -> list[Path]:
    normalised = module.replace("::", "/").replace(".", "/").strip("/")
    base = REPO_ROOT / "stdlib" / normalised
    return [
        base / "README.md",
        base / "lib.sio",
        base / "mod.sio",
        REPO_ROOT / "stdlib" / f"{normalised}.sio",
    ]


def get_stdlib_doc(module: str) -> str:
    """Return Markdown documentation or source notes for a stdlib module."""

    if not module or not MODULE_RE.fullmatch(module) or ".." in module.split("/"):
        return "# Sounio stdlib resource\n\nInvalid module path."
    for path in _candidate_paths(module):
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(REPO_ROOT / "stdlib")
        except ValueError:
            continue
        if resolved.is_file():
            content = resolved.read_text(encoding="utf-8", errors="replace")
            if resolved.suffix == ".md":
                return content
            return (
                f"# stdlib/{module}\n\n"
                f"Source: `{display_path(resolved)}`\n\n"
                "```sio\n"
                f"{content[:12000]}\n"
                "```\n"
            )
    available = sorted(p.name for p in (REPO_ROOT / "stdlib").iterdir() if p.is_dir())[:80]
    return (
        f"# stdlib/{module}\n\n"
        "No README.md, lib.sio, mod.sio, or direct module source was found.\n\n"
        "Available top-level modules include:\n\n"
        + "\n".join(f"- `{name}`" for name in available)
    )
