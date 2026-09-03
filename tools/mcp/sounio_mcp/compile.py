"""Implementation of the `sounio_compile` MCP tool."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

from .check import sounio_check
from .common import SounioToolError, resolve_souc, run_command, sio_source_path
from .diagnostics import diagnostics_from_text

TARGET_FLAGS = {
    "elf": [],
    "macho": ["--target", "x86_64-macos"],
    "ptx": ["--target", "ptx"],
    "metal": ["--target", "metal"],
    "wasm": ["--target", "wasm32-wasi"],
}
TARGET_EXT = {
    "elf": "elf",
    "macho": "macho",
    "ptx": "ptx",
    "metal": "metal",
    "wasm": "wasm",
}
OPTIMIZATIONS = {"debug", "default", "release"}


def _build_output_path(source: Path, target: str) -> Path:
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:12]
    out_dir = Path(tempfile.mkdtemp(prefix=f"sounio-mcp-{digest}-"))
    return out_dir / f"{source.stem}.{TARGET_EXT[target]}"


async def sounio_compile(
    source_path: str,
    target: str = "elf",
    optimization: str = "default",
) -> dict[str, Any]:
    """Compile a Sounio source file to a local artefact."""

    started_check = await sounio_check(source_path)
    warnings: list[str] = []
    if target not in TARGET_FLAGS:
        return {
            "status": "error",
            "output_path": None,
            "stdout": "",
            "stderr": f"unsupported target: {target}",
            "diagnostics": [],
            "duration_ms": started_check.get("duration_ms", 0),
            "warnings": [f"target must be one of: {', '.join(sorted(TARGET_FLAGS))}"],
        }
    if optimization not in OPTIMIZATIONS:
        return {
            "status": "error",
            "output_path": None,
            "stdout": "",
            "stderr": f"unsupported optimization: {optimization}",
            "diagnostics": [],
            "duration_ms": started_check.get("duration_ms", 0),
            "warnings": [f"optimization must be one of: {', '.join(sorted(OPTIMIZATIONS))}"],
        }
    if optimization != "default":
        warnings.append(
            "The checked self-hosted launcher does not yet expose optimisation flags; "
            f"accepted '{optimization}' as provenance metadata only."
        )
    if not started_check.get("valid"):
        return {
            "status": "error",
            "output_path": None,
            "stdout": str(started_check.get("stdout", "")),
            "stderr": str(started_check.get("stderr", "")),
            "diagnostics": started_check.get("diagnostics", []),
            "duration_ms": started_check.get("duration_ms", 0),
            "warnings": warnings,
        }

    try:
        source = sio_source_path(source_path)
        souc = resolve_souc()
    except SounioToolError as exc:
        return {
            "status": "error",
            "output_path": None,
            "stdout": "",
            "stderr": str(exc),
            "diagnostics": [],
            "duration_ms": started_check.get("duration_ms", 0),
            "warnings": warnings,
        }

    output_path = _build_output_path(source, target)
    args: list[str | Path] = [souc, "compile", source, "-o", output_path, *TARGET_FLAGS[target]]
    result = await run_command(args, timeout_sec=60)
    diagnostics = diagnostics_from_text(result.stdout + "\n" + result.stderr)
    ok = output_path.exists() and output_path.stat().st_size > 0 and not result.timed_out
    return {
        "status": "ok" if ok else "error",
        "output_path": str(output_path) if ok else None,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "diagnostics": diagnostics,
        "duration_ms": int(started_check.get("duration_ms", 0)) + result.duration_ms,
        "warnings": warnings,
        "returncode": result.returncode,
    }
