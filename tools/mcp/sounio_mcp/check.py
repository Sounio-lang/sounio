"""Implementation of the `sounio_check` MCP tool."""

from __future__ import annotations

from typing import Any

from .common import SounioToolError, resolve_souc, run_command, sio_source_path
from .diagnostics import diagnostics_from_payload, shared_diagnostic_envelope


async def sounio_check(source_path: str) -> dict[str, Any]:
    """Type-check a Sounio source file and return structured diagnostics."""

    try:
        source = sio_source_path(source_path)
        souc = resolve_souc()
    except SounioToolError as exc:
        return {
            "valid": False,
            "diagnostics": [
                {
                    "severity": "error",
                    "code": "E001",
                    "message": str(exc),
                    "line": 1,
                    "column": 1,
                    "span": {"start": [1, 1], "end": [1, 1]},
                    "related": [],
                    "source": "sounio:mcp",
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 0},
                    },
                    "lsp_severity": 1,
                    "confidence": None,
                }
            ],
            "duration_ms": 0,
            "stdout": "",
            "stderr": "",
        }

    result = await run_command([souc, "check", source, "--json"], timeout_sec=30)
    diagnostics = diagnostics_from_payload(result.stdout, result.stderr)
    if result.timed_out and not diagnostics:
        diagnostics = [
            {
                "severity": "error",
                "code": "E001",
                "message": "souc check timed out",
                "line": 1,
                "column": 1,
                "span": {"start": [1, 1], "end": [1, 1]},
                "related": [],
                "source": "sounio:type",
                "range": {
                    "start": {"line": 0, "character": 0},
                    "end": {"line": 0, "character": 0},
                },
                "lsp_severity": 1,
                "confidence": None,
            }
        ]
    if result.returncode not in {0, None} and not diagnostics:
        diagnostics = diagnostics_from_payload("", result.stderr)
        if not diagnostics:
            diagnostics = [
                {
                    "severity": "error",
                    "code": "E001",
                    "message": f"souc check failed with exit code {result.returncode}",
                    "line": 1,
                    "column": 1,
                    "span": {"start": [1, 1], "end": [1, 1]},
                    "related": [],
                    "source": "sounio:type",
                    "range": {
                        "start": {"line": 0, "character": 0},
                        "end": {"line": 0, "character": 0},
                    },
                    "lsp_severity": 1,
                    "confidence": None,
                }
            ]
    return {
        "valid": not diagnostics and not result.timed_out and result.returncode == 0,
        "diagnostics": diagnostics,
        "diagnostic_schema": "sounio.diagnostic.v1",
        "diagnostic_envelope": shared_diagnostic_envelope(result.stdout, source, diagnostics),
        "duration_ms": result.duration_ms,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "schema": "sounio.mcp.check.v1",
    }
