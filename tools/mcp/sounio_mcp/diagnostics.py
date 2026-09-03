"""Diagnostic normalisation for Sounio MCP tool results."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SHARED_DIAGNOSTIC_SCHEMA = "sounio.diagnostic.v1"
SEVERITY_NAMES = {
    1: "error",
    2: "warning",
    3: "info",
    4: "hint",
}
SEVERITY_VALUES = {name: value for value, name in SEVERITY_NAMES.items()}
VALID_SHARED_SOURCES = {
    "sounio:lex",
    "sounio:parse",
    "sounio:resolve",
    "sounio:type",
    "sounio:effect",
    "sounio:epistemic",
    "sounio:lower",
    "sounio:codegen",
    "sounio:lint",
}
CODE_RE = re.compile(r"(?:error|warning|info|note)?\[?([EWIH][0-9]{3,5})\]?")
LINE_RE = re.compile(r"\bat line ([0-9]+)\b")


def _severity_name(value: object) -> str:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"error", "warning", "info", "hint"}:
            return lowered
        if lowered == "information":
            return "info"
    if isinstance(value, int):
        return SEVERITY_NAMES.get(value, "error")
    return "error"


def _position(raw: Any) -> tuple[int, int]:
    if isinstance(raw, dict):
        line = raw.get("line", 0)
        character = raw.get("character", 0)
        if isinstance(line, int) and isinstance(character, int):
            return line, character
    return 0, 0


def normalise_diagnostic(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert an LSP-shaped diagnostic into the MCP-friendly public shape."""

    range_raw = raw.get("range")
    if not isinstance(range_raw, dict):
        range_raw = {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}}
    start_zero, start_char_zero = _position(range_raw.get("start"))
    end_zero, end_char_zero = _position(range_raw.get("end"))
    line = start_zero + 1
    column = start_char_zero + 1
    severity = _severity_name(raw.get("severity"))
    code = raw.get("code")
    message = raw.get("message", "")
    related = raw.get("related", [])
    return {
        "severity": severity,
        "code": code if isinstance(code, str) and code else "E001",
        "message": str(message),
        "line": line,
        "column": column,
        "span": {"start": [line, column], "end": [end_zero + 1, end_char_zero + 1]},
        "related": related if isinstance(related, list) else [],
        "source": raw.get("source", "sounio:type"),
        "range": range_raw,
        "lsp_severity": raw.get("severity", 1),
        "confidence": raw.get("confidence"),
    }


def diagnostics_from_payload(stdout: str, stderr: str) -> list[dict[str, Any]]:
    """Parse `souc check --json`, falling back to compiler text diagnostics."""

    stripped = stdout.strip()
    if stripped:
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return diagnostics_from_text(stdout + "\n" + stderr)
        diagnostics = payload.get("diagnostics") if isinstance(payload, dict) else None
        if isinstance(diagnostics, list):
            return [
                normalise_diagnostic(item)
                for item in diagnostics
                if isinstance(item, dict)
            ]
    return diagnostics_from_text(stdout + "\n" + stderr)


def shared_diagnostic_envelope(
    stdout: str,
    source_path: Path,
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return the CC-1/CC-2 shared diagnostic envelope for a checked source."""

    stripped = stdout.strip()
    if stripped:
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and payload.get("schema") == SHARED_DIAGNOSTIC_SCHEMA:
            raw_diagnostics = payload.get("diagnostics")
            if isinstance(raw_diagnostics, list):
                return {
                    "schema": SHARED_DIAGNOSTIC_SCHEMA,
                    "uri": str(payload.get("uri", source_path.as_uri())),
                    "version": payload.get("version"),
                    "diagnostics": [
                        _to_shared_diagnostic(item)
                        for item in raw_diagnostics
                        if isinstance(item, dict)
                    ],
                    "epistemic": payload.get("epistemic"),
                }
    return {
        "schema": SHARED_DIAGNOSTIC_SCHEMA,
        "uri": source_path.as_uri(),
        "version": None,
        "diagnostics": [_to_shared_diagnostic(item) for item in diagnostics],
        "epistemic": None,
    }


def _to_shared_diagnostic(raw: dict[str, Any]) -> dict[str, Any]:
    severity = raw.get("severity")
    if isinstance(severity, str):
        severity_value = SEVERITY_VALUES.get(_severity_name(severity), 1)
    elif isinstance(severity, int):
        severity_value = severity if severity in SEVERITY_NAMES else 1
    else:
        lsp_severity = raw.get("lsp_severity", 1)
        severity_value = int(lsp_severity) if isinstance(lsp_severity, int) else 1

    range_raw = raw.get("range")
    if not isinstance(range_raw, dict):
        line = raw.get("line", 1)
        column = raw.get("column", 1)
        line_zero = int(line) - 1 if isinstance(line, int) and line > 0 else 0
        char_zero = int(column) - 1 if isinstance(column, int) and column > 0 else 0
        range_raw = {
            "start": {"line": line_zero, "character": char_zero},
            "end": {"line": line_zero, "character": char_zero},
        }

    code = raw.get("code")
    source = raw.get("source")
    shared_source = (
        source if isinstance(source, str) and source in VALID_SHARED_SOURCES else "sounio:type"
    )
    shared = {
        "severity": severity_value,
        "code": code if isinstance(code, str) and CODE_RE.fullmatch(code) else "E001",
        "message": str(raw.get("message", ""))[:4096],
        "range": _clean_range(range_raw),
        "source": shared_source,
    }
    confidence = raw.get("confidence")
    if confidence is None or isinstance(confidence, int):
        shared["confidence"] = confidence
    related = _clean_related(raw.get("related"))
    if related:
        shared["related"] = related
    return shared


def _clean_range(raw: object) -> dict[str, dict[str, int]]:
    if not isinstance(raw, dict):
        return {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 0}}
    start_line, start_character = _position(raw.get("start"))
    end_line, end_character = _position(raw.get("end"))
    return {
        "start": {"line": max(start_line, 0), "character": max(start_character, 0)},
        "end": {"line": max(end_line, 0), "character": max(end_character, 0)},
    }


def _clean_related(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for item in raw[:8]:
        if not isinstance(item, dict):
            continue
        uri = item.get("uri")
        message = item.get("message")
        if isinstance(uri, str) and isinstance(message, str):
            cleaned.append(
                {
                    "uri": uri,
                    "range": _clean_range(item.get("range")),
                    "message": message[:1024],
                }
            )
    return cleaned


def diagnostics_from_text(text: str) -> list[dict[str, Any]]:
    """Best-effort parser for legacy compiler diagnostic text."""

    diagnostics: list[dict[str, Any]] = []
    for line in text.splitlines():
        if "error" not in line.lower() and "warning" not in line.lower():
            continue
        code_match = CODE_RE.search(line)
        line_match = LINE_RE.search(line)
        source_line = int(line_match.group(1)) if line_match else 1
        code = code_match.group(1) if code_match else "E001"
        severity = "warning" if "warning" in line.lower() else "error"
        diagnostics.append(
            {
                "severity": severity,
                "code": code,
                "message": line.strip(),
                "line": source_line,
                "column": 1,
                "span": {"start": [source_line, 1], "end": [source_line, 1]},
                "related": [],
                "source": "sounio:type",
                "range": {
                    "start": {"line": max(source_line - 1, 0), "character": 0},
                    "end": {"line": max(source_line - 1, 0), "character": 0},
                },
                "lsp_severity": 1 if severity == "error" else 2,
                "confidence": None,
            }
        )
    return diagnostics
