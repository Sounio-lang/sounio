#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory aggregate-return fallback sites in lean_single.sio.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    return parser.parse_args()


def find_function_block(lines: list[str], symbol: str) -> tuple[int, int]:
    start = -1
    for idx, line in enumerate(lines):
        if line.startswith(f"fn {symbol}("):
            start = idx
            break
    if start < 0:
        raise SystemExit(f"missing symbol: {symbol}")
    end = len(lines)
    for idx in range(start + 1, len(lines)):
        if lines[idx].startswith("fn "):
            end = idx
            break
    return start, end


def inspect_site(lines: list[str], symbol: str, target: str, diagnostic: str) -> dict[str, object]:
    start, end = find_function_block(lines, symbol)
    block = lines[start:end]
    fallback_lines: list[int] = []
    diagnostic_line = 0
    for offset, line in enumerate(block, start=start + 1):
        if "RET_AGG_BSS_OFF" in line:
            fallback_lines.append(offset)
        if diagnostic in line:
            diagnostic_line = offset
    classification = "unsupported_but_fenced" if diagnostic_line > 0 else "transitional_legacy_fallback"
    return {
        "symbol": symbol,
        "target": target,
        "source_line": start + 1,
        "fallback_lines": fallback_lines,
        "legacy_storage_referenced": bool(fallback_lines),
        "diagnostic_line": diagnostic_line,
        "classification": classification,
        "normal_supported_path": "caller-owned hidden aggregate-return buffer",
        "legacy_storage": "GL_BSS_BASE + RET_AGG_BSS_OFF",
    }


def main() -> int:
    args = parse_args()
    source_path = Path(args.source)
    lines = source_path.read_text(encoding="utf-8").splitlines()

    entries = [
        inspect_site(lines, "stabilize_return_agg_x86", "x86_64", "x86 aggregate-return fallback requires caller-provided SRET buffer"),
        inspect_site(lines, "stabilize_return_agg_a64", "aarch64", "AArch64 aggregate-return fallback requires caller-provided SRET buffer"),
    ]

    report = {
        "schema": "sounio.selfhost_fallback_inventory.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": str(source_path),
        "entries": entries,
        "classification_counts": {},
    }
    for entry in entries:
        key = entry["classification"]
        report["classification_counts"][key] = report["classification_counts"].get(key, 0) + 1

    Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md_lines = [
        "# Selfhost Aggregate Return Fallback Inventory",
        "",
    ]
    for entry in entries:
        md_lines.append(f"## {entry['symbol']}")
        md_lines.append("")
        md_lines.append(f"- target: {entry['target']}")
        md_lines.append(f"- classification: {entry['classification']}")
        md_lines.append(f"- source_line: {entry['source_line']}")
        if entry["fallback_lines"]:
            md_lines.append(f"- fallback_lines: {', '.join(str(v) for v in entry['fallback_lines'])}")
        else:
            md_lines.append("- fallback_lines: none")
        md_lines.append(f"- legacy_storage_referenced: {entry['legacy_storage_referenced']}")
        md_lines.append(f"- diagnostic_line: {entry['diagnostic_line']}")
        md_lines.append(f"- normal_supported_path: {entry['normal_supported_path']}")
        md_lines.append("")
    Path(args.md_out).write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    for entry in entries:
        if entry["classification"] != "unsupported_but_fenced":
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
