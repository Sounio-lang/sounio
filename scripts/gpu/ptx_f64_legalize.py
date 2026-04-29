#!/usr/bin/env python3
"""Legalize f64 PTX emitted by the pinned public GPU compiler artifact.

The beta.4 public GPU artifact can declare kernel parameters as `.f64` while
loading them through `.u64` scratch registers. That makes later arithmetic pick
`.f32` opcodes even when the visible kernel contract is f64. This pass is a
small, deterministic bridge: it rewrites the affected parameter loads and local
float dataflow into structurally f64 PTX, while leaving the raw PTX artifact
available for root-cause comparison.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import struct
from typing import Iterable


ENTRY_RE = re.compile(r"^\s*(?:\.visible\s+)?\.(?:entry|func)\b")
PARAM_DECL_RE = re.compile(r"^\s*\.param\s+(\.\w+)\s+([A-Za-z_][A-Za-z0-9_]*)")
LD_PARAM_U64_RE = re.compile(
    r"^(\s*)ld\.param\.u64\s+(r64_(\d+)),\s+\[param_(\d+)\];(\s*)$"
)
FLOAT_ARITH_F32_RE = re.compile(
    r"^(\s*)(add|sub|mul|div)\.f32\s+([^;]+);(\s*)$"
)
SQRT_F32_RE = re.compile(r"^(\s*)sqrt\.approx\.f32\s+([^,]+),\s+([^;]+);(\s*)$")
MOV_F64_F32_IMM_RE = re.compile(
    r"^(\s*)mov\.f64\s+(f64_(\d+)),\s+0[Ff]([0-9A-Fa-f]{8});(\s*)$"
)
MOV_F32_IMM_RE = re.compile(
    r"^(\s*)mov\.f32\s+(f32_(\d+)),\s+0[Ff]([0-9A-Fa-f]{8});(\s*)$"
)
REG_F64_RE = re.compile(r"^(\s*\.reg\s+\.f64\s+f64_<)(\d+)(>;.*)$")
REG_F32_NAME_RE = re.compile(r"^f32_(\d+)$")
REG_F64_NAME_RE = re.compile(r"^f64_(\d+)$")
REG_R64_NAME_RE = re.compile(r"^r64_(\d+)$")
F32_PROMOTION_BASE = 64


def f32_hex_to_f64_hex(bits: str) -> str:
    """Convert a PTX f32 hex immediate body to an f64 hex immediate."""

    as_int = int(bits, 16)
    as_float = struct.unpack(">f", struct.pack(">I", as_int))[0]
    as_double = struct.unpack(">Q", struct.pack(">d", float(as_float)))[0]
    return f"0d{as_double:016X}"


def split_operands(text: str) -> list[str]:
    return [part.strip() for part in text.split(",")]


def max_f64_index_from_token(token: str, current: int) -> int:
    match = REG_F64_NAME_RE.match(token.strip())
    if match:
        return max(current, int(match.group(1)))
    return current


def ensure_f64_token(
    token: str,
    r64_to_f64: dict[str, str],
    f32_promotions: dict[str, str],
    max_f64_index: int,
) -> tuple[str, int]:
    stripped = token.strip()
    if stripped in r64_to_f64:
        mapped = r64_to_f64[stripped]
        return mapped, max_f64_index_from_token(mapped, max_f64_index)

    r64_match = REG_R64_NAME_RE.match(stripped)
    if r64_match:
        mapped = f"f64_{r64_match.group(1)}"
        r64_to_f64[stripped] = mapped
        return mapped, max(max_f64_index, int(r64_match.group(1)))

    f32_match = REG_F32_NAME_RE.match(stripped)
    if f32_match:
        f32_num = f32_match.group(1)
        promoted = f32_promotions.setdefault(
            f32_num, f"f64_{F32_PROMOTION_BASE + int(f32_num)}"
        )
        promoted_match = REG_F64_NAME_RE.match(promoted)
        promoted_index = int(promoted_match.group(1)) if promoted_match else -1
        return promoted, max(max_f64_index, promoted_index)

    return stripped, max_f64_index_from_token(stripped, max_f64_index)


def operand_is_f64(token: str, r64_to_f64: dict[str, str]) -> bool:
    stripped = token.strip()
    return (
        stripped.startswith("f64_")
        or stripped in r64_to_f64
        or stripped.startswith("r64_")
    )


def legalize_lines(lines: Iterable[str]) -> tuple[list[str], dict[str, int]]:
    out: list[str] = []
    current_f64_params: set[int] = set()
    in_param_header = False
    param_index = 0
    r64_to_f64: dict[str, str] = {}
    f32_promotions: dict[str, str] = {}
    max_f64_index = -1

    stats = {
        "f64_param_loads_rewritten": 0,
        "f32_arithmetic_rewritten": 0,
        "sqrt_rewritten": 0,
        "f64_literals_repaired": 0,
        "f32_literals_promoted": 0,
        "f64_register_decl_expanded": 0,
    }

    for line in lines:
        if ENTRY_RE.match(line):
            current_f64_params = set()
            r64_to_f64 = {}
            in_param_header = "(" in line and ")" not in line
            param_index = 0
            out.append(line)
            continue

        if in_param_header:
            param_match = PARAM_DECL_RE.match(line)
            if param_match:
                if param_match.group(1) == ".f64":
                    current_f64_params.add(param_index)
                param_index += 1
            if line.strip() == ")":
                in_param_header = False
            out.append(line)
            continue

        param_load = LD_PARAM_U64_RE.match(line)
        if param_load and int(param_load.group(4)) in current_f64_params:
            reg_num = param_load.group(3)
            r64_name = param_load.group(2)
            f64_name = f"f64_{reg_num}"
            r64_to_f64[r64_name] = f64_name
            max_f64_index = max(max_f64_index, int(reg_num))
            out.append(
                f"{param_load.group(1)}ld.param.f64 {f64_name}, "
                f"[param_{param_load.group(4)}];{param_load.group(5)}\n"
            )
            stats["f64_param_loads_rewritten"] += 1
            continue

        mov_f64 = MOV_F64_F32_IMM_RE.match(line)
        if mov_f64:
            max_f64_index = max(max_f64_index, int(mov_f64.group(3)))
            out.append(
                f"{mov_f64.group(1)}mov.f64 {mov_f64.group(2)}, "
                f"{f32_hex_to_f64_hex(mov_f64.group(4))};{mov_f64.group(5)}\n"
            )
            stats["f64_literals_repaired"] += 1
            continue

        arith = FLOAT_ARITH_F32_RE.match(line)
        if arith:
            operands = split_operands(arith.group(3))
            if len(operands) == 3 and (
                current_f64_params or any(operand_is_f64(op, r64_to_f64) for op in operands)
            ):
                converted: list[str] = []
                for operand in operands:
                    replacement, max_f64_index = ensure_f64_token(
                        operand, r64_to_f64, f32_promotions, max_f64_index
                    )
                    converted.append(replacement)
                opcode = arith.group(2)
                if opcode == "div":
                    opcode = "div.rn"
                out.append(
                    f"{arith.group(1)}{opcode}.f64 {converted[0]}, "
                    f"{converted[1]}, {converted[2]};{arith.group(4)}\n"
                )
                stats["f32_arithmetic_rewritten"] += 1
                continue

        sqrt = SQRT_F32_RE.match(line)
        if sqrt and (current_f64_params or operand_is_f64(sqrt.group(3), r64_to_f64)):
            dst, max_f64_index = ensure_f64_token(
                sqrt.group(2), r64_to_f64, f32_promotions, max_f64_index
            )
            src, max_f64_index = ensure_f64_token(
                sqrt.group(3), r64_to_f64, f32_promotions, max_f64_index
            )
            out.append(f"{sqrt.group(1)}sqrt.rn.f64 {dst}, {src};{sqrt.group(4)}\n")
            stats["sqrt_rewritten"] += 1
            continue

        out.append(line)

    expanded: list[str] = []
    f64_decl_seen = False
    for line in out:
        reg_decl = REG_F64_RE.match(line)
        if reg_decl:
            f64_decl_seen = True
            existing = int(reg_decl.group(2))
            required = max(existing, max_f64_index + 1)
            if required > existing:
                stats["f64_register_decl_expanded"] += 1
            expanded.append(f"{reg_decl.group(1)}{required}{reg_decl.group(3)}\n")
            continue

        mov_f32 = MOV_F32_IMM_RE.match(line)
        expanded.append(line)
        if mov_f32 and mov_f32.group(3) in f32_promotions:
            expanded.append(
                f"{mov_f32.group(1)}mov.f64 {f32_promotions[mov_f32.group(3)]}, "
                f"{f32_hex_to_f64_hex(mov_f32.group(4))};{mov_f32.group(5)}\n"
            )
            stats["f32_literals_promoted"] += 1

    if not f64_decl_seen and max_f64_index >= 0:
        stats["f64_register_decl_expanded"] += 1

    stats["f32_literal_registers_promoted"] = len(f32_promotions)
    stats["max_f64_register_index"] = max_f64_index
    return expanded, stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path)
    args = parser.parse_args()

    lines = args.input.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    legalized, stats = legalize_lines(lines)
    args.output.write_text("".join(legalized), encoding="utf-8")

    if args.report:
        args.report.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
