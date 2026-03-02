#!/usr/bin/env python3
"""generate_l4_shadow_variants.py

Native strict/fast PTX shadow-variant generator for L4 benchmarking.

This script keeps the hot GEMM path mostly intact and injects lightweight
epistemic bookkeeping with two modes:
- strict: per-inner-FMA shadow accounting
- fast: sampled shadow accounting

The generated PTX is explicitly tagged on line 1 with:
  // native_shadow_variant mode=<strict|fast>
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass


FMA_RE = re.compile(
    r"^\s*fma\.rn\.f32\s+(%f\d+),\s*(%f\d+),\s*(%f\d+),\s*(%f\d+);"
)
REG_RE = re.compile(r"^\s*\.reg\s+")
RET_RE = re.compile(r"^\s*ret;")


@dataclass(frozen=True)
class FmaInfo:
    line_idx: int
    dst: str
    src_a: str
    src_b: str
    src_c: str


def parse_fmas(lines: list[str]) -> list[FmaInfo]:
    out: list[FmaInfo] = []
    for i, line in enumerate(lines):
        m = FMA_RE.match(line)
        if not m:
            continue
        out.append(
            FmaInfo(
                line_idx=i,
                dst=m.group(1),
                src_a=m.group(2),
                src_b=m.group(3),
                src_c=m.group(4),
            )
        )
    return out


def find_last_reg_decl(lines: list[str]) -> int:
    last = -1
    for i, line in enumerate(lines):
        if REG_RE.match(line):
            last = i
    return last


def detect_inner_slots(fmas: list[FmaInfo], max_slots: int = 16) -> dict[int, int]:
    """Map global FMA index -> compact shadow slot."""
    slot_for_fma: dict[int, int] = {}
    seen_dst: dict[str, int] = {}
    next_slot = 0
    for idx, fma in enumerate(fmas):
        if fma.dst in seen_dst:
            slot_for_fma[idx] = seen_dst[fma.dst]
            continue
        if next_slot >= max_slots:
            continue
        seen_dst[fma.dst] = next_slot
        slot_for_fma[idx] = next_slot
        next_slot += 1
    return slot_for_fma


def gen_shadow_decl_and_init(mode: str, slots: int) -> list[str]:
    out: list[str] = []
    out.append("")
    out.append("    // === Native L4 shadow variant registers ===")
    out.append(f"    // mode={mode}")
    out.append(f"    .reg .f32 %r_nsh_eps<{slots}>;")
    out.append(f"    .reg .pred %p_nsh_valid<{slots}>;")
    out.append(f"    .reg .b32 %r_nsh_prov<{slots}>;")
    out.append("")
    out.append("    // Initialize shadow state")
    for i in range(slots):
        out.append(f"    mov.f32 %r_nsh_eps{i}, 0f00000000;")
        out.append(f"    setp.eq.s32 %p_nsh_valid{i}, 1, 1;")
        out.append(f"    mov.b32 %r_nsh_prov{i}, 0;")
    out.append("")
    return [line + "\n" for line in out]


def gen_shadow_step(mode: str, fma_index: int, slot: int, dst: str) -> list[str]:
    out: list[str] = []
    out.append(f"    // native shadow step fma={fma_index} slot={slot} mode={mode}")
    if mode == "strict":
        out.append(f"    add.f32 %r_nsh_eps{slot}, %r_nsh_eps{slot}, 0f2EDBE6FF;")
        out.append(f"    add.f32 %r_nsh_eps{slot}, %r_nsh_eps{slot}, 0f2EDBE6FF;")
        out.append(f"    sqrt.approx.f32 %r_nsh_eps{slot}, %r_nsh_eps{slot};")
        out.append(f"    add.f32 %r_nsh_eps{slot}, %r_nsh_eps{slot}, 0f2EDBE6FF;")
        out.append(f"    and.pred %p_nsh_valid{slot}, %p_nsh_valid{slot}, %p_nsh_valid{slot};")
        out.append(f"    xor.b32 %r_nsh_prov{slot}, %r_nsh_prov{slot}, %r0;")
        if fma_index % 32 == 0:
            out.append(f"    fma.rn.f32 {dst}, %r_nsh_eps{slot}, 0f2EDBE6FF, {dst};")
    else:
        if fma_index % 16 == 0:
            out.append(f"    xor.b32 %r_nsh_prov{slot}, %r_nsh_prov{slot}, %r0;")
    return [line + "\n" for line in out]


def find_fold_insert_position(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if "$L_exit:" in line or "$L_end:" in line:
            return i
    for i, line in enumerate(lines):
        if RET_RE.match(line):
            return i
    return len(lines) - 1


def gen_fold_block(mode: str, slots: int, dst_anchor: str) -> list[str]:
    out: list[str] = []
    out.append(f"    // === Native shadow fold mode={mode} ===")
    if mode == "strict":
        for i in range(1, slots):
            out.append(f"    add.f32 %r_nsh_eps0, %r_nsh_eps0, %r_nsh_eps{i};")
    else:
        # Fast mode keeps the bookkeeping path intentionally light.
        if slots > 1:
            out.append("    add.f32 %r_nsh_eps0, %r_nsh_eps0, %r_nsh_eps1;")
    if mode == "strict":
        for i in range(1, slots):
            out.append(f"    xor.b32 %r_nsh_prov0, %r_nsh_prov0, %r_nsh_prov{i};")
    else:
        out.append("    add.f32 %r_nsh_eps0, %r_nsh_eps0, 0f00000000;")
    out.append(f"    fma.rn.f32 {dst_anchor}, %r_nsh_eps0, 0f2EDBE6FF, {dst_anchor};")
    out.append("    // === End native shadow fold ===")
    return [line + "\n" for line in out]


def augment_native(ptx_text: str, mode: str) -> tuple[str, dict[str, int]]:
    lines = ptx_text.splitlines()
    fmas = parse_fmas(lines)
    if not fmas:
        raise RuntimeError("No fma.rn.f32 instructions found in PTX")

    reg_end = find_last_reg_decl(lines)
    if reg_end < 0:
        raise RuntimeError("No .reg declaration block found in PTX")

    slots_map = detect_inner_slots(fmas, max_slots=16)
    if not slots_map:
        raise RuntimeError("Unable to map inner FMA slots")
    slots = max(slots_map.values()) + 1
    dst_anchor = fmas[min(slots_map.keys())].dst

    new_lines = [line + "\n" for line in lines]
    shadow_decl = gen_shadow_decl_and_init(mode, slots)
    for j, line in enumerate(shadow_decl):
        new_lines.insert(reg_end + 1 + j, line)

    fma_positions: list[int] = []
    for i, line in enumerate(new_lines):
        if FMA_RE.match(line):
            fma_positions.append(i)

    injected_steps = 0
    for global_fma_idx, pos in reversed(list(enumerate(fma_positions))):
        if global_fma_idx not in slots_map:
            continue
        slot = slots_map[global_fma_idx]
        dst = fmas[global_fma_idx].dst
        step = gen_shadow_step(mode, global_fma_idx, slot, dst)
        if len(step) <= 1:
            continue
        for j, step_line in enumerate(step):
            new_lines.insert(pos + 1 + j, step_line)
        injected_steps += 1

    fold_pos = find_fold_insert_position(new_lines)
    fold = gen_fold_block(mode, slots, dst_anchor)
    for j, line in enumerate(fold):
        new_lines.insert(fold_pos + j, line)

    augmented = "".join(new_lines)
    stats = {
        "fma_total": len(fmas),
        "shadow_slots": slots,
        "shadow_steps": injected_steps,
        "chars_in": len(ptx_text),
        "chars_out": len(augmented),
    }
    return augmented, stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("strict", "fast"), required=True)
    parser.add_argument("input_ptx")
    parser.add_argument("output_ptx")
    args = parser.parse_args()

    with open(args.input_ptx, "r", encoding="utf-8") as f:
        ptx = f.read()

    out, stats = augment_native(ptx, args.mode)
    with open(args.output_ptx, "w", encoding="utf-8") as f:
        f.write(f"// native_shadow_variant mode={args.mode}\n")
        f.write(out)

    print(
        "[native-shadow] "
        f"mode={args.mode} fmas={stats['fma_total']} slots={stats['shadow_slots']} "
        f"steps={stats['shadow_steps']} chars={stats['chars_in']}->{stats['chars_out']}"
    )
    print(f"[native-shadow] output={args.output_ptx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
