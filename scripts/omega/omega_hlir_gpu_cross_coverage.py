#!/usr/bin/env python3
"""Emit anti-drift cross-coverage report for HLIR->GPU lowering and backend emitters."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "sounio.omega.hlir-gpu-cross-coverage.v1"
DEFAULT_HLIR_TO_GPU_PATH = Path("self-hosted/gpu/hlir_to_gpu.sio")
DEFAULT_METAL_PATH = Path("self-hosted/gpu/metal.sio")
DEFAULT_PTX_PATH = Path("self-hosted/gpu/ptx_advanced.sio")
DEFAULT_REPORT_PATH = Path("artifacts/omega/hlir_gpu_cross_coverage.v1.json")

SHARED_CORE_PAIRS = (
    ("HlirOpAdd", "GpuOpcodeAdd", "GpuOpcode::GpuAdd"),
    ("HlirOpSub", "GpuOpcodeSub", "GpuOpcode::GpuSub"),
    ("HlirOpMul", "GpuOpcodeMul", "GpuOpcode::GpuMul"),
    ("HlirOpDiv", "GpuOpcodeDiv", "GpuOpcode::GpuDiv"),
    ("HlirOpEq", "GpuOpcodeSetpEq", "GpuOpcode::GpuSetpEq"),
    ("HlirOpNe", "GpuOpcodeSetpNe", "GpuOpcode::GpuSetpNe"),
    ("HlirOpLt", "GpuOpcodeSetpLt", "GpuOpcode::GpuSetpLt"),
    ("HlirOpLe", "GpuOpcodeSetpLe", "GpuOpcode::GpuSetpLe"),
    ("HlirOpGt", "GpuOpcodeSetpGt", "GpuOpcode::GpuSetpGt"),
    ("HlirOpGe", "GpuOpcodeSetpGe", "GpuOpcode::GpuSetpGe"),
    ("HlirOpLoad", "GpuOpcodeLoadGlobal", "GpuOpcode::GpuLoadGlobal"),
    ("HlirOpStore", "GpuOpcodeStoreGlobal", "GpuOpcode::GpuStoreGlobal"),
    ("HlirOpSExt", "GpuOpcodeCvt", "GpuOpcode::GpuCvt"),
    ("HlirOpSelect", "GpuOpcodeSelp", "GpuOpcode::GpuSelp"),
    ("HlirOpFMA", "GpuOpcodeFma", "GpuOpcode::GpuFma"),
    ("HlirOpGetThreadId", "GpuOpcodeGetTid", "GpuOpcode::GpuGetTid"),
    ("HlirOpGetBlockId", "GpuOpcodeGetBid", "GpuOpcode::GpuGetBid"),
    ("HlirOpGetBlockDim", "GpuOpcodeGetNtid", "GpuOpcode::GpuGetNtid"),
    ("HlirOpAtomicAdd", "GpuOpcodeAtomicAdd", "GpuOpcode::GpuAtomicAdd"),
    ("HlirOpSyncThreads", "GpuOpcodeBarrierSync", "GpuOpcode::GpuBarrierSync"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate HLIR->GPU and backend opcode cross coverage"
    )
    parser.add_argument(
        "--hlir-to-gpu",
        default=str(DEFAULT_HLIR_TO_GPU_PATH),
        help="Path to self-hosted/gpu/hlir_to_gpu.sio",
    )
    parser.add_argument(
        "--metal",
        default=str(DEFAULT_METAL_PATH),
        help="Path to self-hosted/gpu/metal.sio",
    )
    parser.add_argument(
        "--ptx-advanced",
        default=str(DEFAULT_PTX_PATH),
        help="Path to self-hosted/gpu/ptx_advanced.sio",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_REPORT_PATH),
        help="Output report path",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if strict anchor markers are missing",
    )
    return parser.parse_args()


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_text(path: Path) -> str:
    return path.read_text()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")


def main() -> int:
    args = parse_args()

    hlir_path = Path(args.hlir_to_gpu)
    metal_path = Path(args.metal)
    ptx_path = Path(args.ptx_advanced)
    out_path = Path(args.out)

    errors: list[str] = []

    try:
        hlir_text = load_text(hlir_path)
    except OSError as exc:
        hlir_text = ""
        errors.append(f"{hlir_path}: read failed: {exc}")

    try:
        metal_text = load_text(metal_path)
    except OSError as exc:
        metal_text = ""
        errors.append(f"{metal_path}: read failed: {exc}")

    try:
        ptx_text = load_text(ptx_path)
    except OSError as exc:
        ptx_text = ""
        errors.append(f"{ptx_path}: read failed: {exc}")

    checks = []
    full_pass_count = 0

    if hlir_text and metal_text and ptx_text:
        for hlir_op, gpu_const, backend_marker in SHARED_CORE_PAIRS:
            has_hlir_branch = f"op_tag == {hlir_op}" in hlir_text
            has_gpu_const = gpu_const in hlir_text
            has_metal_backend = backend_marker in metal_text
            has_ptx_backend = backend_marker in ptx_text
            all_pass = (
                has_hlir_branch
                and has_gpu_const
                and has_metal_backend
                and has_ptx_backend
            )

            if all_pass:
                full_pass_count += 1
            else:
                if not has_hlir_branch:
                    errors.append(f"{hlir_path}: missing lowering branch for {hlir_op}")
                if not has_gpu_const:
                    errors.append(
                        f"{hlir_path}: missing opcode constant {gpu_const!r} for {hlir_op}"
                    )
                if not has_metal_backend:
                    errors.append(
                        f"{metal_path}: missing backend marker {backend_marker!r}"
                    )
                if not has_ptx_backend:
                    errors.append(
                        f"{ptx_path}: missing backend marker {backend_marker!r}"
                    )

            checks.append(
                {
                    "hlir_op": hlir_op,
                    "gpu_opcode_const": gpu_const,
                    "backend_marker": backend_marker,
                    "has_hlir_branch": has_hlir_branch,
                    "has_gpu_const": has_gpu_const,
                    "has_metal_backend": has_metal_backend,
                    "has_ptx_backend": has_ptx_backend,
                    "status": "pass" if all_pass else "fail",
                }
            )

    strict_anchor_present = "fn hlir_lower_instr(" in hlir_text
    if args.strict and not strict_anchor_present:
        errors.append(
            f"{hlir_path}: missing strict anchor marker 'fn hlir_lower_instr('"
        )

    status = "pass" if not errors else "fail"

    report = {
        "schema": SCHEMA,
        "generated_at_utc": now_utc_iso(),
        "status": status,
        "strict": bool(args.strict),
        "inputs": {
            "hlir_to_gpu": str(hlir_path),
            "metal": str(metal_path),
            "ptx_advanced": str(ptx_path),
        },
        "summary": {
            "pair_count": len(SHARED_CORE_PAIRS),
            "full_pass_count": full_pass_count,
            "strict_anchor_present": strict_anchor_present,
        },
        "checks": checks,
        "errors": sorted(set(errors)),
    }

    write_json(out_path, report)

    print(
        "omega_hlir_gpu_cross_coverage: "
        f"status={status} pairs={len(SHARED_CORE_PAIRS)} "
        f"full_pass={full_pass_count} report={out_path}"
    )

    if errors:
        for error in sorted(set(errors)):
            print(f"error: {error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
