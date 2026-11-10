#!/usr/bin/env python3
"""Catch references to nonexistent KernelIR GPU opcodes before a self-build."""
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
if len(sys.argv) > 1:
    root = Path(sys.argv[1])

enum_source = (root / "self-hosted/gpu/kernel_ir.sio").read_text()
bridge_source = (root / "self-hosted/gpu/hlir_to_gpu.sio").read_text()
# Strip comments so documentation mentioning absent variants is harmless.
enum_source = re.sub(r"//[^\n]*|/\*.*?\*/", "", enum_source, flags=re.S)
bridge_source = re.sub(r"//[^\n]*|/\*.*?\*/", "", bridge_source, flags=re.S)
enum_match = re.search(r"pub enum GpuOpcode\s*\{([^}]+)\}", enum_source)
if enum_match is None:
    sys.exit("GPU_OPCODE_SURFACE_FAIL: cannot locate public GpuOpcode enum")
variants = set(re.findall(r"\b(Gpu\w+)\b", enum_match[1]))
references = set(re.findall(r"\bGpuOpcode::(\w+)\b", bridge_source))
if not variants or not references:
    sys.exit("GPU_OPCODE_SURFACE_FAIL: empty enum or bridge reference surface")
missing = sorted(references - variants)
if missing:
    sys.exit("GPU_OPCODE_SURFACE_FAIL: unresolved " + " ".join(missing))
print(f"GPU_OPCODE_SURFACE_OK: {len(references)} bridge references resolve")
