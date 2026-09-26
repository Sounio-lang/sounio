#!/usr/bin/env python3
"""Verify committed fano_raw_kernel.bin matches expected Fano kernel structure."""
import sys
from pathlib import Path

bin_path = Path("tests/run-pass/fano_raw_kernel.bin")
if not bin_path.exists():
    print("FANO_BIN_GATE_FAIL: fano_raw_kernel.bin not found")
    sys.exit(1)

data = bin_path.read_bytes()
if len(data) != 186:
    print("FANO_BIN_GATE_FAIL: expected 186 bytes, got %d" % len(data))
    sys.exit(1)

# Instruction 1: VBROADCASTSD t_bj, zb_in  (62 F2 FD 48 19 E1)
if data[0:4] != bytes([0x62, 0xF2, 0xFD, 0x48]):
    print("FANO_BIN_GATE_FAIL: bad EVEX prefix at byte 0")
    sys.exit(1)
if data[4] != 0x19:
    print("FANO_BIN_GATE_FAIL: expected VBROADCASTSD opcode 0x19 at byte 4, got 0x%02X" % data[4])
    sys.exit(1)

# Instruction 31: VMOVAPD dst, accum  (62 D1 FD 48 28 D0)
if data[180:184] != bytes([0x62, 0xD1, 0xFD, 0x48]):
    print("FANO_BIN_GATE_FAIL: bad EVEX prefix at byte 180")
    sys.exit(1)
if data[184] != 0x28:
    print("FANO_BIN_GATE_FAIL: expected VMOVAPD opcode 0x28 at byte 184, got 0x%02X" % data[184])
    sys.exit(1)

# Verify all 31 instructions start with 0x62 EVEX prefix
for i in range(31):
    if data[i * 6] != 0x62:
        print("FANO_BIN_GATE_FAIL: instruction %d at byte %d missing 0x62 prefix" % (i, i * 6))
        sys.exit(1)

# Verify opcodes at expected positions
expected_opcodes = [
    (4, 0x19, "VBROADCASTSD"),
    (10, 0x59, "VMULPD"),
    (16, 0x16, "VPERMPD col1-b"),
    (22, 0x16, "VPERMPD col1-a"),
    (28, 0x57, "VXORPD col1"),
    (34, 0xB8, "VFMADD231PD col1"),
]
for offset, expected, name in expected_opcodes:
    if data[offset] != expected:
        print("FANO_BIN_GATE_FAIL: expected %s 0x%02X at byte %d, got 0x%02X" % (name, expected, offset, data[offset]))
        sys.exit(1)

print("FANO_BIN_GATE_OK: 186-byte Fano kernel structure verified")
