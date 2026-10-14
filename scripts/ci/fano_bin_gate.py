#!/usr/bin/env python3
"""Verify committed fano_raw_kernel.bin against full Fano kernel opcode sequence."""
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

# Verify all31 instructions start with 0x62 EVEX prefix
for i in range(31):
    if data[i * 6] != 0x62:
        print("FANO_BIN_GATE_FAIL: instruction %d at byte %d missing 0x62 prefix" % (i, i * 6))
        sys.exit(1)

# Expected opcode sequence for the31-instruction Fano kernel:
# Col0: VBROADCASTSD (0x19), VMULPD (0x59)
# Col1-7: VPERMPD (0x16), VPERMPD (0x16), VXORPD (0x57), VFMADD231PD (0xB8) ×7
# Final: VMOVAPD (0x28)
expected_opcodes = [
    0x19,  # VBROADCASTSD
    0x59,  # VMULPD
    # Col1
    0x16, 0x16, 0x57, 0xB8,
    # Col2
    0x16, 0x16, 0x57, 0xB8,
    # Col3
    0x16, 0x16, 0x57, 0xB8,
    # Col4
    0x16, 0x16, 0x57, 0xB8,
    # Col5
    0x16, 0x16, 0x57, 0xB8,
    # Col6
    0x16, 0x16, 0x57, 0xB8,
    # Col7
    0x16, 0x16, 0x57, 0xB8,
    # Final store
    0x28,
]

for i, expected in enumerate(expected_opcodes):
    actual = data[i * 6 + 4]
    if actual != expected:
        print("FANO_BIN_GATE_FAIL: instruction %d opcode at byte %d: expected 0x%02X, got 0x%02X" % (i, i * 6 + 4, expected, actual))
        sys.exit(1)

# Verify EVEX P2 (byte2) has W=1 (0x80) for all f64 instructions
for i in range(31):
    p2 = data[i * 6 + 2]
    if (p2 & 0x80) == 0:
        print("FANO_BIN_GATE_FAIL: instruction %d P2 missing W bit (0x%02X)" % (i, p2))
        sys.exit(1)

# Verify EVEX P3 (byte3) has L'L=10 (ZMM) for all instructions
for i in range(31):
    p3 = data[i * 6 + 3]
    vl = (p3 >> 5) & 3
    if vl != 2:
        print("FANO_BIN_GATE_FAIL: instruction %d P3 L'L=%d, expected 2 (ZMM)" % (i, vl))
        sys.exit(1)

print("FANO_BIN_GATE_OK: 186-byte Fano kernel verified (31 opcodes, EVEX fields, ZMM VL)")
