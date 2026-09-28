#!/usr/bin/env python3
"""Find the little-endian image of [3, 5, 7, 11] in a native-v2 ELF.

Passes only for an ELF emitted by a madaros rebuilt from
madaros/const-i64-array-image. The committed prebuilt still lowers the
literal as per-element stores, so this needle is absent there.
"""
import sys
from pathlib import Path

VALUES = (3, 5, 7, 11)


def main():
    if len(sys.argv) != 2:
        print("usage: check_const_i64_array_image.py <elf>", file=sys.stderr)
        return 2
    needle = b"".join(v.to_bytes(8, "little", signed=True) for v in VALUES)
    data = Path(sys.argv[1]).read_bytes()
    idx = data.find(needle)
    if idx < 0:
        print("FAIL image not found")
        return 1
    print(f"PASS offset={idx} bytes={len(needle)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
