#!/usr/bin/env python3
"""Find the little-endian image of [3, 5, 7, 11] in a native-v2 ELF and
require it to sit inside a non-writable PT_LOAD segment.
Presence alone does not witness the const-array image feature: the same
bytes could ride anywhere in the file. Read-only load placement does -- the
flat rodata the emitter appends is mapped without PF_W."""
import struct
import sys
from pathlib import Path

VALUES = (3, 5, 7, 11)
PT_LOAD = 1
PF_W = 2


def load_spans(data):
    """(file_offset, filesz, flags) of every PT_LOAD in a little-endian ELF64."""
    if data[:6] != b"\x7fELF\x02\x01":
        return None
    e_phoff, = struct.unpack_from("<Q", data, 0x20)
    e_phentsize, e_phnum, = struct.unpack_from("<HH", data, 0x36)
    spans = []
    for i in range(e_phnum):
        ph = e_phoff + i * e_phentsize
        p_type, p_flags, = struct.unpack_from("<II", data, ph)
        if p_type != PT_LOAD:
            continue
        p_offset, p_filesz = struct.unpack_from("<QQ", data, ph + 8)
        spans.append((p_offset, p_filesz, p_flags))
    return spans


def main():
    if len(sys.argv) != 2:
        print("usage: check_const_i64_array_image.py <elf>", file=sys.stderr)
        return 2
    needle = b"".join(v.to_bytes(8, "little", signed=True) for v in VALUES)
    data = Path(sys.argv[1]).read_bytes()
    spans = load_spans(data)
    if spans is None:
        print("FAIL not a little-endian ELF64")
        return 1
    ro = None
    idx = -1
    while True:
        idx = data.find(needle, idx + 1)
        if idx < 0:
            break
        for off, size, flags in spans:
            if off <= idx and idx + len(needle) <= off + size and not flags & PF_W:
                ro = idx
    if ro is None:
        print("FAIL image absent from every non-writable PT_LOAD")
        return 1
    print(f"PASS offset={ro} bytes={len(needle)} in read-only PT_LOAD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
