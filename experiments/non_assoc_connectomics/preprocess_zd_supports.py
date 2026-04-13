#!/usr/bin/env python3
"""
Phase 2 helper: convert sedenion_zero_divisor_geometry.v1.json to a flat
binary the Sounio pipeline can load without a JSON parser.

Input : artifacts/research/sedenion_zero_divisor_geometry.v1.json
Output: artifacts/research/sedenion_zd_168.bin

Binary format (v2 — all i64 for easy Sounio access):
    Header : n_pairs (i64 little-endian) = 168
    Body   : 168 × 6·i64 little-endian records
             [lhs, rhs, sup0, sup1, sup2, sup3]
             48 bytes/record, total 168 × 48 = 8064 bytes
    Grand total : 8072 bytes

Sounio's native compiler has read_i64 but no read_i32 builtin, so we pay
the 2× size for loader simplicity.

Each entry corresponds to one of the 168 unordered projective zero-divisor
pair classes in the 16-dimensional sedenion algebra. `sup0..sup3` are the
4 indices in [1, 15] whose positions define the two intersecting planes.

Idempotent; run once locally, commit the .bin artifact.
"""

from __future__ import annotations

import json
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "artifacts", "research", "sedenion_zero_divisor_geometry.v1.json")
DST = os.path.join(ROOT, "artifacts", "research", "sedenion_zd_168.bin")


def main() -> int:
    if not os.path.exists(SRC):
        print(f"ERROR: {SRC} not found", file=sys.stderr)
        return 1

    with open(SRC, "rb") as f:
        data = json.load(f)

    pairs = data.get("unordered_projective_pairs", [])
    if len(pairs) != 168:
        print(f"ERROR: expected 168 pairs, got {len(pairs)}", file=sys.stderr)
        return 2

    with open(DST, "wb") as f:
        f.write(struct.pack("<q", len(pairs)))
        for p in pairs:
            sup = p["support_union"]
            if len(sup) != 4:
                print(f"ERROR: pair has {len(sup)} support indices, expected 4", file=sys.stderr)
                return 3
            f.write(struct.pack("<qqqqqq", p["lhs"], p["rhs"], sup[0], sup[1], sup[2], sup[3]))

    size = os.path.getsize(DST)
    expected = 8 + 168 * 48
    assert size == expected, f"size mismatch: got {size}, expected {expected}"
    print(f"wrote {DST} ({size} bytes, 168 pairs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
