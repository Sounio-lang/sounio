#!/usr/bin/env python3
"""Host reference for a kind=2 operator, named by its 16-bit phase code.

The 320 inputs are the a_bits/b_bits of the frozen numeric-fixtures.jsonl. The
arithmetic is the one numeric_fixtures.sio and materialize_ptx.sio fix: for
output k, ascending j, product = a[k^j] * b[j], signed = product * sigma,
sum = sum + signed, starting from +0.0, binary64 round-to-nearest, no FMA.
sigma = cd_sigma(k^j, j) * (-1)^((k^j)^T B j), the sign admission.sio
reconstructs. Before it writes anything, it recomputes phase 0 and refuses
unless that reproduces the Sounio fixture output_bits for every word. The
frame is the probe's reference.hex format, and parity is still decided by
material_parity.sio, not here.
"""
import argparse
import json
import struct
from pathlib import Path


def cd_sigma(a, b, bits=4):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    a_hi, b_hi, a_lo, b_lo = a >= half, b >= half, a & (half - 1), b & (half - 1)
    if not a_hi and not b_hi:
        return cd_sigma(a_lo, b_lo, bits - 1)
    if not a_hi and b_hi:
        return cd_sigma(b_lo, a_lo, bits - 1)
    if a_hi and not b_hi:
        return cd_sigma(a_lo, 0, bits - 1) if b_lo == 0 else -cd_sigma(a_lo, b_lo, bits - 1)
    return -cd_sigma(0, a_lo, bits - 1) if b_lo == 0 else cd_sigma(b_lo, a_lo, bits - 1)


def bilinear(code, x, y):
    s = 0
    for i in range(4):
        if (x >> i) & 1:
            for j in range(4):
                if (y >> j) & 1:
                    s ^= (code >> (i * 4 + j)) & 1
    return s


def sigma_phase(code, a, b):
    base = cd_sigma(a, b)
    return -base if bilinear(code, a, b) else base


def to_f64(bits):
    return struct.unpack("<d", struct.pack("<q", bits))[0]


def to_bits(value):
    return struct.unpack("<q", struct.pack("<d", value))[0]


def outputs(phase, row):
    a = [to_f64(x) for x in row["a_bits"]]
    b = [to_f64(x) for x in row["b_bits"]]
    out = []
    for k in range(16):
        total = 0.0
        for j in range(16):
            product = a[k ^ j] * b[j]
            total = total + product * float(sigma_phase(phase, k ^ j, j))
        out.append(to_bits(total))
    return out


def is_nan(bits):
    return (bits & 0x7FF0000000000000) == 0x7FF0000000000000 and (bits & 0x000FFFFFFFFFFFFF) != 0


def load_rows(fixtures):
    return [json.loads(line) for line in Path(fixtures).read_text().splitlines()]


def self_check(rows):
    for row in rows:
        for got, want in zip(outputs(0, row), row["output_bits"]):
            if got != want and not (is_nan(got) and is_nan(want)):
                raise RuntimeError("phase 0 does not reproduce the Sounio fixtures; refusing")
    return True


def frame(rows_bits):
    return "".join("".join(format(x & ((1 << 64) - 1), "016x") for x in row) + "\n" for row in rows_bits)


def reference(phase, fixtures):
    if not isinstance(phase, int) or phase < 0 or phase > 65535:
        raise ValueError("phase must be 0..65535")
    rows = load_rows(fixtures)
    self_check(rows)
    return frame([outputs(phase, row) for row in rows])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", type=int, required=True)
    ap.add_argument("--fixtures", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit("refusing to overwrite " + str(args.out))
    args.out.write_text(reference(args.phase, args.fixtures))


if __name__ == "__main__":
    main()
