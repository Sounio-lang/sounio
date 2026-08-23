#!/usr/bin/env python3
"""Emit the Sounio fixture for the pow/tgamma intrinsics from the oracle.

The fixture is generated rather than hand-written for one reason: its expected
values must come from the same model the emitter was transliterated from, and a
hand-copied bit pattern is a transcription error waiting to happen in a place
where the test would then certify the error.

WHY NOT libm. The obvious oracle is the system tgamma, and it is the wrong one.
Measured on this host: glibc's tgamma returns 355687428096000.1 where 17! is the
integer 355687428096000, and is 1 ulp off on 18!, 19!, 21! and 22! as well --
five consecutive exactly-representable factorials. Over the design's bands it
reaches 4.6 ulp. A fixture asserting agreement with libm would fail a correct
implementation on exactly the rows that matter most, the ones with an exact
answer. Expected values therefore come from the oracle, which decides against
mpmath at 60 digits.

Tolerances are per row, taken from the design's own classification: rows with an
exactly representable answer are asserted bit-exact, the rest to the published
bound (pow <= 2 ulp, tgamma <= 5 ulp). Comparison is by integer distance between
bit patterns, which IS the ulp distance for two finite values of the same sign,
and the generator refuses to emit a tolerance row where that assumption fails.
"""
import importlib.util
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORACLE = os.path.join(ROOT, "scripts", "research", "pow_tgamma_oracle.py")
OUT = os.path.join(ROOT, "tests", "run-pass", "ffi_pow_tgamma_vectors.sio")


def load_oracle():
    spec = importlib.util.spec_from_file_location("oracle", ORACLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bits(x):
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def value(b):
    return struct.unpack("<d", struct.pack("<Q", b))[0]


def ulp_distance_is_meaningful(a_bits, b_bits):
    """Integer bit distance equals ulp distance only for same-sign finite values."""
    a, b = value(a_bits), value(b_bits)
    if a != a or b != b:            # NaN
        return False
    if a in (float("inf"), float("-inf")) or b in (float("inf"), float("-inf")):
        return False
    return (a_bits >> 63) == (b_bits >> 63)


def emit_rows(name, rows):
    """rows: list of (x_bits, y_bits_or_None, expected_bits, tol_ulp, note)"""
    lines = []
    for x, y, e, tol, note in rows:
        if tol > 0 and not ulp_distance_is_meaningful(e, e):
            raise SystemExit(f"{name}: row {note} wants a tolerance but its expected value "
                             "is NaN or infinite, where bit distance is not ulp distance")
        lines.append((x, y, e, tol, note))
    return lines


def main():
    oracle = load_oracle()
    pow_rows = []
    for _, xb, yb, eb, note in oracle.POW_VECTORS:
        exact = ulp_distance_is_meaningful(eb, eb) and value(eb) == int(value(eb)) \
            if abs(value(eb)) < 2**53 else False
        pow_rows.append((xb, yb, eb, 0 if exact else 2, note))
    tg_rows = []
    for _, xb, eb, note in oracle.TGAMMA_VECTORS:
        exact = "exact" in note.lower() or "MUST be" in note
        tg_rows.append((xb, None, eb, 0 if exact else 5, note))
    print(f"pow vectors:    {len(pow_rows)}  ({sum(1 for r in pow_rows if r[3]==0)} bit-exact)")
    print(f"tgamma vectors: {len(tg_rows)}  ({sum(1 for r in tg_rows if r[3]==0)} bit-exact)")
    print(f"would write:    {OUT}")
    print()
    print("Emitters for ffi_pow and ffi_tgamma do not exist yet, so this run only")
    print("reports what it WOULD emit. Committing a fixture that cannot pass would")
    print("put a red test in the tree and invite someone to weaken it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
