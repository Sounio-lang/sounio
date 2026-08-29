#!/usr/bin/env python3
"""Python arbitrary-precision oracle for stdlib/math/bignat.sio.

Computes, with Python's native (exact) big integers, the same cases exercised
by tests/run-pass/bignat_selftest*.sio, and emits matching `OP <name> <value>`
lines so the two outputs can be diffed directly.

Usage:
    python3 scripts/research/bignat_oracle.py            # all cases
    python3 scripts/research/bignat_oracle.py core        # just the core group
    python3 scripts/research/bignat_oracle.py signed      # just the signed group
    python3 scripts/research/bignat_oracle.py eq_true|eq_false|iszero_true|iszero_false

Cross-check with the compiled selftests:
    SOUNIO_STDLIB_PATH=$PWD/stdlib ./bin/souc run tests/run-pass/bignat_selftest.sio \
        | grep '^OP ' > /tmp/souc_core.txt
    python3 scripts/research/bignat_oracle.py core > /tmp/py_core.txt
    diff /tmp/souc_core.txt /tmp/py_core.txt   # expect: no output (exact match)
"""
import math
import sys


def bool_str(b: bool) -> str:
    return "1" if b else "0"


def emit_core():
    p30 = 10**30
    p20 = 10**20
    print(f"OP add {p30 + p20}")
    print(f"OP sub {p30 - p20}")
    print(f"OP mul {p30 * p30}")
    p60 = p30 * p30
    print(f"OP divmod_q {p60 // p30}")
    print(f"OP divmod_r {p60 % p30}")
    print(f"OP gcd {math.gcd(10**40, 6 * 10**39)}")
    print(f"OP cmp {1 if p30 > p20 else (0 if p30 == p20 else -1)}")


def emit_divmod_rem():
    p30 = 10**30
    p60 = p30 * p30
    print(f"OP divmod_rem_q {p60 // 7}")
    print(f"OP divmod_rem_r {p60 % 7}")


def emit_signed():
    p30 = 10**30
    p20 = 10**20
    print(f"OP sub_neg {p20 - p30}")
    print(f"OP add_signed {-5 + 3}")
    print(f"OP sub_signed {3 - 5}")
    print(f"OP mul_signed {-7 * 4}")


def emit_eq_true():
    p30 = 10**30
    print(f"OP eq_true {bool_str(p30 == p30)}")


def emit_eq_false():
    p30 = 10**30
    p20 = 10**20
    print(f"OP eq_false {bool_str(p30 == p20)}")


def emit_iszero_true():
    print(f"OP is_zero_true {bool_str(0 == 0)}")


def emit_iszero_false():
    p30 = 10**30
    print(f"OP is_zero_false {bool_str(p30 == 0)}")


GROUPS = {
    "core": emit_core,
    "divmod_rem": emit_divmod_rem,
    "signed": emit_signed,
    "eq_true": emit_eq_true,
    "eq_false": emit_eq_false,
    "iszero_true": emit_iszero_true,
    "iszero_false": emit_iszero_false,
}


def main(argv):
    names = argv[1:] if len(argv) > 1 else list(GROUPS.keys())
    for name in names:
        if name not in GROUPS:
            print(f"unknown group: {name} (choices: {', '.join(GROUPS)})", file=sys.stderr)
            return 1
        GROUPS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
