#!/usr/bin/env python3
"""Arbitrary-precision oracle for stdlib/data/bigrat.sio — emits `key=decimal` lines matching
tests/stdlib/data/test_bigrat_stdlib.sio, so the compiled run-proof's printed big values can be
diffed digit-for-digit (the ONLY trustworthy correctness signal against the codegen capacity wall)."""
from fractions import Fraction as F
def emit(key, fr):
    print(f"{key}_num={fr.numerator}")
    print(f"{key}_den={fr.denominator}")
emit("add", F(1,10**20) + F(1,10**20))     # 1 / 50000000000000000000  (den overflows i64)
emit("mul", F(1,10**15) * F(1,10**15))      # 1 / 10^30                 (den overflows i64)
emit("reduce", F(6*10**39, 4*10**39))       # 3 / 2                     (bignum gcd -> small)
emit("neg", F(-3, 6))                        # -1 / 2                    (sign on numerator)
emit("sum", F(1,3) + F(1,6))                 # 1 / 2
