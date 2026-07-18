#!/usr/bin/env python3
"""Arbitrary-precision oracle for stdlib/data/bigrat.sio. Emits `key=decimal` lines that the run-proofs
print, so the compiled big values can be diffed digit-for-digit (the ONLY trustworthy correctness
signal against souc's codegen capacity wall, which can silently miscompute struct-heavy BigInt code).

Usage:
  python3 scripts/research/bigrat_oracle.py         # base cases  (test_bigrat_stdlib.sio)
  python3 scripts/research/bigrat_oracle.py col     # column-reduction cases (test_bigrat_col_stdlib.sio)
"""
import sys
from fractions import Fraction as F

def _emit(key, fr):
    print(f"{key}_num={fr.numerator}")
    print(f"{key}_den={fr.denominator}")

def _first_primes(n):
    ps = []; c = 2
    while len(ps) < n:
        if all(c % p for p in ps if p * p <= c): ps.append(c)
        c += 1
    return ps

def base():
    _emit("add", F(1, 10**20) + F(1, 10**20))   # 1 / 5e19 (den overflows i64)
    _emit("mul", F(1, 10**15) * F(1, 10**15))    # 1 / 10^30
    _emit("reduce", F(6 * 10**39, 4 * 10**39))   # 3 / 2
    _emit("neg", F(-3, 6))                        # -1 / 2
    _emit("sum", F(1, 3) + F(1, 6))               # 1 / 2

def col():
    _emit("col_h5", sum((F(1, k) for k in range(1, 6)), F(0)))                 # 137 / 60
    _emit("col_prime3", F(1,1000000007) + F(1,1000000009) + F(1,1000000021))   # den ~10^27
    s = F(0)
    for p in _first_primes(100): s += F(1, p)                                  # 220-digit denominator
    _emit("col_p100", s)

if __name__ == "__main__":
    (col if (len(sys.argv) > 1 and sys.argv[1] == "col") else base)()
