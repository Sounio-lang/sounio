#!/usr/bin/env python3
"""
Oracle for tests/run-pass/sedenion_ratbig_channel.sio.

Computes the canonical sedenion channel product exactly over Q using
Python's fractions.Fraction, for cross-checking against the Sounio
exact-rational-over-bigint implementation.

Channel: a = alpha*e3 + beta*e10, b = gamma*e6 + delta*e15
  r5  = alpha*gamma + beta*delta
  r12 = alpha*delta + beta*gamma

Emits the same 'R5 <num> <den>' / 'R12 <num> <den>' lines the .sio test
emits, in reduced (num, den) form with den > 0, so the two outputs can be
diffed directly.
"""
from fractions import Fraction as F


def channel(alpha, beta, gamma, delta):
    r5 = alpha * gamma + beta * delta
    r12 = alpha * delta + beta * gamma
    return r5, r12


def emit(label, alpha, beta, gamma, delta):
    r5, r12 = channel(alpha, beta, gamma, delta)
    print("CASE %s" % label)
    print("R5 %d %d" % (r5.numerator, r5.denominator))
    print("R12 %d %d" % (r12.numerator, r12.denominator))


def main():
    # Case 1: locus point (1,1,1,-1) -- integers.
    emit("1_locus_unit", F(1), F(1), F(1), F(-1))

    # Case 2: locus point (t,t,s,-s) with large rationals.
    t = F(123456789, 7)
    s = F(99999999, 13)
    emit("2_locus_large", t, t, s, -s)

    # Case 3: off-locus rational point -- expect nonzero reduced r5,r12.
    emit("3_offlocus", F(1, 2), F(1, 3), F(1, 5), F(1, 7))


if __name__ == "__main__":
    main()
