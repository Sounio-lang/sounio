#!/usr/bin/env python3
"""Independent exact oracle for the bounded synthetic D1 associator."""

from fractions import Fraction


def mediate(left: Fraction, right: Fraction) -> Fraction:
    return (2 * left + right) / 3


def main() -> int:
    a = Fraction(3, 10)
    b = Fraction(3, 5)
    c = Fraction(9, 10)

    left_intermediate = mediate(a, b)
    right_intermediate = mediate(b, c)
    left_grouped = mediate(left_intermediate, c)
    right_grouped = mediate(a, right_intermediate)
    associator = left_grouped - right_grouped

    assert left_intermediate == Fraction(2, 5)
    assert right_intermediate == Fraction(7, 10)
    assert left_grouped == Fraction(17, 30)
    assert right_grouped == Fraction(13, 30)
    assert associator == Fraction(2, 15)

    raw_left = (2550, 4500)
    raw_right = (1950, 4500)
    raw_numerator = raw_left[0] * raw_right[1] - raw_right[0] * raw_left[1]
    raw_denominator = raw_left[1] * raw_right[1]
    assert raw_numerator == 2_700_000
    assert raw_denominator == 20_250_000
    assert Fraction(raw_numerator, raw_denominator) == associator

    associative_left = (a + b) + c
    associative_right = a + (b + c)
    assert associative_left == associative_right == Fraction(9, 5)

    expanded_states = {
        1: ("((a,b),c)", left_grouped),
        2: ("(a,(b,c))", right_grouped),
    }
    frozen_transition = {1: Fraction(17, 30), 2: Fraction(13, 30)}
    assert expanded_states[1][0] != expanded_states[2][0]
    assert set(frozen_transition) == set(expanded_states)
    for grouping_code, (_, observed_output) in expanded_states.items():
        assert frozen_transition[grouping_code] == observed_output

    print("ORACLE_D1_W0 left=17/30 right=13/30 associator=2/15")
    print("ORACLE_D1_W1 associative_control=9/5 associator=0")
    print("ORACLE_D1_W3 expansion=restores-factorability irreducible=false")
    print("DYADIC RELATIONAL ASSOCIATOR D1 ORACLE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
