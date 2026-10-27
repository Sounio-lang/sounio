"""Exact rational oracle for float_literal_long_decimal_midpoint_2826.sio.

Every expected bit pattern in that test is derived here from integer
midpoints between adjacent binary64 values, using Fraction arithmetic only
(no float(), no strtod): a midpoint is an odd multiple of half an ulp, its
decimal expansion terminates, and the cases step one decimal quantum to
either side of it. Run: python3 tests/oracles/float_literal_long_decimal_oracle.py
Output rows: label, signed f64 bits, decimal literal.
"""
from fractions import Fraction


def floor_log2(fr: Fraction) -> int:
    e = fr.numerator.bit_length() - fr.denominator.bit_length()
    while Fraction(2) ** e > fr:
        e -= 1
    while Fraction(2) ** (e + 1) <= fr:
        e += 1
    return e


def round_ratio_to_int(fr: Fraction) -> int:
    q, r = divmod(fr.numerator, fr.denominator)
    twice = 2 * r
    if twice > fr.denominator or (twice == fr.denominator and q % 2 == 1):
        q += 1
    return q


def round_f64_bits(fr: Fraction) -> int:
    if fr == 0:
        return 0
    sign = 0
    if fr < 0:
        sign = 1 << 63
        fr = -fr
    if fr < Fraction(2) ** -1022:
        m = round_ratio_to_int(fr / (Fraction(2) ** -1074))
        return sign | m if m < (1 << 52) else sign | (1 << 52)
    e = floor_log2(fr)
    m = round_ratio_to_int(fr / (Fraction(2) ** (e - 52)))
    if m == 1 << 53:
        m >>= 1
        e += 1
    if e > 1023:
        return sign | (0x7FF << 52)
    return sign | ((e + 1023) << 52) | (m - (1 << 52))


def signed(bits: int) -> int:
    return bits - (1 << 64) if bits >= (1 << 63) else bits


def lead_exp10(fr: Fraction) -> int:
    e = 0
    while Fraction(10) ** (e + 1) <= fr:
        e += 1
    while Fraction(10) ** e > fr:
        e -= 1
    return e


def sci(fr: Fraction, digits: int) -> str:
    lead = lead_exp10(fr)
    coef = fr / (Fraction(10) ** (lead - digits + 1))
    if coef.denominator != 1:
        raise ValueError(f"not exact at {digits} digits")
    s = str(coef.numerator).rjust(digits, "0").rstrip("0") or "0"
    return s[0] + ("." + s[1:] if len(s) > 1 else ".0") + f"e{lead}"


def sci_exact(fr: Fraction) -> str:
    for digits in range(1, 1201):
        lead = lead_exp10(fr)
        if (fr / (Fraction(10) ** (lead - digits + 1))).denominator == 1:
            return sci(fr, digits)
    raise ValueError("no exact digits <= 1200")


def around(fr: Fraction, digits: int = 60):
    q = Fraction(10) ** (lead_exp10(fr) - digits + 1)
    return sci_exact(fr - q), sci_exact(fr), sci_exact(fr + q)


def parse_decimal(txt: str) -> Fraction:
    mant, _, ex = txt.partition("e")
    neg = mant.startswith("-")
    mant = mant.lstrip("-")
    ip, _, fp = mant.partition(".")
    fr = Fraction(int((ip + fp) or "0")) * Fraction(10) ** ((int(ex) if ex else 0) - len(fp))
    return -fr if neg else fr


def main() -> None:
    cases = []
    mid_one = Fraction(2**53 + 1, 2**53)
    cases += list(zip(("one_below", "one_at", "one_above"), around(mid_one)))
    cases.append(("one_at_long_zero", sci(mid_one, 60)[:-2] + "0" * 840 + "e0"))
    cases.append(("one_above_far_tail", sci(mid_one + Fraction(10) ** -900, 901)))
    cases.append(("founder_midpoint_plus",
                  "1.43425768535078390630843614417244680225849151611328125000001"))
    cases += list(zip(("half_sub_below", "half_sub_at", "half_sub_above"),
                       around(Fraction(1, 2**1075))))
    cases += list(zip(("norm_below", "norm_at", "norm_above"),
                       around(Fraction(2**53 - 1, 2**1075))))
    cases += list(zip(("over_below", "over_at", "over_above"),
                       around(Fraction(2**54 - 1) * Fraction(2) ** 970)))
    # Positive-exponent fast-window boundary: sig needs trailing-zero growth
    # into the exact 10^k window. 1.0e24 is the case the #2904 `keep` form
    # read as exactly 1e23 (counted zeros, multiplied by the clamped power).
    cases += [("pow10_e24", "1.0e24"), ("pow10_e23", "1.0e23"),
              ("pow10_25e25", "2.5e25"), ("pow10_120e30", "1.20e30"),
              ("pow10_5e24", "5.0e24")]
    for label, txt in cases:
        print(f"{label} {signed(round_f64_bits(parse_decimal(txt)))} {txt}")
    neg = parse_decimal("1.43425768535078390630843614417244680225849151611328125000001")
    print(f"founder_negative {signed(round_f64_bits(-neg))} "
          "-1.43425768535078390630843614417244680225849151611328125000001")


if __name__ == "__main__":
    main()
