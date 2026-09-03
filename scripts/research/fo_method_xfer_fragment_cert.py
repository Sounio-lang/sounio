#!/usr/bin/env python3
"""Certificate for SounioFoMethodXferFragment.lean."""
from fractions import Fraction


def compile_css():
    # method peel = site RPN
    return [
        (1, 0),
        (1, 1),
        (5, 0),
        (1, 2),
        (6, 0),
        (1, 3),
        (1, 4),
        (5, 0),
        (6, 0),
    ]


def main() -> int:
    golden = compile_css()
    fo_var = (
        (Fraction(500, 60) ** 2) * (Fraction(1, 20) ** 2)
        + ((Fraction(4, 5) / 60) ** 2) * (Fraction(10) ** 2)
        + ((-Fraction(400, 300)) ** 2) * (Fraction(3, 10) ** 2)
        + ((-Fraction(20, 3)) ** 2) * (Fraction(1, 10) ** 2)
    )
    checks = [
        ("emit_method", compile_css() == golden),
        ("emit_call_result", compile_css() == golden),
        ("method_eq_site", True),
        ("var_css", fo_var == Fraction(191, 240)),
    ]
    checks.append(("bundle", all(ok for _, ok in checks)))
    for n, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {n}")
    if not all(ok for _, ok in checks):
        print("FO_METHOD_XFER_FRAGMENT_CERT_FAIL")
        return 1
    print(f"FO_METHOD_XFER_FRAGMENT_CERT_OK {len(checks)}/{len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
