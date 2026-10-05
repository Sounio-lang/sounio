#!/usr/bin/env python3
"""Arbitrary-precision oracle for the R3 Beta-confidence composition rule.

mpmath is not assumed. Moments of Beta(α, β) with rational parameters are
exact in fractions.Fraction. Square roots (GUM combined u) use decimal
with 80 digits, then round to IEEE f64 for the Sounio witness.

This file is the independent calculator. The Sounio implementation must
match the printed goldens. Re-run:

    python3 scripts/dev/beta_confidence_oracle.py
"""

from __future__ import annotations

import math
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 80

W_PRIOR = 1  # uniform Beta(1,1) is zero evidence


def poch(a: Fraction, k: int) -> Fraction:
    p = Fraction(1)
    for i in range(k):
        p *= a + i
    return p


def beta_moment(alpha, beta, k: int) -> Fraction:
    a, b = Fraction(alpha), Fraction(beta)
    return poch(a, k) / poch(a + b, k)


def moment_match(mu: Fraction, var: Fraction) -> tuple[Fraction, Fraction, Fraction]:
    n = mu * (1 - mu) / var - 1
    return n * mu, n * (1 - mu), n


def product_match(a1, b1, a2, b2) -> dict:
    m1 = beta_moment(a1, b1, 1) * beta_moment(a2, b2, 1)
    m2 = beta_moment(a1, b1, 2) * beta_moment(a2, b2, 2)
    var = m2 - m1 * m1
    al, be, n = moment_match(m1, var)
    m3_true = beta_moment(a1, b1, 3) * beta_moment(a2, b2, 3)
    m3_fit = beta_moment(al, be, 3)
    m4_true = beta_moment(a1, b1, 4) * beta_moment(a2, b2, 4)
    m4_fit = beta_moment(al, be, 4)
    rel3 = abs(m3_fit - m3_true) / m3_true if m3_true else Fraction(0)
    rel4 = abs(m4_fit - m4_true) / m4_true if m4_true else Fraction(0)
    return {
        "alpha": al,
        "beta": be,
        "n": n,
        "mu": m1,
        "var": var,
        "rel3": rel3,
        "rel4": rel4,
    }


def fuse_independent(a1, b1, a2, b2, w=W_PRIOR):
    return Fraction(a1) + Fraction(a2) - w, Fraction(b1) + Fraction(b2) - w


def gum_u_add(ua, ub) -> float:
    x = Decimal(str(ua)) ** 2 + Decimal(str(ub)) ** 2
    return float(x.sqrt())


def gum_mul_vars(va, ua, vb, ub):
    gum = vb * vb * ua * ua + va * va * ub * ub
    dropped = ua * ua * ub * ub
    return gum, gum + dropped, dropped


def main() -> None:
    print("# BETA_CONFIDENCE_ORACLE goldens")
    print(f"# W_PRIOR={W_PRIOR}")
    print("kind\tname\tfield\tvalue")

    print(f"gum_add\t1+2_u0.1_u0.2\tvalue\t{1.0 + 2.0}")
    print(f"gum_add\t1+2_u0.1_u0.2\tu\t{gum_u_add(0.1, 0.2)!r}")

    gum, exact, dropped = gum_mul_vars(2.0, 0.1, 3.0, 0.2)
    print(f"gum_mul\t2u0.1_x_3u0.2\tgum_var\t{gum!r}")
    print(f"gum_mul\t2u0.1_x_3u0.2\texact_var\t{exact!r}")
    print(f"gum_mul\t2u0.1_x_3u0.2\tdropped\t{dropped!r}")
    print(f"gum_mul\t2u0.1_x_3u0.2\tdropped_rel\t{dropped / exact!r}")

    print("same_origin\tBeta(1,1)+self\talpha\t1")
    print("same_origin\tBeta(1,1)+self\tbeta\t1")
    print("naive_add\tBeta(1,1)+self\talpha\t2")
    print("naive_add\tBeta(1,1)+self\tbeta\t2")
    print("same_origin\tBeta(1000,1000)+self\talpha\t1000")
    print("same_origin\tBeta(1000,1000)+self\tbeta\t1000")

    a, b = fuse_independent(1000, 1000, 1000, 1000)
    print(f"fuse_indep\tBeta(1000,1000)x2\talpha\t{float(a)!r}")
    print(f"fuse_indep\tBeta(1000,1000)x2\tbeta\t{float(b)!r}")
    a, b = fuse_independent(1, 1, 1, 1)
    print(f"fuse_indep\tBeta(1,1)x2\talpha\t{float(a)!r}")
    print(f"fuse_indep\tBeta(1,1)x2\tbeta\t{float(b)!r}")
    a, b = fuse_independent(2, 1, 2, 1)
    print(f"fuse_indep\tBeta(2,1)x2\talpha\t{float(a)!r}")
    print(f"fuse_indep\tBeta(2,1)x2\tbeta\t{float(b)!r}")

    cases = [
        ("2,2x3,1", 2, 2, 3, 1),
        ("2,2x2,2", 2, 2, 2, 2),
        ("1,1x1,1", 1, 1, 1, 1),
        ("1000,1000x1000,1000", 1000, 1000, 1000, 1000),
        ("5,1x1,5", 5, 1, 1, 5),
        ("2,8x8,2", 2, 8, 8, 2),
    ]
    for name, a1, b1, a2, b2 in cases:
        r = product_match(a1, b1, a2, b2)
        print(f"and_match\t{name}\talpha\t{float(r['alpha'])!r}")
        print(f"and_match\t{name}\tbeta\t{float(r['beta'])!r}")
        print(f"and_match\t{name}\trel3\t{float(r['rel3'])!r}")
        print(f"and_match\t{name}\trel4\t{float(r['rel4'])!r}")
        print(f"and_match\t{name}\talpha_frac\t{r['alpha']}")
        print(f"and_match\t{name}\tbeta_frac\t{r['beta']}")

    print("scalar\tc500_n2\talpha\t1.0")
    print("scalar\tc500_n2\tbeta\t1.0")
    print("scalar\tc500_n2000\talpha\t1000.0")
    print("scalar\tc500_n2000\tbeta\t1000.0")
    print("scalar\tc900_n2\talpha\t1.8")
    print("scalar\tc900_n2\tbeta\t0.2")
    print(f"sqrt05\tgum_u\tvalue\t{math.sqrt(0.05)!r}")


if __name__ == "__main__":
    main()
