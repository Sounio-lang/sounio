#!/usr/bin/env python3
"""Integer-only origin/GUM oracle. No floating point, no math.sqrt.

JCGM 100:2008 first-order combined uncertainty:
  independent: u'^2 = ua^2 + ub^2
  same-origin envelope (founder form): u' = ua + ub

Squares and sums are integers. The 3-4-5 triple is the integer identity
that names u'=5 without taking a square root in this file.
"""

from __future__ import annotations


def fuse(a: int, b: int) -> int:
    if a != 0 and a == b:
        return a
    return 0


def correlated(a: int, b: int) -> int:
    if a != 0 and a == b:
        return 1
    return 0


def add_u(ua: int, ub: int, oa: int, ob: int) -> tuple[str, int]:
    if correlated(oa, ob) == 1:
        return ("sum", ua + ub)
    return ("rss_sq", ua * ua + ub * ub)


def main() -> None:
    kind, val = add_u(3, 3, 1, 1)
    assert kind == "sum" and val == 6
    assert fuse(1, 1) == 1
    print("ORACLE correlated_add ua=3 ub=3 oa=1 ob=1 -> u=6 origin=1")

    kind, val = add_u(3, 4, 1, 2)
    assert kind == "rss_sq" and val == 25
    assert fuse(1, 2) == 0
    assert correlated(1, 2) == 0
    print("ORACLE independent_add ua=3 ub=4 oa=1 ob=2 -> u2=25 origin=0 no-fire")

    kind, val = add_u(3, 3, 1, 1)
    assert kind == "sum" and val == 6
    print("ORACLE esub_same ua=3 ub=3 oa=1 ob=1 -> u=6 (envelope, not |3-3|=0)")

    assert fuse(fuse(1, 2), 1) == 0
    print("ORACLE mixed (a+b)+a origin=0 silent")

    assert fuse(0, 1) == 0
    assert fuse(1, 0) == 0
    assert fuse(0, 0) == 0
    assert correlated(0, 0) == 0
    print("ORACLE sentinel 0 never fires")

    # 3-4-5 names the independent u without a square root.
    assert 3 * 3 + 4 * 4 == 5 * 5
    print("ORACLE pythagorean 3^2+4^2=5^2")
    print("ORACLE_OK")


if __name__ == "__main__":
    main()
