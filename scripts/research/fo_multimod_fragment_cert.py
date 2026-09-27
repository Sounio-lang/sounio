#!/usr/bin/env python3
"""Certificate for SounioFoMultimodFragment.lean."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Union


@dataclass(frozen=True)
class Param:
    i: int


@dataclass(frozen=True)
class Bin:
    op: str
    a: object
    b: object


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple


def mul(a, b):
    return Bin("mul", a, b)


def div(a, b):
    return Bin("div", a, b)


body_inf = div(mul(Param(0), Param(1)), Param(2))
body_cl = mul(Param(0), Param(1))
body_css = div(
    Call("fo_infusion_rate", (Param(0), Param(1), Param(2))),
    Call("fo_clearance", (Param(3), Param(4))),
)
items = [
    ("fo_infusion_rate", body_inf),
    ("fo_clearance", body_cl),
    ("fo_css", body_css),
]


def install(items_list):
    return list(items_list)


def lookup(reg, name):
    found = None
    for n, b in reg:
        if n == name:
            found = b
    return found


def subst(args, e):
    if isinstance(e, Param):
        return args[e.i] if e.i < len(args) else e
    if isinstance(e, Bin):
        return Bin(e.op, subst(args, e.a), subst(args, e.b))
    if isinstance(e, Call):
        return Call(e.name, tuple(subst(args, a) for a in e.args))
    return e


def expand_once(reg, e):
    if isinstance(e, Call):
        b = lookup(reg, e.name)
        return subst(list(e.args), b) if b is not None else e
    if isinstance(e, Bin):
        return Bin(e.op, expand_once(reg, e.a), expand_once(reg, e.b))
    return e


def expand_full(reg, e):
    for _ in range(3):
        e = expand_once(reg, e)
    return e


def css_site():
    return div(
        div(mul(Param(0), Param(1)), Param(2)),
        mul(Param(3), Param(4)),
    )


def main() -> int:
    call = Call("fo_css", (Param(0), Param(1), Param(2), Param(3), Param(4)))
    site = css_site()
    reg_l = install(items)
    reg_i = install(items)
    reg_u = install(items + items)
    el, ei, eu = expand_full(reg_l, call), expand_full(reg_i, call), expand_full(reg_u, call)
    fo_var = (
        (Fraction(500, 60) ** 2) * (Fraction(1, 20) ** 2)
        + ((Fraction(4, 5) / 60) ** 2) * (Fraction(10) ** 2)
        + ((-Fraction(400, 300)) ** 2) * (Fraction(3, 10) ** 2)
        + ((-Fraction(20, 3)) ** 2) * (Fraction(1, 10) ** 2)
    )
    checks = [
        ("local", el == site),
        ("import", ei == site),
        ("union", eu == site),
        ("local_eq_import", el == ei),
        ("union_eq_local", eu == el),
        ("var_css", fo_var == Fraction(191, 240)),
    ]
    checks.append(("bundle", all(ok for _, ok in checks)))
    for n, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {n}")
    if not all(ok for _, ok in checks):
        print("FO_MULTIMOD_FRAGMENT_CERT_FAIL")
        return 1
    print(f"FO_MULTIMOD_FRAGMENT_CERT_OK {len(checks)}/{len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
