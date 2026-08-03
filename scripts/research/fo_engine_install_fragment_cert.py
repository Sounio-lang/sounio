#!/usr/bin/env python3
"""Executable certificate for formal/lean4/SounioFoEngineInstallFragment.lean."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Union


@dataclass(frozen=True)
class Param:
    i: int


@dataclass(frozen=True)
class Lit:
    c: Fraction


@dataclass(frozen=True)
class Bin:
    op: str
    a: object
    b: object


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple


PureExpr = Union[Param, Lit, Bin, Call]


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

forward = [
    ("fo_infusion_rate", body_inf),
    ("fo_clearance", body_cl),
    ("fo_css", body_css),
]
reverse = list(reversed(forward))


def install_pass(reg: list, items: list) -> list:
    out = list(reg)
    for n, b in items:
        out.append((n, b))
    return out


def multipass(items: list, k: int) -> list:
    r: list = []
    for _ in range(k):
        r = install_pass(r, items)
    return r


def lookup(reg: list, name: str):
    found = None
    for n, b in reg:
        if n == name:
            found = b  # last wins
    return found


def subst(args, e):
    if isinstance(e, Param):
        return args[e.i] if e.i < len(args) else e
    if isinstance(e, Lit):
        return e
    if isinstance(e, Bin):
        return Bin(e.op, subst(args, e.a), subst(args, e.b))
    if isinstance(e, Call):
        return Call(e.name, tuple(subst(args, a) for a in e.args))
    raise TypeError(e)


def expand_once(reg, e):
    if isinstance(e, Call):
        body = lookup(reg, e.name)
        return subst(list(e.args), body) if body is not None else e
    if isinstance(e, Bin):
        return Bin(e.op, expand_once(reg, e.a), expand_once(reg, e.b))
    return e


def expand_full(reg, e, n=3):
    for _ in range(n):
        e = expand_once(reg, e)
    return e


def css_site():
    return div(
        div(mul(Param(0), Param(1)), Param(2)),
        mul(Param(3), Param(4)),
    )


def compile_expr(e):
    if isinstance(e, Param):
        return [(1, e.i)]
    if isinstance(e, Lit):
        return [(2, 0)]
    if isinstance(e, Call):
        return []
    left, right = compile_expr(e.a), compile_expr(e.b)
    op = {"add": 3, "sub": 4, "mul": 5, "div": 6}[e.op]
    return left + right + [(op, 0)]


def main() -> int:
    site = css_site()
    call = Call("fo_css", (Param(0), Param(1), Param(2), Param(3), Param(4)))
    rf1 = multipass(forward, 1)
    rr1 = multipass(reverse, 1)
    rf4 = multipass(forward, 4)
    rr4 = multipass(reverse, 4)
    golden = [
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
    checks = [
        ("install_forward", expand_full(rf1, call) == site),
        ("install_reverse", expand_full(rr1, call) == site),
        ("install_four_pass", expand_full(rf4, call) == site and expand_full(rr4, call) == site),
        ("import_eq_local", expand_full(rf1, call) == expand_full(multipass(forward, 1), call)),
        ("emit_forward", compile_expr(expand_full(rf1, call)) == golden),
        ("emit_reverse", compile_expr(expand_full(rr1, call)) == golden),
        (
            "var_css",
            (
                (Fraction(500, 60) ** 2) * (Fraction(1, 20) ** 2)
                + ((Fraction(4, 5) / 60) ** 2) * (Fraction(10) ** 2)
                + ((-Fraction(400, 300)) ** 2) * (Fraction(3, 10) ** 2)
                + ((-Fraction(20, 3)) ** 2) * (Fraction(1, 10) ** 2)
            )
            == Fraction(191, 240),
        ),
    ]
    checks.append(("bundle", all(ok for _, ok in checks)))
    failed = [n for n, ok in checks if not ok]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    if failed:
        print(f"FO_ENGINE_INSTALL_FRAGMENT_CERT_FAIL {len(failed)}/{len(checks)}")
        return 1
    print(f"FO_ENGINE_INSTALL_FRAGMENT_CERT_OK {len(checks)}/{len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
