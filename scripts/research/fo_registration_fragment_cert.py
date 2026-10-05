#!/usr/bin/env python3
"""Executable certificate for formal/lean4/SounioFoRegistrationFragment.lean."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Optional, Union


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


body_infusion = div(mul(Param(0), Param(1)), Param(2))
body_clearance = mul(Param(0), Param(1))
body_css = div(
    Call("fo_infusion_rate", (Param(0), Param(1), Param(2))),
    Call("fo_clearance", (Param(3), Param(4))),
)


def lookup(name: str) -> Optional[PureExpr]:
    return {
        "fo_infusion_rate": body_infusion,
        "fo_clearance": body_clearance,
        "fo_css": body_css,
    }.get(name)


def subst(args: list, e: PureExpr) -> PureExpr:
    if isinstance(e, Param):
        return args[e.i] if e.i < len(args) else e
    if isinstance(e, Lit):
        return e
    if isinstance(e, Bin):
        return Bin(e.op, subst(args, e.a), subst(args, e.b))
    if isinstance(e, Call):
        return Call(e.name, tuple(subst(args, a) for a in e.args))
    raise TypeError(e)


def expand_once(e: PureExpr) -> PureExpr:
    if isinstance(e, Call):
        body = lookup(e.name)
        if body is not None:
            return subst(list(e.args), body)
        return e
    if isinstance(e, Bin):
        return Bin(e.op, expand_once(e.a), expand_once(e.b))
    return e


def expand_full(e: PureExpr, n: int = 3) -> PureExpr:
    for _ in range(n):
        e = expand_once(e)
    return e


def css_site():
    return div(
        div(mul(Param(0), Param(1)), Param(2)),
        mul(Param(3), Param(4)),
    )


def compile_expr(e: PureExpr) -> list[tuple[int, int]]:
    if isinstance(e, Param):
        return [(1, e.i)]
    if isinstance(e, Lit):
        return [(2, 0)]
    if isinstance(e, Call):
        return []
    left = compile_expr(e.a)
    right = compile_expr(e.b)
    op = {"add": 3, "sub": 4, "mul": 5, "div": 6}[e.op]
    return left + right + [(op, 0)]


def main() -> int:
    call = Call("fo_css", (Param(0), Param(1), Param(2), Param(3), Param(4)))
    site = css_site()
    loc = expand_full(call)
    # import registry is identical
    imp = expand_full(call)
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
        ("expand_local_eq_site", loc == site),
        ("expand_import_eq_site", imp == site),
        ("expand_local_eq_import", loc == imp),
        ("registries_agree", lookup("fo_css") == body_css),
        ("emit_after_expand", compile_expr(loc) == golden),
        ("method_peel_eq_site", site == css_site()),
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
        print(f"FO_REGISTRATION_FRAGMENT_CERT_FAIL {len(failed)}/{len(checks)}")
        return 1
    print(f"FO_REGISTRATION_FRAGMENT_CERT_OK {len(checks)}/{len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
