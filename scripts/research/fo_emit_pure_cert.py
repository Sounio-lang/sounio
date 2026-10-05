#!/usr/bin/env python3
"""Executable certificate for formal/lean4/SounioFoEmitPure.lean.

L2 pure-emit: fo_bc_compile_expr pure fragment emits oral Css RPN.
"""
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
    op: str  # add|sub|mul|div
    a: object
    b: object


PureExpr = Union[Param, Lit, Bin]

OP_PARAM, OP_CONST, OP_ADD, OP_SUB, OP_MUL, OP_DIV = 1, 2, 3, 4, 5, 6


def compile_expr(e: PureExpr) -> list[tuple[int, int]]:
    if isinstance(e, Param):
        return [(OP_PARAM, e.i)]
    if isinstance(e, Lit):
        return [(OP_CONST, 0)]
    left = compile_expr(e.a)  # type: ignore[arg-type]
    right = compile_expr(e.b)  # type: ignore[arg-type]
    op = {"add": OP_ADD, "sub": OP_SUB, "mul": OP_MUL, "div": OP_DIV}[e.op]
    return left + right + [(op, 0)]


def mul(a, b):
    return Bin("mul", a, b)


def div(a, b):
    return Bin("div", a, b)


def css_site():
    return div(
        div(mul(Param(0), Param(1)), Param(2)),
        mul(Param(3), Param(4)),
    )


def fo_infusion_rate(F, Dose, tau):
    return div(mul(F, Dose), tau)


def fo_clearance(CL0, eEta):
    return mul(CL0, eEta)


def fo_css(F, Dose, tau, CL0, eEta):
    return div(fo_infusion_rate(F, Dose, tau), fo_clearance(CL0, eEta))


def main() -> int:
    site = css_site()
    imp = fo_css(Param(0), Param(1), Param(2), Param(3), Param(4))
    golden = [
        (OP_PARAM, 0),
        (OP_PARAM, 1),
        (OP_MUL, 0),
        (OP_PARAM, 2),
        (OP_DIV, 0),
        (OP_PARAM, 3),
        (OP_PARAM, 4),
        (OP_MUL, 0),
        (OP_DIV, 0),
    ]
    checks: list[tuple[str, bool]] = []
    checks.append(("import_expand_eq_site", imp == site))
    checks.append(("emit_site_eq_golden", compile_expr(site) == golden))
    checks.append(("emit_import_eq_golden", compile_expr(imp) == golden))
    checks.append(("emit_method_eq_golden", compile_expr(site) == golden))

    fo_var = (
        (Fraction(500, 60) ** 2) * (Fraction(1, 20) ** 2)
        + ((Fraction(4, 5) / 60) ** 2) * (Fraction(10) ** 2)
        + ((-Fraction(400, 300)) ** 2) * (Fraction(3, 10) ** 2)
        + ((-Fraction(20, 3)) ** 2) * (Fraction(1, 10) ** 2)
    )
    checks.append(("var_css", fo_var == Fraction(191, 240)))
    checks.append(("l2_pure_emit_bundle", all(ok for _, ok in checks)))

    failed = [n for n, ok in checks if not ok]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    if failed:
        print(f"FO_EMIT_PURE_CERT_FAIL {len(failed)}/{len(checks)}")
        return 1
    print(f"FO_EMIT_PURE_CERT_OK {len(checks)}/{len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
