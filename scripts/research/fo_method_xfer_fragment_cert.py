#!/usr/bin/env python3
"""Executable certificate for formal/lean4/SounioFoMethodXferFragment.lean.

L2 method FO_XFER: after the op17 LOAD_PARAM_FIELD peel of `self.cl0`, the
body of `Pk.css` -- and of the call-result form `make_pk(cl0, v0).css(...)` --
is the oral Css site expression, and it emits the golden FO RPN.

Param layout used by FO after the peel:
    p0 = F, p1 = Dose, p2 = tau, p3 = CL0 (from self.cl0), p4 = eEta

Structure mirrors scripts/research/fo_emit_pure_cert.py: the expression under
test is *built* from the science model, and it is compared against a golden
RPN list and a site expression that are both written out independently by
hand. Nothing here compares a function against itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Union

# Opcodes, as in the Lean file.
OP_PARAM, OP_MUL, OP_DIV = 1, 5, 6


@dataclass(frozen=True)
class Param:
    i: int


@dataclass(frozen=True)
class Bin:
    op: str  # mul | div
    a: object
    b: object


PureExpr = Union[Param, Bin]


def mul(a: PureExpr, b: PureExpr) -> PureExpr:
    return Bin("mul", a, b)


def div(a: PureExpr, b: PureExpr) -> PureExpr:
    return Bin("div", a, b)


def compile_expr(e: PureExpr) -> list[tuple[int, int]]:
    """FO bytecode emit: post-order RPN, one instruction per node."""
    if isinstance(e, Param):
        return [(OP_PARAM, e.i)]
    left = compile_expr(e.a)  # type: ignore[arg-type]
    right = compile_expr(e.b)  # type: ignore[arg-type]
    op = {"mul": OP_MUL, "div": OP_DIV}[e.op]
    return left + right + [(op, 0)]


# --- independently written oracles -------------------------------------------
# The site expression as it appears at the call site in the science model,
# written out directly rather than derived from the method under test.
def css_site() -> PureExpr:
    return div(
        div(mul(Param(0), Param(1)), Param(2)),
        mul(Param(3), Param(4)),
    )


# The golden RPN, written out by hand from the Lean `cssSiteProg` literal.
# Raw opcode numbers on purpose: a wrong OP_* constant must fail this.
CSS_SITE_PROG: list[tuple[int, int]] = [
    (1, 0),  # param F
    (1, 1),  # param Dose
    (5, 0),  # mul
    (1, 2),  # param tau
    (6, 0),  # div   -> rate
    (1, 3),  # param CL0 (peeled self.cl0)
    (1, 4),  # param eEta
    (5, 0),  # mul   -> clearance
    (6, 0),  # div   -> Css
]


# --- the science model under test --------------------------------------------
def fo_infusion_rate(f: PureExpr, dose: PureExpr, tau: PureExpr) -> PureExpr:
    return div(mul(f, dose), tau)


def fo_clearance(cl0: PureExpr, e_eta: PureExpr) -> PureExpr:
    return mul(cl0, e_eta)


@dataclass(frozen=True)
class Pk:
    """Receiver struct `Pk { cl0, v0 }`; v0 is carried but unused by css."""

    cl0: PureExpr
    v0: PureExpr


def make_pk(cl0: PureExpr, v0: PureExpr) -> Pk:
    return Pk(cl0, v0)


def pk_css(recv: Pk, f: PureExpr, dose: PureExpr, tau: PureExpr,
           e_eta: PureExpr) -> PureExpr:
    """Body of `Pk.css`: rate(f, dose, tau) / clearance(self.cl0, eEta)."""
    return div(fo_infusion_rate(f, dose, tau), fo_clearance(recv.cl0, e_eta))


# Free f64 args land on p0..p2 and p4; op17 peels self.cl0 onto p3. v0 is a
# live struct channel that css never reads, so it gets the next free slot and
# must not appear in the emitted program.
def method_css_peeled() -> PureExpr:
    recv = Pk(cl0=Param(3), v0=Param(5))
    return pk_css(recv, Param(0), Param(1), Param(2), Param(4))


def call_result_css_peeled() -> PureExpr:
    """`make_pk(cl0, v0).css(f, dose, tau, eta)` after xfer expansion."""
    recv = make_pk(Param(3), Param(5))
    return pk_css(recv, Param(0), Param(1), Param(2), Param(4))


def compile_css() -> list[tuple[int, int]]:
    """FO program emitted for the peeled method body."""
    return compile_expr(method_css_peeled())


def main() -> int:
    fo_var = (
        (Fraction(500, 60) ** 2) * (Fraction(1, 20) ** 2)
        + ((Fraction(4, 5) / 60) ** 2) * (Fraction(10) ** 2)
        + ((-Fraction(400, 300)) ** 2) * (Fraction(3, 10) ** 2)
        + ((-Fraction(20, 3)) ** 2) * (Fraction(1, 10) ** 2)
    )
    checks = [
        ("emit_method", compile_css() == CSS_SITE_PROG),
        ("emit_call_result",
         compile_expr(call_result_css_peeled()) == CSS_SITE_PROG),
        ("method_eq_site", method_css_peeled() == css_site()),
        ("call_result_eq_method",
         call_result_css_peeled() == method_css_peeled()),
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
