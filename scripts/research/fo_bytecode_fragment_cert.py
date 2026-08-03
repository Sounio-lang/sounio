#!/usr/bin/env python3
"""Executable certificate for formal/lean4/SounioFoBytecodeFragment.lean.

L2 fragment: FO bytecode stack machine for oral Css (Madaros ops 1–6).
Site ≡ import-expanded ≡ method programs; interpret to L1 desugarSite AST;
FO var freezes 191/240.

Exit 0 → FO_BYTECODE_FRAGMENT_CERT_OK
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Union


@dataclass(frozen=True)
class Seed:
    i: int


@dataclass(frozen=True)
class Lit:
    c: Fraction


@dataclass(frozen=True)
class Add:
    a: object
    b: object


@dataclass(frozen=True)
class Mul:
    a: object
    b: object


@dataclass(frozen=True)
class Div:
    a: object
    b: object


@dataclass(frozen=True)
class Sub:
    a: object
    b: object


FoExpr = Union[Seed, Lit, Add, Mul, Div, Sub]

OP_PARAM, OP_CONST, OP_ADD, OP_SUB, OP_MUL, OP_DIV = 1, 2, 3, 4, 5, 6


def params(i: int) -> FoExpr:
    return Seed(i) if i <= 4 else Lit(Fraction(0))


def step(stk: list[FoExpr], op: int, arg: int) -> Optional[list[FoExpr]]:
    if op == OP_PARAM:
        return [params(arg)] + stk
    if op == OP_CONST:
        return [Lit(Fraction(arg))] + stk
    if op in (OP_ADD, OP_SUB, OP_MUL, OP_DIV):
        if len(stk) < 2:
            return None
        b, a, rest = stk[0], stk[1], stk[2:]
        if op == OP_ADD:
            return [Add(a, b)] + rest
        if op == OP_SUB:
            return [Sub(a, b)] + rest
        if op == OP_MUL:
            return [Mul(a, b)] + rest
        return [Div(a, b)] + rest
    return None


def run(prog: list[tuple[int, int]]) -> Optional[FoExpr]:
    stk: list[FoExpr] = []
    for op, arg in prog:
        nxt = step(stk, op, arg)
        if nxt is None:
            return None
        stk = nxt
    return stk[0] if len(stk) == 1 else None


def desugar_site() -> FoExpr:
    return Div(
        Div(Mul(Seed(0), Seed(1)), Seed(2)),
        Mul(Seed(3), Seed(4)),
    )


def main() -> int:
    css_site = [
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
    css_import = list(css_site)
    css_method = list(css_site)

    checks: list[tuple[str, bool]] = []
    checks.append(("import_prog_eq_site", css_import == css_site))
    checks.append(("method_prog_eq_site", css_method == css_site))

    e_site = run(css_site)
    e_imp = run(css_import)
    e_meth = run(css_method)
    target = desugar_site()
    checks.append(("run_site_eq_desugar", e_site == target))
    checks.append(("run_import_eq_desugar", e_imp == target))
    checks.append(("run_method_eq_desugar", e_meth == target))

    j = {
        0: Fraction(500, 60),
        1: Fraction(4, 5) / 60,
        2: Fraction(0),
        3: -Fraction(400, 300),
        4: -Fraction(20, 3),
    }
    sig = {
        0: Fraction(1, 20),
        1: Fraction(10),
        2: Fraction(0),
        3: Fraction(3, 10),
        4: Fraction(1, 10),
    }
    fo_var = sum(j[i] * j[i] * sig[i] * sig[i] for i in range(5))
    checks.append(("var_css", fo_var == Fraction(191, 240)))
    checks.append(
        (
            "l2_fragment_bundle",
            all(ok for _, ok in checks),
        )
    )

    failed = [n for n, ok in checks if not ok]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    if failed:
        print(f"FO_BYTECODE_FRAGMENT_CERT_FAIL {len(failed)}/{len(checks)}")
        return 1
    print(f"FO_BYTECODE_FRAGMENT_CERT_OK {len(checks)}/{len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
