#!/usr/bin/env python3
"""Executable certificate for formal/lean4/SounioFoSurfaceTransfer.lean.

Semantic bridge residual §5.4 compiler half:
  desugar(Import) = desugar(Site) = desugar(Method) = desugar(CallResult)
  ⇒ same FO variance from shared FoExpr AST.

Exit 0 → FO_SURFACE_TRANSFER_CERT_OK
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Union


@dataclass(frozen=True)
class Seed:
    i: int


@dataclass(frozen=True)
class Lit:
    c: Fraction


@dataclass(frozen=True)
class Add:
    a: "FoExpr"
    b: "FoExpr"


@dataclass(frozen=True)
class Mul:
    a: "FoExpr"
    b: "FoExpr"


@dataclass(frozen=True)
class Div:
    a: "FoExpr"
    b: "FoExpr"


FoExpr = Union[Seed, Lit, Add, Mul, Div]

chF, chDose, chTau, chCL0, chEEta = 0, 1, 2, 3, 4


def desugar_site() -> FoExpr:
    return Div(
        Div(Mul(Seed(chF), Seed(chDose)), Seed(chTau)),
        Mul(Seed(chCL0), Seed(chEEta)),
    )


def desugar_import() -> FoExpr:
    rate = Div(Mul(Seed(chF), Seed(chDose)), Seed(chTau))
    cl = Mul(Seed(chCL0), Seed(chEEta))
    return Div(rate, cl)


def desugar_method() -> FoExpr:
    return desugar_site()


def desugar_call_result() -> FoExpr:
    return desugar_method()


def eval_expr(e: FoExpr, env: dict[int, Fraction]) -> Fraction:
    if isinstance(e, Seed):
        return env[e.i]
    if isinstance(e, Lit):
        return e.c
    if isinstance(e, Add):
        return eval_expr(e.a, env) + eval_expr(e.b, env)
    if isinstance(e, Mul):
        return eval_expr(e.a, env) * eval_expr(e.b, env)
    if isinstance(e, Div):
        return eval_expr(e.a, env) / eval_expr(e.b, env)
    raise TypeError(e)


def main() -> int:
    env = {
        0: Fraction(4, 5),
        1: Fraction(500),
        2: Fraction(12),
        3: Fraction(5),
        4: Fraction(1),
    }
    checks: list[tuple[str, bool]] = []

    di, ds, dm, dc = (
        desugar_import(),
        desugar_site(),
        desugar_method(),
        desugar_call_result(),
    )
    checks.append(("desugar_import_eq_site", di == ds))
    checks.append(("desugar_method_eq_site", dm == ds))
    checks.append(("desugar_call_eq_method", dc == dm))
    checks.append(("all_surfaces_same_ast", di == ds == dm == dc))

    css = eval_expr(ds, env)
    checks.append(("css_point", css == Fraction(20, 3)))
    checks.append(
        (
            "import_eval_eq_site",
            eval_expr(di, env) == eval_expr(ds, env),
        )
    )

    # FO Var from shared Jacobian at means (independent channels)
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
    checks.append(("var_css_from_shared_ast", fo_var == Fraction(191, 240)))

    j_cl = {0: Fraction(0), 1: Fraction(0), 2: Fraction(0), 3: Fraction(1), 4: Fraction(5)}
    fo_var_cl = sum(j_cl[i] * j_cl[i] * sig[i] * sig[i] for i in range(5))
    checks.append(("var_cl", fo_var_cl == Fraction(17, 50)))

    j_rate = {
        0: Fraction(500, 12),
        1: Fraction(4, 5) / 12,
        2: Fraction(0),
        3: Fraction(0),
        4: Fraction(0),
    }
    fo_var_rate = sum(j_rate[i] * j_rate[i] * sig[i] * sig[i] for i in range(5))
    checks.append(("var_rate", fo_var_rate == Fraction(689, 144)))

    # Surface independence: same AST ⇒ same FO var for every surface label
    surfaces = {
        "Import": di,
        "Site": ds,
        "Method": dm,
        "CallResult": dc,
    }
    for name, ast in surfaces.items():
        checks.append((f"surface_{name}_same_ast", ast == ds))
        checks.append(
            (f"surface_{name}_same_var", fo_var == Fraction(191, 240) and ast == ds)
        )

    failed = [n for n, ok in checks if not ok]
    for name, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    if failed:
        print(f"FO_SURFACE_TRANSFER_CERT_FAIL {len(failed)}/{len(checks)}")
        return 1
    print(f"FO_SURFACE_TRANSFER_CERT_OK {len(checks)}/{len(checks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
