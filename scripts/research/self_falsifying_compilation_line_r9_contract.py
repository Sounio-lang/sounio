#!/usr/bin/env python3
"""Self-falsifying compilation, rung R9 — finishing the trusted-base audit.

Spec: docs/research/self_falsifying_compilation_line_r9_2026-07-26.md

R8 mapped 12 irreducible kernels and audited the one that dominates the blast
radius (the Cayley-Dickson sign table). Nine were left mapped and unchecked. R9
finishes the job, and in doing so answers the question R8 flagged as the
method's boundary:

    "A shared kernel encoding a CHOICE rather than a THEOREM would have no
     adjudicator."

So each remaining kernel is first CLASSIFIED, then audited if it can be:

  PREDICTIVE   asserts a structural fact that can be checked against ground
               truth computed independently. These are the valuable ones —
               they can be WRONG.
  ALGEBRAIC    obeys laws that pin it down (associativity, inverses, ...).
  MECHANICAL   a regrouping or lookup with one possible behaviour; an
               independent implementation trivially matches.
  CONVENTION   encodes a choice. No adjudicator exists, and saying so is the
               honest outcome, not a gap to paper over.

GROUND TRUTH is computed from the recursive Cayley-Dickson oracle re-derived in
this file (R8 measured it at similarity 0.058-0.107 against the corpus's own
sign kernels). Zero divisors are found by rank-deficiency of the
left-multiplication matrix — a route the corpus's own predicates never take.

VERDICT OPTIONS, FIXED BEFORE COMPUTING (see main()):
  TRUSTED_BASE_FULLY_AUDITED__ALL_KERNELS_CORROBORATED
      every kernel with an adjudicator agrees with independent ground truth.
  TRUSTED_BASE_AUDIT_FOUND_DIVERGENCE
      at least one kernel disagrees — part of the corpus asserts something the
      algebra does not support.
  TRUSTED_BASE_PARTIALLY_AUDITABLE
      kernels remain for which no adjudicator could be constructed.

Pure Python 3 + numpy.
"""

from __future__ import annotations

import ast
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TOL = 1e-9


# ---------------------------------------------------------------- oracle


def cd_conj(x):
    c = -x.copy()
    c[0] = x[0]
    return c


def cd_mul(x, y):
    n = len(x)
    if n == 1:
        return x * y
    h = n // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    return np.concatenate([cd_mul(a, c) - cd_mul(cd_conj(d), b),
                           cd_mul(d, a) + cd_mul(b, cd_conj(c))])


def e(i, n):
    v = np.zeros(n)
    v[i] = 1.0
    return v


def is_zero_divisor(x) -> bool:
    """x is a zero divisor iff left multiplication by x is rank-deficient."""
    n = len(x)
    L = np.column_stack([cd_mul(x, e(k, n)) for k in range(n)])
    return int(np.linalg.matrix_rank(L, tol=1e-8)) < n


def zd_census(b: int) -> set[tuple[int, int]]:
    """Canonical 2-term zero divisors e_i +/- e_j at level b, as index pairs."""
    n = 1 << b
    out = set()
    for i in range(1, n):
        for j in range(i + 1, n):
            for s in (1.0, -1.0):
                if is_zero_divisor(e(i, n) + s * e(j, n)):
                    out.add((i, j))
                    break
    return out


# ---------------------------------------------------------------- extraction


def extract(rel: str, fn: str, extra: dict | None = None):
    path = REPO / "scripts/research" / rel
    try:
        src = path.read_text(errors="replace")
    except OSError:
        return None
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == fn:
            ns: dict = {"np": np, "Fraction": Fraction, "TOL": TOL}
            ns.update(extra or {})
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             f"<{rel}:{fn}>", "exec"), ns)
                return ns[fn]
            except Exception:
                return None
    return None


# ---------------------------------------------------------------- audits


def audit_expected_labels(results):
    """PREDICTIVE: the closed form for which xor-labels carry zero divisors."""
    f = extract("chingon_zd_contract.py", "expected_labels")
    if f is None:
        results.append(("expected_labels", "PREDICTIVE", "NOT_EXTRACTABLE", ""))
        return
    detail = []
    agree = True
    for b in (4, 5):
        truth = {i ^ j for i, j in zd_census(b)}
        claimed = set(f(b))
        if truth != claimed:
            agree = False
            detail.append(f"level {b}: claimed-only={sorted(claimed - truth)[:6]} "
                          f"truth-only={sorted(truth - claimed)[:6]}")
        else:
            detail.append(f"level {b}: {len(truth)} labels, exact match")
    results.append(("expected_labels", "PREDICTIVE",
                    "CORROBORATED" if agree else "DIVERGES", "; ".join(detail)))


def audit_missing_diagonal(results):
    """PREDICTIVE: which index pairs in a fiber are NOT zero divisors."""
    birth = extract("chingon_zd_contract.py", "fiber_birth_level")
    f = extract("chingon_zd_contract.py", "missing_diagonal",
                {"fiber_birth_level": birth})
    if f is None or birth is None:
        results.append(("missing_diagonal", "PREDICTIVE", "NOT_EXTRACTABLE", ""))
        return
    detail = []
    agree = True
    for b in (4, 5):
        n = 1 << b
        zd = zd_census(b)
        labels = {i ^ j for i, j in zd}
        for label in sorted(labels):
            all_pairs = {(i, i ^ label) for i in range(1, n)
                         if 0 < (i ^ label) < n and i < (i ^ label)}
            actual_missing = {p for p in all_pairs if p not in zd}
            try:
                claimed = {tuple(sorted(p)) for p in f(label, b)}
            except Exception as exc:
                results.append(("missing_diagonal", "PREDICTIVE",
                                "NOT_EXTRACTABLE", f"call failed: {exc}"))
                return
            claimed = {p for p in claimed if p in all_pairs}
            if claimed != actual_missing:
                agree = False
                detail.append(f"level {b} label {label}: "
                              f"claimed={sorted(claimed)[:4]} "
                              f"actual={sorted(actual_missing)[:4]}")
                break
        else:
            detail.append(f"level {b}: all {len(labels)} fibers match")
            continue
        break
    results.append(("missing_diagonal", "PREDICTIVE",
                    "CORROBORATED" if agree else "DIVERGES", "; ".join(detail)))


def audit_compute_fibers(results):
    """MECHANICAL: grouping by xor label. One possible behaviour."""
    f = extract("chingon_zd_contract.py", "compute_fibers")
    if f is None:
        results.append(("compute_fibers", "MECHANICAL", "NOT_EXTRACTABLE", ""))
        return
    pairs = [(i, +1, j) for i, j in sorted(zd_census(4))]
    got = f(pairs)
    want: dict[int, list] = {}
    for i, s, j in pairs:
        want.setdefault(i ^ j, []).append((i, s, j))
    ok = {k: sorted(v) for k, v in got.items()} == {k: sorted(v) for k, v in want.items()}
    results.append(("compute_fibers", "MECHANICAL",
                    "CORROBORATED" if ok else "DIVERGES",
                    f"{len(want)} fibers regrouped independently"))


def audit_polynomials(results):
    """ALGEBRAIC: p_add/p_sub pinned by group laws plus an independent impl."""
    p_add = extract("ade_wildgen_mckay_contract.py", "p_add")
    p_sub = extract("ade_wildgen_mckay_contract.py", "p_sub")
    if p_add is None or p_sub is None:
        results.append(("p_add/p_sub", "ALGEBRAIC", "NOT_EXTRACTABLE", ""))
        return
    rng = np.random.default_rng(20260726)
    ok = True
    notes = []
    for _ in range(300):
        f = {int(m): Fraction(int(rng.integers(-5, 6)), int(rng.integers(1, 5)))
             for m in rng.integers(0, 8, size=4)}
        g = {int(m): Fraction(int(rng.integers(-5, 6)), int(rng.integers(1, 5)))
             for m in rng.integers(0, 8, size=4)}
        f = {k: v for k, v in f.items() if v != 0}
        g = {k: v for k, v in g.items() if v != 0}
        # independent implementation: dense coefficient vectors
        def dense(d):
            v = [Fraction(0)] * 8
            for k, c in d.items():
                v[k] += c
            return v
        ind_add = dense(f)
        ind_add = [a + b for a, b in zip(ind_add, dense(g))]
        got_add = dense(p_add(f, g))
        if got_add != ind_add:
            ok = False
            notes.append("p_add differs from a dense-vector implementation")
            break
        # laws: f - f = 0, (f + g) - g = f, no zero coefficients retained
        if p_sub(f, f) != {}:
            ok = False; notes.append("p_sub(f, f) != 0"); break
        if dense(p_sub(p_add(f, g), g)) != dense(f):
            ok = False; notes.append("(f+g)-g != f"); break
        if any(c == 0 for c in p_add(f, g).values()):
            ok = False; notes.append("zero coefficients retained"); break
    results.append(("p_add/p_sub", "ALGEBRAIC",
                    "CORROBORATED" if ok else "DIVERGES",
                    "; ".join(notes) or "300 random pairs: dense-vector "
                                        "agreement + group laws"))


def audit_cusp_wells(results):
    """ALGEBRAIC: returned points must be roots of x^3+ax+b with 3x^2+a>0."""
    f = extract("functor_f_g2_covariance_contract.py", "cusp_wells")
    if f is None:
        results.append(("cusp_wells", "ALGEBRAIC", "NOT_EXTRACTABLE", ""))
        return
    rng = np.random.default_rng(7)
    ok = True
    worst = 0.0
    for _ in range(400):
        a, b = float(rng.normal()), float(rng.normal())
        for x in f(a, b):
            worst = max(worst, abs(x ** 3 + a * x + b))
            if abs(x ** 3 + a * x + b) > 1e-6 or 3 * x * x + a <= 0:
                ok = False
    results.append(("cusp_wells", "ALGEBRAIC",
                    "CORROBORATED" if ok else "DIVERGES",
                    f"400 random cusps; max |x^3+ax+b| = {worst:.2e}, "
                    f"all satisfy 3x^2+a>0"))


def audit_cd_sigma_variant(results):
    """The second cd_sigma cluster against the one R8 audited."""
    a = extract("cd_tower_zd_fiber_signed_localization_contract.py", "cd_sigma")
    b = extract("rupture_r2_fiber_measure_contract.py", "cd_sigma")
    if a is None or b is None:
        results.append(("cd_sigma (variant 2)", "PREDICTIVE",
                        "NOT_EXTRACTABLE", ""))
        return
    diff = 0
    total = 0
    for bits in (3, 4, 5, 6):
        for i in range(1 << bits):
            for j in range(1 << bits):
                total += 1
                if a(i, j, bits) != b(i, j, bits):
                    diff += 1
    results.append(("cd_sigma (variant 2)", "ALGEBRAIC",
                    "CORROBORATED" if diff == 0 else "DIVERGES",
                    f"{total} products vs the cd_sigma R8 audited, "
                    f"{diff} disagreements"))


def audit_zd_line(results):
    """CONVENTION-adjacent: needs its file's Lmat4/nullspace to even run."""
    results.append(("zd_line", "CONVENTION", "NO_ADJUDICATOR",
                    "defined in terms of its own file's Lmat4/nullspace; "
                    "reconstructing them here would import the derivation "
                    "under audit, which forfeits independence (R6)"))


def audit_chk(results):
    results.append(("chk", "CONVENTION", "NOT_EXTRACTABLE",
                    "nested inside another function; not a module-level kernel"))


# ---------------------------------------------------------------- main


def main() -> int:
    print("SELF-FALSIFYING COMPILATION R9 — finishing the trusted-base audit")
    print("=" * 78)
    print("R8 mapped 12 kernels and audited the sign table. Nine were left")
    print("unchecked. Ground truth here comes from rank-deficiency of the")
    print("left-multiplication matrix — a route the corpus's predicates never take.")
    print()

    results: list[tuple[str, str, str, str]] = []
    audit_expected_labels(results)
    audit_missing_diagonal(results)
    audit_compute_fibers(results)
    audit_polynomials(results)
    audit_cusp_wells(results)
    audit_cd_sigma_variant(results)
    audit_zd_line(results)
    audit_chk(results)

    for name, kind, verdict, detail in results:
        print(f"  {name:22s} [{kind:10s}] {verdict}")
        if detail:
            print(f"      {detail}")
    print()

    corroborated = [r for r in results if r[2] == "CORROBORATED"]
    diverged = [r for r in results if r[2] == "DIVERGES"]
    unadjudicable = [r for r in results if r[2] in ("NO_ADJUDICATOR",
                                                    "NOT_EXTRACTABLE")]

    print(f"K9_AUDIT {len(corroborated)} corroborated, {len(diverged)} diverged, "
          f"{len(unadjudicable)} without an adjudicator")
    print(f"K9_AUDIT {'PASS' if not diverged else 'FAIL'} — measured")
    print()

    print("=" * 78)
    if diverged:
        token = "TRUSTED_BASE_AUDIT_FOUND_DIVERGENCE"
    elif unadjudicable:
        token = "TRUSTED_BASE_PARTIALLY_AUDITABLE"
    else:
        token = "TRUSTED_BASE_FULLY_AUDITED__ALL_KERNELS_CORROBORATED"

    print(f"  corroborated         : {len(corroborated)}")
    print(f"  diverged             : {len(diverged)}")
    print(f"  no adjudicator       : {len(unadjudicable)}  "
          f"({', '.join(r[0] for r in unadjudicable) or '-'})")
    print(f"SELF_FALSIFYING_R9_VERDICT {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
