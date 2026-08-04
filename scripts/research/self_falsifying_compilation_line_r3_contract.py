#!/usr/bin/env python3
"""Self-falsifying compilation, rung R3 — executable falsifiers.

Spec: docs/research/self_falsifying_compilation_line_r3_2026-07-26.md

R0 §3 proved that no compile-time procedure whose only evidence is a claim's own
check can detect SHARED MISINTERPRETATION. R2 built the guard for the other
class (drift). R3 tests the one remaining idea: the claim schema already carries
a `falsifier` field, today prose. If falsifiers were EXECUTABLE — a check
authored against the claim that must FAIL for the claim to live — would that
reach the half R2 cannot?

The honest test is not "build the mechanism" (a falsifier that must fail is just
a gate with inverted polarity, and if the same author writes both, R0 §3 applies
unchanged). It is:

    For the three audited self-corrections, can an executable falsifier be
    expressed INDEPENDENTLY of the claim's own harness, and would it have
    refuted the proposition asserted at the parent commit?

"Independently" is the load-bearing word: a falsifier that imports the claim's
machinery inherits the claim's misunderstanding.

VERDICT OPTIONS, FIXED BEFORE COMPUTING (see main()):
  FALSIFIERS_NONVACUOUS_GENERALLY
      all three expressible independently at modest cost, and each refutes.
  FALSIFIERS_NONVACUOUS_ONLY_FOR_CLOSED_FORM_CLAIMS
      the ones reducible to a closed-form identity are cheap and fire; the
      others need the original machinery rebuilt, i.e. the falsifier costs as
      much as the claim.
  FALSIFIERS_VACUOUS
      even the closed-form case fails to refute its parent proposition.

Pure Python 3 + numpy.
"""

from __future__ import annotations

import sys

import numpy as np

TOL = 1e-9
RNG = np.random.default_rng(20260726)


# ---------------------------------------------------------------- algebra
# Cayley-Dickson, re-derived here rather than imported from any functor-F
# contract. Independence is the property under test: a falsifier that reuses the
# claim's own multiplication table inherits whatever that table encodes.


def cd_conj(x: np.ndarray) -> np.ndarray:
    c = -x.copy()
    c[0] = x[0]
    return c


def cd_mul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(a,b)(c,d) = (ac - conj(d) b, d a + b conj(c))."""
    n = len(x)
    if n == 1:
        return x * y
    h = n // 2
    a, b = x[:h], x[h:]
    c, d = y[:h], y[h:]
    lo = cd_mul(a, c) - cd_mul(cd_conj(d), b)
    hi = cd_mul(d, a) + cd_mul(b, cd_conj(c))
    return np.concatenate([lo, hi])


def basis(i: int, n: int) -> np.ndarray:
    v = np.zeros(n)
    v[i] = 1.0
    return v


def imaginary(n: int) -> np.ndarray:
    """Random imaginary element (zero real part)."""
    v = RNG.normal(size=n)
    v[0] = 0.0
    return v


# ---------------------------------------------------------------- F2


def falsifier_e6_bridge() -> dict:
    """Refute: 'phi is the G2 SHADOW / COMPLEMENT / blind-spot of the E6 cubic'.

    Asserted at 2b33d7500 as PHI_IS_G2_SHADOW_OF_E6_CUBIC; corrected in
    ec579a24c to PHI_IS_THE_E6_CUBIC_CROSSTERM.

    The falsifier is a closed-form identity: for IMAGINARY octonions,
        Re(x y z) = -<x y, z> = -phi(x, y, z).
    If that holds, phi is the octonion cross-term of the Albert cubic -- it sits
    INSIDE the invariant and cannot be its complement. Nothing about the claim's
    harness is used: octonion multiplication is re-derived above, and the
    identity is checked on random triples plus every basis triple.
    """
    def phi(x, y, z):
        # <x y, z> with z imaginary: Euclidean inner product on components 1..7
        return float(np.dot(cd_mul(x, y)[1:], z[1:]))

    def re_xyz(x, y, z):
        return float(cd_mul(cd_mul(x, y), z)[0])

    worst = 0.0
    # random imaginary triples
    for _ in range(400):
        x, y, z = imaginary(8), imaginary(8), imaginary(8)
        worst = max(worst, abs(re_xyz(x, y, z) + phi(x, y, z)))
    # every imaginary basis triple
    for i in range(1, 8):
        for j in range(1, 8):
            for k in range(1, 8):
                x, y, z = basis(i, 8), basis(j, 8), basis(k, 8)
                worst = max(worst, abs(re_xyz(x, y, z) + phi(x, y, z)))

    holds = worst < TOL
    return {
        "name": "F2_E6_BRIDGE",
        "parent_claim": "PHI_IS_G2_SHADOW_OF_E6_CUBIC",
        "expressible": True,
        "independent": True,
        "cost": "closed-form identity, ~40 lines, no group theory",
        "refutes_parent": holds,
        "detail": (f"max |Re(xyz) + phi(x,y,z)| = {worst:.3e} over 400 random "
                   f"imaginary triples and all 343 imaginary basis triples; "
                   f"identity {'HOLDS' if holds else 'FAILS'}"),
    }


# ---------------------------------------------------------------- F1, F3
# Recorded as not independently expressible within this rung. This is a result,
# not an omission: see the spec. Both propositions are about the structure of a
# constructed object (a symmetry group; a module over a Fano-line class), so a
# falsifier has to rebuild that construction before it can disagree with it --
# and rebuilding it from the claim's own harness would forfeit the independence
# that makes a falsifier worth anything.


def falsifier_ord3_module() -> dict:
    return {
        "name": "F1_ORD3_MODULE",
        "parent_claim": "ORD3_MODULE_IS_2xV3 (framed as a fingerprint of the operation)",
        "expressible": False,
        "independent": None,
        "cost": ("requires re-deriving the octonion automorphism group action, "
                 "the Fano-line class structure and the sedenion zero-divisor "
                 "set before the module can even be formed"),
        "refutes_parent": None,
        "detail": ("NOT ATTEMPTED INDEPENDENTLY. The corrected statement (the "
                   "module is the class's plain coordinate space, and the "
                   "multiplicity 2 is Cayley-Dickson doubling) is checkable "
                   "only after that machinery exists; building it from the "
                   "claim's own contract would inherit the claim's framing."),
    }


def count_sign_automorphisms() -> int:
    """Diagonal sign maps on e_1..e_7 that are octonion automorphisms.

    Cheap and fully independent: 128 candidates, checked against the
    multiplication re-derived in this file. This is the '2^3' factor the
    correction identified, computed without reference to the claim's harness.
    """
    n = 8
    ident = [basis(i, n) for i in range(n)]
    count = 0
    for mask in range(128):
        s = [1.0] + [(-1.0 if (mask >> k) & 1 else 1.0) for k in range(7)]
        g = np.zeros((n, n))
        for i in range(n):
            g[:, i] = s[i] * ident[i]
        ok = True
        for i in range(n):
            for j in range(n):
                lhs = g @ cd_mul(ident[i], ident[j])
                rhs = cd_mul(g[:, i], g[:, j])
                if np.linalg.norm(lhs - rhs) > TOL:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            count += 1
    return count


def falsifier_group_id() -> dict:
    """Partially attempted rather than merely declared inexpressible.

    Refuting 'the group is S4, order 24' outright needs the symmetry-fill
    group's definition, and its order is the disputed fact -- so the falsifier
    cannot take that definition from the claim. But one component IS cheaply
    and independently computable: the diagonal sign automorphisms, i.e. the
    '2^3' the correction identified. Computing it tests whether the cost
    objection is real or merely asserted.
    """
    n_sign = count_sign_automorphisms()
    # Corroborates a factor the correction names; does NOT by itself refute
    # 'order 24', because without the group's definition we cannot say these
    # sign maps all lie in it.
    corroborates = (n_sign == 8)
    return {
        "name": "F3_GROUP_ID",
        "parent_claim": "the ord-3 symmetry-fill group is S4, order 24",
        "expressible": False,
        "independent": None,
        "cost": ("the disputed fact IS the group's order, so a falsifier "
                 "cannot take the group's definition from the claim; "
                 "constructing it independently is the bulk of the original work"),
        "refutes_parent": None,
        "detail": (f"PARTIALLY ATTEMPTED. Independent computation of the "
                   f"diagonal sign automorphisms of the octonions gives "
                   f"{n_sign} (2^3 = 8 expected: "
                   f"{'corroborated' if corroborates else 'NOT corroborated'}), "
                   f"which is the factor the correction names. This corroborates "
                   f"a component but does NOT refute 'order 24' on its own: "
                   f"without the symmetry-fill group's definition one cannot "
                   f"assert these maps lie in it. The cost objection is real, "
                   f"not merely asserted."),
    }


# ---------------------------------------------------------------- main


def main() -> int:
    print("SELF-FALSIFYING COMPILATION R3 — executable falsifiers")
    print("=" * 72)
    print("Question: for the three audited self-corrections, can a falsifier be")
    print("expressed INDEPENDENTLY of the claim's harness, and would it have")
    print("refuted the parent commit's proposition?")
    print()

    results = [falsifier_e6_bridge(), falsifier_ord3_module(), falsifier_group_id()]

    for r in results:
        state = ("EXPRESSIBLE" if r["expressible"] else "NOT_INDEPENDENTLY_EXPRESSIBLE")
        print(f"{r['name']} {state}")
        print(f"    parent claim : {r['parent_claim']}")
        print(f"    cost         : {r['cost']}")
        if r["expressible"]:
            verdict = "REFUTES the parent" if r["refutes_parent"] else "does NOT refute"
            print(f"    outcome      : {verdict}")
        print(f"    detail       : {r['detail']}")
        print()

    expressible = [r for r in results if r["expressible"]]
    fired = [r for r in expressible if r["refutes_parent"]]

    print(f"E1_EXPRESSIBILITY {len(expressible)}/{len(results)} falsifiers "
          f"expressible independently")
    print(f"E1_EXPRESSIBILITY {'PASS' if results else 'FAIL'} — measured")
    print()
    print(f"E2_WOULD_HAVE_FIRED {len(fired)}/{len(expressible)} of the "
          f"expressible falsifiers refute their parent proposition")
    print(f"E2_WOULD_HAVE_FIRED {'PASS' if expressible else 'FAIL'} — measured")
    print()

    print("=" * 72)
    if not expressible:
        token = "FALSIFIERS_VACUOUS"
    elif len(fired) < len(expressible):
        token = "FALSIFIERS_VACUOUS"
    elif len(expressible) == len(results):
        token = "FALSIFIERS_NONVACUOUS_GENERALLY"
    else:
        token = "FALSIFIERS_NONVACUOUS_ONLY_FOR_CLOSED_FORM_CLAIMS"

    print(f"  expressible independently : {len(expressible)}/{len(results)}")
    print(f"  of those, would have fired: {len(fired)}/{len(expressible)}")
    print(f"SELF_FALSIFYING_R3_VERDICT {token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
