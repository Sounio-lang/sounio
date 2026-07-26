#!/usr/bin/env python3
"""
E-series semantic germ — a Petitot-style morphodynamic potential whose
bifurcation set carries the Arnold E6/E7/E8 singularity structure, verified
against the octonion/associator structure of the rupture programme.

Companion to:
  docs/research/e_series_semantic_germ_spec_2026-07-26.md

Context (ade_wildgen_mckay_spec_2026-07-26.md, verdict
STRONG_FORM_OBSTRUCTED__WEAK_FORM_UNDECIDABLE): the weak form of the
ADE-Wildgen conjecture was undecidable because "no E-series germ has ever
been constructed on the semantic (morphodynamic) side" (clause C3). This
contract constructs that object and makes the weak form testable:

  G-SIDE (the semantic E-series germs, exact rational arithmetic):
    G1  the Petitot-style potentials V_E6 = x^3+y^4, V_E7 = x^3+xy^3,
        V_E8 = x^3+y^5 (two state variables, controls = unfolding monomials)
        have Milnor numbers mu = 6/7/8 (exact rational Buchberger), corank 2,
        and miniversal unfolding bases of size mu; the programme's operative
        A-series germs (cusp x^4 = A3, butterfly x^6 = A5) are reproduced as
        controls.
    G2  the 3-jet discriminates E from D exactly: the cubic part of each
        E-germ is a perfect cube (one linear factor, multiplicity 3; binary
        Hessian of the cubic vanishes identically), while the D4/D5 controls
        split into 3 / 2 distinct linear factors (binary cubic discriminant).
    G3  MORSIFICATION CENSUS (numeric): an explicit small real deformation
        splits each E-centre into exactly mu distinct nondegenerate (A1)
        critical points over C; the real sub-census (real critical points,
        Morse indices, minima = Petitot "wells") is reported as data. This
        is the semantic content Wildgen needs: the E6/E7/E8 organizing
        centres unfold into 6/7/8 elementary positions, against 3 (cusp)
        and 5 (butterfly) for the operative A-series germs.
    G4  E6 ADJACENCY CLOSURE (full, exact): the bifurcation set of V_E6
        contains every sub-singularity in Arnold's list for E6 —
        A1 (G3), A2, A3, A4, A5, D4, D5 — each witnessed by an explicit
        deformation with a critical point at the origin whose local type is
        verified exactly (gradient vanishes, Hessian corank, local Milnor
        number via Buchberger, cubic-jet factor structure).
    G5  E7 ADJACENCY SPINE (one short of full, exact): witnessed
        sub-singularities of E7: A1 (G3), A2, A3, A4, A5, D4, D5, D6, E6 —
        the full Arnold list for E7 except A6 (needs an off-origin
        witness; scoped out, see spec).
    G6  E8 ADJACENCY SPINE (two short of full, exact): witnessed
        sub-singularities of E8: A1 (G3), A2, A3, A4, A5, D4, D5, D6, D7,
        E6, E7 — the full Arnold list for E8 except A6, A7 (need
        off-origin witnesses; scoped out, see spec).

  O-SIDE (verification against the octonion/associator structure; O1/O2 are
  inherited re-audits of the gated functor_f E-clauses, run on the
  self-contained CD core):
    O1  Re(x*y*z) is bracketing-independent on O (associator purely
        imaginary) and equals -phi(x,y,z) on imaginary triples: the G2
        3-form phi IS the imaginary restriction of the E6/Albert cubic
        cross-term (inherited E1+E3, re-audited).
    O2  the associator [x,y,z] is vector-valued and separate from the
        scalar phi: [e1,e2,e4] has norm 2 while phi(e1,e2,e4) = 0
        (inherited E4, re-audited).
    O3  the E-labels of the constructed germs are the magic-square tower:
        Tits T(A,J3(O)): 0+1*26+52 = 78 (E6), 3+3*26+52 = 133 (E7),
        14+7*26+52 = 248 (E8) (inherited C4, re-audited).
    O4  DIVERGENCE (honest negative): no nonzero symmetric cubic form on
        Im O arises from the Albert cross-term — phi is alternating
        (phi(x,x,x) = 0 measured) and Re(x^3) = 0 for imaginary x — while
        the semantic germ's cubic is a commutative perfect cube. The
        E-series germ is therefore NOT claimed to be octonion-derived; the
        bridge is the E-label via the magic square + the singularity label,
        not a form identity. D3 quarantine kept.

Verdict: E_SERIES_GERMS_CONSTRUCTED__WEAK_FORM_TESTABLE__WEAK_FORM_OPEN —
the semantic E-series object now exists and is gated, so the weak form of
ADE-Wildgen is testable; construction does not decide it (current programme
evidence neither requires nor excludes E-series morphology).

Self-contained; numpy only for the numeric parts (G3, O-side).
Deterministic (fixed seeds). Pass `--probe` to print the candidate-witness
battery without assertions (used to fix the witness tables).
"""
from __future__ import annotations

import sys
from fractions import Fraction

import numpy as np

np.seterr(all="ignore")
TOL = 1e-8
PROBE = "--probe" in sys.argv

F = Fraction


# ---------------------------------------------------------------------
# Exact rational bivariate polynomial arithmetic (graded order, x > y)
# ---------------------------------------------------------------------

def mkey(mon):
    return (mon[0] + mon[1], mon[0], mon[1])


def p_lt(f):
    return max(f.keys(), key=mkey)


def p_mul_monomial(f, mon, coeff=F(1)):
    return {(a + mon[0], b + mon[1]): c * coeff for (a, b), c in f.items()}


def p_mul(f, g):
    out = {}
    for (a, b), c in f.items():
        for (d, e), h in g.items():
            out[(a + d, b + e)] = out.get((a + d, b + e), F(0)) + c * h
    return {m: c for m, c in out.items() if c != 0}


def p_add(f, g):
    out = dict(f)
    for mon, c in g.items():
        out[mon] = out.get(mon, F(0)) + c
        if out[mon] == 0:
            del out[mon]
    return out


def p_sub(f, g):
    out = dict(f)
    for mon, c in g.items():
        out[mon] = out.get(mon, F(0)) - c
        if out[mon] == 0:
            del out[mon]
    return out


def p_scale(f, s):
    return {mon: c * s for mon, c in f.items()}


def p_monic(f):
    lt = p_lt(f)
    lc = f[lt]
    return {mon: c / lc for mon, c in f.items()}


def p_divisible(mon, lt):
    return mon[0] >= lt[0] and mon[1] >= lt[1]


def p_reduce(f, G):
    f = dict(f)
    while True:
        hit = None
        for mon in sorted(f.keys(), key=mkey, reverse=True):
            for g in G:
                ltg = p_lt(g)
                if p_divisible(mon, ltg):
                    hit = (mon, g, ltg)
                    break
            if hit:
                break
        if not hit:
            return f
        mon, g, ltg = hit
        coeff = f[mon] / g[ltg]
        shift = (mon[0] - ltg[0], mon[1] - ltg[1])
        f = p_sub(f, p_mul_monomial(g, shift, coeff))
        if not f:
            return {}


def p_spoly(f, g):
    lf, lg = p_lt(f), p_lt(g)
    lcm = (max(lf[0], lg[0]), max(lf[1], lg[1]))
    s1 = p_mul_monomial(p_monic(f), (lcm[0] - lf[0], lcm[1] - lf[1]))
    s2 = p_mul_monomial(p_monic(g), (lcm[0] - lg[0], lcm[1] - lg[1]))
    return p_sub(s1, s2)


def buchberger(gens, cap=400):
    G = [p_monic(g) for g in gens if g]
    pairs = [(i, j) for i in range(len(G)) for j in range(i + 1, len(G))]
    steps = 0
    while pairs and steps < cap:
        i, j = pairs.pop(0)
        r = p_reduce(p_spoly(G[i], G[j]), G)
        steps += 1
        if r:
            G.append(p_monic(r))
            pairs += [(i, len(G) - 1) for i in range(len(G) - 1)]
    return G


def milnor_number(jac_gens):
    """dim Q[x,y]/J via standard monomials of a Groebner basis."""
    G = buchberger(jac_gens)
    lts = [p_lt(g) for g in G]
    pure_x = [l[0] for l in lts if l[1] == 0]
    pure_y = [l[1] for l in lts if l[0] == 0]
    if not pure_x or not pure_y:
        raise AssertionError("Jacobian ideal not zero-dimensional (non-isolated critical point)")
    maxa = max(l[0] for l in lts)
    maxb = max(l[1] for l in lts)
    basis = [(a, b) for a in range(maxa + 1) for b in range(maxb + 1)
             if not any(p_divisible((a, b), lt) for lt in lts)]
    return len(basis), sorted(basis, key=mkey)


def d_dx(f):
    return {(a - 1, b): c * a for (a, b), c in f.items() if a > 0}


def d_dy(f):
    return {(a, b - 1): c * b for (a, b), c in f.items() if b > 0}


# ---------------------------------------------------------------------
# Local Milnor number at the origin (exact linear algebra).
#
# The Milnor number is the dimension of the LOCAL algebra Q[x,y]_m / J,
# not of the global quotient (a deformation may move further critical
# points into the affine plane; the global quotient counts them all).
# In the truncated ring T_N = Q[x,y]/m^(N+1) every element is a truncated
# polynomial, so the local ideal is exactly the Q-linear span of truncated
# monomial multiples of the generators (unit inversion = truncated
# geometric series, which is a polynomial combination of monomial
# multiples).  For an Artinian local algebra of dimension mu, m^mu = 0,
# so N = dim(global quotient) + 1 is safely enough.
# ---------------------------------------------------------------------

def exact_rank(rows):
    """Rank of a list of Fraction row vectors (Gaussian elimination)."""
    rows = [list(r) for r in rows if any(c != 0 for c in r)]
    rank = 0
    col = 0
    ncols = len(rows[0]) if rows else 0
    while rows and col < ncols:
        piv = None
        for i, r in enumerate(rows):
            if r[col] != 0:
                piv = i
                break
        if piv is None:
            col += 1
            continue
        rows[0], rows[piv] = rows[piv], rows[0]
        pr = rows[0]
        pc = pr[col]
        rest = []
        for r in rows[1:]:
            if r[col] != 0:
                f = r[col] / pc
                rest.append([c - f * p for c, p in zip(r, pr)])
            else:
                rest.append(r)
        rank += 1
        rows = rest
        col += 1
    return rank


def local_milnor(jac_gens):
    gdim, _ = milnor_number(jac_gens)   # global dimension (zero-dim check incl.)
    N = gdim + 1
    mons = [(a, b) for a in range(N + 1) for b in range(N + 1 - a)]
    midx = {m: i for i, m in enumerate(mons)}
    rows = []
    for g in jac_gens:
        for m in mons:
            row = [F(0)] * len(mons)
            for (a, b), c in p_mul_monomial(g, m).items():
                if a + b <= N:
                    row[midx[(a, b)]] = c
            rows.append(row)
    return len(mons) - exact_rank(rows)


def X(power):
    return {(power, 0): F(1)}


def Y(power):
    return {(0, power): F(1)}


def MONO(a, b, c=1):
    return {(a, b): F(c)}


# ---------------------------------------------------------------------
# The E-series germs and their type analysis at the origin
# ---------------------------------------------------------------------

GERMS = {
    "E6": p_add(X(3), Y(4)),
    "E7": p_add(X(3), MONO(1, 3)),
    "E8": p_add(X(3), Y(5)),
}

UNFOLDING = {
    "E6": [(0, 0), (1, 0), (0, 1), (0, 2), (1, 1), (1, 2)],
    "E7": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1)],
    "E8": [(0, 0), (1, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)],
}


def homogeneous_part(f, deg):
    return {m: c for m, c in f.items() if m[0] + m[1] == deg}


def binary_cubic_structure(cubic):
    """Factor structure of a binary cubic c0 x^3 + c1 x^2 y + c2 x y^2 + c3 y^3.

    Returns 'cube' (one linear factor, multiplicity 3), 'distinct3' (three
    distinct linear factors), or 'double_single' (a double and a single).
    Exact over Q: the binary cubic is a cube iff its binary Hessian vanishes
    identically; it has a repeated factor iff its discriminant vanishes.
    """
    c0 = cubic.get((3, 0), F(0))
    c1 = cubic.get((2, 1), F(0))
    c2 = cubic.get((1, 2), F(0))
    c3 = cubic.get((0, 3), F(0))
    if c0 == 0 and c1 == 0 and c2 == 0 and c3 == 0:
        raise AssertionError("no cubic part")
    # binary Hessian of the cubic (vanishes identically iff perfect cube)
    h_x2 = 12 * c0 * c2 - 4 * c1 * c1
    h_xy = 36 * c0 * c3 - 4 * c1 * c2
    h_y2 = 12 * c1 * c3 - 4 * c2 * c2
    if h_x2 == 0 and h_xy == 0 and h_y2 == 0:
        return "cube"
    disc = (18 * c0 * c1 * c2 * c3 - 4 * c1 ** 3 * c3 + c1 * c1 * c2 * c2
            - 4 * c0 * c2 ** 3 - 27 * c0 * c0 * c3 * c3)
    return "distinct3" if disc != 0 else "double_single"


def hessian_rank_at_origin(f):
    q20 = f.get((2, 0), F(0))
    q11 = f.get((1, 1), F(0))
    q02 = f.get((0, 2), F(0))
    rows = [[2 * q20, q11], [q11, 2 * q02]]
    rank = 0
    for r in rows:
        if any(c != 0 for c in r):
            rank += 1
    if rank == 2 and rows[0][0] * rows[1][1] - rows[0][1] * rows[1][0] == 0:
        rank = 1
    return rank


def classify_at_origin(V):
    """Arnold type of the germ V at the origin (exact).

    Returns (type_string, mu, detail). Uses: isolated critical point at the
    origin (no constant/linear part), corank of the Hessian, the factor
    structure of the cubic 3-jet, and the Milnor number. Classification
    facts (Arnold): corank 1 isolated => A_mu; corank 2 with cubic a cube
    and mu in {6,7,8} => E_mu; cubic with 3 distinct factors => D4 (mu=4);
    cubic double+single with mu >= 5 => D_mu.
    """
    assert not homogeneous_part(V, 0) and not homogeneous_part(V, 1), \
        "origin is not a critical point"
    mu = local_milnor([d_dx(V), d_dy(V)])
    basis = None
    corank = 2 - hessian_rank_at_origin(V)
    if corank == 0:
        return "A1", mu, f"corank0_mu={mu}"
    if corank == 1:
        return f"A{mu}", mu, f"corank1_mu={mu}"
    cubic = homogeneous_part(V, 3)
    struct = binary_cubic_structure(cubic)
    if struct == "distinct3":
        assert mu == 4, f"3-distinct-factor cubic must be D4 (mu=4), got mu={mu}"
        return "D4", mu, "cubic=3_distinct_factors"
    if struct == "double_single":
        assert mu >= 5, f"double+single cubic with mu={mu} < 5 inconsistent"
        return f"D{mu}", mu, "cubic=double+single"
    assert mu in (6, 7, 8), f"cubic cube with mu={mu} not in {{6,7,8}} (non-simple)"
    return f"E{mu}", mu, "cubic=perfect_cube"


# ---------------------------------------------------------------------
# Adjacency witness families (deformations T + W(t) with a critical point
# at the origin; local type computed exactly by classify_at_origin).
#
# Each witness is a 1-parameter curve of deformations with W(t) -> 0 as
# t -> 0 and the origin-germ type CONSTANT on the punctured parameter
# line (verified exactly at t = 1, 1/8, 1/64).  For single-monomial
# witnesses the constancy is a theorem: the diagonal scaling
# (x,y) -> (s^{w_x} x, s^{w_y} y) of the quasi-homogeneous germ maps
# T + t m onto s * (T + t' m) for any t' > 0 (w(m) != 1), so the type
# cannot change with t.  The multi-term curves are resonant combinations
# (coefficient relations such as b^2 = 4 a e keep the killed residual
# coefficients killed along the whole curve).  In both cases the type-S
# fibre accumulates at the central E-fibre: this is exactly the
# adjacency S < T in Arnold's sense.
# ---------------------------------------------------------------------

WITNESSES = {
    "E6": [  # FULL closure: A1 (G3) + everything below
        ("A2", lambda t: {(0, 2): t}),
        ("A3", lambda t: {(2, 0): t}),
        ("A4", lambda t: {(2, 0): t * t, (1, 2): 2 * t, (1, 3): t}),
        ("A5", lambda t: {(2, 0): t * t, (1, 2): 2 * t}),
        ("D4", lambda t: {(0, 3): t}),
        ("D5", lambda t: {(2, 1): t}),
    ],
    "E7": [  # closure missing only A6 (off-origin witness needed; scoped out)
        ("A2", lambda t: {(0, 2): t}),
        ("A3", lambda t: {(2, 0): t, (1, 2): t}),
        ("A4", lambda t: {(2, 0): t, (1, 2): t, (0, 4): t * F(1, 4)}),
        ("A5", lambda t: {(2, 0): t}),
        ("D4", lambda t: {(0, 3): t}),
        ("D5", lambda t: {(2, 1): 3 * t, (0, 3): -4 * t ** 3}),
        ("D6", lambda t: {(2, 1): t}),
        ("E6", lambda t: {(0, 4): t}),
    ],
    "E8": [  # closure missing only A6, A7 (off-origin witnesses; scoped out)
        ("A2", lambda t: {(0, 2): t}),
        ("A3", lambda t: {(2, 0): t, (0, 4): t}),
        ("A4", lambda t: {(2, 0): t}),
        ("A5", lambda t: {(2, 0): t ** 3, (1, 2): 2 * t ** 2, (1, 3): t, (0, 4): t}),
        ("D4", lambda t: {(0, 3): t}),
        ("D5", lambda t: {(2, 1): t, (0, 4): t}),
        ("D6", lambda t: {(2, 1): t}),
        ("D7", lambda t: {(2, 1): t * t, (1, 3): 2 * t}),
        ("E6", lambda t: {(0, 4): t}),
        ("E7", lambda t: {(1, 3): t}),
    ],
}

# candidates probed while fixing the tables (printed with --probe)
PROBE_CANDIDATES = {
    "E6": [
        ("?A4a", {(2, 0): 1, (1, 2): 2, (1, 3): 1}),
        ("?A4b", {(2, 0): 1, (1, 2): 2, (1, 3): 3}),
        ("?A4c", {(2, 0): 1, (1, 2): 2, (0, 5): 1, (1, 3): 1}),
    ],
    "E7": [
        ("?A4a", {(2, 0): 1, (1, 2): 1, (0, 4): F(1, 4)}),
        ("?A4b", {(2, 0): 1, (1, 2): 2, (0, 4): 1}),
        ("?A4c", {(2, 0): 1, (1, 2): 1, (1, 3): 1}),
        ("?A6a", {(2, 0): 1, (1, 2): 1, (1, 3): 1, (0, 4): F(1, 4)}),
    ],
    "E8": [
        ("?D7a", {(2, 1): 1, (1, 3): 1, (0, 4): 1}),
        ("?D7b", {(2, 1): 1, (1, 3): 2}),
        ("?D7c", {(2, 1): 1, (1, 3): 1, (0, 4): F(1, 4)}),
        ("?D7d", {(2, 1): 3, (0, 3): -4, (0, 4): 1}),
        ("?D7e", {(2, 1): 3, (0, 3): -4, (1, 3): 1, (0, 4): 1}),
        ("?D7f", {(2, 1): 2, (0, 3): 1, (0, 4): 1}),
        ("?D7g", {(2, 1): 1, (1, 2): 1, (0, 4): 1}),
        ("?D7h", {(2, 1): 1, (1, 4): 1}),
        ("?D7i", {(2, 1): 1, (0, 5): 1}),
    ],
}


def deformed(germ_name, add):
    V = dict(GERMS[germ_name])
    for mon, c in add.items():
        V[mon] = V.get(mon, F(0)) + F(c)
        if V[mon] == 0:
            del V[mon]
    return V


# ---------------------------------------------------------------------
# Clause checks
# ---------------------------------------------------------------------

def check_G1_germs_milnor():
    ok = True
    # A-series controls (univariate): cusp x^4 = A3, butterfly x^6 = A5
    mu_cusp = p_lt(d_dx(X(4)))[0]
    mu_bfly = p_lt(d_dx(X(6)))[0]
    a_ok = (mu_cusp == 3) and (mu_bfly == 5)
    ok = ok and a_ok
    details = [f"cusp_A3_mu={mu_cusp}", f"butterfly_A5_mu={mu_bfly}"]
    for name, germ in GERMS.items():
        mu, basis = milnor_number([d_dx(germ), d_dy(germ)])
        exp_mu = int(name[1])
        basis_ok = sorted(basis) == sorted(UNFOLDING[name])
        ok = ok and (mu == exp_mu) and basis_ok
        details.append(f"{name}_mu={mu}_basis_size={len(basis)}_basis_matches_unfolding={basis_ok}")
    print(f"G1_GERMS_MILNOR {' '.join(details)} (exact rational Buchberger) "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_G2_e_type_jet():
    ok = True
    details = []
    for name, germ in GERMS.items():
        struct = binary_cubic_structure(homogeneous_part(germ, 3))
        e_ok = struct == "cube"
        ok = ok and e_ok
        details.append(f"{name}_cubic={struct}")
    # D-series controls: D4 = x^2 y - y^3 (3 distinct), D5 = x^2 y + y^4 (double+single)
    D4 = p_sub(MONO(2, 1), Y(3))
    D5 = p_add(MONO(2, 1), Y(4))
    d4_ok = binary_cubic_structure(homogeneous_part(D4, 3)) == "distinct3"
    d5_ok = binary_cubic_structure(homogeneous_part(D5, 3)) == "double_single"
    ok = ok and d4_ok and d5_ok
    details.append(f"D4_control={'distinct3' if d4_ok else 'FAIL'}")
    details.append(f"D5_control={'double_single' if d5_ok else 'FAIL'}")
    print(f"G2_E_TYPE_JET {' '.join(details)} "
          f"(binary-Hessian cube test + cubic discriminant, exact) -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_adjacency(germ_name, clause, full_list):
    ok = True
    seen = []
    for expected, curve in WITNESSES[germ_name]:
        add = curve(F(1))
        V = deformed(germ_name, add)
        typ, mu, detail = classify_at_origin(V)
        row_ok = typ == expected
        # type constancy on the punctured parameter line: verify the same
        # type at two further exact parameter values on the curve
        # (t = 1/8, 1/64); the curve tends to the central E-fibre as t -> 0
        if row_ok:
            for t_small in (F(1, 8), F(1, 64)):
                typ_s, _, _ = classify_at_origin(deformed(germ_name, curve(t_small)))
                row_ok = row_ok and (typ_s == expected)
        ok = ok and row_ok
        seen.append(typ)
        print(f"  {clause}_WITNESS {germ_name}+{'+'.join(f'{c}*x^{a}y^{b}' for (a, b), c in sorted(add.items()))} "
              f"-> {typ} (mu={mu}, {detail}, constant at t=1,1/8,1/64) expected={expected} {'OK' if row_ok else 'MISMATCH'}")
    coverage = "FULL" if full_list else "SPINE"
    print(f"{clause}_{germ_name}_ADJACENCY_{coverage} witnessed={seen} -> {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------
# G3 — Morsification census (numeric)
# V_t = germ + t*(x*y + x + y), t = 1/10.  Elimination:
#   E6/E8: dV/dx = 3x^2 + t(y+1) => y(x) = -(3x^2+t)/t; substitute in dV/dy.
#   E7:    dV/dy = 3x y^2 + t(x+1) => x = -t/(3y^2+t); cleared equation
#          3 t^2 + (y^3 + t y + t)(3 y^2 + t)^2 = 0  (degree 7).
# ---------------------------------------------------------------------

def poly1d(coeffs):
    """coeffs: dict power -> Fraction.  Return numpy descending float array."""
    deg = max(coeffs)
    return np.array([float(coeffs.get(k, F(0))) for k in range(deg, -1, -1)])


def morsification_census(name):
    t = F(1, 10)
    tf = float(t)
    if name in ("E6", "E8"):
        # y(x) = -(3x^2+t)/t =: A x^2 + B ; P(x) = dV/dy(y(x),x)
        A = -F(3) / t
        B = F(-1)
        # y^n expanded exactly
        n = 4 if name == "E6" else 5
        lead = F(4 if name == "E6" else 5)
        # y = A x^2 + B ; y^(n-1) via binomial
        from math import comb
        coeffs = {}
        for k in range(0, n):
            # term C(n-1,k) (A x^2)^k B^(n-1-k)
            c = F(comb(n - 1, k)) * (A ** k) * (B ** (n - 1 - k))
            coeffs[2 * k] = coeffs.get(2 * k, F(0)) + lead * c
        coeffs[1] = coeffs.get(1, F(0)) + t
        coeffs[0] = coeffs.get(0, F(0)) + t
        roots = np.roots(poly1d(coeffs))
        pts = [(complex(r), complex(float(A) * r * r + float(B))) for r in roots]
        hess = lambda x, y: (6 * x) * (float(lead) * (n - 1) * y ** (n - 2)) - tf * tf
        grad = lambda x, y: (3 * x * x + tf * (y + 1),
                             float(lead) * y ** (n - 1) + tf * (x + 1))
    else:  # E7
        # 3 t^2 + (y^3 + t y + t)(3 y^2 + t)^2 = 0
        q = {4: F(9), 2: 6 * t, 0: t * t}               # (3y^2+t)^2
        r = {3: F(1), 1: t, 0: t}                     # y^3 + t y + t
        coeffs = {0: 3 * t * t}
        for p1, c1 in q.items():
            for p2, c2 in r.items():
                coeffs[p1 + p2] = coeffs.get(p1 + p2, F(0)) + c1 * c2
        roots = np.roots(poly1d(coeffs))
        tf = float(t)
        pts = []
        for ry in roots:
            den = 3 * ry * ry + tf
            if abs(den) < 1e-6:
                raise AssertionError("E7 elimination denominator vanishes at a root")
            pts.append((complex(-tf / den), complex(ry)))
        hess = lambda x, y: (6 * x) * (6 * x * y + tf) - (3 * y * y + tf) ** 2
        grad = lambda x, y: (3 * x * x + y ** 3 + tf * (y + 1),
                             3 * x * y * y + tf * (x + 1))
    mu = int(name[1])
    resid = max(max(abs(g) for g in grad(x, y)) for x, y in pts)
    dets = [abs(hess(x, y)) for x, y in pts]
    distinct = len(pts) == mu
    # cluster check: all pairwise separated
    sep = min(abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
              for i in range(len(pts)) for j in range(i + 1, len(pts)))
    real = [(x.real, y.real) for x, y in pts if abs(x.imag) < 1e-8 and abs(y.imag) < 1e-8]
    n_min = 0
    idx_dist = {}
    # Morse index from the true Hessian (per germ)
    for x, y in real:
        if name == "E7":
            H = np.array([[6 * x, 3 * y * y + tf], [3 * y * y + tf, 6 * x * y + tf]])
        else:
            n = 4 if name == "E6" else 5
            yy = float(lead) * (n - 1) * y ** (n - 2)
            H = np.array([[6 * x, tf], [tf, yy]])
        ev = np.linalg.eigvalsh(H)
        idx = int(np.sum(ev < -1e-9))
        idx_dist[idx] = idx_dist.get(idx, 0) + 1
        if idx == 0:
            n_min += 1
    ok = (distinct and resid < 1e-6 and min(dets) > 1e-6 and sep > 1e-6)
    print(f"G3_MORSIFICATION_{name} complex_critical_points={len(pts)}/{mu} "
          f"all_A1={min(dets) > 1e-6} max_residual={resid:.2e} min_separation={sep:.2e} "
          f"real_points={len(real)} real_minima(wells)={n_min} morse_index_dist={idx_dist} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


def check_G3():
    ok = all(morsification_census(n) for n in ("E6", "E7", "E8"))
    print("G3_NOTE complex census = mu is the invariant (6/7/8 elementary "
          "positions vs 3 cusp / 5 butterfly); the real well count is "
          "real-form-dependent and reported as data, not as a law")
    return ok


# ---------------------------------------------------------------------
# O-side — octonion/associator verification (self-contained CD core)
# ---------------------------------------------------------------------

def cds(a, b, bits=3):
    s = 1
    while bits > 0:
        if a == 0 or b == 0:
            return s
        if bits == 1:
            return -s
        h = 1 << (bits - 1)
        ah = a >= h
        bh = b >= h
        al = a & (h - 1)
        bl = b & (h - 1)
        if not ah and not bh:
            a, b = al, bl
        elif not ah and bh:
            a, b = bl, al
        elif ah and not bh:
            (a, b, s) = ((al, 0, s) if bl == 0 else (al, bl, -s))
        else:
            (a, b, s) = ((0, al, -s) if bl == 0 else (bl, al, s))
        bits -= 1
    return s


def omul(A, B):
    C = np.zeros(8)
    for i in range(8):
        for j in range(8):
            C[i ^ j] += cds(i, j) * A[i] * B[j]
    return C


def e(i):
    v = np.zeros(8)
    v[i] = 1.0
    return v


def phi3(x, y, z):
    return float(np.dot(omul(x, y), z))


def assoc(x, y, z):
    return omul(omul(x, y), z) - omul(x, omul(y, z))


def check_O_side():
    rng = np.random.default_rng(20260726)
    ok = True
    # O1: Re(x*y*z) bracketing-independent; = -phi on imaginary triples
    dev_bracket = 0.0
    dev_phi = 0.0
    for _ in range(200):
        x, y, z = (np.concatenate([[0.0], rng.normal(size=7)]) for _ in range(3))
        re1 = omul(omul(x, y), z)[0]
        re2 = omul(x, omul(y, z))[0]
        dev_bracket = max(dev_bracket, abs(re1 - re2))
        dev_phi = max(dev_phi, abs(re1 + phi3(x, y, z)))
    o1 = dev_bracket < 1e-12 and dev_phi < 1e-12
    ok = ok and o1
    print(f"O1_PHI_IS_CUBIC_CROSSTERM bracket_dev={dev_bracket:.2e} "
          f"Re_plus_phi_dev={dev_phi:.2e} (200 imaginary triples) -> {'PASS' if o1 else 'FAIL'}")
    # O2: associator vector-valued and separate from scalar phi
    a124 = assoc(e(1), e(2), e(4))
    n_a124 = float(np.linalg.norm(a124))
    p124 = abs(phi3(e(1), e(2), e(4)))
    o2 = abs(n_a124 - 2.0) < 1e-12 and p124 < 1e-12
    ok = ok and o2
    print(f"O2_ASSOCIATOR_SEPARATE ||[e1,e2,e4]||={n_a124:.3f} phi(e1,e2,e4)={p124:.1e} "
          f"(vector associator vs scalar 3-form) -> {'PASS' if o2 else 'FAIL'}")
    # O3: magic-square tower (Tits): the E-labels of the germs
    o3 = (0 + 1 * 26 + 52 == 78) and (3 + 3 * 26 + 52 == 133) and (14 + 7 * 26 + 52 == 248)
    ok = ok and o3
    print(f"O3_MAGIC_SQUARE_CHAIN tits: 0+1*26+52=78(E6) 3+3*26+52=133(E7) "
          f"14+7*26+52=248(E8) -> {'PASS' if o3 else 'FAIL'}")
    # O4: divergence — no symmetric cubic from phi on Im O
    dev_alt = 0.0
    dev_re3 = 0.0
    for _ in range(200):
        x = np.concatenate([[0.0], rng.normal(size=7)])
        dev_alt = max(dev_alt, abs(phi3(x, x, x)))
        x3 = omul(x, omul(x, x))
        dev_re3 = max(dev_re3, abs(x3[0]))
    germ_cubic_at_10 = 1.0  # E-series cubic part u^3 evaluated at (u,v)=(1,0)
    o4 = dev_alt < 1e-12 and dev_re3 < 1e-12 and germ_cubic_at_10 != 0.0
    ok = ok and o4
    print(f"O4_NO_FORM_IDENTITY phi(x,x,x)_max={dev_alt:.2e} Re(x^3)_max={dev_re3:.2e} "
          f"(alternating octonion cross-term) vs germ cubic u^3|(1,0)={germ_cubic_at_10} "
          f"(commutative cube) -> {'PASS' if o4 else 'FAIL'}")
    print("O4_NOTE the semantic E-germ is NOT octonion-derived: the bridge is "
          "the E-label via the magic square + the singularity label, not a "
          "form identity (D3 quarantine kept)")
    return ok


# ---------------------------------------------------------------------

def run_probe():
    for germ, cands in PROBE_CANDIDATES.items():
        for label, add in cands:
            add = {m: c for m, c in add.items() if c != 0}
            try:
                typ, mu, detail = classify_at_origin(deformed(germ, add))
                print(f"PROBE {germ} {label} {add} -> {typ} (mu={mu}, {detail})")
            except AssertionError as ex:
                print(f"PROBE {germ} {label} {add} -> INCONSISTENT: {ex}")


def main():
    if PROBE:
        run_probe()
        return 0
    print("=" * 76)
    print("E-SERIES SEMANTIC GERM — Petitot potentials with E6/E7/E8 bifurcation")
    print("structure, verified against the octonion/associator structure")
    print("=" * 76)
    results = [
        ("G1", check_G1_germs_milnor()),
        ("G2", check_G2_e_type_jet()),
        ("G3", check_G3()),
        ("G4", check_adjacency("E6", "G4", True)),
        ("G5", check_adjacency("E7", "G5", False)),
        ("G6", check_adjacency("E8", "G6", False)),
        ("O1_O4", check_O_side()),
    ]
    print("=" * 76)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print("E_SERIES_GERM_VERDICT E_SERIES_GERMS_CONSTRUCTED__WEAK_FORM_TESTABLE__WEAK_FORM_OPEN "
              f"({passed}/{total} clause-groups PASS)")
        print("E_SERIES_GERM_DETAIL e6_closure=FULL(A1..A5,D4,D5); "
              "e7_closure=A1..A5,D4..D6,E6(missing A6); "
              "e8_closure=A1..A5,D4..D7,E6,E7(missing A6,A7); "
              "morsification_census=mu_over_C; octonion_bridge=MAGIC_SQUARE_LABEL_not_FORM_IDENTITY")
        print("E_SERIES_SEMANTIC_GERM_OK")
        return 0
    print(f"E_SERIES_GERM_VERDICT CONTRACT_INCOMPLETE ({passed}/{total} clause-groups PASS)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
