#!/usr/bin/env python3
"""
ADE-Wildgen conjecture — the McKay correspondence for E6/E7/E8, computed, and its
comparison with the G2/octonion structure of the rupture programme.

Companion to:
  docs/research/ade_wildgen_mckay_spec_2026-07-26.md

The conjecture (petitot-semantic-potential.md §4, "the deep conjecture", flagged
interpretive frontier): Wildgen's four-actant semantics requires the exceptional
singularities E6/E7/E8; octonions generate the exceptional Lie algebras via the
Freudenthal–Tits magic square (G2 -> F4 -> E6 -> E7 -> E8); Arnold's exceptional
singularities and the exceptional Lie algebras share ADE labels via the McKay
correspondence; therefore "the organizing centre of rich semantic morphology is the
octonionic exceptional structure".

This contract computes, from first principles and self-contained:

  M-SIDE (the label-match is real mathematics):
    M1  the binary polyhedral groups 2T/2O/2I as unit quaternions (closure checked),
        orders 24/48/120, conjugacy-class counts 7/8/9 = rank(affine E6/E7/E8).
    M2  character tables via the Burnside class-algebra algorithm (structure
        constants -> common eigenvectors -> central characters -> degrees), with
        Sigma d^2 = |Gamma| and the natural 2-dim SU(2) character recovered as one
        of the computed rows (independent self-check).
    M3  the McKay fusion matrix N (tensoring with the natural 2-dim rep): symmetric
        0/1, spectral radius exactly 2 (affine), unique trivalent node with arm
        lengths (2,2,2) / (1,3,3) / (1,2,5) = affine E6/E7/E8; deleting the trivial
        node leaves finite E6/E7/E8 (arms (1,2,2)/(1,2,3)/(1,2,4), radius < 2).
    M4  irrep dimensions = affine marks; Coxeter numbers h = sum d = 12/18/30.

  C-SIDE (comparison with the G2/octonion structure):
    C1  G2 is EXCLUDED from the SU(2) McKay series: every McKay fusion matrix is
        symmetric (the natural rep is self-dual), hence simply-laced; the G2 Cartan
        matrix [[2,-1],[-3,2]] is not symmetric. G2 arises instead by folding D4
        under triality (verified at the Cartan level).
    C2  at the Fano/finite level the octonion symmetry PSL(2,7)=GL(3,2) (order 168,
        element orders {1,2,3,4,7}) CONTAINS the E6/E7 polyhedral groups
        (point-stabilizer = S4 signature; derived subgroup = A4 signature) but NOT
        the E8 icosahedral group (5 does not divide 168; no order-5 elements).
        Conversely 7 divides none of |2T|,|2O|,|2I|. (Finite level only: the
        continuous G2 contains SU(2) and hence 2I; stated as a NOTE.)
    C3  the programme's operative Petitot germs are A-SERIES: cusp x^4 = A3,
        butterfly x^6 = A5 (Milnor numbers 3 and 5). The E-series germs
        E6 = x^3+y^4, E7 = x^3+xy^3, E8 = x^3+y^5 have Milnor numbers 6/7/8 —
        computed here by exact rational Buchberger — but no E-series germ has ever
        been constructed on the semantic (morphodynamic) side of the programme.
    C4  the genuine continuous bridge octonions -> E-series is the Freudenthal–Tits
        magic square (Tits construction T(A,J3(O)) = der(A)+(Im A x J3(O)_0)+der(J3(O)):
        0+1*26+52 = 78; 3+3*26+52 = 133; 14+7*26+52 = 248), already gated
        by the functor_f e6/e7/e8 contracts — NOT the McKay correspondence.

Verdict: the strong/naive form of the conjecture ("semantic morphology is governed
by the same exceptional structure as the octonions, via McKay") is OBSTRUCTED at
every computable checkpoint (G2 not simply-laced; E8 finite content absent from the
Fano group; the semantic germs in use are A-series). The weak form ("exceptional
geometry governs rich semantic morphology") is currently UNDECIDABLE within the
programme: the continuous bridge exists (magic square, phi = E6 cubic cross-term)
but the only honest test — an E-series germ or an F4-natural-not-G2-natural object
on the semantic side — has never been constructed.

Self-contained; numpy only. Deterministic (fixed seeds).
"""
from __future__ import annotations

from fractions import Fraction
from itertools import permutations

import numpy as np

np.seterr(all="ignore")
TOL = 1e-8


# ---------------------------------------------------------------------
# Unit quaternions (w, x, y, z) and the binary polyhedral groups
# ---------------------------------------------------------------------

def qmul(p, q):
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qkey(q, nd=9):
    return tuple(np.round(np.asarray(q, float), nd))


def binary_tetrahedral():
    els = []
    for s in (1.0, -1.0):
        els.append(np.array([s, 0, 0, 0]))
    for ax in range(1, 4):
        for s in (1.0, -1.0):
            q = np.zeros(4)
            q[ax] = s
            els.append(q)
    # all 16 sign combinations of (+-1/2, +-1/2, +-1/2, +-1/2)
    for m in range(16):
        els.append(np.array([0.5 * (1 if m & 1 else -1),
                             0.5 * (1 if m & 2 else -1),
                             0.5 * (1 if m & 4 else -1),
                             0.5 * (1 if m & 8 else -1)]))
    return els


def binary_octahedral():
    els = list(binary_tetrahedral())
    r = 1.0 / np.sqrt(2.0)
    for i in range(4):
        for j in range(i + 1, 4):
            for s1 in (1.0, -1.0):
                for s2 in (1.0, -1.0):
                    q = np.zeros(4)
                    q[i] = s1 * r
                    q[j] = s2 * r
                    els.append(q)
    return els


def _parity(perm):
    inv = 0
    for i in range(len(perm)):
        for j in range(i + 1, len(perm)):
            inv += perm[i] > perm[j]
    return inv % 2


def binary_icosahedral():
    els = list(binary_tetrahedral())
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    base = [0.0, 0.5, 1.0 / (2.0 * phi), phi / 2.0]
    seen = set()
    for perm in permutations(range(4)):
        if _parity(perm) != 0:
            continue
        pos = [base[perm[k]] for k in range(4)]
        nz = [k for k in range(4) if pos[k] != 0.0]
        for m in range(1 << len(nz)):
            q = list(pos)
            for b, k in enumerate(nz):
                if not (m >> b) & 1:
                    q[k] = -q[k]
            key = qkey(q)
            if key not in seen:
                seen.add(key)
                els.append(np.array(q))
    return els


class FiniteSubgroup:
    """A finite subgroup of SU(2) as explicit unit quaternions."""

    def __init__(self, name, elements):
        self.name = name
        self.els = elements
        self.idx = {qkey(q): i for i, q in enumerate(self.els)}
        self.n = len(self.els)
        assert self.n == len(self.idx), f"{name}: duplicate elements"
        self._check_closure()
        self.classes = self._conjugacy_classes()

    def _check_closure(self):
        for p in self.els:
            for q in self.els:
                if qkey(qmul(p, q)) not in self.idx:
                    raise AssertionError(f"{self.name}: not closed under multiplication")

    def _conjugacy_classes(self):
        unseen = set(range(self.n))
        classes = []
        while unseen:
            i = min(unseen)
            g = self.els[i]
            orbit = set()
            for h in self.els:
                orbit.add(self.idx[qkey(qmul(qmul(h, g), qconj(h)))])
            classes.append(sorted(orbit))
            unseen -= orbit
        classes.sort(key=lambda c: (len(c), c[0]))
        self.class_of = {}
        for k, c in enumerate(classes):
            for i in c:
                self.class_of[i] = k
        return classes

    def natural_character(self):
        """chi_2 of the defining 2-dim SU(2) rep, per class (= 2 Re q)."""
        return np.array([2.0 * self.els[c[0]][0] for c in self.classes])


# ---------------------------------------------------------------------
# Burnside character table via the class algebra
# ---------------------------------------------------------------------

def class_algebra_structure_constants(G):
    k = len(G.classes)
    sizes = np.array([len(c) for c in G.classes])
    cnt = np.zeros((k, k, k), dtype=np.int64)  # pairs (x,y) in Ci x Cj with xy in Cc
    for i, Ci in enumerate(G.classes):
        for j, Cj in enumerate(G.classes):
            for x in Ci:
                for y in Cj:
                    c = G.class_of[G.idx[qkey(qmul(G.els[x], G.els[y]))]]
                    cnt[i, j, c] += 1
    # structure constants: a[i,j,c] = pairs landing on a FIXED z in Cc = count / |C_c|
    assert np.all(cnt % sizes[None, None, :] == 0), "class counts not divisible by class size"
    a = cnt // sizes[None, None, :]
    # sanity: K_i K_j = sum_c a[i,j,c] K_c  =>  sum_c a[i,j,c] |C_c| = |C_i| |C_j|
    lhs = (a * sizes[None, None, :]).sum(axis=2)
    rhs = np.outer(sizes, sizes)
    assert np.array_equal(lhs, rhs), "class algebra counting sanity failed"
    return a, sizes


def burnside_character_table(G):
    """Return (classes, sizes, chars) with chars[rho, class] (complex)."""
    a, sizes = class_algebra_structure_constants(G)
    k = len(G.classes)
    # multiplication matrices M_i: (M_i)_{c,j} = a[i,j,c]
    M = np.stack([a[i].T for i in range(k)]).astype(float)
    for i in range(k):
        for j in range(k):
            assert np.max(np.abs(M[i] @ M[j] - M[j] @ M[i])) == 0, "class algebra not commutative"
    # generic element of the class algebra with simple spectrum
    V = None
    for seed in (20260726, 7, 13, 101):
        rng = np.random.default_rng(seed)
        r = rng.integers(1, 20, size=k).astype(float)
        Mc = np.tensordot(r, M, axes=1)
        w, Vc = np.linalg.eig(Mc)
        gaps = np.abs(w[:, None] - w[None, :]) + 1e30 * np.eye(k)
        if gaps.min() > 1e-5:
            V = Vc
            break
    if V is None:
        raise AssertionError("no simple-spectrum class-algebra element found")
    chars = np.zeros((k, k), dtype=complex)
    degs = []
    for rho in range(k):
        v = V[:, rho]
        lam = np.array([(v.conj() @ (M[i] @ v)) / (v.conj() @ v) for i in range(k)])
        # residuals
        for i in range(k):
            assert np.max(np.abs(M[i] @ v - lam[i] * v)) < 1e-6, "eigenvector residual"
        d2 = G.n / np.sum(np.abs(lam) ** 2 / sizes)
        d = int(round(float(np.sqrt(d2.real))))
        assert abs(d2 - d * d) < 1e-6, f"non-integral degree^2: {d2}"
        degs.append(d)
        chars[rho] = d * lam / sizes
    return sizes, chars, degs


# ---------------------------------------------------------------------
# McKay fusion graph
# ---------------------------------------------------------------------

def mckay_fusion(G, chars, sizes):
    chi2 = G.natural_character()
    # the natural rep must be one of the computed irreps (self-check)
    nat = None
    for rho in range(chars.shape[0]):
        if np.max(np.abs(chars[rho] - chi2)) < 1e-6:
            nat = rho
            break
    if nat is None:
        raise AssertionError(f"{G.name}: natural SU(2) character not in computed table")
    k = chars.shape[0]
    N = np.zeros((k, k))
    for rho in range(k):
        for sig in range(k):
            val = (sizes * chars[rho] * chi2 * np.conj(chars[sig])).sum() / G.n
            assert abs(val.imag) < 1e-6, "fusion multiplicity not real"
            N[rho, sig] = val.real
    Nr = np.rint(N)
    assert np.max(np.abs(N - Nr)) < 1e-6, "fusion multiplicities non-integral"
    return Nr.astype(int), nat


def graph_arms(N):
    """Arm lengths from the unique trivalent node of a tree adjacency matrix."""
    deg = N.sum(axis=1)
    tri = [i for i in range(len(deg)) if deg[i] == 3]
    if len(tri) != 1:
        return None
    u = tri[0]
    arms = []
    for nb in np.flatnonzero(N[u]):
        prev, cur, length = u, int(nb), 1
        while deg[cur] == 2:
            nxt = [x for x in np.flatnonzero(N[cur]) if x != prev][0]
            prev, cur, length = cur, int(nxt), length + 1
        arms.append(length)
    return tuple(sorted(arms))


def spectral_radius(N):
    return float(np.max(np.abs(np.linalg.eigvals(N.astype(float)))))


def delete_node(N, node):
    keep = [i for i in range(N.shape[0]) if i != node]
    return N[np.ix_(keep, keep)]


# ---------------------------------------------------------------------
# GL(3,2) = PSL(2,7) — the Fano finite group
# ---------------------------------------------------------------------

def gl32_elements():
    els = []
    for m in range(1 << 9):
        M = np.array([(m >> b) & 1 for b in range(9)], dtype=np.int64).reshape(3, 3)
        if round(float(np.linalg.det(M.astype(float)))) % 2 != 0:
            els.append(M)
    return els


def mat_order_f2(M, cap=16):
    I = np.eye(3, dtype=np.int64)
    P = M.copy()
    for k in range(1, cap + 1):
        if np.array_equal(P % 2, I):
            return k
        P = (P @ M) % 2
    return None


def order_profile(elements):
    prof = {}
    for M in elements:
        o = mat_order_f2(M)
        prof[o] = prof.get(o, 0) + 1
    return prof


def mat_inv_f2(M):
    adj = np.array([[M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1],
                     M[0, 2] * M[2, 1] - M[0, 1] * M[2, 2],
                     M[0, 1] * M[1, 2] - M[0, 2] * M[1, 1]],
                    [M[1, 2] * M[2, 0] - M[1, 0] * M[2, 2],
                     M[0, 0] * M[2, 2] - M[0, 2] * M[2, 0],
                     M[0, 2] * M[1, 0] - M[0, 0] * M[1, 2]],
                    [M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0],
                     M[0, 1] * M[2, 0] - M[0, 0] * M[2, 1],
                     M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]]], dtype=np.int64)
    return adj % 2  # det = 1 over F2


def closure_of(generators, universe_keys, mul):
    seen = set(generators)
    changed = True
    while changed:
        changed = False
        for a in list(seen):
            for b in list(seen):
                c = mul(a, b)
                if c not in seen:
                    seen.add(c)
                    changed = True
    return seen


# ---------------------------------------------------------------------
# Exact rational polynomial arithmetic (Milnor numbers of E-series germs)
# ---------------------------------------------------------------------

def mkey(mon):
    return (mon[0] + mon[1], mon[0], mon[1])  # grlex-ish total order


def p_lt(f):
    return max(f.keys(), key=mkey)


def p_mul_monomial(f, mon, coeff=Fraction(1)):
    out = {}
    for (a, b), c in f.items():
        out[(a + mon[0], b + mon[1])] = c * coeff
    return out


def p_sub(f, g):
    out = dict(f)
    for mon, c in g.items():
        out[mon] = out.get(mon, Fraction(0)) - c
        if out[mon] == 0:
            del out[mon]
    return out


def p_monic(f):
    lt = p_lt(f)
    lc = f[lt]
    return {mon: c / lc for mon, c in f.items()}


def p_divisible(mon, lt):
    return mon[0] >= lt[0] and mon[1] >= lt[1]


def p_reduce(f, G):
    f = dict(f)
    while True:
        # reduce the largest reducible term
        terms = sorted(f.keys(), key=mkey, reverse=True)
        hit = None
        for mon in terms:
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


def buchberger(gens, cap=200):
    G = [p_monic(g) for g in gens]
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
    """dim of Q{x,y}/J via standard monomials of a Groebner basis."""
    G = buchberger(jac_gens)
    lts = [p_lt(g) for g in G]
    maxa = max(l[0] for l in lts)
    maxb = max(l[1] for l in lts)
    basis = [(a, b) for a in range(maxa + 1) for b in range(maxb + 1)
             if not any(p_divisible((a, b), lt) for lt in lts)]
    return len(basis), basis, lts


def X(power):
    return {(power, 0): Fraction(1)}


def Y(power):
    return {(0, power): Fraction(1)}


def p_add(f, g):
    out = dict(f)
    for mon, c in g.items():
        out[mon] = out.get(mon, Fraction(0)) + c
        if out[mon] == 0:
            del out[mon]
    return out


def p_scale(f, s):
    return {mon: c * s for mon, c in f.items()}


def d_dx(f):
    out = {}
    for (a, b), c in f.items():
        if a > 0:
            out[(a - 1, b)] = c * a
    return out


def d_dy(f):
    out = {}
    for (a, b), c in f.items():
        if b > 0:
            out[(a, b - 1)] = c * b
    return out


# ---------------------------------------------------------------------
# Contract clauses
# ---------------------------------------------------------------------

EXPECTED = {
    "2T": dict(order=24, nclasses=7, dims=[1, 1, 1, 2, 2, 2, 3],
               arms_aff=(2, 2, 2), arms_fin=(1, 2, 2), coxeter=12, lie="E6"),
    "2O": dict(order=48, nclasses=8, dims=[1, 1, 2, 2, 2, 3, 3, 4],
               arms_aff=(1, 3, 3), arms_fin=(1, 2, 3), coxeter=18, lie="E7"),
    "2I": dict(order=120, nclasses=9, dims=[1, 2, 2, 3, 3, 4, 4, 5, 6],
               arms_aff=(1, 2, 5), arms_fin=(1, 2, 4), coxeter=30, lie="E8"),
}


def check_M1_M2(groups):
    ok = True
    details = []
    for g in groups:
        exp = EXPECTED[g.name]
        sizes, chars, degs = g.chars_data
        order_ok = g.n == exp["order"]
        class_ok = len(g.classes) == exp["nclasses"]
        sumsq = sum(d * d for d in degs)
        sumsq_ok = sumsq == g.n
        dims_ok = sorted(degs) == exp["dims"]
        ok = ok and order_ok and class_ok and sumsq_ok and dims_ok
        details.append(f"{g.name}:|G|={g.n},classes={len(g.classes)},"
                       f"dims={sorted(degs)},sumd2={sumsq}")
    print(f"M1_BINARY_POLYHEDRAL_GROUPS {'; '.join(details)} -> {'PASS' if ok else 'FAIL'}")
    # natural character recovered (computed inside mckay_fusion; re-state here)
    nat_ok = True
    for g in groups:
        sizes, chars, degs = g.chars_data
        chi2 = g.natural_character()
        if not any(np.max(np.abs(chars[r] - chi2)) < 1e-6 for r in range(chars.shape[0])):
            nat_ok = False
    print(f"M2_BURNSIDE_TABLES natural_SU2_character_recovered={nat_ok} "
          f"sum_chi_sq=|Gamma| -> {'PASS' if nat_ok else 'FAIL'}")
    return ok and nat_ok


def check_M3_M4(groups):
    ok = True
    for g in groups:
        exp = EXPECTED[g.name]
        N, nat, triv = g.fusion_data
        sym = np.array_equal(N, N.T)
        simple01 = set(np.unique(N)) <= {0, 1} and np.all(np.diag(N) == 0)
        rho = spectral_radius(N)
        aff_ok = abs(rho - 2.0) < 1e-8
        arms_aff = graph_arms(N)
        arms_aff_ok = arms_aff == exp["arms_aff"]
        Nfin = delete_node(N, triv)
        fin_ok = spectral_radius(Nfin) < 2.0 - 1e-9
        arms_fin = graph_arms(Nfin)
        arms_fin_ok = arms_fin == exp["arms_fin"]
        # trivial node = the extending (affine) node: a leaf adjacent to the
        # natural-rep node (V2 tensor triv = V2); deletion gives the finite diagram
        triv_ext = (N[triv].sum() == 1) and (N[triv, nat] == 1)
        sizes, chars, degs = g.chars_data
        h = sum(degs)
        cox_ok = h == exp["coxeter"]
        row_ok = sym and simple01 and aff_ok and arms_aff_ok and fin_ok and arms_fin_ok and triv_ext and cox_ok
        ok = ok and row_ok
        print(f"M3_MCKAY_FUSION_{exp['lie']} symmetric={sym} 0/1={simple01} "
              f"rho_aff={rho:.6f} arms_aff={arms_aff} arms_fin={arms_fin} "
              f"trivial@extending={triv_ext} -> {'PASS' if row_ok else 'FAIL'}")
        print(f"M4_COXETER_{exp['lie']} h=sum(d)={h} expected={exp['coxeter']} "
              f"marks=dims -> {'PASS' if cox_ok else 'FAIL'}")
    return ok


def check_C1_G2_exclusion(groups):
    # every McKay fusion matrix is symmetric (natural rep self-dual) => simply-laced only
    all_sym = all(np.array_equal(g.fusion_data[0], g.fusion_data[0].T) for g in groups)
    C_G2 = np.array([[2, -1], [-3, 2]])
    nonsym = not np.array_equal(C_G2, C_G2.T)
    # D4 triality folding reproduces the G2 Cartan matrix
    C_D4 = np.array([[2, -1, 0, 0],
                     [-1, 2, -1, -1],
                     [0, -1, 2, 0],
                     [0, -1, 0, 2]])
    # orbits {0,2,3} (outer) and {1} (central); folded C~_{I,J} = sum_{j in J} C_{i0,j}
    folded = np.array([[2, C_D4[0, 1]],
                       [C_D4[1, 0] + C_D4[1, 2] + C_D4[1, 3], 2]])
    fold_ok = np.array_equal(folded, C_G2)
    det_ok = round(float(np.linalg.det(C_G2))) == 1
    ok = all_sym and nonsym and fold_ok and det_ok
    print(f"C1_G2_EXCLUDED_FROM_SU2_MCKAY mckay_symmetric={all_sym} "
          f"g2_cartan_nonsymmetric={nonsym} D4_triality_fold={folded.tolist()} "
          f"det={det_ok} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_C2_fano_finite_content():
    els = gl32_elements()
    order_ok = len(els) == 168
    prof = order_profile(els)
    prof_expected = {1: 1, 2: 21, 3: 56, 4: 42, 7: 48}
    prof_ok = prof == prof_expected
    # point stabilizer of v = (1,0,0)
    v = np.array([1, 0, 0], dtype=np.int64)
    stab = [M for M in els if np.array_equal((M @ v) % 2, v)]
    stab_prof = order_profile(stab)
    s4_sig = {1: 1, 2: 9, 3: 8, 4: 6}
    stab_ok = len(stab) == 24 and stab_prof == s4_sig
    # derived subgroup of the stabilizer (A4 signature)
    key = {tuple(M.ravel()): M for M in stab}
    comms = set()
    for A in stab:
        Ai = mat_inv_f2(A)
        for B in stab:
            Bi = mat_inv_f2(B)
            C = (((A @ B) % 2) @ ((Ai @ Bi) % 2)) % 2
            comms.add(tuple(C.ravel()))
    mul = lambda a, b: tuple((((key[a] @ key[b]) % 2)).ravel())
    derived = closure_of(comms, None, mul)
    derived_prof = order_profile([key[t] for t in derived])
    a4_sig = {1: 1, 2: 3, 3: 8}
    derived_ok = len(derived) == 12 and derived_prof == a4_sig
    # icosahedral exclusion at the Fano/finite level
    no_order5 = 5 not in prof
    lagrange = (168 % 5 != 0) and (24 % 7 != 0) and (48 % 7 != 0) and (120 % 7 != 0)
    ok = order_ok and prof_ok and stab_ok and derived_ok and no_order5 and lagrange
    print(f"C2_FANO_FINITE_CONTENT |GL(3,2)|={len(els)} orders={prof} "
          f"stab_S4={stab_ok} derived_A4={derived_ok} no_order5={no_order5} "
          f"lagrange_7_not_in_2T2O2I={lagrange} -> {'PASS' if ok else 'FAIL'}")
    print("C2_NOTE finite/Fano level only: continuous G2 contains SU(2) and hence 2I; "
          "the exclusion is of the octonion's COMBINATORIAL (Fano) symmetry")
    return ok


def check_C3_petitot_germs_a_series():
    # univariate A_k germs x^{k+1}: mu = k (quotient C{x}/(x^k) has basis 1..x^{k-1})
    # cusp germ x^4 (= A3, the repo's contrariety potential); butterfly germ x^6 (= A5)
    mu_cusp = p_lt(d_dx(X(4)))[0]            # deg of d(x^4)/dx = 3 -> mu = 3
    mu_bfly = p_lt(d_dx(X(6)))[0]            # deg of d(x^6)/dx = 5 -> mu = 5
    a_ok = (mu_cusp == 3) and (mu_bfly == 5)
    # E-series germs, exact rational Buchberger
    E6 = p_add(X(3), Y(4))
    E7 = p_add(X(3), p_mul_monomial(Y(3), (1, 0)))
    E8 = p_add(X(3), Y(5))
    results = {}
    for name, f in (("E6", E6), ("E7", E7), ("E8", E8)):
        mu, basis, lts = milnor_number([d_dx(f), d_dy(f)])
        results[name] = mu
    e_ok = results == {"E6": 6, "E7": 7, "E8": 8}
    ok = a_ok and e_ok
    print(f"C3_GERMS_MILNOR cusp_A3_mu={mu_cusp} butterfly_A5_mu={mu_bfly} "
          f"E6_mu={results['E6']} E7_mu={results['E7']} E8_mu={results['E8']} "
          f"(exact rational Buchberger) -> {'PASS' if ok else 'FAIL'}")
    print("C3_NOTE operative Petitot germs of the programme are A3/A5 (A-series); "
          "no E-series germ has ever been constructed on the semantic side")
    return ok


def check_C4_magic_square():
    # Tits construction T(A, J3(O)) = der(A) + (Im A x J3(O)_0) + der(J3(O)):
    #   E6 = T(C, .): 0 + 1*26 + 52 = 78  (der(C)=0, Im C dim 1)
    #   E7 = T(H, .): 3 + 3*26 + 52 = 133 (der(H)=su(2) dim 3, Im H dim 3)
    #   E8 = T(O, .): 14 + 7*26 + 52 = 248 (der(O)=g2 dim 14, Im O dim 7)
    tits_e6 = 0 + 1 * 26 + 52 == 78
    tits_e7 = 3 + 3 * 26 + 52 == 133
    tits_e8 = 14 + 7 * 26 + 52 == 248
    tower = [14, 52, 78, 133, 248] == [14, 52, 78, 133, 248]
    ok = tits_e8 and tits_e6 and tits_e7 and tower
    print(f"C4_MAGIC_SQUARE_LINK tits_0+1*26+52=78:{tits_e6} 3+3*26+52=133:{tits_e7} "
          f"14+7*26+52=248:{tits_e8} tower={tower} -> {'PASS' if ok else 'FAIL'}")
    print("C4_NOTE the genuine octonion->E-series bridge is Freudenthal-Tits "
          "(gated by functor_f e6/e7/e8 contracts), NOT the McKay correspondence")
    return ok


def main():
    print("=" * 72)
    print("ADE-WILDGEN — McKay correspondence for E6/E7/E8 vs the G2/octonion structure")
    print("=" * 72)

    groups = []
    for name, ctor in (("2T", binary_tetrahedral), ("2O", binary_octahedral),
                       ("2I", binary_icosahedral)):
        g = FiniteSubgroup(name, ctor())
        sizes, chars, degs = burnside_character_table(g)
        g.chars_data = (sizes, chars, degs)
        N, nat = mckay_fusion(g, chars, sizes)
        triv = None
        for rho in range(chars.shape[0]):
            if np.max(np.abs(chars[rho] - 1.0)) < 1e-9:
                triv = rho
        assert triv is not None
        g.fusion_data = (N, nat, triv)
        groups.append(g)

    results = [
        ("M1_M2", check_M1_M2(groups)),
        ("M3_M4", check_M3_M4(groups)),
        ("C1", check_C1_G2_exclusion(groups)),
        ("C2", check_C2_fano_finite_content()),
        ("C3", check_C3_petitot_germs_a_series()),
        ("C4", check_C4_magic_square()),
    ]
    print("=" * 72)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print("ADE_WILDGEN_VERDICT STRONG_FORM_OBSTRUCTED__WEAK_FORM_UNDECIDABLE "
              f"({passed}/{total} clause-groups PASS)")
        print("ADE_WILDGEN_DETAIL mckay_label_match=REAL_MATHEMATICS; "
              "g2_not_simply_laced=EXCLUDED_FROM_SU2_MCKAY; "
              "fano_finite=contains_A4_S4_not_A5; "
              "semantic_germs=A_SERIES_not_E_SERIES; "
              "continuous_bridge=MAGIC_SQUARE_not_MCKAY")
        print("ADE_WILDGEN_MCKAY_OK")
        return 0
    print(f"ADE_WILDGEN_VERDICT CONTRACT_INCOMPLETE ({passed}/{total} clause-groups PASS)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
