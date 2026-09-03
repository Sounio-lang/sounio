#!/usr/bin/env python3
"""
Functor F — the Fano / PSL(2,7) thread: one symmetry unifies the whole order tower.

A POSITIVE cross-order result (not an obstruction). The functor-F algebraic column is
built on the octonion G2 3-form phi, whose 7 Fano lines are its structure constants.
The programme's ord-2 sensor is the sedenion zero-divisor (ZD) geometry, known to carry
a 7-fibre / 168=|PSL(2,7)| orbit structure (prior work: frente-B PR #660; Moreno;
Kirshtein). This rung threads them: it shows the SAME Fano plane / PSL(2,7) indexes
BOTH, so ord-1 (phi), ord-2 (ZD fibre) and ord-3 (the secondary ternary operation, which
the ord-3 rung located on the ZD fibre) are three faces of one symmetry.

  W1  The 42 sedenion ZD e_i+e_j fall into 7 fibres of 6.
  W2  Each ZD e_i + e_{8+k} has fibre octonion-support = the COMPLEMENT of the Fano line
      through (i,k); the 7 fibre-lines are EXACTLY the 7 phi 3-form structure-constant
      lines. (This is the new connection: prior work has the 7 fibres, this identifies
      their index with the functor-F 3-form.)
  W3  An explicit PSL(2,7) Fano collineation (a signed-permutation octonion automorphism,
      order 3) lifted diagonally to Aut(S) permutes the ZD fibres by the SAME permutation
      pi it induces on the phi lines -- equivariance verified on all 42 ZD.
  W4  Hence the order tower threads one PSL(2,7): ord-1 phi lines, ord-2 ZD fibres, and
      the ord-3 secondary operation (parent rung, SECONDARY_TERNARY_LOCATED) sit on the
      same 7 Fano lines under the same group action.

Honest scope: the 7-fibre/168 ZD structure is PRIOR; the contribution is the explicit
phi-line == fibre-line identity + the equivariant thread across orders. Not an identity
of the 'X = ZD' (D3) kind -- it is a shared symmetry/indexing, operational.

Self-contained (Cayley-Dickson bits=3 and 4); embeds a core axiom-audit.
"""
import numpy as np

np.seterr(all='ignore')
TOL = 1e-9


def cds(a, b, bits):
    s = 1
    while bits > 0:
        if a == 0 or b == 0:
            return s
        if bits == 1:
            return -s
        h = 1 << (bits - 1)
        ah, bh = a >= h, b >= h
        al, bl = a & (h - 1), b & (h - 1)
        if not ah and not bh:
            a, b = al, bl
        elif not ah and bh:
            a, b = bl, al
        elif ah and not bh:
            a, b, s = ((al, 0, s) if bl == 0 else (al, bl, -s))
        else:
            a, b, s = ((0, al, -s) if bl == 0 else (bl, al, s))
        bits -= 1
    return s


def mul(A, B, bits):
    n = 1 << bits
    C = np.zeros(n)
    for i in range(n):
        if A[i] == 0.0:
            continue
        for j in range(n):
            if B[j] == 0.0:
                continue
            C[i ^ j] += cds(i, j, bits) * A[i] * B[j]
    return C


def e(i, n):
    v = np.zeros(n); v[i] = 1.0; return v


def Lmat4(b):
    return np.column_stack([mul(b, e(k, 16), 4) for k in range(16)])


def nullspace(M):
    _, s, vh = np.linalg.svd(M)
    return vh[np.sum(s > TOL):]


def audit_core():
    def chk(bits):
        n = 1 << bits
        ident = all(np.allclose(mul(e(0, n), e(j, n), bits), e(j, n)) for j in range(n))
        sq = all(np.allclose(mul(e(i, n), e(i, n), bits), -e(0, n)) for i in range(1, n))
        anti = all(np.allclose(mul(e(i, n), e(j, n), bits), -mul(e(j, n), e(i, n), bits))
                   for i in range(1, n) for j in range(1, n) if i != j)
        return ident and sq and anti
    return chk(3) and chk(4)


def phi_lines():
    L = set()
    for i in range(1, 8):
        for j in range(1, 8):
            if i != j:
                k = int(np.argmax(np.abs(mul(e(i, 8), e(j, 8), 3))[1:])) + 1
                L.add(frozenset({i, j, k}))
    return L


def zd_line(b):
    F = nullspace(Lmat4(b))
    supp = frozenset(x % 8 for v in F for x in np.nonzero(np.abs(v) > TOL)[0])
    return frozenset(range(1, 8)) - supp


def zd_list():
    return [(i, j) for i in range(1, 16) for j in range(i + 1, 16)
            if 16 - np.linalg.matrix_rank(Lmat4(e(i, 16) + e(j, 16)), tol=TOL) > 0]


def fano_collineation_auto():
    """A PSL(2,7) element: bit-rotation on 3-bit indices + signs making it an octonion auto."""
    Mbits = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    pi = {}
    for i in range(1, 8):
        b = np.array([(i >> k) & 1 for k in range(3)])
        j = Mbits @ b % 2
        pi[i] = int(j[0] | (j[1] << 1) | (j[2] << 2))
    for mask in range(128):
        s = [1] + [1 if (mask >> k) & 1 == 0 else -1 for k in range(7)]
        M = np.zeros((8, 8)); M[:, 0] = e(0, 8)
        for i in range(1, 8):
            M[:, i] = s[i] * e(pi[i], 8)
        if max(np.linalg.norm(M @ mul(e(i, 8), e(j, 8), 3) - mul(M[:, i], M[:, j], 3))
               for i in range(8) for j in range(8)) < TOL:
            return M, pi
    return None, pi


def all_collineation_autos():
    """All 168 GL(3,2)=PSL(2,7) Fano collineations as signed-perm octonion automorphisms."""
    import itertools as _it
    out = []
    for bv in _it.product([0, 1], repeat=9):
        M = np.array(bv).reshape(3, 3)
        if int(round(np.linalg.det(M))) % 2 != 1:
            continue
        pi = {}
        for i in range(1, 8):
            b = np.array([(i >> k) & 1 for k in range(3)])
            jj = M @ b % 2
            pi[i] = int(jj[0] | (jj[1] << 1) | (jj[2] << 2))
        for mask in range(128):
            s = [1] + [1 if (mask >> k) & 1 == 0 else -1 for k in range(7)]
            g = np.zeros((8, 8)); g[:, 0] = e(0, 8)
            for i in range(1, 8):
                g[:, i] = s[i] * e(pi[i], 8)
            if max(np.linalg.norm(g @ mul(e(i, 8), e(j, 8), 3) - mul(g[:, i], g[:, j], 3))
                   for i in range(8) for j in range(8)) < TOL:
                out.append((g, pi)); break
    return out


def main():
    print("=" * 70)
    print("FUNCTOR F — the Fano / PSL(2,7) thread (ord-1 phi <-> ord-2 ZD <-> ord-3)")
    print("=" * 70)
    core = audit_core()
    print(f"W0_CORE_AUDIT octonion+sedenion identity/sq/anticomm {'PASS' if core else 'FAIL'}")

    ZD = zd_list()
    # W1 — 42 ZD -> 7 fibres of 6
    byline = {}
    for (i, j) in ZD:
        byline.setdefault(zd_line(e(i, 16) + e(j, 16)), []).append((i, j))
    w1 = (len(ZD) == 42 and len(byline) == 7 and sorted(len(v) for v in byline.values()) == [6] * 7)
    print(f"W1_ZD_7FIBRES {len(ZD)} ZD -> {len(byline)} fibres, sizes {sorted(len(v) for v in byline.values())} "
          f"{'PASS' if w1 else 'FAIL'}")

    # W2 — fibre-line == complement of Fano line through (i,k); 7 fibre-lines == phi lines
    PL = phi_lines()
    from itertools import combinations
    line_of = {}
    for L in PL:
        for a, b in [(x, y) for x in L for y in L if x != y]:
            line_of[(a, b)] = L
    # fibre octonion-support = complement of the Fano line through (i,k);
    # equivalently zd_line (= range \ support) IS that Fano line L.
    comp_ok = True
    for (i, j) in ZD:
        lo = i if i < 8 else i - 8; hi = j if j < 8 else j - 8
        L = line_of.get((lo, hi))
        if L is None or zd_line(e(i, 16) + e(j, 16)) != L:
            comp_ok = False
    w2 = comp_ok and (set(byline.keys()) == PL)
    print(f"W2_FIBRE_EQ_PHI_LINE fibre-line == Fano line thru (i,k) (support=its complement) all 42: {comp_ok}; "
          f"7 fibre-lines == 7 phi lines: {set(byline.keys()) == PL} {'PASS' if w2 else 'FAIL'}")

    # W3 — PSL(2,7) collineation acts equivariantly on phi lines and ZD fibres.
    # (Computational evidence over the 42 ZD for one explicit order-3 generator, not a
    #  forall-PSL(2,7) structural proof; the diagonal lift is explicitly re-verified as a
    #  sedenion automorphism, since its signs are solved, not free.)
    g, pi = fano_collineation_auto()
    G = np.zeros((16, 16)); G[:8, :8] = g; G[8:, 8:] = g
    G_is_auto = max(np.linalg.norm(G @ mul(e(i, 16), e(j, 16), 4) - mul(G[:, i], G[:, j], 4))
                    for i in range(16) for j in range(16)) < 1e-9
    def line_img(L):
        return frozenset(pi[p] for p in L)
    equiv = all(zd_line(G @ (e(i, 16) + e(j, 16))) == line_img(zd_line(e(i, 16) + e(j, 16)))
                for (i, j) in ZD)
    lines_closed = (set(line_img(L) for L in PL) == PL)
    order3 = np.allclose(np.linalg.matrix_power(g, 3), np.eye(8))
    w3 = equiv and lines_closed and order3 and G_is_auto
    print(f"W3_PSL27_EQUIVARIANCE (g,g) in Aut(S)={G_is_auto}; Fano collineation (order-{3 if order3 else '?'}) "
          f"permutes phi lines (closed={lines_closed}) and ZD fibres by the SAME pi on all 42 ZD: {equiv} "
          f"{'PASS' if w3 else 'FAIL'}")

    # W4 — the ord-3 secondary operation sits on these fibres (parent rung): ker L_b uniform 4-dim
    dims = set(16 - np.linalg.matrix_rank(Lmat4(e(i, 16) + e(j, 16)), tol=TOL) for (i, j) in ZD)
    w4 = (dims == {4})
    print(f"W4_ORD3_ON_FIBRE the ord-3 secondary op lives on these fibres (ker L_b dim {sorted(dims)}, uniform) "
          f"=> ord-1 phi, ord-2 ZD, ord-3 secondary thread one PSL(2,7) {'PASS' if w4 else 'FAIL'}")

    # W5 — the FULL PSL(2,7): all 168 collineations act equivariantly (phi lines == ZD fibres)
    # and transitively on the 7 fibres. Upgrades W3 from one generator to the whole group.
    autos = all_collineation_autos()
    n_auto = 0; n_equiv = 0
    for (gg, ppi) in autos:
        G2 = np.zeros((16, 16)); G2[:8, :8] = gg; G2[8:, 8:] = gg
        if max(np.linalg.norm(G2 @ mul(e(i, 16), e(j, 16), 4) - mul(G2[:, i], G2[:, j], 4))
               for i in range(16) for j in range(16)) < 1e-9:
            n_auto += 1
        if all(zd_line(G2 @ (e(i, 16) + e(j, 16))) == frozenset(ppi[p] for p in zd_line(e(i, 16) + e(j, 16)))
               for (i, j) in ZD):
            n_equiv += 1
    L0 = sorted(PL, key=lambda x: sorted(x))[0]
    orbit = set(frozenset(ppi[p] for p in L0) for (gg, ppi) in autos)
    w5 = (len(autos) == 168 and n_auto == 168 and n_equiv == 168 and len(orbit) == 7)
    print(f"W5_FULL_PSL27_TRANSITIVE |group|={len(autos)}; (g,g) in Aut(S): {n_auto}/168; "
          f"phi-line-perm==ZD-fibre-perm: {n_equiv}/168; orbit on fibres={len(orbit)} (7=transitive) "
          f"{'PASS' if w5 else 'FAIL'}")

    # W6 — the psi (ord-2 co-associator 4-form) threads the SAME fibres, dually:
    # its 7 coassociative 4-planes = the ZD fibre octonion-supports = the Fano-line
    # complements; psi calibrates (+-1) each. So the fibre carries BOTH a phi line
    # (ord-1, its label) and a psi 4-plane (ord-2, its support) -- the G2 phi/psi duality.
    import itertools as _it2
    psi = np.zeros((8, 8, 8, 8))
    for a in range(1, 8):
        for b_ in range(1, 8):
            for c in range(1, 8):
                A = mul(mul(e(a, 8), e(b_, 8), 3), e(c, 8), 3) - mul(e(a, 8), mul(e(b_, 8), e(c, 8), 3), 3)
                for d in range(1, 8):
                    psi[a, b_, c, d] = -0.5 * float(np.dot(A, e(d, 8)))
    psi_supp = set()
    calib = set()
    for T in _it2.combinations(range(1, 8), 4):
        vals = [psi[p] for p in _it2.permutations(T)]
        if max(abs(v) for v in vals) > TOL:
            psi_supp.add(frozenset(T))
            calib.add(round(max(abs(v) for v in vals), 6))
    fibre_supps = set(frozenset(x % 8 for v in nullspace(Lmat4(e(i, 16) + e(j, 16)))
                                for x in np.nonzero(np.abs(v) > TOL)[0]) for (i, j) in ZD)
    fano_compl = set(frozenset(range(1, 8)) - L for L in PL)
    w6 = (len(psi_supp) == 7 and psi_supp == fano_compl and psi_supp == fibre_supps and calib == {1.0})
    print(f"W6_PSI_COASSOCIATIVE_FIBRE psi nonzero on {len(psi_supp)}/35 four-sets == Fano complements "
          f"({psi_supp == fano_compl}) == ZD fibre supports ({psi_supp == fibre_supps}); calibration |psi|={sorted(calib)} "
          f"{'PASS' if w6 else 'FAIL'}")

    print("=" * 70)
    if core and w1 and w2 and w3 and w4 and w5 and w6:
        print("FUNCTOR_F_FANO_VERDICT PSL27_THREADS_THE_TOWER")
        print("FUNCTOR_F_FANO_NOTE 42 sedenion ZD -> 7 Fano fibres (prior 168 structure); NEW: fibre-line "
              "== functor-F phi 3-form line (complement map, all 42); one explicit order-3 PSL(2,7) "
              "collineation (verified in Aut(S)) acts by the SAME pi on phi lines AND ZD fibres; so ord-1 "
              "phi, ord-2 ZD fibre, ord-3 secondary op SHARE one Fano indexing and one PSL(2,7) action "
              "(operational unification across layers, NOT a single object, NOT an identity; D3 respected); "
              "verified for the FULL 168-element PSL(2,7): all in Aut(S), all equivariant, transitive on "
              "the 7 fibres (W5); AND the psi 4-form (ord-2) threads the same fibres dually -- its 7 "
              "coassociative 4-planes ARE the ZD fibre supports (= Fano complements), calibrated +-1 (W6): "
              "the fibre carries a phi line (label) and a psi 4-plane (support), the G2 phi/psi duality")
        return 0
    print("FUNCTOR_F_FANO_VERDICT INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
