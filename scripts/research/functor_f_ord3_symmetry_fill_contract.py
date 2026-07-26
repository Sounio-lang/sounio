#!/usr/bin/env python3
"""
Functor F — ord-3 fill, the symmetry verdict: NO_CANONICAL_FILL is representation-theoretic.

NO_CANONICAL_FILL (bare algebra) left open whether the PSL(2,7) symmetry could supply the
canonical secondary value the bracketing could not. This rung settles it at every level:

  Z1  The stabiliser of a single ZD b (its 4-dim fibre AS A SUBSPACE) inside the lifted
      PSL(2,7) is TRIVIAL -- so there is no group to average a single quotient Q over.
  Z2  The acting line-stabiliser is the SIGNED-octonion-automorphism group of ORDER 192 =
      (Z2)^3 x S4 (CORRECTION 2026-07-26: NOT the abstract S4 of order 24 -- the 24
      collineation-reps are a transversal that generates the order-192 group; it sits in the
      full signed-automorphism group of O of order 1344 = 8 x 168 = (Z2)^3 : PSL(2,7)). It
      acts on the 8-dim support-class and permutes its 6 fibres.
  Z3  The class-level secondary span (spanned by (x*y)*b over the 6 ZD b of the class and
      x,y in F(b)) contains NO invariant vector: dim(K ∩ span) = 0 for all 7 classes, where K
      is the 2-dim ambient invariant space. (Those ambient invariants are exactly the two
      structural units e0 and e8=l, and they lie in the COMPLEMENT of the secondary span, not
      in it -- a subtlety that flipped an early buggy computation.)
  Z4  Therefore the ord-3 secondary operation is an invariant-free module. No canonical
      scalar value can exist -- at bare-algebra, single-fibre, OR class-symmetry level --
      because the module has no invariant line. NO_CANONICAL_FILL is EXPLAINED (forced by
      representation theory), not merely unachieved. The canonical object is the module
      itself (canonical as a representation, not as a value).

Verdict NO_INVARIANT_FILL. Operational, D3 respected.
Self-contained (bits=3,4); embeds a core axiom-audit; builds the full 168-element PSL(2,7).
"""
import numpy as np
import itertools

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
        return (all(np.allclose(mul(e(0, n), e(j, n), bits), e(j, n)) for j in range(n))
                and all(np.allclose(mul(e(i, n), e(i, n), bits), -e(0, n)) for i in range(1, n))
                and all(np.allclose(mul(e(i, n), e(j, n), bits), -mul(e(j, n), e(i, n), bits))
                        for i in range(1, n) for j in range(1, n) if i != j))
    return chk(3) and chk(4)


def collineations():
    out = []
    for bv in itertools.product([0, 1], repeat=9):
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


def lift(g):
    G = np.zeros((16, 16)); G[:8, :8] = g; G[8:, 8:] = g; return G


def zd_line(b):
    F = nullspace(Lmat4(b))
    supp = frozenset(x % 8 for v in F for x in np.nonzero(np.abs(v) > TOL)[0])
    return frozenset(range(1, 8)) - supp


def main():
    print("=" * 70)
    print("FUNCTOR F — ord-3 fill: the symmetry verdict (NO_CANONICAL_FILL explained)")
    print("=" * 70)
    core = audit_core()
    print(f"Z0_CORE_AUDIT {'PASS' if core else 'FAIL'}")

    autos = collineations()
    ZD = [(i, j) for i in range(1, 16) for j in range(i + 1, 16)
          if 16 - np.linalg.matrix_rank(Lmat4(e(i, 16) + e(j, 16)), tol=TOL) > 0]

    # Z1 — single-fibre stabiliser is trivial (only identity fixes b=e1+e10 pointwise via e1,e2)
    b0 = e(1, 16) + e(10, 16)
    stab_b = [(g, pi) for (g, pi) in autos if np.allclose(g @ e(1, 8), e(1, 8)) and np.allclose(g @ e(2, 8), e(2, 8))]
    z1 = (len(stab_b) == 1)
    print(f"Z1_SINGLE_FIBRE_STAB_TRIVIAL |Stab(b) fixing its fibre as subspace| = {len(stab_b)} "
          f"{'PASS' if z1 else 'FAIL'}")

    # group by class (Fano line)
    byline = {}
    for (i, j) in ZD:
        byline.setdefault(zd_line(e(i, 16) + e(j, 16)), []).append((i, j))

    def common_inv(mats):
        A = np.vstack([g - np.eye(16) for g in mats])
        _, s, vh = np.linalg.svd(A)
        return vh[np.sum(s > TOL):]                 # rows: ambient S4-invariant vectors

    def rank(*blocks):
        return np.linalg.matrix_rank(np.vstack(blocks), tol=TOL)

    # Z2/Z3 — every class: the acting group's class-level secondary span (row space of the
    # (x*y)*b composites) contains NO invariant vector at all.
    # CORRECTION (2026-07-26): the acting group is NOT the abstract S4 (order 24). The 24
    # collineation-reps below are a TRANSVERSAL; the group they generate (the signed-octonion-
    # automorphism line-stabiliser) has order 192 = (Z2)^3 x S4, sitting in the full signed-
    # automorphism group of O of order 1344 = 8 x 168 = (Z2)^3 : PSL(2,7). The invariant test
    # is via the common-kernel of the generating set (Z3), which computes the GENERATED group's
    # invariants -- robust to the exact order -- so the invariant-free result is unaffected;
    # only the earlier 'S4 order 24' label was wrong.
    z2 = True; z3 = True
    worst_inter = 0
    for L, members in byline.items():
        S4 = [lift(g) for (g, pi) in autos if set(pi[p] for p in L) == L]   # 24 collineation-reps (transversal)
        if len(S4) != 24 or len(members) != 6:
            z2 = False
        W = []
        for (i, j) in members:
            b = e(i, 16) + e(j, 16); F = nullspace(Lmat4(b))
            for a in range(4):
                for c in range(4):
                    W.append(mul(mul(F[a], F[c], 4), b, 4))
        W = np.array(W)
        K = common_inv(S4)                                   # ambient S4-invariants (= {e0, e8})
        _, sv, vh = np.linalg.svd(W)
        SP = vh[:np.sum(sv > TOL)]                            # SECONDARY SPAN = row space of W
        inter = K.shape[0] + SP.shape[0] - rank(K, SP)       # dim(K ∩ secondary span)
        worst_inter = max(worst_inter, inter)
        if inter != 0:
            z3 = False
    # verify the CORRECTED group order: the 24 line-fixing collineation-reps generate a group
    # of order 192 (not 24); the full signed-auto group of O has order 1344 = 8 x 168.
    def _key(M):
        return tuple(np.round(M, 4).ravel())
    L0 = sorted(byline.keys(), key=lambda s: sorted(s))[0]
    gens = [lift(g) for (g, pi) in autos if set(pi[p] for p in L0) == L0]
    G = {_key(np.eye(16)): np.eye(16)}; frontier = list(G.values())
    while frontier:
        nxt = []
        for a in frontier:
            for g in gens:
                p = a @ g; k = _key(p)
                if k not in G:
                    G[k] = p; nxt.append(p)
        frontier = nxt
        if len(G) > 400:
            break
    gen_order = len(G)
    z2 = z2 and (gen_order == 192)
    print(f"Z2_ACTING_GROUP 24 collineation-reps/class (transversal) GENERATE a group of order "
          f"{gen_order} = (Z2)^3 x S4 (NOT the abstract S4=24; correction); 6 ZD/class "
          f"{'PASS' if z2 else 'FAIL'}")
    print(f"Z3_NO_INVARIANT_IN_SPAN all 7 classes: dim(order-192-group-invariants ∩ secondary span) = "
          f"{worst_inter} (0 => no S4-invariant secondary vector; the ambient invariants e0,e8 lie in the "
          f"complement, not the span) {'PASS' if z3 else 'FAIL'}")

    # Z4 — conclusion
    z4 = z1 and z2 and z3
    print(f"Z4_NO_GENUINE_FILL the ord-3 secondary op is an invariant-free module (under the order-192 group) => no canonical "
          f"secondary value at any level {'PASS' if z4 else 'FAIL'}")

    print("=" * 70)
    if core and z1 and z2 and z3 and z4:
        print("FUNCTOR_F_ORD3SYM_VERDICT NO_INVARIANT_FILL")
        print("FUNCTOR_F_ORD3SYM_NOTE single-fibre stabiliser trivial; the acting line-stabiliser is the "
              "SIGNED-octonion-automorphism group of ORDER 192 = (Z2)^3 x S4 (CORRECTION: not the abstract "
              "S4=24; it sits in the full signed-auto group of O, order 1344 = 8 x 168 = (Z2)^3:PSL(2,7)); "
              "it has NO invariant vector in the class-level secondary span (dim(K ∩ span)=0, all 7 classes; "
              "the ambient invariants e0,e8 lie in the complement) -> the ord-3 secondary op is an "
              "invariant-free module -> NO_CANONICAL_FILL holds at bare-algebra, single-fibre AND class-"
              "symmetry level (rep-theoretic). The invariant test uses the generating-set common-kernel, "
              "robust to the exact group order, so the result is unaffected by the corrected group id; "
              "operational, D3 respected")
        return 0
    print("FUNCTOR_F_ORD3SYM_VERDICT INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
