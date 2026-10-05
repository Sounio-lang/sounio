#!/usr/bin/env python3
"""
Functor F — anatomy of the ord-3 secondary operation (x,y)->(x.y).b, honest.

Continuing "find something genuinely new" at higher risk, this rung dissects the ord-3
secondary operation as an algebraic object on the sedenion zero-divisor fibres. What
survives scrutiny (advisor + §10 Grok review), cleanly separated into KNOWN / NEW / RETRACTED:

  T1  TWO-SIDED ANNIHILATION (KNOWN -- Moreno 1998, Kivunge 2004): for every one of the 42
      zero-divisors b=e_i+e_j, F(b)=ker L_b = ker R_b (dim 4); b annihilates its own fibre on
      BOTH sides (b.x = x.b = 0 for x in F(b)).
  T2  BRACKETING COLLAPSE (consequence of T1): of the 8 bracketings/orderings of the triple
      (x,y,b) with x,y in F(b), exactly 4 vanish identically -- precisely those that multiply b
      into x or y first: x(yb)=(xb)y=x(by)=(bx)y=0. Only the "multiply the pair x.y first"
      bracketings survive.
  T3  THE REVERSAL LAW (verified, then PROVED -- a 3-line corollary of standard structure, NOT a
      new identity): b.(x.y) = (y.x).b for all x,y in F(b), on ALL 42 ZD. Equivalently, since
      y.x = -x.y - 2<x,y>e0 for imaginary x,y, the anticommutator form {b, x.y} = -2<x,y> b.
      PROOF (T3b, checked elementwise): every Cayley-Dickson algebra is FLEXIBLE ([a,b,c]=-[c,b,a],
      classical; verified for the sedenions here), and using the two-sided annihilation (T1):
        (bx)y=0 => b(xy) = -[b,x,y];   y(xb)=0 => (yx)b = [y,x,b];   flex => [b,x,y] = -[y,x,b];
      hence b(xy) - (yx)b = -[b,x,y] - [y,x,b] = 0. So the reversal law is the FLEXIBILITY law's
      shadow on the annihilator fibres -- a folklore-adjacent corollary (§10 Grok initially called
      it "a new computed relation" thinking of conjugation/alternativity, missing flexibility; the
      3-line proof settles it). Both T1 and flexibility are documented (Moreno'98/Kivunge'04;
      flexibility classical, e.g. Schafer / nLab; sedenion ZD geometry arXiv:2411.18881, 2024).
  T4  OPERATION SPLIT: the ord-3 op (x.y).b = S + C, S=1/2((xy)b+(yx)b) symmetric part,
      C=1/2((xy)b-(yx)b)=1/2[x,y].b commutator part; both nonzero. (Honest sym/antisym split --
      NOT a claim that S,C are irreducible G-submodules; Grok flagged that would need a separate
      check.) This reconnects the ord-3 op to the associator/commutator theme the arc began with.

  RETRACTED here: an earlier probe claimed dim Hom_G(D,M)=6 ("a 6-dim space of G-equivariant
  ternary operations"). That domain D=+_b F(b)(x)F(b) tied the operation to specific b-VECTORS,
  whose G-orbit is 24 (four sign-variants per fibre) -- while the FIBRE orbit is a clean 6. So D
  as built was NOT a clean G-module and the count was ill-posed. §10 Grok flagged it [WRONG]; a
  direct orbit computation confirmed. Claim withdrawn (2nd retraction this session -- honest).

Verdict ORD3_TERNARY_ANATOMY. Numerical certificate (machine precision), D3 respected.
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


def m4(x, y):
    return mul(x, y, 4)


def assoc(a, b, c):
    return m4(m4(a, b), c) - m4(a, m4(b, c))   # [a,b,c]


def Lmat4(b):
    return np.column_stack([mul(b, e(k, 16), 4) for k in range(16)])


def Rmat4(b):
    return np.column_stack([mul(e(k, 16), b, 4) for k in range(16)])


def nullspace(M):
    _, s, vh = np.linalg.svd(M)
    return vh[np.sum(s > TOL):]


def sp(rows):
    _, s, vh = np.linalg.svd(np.array(rows))
    return vh[:int(np.sum(s > TOL))]


def audit_core():
    def chk(bits):
        n = 1 << bits
        return (all(np.allclose(mul(e(0, n), e(j, n), bits), e(j, n)) for j in range(n))
                and all(np.allclose(mul(e(i, n), e(i, n), bits), -e(0, n)) for i in range(1, n))
                and all(np.allclose(mul(e(i, n), e(j, n), bits), -mul(e(j, n), e(i, n), bits))
                        for i in range(1, n) for j in range(1, n) if i != j))
    return chk(3) and chk(4)


def main():
    print("=" * 70)
    print("FUNCTOR F — anatomy of the ord-3 secondary operation (x,y)->(x.y).b")
    print("=" * 70)
    print(f"T0_CORE_AUDIT {'PASS' if audit_core() else 'FAIL'}")

    ZD = [(i, j) for i in range(1, 16) for j in range(i + 1, 16)
          if 16 - np.linalg.matrix_rank(Lmat4(e(i, 16) + e(j, 16)), tol=TOL) > 0]

    # T1 — two-sided annihilation (KNOWN), all 42 ZD
    worst_ann = 0.0; kernels_equal = True
    fibres = {}
    for (i, j) in ZD:
        b = e(i, 16) + e(j, 16)
        KL = nullspace(Lmat4(b)); KR = nullspace(Rmat4(b))
        fibres[(i, j)] = KL
        if KL.shape[0] != 4 or KR.shape[0] != 4 or sp(np.vstack([KL, KR])).shape[0] != 4:
            kernels_equal = False
        for a in range(4):
            worst_ann = max(worst_ann, np.linalg.norm(m4(KL[a], b)), np.linalg.norm(m4(b, KL[a])))
    t1 = (len(ZD) == 42 and kernels_equal and worst_ann < 1e-9)
    print(f"T1_TWO_SIDED_ANNIHILATION (KNOWN: Moreno'98/Kivunge'04) {len(ZD)} ZD: F(b)=ker L_b=ker R_b "
          f"(dim 4), max||x.b||,||b.x||={worst_ann:.1e} {'PASS' if t1 else 'FAIL'}")

    # T2 — bracketing collapse: the 4 "b-first" bracketings vanish
    vanish = 0.0
    for (i, j) in ZD:
        b = e(i, 16) + e(j, 16); F = fibres[(i, j)]
        for a in range(4):
            for c in range(4):
                x, y = F[a], F[c]
                for val in (m4(x, m4(y, b)), m4(m4(x, b), y), m4(x, m4(b, y)), m4(m4(b, x), y)):
                    vanish = max(vanish, np.linalg.norm(val))
    t2 = vanish < 1e-9
    print(f"T2_BRACKETING_COLLAPSE the 4 b-first bracketings x(yb),(xb)y,x(by),(bx)y all vanish: "
          f"max||.||={vanish:.1e} (only 'pair-first' survives) {'PASS' if t2 else 'FAIL'}")

    # T3 — reversal law + anticommutator form, all 42 ZD
    worst_rev = 0.0; worst_anti = 0.0
    for (i, j) in ZD:
        b = e(i, 16) + e(j, 16); F = fibres[(i, j)]
        for a in range(4):
            for c in range(4):
                x, y = F[a], F[c]
                worst_rev = max(worst_rev, np.linalg.norm(m4(b, m4(x, y)) - m4(m4(y, x), b)))
                anti = m4(b, m4(x, y)) + m4(m4(x, y), b)   # {b, xy}
                worst_anti = max(worst_anti, np.linalg.norm(anti + 2 * (x @ y) * b))  # = -2<x,y> b
    t3 = worst_rev < 1e-9 and worst_anti < 1e-9
    print(f"T3_REVERSAL_LAW all 42 ZD: max||b(xy)-(yx)b||={worst_rev:.1e}; equivalently "
          f"{{b,xy}}=-2<x,y>b, max dev={worst_anti:.1e} {'PASS' if t3 else 'FAIL'}")

    # T3b — the 3-line PROOF: reversal law = flexibility's shadow on the annihilator fibres
    rng = np.random.default_rng(0)
    flex = max(np.linalg.norm(assoc(a := rng.standard_normal(16), bb := rng.standard_normal(16),
               cc := rng.standard_normal(16)) + assoc(cc, bb, a)) for _ in range(80))
    w_l = w_r = w_f = 0.0
    for (i, j) in ZD:
        b = e(i, 16) + e(j, 16); F = fibres[(i, j)]
        for a in range(4):
            for c in range(4):
                x, y = F[a], F[c]
                w_l = max(w_l, np.linalg.norm(m4(b, m4(x, y)) + assoc(b, x, y)))     # b(xy)=-[b,x,y]
                w_r = max(w_r, np.linalg.norm(m4(m4(y, x), b) - assoc(y, x, b)))     # (yx)b=[y,x,b]
                w_f = max(w_f, np.linalg.norm(assoc(b, x, y) + assoc(y, x, b)))       # flex
    t3b = flex < 1e-9 and w_l < 1e-9 and w_r < 1e-9 and w_f < 1e-9
    print(f"T3b_PROOF_VIA_FLEXIBILITY sedenions flexible (max||[a,b,c]+[c,b,a]||={flex:.1e}); "
          f"b(xy)=-[b,x,y] ({w_l:.0e}), (yx)b=[y,x,b] ({w_r:.0e}), [b,x,y]=-[y,x,b] ({w_f:.0e}) "
          f"=> reversal law is a 3-line corollary, NOT new {'PASS' if t3b else 'FAIL'}")

    # T4 — operation split S + C, both nonzero
    dom = [(bi, a, c) for bi in range(len(ZD)) for a in range(4) for c in range(4)]
    ZDl = list(ZD)
    def opm(fn):
        out = []
        for (bi, a, c) in dom:
            (i, j) = ZDl[bi]; b = e(i, 16) + e(j, 16); F = fibres[(i, j)]
            out.append(fn(F[a], F[c], b))
        return np.array(out).T
    V1 = opm(lambda x, y, b: m4(m4(x, y), b))
    S = opm(lambda x, y, b: 0.5 * (m4(m4(x, y), b) + m4(m4(y, x), b)))
    C = opm(lambda x, y, b: 0.5 * (m4(m4(x, y), b) - m4(m4(y, x), b)))
    split_ok = np.linalg.norm(V1 - (S + C)) < 1e-9
    t4 = split_ok and np.linalg.norm(S) > 1e-6 and np.linalg.norm(C) > 1e-6
    print(f"T4_OPERATION_SPLIT (xy)b=S+C (dev {np.linalg.norm(V1-(S+C)):.1e}); ||S||={np.linalg.norm(S):.2f} "
          f"(sym), ||C||={np.linalg.norm(C):.2f} (commutator 1/2[x,y]b), both nonzero {'PASS' if t4 else 'FAIL'}")

    print("=" * 70)
    if audit_core() and t1 and t2 and t3 and t3b and t4:
        print("FUNCTOR_F_ORD3TERN_VERDICT ORD3_TERNARY_ANATOMY")
        print("FUNCTOR_F_ORD3TERN_NOTE the ord-3 secondary op (x,y)->(x.y).b on sedenion ZD fibres is "
              "structurally DETERMINED BY KNOWN FACTS: b two-sided-annihilates its fibre (KNOWN, "
              "Moreno/Kivunge), collapsing 4 of 8 triple bracketings; the reversal identity b(xy)=(yx)b "
              "(all 42 ZD, equiv. {b,xy}=-2<x,y>b) is a 3-LINE COROLLARY of FLEXIBILITY (classical for all "
              "CD algebras) + that annihilation -- NOT a new identity (proof T3b: b(xy)=-[b,x,y], "
              "(yx)b=[y,x,b], flex [b,x,y]=-[y,x,b]); and the operation splits as symmetric + commutator "
              "(1/2[x,y]b), both nonzero, reconnecting to the associator theme. RETRACTED: an earlier "
              "'dim Hom_G(D,M)=6' claim (domain tied to b-vectors, orbit 24, not the clean 6-fibre orbit -> "
              "ill-posed; Grok[WRONG]+orbit check). HONEST OUTCOME: this vein harbours no genuinely-new deep "
              "invariant -- the ord-3 op is fixed by standard octonion/sedenion structure. Numerical "
              "certificate; D3 respected")
        return 0
    print("FUNCTOR_F_ORD3TERN_VERDICT INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
