#!/usr/bin/env python3
"""
Functor F — ord-3 cross-column: the secondary ternary operation and where it lives.

The programme's ord-3 row is 'Massey / Borromean'. A pre-existing EMPIRICAL result
(docs/gpu/BORROMEAN_AINFINITY.md) found the octonion associator is NOT the path Massey
product (chance accuracy). This rung gives the ALGEBRAIC complement and locates where a
genuine secondary ternary structure can live.

Object named for what it is (NOT 'Massey' — a plain algebra has no differential): a
SECONDARY TERNARY OPERATION on the two-sided annihilator of a zero-divisor, defined on
the slice where the PRIMARY associator vanishes identically (products a*b=0 AND b*c=0).
What transfers from Massey theory is only the DEFINEDNESS PATTERN (secondary lives where
primary vanishes), not the construction.

Findings (probe-first; indeterminacy measured WITH a generic baseline; verdict named in
advance — operational, D3 respected):
  T1  In O (division algebra) there are NO zero-divisors, so the slice a*b=0 is EMPTY:
      the secondary-ternary domain does not exist. This is the algebraic reason the
      octonion associator (a total primary operation) cannot be a secondary/Massey object.
  T2  In S the slice is NONEMPTY and structured: 42 of 105 (e_i+e_j) are zero-divisors,
      and for each, ker L_b == ker R_b == a 4-dim FIBRE (uniform) -- the ord-2 ZD fibre
      (ties to the merged seam_coincidence / lo(+)hi structure).
  T3  On the slice the PRIMARY associator vanishes: [a,b,c]=(a*b)*c-a*(b*c)=0-0=0.
      So any ord-3 signal there is genuinely secondary, not the associator.
  T4  The secondary structure is DISTINGUISHABLE, not indeterminacy-swamped:
      dim(a*S + S*c) = 14 uniformly on the ZD slice vs 16 for generic pairs, so the
      quotient where a secondary invariant would live is 2-dim (ZD) vs 0-dim (generic).
  T5  But it is NOT a canonical invariant and NOT Borromean: within a fibre no element
      annihilates another (Borromean variety dim 0 -> no all-pairwise-0 triple), the
      triple is consecutive-only (a*b=0,b*c=0, a*c!=0), and S has no differential to fix
      the secondary value in that 2-dim quotient.

Verdict SECONDARY_TERNARY_LOCATED: ord-3 empty in O, located on the ord-2 sedenion ZD
fibre in S with a nontrivial 2-dim quotient, but with a stated definability obstruction
(no canonical value; no Borromean). Operational, never an identity (D3 respected).

Self-contained: Cayley-Dickson to bits=4 (sedenions); embeds a core axiom-audit.
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


def basis(n, i):
    v = np.zeros(n); v[i] = 1.0; return v


def Lmat(b, bits):
    n = 1 << bits
    return np.column_stack([mul(b, basis(n, k), bits) for k in range(n)])


def Rmat(b, bits):
    n = 1 << bits
    return np.column_stack([mul(basis(n, k), b, bits) for k in range(n)])


def nullspace(M):
    _, s, vh = np.linalg.svd(M)
    return vh[np.sum(s > TOL):]


def audit_core(bits):
    n = 1 << bits
    ident = all(np.allclose(mul(basis(n, 0), basis(n, j), bits), basis(n, j)) for j in range(n))
    sq = all(np.allclose(mul(basis(n, i), basis(n, i), bits), -basis(n, 0)) for i in range(1, n))
    anti = all(np.allclose(mul(basis(n, i), basis(n, j), bits), -mul(basis(n, j), basis(n, i), bits))
               for i in range(1, n) for j in range(1, n) if i != j)
    return ident, sq, anti


def zd_pairs(bits):
    n = 1 << bits
    out = []
    for i in range(1, n):
        for j in range(i + 1, n):
            b = basis(n, i) + basis(n, j)
            if n - np.linalg.matrix_rank(Lmat(b, bits), tol=TOL) > 0:
                out.append((i, j))
    return out


def main():
    print("=" * 70)
    print("FUNCTOR F — ord-3 secondary ternary operation (algebra <-> Massey/Borromean)")
    print("=" * 70)

    id8, sq8, an8 = audit_core(3)
    id16, sq16, an16 = audit_core(4)
    core = id8 and sq8 and an8 and id16 and sq16 and an16
    print(f"T0_CORE_AUDIT O(bits3): id={id8} sq=-1={sq8} anti={an8} | S(bits4): id={id16} sq=-1={sq16} "
          f"anti={an16} {'PASS' if core else 'FAIL'}")

    # T1 — octonion domain empty (division algebra)
    maxk8 = max(8 - np.linalg.matrix_rank(Lmat(basis(8, i) + basis(8, j), 3), tol=TOL)
                for i in range(1, 8) for j in range(i, 8))
    t1 = (maxk8 == 0)
    print(f"T1_OCTONION_DOMAIN_EMPTY max dim ker L_b = {maxk8} (0 => no ZD, secondary domain empty) "
          f"{'PASS' if t1 else 'FAIL'}")

    # T2 — sedenion slice nonempty, ker L_b == ker R_b == 4-dim fibre, uniform
    ZD = zd_pairs(4)
    kL = set(); kR = set(); same = True
    for (i, j) in ZD:
        b = basis(16, i) + basis(16, j)
        dL = 16 - np.linalg.matrix_rank(Lmat(b, 4), tol=TOL)
        dR = 16 - np.linalg.matrix_rank(Rmat(b, 4), tol=TOL)
        kL.add(dL); kR.add(dR)
        # subspace equality of ker L_b and ker R_b
        A = nullspace(Lmat(b, 4)); B = nullspace(Rmat(b, 4))
        if np.linalg.matrix_rank(np.vstack([A, B]), tol=TOL) != A.shape[0]:
            same = False
    t2 = (len(ZD) == 42 and kL == {4} and kR == {4} and same)
    print(f"T2_SEDENION_DOMAIN_ON_ZD_FIBRE #ZD={len(ZD)} dimkerL={sorted(kL)} dimkerR={sorted(kR)} "
          f"kerL==kerR(all)={same} {'PASS' if t2 else 'FAIL'}")

    # T3 — primary associator vanishes on the slice
    b = basis(16, ZD[0][0]) + basis(16, ZD[0][1])
    fib = nullspace(Lmat(b, 4))
    rng = np.random.default_rng(5)
    worst_assoc = 0.0
    for _ in range(30):
        a = rng.standard_normal(4) @ fib; c = rng.standard_normal(4) @ fib
        A = mul(mul(a, b, 4), c, 4) - mul(a, mul(b, c, 4), 4)
        worst_assoc = max(worst_assoc, float(np.linalg.norm(A)))
    t3 = worst_assoc < TOL
    print(f"T3_PRIMARY_BLIND associator [a,b,c] on slice: worst||.||={worst_assoc:.1e} (0 => primary blind) "
          f"{'PASS' if t3 else 'FAIL'}")

    # T4 — indeterminacy 14 on ZD slice vs 16 generic => 2-dim quotient
    def ideal_rank(x, y):
        return np.linalg.matrix_rank(np.hstack([Lmat(x, 4), Rmat(y, 4)]), tol=TOL)
    zd_ranks = set()
    for (i, j) in ZD[:12]:
        bb = basis(16, i) + basis(16, j)
        kr = nullspace(Rmat(bb, 4)); kl = nullspace(Lmat(bb, 4))
        for _ in range(6):
            a = rng.standard_normal(4) @ kr; c = rng.standard_normal(4) @ kl
            zd_ranks.add(ideal_rank(a, c))
    gen_ranks = set(ideal_rank(rng.standard_normal(16), rng.standard_normal(16)) for _ in range(30))
    t4 = (zd_ranks == {14} and gen_ranks == {16})
    print(f"T4_SECONDARY_DISTINGUISHABLE dim(a*S+S*c): ZD-slice={sorted(zd_ranks)} generic={sorted(gen_ranks)} "
          f"=> ZD quotient=2-dim vs generic 0-dim {'PASS' if t4 else 'FAIL'}")

    # T5 — no Borromean within a fibre (variety dim 0); consecutive-only + no differential
    v0 = fib[0]
    Mv = np.column_stack([mul(v0, fib[k], 4) for k in range(4)])
    borromean_dim = 4 - np.linalg.matrix_rank(Mv, tol=TOL)
    a = nullspace(Rmat(b, 4))[0]; c = nullspace(Lmat(b, 4))[1]
    ac = float(np.linalg.norm(mul(a, c, 4)))
    t5 = (borromean_dim == 0 and ac > TOL)
    print(f"T5_NO_BORROMEAN in-fibre annihilator-of-v0 dim={borromean_dim} (0 => no all-pairwise-0 triple); "
          f"consecutive-only ||a*c||={ac:.3f}!=0 {'PASS' if t5 else 'FAIL'}")

    print("=" * 70)
    if core and t1 and t2 and t3 and t4 and t5:
        print("FUNCTOR_F_ORD3_VERDICT SECONDARY_TERNARY_LOCATED")
        print("FUNCTOR_F_ORD3_NOTE ord3 secondary-ternary domain empty in O(division algebra); in S it "
              "IS the ord-2 ZD fibre (kerL=kerR=4, 42 ZD); primary associator blind on slice; "
              "indeterminacy 14 vs generic 16 => 2-dim quotient (distinguishable); NO Borromean, no "
              "differential => no canonical value; operational_not_identity; D3_respected; algebraic "
              "complement to the empirical BORROMEAN_AINFINITY negative (not the 48.9% number)")
        return 0
    print("FUNCTOR_F_ORD3_VERDICT INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
