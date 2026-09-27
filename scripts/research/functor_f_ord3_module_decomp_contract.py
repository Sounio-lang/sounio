#!/usr/bin/env python3
"""
Functor F — what the ord-3 secondary module M actually is (HONEST, deflated version).

An earlier framing of this rung called M = 2*V3 "the exact representation-theoretic
fingerprint of the ord-3 secondary operation". On scrutiny (advisor + full anatomy) that
was an OVERCLAIM -- the same label-drift failure mode already self-caught twice in this
thread (associator-vs-phi at E6; S4-vs-192 group id). The honest anatomy:

  M1  The class-level ord-3 secondary module M = span{(x*y)*b : b in the 6 ZD of a Fano-line
      support-class L, x,y in the fibre F(b)=ker L_b} has dim 6 and is G-stable, G = the
      signed-octonion-automorphism line-stabiliser of order 192 = (Z2)^3 x S4 (inside the full
      signed-auto group 1344 = (Z2)^3 : PSL(2,7)).
  M2  NON-DEGENERACY (the one genuinely operation-dependent fact): each single b already gives
      a 4-dim image span{(x*y)*b}; the six images (pairwise intersection 2) collapse to fill
      EXACTLY the 6-dim COORDINATE SPACE of the class's indices, M = span{e_i, e_{i+8} : i in L}
      -- support exactly {L, L+8}. The images even reach the 6th dimension beyond the 5-dim
      span of the ZD elements themselves (M contains all 6 ZD b). So the operation is
      non-degenerate / surjective onto the class coordinate space.
  M3  As a G-module M = 2*V3: <chi,chi>=4, <chi,1>=0, End_G(M) non-abelian of dim 4 (computed
      in-harness as the commutant null-space, = M2(R)), a generic self-adjoint commutant
      element splits {3,3} over MULTIPLE seeds (rules out the quaternionic {6} alternative) ->
      V3 absolutely irreducible, multiplicity 2.
  M4  DEFLATION: the "2" is just the Cayley-Dickson lower/upper doubling (the lift is
      diag(g,g), so G acts IDENTICALLY on span{e1,e2,e3} and span{e9,e10,e11}); V3 is the
      octonion-automorphism action on a Fano line's 3 coordinates (absolutely irreducible).
      So 2*V3 is a fingerprint of the class COORDINATE structure (doubling x Fano-line action),
      NOT a fine invariant of the ternary operation's content. The operation contributes only
      M2 (it fills this space). Honest, modest, D3 respected.

Verdict ORD3_IMAGES_FILL_CLASS_COORD_SPACE (2xV3 = CD-doubling of the Fano-line octonion
action). Numerical certificate (machine precision), not symbolic.
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


def o(x, y):
    return mul(x, y, 3)


def nullspace(M):
    _, s, vh = np.linalg.svd(M)
    return vh[np.sum(s > TOL):]


def sp(rows):
    _, s, vh = np.linalg.svd(np.array(rows))
    return vh[:int(np.sum(s > TOL))]


def Lmat4(b):
    return np.column_stack([mul(b, e(k, 16), 4) for k in range(16)])


def collineation_reps():
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
            if max(np.linalg.norm(g @ o(e(i, 8), e(j, 8)) - o(g[:, i], g[:, j]))
                   for i in range(8) for j in range(8)) < TOL:
                out.append((g, pi)); break
    return out


def lift(g):
    G = np.zeros((16, 16)); G[:8, :8] = g; G[8:, 8:] = g; return G


def commutant_dim(rep):
    # dim {T : rho(g) T = T rho(g) for all g} = null-space of the stacked Kronecker blocks
    r = rep[0].shape[0]
    I = np.eye(r)
    A = np.vstack([np.kron(g, I) - np.kron(I, g.T) for g in rep])
    _, s, _ = np.linalg.svd(A)
    return r * r - int(np.sum(s > 1e-7))


def main():
    print("=" * 70)
    print("FUNCTOR F — what the ord-3 secondary module M actually is (honest anatomy)")
    print("=" * 70)
    autos = collineation_reps()

    gens = [lift(g) for (g, pi) in autos if set(pi[p] for p in {1, 2, 3}) == {1, 2, 3}]
    def key(M):
        return tuple(np.round(M, 4).ravel())
    G = {key(np.eye(16)): np.eye(16)}; fr = list(G.values())
    while fr:
        nx = []
        for a in fr:
            for g in gens:
                p = a @ g; k = key(p)
                if k not in G:
                    G[k] = p; nx.append(p)
        fr = nx
        if len(G) > 400:
            break
    Gp = list(G.values())

    ZDcls = [(1, 10), (1, 11), (2, 9), (2, 11), (3, 9), (3, 10)]
    Lline = [1, 2, 3]
    allW = []
    per_b = []
    for (i, j) in ZDcls:
        b = e(i, 16) + e(j, 16); F = nullspace(Lmat4(b))
        Wb = [mul(mul(F[a], F[c], 4), b, 4) for a in range(4) for c in range(4)]
        per_b.append(sp(Wb).shape[0])
        allW += Wb
    B = sp(allW).T; r = B.shape[1]
    stable = max(np.linalg.norm(g @ B - B @ (B.T @ g @ B)) for g in Gp)
    m1 = (len(Gp) == 192 and r == 6 and stable < 1e-9)
    print(f"M1_MODULE |G|={len(Gp)} (=192); dim M={r}; G-stable (dev {stable:.1e}) "
          f"{'PASS' if m1 else 'FAIL'}")

    # M2 — non-degeneracy: M equals the class coordinate space span{e_i,e_{i+8}:i in L}
    idx = Lline + [i + 8 for i in Lline]
    Bc = np.array([e(i, 16) for i in idx]).T
    Mrows = B.T
    is_coord = (sp(np.vstack([Mrows, Bc.T])).shape[0] == 6)
    contains_all_zd = all(
        np.linalg.norm((v := (e(i, 16) + e(j, 16)) / np.sqrt(2)) - Mrows.T @ (Mrows @ v)) < 1e-7
        for (i, j) in ZDcls)
    m2 = (per_b == [4] * 6 and is_coord and contains_all_zd)
    print(f"M2_NONDEGENERACY per-b image dims={per_b} (each 4); the six fill EXACTLY the class "
          f"coordinate 6-space span{{e_i,e_{{i+8}}:i in L}}={is_coord}; contains all 6 ZD b={contains_all_zd} "
          f"{'PASS' if m2 else 'FAIL'}")

    # M3 — 2*V3 signature, with dim End computed in-harness and multi-seed genericity
    rep = [B.T @ g @ B for g in Gp]
    chi = np.array([np.trace(R).real for R in rep])
    inner = np.mean(chi ** 2); triv = np.mean(chi)
    dEnd = commutant_dim(rep)
    rng = np.random.default_rng(0)
    def sym(X):
        return sum(R @ X @ R.T for R in rep) / len(rep)
    splits = []
    noncomm = 0.0
    for seed in range(4):
        T1 = sym(rng.standard_normal((r, r))); T2 = sym(rng.standard_normal((r, r)))
        noncomm = max(noncomm, np.linalg.norm(T1 @ T2 - T2 @ T1))
        ev = np.sort(np.linalg.eigvalsh(T1 + T1.T))
        gaps = np.diff(ev)
        cut = np.where(gaps > 0.05 * (ev[-1] - ev[0] + 1e-12))[0]
        blocks = np.diff([-1, *cut.tolist(), r - 1])   # relative-gap block sizes
        splits.append(sorted(blocks.tolist()))
    all_33 = all(s == [3, 3] for s in splits)
    m3 = (abs(inner - 4) < 1e-6 and abs(triv) < 1e-6 and dEnd == 4 and noncomm > 1e-6 and all_33)
    print(f"M3_2xV3 <chi,chi>={inner:.3f}, <chi,1>={triv:.3f}, dim End_G(M)={dEnd} (in-harness), "
          f"non-abelian (||[T1,T2]||={noncomm:.2f}), {{3,3}} split over 4 seeds={all_33} "
          f"{'PASS' if m3 else 'FAIL'}")

    # M4 — deflation: the 2 is the CD doubling; V3 = octonion action on a Fano line's 3 coords
    Blo = np.array([e(i, 16) for i in Lline]).T
    Bhi = np.array([e(i + 8, 16) for i in Lline]).T
    rep_lo = [Blo.T @ g @ Blo for g in Gp]
    rep_hi = [Bhi.T @ g @ Bhi for g in Gp]
    identical = max(np.linalg.norm(a - b) for a, b in zip(rep_lo, rep_hi))
    chi_lo = np.array([np.trace(R).real for R in rep_lo])
    v3_irr = abs(np.mean(chi_lo ** 2) - 1.0) < 1e-6
    m4 = (identical < 1e-9 and v3_irr)
    print(f"M4_DEFLATION_CD_DOUBLING V3=G|span{{e1,e2,e3}} absolutely irreducible ({v3_irr}); "
          f"upper half span{{e9,e10,e11}} identical (dev {identical:.1e}) => M=V3⊕V3, the '2' is the "
          f"Cayley-Dickson doubling, NOT a fine operation-invariant {'PASS' if m4 else 'FAIL'}")

    print("=" * 70)
    if m1 and m2 and m3 and m4:
        print("FUNCTOR_F_ORD3MOD_VERDICT ORD3_IMAGES_FILL_CLASS_COORD_SPACE")
        print("FUNCTOR_F_ORD3MOD_NOTE HONEST/deflated: the ord-3 secondary images over a Fano-line class "
              "are NON-DEGENERATE -- each b gives a 4-dim image, and the six fill EXACTLY the class's 6-dim "
              "coordinate space span{e_i,e_{i+8}:i in L} (reaching beyond the 5-dim span of the ZD elements). "
              "As a G-module (order 192) that space is 2*V3, but the '2' is merely the Cayley-Dickson "
              "lower/upper doubling (lift=diag(g,g), identical action) and V3 is the octonion-automorphism "
              "action on a Fano line's 3 coordinates (absolutely irreducible). So 2*V3 fingerprints the "
              "class COORDINATE structure, not the ternary operation's content; the only operation-dependent "
              "fact is the non-degeneracy (M2). Earlier '2*V3 = fingerprint of the ord-3 operation' was an "
              "OVERCLAIM, corrected here (label-drift, same mode as the E6/S4 self-catches). Numerical "
              "certificate; D3 respected")
        return 0
    print("FUNCTOR_F_ORD3MOD_VERDICT INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
