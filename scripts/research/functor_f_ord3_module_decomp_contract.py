#!/usr/bin/env python3
"""
Functor F — the exact representation of the ord-3 secondary operation: M = 2 * V3.

A new computation about a BESPOKE object: the ord-3 secondary ternary operation on the
sedenion zero-divisor fibres, x,y |-> (x*y)*b, is this programme's own construction. The
group acting on it (order 192 / 1344, PSL(2,7)) and its irreps are entirely standard; what
is new is only that THIS concrete 6-dim module realises the specific decomposition below.
The certificate is NUMERICAL (double precision, machine-epsilon tolerances), not symbolic.
Reviewed §10 (Grok [OK] on the decomposition logic, the M2(R)-vs-H distinction, and the
character arithmetic; framing tightened to 'this module realises 2*V3', numerical-certificate
caveat noted). The result, exact:

  M1  The class-level ord-3 secondary module M (spanned by (x*y)*b over the 6 ZD b of a
      Fano-line support-class and x,y in the fibre F(b)) is 6-dimensional and is a genuine
      module for the acting symmetry -- the signed-octonion-automorphism line-stabiliser of
      order 192 = (Z2)^3 x S4 (inside the full signed-auto group 1344 = (Z2)^3:PSL(2,7)).
  M2  Character invariants: <chi_M, chi_M> = 4 (sum of squared multiplicities) and
      <chi_M, trivial> = 0 (NO invariant vector -- this is exactly why NO_CANONICAL_FILL).
  M3  The commutant End_G(M) is NON-abelian of dimension 4, and a generic self-adjoint
      commutant element has eigenvalue multiplicities {3, 3}. Hence M is NOT multiplicity-
      free: it is 2 copies of ONE 3-dimensional irreducible V3 of the order-192 group.
  M4  V3 is a NON-trivial irrep, so M has no trivial summand -> no invariant -> no
      canonical secondary value. This is the deepest form of NO_INVARIANT_FILL: the ord-3
      secondary operation is exactly the isotypic block 2*V3, invariant-free by construction.

Verdict ORD3_MODULE_IS_2xV3. Genuinely new (the object is this programme's construction);
verified numerically. No semantic claim (D3-quarantined).
"""
import numpy as np
import itertools

np.seterr(all='ignore')
TOL = 1e-9


def cds(a, b, bits=3):
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


def main():
    print("=" * 70)
    print("FUNCTOR F — the exact representation of the ord-3 secondary operation: M = 2*V3")
    print("=" * 70)
    autos = collineation_reps()

    # build the order-192 acting group (close the line-fixing transversal)
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

    # the ord-3 secondary module M
    ZDcls = [(1, 10), (1, 11), (2, 9), (2, 11), (3, 9), (3, 10)]
    W = []
    for (i, j) in ZDcls:
        b = e(i, 16) + e(j, 16); F = nullspace(Lmat4(b))
        for a in range(4):
            for c in range(4):
                W.append(mul(mul(F[a], F[c], 4), b, 4))
    _, sv, vh = np.linalg.svd(np.array(W)); r = int(np.sum(sv > TOL)); B = vh[:r].T
    rep = [B.T @ g @ B for g in Gp]
    stable = max(np.linalg.norm(g @ B - B @ (B.T @ g @ B)) for g in Gp)

    m1 = (len(Gp) == 192 and r == 6 and stable < 1e-9)
    print(f"M1_MODULE |G|={len(Gp)} (=192=(Z2)^3 x S4), dim M={r}, G-stable(dev {stable:.1e}) "
          f"{'PASS' if m1 else 'FAIL'}")

    chi = np.array([np.trace(R).real for R in rep])
    inner = np.mean(chi ** 2); triv = np.mean(chi)
    m2 = (abs(inner - 4) < 1e-6 and abs(triv) < 1e-6)
    print(f"M2_CHARACTER <chi,chi>={inner:.3f} (sum of squared mults), <chi,trivial>={triv:.3f} "
          f"(NO invariant) {'PASS' if m2 else 'FAIL'}")

    rng = np.random.default_rng(1)
    def sym(X):
        return sum(R @ X @ R.T for R in rep) / len(rep)
    T1 = sym(rng.standard_normal((r, r))); T2 = sym(rng.standard_normal((r, r)))
    noncomm = np.linalg.norm(T1 @ T2 - T2 @ T1)
    evs = np.round(np.linalg.eigvalsh(T1 + T1.T), 4)
    from collections import Counter
    mult = sorted(Counter(evs).values())
    m3 = (noncomm > 1e-6 and mult == [3, 3])
    print(f"M3_COMMUTANT End_G(M) non-abelian (||[T1,T2]||={noncomm:.2f}), generic eigenvalue mults "
          f"{dict(Counter(evs))} = {{3,3}} => M = 2 x V3 (mult-2 of a 3-dim irrep) {'PASS' if m3 else 'FAIL'}")

    m4 = m2 and m3  # no trivial + 2xV3 => invariant-free, explains NO_CANONICAL_FILL
    print(f"M4_V3_NONTRIVIAL M = 2*V3 with V3 a NON-trivial 3-dim irrep => no trivial summand => "
          f"no invariant => this is the exact form of NO_CANONICAL_FILL {'PASS' if m4 else 'FAIL'}")

    print("=" * 70)
    if m1 and m2 and m3 and m4:
        print("FUNCTOR_F_ORD3MOD_VERDICT ORD3_MODULE_IS_2xV3")
        print("FUNCTOR_F_ORD3MOD_NOTE the ord-3 secondary operation on the sedenion ZD fibres, as a module "
              "for its symmetry group 2^3:S4 (order 192), is EXACTLY 2*V3 -- two copies of a single 3-dim "
              "irreducible, with no trivial part (<chi,chi>=4, <chi,triv>=0, non-abelian commutant dim 4, "
              "eigenmults {3,3}). Genuinely new (the ord-3 operation is this programme's construction). "
              "V3 nontrivial => invariant-free => the exact representation-theoretic form of "
              "NO_CANONICAL_FILL. No semantic claim (D3-quarantined)")
        return 0
    print("FUNCTOR_F_ORD3MOD_VERDICT INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
