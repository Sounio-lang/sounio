#!/usr/bin/env python3
"""
Functor F — ord-3 follow-up: can the 2-dim quotient be CANONICALLY filled?

The ord-3 rung (SECONDARY_TERNARY_LOCATED) found a nontrivial 2-dim quotient
Q = S/(a*S + S*c) on the sedenion ZD fibre where a secondary ternary invariant could
live. This rung tests whether the sedenion algebra's OWN structures fill Q with a
CANONICAL nonzero value, or whether any fill requires a CHOSEN (auxiliary) datum.

Method (advisor): a value is canonical iff it does NOT move as its auxiliary datum
ranges. Three pre-named outcomes decided before computing: constant-nonzero (canonical
invariant -> genuine positive), constant-zero (algebra lands at origin), datum-varying
(no canonical fill). The verdict follows the sweep.

Findings:
  U1  Q is 2-dimensional (re-confirmed).
  U2  The consecutive products a*c and c*a fall ENTIRELY into the indeterminacy
      a*S + S*c: Q-image 0. The naive fill vanishes.
  U3  Q IS reachable: the intrinsic ternary composites (a*c)*b, b*(a*c), (c*a)*b, b*(c*a)
      have nonzero Q-image -- so this is NOT 'algebra lands at origin'.
  U4  But NO canonical value: those four composites point in DIFFERENT Q-directions and
      span rank 2 (the full quotient), robustly across all ZD. The auxiliary datum is the
      BRACKETING/ORDERING of the ternary composite -- forced to matter by non-associativity
      -- and it sweeps all of Q. So the fill is SELECTED, not forced.
  U5  S has no differential: a*b=0 is a strict equation, not d(u)=a*b, so the classical
      Massey construction (which would pick a distinguished representative mod
      indeterminacy) does not run. The non-canonicity is structural.

Verdict NO_CANONICAL_FILL: the 2-dim quotient is real and reachable, but the sedenion
algebra supplies no canonical (bracketing-independent) nonzero secondary invariant;
filling it requires imposed A_infinity structure. SHARPENS SECONDARY_TERNARY_LOCATED
(does not overturn it). Operational, D3 respected.

Self-contained (Cayley-Dickson bits=4); embeds a core axiom-audit.
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


def mul(A, B, bits=4):
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


def basis(i, n=16):
    v = np.zeros(n); v[i] = 1.0; return v


def Lmat(b, bits=4):
    n = 1 << bits
    return np.column_stack([mul(b, basis(k, n), bits) for k in range(n)])


def Rmat(b, bits=4):
    n = 1 << bits
    return np.column_stack([mul(basis(k, n), b, bits) for k in range(n)])


def nullspace(M):
    _, s, vh = np.linalg.svd(M)
    return vh[np.sum(s > TOL):]


def Qcomp(a, c):
    Ideal = np.hstack([Lmat(a), Rmat(c)])
    U, S2, _ = np.linalg.svd(Ideal @ Ideal.T)
    return U[:, np.sum(S2 > TOL):]


def audit_core():
    n = 16
    ident = all(np.allclose(mul(basis(0), basis(j)), basis(j)) for j in range(n))
    sq = all(np.allclose(mul(basis(i), basis(i)), -basis(0)) for i in range(1, n))
    anti = all(np.allclose(mul(basis(i), basis(j)), -mul(basis(j), basis(i)))
               for i in range(1, n) for j in range(1, n) if i != j)
    return ident and sq and anti


def zd_pairs():
    out = []
    for i in range(1, 16):
        for j in range(i + 1, 16):
            if 16 - np.linalg.matrix_rank(Lmat(basis(i) + basis(j)), tol=TOL) > 0:
                out.append((i, j))
    return out


def main():
    print("=" * 70)
    print("FUNCTOR F — ord-3 follow-up: can the 2-dim quotient be canonically filled?")
    print("=" * 70)
    core = audit_core()
    print(f"U0_CORE_AUDIT sedenion identity/sq/anticomm {'PASS' if core else 'FAIL'}")

    rng = np.random.default_rng(11)
    ZD = zd_pairs()
    b = basis(1) + basis(10)
    kerR = nullspace(Rmat(b)); kerL = nullspace(Lmat(b))
    a0 = rng.standard_normal(4) @ kerR; c0 = rng.standard_normal(4) @ kerL

    # U1 — Q is 2-dim
    u1 = (Qcomp(a0, c0).shape[1] == 2)
    print(f"U1_QUOTIENT_2DIM dim Q = {Qcomp(a0, c0).shape[1]} {'PASS' if u1 else 'FAIL'}")

    # U2 — consecutive products vanish in Q
    worst_ac = 0.0
    for _ in range(12):
        a = rng.standard_normal(4) @ kerR; c = rng.standard_normal(4) @ kerL
        comp = Qcomp(a, c)
        for prod in (mul(a, c), mul(c, a)):
            worst_ac = max(worst_ac, np.linalg.norm(comp.T @ prod) / (np.linalg.norm(prod) + 1e-12))
    u2 = worst_ac < 1e-6
    print(f"U2_CONSECUTIVE_VANISH ||projQ(a*c,c*a)||/||.|| worst={worst_ac:.1e} (0 => naive fill vanishes) "
          f"{'PASS' if u2 else 'FAIL'}")

    # U3 — Q IS reachable by intrinsic ternary composites (not 'lands at origin')
    a = rng.standard_normal(4) @ kerR; c = rng.standard_normal(4) @ kerL
    comp = Qcomp(a, c)
    comps = [mul(mul(a, c), b), mul(b, mul(a, c)), mul(mul(c, a), b), mul(b, mul(c, a))]
    reach = min(np.linalg.norm(comp.T @ x) for x in comps)
    u3 = reach > 1e-6
    print(f"U3_INTRINSIC_REACHES_Q min ||projQ(ternary composites)|| = {reach:.3f} (>0 => Q reachable) "
          f"{'PASS' if u3 else 'FAIL'}")

    # U4 — but bracketing/ordering is the datum: composites span the FULL quotient, robustly
    ranks = set()
    for (i, j) in ZD:
        bb = basis(i) + basis(j); kr = nullspace(Rmat(bb)); kl = nullspace(Lmat(bb))
        aa = rng.standard_normal(4) @ kr; cc = rng.standard_normal(4) @ kl; cp = Qcomp(aa, cc)
        vs = [cp.T @ mul(mul(aa, cc), bb), cp.T @ mul(bb, mul(aa, cc)),
              cp.T @ mul(mul(cc, aa), bb), cp.T @ mul(bb, mul(cc, aa))]
        ranks.add(int(np.linalg.matrix_rank(np.array(vs), tol=1e-6)))
    u4 = (ranks == {2})
    print(f"U4_BRACKETING_IS_DATUM ternary-composite Q-span rank over all {len(ZD)} ZD = {sorted(ranks)} "
          f"(all 2 => bracketing choice sweeps full quotient => NO canonical value) {'PASS' if u4 else 'FAIL'}")

    # U5 — no differential (strict zeros) => classical Massey cannot pick a distinguished rep
    ab = np.linalg.norm(mul(a, b)); bc = np.linalg.norm(mul(b, c))
    u5 = ab < TOL and bc < TOL
    print(f"U5_NO_DIFFERENTIAL a*b={ab:.1e}, b*c={bc:.1e} strict zeros (not d(u)) => Massey construction "
          f"cannot select a rep {'PASS' if u5 else 'FAIL'}")

    print("=" * 70)
    if core and u1 and u2 and u3 and u4 and u5:
        print("FUNCTOR_F_ORD3FILL_VERDICT NO_CANONICAL_FILL")
        print("FUNCTOR_F_ORD3FILL_NOTE Q 2-dim real & reachable; consecutive a*c,c*a -> 0 in Q; intrinsic "
              "ternary composites reach Q but the 4 bracketings/orderings span the FULL quotient (rank 2, "
              "robust over all 42 ZD) => value is bracketing-selected not forced; S has no differential "
              "(a*b strict 0) => Massey cannot pick a rep; SHARPENS SECONDARY_TERNARY_LOCATED (fill needs "
              "imposed A-infinity structure); operational_not_identity; D3_respected")
        return 0
    print("FUNCTOR_F_ORD3FILL_VERDICT INCOMPLETE")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
