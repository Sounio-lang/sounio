r"""
EXPLICIT ISOMORPHISM for the fiber-geometry PARITY COLLAPSE LAW (constructive; VERIFIED n=6,7,8).

Context.  The orbit theorem (cd_tower_auto_action_on_zd_fibers.py) is PROVEN forall n and untouched.
A SECONDARY claim "distinct orbits => distinct fiber geometries" was FALSE and was retracted
(cd_tower_fiber_geometry_collision.py): an adversarial nauty audit found Fano orbits whose annihilation
graph is ISOMORPHIC to a fixed-seam orbit's.  The audit's replacement was the PARITY COLLAPSE LAW
(nauty-complete n<=8): gamma(Seam(Y)) = gamma(Fano(Y & (Y-1))) exactly when wt(Y) is even (Y != 0),
where Y ranges over the seam-tower coordinate (bits {3..n-2}).

This file upgrades that from a nauty black box to an EXPLICIT, VERIFIED graph isomorphism.  For every
even-weight seam Y it constructs a concrete bijection and checks (via the closed-form adjacency rule)
that it is an isomorphism of the two annihilation graphs.

THE MAP.  A fiber's vertices are the mixed-half ZD primitives e_lo + s.e_hi (hi = lo XOR L, s = +-1),
so a vertex is a pair (lo, s), lo in [1, H=2^{n-1}), s in {+-1}.  The collapse isomorphism is
        Phi(lo, s) = ( tau(lo),  lambda(lo) * s ),      tau = swap(bit 0, bit lsb(Y)),
where lsb(Y) is the position of the lowest set bit of Y (a seam-tower bit >= 3), and lambda: [1,H)->{+-1}
is a SWITCHING function (a per-vertex sign gauge).  tau trades the octonion bit 0 for the lowest active
seam bit -- exactly sending the seam label Y (x=0) to the Fano representative label (Y & (Y-1)) | 1 (x=1).

CLOSED-FORM ADJACENCY (proved from the sign cocycle; verified 0 mismatches vs the algebra product).
Two primitives (lo,s),(lp,t) in a fiber with full label L (has the top-seam bit H) are adjacent iff,
writing hi = lo^L, hq = lp^L:
        sg(lo,lp) + s t sg(hi,hq) = 0   and   t sg(lo,hq) + s sg(hi,lp) = 0   and   (the lp<->lo swap of
        both).  Equivalently: the pair is "resonant" (a sign-independent predicate R on (lo,lp)) AND the
        sign product s*t equals a fixed eps(lo,lp).  So each annihilation graph is the Z2 signed
        double-cover of a "lo-graph" (R, eps).

WHY THE ORBIT THEOREM CANNOT SEE THIS.  Seam(Y) is a size-1 orbit and Fano(Y & (Y-1)) is size-7; being
different orbits, NO algebra automorphism maps between them.  Indeed tau = swap(0, lsb(Y)) is NOT a
signed-monomial automorphism (its sigma-ratio has a nonzero associator on the full index range).  So Phi
is a genuinely NON-automorphic graph isomorphism -- the collapse is invisible to the group action, which
is precisely why "distinct orbits => distinct geometries" failed.

STATUS -- HONEST.
  * Explicit iso Phi, and that it IS a graph isomorphism for EVERY even-weight collapse pair: VERIFIED
    n=6,7,8 (1 + 3 + 7 = 11 pairs, 0 adjacency mismatches).
  * tau commutes with the hi-map (tau(a ^ L_seam) = tau(a) ^ L_fano): holds by linearity, all n.
  * FORALL-n PROOF reduces to two sigma-cocycle lemmas, both VERIFIED n<=8 but NOT proved here:
      (L1) resonance-preservation: R_{L_seam}(a,b) = R_{L_fano}(tau a, tau b) for all a,b;
      (L2) switching balance: the discrepancy eps_fano(tau a, tau b) * eps_seam(a,b) is a coboundary on
           the resonance graph (so lambda exists).
    (L1) is an associator-product identity and looks the more tractable; (L2) is the delicate half.
  * The NON-collapse direction (odd-weight seams stay distinct = Fano-stratum injectivity) is OUTSIDE
    this file -- it needs spectral / nauty data (reviewer, n<=8), and is the open part of the full law.
  Tag: VERIFIED(n=6,7,8) constructive isomorphism; CONJECTURE(forall n) via (L1)+(L2).
"""
from collections import Counter


def cd_sigma(a, b, bits):
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    aH, bH, aL, bL = a >= half, b >= half, a & (half - 1), b & (half - 1)
    if not aH and not bH:
        return cd_sigma(aL, bL, bits - 1)
    if not aH and bH:
        return cd_sigma(bL, aL, bits - 1)
    if aH and not bH:
        return cd_sigma(aL, bL, bits - 1) if bL == 0 else -cd_sigma(aL, bL, bits - 1)
    return -cd_sigma(bL, aL, bits - 1) if bL == 0 else cd_sigma(bL, aL, bits - 1)


def _adj(lo, s, lp, t, L, n):
    """Closed-form annihilation adjacency of primitives (lo,s),(lp,t) in fiber L (full label)."""
    sg = cd_sigma
    hi, hq = lo ^ L, lp ^ L
    return (sg(lo, lp, n) + s * t * sg(hi, hq, n) == 0
            and t * sg(lo, hq, n) + s * sg(hi, lp, n) == 0
            and sg(lp, lo, n) + t * s * sg(hq, hi, n) == 0
            and s * sg(lp, hi, n) + t * sg(hq, lo, n) == 0)


def _resonant(lo, lp, L, n):
    sg = cd_sigma
    hi, hq = lo ^ L, lp ^ L
    P1 = sg(lo, lp, n) * sg(hi, hq, n)
    P2 = sg(lp, lo, n) * sg(hq, hi, n)
    P3 = sg(lo, hq, n) * sg(hi, lp, n)
    P4 = sg(lp, hi, n) * sg(hq, lo, n)
    return (P1 == P2 == P3 == P4, -P1)


def _swap0j(x, j):
    b0, bj = x & 1, (x >> j) & 1
    return x ^ (1 | (1 << j)) if b0 != bj else x


def _switching(n, Y, j):
    """Solve lambda:[1,H)->{+-1} with eps_fano(tau a,tau b)=lam(a)lam(b)eps_seam(a,b); None if unbalanced."""
    H = 1 << (n - 1)
    Ls, Lf = Y | H, ((Y & (Y - 1)) | 1) | H
    los = range(1, H)
    E_s, E_f = {}, {}
    for a in los:
        for b in los:
            if a < b:
                ok, e = _resonant(a, b, Ls, n)
                if ok:
                    E_s[(a, b)] = e
                ok2, e2 = _resonant(a, b, Lf, n)
                if ok2:
                    E_f[(a, b)] = e2
    adj = {lo: [] for lo in los}
    for (a, b), e in E_s.items():
        pa, pb = _swap0j(a, j), _swap0j(b, j)
        k = (pa, pb) if pa < pb else (pb, pa)
        if k not in E_f:
            return None  # (L1) resonance not preserved
        d = E_f[k] * e
        adj[a].append((b, d))
        adj[b].append((a, d))
    lam = {}
    for start in los:
        if start in lam:
            continue
        lam[start] = 1
        stack = [start]
        while stack:
            u = stack.pop()
            for v, d in adj[u]:
                want = lam[u] * d
                if v in lam:
                    if lam[v] != want:
                        return None  # (L2) unbalanced
                else:
                    lam[v] = want
                    stack.append(v)
    return lam


def verify_collapse(n, Y):
    """Build Phi for even-weight seam Y and verify it is a graph iso Seam(Y) -> Fano(Y & (Y-1))."""
    j = (Y & -Y).bit_length() - 1
    lam = _switching(n, Y, j)
    if lam is None:
        return False, "no switching (L1/L2 failed)"
    H = 1 << (n - 1)
    Ls, Lf = Y | H, ((Y & (Y - 1)) | 1) | H
    V = [(lo, s) for lo in range(1, H) for s in (1, -1)]
    Phi = {(lo, s): (_swap0j(lo, j), lam[lo] * s) for lo, s in V}
    edges = mism = 0
    for i in range(len(V)):
        for k in range(i + 1, len(V)):
            (lo, s), (lp, t) = V[i], V[k]
            a = _adj(lo, s, lp, t, Ls, n)
            (LO, S), (LP, T) = Phi[V[i]], Phi[V[k]]
            b = _adj(LO, S, LP, T, Lf, n)
            edges += a
            mism += (a != b)
    return mism == 0, f"edges={edges}, mismatches={mism}"


def even_weight_seams(n):
    tb = list(range(3, n - 1))  # seam-tower bit positions
    out = []
    for mask in range(1, 1 << len(tb)):
        if bin(mask).count("1") % 2 == 0:
            out.append(sum(1 << tb[i] for i in range(len(tb)) if (mask >> i) & 1))
    return sorted(out)


def main():
    ok_all = True
    for n in (6, 7, 8):
        seams = even_weight_seams(n)
        print(f"n={n}: {len(seams)} even-weight collapse pairs (deficit 2^(n-5)-1 = {2**(n-5)-1})")
        for Y in seams:
            y0 = Y & (Y - 1)
            ok, info = verify_collapse(n, Y)
            ok_all = ok_all and ok
            tag = "ISO OK" if ok else "FAIL"
            print(f"   Seam(Y={Y:3}) ~= Fano(y0={y0:3})  Phi=swap(0,{(Y&-Y).bit_length()-1})+switch : {tag}  ({info})")
    print()
    print("ALL collapse isomorphisms verified (n=6,7,8)." if ok_all
          else "SOME collapse isomorphism FAILED -- investigate.")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
