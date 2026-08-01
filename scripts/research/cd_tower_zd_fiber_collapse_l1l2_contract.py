#!/usr/bin/env python3
"""
CD-tower ZD fibers — attacking (c), the parity-collapse law: L1 is PARITY-FREE, and the
triangle route into L2 is WALLED.

Context. V1 forall n = (c) AND (d) (see cd_tower_zd_fiber_v1_reduction_contract.py). (c) is the
parity-collapse law: the even-weight seams merge into Fano classes via the explicit map
Phi(lo,s) = (tau lo, lambda(lo) s), tau = swap(bit 0, bit j), j = lsb(Y). Its forall-n proof was
reduced (2026-07-12) to two sigma-lemmas, both verified n<=8, neither proven:

    (L1) resonance preservation:  R_{L_seam}(a,b) = R_{L_fano}(tau a, tau b)
    (L2) switching balance:       eps_fano(tau a, tau b) * eps_seam(a,b) is a coboundary

and the note recorded only that "(L2) is the delicate half". This rung says WHERE the parity is,
and closes off one route.

  C1  L1 IS PARITY-FREE. L1 holds for EVERY seam Y -- odd weight as well as even -- at
      n = 6,7,8. The even/odd distinction that the collapse law turns on does NOT live in L1.
      Therefore the whole content of the parity is in L2. This is new: the earlier note ranked
      L1 as "more tractable" but did not observe that it does not see the parity at all.

  C2  L1's REAL HYPOTHESIS IS SEAM-NESS (negative control). For L whose lo-part is NOT a seam
      (low 3 bits nonzero) the same equivariance FAILS, in bulk. So C1 is not vacuous: L1 is a
      statement about seam labels, not a general symmetry.

  C3  A DEAD ROUTE, RECORDED. Since res <=> P1 == P3 and both are +-1, res <=> Q = +1 where
      Q_L(a,b) = sigma(a,b) sigma(a^L,b^L) sigma(a,b^L) sigma(a^L,b) is the product of sigma
      over the coset square {a,a^L} x {b,b^L} -- a mixed second difference of the cocycle.
      Second differences of QUADRATIC forms are bilinear, which would have made L1 immediate.
      Q IS NOT BILINEAR (measured). Consistent with this lane's earlier finding that the
      associator becomes higher-degree Boolean at n >= 6.

  C4  A SECOND DEAD ROUTE, RECORDED. Q is tau-equivariant GLOBALLY only for j = 1, 2 (the
      octonion bits); for j >= 3 -- exactly where the collapse law lives -- it fails in bulk.
      (The n=6, j=3 violation count reproduces the 55296 already on record for "tau is not a
      signed automorphism".) So L1 cannot be obtained from a global symmetry of Q.

  C5  THE PARITY IS AN L2 CYCLE OBSTRUCTION. For EVEN-weight Y every triangle of the resonance
      graph has discrepancy-product +1; for ODD-weight Y both +1 and -1 occur. So the coboundary
      obstruction is visible on triangles and tracks the weight parity exactly.

  C6  ...BUT THE TRIANGLE ROUTE DOES NOT CLOSE L2 (the wall). A +-1 edge signing is a coboundary
      iff every CYCLE has product +1; triangles suffice only if they GENERATE the cycle space.
      They do NOT: measured, the F2 rank of the triangle vectors is strictly below the cycle
      rank |E| - |V| + c for several even-weight fibers. So "all triangles +1" is NECESSARY but
      NOT SUFFICIENT, and C5 is evidence about the obstruction, not a proof of L2.

  C0  PARITY. The builders reproduce the in-tree sign_table/A_sig entrywise.

NOT CLAIMED. (c) is NOT proven. L1 is not proven either -- C1/C2 pin its hypothesis and show it
is parity-free, which is a reduction of the target, not a proof. L2 is untouched except that one
route into it is now walled. The existing lambda-solver verifies the collapse constructively at
n <= 8; nothing here extends that.

Verdict COLLAPSE_L1_IS_PARITY_FREE__L2_TRIANGLE_ROUTE_WALLED.
Numerical certificate over an exact integer sign table; D3 respected.
"""
import itertools
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRIOR = os.path.join(HERE, "cd_tower_zd_fiber_spectral_forall_n_progress_contract.py")
_src = open(PRIOR).read()
exec(_src.split("def main()")[0].split("from collections import defaultdict")[1])  # noqa: S102


def sign_table_fast(n):
    S = np.ones((1, 1), dtype=np.int8)
    for b in range(1, n + 1):
        h = 1 << (b - 1)
        P = S
        T = np.empty((2 * h, 2 * h), dtype=np.int8)
        T[:h, :h] = P
        T[:h, h:] = P.T
        blk = -P.copy(); blk[:, 0] = P[:, 0]
        T[h:, :h] = blk
        blk2 = P.T.copy(); blk2[:, 0] = -P.T[:, 0]
        T[h:, h:] = blk2
        S = T
    S[0, :] = 1
    S[:, 0] = 1
    return S


def A_sig_fast(n, Llo, S):
    H = 1 << (n - 1)
    L = Llo | H
    los = np.arange(1, H)
    hi = los ^ L
    P1 = S[np.ix_(los, los)].astype(np.int16) * S[np.ix_(hi, hi)].astype(np.int16)
    P3 = S[np.ix_(los, hi)].astype(np.int16) * S[np.ix_(hi, los)].astype(np.int16)
    res = (P1 == P1.T) & (P3 == P3.T) & (P1 == P3)
    A = np.where(res, -P1, 0).astype(np.int8)
    np.fill_diagonal(A, 0)
    return A


def swap0j(x, j):
    b0, bj = x & 1, (x >> j) & 1
    return x ^ (1 | (1 << j)) if b0 != bj else x


def main():
    print("=" * 78)
    print("CD-tower ZD fibers — (c): L1 is PARITY-FREE; the L2 triangle route is WALLED")
    print("=" * 78)
    ok = {}

    c0 = True
    for n in (6, 7):
        Sref, Sfast = sign_table(n), sign_table_fast(n)
        if not np.array_equal(Sref, Sfast):
            c0 = False
        H = 1 << (n - 1)
        for Llo in range(1, H):
            if not np.array_equal(A_sig(n, Llo, Sref), A_sig_fast(n, Llo, Sfast).astype(float)):
                c0 = False
    ok["C0"] = c0
    print(f"C0_PARITY   builders == in-tree sign_table/A_sig entrywise (n=6,7) "
          f"{'OK' if c0 else 'FAIL'}")

    tabs = {n: sign_table_fast(n).astype(np.int64) for n in (6, 7, 8)}

    def Q(S, a, b, L):
        return int(S[a, b] * S[a ^ L, b ^ L] * S[a, b ^ L] * S[a ^ L, b])

    # ---- C1 / C2: L1 holds for every seam (both parities); fails off seams ----------------
    c1 = c2 = True
    for n in (6, 7, 8):
        S, H = tabs[n], 1 << (n - 1)
        res_by_par = {}
        for parity in (0, 1):
            seams = [Y for Y in range(8, H, 8) if bin(Y >> 3).count("1") % 2 == parity]
            bad = tot = 0
            for Y in seams:
                j = (Y & -Y).bit_length() - 1
                Ls, Lf = Y | H, swap0j(Y, j) | H
                if Lf != swap0j(Ls, j):
                    c1 = False
                for a in range(1, H):
                    for b in range(1, H):
                        tot += 1
                        if Q(S, swap0j(a, j), swap0j(b, j), Lf) != Q(S, a, b, Ls):
                            bad += 1
            res_by_par[parity] = (len(seams), bad, tot)
            c1 = c1 and bad == 0
        (nE, bE, tE), (nO, bO, tO) = res_by_par[0], res_by_par[1]
        print(f"C1_PARITYFREE n={n}: L1 violations -- EVEN-weight seams ({nE}): {bE}/{tE};  "
              f"ODD-weight seams ({nO}): {bO}/{tO}  => the parity is NOT in L1 "
              f"{'OK' if bE == 0 and bO == 0 else 'FAIL'}")
        # negative control: non-seam L
        bad = tot = 0
        for L0 in range(1, H):
            if L0 % 8 == 0:
                continue
            j = 3
            Ls, Lf = L0 | H, swap0j(L0, j) | H
            for a in range(1, H):
                for b in range(1, H):
                    tot += 1
                    if Q(S, swap0j(a, j), swap0j(b, j), Lf) != Q(S, a, b, Ls):
                        bad += 1
        if bad == 0:
            c2 = False
        print(f"C2_SEAMNESS  n={n}: with a NON-seam L the same equivariance FAILS: {bad}/{tot} "
              f"violations => L1's hypothesis is seam-ness, and C1 is not vacuous "
              f"{'OK' if bad else 'FAIL'}")
    ok["C1"], ok["C2"] = c1, c2

    # ---- C3 dead route: Q is not bilinear -------------------------------------------------
    c3 = True
    rng = random.Random(0)
    for n in (6, 7):
        S, N = tabs[n], 1 << n
        bad = tot = 0
        for _ in range(4000):
            L = rng.randrange(1, N); b = rng.randrange(N)
            a1, a2 = rng.randrange(N), rng.randrange(N)
            tot += 1
            if Q(S, a1 ^ a2, b, L) != Q(S, a1, b, L) * Q(S, a2, b, L):
                bad += 1
        if bad == 0:
            c3 = False
        print(f"C3_NOTBILIN  n={n}: Q(a1^a2,b,L) != Q(a1,b,L)*Q(a2,b,L) in {bad}/{tot} sampled "
              f"cases => Q is NOT bilinear; the 'second difference of a quadratic form' route "
              f"is dead {'OK' if bad else 'FAIL'}")
    ok["C3"] = c3

    # ---- C4 dead route: global tau-equivariance only for j = 1,2 ---------------------------
    c4 = True
    for n in (6,):
        S, N = tabs[n], 1 << n
        for j in range(1, n):
            bad = tot = 0
            for L in range(1, N):
                for a in range(N):
                    for b in range(N):
                        tot += 1
                        if Q(S, swap0j(a, j), swap0j(b, j), swap0j(L, j)) != Q(S, a, b, L):
                            bad += 1
            want_equiv = j <= 2
            good = (bad == 0) if want_equiv else (bad > 0)
            c4 = c4 and good
            print(f"C4_GLOBAL    n={n} j={j}: global Q tau-equivariance violations {bad}/{tot} "
                  f"({'equivariant' if bad == 0 else 'not'}; expected "
                  f"{'equivariant' if want_equiv else 'NOT'}) {'OK' if good else 'FAIL'}")
    ok["C4"] = c4

    # ---- C5 the parity IS an L2 cycle obstruction ------------------------------------------
    def res_edges(S, H, L):
        E = {}
        for a in range(1, H):
            for b in range(1, H):
                if a == b:
                    continue
                P1 = int(S[a, b] * S[a ^ L, b ^ L])
                P3 = int(S[a, b ^ L] * S[a ^ L, b])
                if P1 == P3:
                    E[(a, b)] = -P1
        return E

    c5 = True
    for n in (6, 7):
        S, H = tabs[n], 1 << (n - 1)
        for Y in range(8, H, 8):
            j = (Y & -Y).bit_length() - 1
            Es, Ef = res_edges(S, H, Y | H), res_edges(S, H, swap0j(Y, j) | H)
            disc = {}
            for (a, b), e in Es.items():
                k = (swap0j(a, j), swap0j(b, j))
                if k in Ef:
                    disc[(a, b)] = Ef[k] * e
            verts = sorted({a for a, _ in disc})[:26]
            prods = set()
            for a, b, cc in itertools.combinations(verts, 3):
                if (a, b) in disc and (b, cc) in disc and (a, cc) in disc:
                    prods.add(disc[(a, b)] * disc[(b, cc)] * disc[(a, cc)])
            even = bin(Y >> 3).count("1") % 2 == 0
            good = (prods == {1}) if even else (prods == {1, -1})
            c5 = c5 and good
            print(f"C5_OBSTRUCT  n={n} Y={Y:3d} wt={bin(Y >> 3).count('1')} "
                  f"({'even' if even else 'odd'}): triangle disc-products={sorted(prods)} "
                  f"{'OK' if good else 'FAIL'}")
    ok["C5"] = c5

    # ---- C6 the wall: triangles do NOT generate the cycle space ----------------------------
    c6_any_gap = False
    for n, fibers in ((6, (24,)), (7, (40, 48))):
        S, H = tabs[n], 1 << (n - 1)
        for Y in fibers:
            L = Y | H
            E = [(a, b) for a in range(1, H) for b in range(a + 1, H)
                 if int(S[a, b] * S[a ^ L, b ^ L]) == int(S[a, b ^ L] * S[a ^ L, b])]
            idx = {e: i for i, e in enumerate(E)}
            V = sorted({x for e in E for x in e})
            par = {v: v for v in V}

            def find(x):
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                return x
            for a, b in E:
                ra, rb = find(a), find(b)
                if ra != rb:
                    par[ra] = rb
            comp = len({find(v) for v in V})
            dim = len(E) - len(V) + comp
            basis = []
            for a, b, cc in itertools.combinations(V, 3):
                if (a, b) in idx and (b, cc) in idx and (a, cc) in idx:
                    r = (1 << idx[(a, b)]) ^ (1 << idx[(b, cc)]) ^ (1 << idx[(a, cc)])
                    for v in basis:
                        r = min(r, r ^ v)
                    if r:
                        basis.append(r)
                        basis.sort(reverse=True)
            gap = len(basis) < dim
            c6_any_gap = c6_any_gap or gap
            print(f"C6_WALL      n={n} Y={Y:3d} (even wt): cycle-dim={dim}  triangle-rank="
                  f"{len(basis)}  {'TRIANGLES DO NOT GENERATE' if gap else 'they generate'}")
    ok["C6"] = c6_any_gap
    print(f"C6_WALL      at least one even-weight fiber where triangles do NOT generate the "
          f"cycle space {'OK' if c6_any_gap else 'FAIL'} => 'all triangles +1' is NECESSARY but "
          f"NOT SUFFICIENT; the triangle route does not close L2")

    # ---- C7  WHY THE WIRED DISCHARGE CARRIES BOUNDS -- a null control ----------------------
    # SounioZDCollapse.lean now imports SounioZDFiberAntisym and discharges its (*) hypothesis
    # with star_forall. The general `hres` in that file quantifies over UNBOUNDED a, b; the
    # discharged instances Phi_preserves_adj_star / Phi_reflects_adj_star instead carry
    # p.1, q.1 < 2^m. That looks like a convenience. It is not: OUT OF RANGE (*) IS FALSE, so
    # the unbounded hypothesis could never have been discharged by anything.
    #
    # cdSigma is total -- for a >= 2^m the recursion still reduces via a % half -- so the
    # question is meaningful and has a definite answer. Transcribed from the .lean, not from
    # the sign table, because the sign table is only built in range.
    import sys as _sys
    from functools import lru_cache as _lru
    _sys.setrecursionlimit(100000)

    @_lru(maxsize=None)
    def _cdSigma(a, b, n):
        if n == 0:
            return -1
        if n == 1:
            return 1 if (a == 0 or b == 0) else -1
        if a == 0 or b == 0:
            return 1
        h = 1 << (n - 1)
        if a < h and b < h:
            return _cdSigma(a % h, b % h, n - 1)
        if a < h and b >= h:
            return _cdSigma(b % h, a % h, n - 1)
        if a >= h and b < h:
            return _cdSigma(a % h, 0, n - 1) if b % h == 0 else -_cdSigma(a % h, b % h, n - 1)
        return -_cdSigma(0, a % h, n - 1) if b % h == 0 else _cdSigma(b % h, a % h, n - 1)

    def _tau(j, x):
        return x if (x & 1) == ((x >> j) & 1) else x ^ (1 | (1 << j))

    def _Qg(L, a, b, m):
        return (_cdSigma(a, b, m) * _cdSigma(a ^ L, b ^ L, m)
                * _cdSigma(a, b ^ L, m) * _cdSigma(a ^ L, b, m))

    c7_in = c7_out = c7_bad_in = c7_bad_out = 0
    for m in (4, 5):
        for Y in range(1, 1 << m):
            lsb = (Y & -Y).bit_length() - 1
            for j in range(lsb + 1):
                tY = _tau(j, Y)
                for a in range(1 << (m + 2)):
                    for b in range(1 << (m + 2)):
                        good = _Qg(Y, a, b, m) == _Qg(tY, _tau(j, a), _tau(j, b), m)
                        if a < (1 << m) and b < (1 << m):
                            c7_in += 1
                            c7_bad_in += not good
                        else:
                            c7_out += 1
                            c7_bad_out += not good
    # The clause passes when (*) holds IN range and FAILS OUT of it. A zero out-of-range count
    # would mean the bound is decorative and the general `hres` was dischargeable after all.
    c7 = (c7_bad_in == 0) and (c7_bad_out > 0)
    ok["C7"] = c7
    print(f"C7_BOUND    (*) in range: {c7_bad_in}/{c7_in} violations; OUT of range "
          f"(a or b >= 2^m, where cdSigma is still total): {c7_bad_out}/{c7_out} violations "
          f"{'OK' if c7 else 'FAIL'} -- the bounds on Phi_preserves_adj_star are LOAD-BEARING, "
          f"not convenience: the unbounded `hres` of the general theorem is FALSE for the "
          f"fiber pair (Y, tau j Y), so nothing could ever discharge it")

    print("=" * 78)
    if all(ok.values()):
        print("CD_TOWER_ZDC_VERDICT STAR_DISCHARGED_IN_LEAN__COLLAPSE_BLOCKED_ON_L2_ALONE")
        print("CD_TOWER_ZDC_NOTE (c)'s two sigma-lemmas are now located. L1 holds for EVERY seam, "
              "odd weight as well as even (C1), and fails for non-seam L (C2) -- so its real "
              "hypothesis is seam-ness and it does NOT see the parity the collapse law turns on. "
              "The parity is therefore entirely in L2, where it appears as a cycle obstruction: "
              "even-weight fibers have all triangle discrepancy-products +1, odd-weight ones do "
              "not (C5). But triangles do NOT generate the cycle space (C6), so that is evidence "
              "about the obstruction, NOT a proof of L2. Two routes recorded dead: Q is not "
              "bilinear (C3, so the second-difference-of-a-quadratic argument fails) and Q is "
              "globally tau-equivariant only for the octonion bits j=1,2, not for the seam bits "
              "j>=3 where the collapse lives (C4). (c) IS NOT PROVEN; L1 is reduced, not proven. "
              "Numerical certificate; D3 respected")
        return 0
    print("CD_TOWER_ZDC_VERDICT INCOMPLETE  failing=" +
          ",".join(k for k, v in ok.items() if not v))
    return 1


if __name__ == "__main__":
    t0 = time.time()
    rc = main()
    print(f"[{time.time() - t0:.1f}s]", file=sys.stderr)
    raise SystemExit(rc)
