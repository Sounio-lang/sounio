#!/usr/bin/env python3
r"""Probe for Conjecture 6.8 of Guterman-Zhilina, arXiv:2608.26903 (27 Aug 2026):
the diameter of the commutativity graph of the sedenions restricted to elements
whose imaginary part is a zero divisor equals 3 (they prove 3 <= d <= 4 and
conjecture 3 from floating-point Mathematica experiments).

Reduction implemented here (exact over Q wherever linear):
  * Commutation depends only on imaginary parts -> vertices reduce to pure ZDs.
  * For a pure ZD x with n(x) != 0:  Im C(x) = R x (+) O(x),  dim 5,
    where O(x) = ker L_x  /\  ker R_x  (dim 4, Moreno; our ker machine).
  * d(x,x') = 1  iff  [x,x'] = 0.
  * d(x,x') = 2  iff  U /\ V != 0,  U = Rx+O(x), V = Rx'+O(x')   (exact).
  * d(x,x') = 3  iff  exists u in U, w in V, u,w != 0, [u,w] = 0
    <=>  the exact rational kernel K of the bilinear map
         T : U (x) V -> Im S,  (u,w) |-> [u,w]
    contains a REAL RANK-ONE tensor k m^T.  (Naive dimension counting says it
    should not exist generically; if it always does, that is the octonionic
    structure at work -- and the conjecture holds.)
  * A pair with NO rank-one tensor in K and U /\ V = 0 and [x,x'] != 0 is a
    COUNTEREXAMPLE: diameter 4.

Exact legs: multiplication (integer, cd_sigma), O(x) kernels, U /\ V, K = ker T.
Numeric leg: rank-one hunt in the bilinear form via alternating min-singular-
vector iteration with random restarts (monotone residual per half-step).

Sanity reproductions of the paper (exact): dim O(x) = 4; Prop 6.7 disjointness
Im C(x) /\ Im C(x~) = 0 for the swap x~ (hence d(x,x~) >= 3); Description 4.18
counts (84 basis ZDs, 4 orthogonal partners each).

Output (sorted-stable, gate style):
  SANITY_ODIM4 OK|FAIL
  SANITY_SWAP_PROP67 OK|FAIL
  SANITY_BASIS_HEXAGONS OK|FAIL
  PAIRS <n>  D1 <n>  D2 <n>  D3_WITNESS <n>  D3_UNRESOLVED <n>
  MAX_RESIDUAL <float>            (over all pairs resolved at d=3)
  CONJ68_PROBE OK|CANDIDATE_D4
"""
from __future__ import annotations

import random
import sys
from fractions import Fraction
from itertools import combinations

import numpy as np

DIM = 16
N = 4  # sedenions = A_4

# ---------------------------------------------------------------- multiplication
# cd_sigma transcribed from scripts/research/sedenion_zd_fiber_identity_oracle.py
# (ir_cd_sigma); verified against Table 1 of arXiv:2608.26903 and the canonical
# zero-divisor pair (e1+e10)(e7+e12) = 0.


def cd_sigma(a: int, b: int, bits: int = N) -> int:
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    a_hi, b_hi = a >= half, b >= half
    a_lo, b_lo = a & (half - 1), b & (half - 1)
    if not a_hi and not b_hi:
        return cd_sigma(a_lo, b_lo, bits - 1)
    if not a_hi and b_hi:
        return cd_sigma(b_lo, a_lo, bits - 1)
    if a_hi and not b_hi:
        return cd_sigma(a_lo, b_lo, bits - 1) if b_lo == 0 else -cd_sigma(a_lo, b_lo, bits - 1)
    return -cd_sigma(b_lo, a_lo, bits - 1) if b_lo == 0 else cd_sigma(b_lo, a_lo, bits - 1)


SIGMA = [[cd_sigma(i, j) for j in range(DIM)] for i in range(DIM)]


def mul(x: list, y: list) -> list:
    out = [0] * DIM
    for i, ci in enumerate(x):
        if ci == 0:
            continue
        for j, cj in enumerate(y):
            if cj == 0:
                continue
            out[i ^ j] += SIGMA[i][j] * ci * cj
    return out


def comm(x: list, y: list) -> list:
    xy, yx = mul(x, y), mul(y, x)
    return [a - b for a, b in zip(xy, yx)]


def is_zero(v: list) -> bool:
    return all(c == 0 for c in v)


def e(i: int) -> list:
    v = [0] * DIM
    v[i] = 1
    return v


# ---------------------------------------------------------------- exact kernels


def rref_nullspace(rows: list[list]) -> list[list]:
    """Exact nullspace basis (Fractions cleared to ints) of a matrix given as rows."""
    m = [[Fraction(c) for c in r] for r in rows]
    ncols = DIM
    pivots = []
    r = 0
    for c in range(ncols):
        pr = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if pr is None:
            continue
        m[r], m[pr] = m[pr], m[r]
        pv = m[r][c]
        m[r] = [x / pv for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [x - f * y for x, y in zip(m[i], m[r])]
        pivots.append(c)
        r += 1
        if r == len(m):
            break
    free = [c for c in range(ncols) if c not in pivots]
    basis = []
    for fc in free:
        v = [Fraction(0)] * ncols
        v[fc] = Fraction(1)
        for pi, pc in enumerate(pivots):
            v[pc] = -m[pi][fc]
        # clear denominators -> integer vector
        den = 1
        for x in v:
            den = den * x.denominator // gcd(den, x.denominator)
        iv = [int(x * den) for x in v]
        g = 0
        for x in iv:
            g = gcd(g, abs(x))
        basis.append([x // (g or 1) for x in iv])
    return basis


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def left_mat(x: list) -> list[list]:
    """Rows of L_x as a map v -> x*v, row k = coefficients of output coord k."""
    cols = [mul(x, e(j)) for j in range(DIM)]
    return [[cols[j][k] for j in range(DIM)] for k in range(DIM)]


def right_mat(x: list) -> list[list]:
    cols = [mul(e(j), x) for j in range(DIM)]
    return [[cols[j][k] for j in range(DIM)] for k in range(DIM)]


def orthogonalizer(x: list) -> list[list]:
    r"""O(x) = ker L_x /\ ker R_x, exact integer basis."""
    return rref_nullspace(left_mat(x) + right_mat(x))


def span_intersection(a: list[list], b: list[list]) -> int:
    r"""dim( span(a) /\ span(b) ) via rank(a)+rank(b)-rank(a u b), exact."""
    return rank(a) + rank(b) - rank(a + b)


def rank(rows: list[list]) -> int:
    m = [[Fraction(c) for c in r] for r in rows]
    ncols = len(rows[0]) if rows else 0
    r = 0
    for c in range(ncols):
        pr = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if pr is None:
            continue
        m[r], m[pr] = m[pr], m[r]
        pv = m[r][c]
        m[r] = [x / pv for x in m[r]]
        for i in range(r + 1, len(m)):
            if m[i][c] != 0:
                f = m[i][c]
                m[i] = [x - f * y for x, y in zip(m[i], m[r])]
        r += 1
        if r == len(m):
            break
    return r


# ---------------------------------------------------------------- ZD utilities


def octo_pure_random(rng: random.Random, lo=-9, hi=9) -> list:
    while True:
        a = [0] + [rng.randint(lo, hi) for _ in range(7)]
        if any(a[1:]):
            return a


def norm2(v: list) -> int:
    return sum(c * c for c in v)


def omul(a: list, b: list) -> list:
    """Octonion multiply on 8-vectors via the same sigma table (indices < 8)."""
    out = [0] * 8
    for i, ci in enumerate(a):
        if ci == 0:
            continue
        for j, cj in enumerate(b):
            if cj == 0:
                continue
            out[i ^ j] += SIGMA[i][j] * ci * cj
    return out


def random_rational_zd(rng: random.Random) -> list:
    """Integer sedenion ZD (a, b) with b = a*t/k, t pure, t _|_ 1,a, n(t) = k^2.

    Every ZD is of the form (a, a*t) with t unit pure _|_ a (t = conj(a) b / n(a)),
    so rational-square-norm t samples are dense in the ZD variety.
    """
    while True:
        a = octo_pure_random(rng)
        # search a small integer pure t orthogonal to a with square norm
        for _ in range(4000):
            t = [0] + [rng.randint(-9, 9) for _ in range(7)]
            dot = sum(x * y for x, y in zip(a, t))
            # project away the a-component: t' = n(a) t - dot a  (stays integer, pure)
            tp = [norm2(a) * x - dot * y for x, y in zip(t, a)]
            if not any(tp[1:]):
                continue
            g = 0
            for x in tp:
                g = gcd(g, abs(x))
            tp = [x // (g or 1) for x in tp]
            n = norm2(tp)
            k = int(round(n ** 0.5))
            if k * k == n:
                b = omul(a, tp)
                # sedenion x' = (k a, b): norms k^2 n(a) each; b _|_ a, pure. ZD.
                return [k * c for c in a] + b
        # retry with a fresh a


def zd_check(x: list) -> bool:
    a, b = x[:8], x[8:]
    return (x[0] == 0 and x[8] == 0 and norm2(a) == norm2(b)
            and sum(p * q for p, q in zip(a, b)) == 0
            and not is_zero([0] * 8 + b) if True else False) or _zd_kernel_check(x)


def _zd_kernel_check(x: list) -> bool:
    return len(orthogonalizer(x)) > 0


def swap_of(x: list) -> list:
    return x[8:] + x[:8]


# ---------------------------------------------------------------- d=3 rank-one hunt


def commutant_basis(x: list) -> list[list]:
    return [list(x)] + orthogonalizer(x)


def bilinear_tensor(U: list[list], V: list[list]) -> np.ndarray:
    C = np.zeros((DIM, len(U), len(V)))
    for i, u in enumerate(U):
        for j, w in enumerate(V):
            C[:, i, j] = comm(u, w)
    return C


def _polish(Cs: np.ndarray, k: np.ndarray, m: np.ndarray, iters: int = 60):
    """Gauss-Newton on F(k,m) = sum_ij k_i m_j C[:,i,j] (quadratic near a zero)."""
    res = np.inf
    for _ in range(iters):
        A = np.tensordot(Cs, m, axes=([2], [0]))              # 16 x |U| = dF/dk
        B = np.tensordot(Cs, k, axes=([1], [0]))              # 16 x |V| = dF/dm
        F = A @ k
        res = np.linalg.norm(F)
        if res < 1e-14:
            break
        J = np.hstack([A, B])
        delta, *_ = np.linalg.lstsq(J, -F, rcond=None)
        k = k + delta[: k.size]
        m = m + delta[k.size:]
        nk, nm = np.linalg.norm(k), np.linalg.norm(m)
        if nk < 1e-9 or nm < 1e-9:
            return np.inf, k, m                                # collapsed to trivial zero
        k, m = k / nk, m / nm
    A = np.tensordot(Cs, m, axes=([2], [0]))
    return np.linalg.norm(A @ k), k, m


def rank_one_hunt(C: np.ndarray, restarts: int, iters: int, rng: np.random.Generator):
    """Minimize |sum_ij k_i m_j C[:,i,j]| over unit k, m: alternating SVD steps
    (globalization) + Gauss-Newton polish (local quadratic convergence)."""
    scale = np.abs(C).max() or 1.0
    Cs = C / scale
    best = (np.inf, None, None)
    for _ in range(restarts):
        m = rng.standard_normal(Cs.shape[2])
        m /= np.linalg.norm(m)
        k = None
        res = np.inf
        for _ in range(iters):
            A = np.tensordot(Cs, m, axes=([2], [0]))          # 16 x |U|
            _, s, vt = np.linalg.svd(A, full_matrices=False)
            k = vt[-1]
            B = np.tensordot(Cs, k, axes=([1], [0]))          # 16 x |V|
            _, s2, vt2 = np.linalg.svd(B, full_matrices=False)
            m = vt2[-1]
            res = s2[-1]
            if res < 1e-13:
                break
        if res < 1e-2:                                        # in the basin: polish
            res_p, k_p, m_p = _polish(Cs, k.copy(), m.copy())
            if res_p < res:
                res, k, m = res_p, k_p, m_p
        if res < best[0]:
            best = (res, k.copy(), m.copy())
        if best[0] < 1e-13:
            break
    return best


# ---------------------------------------------------------------- classification


def classify_pair(x: list, xp: list, np_rng: np.random.Generator,
                  restarts: int = 40, iters: int = 60):
    """Returns (d, residual): d in {0,1,2,3}; residual only meaningful for d=3.
    d=0: same line. d never certified as 4 here -- a large residual is only a
    CANDIDATE for d=4 (numeric leg is a hunt, not a proof of absence)."""
    U = commutant_basis(x)
    if rank(U + [xp]) == rank(U) and rank([x, xp]) == 1:
        return 0, 0.0
    if is_zero(comm(x, xp)):
        return 1, 0.0
    V = commutant_basis(xp)
    if span_intersection(U, V) > 0:
        return 2, 0.0
    C = bilinear_tensor(U, V)
    res, _, _ = rank_one_hunt(C, restarts, iters, np_rng)
    if res > 1e-11:                                            # escalation ladder
        res2, _, _ = rank_one_hunt(C, 8 * restarts, 4 * iters, np_rng)
        res = min(res, res2)
    return 3, res


# ---------------------------------------------------------------- sanity legs


def sanity() -> tuple[bool, bool, bool]:
    x = [0] * DIM
    x[1] = 1
    x[10] = 1  # (e1, e2)
    ok_dim = len(orthogonalizer(x)) == 4
    xt = swap_of(x)
    ok_prop67 = span_intersection(commutant_basis(x), commutant_basis(xt)) == 0
    # Description 4.18: 84 basis ZDs, each with exactly 4 orthogonal basis partners
    basis_zds = []
    for lo in range(1, 8):
        for hi in range(1, 8):
            if lo == hi:
                continue
            for s in (1, -1):
                v = [0] * DIM
                v[lo] = 1
                v[8 + hi] = s
                basis_zds.append(v)
    deg = []
    for a in basis_zds:
        d = sum(1 for b in basis_zds if b is not a and is_zero(mul(a, b)))
        deg.append(d)
    ok_hex = len(basis_zds) == 84 and all(d == 4 for d in deg)
    return ok_dim, ok_prop67, ok_hex


# ---------------------------------------------------------------- main sweep


def main() -> None:
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260831
    n_random = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    ok_dim, ok_p67, ok_hex = sanity()
    print(f"SANITY_ODIM4 {'OK' if ok_dim else 'FAIL'}")
    print(f"SANITY_SWAP_PROP67 {'OK' if ok_p67 else 'FAIL'}")
    print(f"SANITY_BASIS_HEXAGONS {'OK' if ok_hex else 'FAIL'}")

    x0 = [0] * DIM
    x0[1] = 1
    x0[10] = 1  # canonical (e1, e2); WLOG by Khalil-Yiu transitivity

    pairs: list[tuple[str, list]] = []
    pairs.append(("swap", swap_of(x0)))
    for lo in range(1, 8):  # basis ZDs vs canonical
        for hi in range(1, 8):
            if lo == hi:
                continue
            for s in (1, -1):
                v = [0] * DIM
                v[lo] = 1
                v[8 + hi] = s
                pairs.append((f"basis({lo},{'+' if s > 0 else '-'}{hi})", v))
    for i in range(n_random):
        pairs.append((f"rand{i}", random_rational_zd(rng)))

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    unresolved = 0
    max_res = 0.0
    worst = None
    for tag, xp in pairs:
        if not _zd_kernel_check(xp):
            print(f"SKIP_NOT_ZD {tag}")
            continue
        d, res = classify_pair(x0, xp, np_rng)
        counts[d] += 1
        if d == 3:
            if res > 1e-9:
                unresolved += 1
                print(f"UNRESOLVED {tag} residual {res:.3e}")
            if res > max_res:
                max_res, worst = res, tag
    total = sum(counts.values())
    print(f"PAIRS {total}  D1 {counts[1]}  D2 {counts[2]}  "
          f"D3_WITNESS {counts[3] - unresolved}  D3_UNRESOLVED {unresolved}")
    print(f"MAX_RESIDUAL {max_res:.3e} ({worst})")
    print(f"CONJ68_PROBE {'OK' if unresolved == 0 and ok_dim and ok_p67 and ok_hex else 'CANDIDATE_D4'}")


if __name__ == "__main__":
    main()
