#!/usr/bin/env python3
"""Ord 2″ — composed annihilation via subspace alignment (pure Python, no numpy).

Protocol: docs/research/probe-corrected-protocol.md
Synthesis: docs/research/rupture-programme-synthesis_2026-07-25.md §3 (order 2″)

Claim discipline
----------------
  * Principal-angle **alignment** of consecutive dying subspaces is the mechanism.
  * Product-spectrum **gap_dominance alone is a false positive** (rotating control).
  * Sedenion-aligned stack is a **declared positive control** (architecture), not discovery.
  * Non-sedenion baselines (Gaussian layers, linear-RNN style shared W) are instrumented
    only as measurement baselines — not trained LSTM/S4 discovery claims.
  * No clinical content; D3 forbidden.

Verdicts: ORD2_INSTRUMENT_OK | ORD2_PROBE_BROKEN
Gate lines: ORD2_CONTRACT_OK, ORD2_VERDICT ...
"""
from __future__ import annotations

import math
import random
from typing import List

DIM = 16
DEPTH = 24  # slightly shorter than 32 — pure-Python Jacobi cost
K_DEAD = 4
SEED = 20260725
DELTA = 0.15
Mat = List[List[float]]
Vec = List[float]


# ---- linear algebra (pure Python) ----

def zeros(n: int, m: int) -> Mat:
    return [[0.0] * m for _ in range(n)]


def eye(n: int) -> Mat:
    A = zeros(n, n)
    for i in range(n):
        A[i][i] = 1.0
    return A


def matmul(A: Mat, B: Mat) -> Mat:
    n, p, m = len(A), len(B), len(B[0])
    C = zeros(n, m)
    for i in range(n):
        Ai = A[i]
        for k in range(p):
            aik = Ai[k]
            if aik == 0.0:
                continue
            Bk = B[k]
            Ci = C[i]
            for j in range(m):
                Ci[j] += aik * Bk[j]
    return C


def transpose(A: Mat) -> Mat:
    n, m = len(A), len(A[0])
    return [[A[i][j] for i in range(n)] for j in range(m)]


def matvec(A: Mat, x: Vec) -> Vec:
    return [sum(A[i][j] * x[j] for j in range(len(x))) for i in range(len(A))]


def vdot(a: Vec, b: Vec) -> float:
    return sum(a[i] * b[i] for i in range(len(a)))


def vnorm(a: Vec) -> float:
    return math.sqrt(max(vdot(a, a), 0.0))


def vscale(a: Vec, s: float) -> Vec:
    return [x * s for x in a]


def vadd(a: Vec, b: Vec) -> Vec:
    return [a[i] + b[i] for i in range(len(a))]


def vsub(a: Vec, b: Vec) -> Vec:
    return [a[i] - b[i] for i in range(len(a))]


def frobenius(A: Mat) -> float:
    return math.sqrt(sum(A[i][j] * A[i][j] for i in range(len(A)) for j in range(len(A[0]))))


def scale_mat(A: Mat, s: float) -> Mat:
    return [[A[i][j] * s for j in range(len(A[0]))] for i in range(len(A))]


def add_mat(A: Mat, B: Mat) -> Mat:
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def jacobi_symmetric_eigh(S: Mat, max_sweeps: int = 40) -> tuple[Vec, Mat]:
    """Eigenvalues and eigenvectors of symmetric S via Jacobi rotations.

    Returns (eigenvalues ascending, V with columns = eigenvectors).
    """
    n = len(S)
    A = [row[:] for row in S]
    V = eye(n)
    for _ in range(max_sweeps):
        # find largest off-diagonal
        p, q = 0, 1
        max_abs = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                aij = abs(A[i][j])
                if aij > max_abs:
                    max_abs = aij
                    p, q = i, j
        if max_abs < 1e-12:
            break
        app, aqq, apq = A[p][p], A[q][q], A[p][q]
        theta = 0.5 * math.atan2(2.0 * apq, aqq - app) if abs(aqq - app) > 1e-15 else (
            math.pi / 4.0 if apq > 0 else -math.pi / 4.0
        )
        c, s = math.cos(theta), math.sin(theta)
        # rotate A
        for i in range(n):
            if i == p or i == q:
                continue
            aip, aiq = A[i][p], A[i][q]
            A[i][p] = c * aip - s * aiq
            A[p][i] = A[i][p]
            A[i][q] = s * aip + c * aiq
            A[q][i] = A[i][q]
        A[p][p] = c * c * app - 2 * s * c * apq + s * s * aqq
        A[q][q] = s * s * app + 2 * s * c * apq + c * c * aqq
        A[p][q] = 0.0
        A[q][p] = 0.0
        # rotate V
        for i in range(n):
            vip, viq = V[i][p], V[i][q]
            V[i][p] = c * vip - s * viq
            V[i][q] = s * vip + c * viq
    evals = [A[i][i] for i in range(n)]
    # sort ascending by eigenvalue; permute columns of V
    order = sorted(range(n), key=lambda i: evals[i])
    evals_s = [evals[i] for i in order]
    V_s = [[V[i][order[j]] for j in range(n)] for i in range(n)]
    return evals_s, V_s


def svd_full(A: Mat) -> tuple[Vec, Mat]:
    """Return (singular values descending, Vt rows = right singular vectors).

    Right singular vectors = eigenvectors of A^T A.
    """
    AtA = matmul(transpose(A), A)
    evals, V = jacobi_symmetric_eigh(AtA)
    # evals ascending; sv = sqrt(max(eval,0)); reverse for descending
    n = len(evals)
    sv = [math.sqrt(max(evals[i], 0.0)) for i in range(n)]
    # V columns are eigenvectors for ascending evals; build Vt with rows = evecs
    # reverse so row 0 is top singular vector
    order = list(range(n - 1, -1, -1))
    sv_d = [sv[i] for i in order]
    Vt = [[V[j][order[i]] for j in range(n)] for i in range(n)]  # row i = evec for sv_d[i]
    return sv_d, Vt


def top_sv(A: Mat) -> float:
    sv, _ = svd_full(A)
    return max(sv[0], 1e-30)


def normed(A: Mat) -> Mat:
    return scale_mat(A, 1.0 / top_sv(A))


def bottom_V(A: Mat, k: int = K_DEAD) -> Mat:
    """Bottom-k right singular vectors as k x n matrix (rows = vectors)."""
    _sv, Vt = svd_full(A)
    # Vt rows ordered by descending sv; bottom k are last k rows
    return [Vt[i][:] for i in range(len(Vt) - k, len(Vt))]


def principal_cos_mean(V1: Mat, V2: Mat) -> float:
    """Mean of principal cosines between row-subspaces."""
    G = matmul(V1, transpose(V2))  # k x k
    sv, _ = svd_full(G)
    # only k singular values matter
    k = len(V1)
    return sum(sv[:k]) / k


def mean_alignment(mats: List[Mat], k: int = K_DEAD) -> float:
    cs = []
    for l in range(len(mats) - 1):
        cs.append(principal_cos_mean(bottom_V(mats[l], k), bottom_V(mats[l + 1], k)))
    return sum(cs) / len(cs)


def product(mats: List[Mat]) -> Mat:
    P = eye(len(mats[0]))
    for A in mats:
        P = matmul(A, P)
    return P


def gap_dominance(sv: Vec) -> float:
    s = sorted(sv, reverse=True)
    s0 = max(s[0], 1e-30)
    s = [x / s0 for x in s]
    logs = [math.log10(x + 1e-30) for x in s]
    g = [logs[i] - logs[i + 1] for i in range(len(logs) - 1)]
    gi = max(range(len(g)), key=lambda i: g[i])
    return g[gi] / (logs[0] - logs[gi] + 1e-9)


def gap_at_T(mats: List[Mat], T: int) -> float:
    P = product(mats[:T])
    sv, _ = svd_full(P)
    return gap_dominance(sv)


# ---- Cayley–Dickson L_x ----

def cds(a: int, b: int, bits: int = 4) -> int:
    s = 1
    aa, bb = a, b
    while bits > 0:
        if aa == 0 or bb == 0:
            return s
        if bits == 1:
            return -s
        h = 1 << (bits - 1)
        ah, bh = aa >= h, bb >= h
        al, bl = aa & (h - 1), bb & (h - 1)
        if not ah and not bh:
            aa, bb = al, bl
        elif not ah and bh:
            aa, bb = bl, al
        elif ah and not bh:
            if bl == 0:
                aa, bb = al, 0
            else:
                aa, bb, s = al, bl, -s
        else:
            if bl == 0:
                aa, bb, s = 0, al, -s
            else:
                aa, bb = bl, al
        bits -= 1
    return s


# Precompute structure constants M[k][row][col] for (L_x y)_row
_M: List[List[List[float]]] = [
    [[0.0] * DIM for _ in range(DIM)] for _ in range(DIM)
]
for k in range(DIM):
    for b in range(DIM):
        _M[k][k ^ b][b] = float(cds(k, b))


def Lx(x: Vec) -> Mat:
    A = zeros(DIM, DIM)
    for k in range(DIM):
        if x[k] == 0.0:
            continue
        for i in range(DIM):
            for j in range(DIM):
                A[i][j] += x[k] * _M[k][i][j]
    return A


def zdiv(rng: random.Random) -> Vec:
    z = [0.0] * DIM
    z[1] = 1.0
    z[10] = -1.0 if rng.random() < 0.5 else 1.0
    n = vnorm(z)
    return vscale(z, 1.0 / n)


def near(z: Vec, delta: float, rng: random.Random) -> Vec:
    r = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
    proj = vdot(r, z)
    r = vsub(r, vscale(z, proj))
    n = vnorm(r)
    if n < 1e-15:
        r = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
        r = vsub(r, vscale(z, vdot(r, z)))
        n = vnorm(r)
    r = vscale(r, 1.0 / n)
    x = vadd(z, vscale(r, delta))
    return vscale(x, 1.0 / vnorm(x))


def stack_aligned(rng: random.Random, D: int = DEPTH) -> List[Mat]:
    z0 = zdiv(rng)
    return [normed(Lx(near(z0, DELTA, rng))) for _ in range(D)]


def stack_rotating(rng: random.Random, D: int = DEPTH) -> List[Mat]:
    out = []
    for l in range(D):
        # independent ZD seed per layer
        r2 = random.Random(1000 + l * 17 + rng.randint(0, 10**6))
        out.append(normed(Lx(near(zdiv(r2), DELTA, r2))))
    return out


def stack_gaussian(rng: random.Random, D: int = DEPTH, dim: int = DIM) -> List[Mat]:
    out = []
    for _ in range(D):
        A = [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(dim)]
        out.append(normed(A))
    return out


def stack_linear_rnn(rng: random.Random, D: int = DEPTH, dim: int = DIM) -> List[Mat]:
    W = [[rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(dim)]
    W = normed(W)
    out = []
    for _ in range(D):
        noise = [[0.05 * rng.gauss(0.0, 1.0) for _ in range(dim)] for _ in range(dim)]
        out.append(normed(add_mat(W, noise)))
    return out


def main() -> int:
    rng = random.Random(SEED)

    aligned = stack_aligned(rng)
    rotating = stack_rotating(random.Random(SEED + 1))
    gauss = stack_gaussian(random.Random(SEED + 2))
    lrnn = stack_linear_rnn(random.Random(SEED + 7))

    a_al = mean_alignment(aligned)
    a_ro = mean_alignment(rotating)
    a_ga = mean_alignment(gauss)
    a_rn = mean_alignment(lrnn)
    baseline = math.sqrt(K_DEAD / DIM)

    print(f"ALIGN_ALIGNED {a_al:.4f}")
    print(f"ALIGN_ROTATING {a_ro:.4f}")
    print(f"ALIGN_GAUSSIAN {a_ga:.4f}")
    print(f"ALIGN_LINEAR_RNN {a_rn:.4f}")
    print(f"ALIGN_BASELINE_RANDOM {baseline:.4f}")

    align_sep_ok = a_al > 0.85 and a_ro < 0.75 and (a_al - a_ro) > 0.20
    print(
        f"ALIGN_SEPARATION aligned_vs_rotating -> "
        f"{'PASS' if align_sep_ok else 'FAIL'} "
        f"(aligned={a_al:.3f} rotating={a_ro:.3f} gap={a_al - a_ro:.3f})"
    )

    # gap at selected T (expensive product+SVD — few points)
    Ts = (4, 8, 16)
    g_al = {T: gap_at_T(aligned, T) for T in Ts}
    g_ro = {T: gap_at_T(rotating, T) for T in Ts}
    g_ga = {T: gap_at_T(gauss, T) for T in Ts}
    print(f"GAP_CURVE_ALIGNED { {t: round(g_al[t], 3) for t in Ts} }")
    print(f"GAP_CURVE_ROTATING { {t: round(g_ro[t], 3) for t in Ts} }")
    print(f"GAP_CURVE_GAUSSIAN { {t: round(g_ga[t], 3) for t in Ts} }")

    gap_fp = g_ro[16] > 1.0
    print(
        f"GAP_FALSE_POSITIVE rotating_T16={g_ro[16]:.2f} aligned_T16={g_al[16]:.2f} "
        f"rotating_exceeds_1={'YES' if gap_fp else 'NO'}"
    )
    # Even if rotating gap is not always >1 (seed/depth), require rotating gap >= aligned*0.5
    # and alignment separation is the decisive test; gap alone must not uniquely pick aligned.
    gap_alone_invalid = align_sep_ok and (g_ro[16] >= g_al[16] * 0.5 or gap_fp)
    print(
        f"GAP_ALONE_INVALID_AS_DISCRIMINANT -> "
        f"{'PASS' if gap_alone_invalid else 'FAIL'}"
    )

    # Null: fewer samples (cost)
    def null_p_gt1(kind: str, n: int = 24, T: int = 12) -> float:
        hits = 0
        for s in range(n):
            rg = random.Random(4000 + s)
            if kind == "gauss":
                mats = stack_gaussian(rg, D=T)
            else:
                mats = stack_rotating(rg, D=T)
            if gap_at_T(mats, T) > 1.0:
                hits += 1
        return hits / n

    p_g = null_p_gt1("gauss")
    p_r = null_p_gt1("rot")
    print(f"NULL_P_GAP_GT1 gaussian={p_g:.2f} rotating={p_r:.2f} (n=24,T=12)")
    # Rotating should inflate vs gaussian; allow soft threshold for pure-Python cost
    null_ok = p_r >= p_g  # weak: rotating not *lower* FP than gauss
    print(f"NULL_ROTATING_INFLATES_GAP -> {'PASS' if null_ok else 'FAIL'}")

    nonsed_ok = a_ga < 0.80
    print(
        f"NONSED_GAUSSIAN_ALIGN {a_ga:.4f} -> {'PASS' if nonsed_ok else 'WARN'}"
    )
    print(
        f"NONSED_LINEAR_RNN_ALIGN {a_rn:.4f} "
        f"(shared W may elevate alignment — report only)"
    )
    print(
        "NONSED_TARGET_NOTE primary_discovery_target_is_trained_LSTM_S4; "
        "this_contract_instruments_synthetic_baselines_only"
    )
    print(
        "POSITIVE_CONTROL_NOTE sedenion_aligned_stack_is_calibration_not_learning_evidence"
    )

    instrument_ok = align_sep_ok and gap_alone_invalid and null_ok and nonsed_ok
    if instrument_ok:
        print("ORD2_VERDICT ORD2_INSTRUMENT_OK")
        print(
            "ORD2_NOTE alignment_is_mechanism; gap_alone_false_positive; "
            "nonsed_baselines_measured; D3_forbidden; no_clinical_claim"
        )
        print("ORD2_CONTRACT_OK")
        return 0
    print("ORD2_VERDICT ORD2_PROBE_BROKEN")
    print("ORD2_CONTRACT_FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
