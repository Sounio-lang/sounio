"""Benchmark: QR Decomposition — Python equivalent.

Hand-rolled Householder QR on 4x4 matrices (no numpy.linalg.qr) for fair comparison.
"""
import time
import math


def vec_norm(v):
    return math.sqrt(sum(x*x for x in v))


def qr_householder(A_flat, n):
    """Householder QR on n x n matrix stored as flat list."""
    R = list(A_flat)
    # Q starts as identity
    Q = [0.0] * (n * n)
    for i in range(n):
        Q[i * n + i] = 1.0

    for k in range(n):
        # Extract column k below diagonal
        col = [R[i * n + k] for i in range(k, n)]
        col_norm = vec_norm(col)
        alpha = -math.copysign(col_norm, col[0])

        # Householder vector
        v = list(col)
        v[0] -= alpha
        v_norm = vec_norm(v)
        if v_norm < 1e-14:
            continue
        v = [vi / v_norm for vi in v]
        col_len = n - k

        # Apply H to R
        for j in range(n):
            dot = sum(v[ii] * R[(ii + k) * n + j] for ii in range(col_len))
            for ii in range(col_len):
                R[(ii + k) * n + j] -= 2.0 * v[ii] * dot

        # Apply H to Q
        for qi in range(n):
            dot = sum(Q[qi * n + (jj + k)] * v[jj] for jj in range(col_len))
            for jj in range(col_len):
                Q[qi * n + (jj + k)] -= 2.0 * dot * v[jj]

    return Q, R


def mat_mul(A, B, n):
    C = [0.0] * (n * n)
    for i in range(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i * n + k] * B[k * n + j]
            C[i * n + j] = s
    return C


def frobenius_diff(A, B, n):
    return math.sqrt(sum((A[i] - B[i])**2 for i in range(n * n)))


def main():
    n = 4
    A = [
        4.0, 1.0, 2.0, 1.0,
        1.0, 3.0, 1.0, 2.0,
        2.0, 1.0, 5.0, 1.0,
        1.0, 2.0, 1.0, 4.0,
    ]

    n_iters = 1000
    max_err = 0.0

    t0 = time.perf_counter()
    for _ in range(n_iters):
        Q, R = qr_householder(A, n)
        QR = mat_mul(Q, R, n)
        err = frobenius_diff(A, QR, n)
        max_err = max(max_err, err)
    t1 = time.perf_counter()

    ms = (t1 - t0) * 1000
    print(f"QR 4x4 ({n_iters} iters): max ||A-QR|| = {max_err:.2e}, time = {ms:.2f} ms")


if __name__ == "__main__":
    main()
