#!/usr/bin/env python3
"""Ref NumPy matmul + linalg.svd"""
import numpy as np
import json

A = np.array([
    [4.0, 1.0, 2.0, 1.0],
    [1.0, 3.0, 1.0, 2.0],
    [2.0, 1.0, 5.0, 1.0],
    [1.0, 2.0, 1.0, 4.0]
])

AA = A @ A
s = np.linalg.svd(A, compute_uv=False)

print(json.dumps({
    "matmul_00": float(AA[0,0]),
    "matmul_11": float(AA[1,1]),
    "matmul_22": float(AA[2,2]),
    "matmul_33": float(AA[3,3]),
    "svd_sigma0": float(s[0]),
    "svd_sigma1": float(s[1]),
    "svd_sigma2": float(s[2]),
    "svd_sigma3": float(s[3])
}))
