"""Benchmark: PK Uncertainty Propagation — Python equivalent.

Sobol-style variance decomposition for PK parameter sensitivity.
"""
import time
from dataclasses import dataclass


@dataclass
class PKParameter:
    code: int
    value: float
    std_uncert: float
    sensitivity: float
    contribution: float


def make_pk_param(code, value, std_uncert, sensitivity):
    contribution = sensitivity**2 * std_uncert**2
    return PKParameter(code, value, std_uncert, sensitivity, contribution)


CL_CODES = {0, 3, 6, 7}
VOL_CODES = {1, 6}
ABS_CODES = {2, 4}


def analyze_pk(params: list[PKParameter]):
    total_var = sum(p.contribution for p in params)
    cl_var = sum(p.contribution for p in params if p.code in CL_CODES)
    vol_var = sum(p.contribution for p in params if p.code in VOL_CODES)
    abs_var = sum(p.contribution for p in params if p.code in ABS_CODES)

    cl_frac = cl_var / total_var if total_var > 0 else 0.0
    vol_frac = vol_var / total_var if total_var > 0 else 0.0
    abs_frac = abs_var / total_var if total_var > 0 else 0.0

    dominant_idx = max(range(len(params)), key=lambda i: params[i].contribution)
    is_cl_dominant = cl_frac > 0.7

    return {
        "cl_frac": cl_frac,
        "vol_frac": vol_frac,
        "abs_frac": abs_frac,
        "dominant_idx": dominant_idx,
        "is_cl_dominant": is_cl_dominant,
        "total_var": total_var,
    }


def main():
    params = [
        make_pk_param(0, 5.0, 1.5, 0.9),   # CL
        make_pk_param(1, 50.0, 5.0, 0.2),   # V
        make_pk_param(2, 1.2, 0.3, 0.15),   # ka
        make_pk_param(3, 0.8, 0.05, 0.1),   # F
        make_pk_param(4, 0.5, 0.1, 0.05),   # t_lag
        make_pk_param(7, 1.2, 0.15, 0.3),   # SCr
        make_pk_param(6, 70.0, 10.0, 0.12), # Weight
        make_pk_param(5, 0.0, 0.0, 0.0),    # Other
    ]

    n_iters = 10_000

    t0 = time.perf_counter()
    for _ in range(n_iters):
        result = analyze_pk(params)
    t1 = time.perf_counter()

    ms = (t1 - t0) * 1000
    cl = result["cl_frac"]
    print(f"PK uncertainty ({n_iters} iters): CL contribution = {cl:.6f}, time = {ms:.2f} ms")
    if result["is_cl_dominant"]:
        print("Recommendation: TDM (clearance dominant)")


if __name__ == "__main__":
    main()
