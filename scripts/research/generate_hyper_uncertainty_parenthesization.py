#!/usr/bin/env python3
"""Generate deterministic uncertainty-parenthesization research artifacts.

This script mirrors the Cayley-Dickson sign recursion used by
`self-hosted/ir/algebra.sio::ir_cd_sigma` and evaluates first-order GUM
propagation through non-associative hypercomplex products.

Key outputs:
- Octonion trace-cost check: for full covariance propagation, `trace(Sigma)`
  is parenthesization-invariant across `(a*b)*c` and `a*(b*c)` in every sampled
  case, matching the composition-algebra intuition.
- Octonion observation-specific costs: component variance and projected scalar
  variance do depend on parenthesization, so an observation-aware objective is
  still meaningful.
- Sedenion trace-cost check: `trace(Sigma)` is not invariant and produces
  genuine left-vs-right winners.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import time
from typing import Dict, List, Sequence, Tuple


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(
    ROOT, "artifacts", "research", "hyper_uncertainty_parenthesization.v1.json"
)


Vector = List[float]
Matrix = List[List[float]]


def git_value(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except Exception:
        return "unknown"


def cd_sigma(a: int, b: int, bits: int) -> int:
    """Exact transcription of self-hosted/ir/algebra.sio::ir_cd_sigma."""
    if a == 0 or b == 0:
        return 1
    if bits <= 1:
        return -1
    half = 1 << (bits - 1)
    a_hi = 1 if a >= half else 0
    b_hi = 1 if b >= half else 0
    a_lo = a & (half - 1)
    b_lo = b & (half - 1)
    if a_hi == 0 and b_hi == 0:
        return cd_sigma(a_lo, b_lo, bits - 1)
    if a_hi == 0 and b_hi == 1:
        return cd_sigma(b_lo, a_lo, bits - 1)
    if a_hi == 1 and b_hi == 0:
        if b_lo == 0:
            return cd_sigma(a_lo, b_lo, bits - 1)
        return -cd_sigma(a_lo, b_lo, bits - 1)
    if b_lo == 0:
        return -cd_sigma(b_lo, a_lo, bits - 1)
    return cd_sigma(b_lo, a_lo, bits - 1)


def basis_mul(i: int, j: int, bits: int) -> Tuple[int, int]:
    return cd_sigma(i, j, bits), i ^ j


def hyper_mul(bits: int, x: Vector, y: Vector) -> Vector:
    dim = 1 << bits
    out = [0.0] * dim
    for i, x_i in enumerate(x):
        if abs(x_i) < 1e-15:
            continue
        for j, y_j in enumerate(y):
            if abs(y_j) < 1e-15:
                continue
            sign, basis = basis_mul(i, j, bits)
            out[basis] += sign * x_i * y_j
    return out


def left_jacobian(bits: int, rhs_mean: Vector) -> Matrix:
    dim = 1 << bits
    jac = zero_matrix(dim)
    for i in range(dim):
        basis = [0.0] * dim
        basis[i] = 1.0
        col = hyper_mul(bits, basis, rhs_mean)
        for row in range(dim):
            jac[row][i] = col[row]
    return jac


def right_jacobian(bits: int, lhs_mean: Vector) -> Matrix:
    dim = 1 << bits
    jac = zero_matrix(dim)
    for j in range(dim):
        basis = [0.0] * dim
        basis[j] = 1.0
        col = hyper_mul(bits, lhs_mean, basis)
        for row in range(dim):
            jac[row][j] = col[row]
    return jac


def zero_matrix(dim: int) -> Matrix:
    return [[0.0 for _ in range(dim)] for _ in range(dim)]


def diag_matrix(diagonal: Vector) -> Matrix:
    dim = len(diagonal)
    return [[diagonal[i] if i == j else 0.0 for j in range(dim)] for i in range(dim)]


def transpose(mat: Matrix) -> Matrix:
    return [list(col) for col in zip(*mat)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    rows = len(a)
    inner = len(b)
    cols = len(b[0]) if b else 0
    out = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            a_ik = a[i][k]
            if abs(a_ik) < 1e-15:
                continue
            row = out[i]
            b_k = b[k]
            for j in range(cols):
                row[j] += a_ik * b_k[j]
    return out


def mat_add(a: Matrix, b: Matrix) -> Matrix:
    dim = len(a)
    return [[a[i][j] + b[i][j] for j in range(dim)] for i in range(dim)]


def trace(mat: Matrix) -> float:
    return sum(mat[i][i] for i in range(len(mat)))


def frob_identity_deviation(mat: Matrix, scalar: float) -> float:
    dim = len(mat)
    err_sq = 0.0
    for i in range(dim):
        for j in range(dim):
            target = scalar if i == j else 0.0
            delta = mat[i][j] - target
            err_sq += delta * delta
    return math.sqrt(err_sq)


def vector_norm_sq(v: Vector) -> float:
    return sum(x * x for x in v)


def quadratic_form(mat: Matrix, q: Vector) -> float:
    dim = len(q)
    acc = 0.0
    for i in range(dim):
        row_dot = 0.0
        for j in range(dim):
            row_dot += mat[i][j] * q[j]
        acc += q[i] * row_dot
    return acc


def round_vector(v: Vector, digits: int = 6) -> Vector:
    return [round(x, digits) for x in v]


def round_matrix_diag(mat: Matrix, digits: int = 6) -> Vector:
    return [round(mat[i][i], digits) for i in range(len(mat))]


def covariance_propagate(bits: int, mean_x: Vector, cov_x: Matrix, mean_y: Vector, cov_y: Matrix) -> Tuple[Vector, Matrix]:
    jx = left_jacobian(bits, mean_y)
    jy = right_jacobian(bits, mean_x)
    left_term = matmul(matmul(jx, cov_x), transpose(jx))
    right_term = matmul(matmul(jy, cov_y), transpose(jy))
    return hyper_mul(bits, mean_x, mean_y), mat_add(left_term, right_term)


def evaluate_triple(bits: int, a_mean: Vector, a_cov: Matrix, b_mean: Vector, b_cov: Matrix, c_mean: Vector, c_cov: Matrix) -> Dict[str, object]:
    ab_mean, ab_cov = covariance_propagate(bits, a_mean, a_cov, b_mean, b_cov)
    left_mean, left_cov = covariance_propagate(bits, ab_mean, ab_cov, c_mean, c_cov)

    bc_mean, bc_cov = covariance_propagate(bits, b_mean, b_cov, c_mean, c_cov)
    right_mean, right_cov = covariance_propagate(bits, a_mean, a_cov, bc_mean, bc_cov)

    return {
        "left": {
            "mean": left_mean,
            "cov": left_cov,
            "trace": trace(left_cov),
            "diag": [left_cov[i][i] for i in range(len(left_cov))],
        },
        "right": {
            "mean": right_mean,
            "cov": right_cov,
            "trace": trace(right_cov),
            "diag": [right_cov[i][i] for i in range(len(right_cov))],
        },
    }


def random_vector(rng: random.Random, dim: int) -> Vector:
    return [round(rng.uniform(-1.0, 1.0), 3) for _ in range(dim)]


def random_diag_cov(rng: random.Random, dim: int) -> Matrix:
    return diag_matrix([round(rng.uniform(0.001, 0.2), 3) for _ in range(dim)])


def sample_inputs(bits: int, seed: int, trial: int) -> Tuple[Vector, Matrix, Vector, Matrix, Vector, Matrix]:
    dim = 1 << bits
    rng = random.Random(seed + 104729 * trial + 17 * bits)
    a_mean = random_vector(rng, dim)
    b_mean = random_vector(rng, dim)
    c_mean = random_vector(rng, dim)
    a_cov = random_diag_cov(rng, dim)
    b_cov = random_diag_cov(rng, dim)
    c_cov = random_diag_cov(rng, dim)
    return a_mean, a_cov, b_mean, b_cov, c_mean, c_cov


def format_example(bits: int, trial: int, component: int | None, projection: Vector | None, inputs: Tuple[Vector, Matrix, Vector, Matrix, Vector, Matrix], result: Dict[str, object]) -> Dict[str, object]:
    a_mean, a_cov, b_mean, b_cov, c_mean, c_cov = inputs
    out: Dict[str, object] = {
        "bits": bits,
        "trial": trial,
        "a_mean": round_vector(a_mean),
        "b_mean": round_vector(b_mean),
        "c_mean": round_vector(c_mean),
        "a_diag_cov": round_matrix_diag(a_cov),
        "b_diag_cov": round_matrix_diag(b_cov),
        "c_diag_cov": round_matrix_diag(c_cov),
        "left_trace": round(float(result["left"]["trace"]), 9),
        "right_trace": round(float(result["right"]["trace"]), 9),
        "left_diag": round_vector(result["left"]["diag"]),
        "right_diag": round_vector(result["right"]["diag"]),
    }
    if component is not None:
        out["component_index"] = component
        out["left_component_variance"] = round(float(result["left"]["diag"][component]), 9)
        out["right_component_variance"] = round(float(result["right"]["diag"][component]), 9)
        out["winner"] = "left" if result["left"]["diag"][component] < result["right"]["diag"][component] else "right"
    if projection is not None:
        left_proj = quadratic_form(result["left"]["cov"], projection)
        right_proj = quadratic_form(result["right"]["cov"], projection)
        out["projection"] = round_vector(projection)
        out["left_projection_variance"] = round(left_proj, 9)
        out["right_projection_variance"] = round(right_proj, 9)
        out["winner"] = "left" if left_proj < right_proj else "right"
    return out


def verify_octonion_scaled_orthogonality(trials: int = 32, seed: int = 91) -> Dict[str, object]:
    dim = 8
    rng = random.Random(seed)
    max_left_dev = 0.0
    max_right_dev = 0.0
    for _ in range(trials):
        mean = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
        norm_sq = vector_norm_sq(mean)
        left = left_jacobian(3, mean)
        right = right_jacobian(3, mean)
        left_dev = frob_identity_deviation(matmul(transpose(left), left), norm_sq)
        right_dev = frob_identity_deviation(matmul(transpose(right), right), norm_sq)
        max_left_dev = max(max_left_dev, left_dev)
        max_right_dev = max(max_right_dev, right_dev)
    return {
        "trials": trials,
        "max_left_frobenius_deviation": max_left_dev,
        "max_right_frobenius_deviation": max_right_dev,
    }


def verify_octonion_trace_invariance(trials: int = 128, seed: int = 211) -> Dict[str, object]:
    max_gap = 0.0
    worst_trial = -1
    for trial in range(trials):
        inputs = sample_inputs(3, seed, trial)
        result = evaluate_triple(3, *inputs)
        gap = abs(result["left"]["trace"] - result["right"]["trace"])
        if gap > max_gap:
            max_gap = gap
            worst_trial = trial
    return {
        "trials": trials,
        "max_trace_gap": max_gap,
        "worst_trial": worst_trial,
        "trace_objective_invariant": max_gap < 1e-8,
    }


def find_octonion_component_examples(seed: int = 503, max_trials: int = 4096) -> Dict[str, object]:
    left_example = None
    right_example = None
    projection_example = None
    for trial in range(max_trials):
        inputs = sample_inputs(3, seed, trial)
        result = evaluate_triple(3, *inputs)

        left_diag = result["left"]["diag"]
        right_diag = result["right"]["diag"]

        if left_example is None:
            for idx in range(8):
                if left_diag[idx] + 1e-8 < right_diag[idx]:
                    left_example = format_example(3, trial, idx, None, inputs, result)
                    break

        if right_example is None:
            for idx in range(8):
                if right_diag[idx] + 1e-8 < left_diag[idx]:
                    right_example = format_example(3, trial, idx, None, inputs, result)
                    break

        if projection_example is None:
            rng = random.Random(seed + 1000003 * trial)
            projection = random_vector(rng, 8)
            left_proj = quadratic_form(result["left"]["cov"], projection)
            right_proj = quadratic_form(result["right"]["cov"], projection)
            if abs(left_proj - right_proj) > 1e-6:
                projection_example = format_example(3, trial, None, projection, inputs, result)

        if left_example is not None and right_example is not None and projection_example is not None:
            break

    return {
        "left_component_example": left_example,
        "right_component_example": right_example,
        "projection_example": projection_example,
    }


def find_sedenion_trace_examples(seed: int = 877, max_trials: int = 256) -> Dict[str, object]:
    left_example = None
    right_example = None
    max_gap = 0.0
    max_gap_trial = -1
    for trial in range(max_trials):
        inputs = sample_inputs(4, seed, trial)
        result = evaluate_triple(4, *inputs)
        diff = result["left"]["trace"] - result["right"]["trace"]
        abs_diff = abs(diff)
        if abs_diff > max_gap:
            max_gap = abs_diff
            max_gap_trial = trial

        if left_example is None and diff < -1e-8:
            left_example = format_example(4, trial, None, None, inputs, result)
            left_example["winner"] = "left"
            left_example["trace_gap"] = round(diff, 9)

        if right_example is None and diff > 1e-8:
            right_example = format_example(4, trial, None, None, inputs, result)
            right_example["winner"] = "right"
            right_example["trace_gap"] = round(diff, 9)

        if left_example is not None and right_example is not None:
            break

    return {
        "max_trace_gap_in_sample": max_gap,
        "max_gap_trial": max_gap_trial,
        "left_trace_example": left_example,
        "right_trace_example": right_example,
    }


def build_report() -> Dict[str, object]:
    orthogonality = verify_octonion_scaled_orthogonality()
    oct_trace = verify_octonion_trace_invariance()
    oct_examples = find_octonion_component_examples()
    sed_examples = find_sedenion_trace_examples()

    return {
        "schema": "sounio.research.hyper-uncertainty-parenthesization.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {
            "commit": git_value(["rev-parse", "HEAD"]),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"]),
        },
        "source_model": {
            "sign_recursion": "self-hosted/ir/algebra.sio::ir_cd_sigma",
            "uncertainty_model": "First-order GUM covariance propagation with independent leaf covariances",
            "leaf_covariance_model": "Diagonal covariance at leaves; full covariance retained after each bilinear product",
            "objective_note": "For octonions, trace(Sigma) is sampled as invariant; observation-specific variance remains parenthesization-sensitive.",
        },
        "octonion": {
            "scaled_orthogonality_check": orthogonality,
            "trace_invariance_check": oct_trace,
            "observation_sensitive_examples": oct_examples,
        },
        "sedenion": {
            "trace_sensitivity_examples": sed_examples,
        },
        "compiler_implications": [
            "Octonion cost extraction should not optimize raw trace(Sigma); all legal bracketings tie under this objective in the sampled first-order model.",
            "Octonion uncertainty optimization becomes meaningful only after specifying an observation functional, such as a component, projection, or downstream scalar readout.",
            "Sedenion trace(Sigma) is genuinely parenthesization-sensitive, so a total-variance objective is plausible there.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT, help="output JSON path")
    args = parser.parse_args()

    report = build_report()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")

    oct_trace = report["octonion"]["trace_invariance_check"]
    sed_trace = report["sedenion"]["trace_sensitivity_examples"]
    print("hyper uncertainty parenthesization artifact written")
    print(f"  out:            {out_path}")
    print(f"  oct max trace:  {oct_trace['max_trace_gap']:.12f}")
    print(f"  sed max trace:  {sed_trace['max_trace_gap_in_sample']:.12f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
