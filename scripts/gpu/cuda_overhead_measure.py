#!/usr/bin/env python3
"""cuda_overhead_measure.py — measure epistemic GEMM shadow register overhead.

Runs cuBLAS SGEMM baseline and Sounio epistemic GEMM on the same GPU,
same dimensions, same timing methodology. Computes per-dimension overhead ratio.

Usage (on GPU node):
    cd ~/work/sounio
    python3 scripts/cuda_overhead_measure.py

    # With pre-generated PTX:
    GEMM_PTX_FILE=/tmp/epistemic_gemm_sm7_4096.ptx \\
        python3 scripts/cuda_overhead_measure.py

    # Custom iterations:
    GEMM_ITERS=12 python3 scripts/cuda_overhead_measure.py

Output: artifacts/omega/overhead_report.v1.json
"""

import json
import math
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DIMS = [1024, 2048, 4096, 8192]
N_ITERS = int(os.environ.get("GEMM_ITERS", 8))
PTX_FILE = os.environ.get("GEMM_PTX_FILE", "")
EXTRA_1024_RUNS = int(os.environ.get("EXTRA_1024_RUNS", 8))


def run_script(script, env_overrides=None):
    """Run a Python script and capture stdout."""
    env = {**os.environ}
    if env_overrides:
        env.update(env_overrides)
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, script)],
        capture_output=True, text=True, env=env, cwd=ROOT_DIR,
    )
    if result.returncode != 0:
        print(f"ERROR running {script}:", file=sys.stderr)
        print(result.stderr[:2000], file=sys.stderr)
        return None, result.stdout + result.stderr
    return result.stdout, result.stderr


def parse_gflops(output, tag="cublas_sgemm"):
    """Extract GFLOPS from [souc-gpu] tagged output lines."""
    results = {}
    current_dim = None
    for line in output.splitlines():
        if tag in line and "x" in line:
            # Parse dimension from line like: [souc-gpu] cublas_sgemm 4096x4096x4096
            parts = line.split()
            for p in parts:
                if "x" in p and p[0].isdigit():
                    dim = int(p.split("x")[0])
                    current_dim = dim
                    break
        if "GFLOPS:" in line and current_dim is not None:
            gflops = float(line.split("GFLOPS:")[1].strip())
            results[current_dim] = gflops
            current_dim = None
    return results


def parse_bandwidth(output):
    """Extract bandwidth from [souc-gpu] memory bandwidth line."""
    for line in output.splitlines():
        if "memory bandwidth" in line and "GB/s" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "GB/s" and i > 0:
                    return float(parts[i - 1])
    return None


def main():
    report_path = os.path.join(ROOT_DIR, "artifacts", "omega", "overhead_report.v1.json")

    print("=" * 72)
    print("  Sounio Epistemic GEMM Overhead Measurement")
    print("=" * 72)
    print()

    # ── Phase 1: cuBLAS baseline ──────────────────────────────────────────
    print("[1/3] Running cuBLAS SGEMM baseline...")
    cublas_env = {
        "GEMM_DIMS": ",".join(str(d) for d in DIMS),
        "GEMM_ITERS": str(N_ITERS),
    }
    cublas_out, _ = run_script("cuda_cublas_baseline.py", cublas_env)
    if cublas_out is None:
        print("FATAL: cuBLAS baseline failed", file=sys.stderr)
        sys.exit(1)

    print(cublas_out)
    cublas_gflops = parse_gflops(cublas_out, tag="cublas_sgemm")
    bandwidth = parse_bandwidth(cublas_out)
    print(f"  cuBLAS results: {cublas_gflops}")
    print(f"  Bandwidth: {bandwidth} GB/s")
    print()

    # ── Phase 2: Epistemic GEMM ───────────────────────────────────────────
    print("[2/3] Running epistemic GEMM for each dimension...")
    epistemic_gflops = {}

    for dim in DIMS:
        n_runs = EXTRA_1024_RUNS if dim == 1024 else 1
        dim_results = []

        for run_i in range(n_runs):
            env = {
                "GEMM_M": str(dim),
                "GEMM_N": str(dim),
                "GEMM_K": str(dim),
                "GEMM_ITERS": str(N_ITERS),
            }
            if PTX_FILE:
                env["GEMM_PTX_FILE"] = PTX_FILE

            suffix = f" (run {run_i+1}/{n_runs})" if n_runs > 1 else ""
            print(f"  Epistemic GEMM {dim}x{dim}x{dim}{suffix}...")
            out, _ = run_script("cuda_gemm_dispatch.py", env)
            if out is None:
                print(f"  WARNING: epistemic GEMM {dim} failed", file=sys.stderr)
                continue

            parsed = parse_gflops(out, tag="epistemic_gemm")
            if dim in parsed:
                dim_results.append(parsed[dim])
                print(f"    -> {parsed[dim]} GFLOPS")
            else:
                print(f"    -> parse failed")

        if dim_results:
            dim_results.sort()
            median_idx = len(dim_results) // 2
            epistemic_gflops[dim] = {
                "median": dim_results[median_idx],
                "mean": round(sum(dim_results) / len(dim_results), 1),
                "min": min(dim_results),
                "max": max(dim_results),
                "n": len(dim_results),
                "all": dim_results,
            }

    print()

    # ── Phase 3: Compute overhead ─────────────────────────────────────────
    print("[3/3] Computing overhead ratios...")
    print()

    overhead_data = {}
    overhead_ratios = []

    print(f"{'Dim':>6}  {'cuBLAS':>10}  {'Epistemic':>10}  {'Overhead':>10}  {'Extra %':>8}")
    print("-" * 52)

    for dim in DIMS:
        if dim not in cublas_gflops or dim not in epistemic_gflops:
            continue

        cb = cublas_gflops[dim]
        ep = epistemic_gflops[dim]["median"]
        ratio = cb / ep if ep > 0 else float("inf")
        pct = (ratio - 1.0) * 100

        overhead_data[str(dim)] = {
            "cublas_gflops": cb,
            "epistemic_gflops_median": ep,
            "epistemic_n": epistemic_gflops[dim]["n"],
            "overhead_x": round(ratio, 3),
            "overhead_pct": round(pct, 1),
        }
        overhead_ratios.append(ratio)

        print(f"{dim:>6}  {cb:>10.1f}  {ep:>10.1f}  {ratio:>9.2f}x  {pct:>7.1f}%")

    # Geometric mean
    if overhead_ratios:
        geomean = math.exp(sum(math.log(r) for r in overhead_ratios) / len(overhead_ratios))
    else:
        geomean = float("nan")

    print("-" * 52)
    print(f"{'geomean':>6}  {'':>10}  {'':>10}  {geomean:>9.2f}x  {(geomean-1)*100:>7.1f}%")
    print()

    # ── Write report ──────────────────────────────────────────────────────
    report = {
        "schema": "sounio.benchmark.overhead-report.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_iters": N_ITERS,
        "ptx_file": PTX_FILE or "(generated on-the-fly)",
        "dimensions": overhead_data,
        "aggregate": {
            "geomean_overhead_x": round(geomean, 3),
            "geomean_overhead_pct": round((geomean - 1.0) * 100, 1),
            "bandwidth_gb_s": bandwidth,
        },
        "1024_extra_runs": epistemic_gflops.get(1024, {}),
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report written to {report_path}")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  OVERHEAD SUMMARY: {geomean:.2f}x geometric mean")
    print(f"  ({(geomean-1)*100:.1f}% more time for 4 shadow registers per element)")
    if bandwidth:
        print(f"  Memory bandwidth: {bandwidth} GB/s")
    print("=" * 72)


if __name__ == "__main__":
    main()
