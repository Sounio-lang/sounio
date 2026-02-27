#!/usr/bin/env python3
"""cuda_cublas_baseline.py — cuBLAS SGEMM baseline on local CUDA GPU.

Uses CUDA Driver API + cuBLAS via ctypes only — no pycuda, no pip packages.
Same timing methodology as cuda_gemm_dispatch.py for apples-to-apples comparison
with the Sounio epistemic GEMM kernel.

Usage (on GPU node):
    python3 scripts/cuda_cublas_baseline.py                     # default dims
    GEMM_DIMS=1024,2048,4096,8192 python3 scripts/cuda_cublas_baseline.py
    GEMM_ITERS=8 python3 scripts/cuda_cublas_baseline.py        # iterations per dim

Output: per-dimension GFLOPS + memory bandwidth measurement for roofline.
"""

import ctypes
import ctypes.util
import json
import os
import random
import struct
import sys
import time

# ── CUDA driver API via ctypes ────────────────────────────────────────────────

CUdevice    = ctypes.c_int
CUcontext   = ctypes.c_void_p
CUdeviceptr = ctypes.c_uint64
CUevent     = ctypes.c_void_p


def _load_libcuda():
    for name in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
        try:
            lib = ctypes.CDLL(name)
            lib.cuInit(0)
            return lib
        except (OSError, AttributeError):
            pass
    raise RuntimeError("libcuda.so.1 not found. Is the NVIDIA driver installed?")


def _load_libcublas():
    for name in ("libcublas.so", "libcublas.so.12", "libcublas.so.11", "cublas64_12.dll"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass
    # Try CUDA toolkit path
    for prefix in ("/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"):
        for ver in ("12", "11", ""):
            path = os.path.join(prefix, f"libcublas.so{'.'+ver if ver else ''}")
            if os.path.isfile(path):
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    pass
    raise RuntimeError("libcublas.so not found. Is the CUDA toolkit installed?")


def _cu(fn, *args):
    ret = fn(*args)
    if ret != 0:
        raise RuntimeError(f"{fn.__name__} returned {ret:#010x}")


# ── Benchmark routines ───────────────────────────────────────────────────────

def run_cublas_sgemm(cuda, cublas, M, N, K, n_iters):
    """Run cuBLAS SGEMM and return GFLOPS with CUDA event timing."""

    # Allocate device memory
    size_A = M * K * 4
    size_B = K * N * 4
    size_C = M * N * 4

    d_A = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_A), size_A)
    d_B = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_B), size_B)
    d_C = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_C), size_C)

    # Fill with same random seed as cuda_gemm_dispatch.py
    rng = random.Random(42)
    h_A = struct.pack(f"{M*K}f", *[rng.gauss(0, 1) for _ in range(M * K)])
    h_B = struct.pack(f"{K*N}f", *[rng.gauss(0, 1) for _ in range(K * N)])
    h_C = b"\x00" * size_C

    _cu(cuda.cuMemcpyHtoD, d_A, ctypes.c_char_p(h_A), size_A)
    _cu(cuda.cuMemcpyHtoD, d_B, ctypes.c_char_p(h_B), size_B)
    _cu(cuda.cuMemcpyHtoD, d_C, ctypes.c_char_p(h_C), size_C)

    # cuBLAS handle
    handle = ctypes.c_void_p(0)
    rc = cublas.cublasCreate_v2(ctypes.byref(handle))
    if rc != 0:
        raise RuntimeError(f"cublasCreate_v2 returned {rc}")

    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)

    # cuBLAS uses column-major. For row-major A(M,K) * B(K,N) = C(M,N):
    # Compute C^T = B^T * A^T in column-major, which is:
    #   cublasSgemm(N, M, K, alpha, B, K, A, K, beta, C, N)  [with OP_T, OP_T]
    # Or just pass as column-major with OP_N:
    #   cublasSgemm(M, N, K, alpha, A, M, B, K, beta, C, M)
    CUBLAS_OP_N = 0

    def launch():
        rc = cublas.cublasSgemm_v2(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            ctypes.byref(alpha),
            ctypes.c_void_p(d_A.value), M,
            ctypes.c_void_p(d_B.value), K,
            ctypes.byref(beta),
            ctypes.c_void_p(d_C.value), M,
        )
        if rc != 0:
            raise RuntimeError(f"cublasSgemm_v2 returned {rc}")

    # Warmup
    launch()
    _cu(cuda.cuCtxSynchronize)

    # Timed iterations
    ev_start = CUevent(0); _cu(cuda.cuEventCreate, ctypes.byref(ev_start), 0)
    ev_stop  = CUevent(0); _cu(cuda.cuEventCreate, ctypes.byref(ev_stop),  0)

    _cu(cuda.cuEventRecord, ev_start, None)
    for _ in range(n_iters):
        launch()
    _cu(cuda.cuEventRecord, ev_stop, None)
    _cu(cuda.cuEventSynchronize, ev_stop)

    ms = ctypes.c_float(0.0)
    _cu(cuda.cuEventElapsedTime, ctypes.byref(ms), ev_start, ev_stop)
    ms_per_iter = ms.value / n_iters

    flops = 2.0 * M * N * K
    gflops = flops / (ms_per_iter * 1e-3) / 1e9

    # Read back first element
    r0 = ctypes.c_float(0.0)
    _cu(cuda.cuMemcpyDtoH, ctypes.byref(r0), d_C, 4)

    # Cleanup
    cuda.cuEventDestroy(ev_start)
    cuda.cuEventDestroy(ev_stop)
    cublas.cublasDestroy_v2(handle)
    cuda.cuMemFree(d_A); cuda.cuMemFree(d_B); cuda.cuMemFree(d_C)

    return {
        "gflops": round(gflops, 1),
        "ms_per_iter": round(ms_per_iter, 4),
        "result_0": r0.value,
        "n_iters": n_iters,
    }


def measure_bandwidth(cuda, size_mb=256):
    """Measure device-to-device memory bandwidth via cuMemcpyDtoD."""
    size_bytes = size_mb * 1024 * 1024
    n_iters = 20

    d_src = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_src), size_bytes)
    d_dst = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_dst), size_bytes)

    # Warmup
    _cu(cuda.cuMemcpyDtoD, d_dst, d_src, size_bytes)
    _cu(cuda.cuCtxSynchronize)

    ev_start = CUevent(0); _cu(cuda.cuEventCreate, ctypes.byref(ev_start), 0)
    ev_stop  = CUevent(0); _cu(cuda.cuEventCreate, ctypes.byref(ev_stop),  0)

    _cu(cuda.cuEventRecord, ev_start, None)
    for _ in range(n_iters):
        _cu(cuda.cuMemcpyDtoD, d_dst, d_src, size_bytes)
    _cu(cuda.cuEventRecord, ev_stop, None)
    _cu(cuda.cuEventSynchronize, ev_stop)

    ms = ctypes.c_float(0.0)
    _cu(cuda.cuEventElapsedTime, ctypes.byref(ms), ev_start, ev_stop)

    gb_s = (size_bytes * n_iters) / (ms.value * 1e-3) / 1e9

    cuda.cuEventDestroy(ev_start)
    cuda.cuEventDestroy(ev_stop)
    cuda.cuMemFree(d_src); cuda.cuMemFree(d_dst)

    return round(gb_s, 1)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    dims_str = os.environ.get("GEMM_DIMS", "1024,2048,4096,8192")
    dims = [int(d.strip()) for d in dims_str.split(",")]
    n_iters = int(os.environ.get("GEMM_ITERS", 8))
    report_path = os.environ.get(
        "CUBLAS_REPORT",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "artifacts", "omega", "cublas_baseline_report.v1.json")
    )

    cuda = _load_libcuda()
    cublas = _load_libcublas()

    # Init device + context
    device = CUdevice(0)
    _cu(cuda.cuDeviceGet, ctypes.byref(device), 0)

    name_buf = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(name_buf, 256, device)
    gpu_name = name_buf.value.decode()

    ctx = CUcontext(0)
    _cu(cuda.cuCtxCreate, ctypes.byref(ctx), 0, device)

    print(f"[souc-gpu] cuBLAS SGEMM baseline on {gpu_name}")
    print(f"  dims:  {dims}")
    print(f"  iters: {n_iters}")
    print()

    # Memory bandwidth
    bw = measure_bandwidth(cuda)
    print(f"[souc-gpu] memory bandwidth: {bw} GB/s (DtoD, 256 MB)")
    print()

    results = {}
    for dim in dims:
        M = N = K = dim
        print(f"=== cuBLAS SGEMM {M}x{N}x{K} (iters={n_iters}) ===")
        res = run_cublas_sgemm(cuda, cublas, M, N, K, n_iters)
        results[str(dim)] = res

        print("=======================================================================")
        print(f"  [souc-gpu] cublas_sgemm {M}x{N}x{K}")
        print(f"  GPU:         {gpu_name}")
        print(f"  GFLOPS:      {res['gflops']}")
        print(f"  ms/iter:     {res['ms_per_iter']}")
        print(f"  result[0]:   {res['result_0']:.6f}")
        print("=======================================================================")
        print()

    # Write report
    report = {
        "schema": "sounio.benchmark.cublas-baseline.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu_name,
        "n_iters": n_iters,
        "bandwidth_gb_s": bw,
        "dimensions": results,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[souc-gpu] report written to {report_path}")

    # Cleanup
    cuda.cuCtxDestroy(ctx)


if __name__ == "__main__":
    main()
