#!/usr/bin/env python3
"""cuda_cublas_baseline.py — cuBLAS SGEMM baseline on local CUDA GPU.

Uses CUDA Runtime API (cudart) + cuBLAS via ctypes only — no pycuda, no pip packages.
Same timing methodology as cuda_gemm_dispatch.py for apples-to-apples comparison
with the Sounio epistemic GEMM kernel.

Key: uses cudart (runtime API) throughout instead of CUDA Driver API to avoid
the runtime/driver context conflict that occurs when cuBLAS (which requires cudart)
is mixed with cuCtxCreate.

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

# ── cudart constants ─────────────────────────────────────────────────────────

cudaMemcpyHostToDevice   = 1
cudaMemcpyDeviceToHost   = 2
cudaMemcpyDeviceToDevice = 3

# ── Library loading ──────────────────────────────────────────────────────────


def _find_lib(names, search_dirs=None):
    """Try loading a shared library by name, then search common directories."""
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            pass

    dirs = list(search_dirs or [])
    dirs.extend(["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"])
    home = os.path.expanduser("~")
    for conda_root in (os.path.join(home, "miniconda3"), os.path.join(home, "anaconda3")):
        dirs.append(os.path.join(conda_root, "targets", "x86_64-linux", "lib"))
        dirs.append(os.path.join(conda_root, "lib"))
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if d:
            dirs.append(d)

    base = names[0].split(".so")[0] if names else "libunknown"
    import glob as _glob
    for prefix in dirs:
        if not os.path.isdir(prefix):
            continue
        # Try versioned .so files
        for f in sorted(_glob.glob(os.path.join(prefix, base + ".so*")), reverse=True):
            try:
                return ctypes.CDLL(f)
            except OSError:
                pass
    return None


def _load_cudart():
    """Load CUDA runtime library."""
    lib = _find_lib(["libcudart.so", "libcudart.so.13", "libcudart.so.12"])
    if lib is None:
        raise RuntimeError("libcudart.so not found. Is the CUDA toolkit installed?")
    return lib


def _load_cublas():
    """Load cuBLAS library."""
    lib = _find_lib(["libcublas.so", "libcublas.so.13", "libcublas.so.12", "libcublas.so.11"])
    if lib is None:
        raise RuntimeError("libcublas.so not found. Is the CUDA toolkit installed?")
    return lib


def _load_libcuda():
    """Load CUDA driver API (only for cuDeviceGetName — no context creation)."""
    for name in ("libcuda.so.1", "libcuda.so"):
        try:
            lib = ctypes.CDLL(name)
            lib.cuInit(0)
            return lib
        except (OSError, AttributeError):
            pass
    return None  # Non-fatal: we can get GPU name from nvidia-smi instead


def _rc(fn, *args):
    """Call a CUDA runtime function and check return code."""
    ret = fn(*args)
    if ret != 0:
        raise RuntimeError(f"{fn.__name__} returned {ret}")


# ── GPU name helper ──────────────────────────────────────────────────────────

def _get_gpu_name(cuda_driver=None):
    """Get GPU name via driver API or nvidia-smi fallback."""
    if cuda_driver is not None:
        try:
            device = ctypes.c_int(0)
            cuda_driver.cuDeviceGet(ctypes.byref(device), 0)
            name_buf = ctypes.create_string_buffer(256)
            cuda_driver.cuDeviceGetName(name_buf, 256, device)
            return name_buf.value.decode()
        except Exception:
            pass
    # Fallback: nvidia-smi
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True, timeout=5,
        )
        return out.strip().split("\n")[0]
    except Exception:
        return "Unknown GPU"


# ── Benchmark routines ───────────────────────────────────────────────────────

def run_cublas_sgemm(cudart, cublas, M, N, K, n_iters):
    """Run cuBLAS SGEMM and return GFLOPS with CUDA event timing."""

    size_A = M * K * 4
    size_B = K * N * 4
    size_C = M * N * 4

    # Allocate device memory via cudaMalloc
    d_A = ctypes.c_void_p(0)
    d_B = ctypes.c_void_p(0)
    d_C = ctypes.c_void_p(0)
    _rc(cudart.cudaMalloc, ctypes.byref(d_A), ctypes.c_size_t(size_A))
    _rc(cudart.cudaMalloc, ctypes.byref(d_B), ctypes.c_size_t(size_B))
    _rc(cudart.cudaMalloc, ctypes.byref(d_C), ctypes.c_size_t(size_C))

    # Fill with same random seed as cuda_gemm_dispatch.py
    rng = random.Random(42)
    h_A = struct.pack(f"{M*K}f", *[rng.gauss(0, 1) for _ in range(M * K)])
    h_B = struct.pack(f"{K*N}f", *[rng.gauss(0, 1) for _ in range(K * N)])
    h_C = b"\x00" * size_C

    _rc(cudart.cudaMemcpy, d_A, ctypes.c_char_p(h_A), ctypes.c_size_t(size_A), cudaMemcpyHostToDevice)
    _rc(cudart.cudaMemcpy, d_B, ctypes.c_char_p(h_B), ctypes.c_size_t(size_B), cudaMemcpyHostToDevice)
    _rc(cudart.cudaMemcpy, d_C, ctypes.c_char_p(h_C), ctypes.c_size_t(size_C), cudaMemcpyHostToDevice)

    # cuBLAS handle
    handle = ctypes.c_void_p(0)
    rc = cublas.cublasCreate_v2(ctypes.byref(handle))
    if rc != 0:
        raise RuntimeError(f"cublasCreate_v2 returned {rc}")

    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)
    CUBLAS_OP_N = 0

    def launch():
        rc = cublas.cublasSgemm_v2(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            ctypes.byref(alpha),
            d_A, M,
            d_B, K,
            ctypes.byref(beta),
            d_C, M,
        )
        if rc != 0:
            raise RuntimeError(f"cublasSgemm_v2 returned {rc}")

    # Warmup
    launch()
    _rc(cudart.cudaDeviceSynchronize)

    # Timed iterations via cudaEvent
    ev_start = ctypes.c_void_p(0)
    ev_stop = ctypes.c_void_p(0)
    _rc(cudart.cudaEventCreate, ctypes.byref(ev_start))
    _rc(cudart.cudaEventCreate, ctypes.byref(ev_stop))

    _rc(cudart.cudaEventRecord, ev_start, None)
    for _ in range(n_iters):
        launch()
    _rc(cudart.cudaEventRecord, ev_stop, None)
    _rc(cudart.cudaEventSynchronize, ev_stop)

    ms = ctypes.c_float(0.0)
    _rc(cudart.cudaEventElapsedTime, ctypes.byref(ms), ev_start, ev_stop)
    ms_per_iter = ms.value / n_iters

    flops = 2.0 * M * N * K
    gflops = flops / (ms_per_iter * 1e-3) / 1e9

    # Read back first element
    r0 = ctypes.c_float(0.0)
    _rc(cudart.cudaMemcpy, ctypes.byref(r0), d_C, ctypes.c_size_t(4), cudaMemcpyDeviceToHost)

    # Cleanup
    cudart.cudaEventDestroy(ev_start)
    cudart.cudaEventDestroy(ev_stop)
    cublas.cublasDestroy_v2(handle)
    cudart.cudaFree(d_A)
    cudart.cudaFree(d_B)
    cudart.cudaFree(d_C)

    return {
        "gflops": round(gflops, 1),
        "ms_per_iter": round(ms_per_iter, 4),
        "result_0": r0.value,
        "n_iters": n_iters,
    }


def measure_bandwidth(cudart, size_mb=256):
    """Measure device-to-device memory bandwidth via cudaMemcpy DtoD."""
    size_bytes = size_mb * 1024 * 1024
    n_iters = 20

    d_src = ctypes.c_void_p(0)
    d_dst = ctypes.c_void_p(0)
    _rc(cudart.cudaMalloc, ctypes.byref(d_src), ctypes.c_size_t(size_bytes))
    _rc(cudart.cudaMalloc, ctypes.byref(d_dst), ctypes.c_size_t(size_bytes))

    # Warmup
    _rc(cudart.cudaMemcpy, d_dst, d_src, ctypes.c_size_t(size_bytes), cudaMemcpyDeviceToDevice)
    _rc(cudart.cudaDeviceSynchronize)

    ev_start = ctypes.c_void_p(0)
    ev_stop = ctypes.c_void_p(0)
    _rc(cudart.cudaEventCreate, ctypes.byref(ev_start))
    _rc(cudart.cudaEventCreate, ctypes.byref(ev_stop))

    _rc(cudart.cudaEventRecord, ev_start, None)
    for _ in range(n_iters):
        _rc(cudart.cudaMemcpy, d_dst, d_src, ctypes.c_size_t(size_bytes), cudaMemcpyDeviceToDevice)
    _rc(cudart.cudaEventRecord, ev_stop, None)
    _rc(cudart.cudaEventSynchronize, ev_stop)

    ms = ctypes.c_float(0.0)
    _rc(cudart.cudaEventElapsedTime, ctypes.byref(ms), ev_start, ev_stop)

    gb_s = (size_bytes * n_iters) / (ms.value * 1e-3) / 1e9

    cudart.cudaEventDestroy(ev_start)
    cudart.cudaEventDestroy(ev_stop)
    cudart.cudaFree(d_src)
    cudart.cudaFree(d_dst)

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

    # Load libraries — cudart first, then cuBLAS (both use runtime context)
    cudart = _load_cudart()
    _rc(cudart.cudaSetDevice, 0)

    # Get GPU name via driver API (no context creation) or nvidia-smi
    cuda_driver = _load_libcuda()
    gpu_name = _get_gpu_name(cuda_driver)

    cublas = _load_cublas()

    print(f"[souc-gpu] cuBLAS SGEMM baseline on {gpu_name}")
    print(f"  dims:  {dims}")
    print(f"  iters: {n_iters}")
    print()

    # Memory bandwidth
    bw = measure_bandwidth(cudart)
    print(f"[souc-gpu] memory bandwidth: {bw} GB/s (DtoD, 256 MB)")
    print()

    results = {}
    for dim in dims:
        M = N = K = dim
        print(f"=== cuBLAS SGEMM {M}x{N}x{K} (iters={n_iters}) ===")
        res = run_cublas_sgemm(cudart, cublas, M, N, K, n_iters)
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


if __name__ == "__main__":
    main()
