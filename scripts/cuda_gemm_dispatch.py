#!/usr/bin/env python3
"""cuda_gemm_dispatch.py — dispatch epistemic GEMM PTX on local CUDA GPU.

Uses CUDA Driver API (libcuda.so.1) via ctypes only — no pycuda, no nvcc,
no CUDA toolkit required. Just libcuda.so.1 on PATH.

Usage (on GPU node):
    cd ~/work/sounio
    python3 scripts/cuda_gemm_dispatch.py            # M=N=K=1024
    GEMM_M=4096 GEMM_N=4096 GEMM_K=4096 \\
        python3 scripts/cuda_gemm_dispatch.py        # 4096^3

The PTX is extracted from:
    cargo test --release --features gpu -p souc --lib epistemic_gemm -- --nocapture
"""

import ctypes
import ctypes.util
import os
import random
import struct
import subprocess
import sys
import time

# ── CUDA driver API via ctypes ────────────────────────────────────────────────

def _load_libcuda():
    for name in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
        try:
            lib = ctypes.CDLL(name)
            lib.cuInit(0)
            return lib
        except (OSError, AttributeError):
            pass
    raise RuntimeError("libcuda.so.1 not found. Is the NVIDIA driver installed?")

cuda = _load_libcuda()

# Pointer helpers
CUdevice   = ctypes.c_int
CUcontext  = ctypes.c_void_p
CUmodule   = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUdeviceptr= ctypes.c_uint64
CUevent    = ctypes.c_void_p


def _cu(fn, *args):
    """Call a CUDA driver function; raise on non-zero return."""
    ret = fn(*args)
    if ret != 0:
        raise RuntimeError(f"{fn.__name__} returned {ret:#010x}")


def _ptr(ctype, val=None):
    """Allocate a mutable pointer of given ctype."""
    return ctypes.byref(ctype(0 if val is None else val))


# ── PTX generation ────────────────────────────────────────────────────────────

def generate_ptx(M: int, N: int, K: int) -> str:
    """Run souc GPU unit tests and extract the PTX string."""
    env = {**os.environ, "GEMM_M": str(M), "GEMM_N": str(N), "GEMM_K": str(K)}
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Locate cargo: try PATH, then common rustup locations
    cargo = "cargo"
    for candidate in ("cargo", os.path.expanduser("~/.cargo/bin/cargo"),
                      "/usr/local/cargo/bin/cargo"):
        if os.path.isfile(candidate) or candidate == "cargo":
            cargo = candidate
            break
    cmd = [
        cargo, "test", "--release", "--features", "gpu", "-p", "souc",
        "--lib", "print_epistemic_gemm_ptx", "--", "--nocapture",
    ]
    print(f"  [ptx] running: {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=repo, env=env)
    combined = r.stdout + r.stderr

    # Fenced marker
    if "--- PTX BEGIN ---" in combined and "--- PTX END ---" in combined:
        a = combined.index("--- PTX BEGIN ---") + len("--- PTX BEGIN ---")
        b = combined.index("--- PTX END ---")
        return combined[a:b].strip()

    # Heuristic: first .version block → closing }
    lines = combined.splitlines()
    ptx_lines, in_ptx, depth = [], False, 0
    for line in lines:
        if not in_ptx and ".version" in line:
            in_ptx = True
        if in_ptx:
            ptx_lines.append(line)
            depth += line.count("{") - line.count("}")
            if depth <= 0 and len(ptx_lines) > 10:
                break

    if ptx_lines:
        return "\n".join(ptx_lines)

    # Last resort: look for the tiled GEMM PTX in test output
    # The Rust test print_epistemic_gemm_ptx might not exist; embed fallback.
    raise RuntimeError(
        "Could not extract PTX from souc test output.\n"
        f"stdout ({len(r.stdout)} chars):\n{r.stdout[:600]}\n"
        f"stderr ({len(r.stderr)} chars):\n{r.stderr[:600]}"
    )


def load_ptx_file(path: str) -> str:
    with open(path) as f:
        raw = f.read()
    # Strip gen_ptx fence markers if present
    if "--- PTX BEGIN ---" in raw and "--- PTX END ---" in raw:
        a = raw.index("--- PTX BEGIN ---") + len("--- PTX BEGIN ---")
        b = raw.index("--- PTX END ---")
        return raw[a:b].strip()
    return raw.strip()


# ── CUDA dispatch ─────────────────────────────────────────────────────────────

def run_gemm(ptx: str, M: int, N: int, K: int, n_iters: int = 5) -> dict:
    """Compile PTX and dispatch epistemic GEMM on the local GPU."""

    # Init
    _cu(cuda.cuInit, 0)

    device = CUdevice(0)
    _cu(cuda.cuDeviceGet, ctypes.byref(device), 0)

    name_buf = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(name_buf, 256, device)
    gpu_name = name_buf.value.decode()

    ctx = CUcontext(0)
    _cu(cuda.cuCtxCreate, ctypes.byref(ctx), 0, device)

    # Load PTX (JIT) or pre-compiled CUBIN
    cubin_file = os.environ.get("GEMM_CUBIN_FILE", "")
    mod = CUmodule(0)
    if cubin_file:
        _cu(cuda.cuModuleLoad, ctypes.byref(mod), cubin_file.encode())
    else:
        # Strip gen_ptx fence markers before JIT compile
        if "--- PTX BEGIN ---" in ptx and "--- PTX END ---" in ptx:
            a = ptx.index("--- PTX BEGIN ---") + len("--- PTX BEGIN ---")
            b = ptx.index("--- PTX END ---")
            ptx = ptx[a:b].strip()
        ptx_bytes = ptx.encode() + b"\x00"
        _cu(cuda.cuModuleLoadData, ctypes.byref(mod), ptx_bytes)

    fn = CUfunction(0)
    _cu(cuda.cuModuleGetFunction, ctypes.byref(fn), mod, b"epistemic_gemm")

    # Allocate device memory (use random-ish values via struct.pack)
    size_A = M * K * 4  # f32
    size_B = K * N * 4
    size_C = M * N * 4

    d_A = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_A), size_A)
    d_B = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_B), size_B)
    d_C = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_C), size_C)
    d_R = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_R), size_C)

    # Fill A, B with small random f32 via struct; fill C with zeros
    rng = random.Random(42)

    def make_f32_buf(n):
        return struct.pack(f"{n}f", *[rng.gauss(0, 1) for _ in range(n)])

    h_A = make_f32_buf(M * K)
    h_B = make_f32_buf(K * N)
    h_C = b"\x00" * size_C

    _cu(cuda.cuMemcpyHtoD, d_A, ctypes.c_char_p(h_A), size_A)
    _cu(cuda.cuMemcpyHtoD, d_B, ctypes.c_char_p(h_B), size_B)
    _cu(cuda.cuMemcpyHtoD, d_C, ctypes.c_char_p(h_C), size_C)
    _cu(cuda.cuMemcpyHtoD, d_R, ctypes.c_char_p(h_C), size_C)

    # Launch config: tile 64×64 per block, thread block 16×16
    grid_x = (N + 63) // 64
    grid_y = (M + 63) // 64

    # Kernel args: a_ptr, b_ptr, c_ptr, result_ptr, alpha, beta, M, N, K
    alpha = ctypes.c_float(1.0)
    beta  = ctypes.c_float(0.0)
    Mc    = ctypes.c_uint32(M)
    Nc    = ctypes.c_uint32(N)
    Kc    = ctypes.c_uint32(K)

    # cuLaunchKernel needs void** array of arg pointers
    # We use a ctypes array of c_void_p pointing to each arg
    args = (ctypes.c_void_p * 9)(
        ctypes.cast(ctypes.byref(d_A),    ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_B),    ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_C),    ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_R),    ctypes.c_void_p),
        ctypes.cast(ctypes.byref(alpha),  ctypes.c_void_p),
        ctypes.cast(ctypes.byref(beta),   ctypes.c_void_p),
        ctypes.cast(ctypes.byref(Mc),     ctypes.c_void_p),
        ctypes.cast(ctypes.byref(Nc),     ctypes.c_void_p),
        ctypes.cast(ctypes.byref(Kc),     ctypes.c_void_p),
    )

    def launch():
        _cu(cuda.cuLaunchKernel,
            fn,
            grid_x, grid_y, 1,   # grid
            16,     16,     1,   # block
            0,                   # shared memory bytes
            None,                # stream (default)
            args,                # kernel params
            None)                # extra

    # Warmup
    launch()
    _cu(cuda.cuCtxSynchronize)

    # CUDA events for timing
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

    flops  = 2.0 * M * N * K
    gflops = flops / (ms_per_iter * 1e-3) / 1e9

    # Read back first element for sanity
    r0 = ctypes.c_float(0.0)
    _cu(cuda.cuMemcpyDtoH, ctypes.byref(r0), d_R, 4)

    # Cleanup
    cuda.cuMemFree(d_A); cuda.cuMemFree(d_B)
    cuda.cuMemFree(d_C); cuda.cuMemFree(d_R)
    cuda.cuModuleUnload(mod)
    cuda.cuCtxDestroy(ctx)

    return {
        "gpu":          gpu_name,
        "gflops":       round(gflops, 1),
        "ms_per_iter":  round(ms_per_iter, 2),
        "M": M, "N": N, "K": K,
        "grid":         f"({grid_x}, {grid_y}, 1)",
        "block":        "(16, 16, 1)",
        "result_r0":    r0.value,
        "n_iters":      n_iters,
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    M = int(os.environ.get("GEMM_M", 1024))
    N = int(os.environ.get("GEMM_N", M))
    K = int(os.environ.get("GEMM_K", M))
    n_iters = int(os.environ.get("GEMM_ITERS", 5))

    ptx_file = os.environ.get("GEMM_PTX_FILE", "")

    print(f"[souc-gpu] epistemic GEMM dispatch on local CUDA device")
    print(f"  dim:   {M}×{N}×{K}")

    if ptx_file:
        print(f"  ptx:   loading from {ptx_file}")
        ptx = load_ptx_file(ptx_file)
    else:
        print(f"  ptx:   generating via cargo test ...")
        try:
            ptx = generate_ptx(M, N, K)
        except RuntimeError as e:
            print(f"\nPTX generation failed: {e}")
            sys.exit(1)
    print(f"  ptx:   {len(ptx)} chars")

    print(f"  iters: {n_iters}")
    print()

    try:
        res = run_gemm(ptx, M, N, K, n_iters)
    except RuntimeError as e:
        print(f"CUDA dispatch failed: {e}")
        sys.exit(1)

    TARGET = 500.0
    print("=======================================================================")
    print(f"  [souc-gpu] epistemic_gemm {res['M']}×{res['N']}×{res['K']}")
    print(f"  GPU:         {res['gpu']}")
    print(f"  GFLOPS:      {res['gflops']}")
    print(f"  ms/iter:     {res['ms_per_iter']}")
    print(f"  grid:        {res['grid']}")
    print(f"  block:       {res['block']}")
    print(f"  result[0,0]: {res['result_r0']:.6f}")
    print("=======================================================================")

    if res["gflops"] >= TARGET:
        print(f"  STATUS: PASS (>= {TARGET} GFLOPS)")
    else:
        gap = TARGET - res["gflops"]
        print(f"  STATUS: BELOW TARGET (< {TARGET} GFLOPS, -{gap:.0f} gap)")
        print()
        print("  Tuning levers:")
        print("    • Use --features gpu sm_80 path (cp.async double-buffer)")
        print("    • Increase TILE_K from 16 to 32 in epistemic_gemm.rs")
        print("    • Check smem_a/smem_b bank conflicts (padding)")
        print("    • Try GEMM_M=2048 for better occupancy measurement")
        sys.exit(1)


if __name__ == "__main__":
    main()
