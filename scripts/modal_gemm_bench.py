#!/usr/bin/env python3
"""scripts/modal_gemm_bench.py — dispatch Sounio epistemic GEMM PTX on Modal.com GPU

Usage:
    # Generate PTX from local souc binary, run on Modal L4
    modal run scripts/modal_gemm_bench.py

    # Override GPU type and matrix size
    MODAL_GPU=A10G GEMM_M=2048 GEMM_N=2048 GEMM_K=2048 \\
        modal run scripts/modal_gemm_bench.py

    # Load pre-generated PTX from file (skip local souc build)
    modal run scripts/modal_gemm_bench.py -- --ptx-file /tmp/gemm.ptx

    # RunPod: export PTX then SSH it over manually (see --export-ptx)
    modal run scripts/modal_gemm_bench.py -- --export-ptx /tmp/gemm.ptx

Requirements:
    pip install modal
    modal token new          # one-time auth

Kernel signature expected (from generate_epistemic_gemm_ptx_sm):
    .visible .entry epistemic_gemm(
        .param .u64 a_ptr,      // A matrix (M×K, f32, row-major)
        .param .u64 b_ptr,      // B matrix (K×N, f32, row-major)
        .param .u64 c_ptr,      // C matrix (M×N, f32, for beta*C term)
        .param .u64 result_ptr, // output   (M×N, f32)
        .param .f32 alpha,
        .param .f32 beta,
        .param .u32 M,
        .param .u32 N,
        .param .u32 K
    )

Launch config: block=(16,16,1), grid=(ceil(N/64), ceil(M/64), 1)
Target: >= 500 GFLOPS sustained on L4 / A10G / A100
"""

import argparse
import os
import subprocess
import sys

import modal

# ── Modal app ────────────────────────────────────────────────────────────────

APP_NAME = "sounio-epistemic-gemm"
app = modal.App(APP_NAME)

_CUDA_IMAGE = (
    modal.Image.from_registry(
        "nvidia/cuda:12.2.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install("pycuda==2024.1", "numpy")
)

# ── Remote function ──────────────────────────────────────────────────────────

@app.function(
    gpu=os.environ.get("MODAL_GPU", "L4"),
    image=_CUDA_IMAGE,
    timeout=300,
)
def dispatch_gemm(
    ptx_code: str,
    M: int = 4096,
    N: int = 4096,
    K: int = 4096,
    n_iters: int = 5,
    kernel_name: str = "epistemic_gemm",
) -> dict:
    """Compile and time the epistemic GEMM PTX on a Modal GPU.

    Returns a dict with gflops, ms_per_iter, grid, block, result_sample.
    """
    import numpy as np
    import pycuda.autoinit  # noqa: F401 — initialises CUDA context
    import pycuda.driver as cuda

    # ── Load PTX module ──────────────────────────────────────────────────────
    # module_from_buffer expects null-terminated bytes
    ptx_bytes = ptx_code.encode() + b"\x00"
    mod = cuda.module_from_buffer(ptx_bytes)
    kernel = mod.get_function(kernel_name)

    # ── Allocate device memory ───────────────────────────────────────────────
    dtype = np.float32
    rng = np.random.default_rng(seed=42)
    A = rng.standard_normal((M, K)).astype(dtype)
    B = rng.standard_normal((K, N)).astype(dtype)
    C = np.zeros((M, N), dtype=dtype)
    result_buf = np.zeros((M, N), dtype=dtype)

    d_A      = cuda.mem_alloc(A.nbytes)
    d_B      = cuda.mem_alloc(B.nbytes)
    d_C      = cuda.mem_alloc(C.nbytes)
    d_result = cuda.mem_alloc(result_buf.nbytes)

    cuda.memcpy_htod(d_A, A)
    cuda.memcpy_htod(d_B, B)
    cuda.memcpy_htod(d_C, C)
    cuda.memcpy_htod(d_result, result_buf)

    # ── Launch config ────────────────────────────────────────────────────────
    # Tile: 64×64 per block, thread block: 16×16 = 256 threads
    block = (16, 16, 1)
    grid  = ((N + 63) // 64, (M + 63) // 64, 1)

    def _launch():
        kernel(
            d_A, d_B, d_C, d_result,
            np.float32(1.0), np.float32(0.0),
            np.uint32(M),    np.uint32(N),    np.uint32(K),
            block=block, grid=grid,
        )

    # ── Warmup ───────────────────────────────────────────────────────────────
    _launch()
    cuda.Context.synchronize()

    # ── Timed runs ───────────────────────────────────────────────────────────
    t_start = cuda.Event()
    t_stop  = cuda.Event()
    t_start.record()
    for _ in range(n_iters):
        _launch()
    t_stop.record()
    t_stop.synchronize()

    ms     = t_stop.time_since(t_start) / n_iters  # ms per iteration
    flops  = 2.0 * M * N * K                        # FMA counts as 2 ops
    gflops = flops / (ms * 1e-3) / 1e9

    # ── Spot-check result ─────────────────────────────────────────────────────
    out = np.empty((M, N), dtype=dtype)
    cuda.memcpy_dtoh(out, d_result)

    return {
        "gflops":        round(gflops, 1),
        "ms_per_iter":   round(ms, 2),
        "M": M, "N": N, "K": K,
        "grid":          str(grid),
        "block":         str(block),
        "result_sample": float(out[0, 0]),
        "kernel":        kernel_name,
    }

# ── Local helpers ─────────────────────────────────────────────────────────────

def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_ptx_via_cargo(M: int, N: int, K: int) -> str:
    """Run the souc GPU unit tests with --nocapture and extract the PTX block.

    The Rust test `test_ptx_generation` (or similar) should print the PTX
    to stdout between a '--- PTX BEGIN ---' / '--- PTX END ---' fence.
    Fallback: scan for any '.version' line and collect until the next '}' line.
    """
    repo = _repo_root()
    env  = {**os.environ, "GEMM_M": str(M), "GEMM_N": str(N), "GEMM_K": str(K)}

    # First try a dedicated PTX-dump test if it exists
    result = subprocess.run(
        [
            "cargo", "test", "--release", "--features", "gpu", "-p", "souc",
            "--lib", "print_epistemic_gemm_ptx", "--", "--nocapture",
        ],
        capture_output=True, text=True, cwd=repo, env=env,
    )
    combined = result.stdout + result.stderr

    # Fenced extraction
    if "--- PTX BEGIN ---" in combined and "--- PTX END ---" in combined:
        a = combined.index("--- PTX BEGIN ---") + len("--- PTX BEGIN ---")
        b = combined.index("--- PTX END ---")
        return combined[a:b].strip()

    # Fallback: first .version block
    lines     = combined.splitlines()
    ptx_lines = []
    in_ptx    = False
    brace_depth = 0
    for line in lines:
        if not in_ptx and ".version" in line:
            in_ptx = True
        if in_ptx:
            ptx_lines.append(line)
            brace_depth += line.count("{") - line.count("}")
            # Stop after the closing brace of the last kernel entry
            if brace_depth <= 0 and len(ptx_lines) > 5:
                break

    if ptx_lines:
        return "\n".join(ptx_lines)

    raise RuntimeError(
        "Could not extract PTX from souc cargo test output.\n"
        f"stdout ({len(result.stdout)} chars):\n{result.stdout[:800]}\n"
        f"stderr ({len(result.stderr)} chars):\n{result.stderr[:800]}"
    )


def generate_ptx_via_binary(souc_bin: str, M: int, N: int, K: int) -> str:
    """Call the souc binary with a 'ptx' subcommand (future CLI hook)."""
    result = subprocess.run(
        [souc_bin, "ptx", "--kernel=epistemic_gemm",
         f"--M={M}", f"--N={N}", f"--K={K}"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"souc ptx subcommand failed:\n{result.stderr[:800]}"
        )
    return result.stdout.strip()


# ── Entry point ───────────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    parser = argparse.ArgumentParser(
        description="Run Sounio epistemic GEMM PTX on Modal GPU-as-a-Service"
    )
    parser.add_argument("--M",        type=int, default=int(os.environ.get("GEMM_M", 4096)))
    parser.add_argument("--N",        type=int, default=int(os.environ.get("GEMM_N", 4096)))
    parser.add_argument("--K",        type=int, default=int(os.environ.get("GEMM_K", 4096)))
    parser.add_argument("--iters",    type=int, default=5,
                        help="Timed iterations per benchmark")
    parser.add_argument("--gpu",      default=os.environ.get("MODAL_GPU", "L4"),
                        choices=["L4", "A10G", "A100", "T4", "H100"],
                        help="Modal GPU type")
    parser.add_argument("--souc-bin", default=os.path.join(_repo_root(), "target", "release", "souc"),
                        help="Path to souc binary (for PTX generation)")
    parser.add_argument("--ptx-file", default=None,
                        help="Skip PTX generation; load PTX from this file")
    parser.add_argument("--export-ptx", default=None,
                        help="Generate PTX locally and write to this file (for RunPod manual dispatch)")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K

    # ── PTX acquisition ───────────────────────────────────────────────────────
    if args.ptx_file:
        print(f"[sounio-modal] Loading PTX from {args.ptx_file} ...")
        with open(args.ptx_file) as f:
            ptx = f.read()
        print(f"[sounio-modal] PTX loaded ({len(ptx):,} chars)")
    else:
        print(f"[sounio-modal] Generating epistemic GEMM PTX ({M}×{N}×{K}) via cargo ...")
        try:
            ptx = generate_ptx_via_cargo(M, N, K)
        except RuntimeError:
            # Fallback: try the souc binary's ptx subcommand
            print("[sounio-modal] Fallback: trying souc binary ptx subcommand ...")
            ptx = generate_ptx_via_binary(args.souc_bin, M, N, K)
        print(f"[sounio-modal] PTX generated ({len(ptx):,} chars)")

    # ── Export-only mode (RunPod workflow) ─────────────────────────────────────
    if args.export_ptx:
        with open(args.export_ptx, "w") as f:
            f.write(ptx)
        print(f"[sounio-modal] PTX written to {args.export_ptx}")
        print("[sounio-modal] RunPod workflow:")
        print(f"  scp {args.export_ptx} <runpod-user>@<ip>:/tmp/gemm.ptx")
        print("  # on RunPod: python3 -c \"")
        print("  #   import pycuda.autoinit, pycuda.driver as cuda, numpy as np")
        print("  #   ptx = open('/tmp/gemm.ptx').read().encode() + b'\\x00'")
        print("  #   mod = cuda.module_from_buffer(ptx)")
        print("  #   kernel = mod.get_function('epistemic_gemm')")
        print("  #   # ... allocate, launch, time ...\"")
        return

    # ── Modal dispatch ────────────────────────────────────────────────────────
    print(f"[sounio-modal] Dispatching to Modal {args.gpu} GPU ({args.iters} timed iters) ...")
    res = dispatch_gemm.remote(ptx, M, N, K, args.iters)

    # ── Results ──────────────────────────────────────────────────────────────
    TARGET_GFLOPS = 500.0
    print()
    print("=======================================================================")
    print(f"  [souc-gpu] epistemic_gemm {res['M']}×{res['N']}×{res['K']}")
    print(f"  GFLOPS:      {res['gflops']}")
    print(f"  ms/iter:     {res['ms_per_iter']}")
    print(f"  grid:        {res['grid']}")
    print(f"  block:       {res['block']}")
    print(f"  result[0,0]: {res['result_sample']:.6f}")
    print(f"  kernel:      {res['kernel']}")
    print("=======================================================================")

    if res["gflops"] >= TARGET_GFLOPS:
        print(f"  STATUS: PASS (>= {TARGET_GFLOPS} GFLOPS)")
    else:
        gap = TARGET_GFLOPS - res["gflops"]
        print(f"  STATUS: BELOW TARGET (< {TARGET_GFLOPS} GFLOPS, -{gap:.0f} GFLOPS)")
        print()
        print("  Tuning levers:")
        print("    • Increase TILE_K (shared memory strip width)")
        print("    • Enable cp.async double-buffering (sm_80+ path)")
        print("    • Check register pressure (reduce %f<48> pool)")
        print("    • Try --gpu A10G or --gpu A100 for more bandwidth")
        sys.exit(1)
