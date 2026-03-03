#!/usr/bin/env python3
"""cuda_plain_gemm_baseline.py — plain SGEMM PTX baseline (no epistemic shadow registers).

Generates and dispatches a minimal tiled SGEMM kernel using the same CUDA Driver
API infrastructure, same grid/block layout (16x16 threads, 64x64 tiles), same
matrix sizes, and same CUDA event timing as cuda_gemm_dispatch.py.

The ONLY difference from the epistemic kernel: no shadow registers, no epsilon
propagation, no validity predicates, no provenance tracking. This gives a
clean ablation for quantifying epistemic overhead.

Usage (on GPU node):
    python3 scripts/cuda_plain_gemm_baseline.py
    GEMM_DIMS=1024,2048,4096,8192 python3 scripts/cuda_plain_gemm_baseline.py
"""

import ctypes
import json
import math
import os
import random
import struct
import sys
import time

# ── CUDA driver API (same as cuda_gemm_dispatch.py) ──────────────────────────

CUdevice    = ctypes.c_int
CUcontext   = ctypes.c_void_p
CUmodule    = ctypes.c_void_p
CUfunction  = ctypes.c_void_p
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
    raise RuntimeError("libcuda.so.1 not found.")


def _cu(fn, *args):
    ret = fn(*args)
    if ret != 0:
        raise RuntimeError(f"{fn.__name__} returned {ret:#010x}")


# ── Plain SGEMM PTX ─────────────────────────────────────────────────────────
#
# Matches the epistemic kernel's signature and launch config exactly:
#   .entry epistemic_gemm(a_ptr, b_ptr, c_ptr, result_ptr, alpha, beta, M, N, K)
#   grid  = (ceil(N/64), ceil(M/64), 1)
#   block = (16, 16, 1)
#
# Each thread computes a 4x4 tile of C using register blocking.
# No shared memory (keeping it simple to match the epistemic kernel's
# register-pressure profile). The goal is measuring the overhead of the
# 4 shadow registers, not tuning an optimal SGEMM.

PLAIN_GEMM_PTX = """\
.version 7.0
.target sm_70
.address_size 64

.visible .entry epistemic_gemm(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u64 result_ptr,
    .param .f32 alpha,
    .param .f32 beta,
    .param .u32 param_M,
    .param .u32 param_N,
    .param .u32 param_K
)
{
    // Registers
    .reg .u64 %ra, %rb, %rc, %rr;
    .reg .f32 %alpha, %beta;
    .reg .u32 %M, %N, %K;
    .reg .u32 %tid_x, %tid_y, %bid_x, %bid_y;
    .reg .u32 %row, %col, %k;
    .reg .u32 %r0, %r1, %r2, %r3;
    .reg .u64 %addr_a, %addr_b, %addr_r;
    .reg .u64 %off64;
    .reg .u32 %off32;
    .reg .f32 %a_val, %b_val;
    .reg .f32 %acc0, %acc1, %acc2, %acc3;
    .reg .f32 %acc4, %acc5, %acc6, %acc7;
    .reg .f32 %acc8, %acc9, %acc10, %acc11;
    .reg .f32 %acc12, %acc13, %acc14, %acc15;
    .reg .pred %p_row, %p_col, %p_loop, %p_r0, %p_r1, %p_r2, %p_r3;
    .reg .pred %p_c0, %p_c1, %p_c2, %p_c3;
    .reg .u32 %row_i, %col_j;

    // Load params
    ld.param.u64 %ra,    [a_ptr];
    ld.param.u64 %rb,    [b_ptr];
    ld.param.u64 %rc,    [c_ptr];
    ld.param.u64 %rr,    [result_ptr];
    ld.param.f32 %alpha, [alpha];
    ld.param.f32 %beta,  [beta];
    ld.param.u32 %M,     [param_M];
    ld.param.u32 %N,     [param_N];
    ld.param.u32 %K,     [param_K];

    // Thread/block IDs
    mov.u32 %tid_x, %tid.x;
    mov.u32 %tid_y, %tid.y;
    mov.u32 %bid_x, %ctaid.x;
    mov.u32 %bid_y, %ctaid.y;

    // Each thread computes a 4x4 sub-tile
    // row = bid_y * 64 + tid_y * 4
    // col = bid_x * 64 + tid_x * 4
    shl.b32 %r0, %bid_y, 6;        // bid_y * 64
    shl.b32 %r1, %tid_y, 2;        // tid_y * 4
    add.u32 %row, %r0, %r1;

    shl.b32 %r0, %bid_x, 6;        // bid_x * 64
    shl.b32 %r1, %tid_x, 2;        // tid_x * 4
    add.u32 %col, %r0, %r1;

    // Initialize 4x4 accumulators to zero
    mov.f32 %acc0, 0f00000000;
    mov.f32 %acc1, 0f00000000;
    mov.f32 %acc2, 0f00000000;
    mov.f32 %acc3, 0f00000000;
    mov.f32 %acc4, 0f00000000;
    mov.f32 %acc5, 0f00000000;
    mov.f32 %acc6, 0f00000000;
    mov.f32 %acc7, 0f00000000;
    mov.f32 %acc8, 0f00000000;
    mov.f32 %acc9, 0f00000000;
    mov.f32 %acc10, 0f00000000;
    mov.f32 %acc11, 0f00000000;
    mov.f32 %acc12, 0f00000000;
    mov.f32 %acc13, 0f00000000;
    mov.f32 %acc14, 0f00000000;
    mov.f32 %acc15, 0f00000000;

    // K loop
    mov.u32 %k, 0;
LOOP_K:
    setp.lt.u32 %p_loop, %k, %K;
    @!%p_loop bra LOOP_END;

    // For each of 4 rows in the sub-tile:
    //   a_val = A[row+i, k]   (row-major: offset = (row+i)*K + k)
    //   For each of 4 cols:
    //     b_val = B[k, col+j]  (row-major: offset = k*N + col+j)
    //     acc[i*4+j] += a_val * b_val

    // Row 0: A[row+0, k]
    add.u32 %row_i, %row, 0;
    setp.lt.u32 %p_r0, %row_i, %M;
    @!%p_r0 bra SKIP_ROW0;
    mul.wide.u32 %off64, %row_i, %K;
    cvt.u64.u32 %addr_a, %k;
    add.u64 %addr_a, %off64, %addr_a;
    shl.b64 %addr_a, %addr_a, 2;
    add.u64 %addr_a, %ra, %addr_a;
    ld.global.f32 %a_val, [%addr_a];

    // Col 0
    add.u32 %col_j, %col, 0;
    setp.lt.u32 %p_c0, %col_j, %N;
    @!%p_c0 bra SKIP_R0C0;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc0, %a_val, %b_val, %acc0;
SKIP_R0C0:
    // Col 1
    add.u32 %col_j, %col, 1;
    setp.lt.u32 %p_c1, %col_j, %N;
    @!%p_c1 bra SKIP_R0C1;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc1, %a_val, %b_val, %acc1;
SKIP_R0C1:
    // Col 2
    add.u32 %col_j, %col, 2;
    setp.lt.u32 %p_c2, %col_j, %N;
    @!%p_c2 bra SKIP_R0C2;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc2, %a_val, %b_val, %acc2;
SKIP_R0C2:
    // Col 3
    add.u32 %col_j, %col, 3;
    setp.lt.u32 %p_c3, %col_j, %N;
    @!%p_c3 bra SKIP_R0C3;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc3, %a_val, %b_val, %acc3;
SKIP_R0C3:
SKIP_ROW0:

    // Row 1: A[row+1, k]
    add.u32 %row_i, %row, 1;
    setp.lt.u32 %p_r1, %row_i, %M;
    @!%p_r1 bra SKIP_ROW1;
    mul.wide.u32 %off64, %row_i, %K;
    cvt.u64.u32 %addr_a, %k;
    add.u64 %addr_a, %off64, %addr_a;
    shl.b64 %addr_a, %addr_a, 2;
    add.u64 %addr_a, %ra, %addr_a;
    ld.global.f32 %a_val, [%addr_a];

    add.u32 %col_j, %col, 0;
    setp.lt.u32 %p_c0, %col_j, %N;
    @!%p_c0 bra SKIP_R1C0;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc4, %a_val, %b_val, %acc4;
SKIP_R1C0:
    add.u32 %col_j, %col, 1;
    setp.lt.u32 %p_c1, %col_j, %N;
    @!%p_c1 bra SKIP_R1C1;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc5, %a_val, %b_val, %acc5;
SKIP_R1C1:
    add.u32 %col_j, %col, 2;
    setp.lt.u32 %p_c2, %col_j, %N;
    @!%p_c2 bra SKIP_R1C2;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc6, %a_val, %b_val, %acc6;
SKIP_R1C2:
    add.u32 %col_j, %col, 3;
    setp.lt.u32 %p_c3, %col_j, %N;
    @!%p_c3 bra SKIP_R1C3;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc7, %a_val, %b_val, %acc7;
SKIP_R1C3:
SKIP_ROW1:

    // Row 2: A[row+2, k]
    add.u32 %row_i, %row, 2;
    setp.lt.u32 %p_r2, %row_i, %M;
    @!%p_r2 bra SKIP_ROW2;
    mul.wide.u32 %off64, %row_i, %K;
    cvt.u64.u32 %addr_a, %k;
    add.u64 %addr_a, %off64, %addr_a;
    shl.b64 %addr_a, %addr_a, 2;
    add.u64 %addr_a, %ra, %addr_a;
    ld.global.f32 %a_val, [%addr_a];

    add.u32 %col_j, %col, 0;
    setp.lt.u32 %p_c0, %col_j, %N;
    @!%p_c0 bra SKIP_R2C0;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc8, %a_val, %b_val, %acc8;
SKIP_R2C0:
    add.u32 %col_j, %col, 1;
    setp.lt.u32 %p_c1, %col_j, %N;
    @!%p_c1 bra SKIP_R2C1;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc9, %a_val, %b_val, %acc9;
SKIP_R2C1:
    add.u32 %col_j, %col, 2;
    setp.lt.u32 %p_c2, %col_j, %N;
    @!%p_c2 bra SKIP_R2C2;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc10, %a_val, %b_val, %acc10;
SKIP_R2C2:
    add.u32 %col_j, %col, 3;
    setp.lt.u32 %p_c3, %col_j, %N;
    @!%p_c3 bra SKIP_R2C3;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc11, %a_val, %b_val, %acc11;
SKIP_R2C3:
SKIP_ROW2:

    // Row 3: A[row+3, k]
    add.u32 %row_i, %row, 3;
    setp.lt.u32 %p_r3, %row_i, %M;
    @!%p_r3 bra SKIP_ROW3;
    mul.wide.u32 %off64, %row_i, %K;
    cvt.u64.u32 %addr_a, %k;
    add.u64 %addr_a, %off64, %addr_a;
    shl.b64 %addr_a, %addr_a, 2;
    add.u64 %addr_a, %ra, %addr_a;
    ld.global.f32 %a_val, [%addr_a];

    add.u32 %col_j, %col, 0;
    setp.lt.u32 %p_c0, %col_j, %N;
    @!%p_c0 bra SKIP_R3C0;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc12, %a_val, %b_val, %acc12;
SKIP_R3C0:
    add.u32 %col_j, %col, 1;
    setp.lt.u32 %p_c1, %col_j, %N;
    @!%p_c1 bra SKIP_R3C1;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc13, %a_val, %b_val, %acc13;
SKIP_R3C1:
    add.u32 %col_j, %col, 2;
    setp.lt.u32 %p_c2, %col_j, %N;
    @!%p_c2 bra SKIP_R3C2;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc14, %a_val, %b_val, %acc14;
SKIP_R3C2:
    add.u32 %col_j, %col, 3;
    setp.lt.u32 %p_c3, %col_j, %N;
    @!%p_c3 bra SKIP_R3C3;
    mul.wide.u32 %off64, %k, %N;
    cvt.u64.u32 %addr_b, %col_j;
    add.u64 %addr_b, %off64, %addr_b;
    shl.b64 %addr_b, %addr_b, 2;
    add.u64 %addr_b, %rb, %addr_b;
    ld.global.f32 %b_val, [%addr_b];
    fma.rn.f32 %acc15, %a_val, %b_val, %acc15;
SKIP_R3C3:
SKIP_ROW3:

    add.u32 %k, %k, 1;
    bra LOOP_K;
LOOP_END:

    // Store 4x4 result tile to result_ptr (row-major: offset = row_i*N + col_j)
    // Row 0
    add.u32 %row_i, %row, 0;
    setp.lt.u32 %p_r0, %row_i, %M;
    @!%p_r0 bra STORE_DONE;

    add.u32 %col_j, %col, 0;
    setp.lt.u32 %p_c0, %col_j, %N;
    @!%p_c0 bra STORE_R0C1;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc0;
STORE_R0C1:
    add.u32 %col_j, %col, 1;
    setp.lt.u32 %p_c1, %col_j, %N;
    @!%p_c1 bra STORE_R0C2;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc1;
STORE_R0C2:
    add.u32 %col_j, %col, 2;
    setp.lt.u32 %p_c2, %col_j, %N;
    @!%p_c2 bra STORE_R0C3;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc2;
STORE_R0C3:
    add.u32 %col_j, %col, 3;
    setp.lt.u32 %p_c3, %col_j, %N;
    @!%p_c3 bra STORE_ROW1;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc3;

    // Row 1
STORE_ROW1:
    add.u32 %row_i, %row, 1;
    setp.lt.u32 %p_r1, %row_i, %M;
    @!%p_r1 bra STORE_DONE;
    add.u32 %col_j, %col, 0;
    setp.lt.u32 %p_c0, %col_j, %N;
    @!%p_c0 bra STORE_R1C1;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc4;
STORE_R1C1:
    add.u32 %col_j, %col, 1;
    setp.lt.u32 %p_c1, %col_j, %N;
    @!%p_c1 bra STORE_R1C2;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc5;
STORE_R1C2:
    add.u32 %col_j, %col, 2;
    setp.lt.u32 %p_c2, %col_j, %N;
    @!%p_c2 bra STORE_R1C3;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc6;
STORE_R1C3:
    add.u32 %col_j, %col, 3;
    setp.lt.u32 %p_c3, %col_j, %N;
    @!%p_c3 bra STORE_ROW2;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc7;

    // Row 2
STORE_ROW2:
    add.u32 %row_i, %row, 2;
    setp.lt.u32 %p_r2, %row_i, %M;
    @!%p_r2 bra STORE_DONE;
    add.u32 %col_j, %col, 0;
    setp.lt.u32 %p_c0, %col_j, %N;
    @!%p_c0 bra STORE_R2C1;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc8;
STORE_R2C1:
    add.u32 %col_j, %col, 1;
    setp.lt.u32 %p_c1, %col_j, %N;
    @!%p_c1 bra STORE_R2C2;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc9;
STORE_R2C2:
    add.u32 %col_j, %col, 2;
    setp.lt.u32 %p_c2, %col_j, %N;
    @!%p_c2 bra STORE_R2C3;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc10;
STORE_R2C3:
    add.u32 %col_j, %col, 3;
    setp.lt.u32 %p_c3, %col_j, %N;
    @!%p_c3 bra STORE_ROW3;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc11;

    // Row 3
STORE_ROW3:
    add.u32 %row_i, %row, 3;
    setp.lt.u32 %p_r3, %row_i, %M;
    @!%p_r3 bra STORE_DONE;
    add.u32 %col_j, %col, 0;
    setp.lt.u32 %p_c0, %col_j, %N;
    @!%p_c0 bra STORE_R3C1;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc12;
STORE_R3C1:
    add.u32 %col_j, %col, 1;
    setp.lt.u32 %p_c1, %col_j, %N;
    @!%p_c1 bra STORE_R3C2;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc13;
STORE_R3C2:
    add.u32 %col_j, %col, 2;
    setp.lt.u32 %p_c2, %col_j, %N;
    @!%p_c2 bra STORE_R3C3;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc14;
STORE_R3C3:
    add.u32 %col_j, %col, 3;
    setp.lt.u32 %p_c3, %col_j, %N;
    @!%p_c3 bra STORE_DONE;
    mul.wide.u32 %off64, %row_i, %N;
    cvt.u64.u32 %addr_r, %col_j;
    add.u64 %addr_r, %off64, %addr_r;
    shl.b64 %addr_r, %addr_r, 2;
    add.u64 %addr_r, %rr, %addr_r;
    st.global.f32 [%addr_r], %acc15;

STORE_DONE:
    ret;
}
"""

# ── Dispatch (reuse pattern from cuda_gemm_dispatch.py) ──────────────────────

def run_plain_gemm(cuda, ptx_str, M, N, K, n_iters):
    mod = CUmodule(0)
    ptx_bytes = ptx_str.encode() + b"\x00"
    _cu(cuda.cuModuleLoadData, ctypes.byref(mod), ptx_bytes)

    fn = CUfunction(0)
    _cu(cuda.cuModuleGetFunction, ctypes.byref(fn), mod, b"epistemic_gemm")

    size_A = M * K * 4
    size_B = K * N * 4
    size_C = M * N * 4

    d_A = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_A), size_A)
    d_B = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_B), size_B)
    d_C = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_C), size_C)
    d_R = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_R), size_C)

    rng = random.Random(42)
    h_A = struct.pack(f"{M*K}f", *[rng.gauss(0, 1) for _ in range(M * K)])
    h_B = struct.pack(f"{K*N}f", *[rng.gauss(0, 1) for _ in range(K * N)])
    h_C = b"\x00" * size_C

    _cu(cuda.cuMemcpyHtoD, d_A, ctypes.c_char_p(h_A), size_A)
    _cu(cuda.cuMemcpyHtoD, d_B, ctypes.c_char_p(h_B), size_B)
    _cu(cuda.cuMemcpyHtoD, d_C, ctypes.c_char_p(h_C), size_C)
    _cu(cuda.cuMemcpyHtoD, d_R, ctypes.c_char_p(h_C), size_C)

    grid_x = (N + 63) // 64
    grid_y = (M + 63) // 64

    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)
    Mc = ctypes.c_uint32(M)
    Nc = ctypes.c_uint32(N)
    Kc = ctypes.c_uint32(K)

    args = (ctypes.c_void_p * 9)(
        ctypes.cast(ctypes.byref(d_A), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_B), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_C), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(d_R), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(alpha), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(beta), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(Mc), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(Nc), ctypes.c_void_p),
        ctypes.cast(ctypes.byref(Kc), ctypes.c_void_p),
    )

    def launch():
        _cu(cuda.cuLaunchKernel, fn,
            grid_x, grid_y, 1, 16, 16, 1, 0, None, args, None)

    # Warmup
    launch()
    _cu(cuda.cuCtxSynchronize)

    # Timed
    ev_start = CUevent(0); _cu(cuda.cuEventCreate, ctypes.byref(ev_start), 0)
    ev_stop = CUevent(0); _cu(cuda.cuEventCreate, ctypes.byref(ev_stop), 0)

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

    r0 = ctypes.c_float(0.0)
    _cu(cuda.cuMemcpyDtoH, ctypes.byref(r0), d_R, 4)

    cuda.cuEventDestroy(ev_start); cuda.cuEventDestroy(ev_stop)
    cuda.cuMemFree(d_A); cuda.cuMemFree(d_B)
    cuda.cuMemFree(d_C); cuda.cuMemFree(d_R)
    cuda.cuModuleUnload(mod)

    return {
        "gflops": round(gflops, 1),
        "ms_per_iter": round(ms_per_iter, 4),
        "result_0": r0.value,
    }


# ── Memory bandwidth ─────────────────────────────────────────────────────────

def measure_bandwidth(cuda, size_mb=256):
    size_bytes = size_mb * 1024 * 1024
    n_iters = 20
    d_src = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_src), size_bytes)
    d_dst = CUdeviceptr(0); _cu(cuda.cuMemAlloc, ctypes.byref(d_dst), size_bytes)
    _cu(cuda.cuMemcpyDtoD, d_dst, d_src, size_bytes)
    _cu(cuda.cuCtxSynchronize)
    ev_start = CUevent(0); _cu(cuda.cuEventCreate, ctypes.byref(ev_start), 0)
    ev_stop = CUevent(0); _cu(cuda.cuEventCreate, ctypes.byref(ev_stop), 0)
    _cu(cuda.cuEventRecord, ev_start, None)
    for _ in range(n_iters):
        _cu(cuda.cuMemcpyDtoD, d_dst, d_src, size_bytes)
    _cu(cuda.cuEventRecord, ev_stop, None)
    _cu(cuda.cuEventSynchronize, ev_stop)
    ms = ctypes.c_float(0.0)
    _cu(cuda.cuEventElapsedTime, ctypes.byref(ms), ev_start, ev_stop)
    gb_s = (size_bytes * n_iters) / (ms.value * 1e-3) / 1e9
    cuda.cuEventDestroy(ev_start); cuda.cuEventDestroy(ev_stop)
    cuda.cuMemFree(d_src); cuda.cuMemFree(d_dst)
    return round(gb_s, 1)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    dims_str = os.environ.get("GEMM_DIMS", "1024,2048,4096,8192")
    dims = [int(d.strip()) for d in dims_str.split(",")]
    n_iters = int(os.environ.get("GEMM_ITERS", 8))
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "artifacts", "omega", "plain_gemm_baseline_report.v1.json")

    cuda = _load_libcuda()
    device = CUdevice(0)
    _cu(cuda.cuDeviceGet, ctypes.byref(device), 0)
    name_buf = ctypes.create_string_buffer(256)
    cuda.cuDeviceGetName(name_buf, 256, device)
    gpu_name = name_buf.value.decode()

    ctx = CUcontext(0)
    _cu(cuda.cuCtxCreate, ctypes.byref(ctx), 0, device)

    print(f"[souc-gpu] plain SGEMM baseline (no shadow registers) on {gpu_name}")
    print(f"  PTX: inline {len(PLAIN_GEMM_PTX)} chars, .target sm_70")
    print(f"  grid/block: same as epistemic (64x64 tiles, 16x16 threads, 4x4 per thread)")
    print(f"  dims: {dims}")
    print(f"  iters: {n_iters}")
    print()

    bw = measure_bandwidth(cuda)
    print(f"[souc-gpu] memory bandwidth: {bw} GB/s (DtoD, 256 MB)")
    print()

    results = {}
    for dim in dims:
        M = N = K = dim
        print(f"=== plain SGEMM {M}x{N}x{K} (iters={n_iters}) ===")
        res = run_plain_gemm(cuda, PLAIN_GEMM_PTX, M, N, K, n_iters)
        results[str(dim)] = res
        print("=======================================================================")
        print(f"  [souc-gpu] plain_sgemm {M}x{N}x{K}")
        print(f"  GPU:         {gpu_name}")
        print(f"  GFLOPS:      {res['gflops']}")
        print(f"  ms/iter:     {res['ms_per_iter']}")
        print(f"  result[0]:   {res['result_0']:.6f}")
        print("=======================================================================")
        print()

    report = {
        "schema": "sounio.benchmark.plain-gemm-baseline.v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu_name,
        "n_iters": n_iters,
        "bandwidth_gb_s": bw,
        "ptx_target": "sm_70",
        "ptx_chars": len(PLAIN_GEMM_PTX),
        "note": "Same kernel signature, grid, block, and tile sizes as epistemic GEMM. "
                "No shadow registers, no epsilon propagation, no validity predicates, "
                "no provenance. Pure FMA inner loop.",
        "dimensions": results,
    }

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[souc-gpu] report written to {report_path}")

    cuda.cuCtxDestroy(ctx)


if __name__ == "__main__":
    main()
