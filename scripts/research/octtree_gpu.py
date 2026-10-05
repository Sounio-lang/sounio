#!/usr/bin/env python3
"""
GPU OctTree — batched tree-fold on CUDA with optional tensor-core path.

The tree-fold is embarrassingly parallel: each level processes all nodes
simultaneously. On GPU this is a sequence of batched matmuls (for L(a)·b)
and element-wise ops, with zero sequential dependency except across tree
levels (log₂(L) steps).

This module provides:
  - OctTreeGPU: same architecture as OctTreeClassifier but GPU-optimized
  - Batched octonion left-multiplication via precomputed L matrices
  - Optional f64 path for exact associator computation

For eventual Sounio compiler lowering: each tree level is one kernel
launch — the octonion product L(a)·b as a wmma (tensor core) tile.
"""

import torch
import torch.nn as nn
import numpy as np
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from ossm_dyck_scaling import oct_mul_fast, _T, _T_KJ, count_params, train_one, GRUClassifier, OSSMCell
from mpon_dyck_scaling import OctTreeClassifier


class OctTreeGPU(nn.Module):
    """GPU-optimized OctTree with vectorized tree fold.

    Key optimization: precompute all octonion left-multiplication matrices
    for the entire batch at once, then fold via batched matmul.

    The octonion product a⊗b = L(a)·b where L(a) is an 8×8 matrix.
    For a batch of N pairs: L_batch (N, 8, 8) @ b_batch (N, 8, 1) -> (N, 8).
    This is a single bmm — exactly what tensor cores accelerate.
    """
    def __init__(self, vocab_size, dim=8, n_classes=2,
                 use_oct=True, max_levels=14):
        super().__init__()
        self.dim = dim
        self.use_oct = use_oct
        self.max_levels = max_levels

        self.embed = nn.Parameter(torch.randn(vocab_size, dim) * 0.1)
        self.gate_prod = nn.Parameter(torch.ones(max_levels) * 0.5)
        self.gate_res = nn.Parameter(torch.ones(max_levels) * 0.5)
        self.bias = nn.Parameter(torch.zeros(max_levels, dim))
        self.readout = nn.Linear(dim, n_classes)

        # Precompute the T_KJ tensor for L(a) construction
        self.register_buffer('_T_KJ', _T_KJ)

    def _oct_mul_batched(self, a, b):
        """Batched octonion multiply: a (N, 8), b (N, 8) -> (N, 8).
        Builds L(a) for the whole batch via einsum, then bmm.
        """
        N = a.shape[0]
        # L(a)[k, j] = sum_i a[i] * T_KJ[i, k, j]
        # = a @ T_KJ.reshape(8, 64) -> (N, 64) -> reshape (N, 8, 8)
        Tkj = self._T_KJ  # (8, 8, 8): (i, k, j)
        L = torch.matmul(a, Tkj.reshape(8, 64)).reshape(N, 8, 8)
        # c = L @ b: (N, 8, 8) @ (N, 8, 1) -> (N, 8)
        return torch.bmm(L, b.unsqueeze(-1)).squeeze(-1)

    def forward(self, tokens):
        x = self.embed[tokens]  # (batch, L, dim)
        h = x
        level = 0
        while h.shape[1] > 1:
            n = h.shape[1]
            if n % 2 == 1:
                pad = torch.zeros(h.shape[0], 1, self.dim, device=h.device, dtype=h.dtype)
                pad[:, 0, 0] = 1.0
                h = torch.cat([h, pad], dim=1)
                n += 1

            # Reshape to pairs: (batch * n//2, dim)
            left = h[:, :n//2].reshape(-1, self.dim)
            right = h[:, n//2:].reshape(-1, self.dim)

            if self.use_oct:
                prod = self._oct_mul_batched(left, right)
            else:
                prod = left * right

            res = left + right
            gp = torch.sigmoid(self.gate_prod[level])
            gr = torch.sigmoid(self.gate_res[level])
            combined = torch.tanh(gp * prod + gr * res + self.bias[level])

            h = combined.reshape(h.shape[0], n//2, self.dim)
            level += 1

        return self.readout(h[:, 0])


def benchmark_gpu_vs_cpu(length=1024, batch=256, device='cuda', n_iters=100):
    """Benchmark OctTreeGPU vs OctTreeClassifier."""
    import time

    vocab = 7
    tokens = torch.randint(0, vocab, (batch, length), device=device)

    print(f"Benchmark: L={length}, batch={batch}, device={device}")

    # GPU model
    model_gpu = OctTreeGPU(vocab, 8, 2, use_oct=True).to(device)

    # Warmup
    for _ in range(5):
        _ = model_gpu(tokens)
    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    t0 = time.time()
    for _ in range(n_iters):
        logits = model_gpu(tokens)
        loss = logits.sum()
        loss.backward()
        model_gpu.zero_grad()
    if device == 'cuda':
        torch.cuda.synchronize()
    dt_gpu = (time.time() - t0) / n_iters * 1000

    print(f"  OctTreeGPU: {dt_gpu:.1f} ms/iter (fwd+bwd)")
    print(f"  Params: {count_params(model_gpu)}")

    if device == 'cuda':
        # Compare with CPU
        model_cpu = OctTreeClassifier(vocab, 8, 2, use_oct=True)
        tokens_cpu = tokens.cpu()
        t0 = time.time()
        for _ in range(min(n_iters, 10)):
            logits = model_cpu(tokens_cpu)
            loss = logits.sum()
            loss.backward()
            model_cpu.zero_grad()
        dt_cpu = (time.time() - t0) / min(n_iters, 10) * 1000
        speedup = dt_cpu / dt_gpu
        print(f"  OctTree CPU: {dt_cpu:.1f} ms/iter")
        print(f"  Speedup: {speedup:.1f}x")

    return dt_gpu


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='GPU OctTree benchmark')
    p.add_argument('--length', type=int, default=1024)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--n-iters', type=int, default=100)
    args = p.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Compute capability: {torch.cuda.get_device_capability()}")
        benchmark_gpu_vs_cpu(args.length, args.batch, device, args.n_iters)
    else:
        print("No CUDA available. Running CPU-only benchmark.")
        benchmark_gpu_vs_cpu(args.length, args.batch, 'cpu', args.n_iters)
