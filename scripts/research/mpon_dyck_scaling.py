#!/usr/bin/env python3
"""
Multi-Parenthesization Octonion Network (MPON)
===============================================
The architecture that ACTUALLY activates octonion non-associativity.

PROBLEM WITH NAIVE OSSM
  h_t = tanh(A ⊗ h_{t-1} + E[x_t]) uses a BINARY octonion product.
  By Artin's theorem, any two octonions generate an associative subalgebra.
  So L(A)·h is a fixed 8×8 real matrix — non-associativity is dormant.

THE FIX
  Instead of a linear fold, compute the octonion product over the sequence
  using a BALANCED BINARY TREE:

    Level 0:  e_1  e_2  e_3  e_4  e_5  e_6  e_7  e_8
    Level 1:  (e_1⊗e_2)  (e_3⊗e_4)  (e_5⊗e_6)  (e_7⊗e_8)
    Level 2:  ((e_1⊗e_2)⊗(e_3⊗e_4))  ((e_5⊗e_6)⊗(e_7⊗e_8))
    Level 3:  ((e_1⊗e_2)⊗(e_3⊗e_4)) ⊗ ((e_5⊗e_6)⊗(e_7⊗e_8))

  At level 2+, each operand is itself a product of different generators.
  The associator [e_1⊗e_2, e_3⊗e_4, ...] is genuinely nonzero.

ARCHITECTURE: MPON
  1. Embed tokens as octonions
  2. Compute K different bracketings (left, right, balanced, random)
  3. Each bracketing produces a different octonion result
  4. The model learns which bracketings carry signal
  5. At each internal node: optional gated mix of ⊗ (non-assoc) and + (assoc)

  This is O(L log L) for balanced (vs O(L) for left fold) — much faster
  on GPU because each level is fully parallel.

BASELINES (same params):
  - OctTree-8:  balanced tree with octonion ⊗  (non-associative)
  - RealTree-8: balanced tree with element-wise × (associative, same structure)
  - OSSM-8:     left fold with ⊗ (the naive model, dormant by Artin)
  - Diag-8:     left fold with element-wise ×
  - GRU-8:      standard GRU
"""

import argparse
import json
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("PyTorch required: pip install torch")

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Import octonion arithmetic and Dyck generators from the main module
from ossm_dyck_scaling import (
    oct_mul_fast, _T, _T_KJ, gen_dyck1, gen_dyck2,
    count_params, GRUClassifier, LSTMClassifier, OSSMCell,
    train_one
)


# ============================================================
# TREE FOLDING FUNCTIONS
# ============================================================

def tree_fold_balanced(x, use_oct=True):
    """Balanced binary tree product.

    x: (batch, L, dim) -> (batch, dim)
    Reduces L elements to 1 via ceil(log2(L)) levels.
    Each level: pairs of elements are multiplied, in parallel.
    """
    batch, L, dim = x.shape
    h = x
    while h.shape[1] > 1:
        n = h.shape[1]
        if n % 2 == 1:
            # Pad with octonion identity (real part 1)
            pad = torch.zeros(batch, 1, dim, device=h.device, dtype=h.dtype)
            pad[:, 0, 0] = 1.0
            h = torch.cat([h, pad], dim=1)
            n += 1
        left = h[:, :n//2].reshape(-1, dim)
        right = h[:, n//2:].reshape(-1, dim)
        if use_oct:
            h = oct_mul_fast(left, right).reshape(batch, n//2, dim)
        else:
            h = (left * right).reshape(batch, n//2, dim)
    return h[:, 0]


def tree_fold_left(x, use_oct=True):
    """Left-associative fold: (((x_0 ⊗ x_1) ⊗ x_2) ⊗ x_3) ..."""
    batch, L, dim = x.shape
    h = x[:, 0]
    for t in range(1, L):
        if use_oct:
            h = oct_mul_fast(h, x[:, t])
        else:
            h = h * x[:, t]
    return h


def tree_fold_right(x, use_oct=True):
    """Right-associative fold: x_0 ⊗ (x_1 ⊗ (x_2 ⊗ ... x_{L-1}))"""
    batch, L, dim = x.shape
    h = x[:, -1]
    for t in range(L-2, -1, -1):
        if use_oct:
            h = oct_mul_fast(x[:, t], h)
        else:
            h = x[:, t] * h
    return h


def tree_fold_mixed(x, use_oct=True, seed=0):
    """Random balanced tree with a specific seed.
    Shuffles pairs at each level before multiplying.
    """
    batch, L, dim = x.shape
    rng = torch.Generator(device=x.device)
    rng.manual_seed(seed)
    h = x
    while h.shape[1] > 1:
        n = h.shape[1]
        if n % 2 == 1:
            pad = torch.zeros(batch, 1, dim, device=h.device, dtype=h.dtype)
            pad[:, 0, 0] = 1.0
            h = torch.cat([h, pad], dim=1)
            n += 1
        # Shuffle columns
        perm = torch.randperm(n, generator=rng, device=h.device)
        h = h[:, perm]
        left = h[:, :n//2].reshape(-1, dim)
        right = h[:, n//2:].reshape(-1, dim)
        if use_oct:
            h = oct_mul_fast(left, right).reshape(batch, n//2, dim)
        else:
            h = (left * right).reshape(batch, n//2, dim)
    return h[:, 0]


# ============================================================
# MODELS
# ============================================================

class OctTreeClassifier(nn.Module):
    """Octonion tree-product classifier with residual mixing.

    At each internal tree node, computes:
      out = tanh(W_prod · (left ⊗ right) + W_res · (left + right) + b)

    The ⊗ term carries non-associative structure (Artin-active at level ≥ 2).
    The + term carries an associative residual path.
    W_prod and W_res are learnable scalar gates per level.

    This avoids the "collapse to identity" problem of pure product trees.
    """
    def __init__(self, vocab_size, dim=8, n_classes=2,
                 use_oct=True, max_levels=14):
        super().__init__()
        self.dim = dim
        self.use_oct = use_oct
        self.max_levels = max_levels

        self.embed = nn.Parameter(torch.randn(vocab_size, dim) * 0.1)
        # Per-level gates: how much product vs residual
        self.gate_prod = nn.Parameter(torch.ones(max_levels) * 0.5)
        self.gate_res = nn.Parameter(torch.ones(max_levels) * 0.5)
        # Per-level bias
        self.bias = nn.Parameter(torch.zeros(max_levels, dim))
        self.readout = nn.Linear(dim, n_classes)

    def forward(self, tokens):
        x = self.embed[tokens]  # (batch, L, dim)
        h = x
        level = 0
        while h.shape[1] > 1:
            n = h.shape[1]
            if n % 2 == 1:
                pad = torch.zeros(h.shape[0], 1, self.dim, device=h.device, dtype=h.dtype)
                h = torch.cat([h, pad], dim=1)
                n += 1
            left = h[:, :n//2].reshape(-1, self.dim)
            right = h[:, n//2:].reshape(-1, self.dim)

            # Non-associative path: octonion product
            if self.use_oct:
                prod = oct_mul_fast(left, right)
            else:
                prod = left * right  # element-wise (associative)

            # Associative residual path: sum
            res = left + right

            # Gated mix
            gp = torch.sigmoid(self.gate_prod[level])
            gr = torch.sigmoid(self.gate_res[level])
            combined = gp * prod + gr * res + self.bias[level]

            # Bound magnitudes
            combined = torch.tanh(combined)

            h = combined.reshape(h.shape[0], n//2, self.dim)
            level += 1

        return self.readout(h[:, 0])


class MPONClassifier(nn.Module):
    """Multi-Parenthesization Octonion Network.

    Computes K different bracketings, concatenates results.
    The model learns which bracketings carry signal via the readout.
    """
    def __init__(self, vocab_size, dim=8, n_classes=2,
                 n_bracketings=4, use_tanh=True):
        super().__init__()
        self.dim = dim
        self.n_bracketings = n_bracketings
        self.use_tanh = use_tanh

        self.embed_oct = nn.Parameter(torch.randn(vocab_size, dim) * 0.3)
        self.embed_real = nn.Parameter(torch.randn(vocab_size, dim) * 0.3)
        # Readout from K octonion results (K*dim features)
        self.readout = nn.Sequential(
            nn.Linear(n_bracketings * dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )

    def forward(self, tokens):
        e_oct = self.embed_oct[tokens]  # (batch, L, 8)
        e_real = self.embed_real[tokens]  # (batch, L, 8) — for real baseline bracketings

        results = []
        # Bracketing 0: octonion balanced tree
        results.append(tree_fold_balanced(e_oct, use_oct=True))
        # Bracketing 1: octonion left fold
        results.append(tree_fold_left(e_oct, use_oct=True))
        # Bracketing 2: octonion right fold
        results.append(tree_fold_right(e_oct, use_oct=True))
        # Bracketing 3: real balanced tree (associative control, same structure)
        if self.n_bracketings >= 4:
            results.append(tree_fold_balanced(e_real, use_oct=False))

        # Apply tanh to each result
        if self.use_tanh:
            results = [torch.tanh(r) for r in results]

        h = torch.cat(results[:self.n_bracketings], dim=-1)  # (batch, K*8)
        return self.readout(h)


# ============================================================
# EXPERIMENT
# ============================================================

def run_mpon_experiment(lengths=(32, 64, 128, 256, 512, 1024),
                        n_types=1, epochs=100, train_size=4096,
                        test_size=1024, seed=20260806, device='cpu'):
    rng = np.random.default_rng(seed)
    gen = gen_dyck1 if n_types == 1 else gen_dyck2
    vocab = 3 if n_types == 1 else 5

    results = {}
    print(f"\n{'='*72}")
    print(f"MPON / TREE FOLD EXPERIMENT — Dyck-{n_types}")
    print(f"Device: {device}")
    print(f"{'='*72}")

    for L in lengths:
        # Ensure power of 2 for clean tree folding
        L_tree = 1 << (L - 1).bit_length() if L > 0 else 1
        if L_tree != L:
            print(f"  (rounding L={L} to {L_tree} for tree)")
            L = L_tree

        print(f"\n--- L = {L} ---")
        tr_t, tr_l = gen(L, train_size, rng)
        te_t, te_l = gen(L, test_size, rng)
        tr_t = torch.from_numpy(tr_t)
        tr_l = torch.from_numpy(tr_l)
        te_t = torch.from_numpy(te_t)
        te_l = torch.from_numpy(te_l)

        models = {
            'OctTree-8':  OctTreeClassifier(vocab, 8, 2, use_oct=True),
            'RealTree-8': OctTreeClassifier(vocab, 8, 2, use_oct=False),
            'OSSM-8':     OSSMCell(vocab, 8, 2, use_octonion=True),
            'Diag-8':     OSSMCell(vocab, 8, 2, use_octonion=False),
            'GRU-8':      GRUClassifier(vocab, 8, 2),
        }
        results[L] = {}

        for name, model in models.items():
            model = model.to(device)
            np_p = count_params(model)
            t0 = time.time()
            hist = train_one(model, tr_t, tr_l, te_t, te_l,
                           epochs=epochs, lr=1e-2, batch_size=64,
                           device=device, name=name)
            dt = time.time() - t0
            final = hist['test_acc'][-1]
            best = max(hist['test_acc'])
            results[L][name] = {
                'params': np_p, 'final_test_acc': final, 'best_test_acc': best,
                'final_train_acc': hist['train_acc'][-1],
                'time_sec': round(dt, 1),
            }
            print(f"  {name:<14} ({np_p:>5d}p)  test={final:.3f}  best={best:.3f}  ({dt:.0f}s)")

    # Summary table
    print(f"\n{'='*72}")
    print("SUMMARY TABLE — Final Test Accuracy")
    print(f"{'='*72}")
    model_names = list(models.keys())
    header = f"{'Model':<16}" + "".join(f"L={L:<8}" for L in lengths)
    print(header)
    print("-" * len(header))
    for name in model_names:
        cells = f"{name:<16}"
        for L in lengths:
            acc = results[L][name]['final_test_acc']
            cells += f"{acc:<10.3f}"
        print(cells)

    # Key comparison
    print(f"\n  OctTree vs RealTree (same structure, different algebra):")
    for L in lengths:
        o = results[L]['OctTree-8']['final_test_acc']
        r = results[L]['RealTree-8']['final_test_acc']
        diff = o - r
        bar = "+" * int(max(diff, 0) * 50) if diff > 0 else "-" * int(min(-diff, 0) * 50)
        print(f"    L={L:>5d}: Oct={o:.3f}  Real={r:.3f}  Δ={diff:+.3f}  {bar}")

    outpath = "scripts/research/mpon_dyck_results.json"
    out = {str(L): v for L, v in results.items()}
    with open(outpath, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {outpath}")
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='MPON / Tree-fold experiment')
    p.add_argument('--lengths', type=int, nargs='+', default=[32, 64, 128, 256, 512, 1024])
    p.add_argument('--n-types', type=int, default=1)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--train-size', type=int, default=4096)
    p.add_argument('--test-size', type=int, default=1024)
    p.add_argument('--seed', type=int, default=20260806)
    p.add_argument('--gpu', action='store_true')
    args = p.parse_args()

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    run_mpon_experiment(
        lengths=tuple(args.lengths), n_types=args.n_types,
        epochs=args.epochs, train_size=args.train_size,
        test_size=args.test_size, seed=args.seed, device=device)
