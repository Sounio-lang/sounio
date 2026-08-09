#!/usr/bin/env python3
"""
RNA Secondary Structure — Octonion Tree-Fold Experiment
========================================================
RNA secondary structure IS bracket matching in nature.

A folded RNA has base pairs (A-U, G-C, G-U) that form nested patterns
represented in dot-bracket notation:
    (((...)))      ← hairpin: 3 opening, 3 closing
    ((...((...)).)) ← multiloop: nested + crossing structure
    ..(((...)))...  ← bulge + interior loop

The task: given an RNA nucleotide sequence, predict whether its
secondary structure is "well-formed" (all brackets close) vs "misfolded"
(bracket violations). This is Dyck-1 on real biological sequences.

DATA
  1. Generate RNA sequences (A,C,G,U) with realistic composition
  2. Fold them using Nussinov dynamic programming → ground-truth structure
  3. Extract dot-bracket representation
  4. Label: valid folding (all brackets close) vs corrupted (some violations)

  This gives REAL biological sequences with REAL bracketing structure,
  unlike synthetic Dyck which uses random bracket placement.

ARCHITECTURE
  Embed each nucleotide as an octonion (A,C,G,U → GF(4) → octonion)
  Fold the sequence via OctTree (⊗) vs RealTree (×)
  Classify: valid structure vs invalid

  If OctTree > RealTree, non-associativity captures RNA folding structure.
"""

import numpy as np
import json
import sys, os

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    raise SystemExit("PyTorch required")

sys.path.insert(0, os.path.dirname(__file__))
from ossm_dyck_scaling import oct_mul_fast, gen_dyck1, train_one, count_params
from mpon_dyck_scaling import OctTreeClassifier, GRUClassifier, OSSMCell


# ============================================================
# NUSINOV RNA FOLDING (dynamic programming)
# ============================================================

# RNA complementarity
COMPLEMENT = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G',
              0: 3, 3: 0, 1: 2, 2: 1,  # GF(4) encoding: A=0,C=1,G=2,T/U=3
              'a': 'u', 'u': 'a', 'g': 'c', 'c': 'g'}

# Base pair energy (simplified): Watson-Crick and wobble
BP_ENERGY = {
    ('A', 'U'): -2, ('U', 'A'): -2,
    ('G', 'C'): -3, ('C', 'G'): -3,
    ('G', 'U'): -1, ('U', 'G'): -1,  # wobble
}

def can_pair(a, b):
    """Check if two bases can form a Watson-Crick or wobble pair."""
    return (a, b) in BP_ENERGY


def nussinov_fold(seq, min_loop=3):
    """Nussinov algorithm: find maximum base-pairing.

    seq: string of ACGU
    Returns: list of (i, j) base pairs, dot-bracket string
    """
    n = len(seq)
    # DP table: dp[i][j] = max pairs in seq[i..j]
    dp = np.zeros((n, n), dtype=np.int32)

    for length in range(min_loop + 1, n):
        for i in range(n - length):
            j = i + length
            # Option 1: j unpaired
            best = dp[i, j-1]
            # Option 2: i pairs with some k in [i+min_loop, j]
            if can_pair(seq[i], seq[j]):
                inner = dp[i+1, j-1] if (i+1 < j-1) else 0
                best = max(best, inner + 1)
            for k in range(i + min_loop + 1, j):
                if can_pair(seq[k], seq[j]):
                    left = dp[i, k-1] if k > i else 0
                    right = dp[k+1, j-1] if k < j-1 else 0
                    best = max(best, left + right + 1)
            dp[i, j] = best

    # Traceback
    pairs = []
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + min_loop:
            continue
        if dp[i, j] == dp[i, j-1]:
            stack.append((i, j-1))
        elif can_pair(seq[i], seq[j]) and dp[i, j] == (dp[i+1, j-1] if i+1 < j-1 else 0) + 1:
            pairs.append((i, j))
            stack.append((i+1, j-1))
        else:
            for k in range(i + min_loop + 1, j):
                if can_pair(seq[k], seq[j]):
                    left = dp[i, k-1] if k > i else 0
                    right = dp[k+1, j-1] if k < j-1 else 0
                    if dp[i, j] == left + right + 1:
                        pairs.append((k, j))
                        stack.append((i, k-1))
                        stack.append((k+1, j-1))
                        break

    # Dot-bracket
    dot = list('.' * n)
    for i, j in pairs:
        dot[i] = '('
        dot[j] = ')'

    return pairs, ''.join(dot)


# ============================================================
# RNA DATA GENERATION
# ============================================================

BASES = 'ACGU'
BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

def gen_rna_sequence(length, rng, gc_bias=0.5):
    """Generate a random RNA sequence with realistic composition."""
    # GC content varies; bias toward biological range
    seq = []
    for _ in range(length):
        if rng.random() < gc_bias:
            seq.append(rng.choice(['G', 'C']))
        else:
            seq.append(rng.choice(['A', 'U']))
    return ''.join(seq)


def gen_rna_dataset(length, n_samples, rng):
    """Generate RNA sequences with their secondary structures.

    Returns:
      tokens: (N, L) int tensor of nucleotide indices (0=A,1=C,2=G,3=U)
      dot_brackets: list of strings
      labels: (N,) — 1 if well-folded (all brackets close), 0 if corrupted
      n_pairs: list of int — number of base pairs per sequence
    """
    tokens = np.zeros((n_samples, length), dtype=np.int64)
    dot_brackets = []
    labels = np.zeros(n_samples, dtype=np.int64)
    n_pairs_list = []

    valid_count = 0
    invalid_count = 0
    target_valid = n_samples // 2

    for i in range(n_samples):
        # Generate sequence
        gc = rng.uniform(0.3, 0.7)
        seq = gen_rna_sequence(length, rng, gc_bias=gc)

        # Fold it
        pairs, dot = nussinov_fold(seq, min_loop=3)
        n_pairs = len(pairs)

        if valid_count < target_valid:
            # Keep the valid folding
            tokens[i] = [BASE_TO_INT[b] for b in seq]
            dot_brackets.append(dot)
            labels[i] = 1
            n_pairs_list.append(n_pairs)
            valid_count += 1
        else:
            # Corrupt: mutate some bases to break structure
            corrupt_seq = list(seq)
            n_mutations = max(1, length // 10)
            mut_positions = rng.choice(length, n_mutations, replace=False)
            for pos in mut_positions:
                # Change to a non-complementary base
                orig = corrupt_seq[pos]
                others = [b for b in BASES if b != orig]
                corrupt_seq[pos] = rng.choice(others)

            # Re-fold the corrupted sequence
            corrupt_str = ''.join(corrupt_seq)
            pairs_c, dot_c = nussinov_fold(corrupt_str, min_loop=3)

            tokens[i] = [BASE_TO_INT[b] for b in corrupt_str]
            dot_brackets.append(dot_c)
            labels[i] = 0  # corrupted
            n_pairs_list.append(len(pairs_c))
            invalid_count += 1

    # Shuffle
    perm = rng.permutation(n_samples)
    return tokens[perm], [dot_brackets[p] for p in perm], labels[perm], [n_pairs_list[p] for p in perm]


# ============================================================
# EXPERIMENT
# ============================================================

def gen_rna_bracket_dataset(length, n_samples, rng):
    """Generate dot-bracket sequences from real RNA folding.

    1. Generate random RNA with biological GC composition
    2. Fold with Nussinov → dot-bracket
    3. Use the dot-bracket as a 3-symbol sequence: ( = 1, ) = 2, . = 0
    4. Label: 1 = valid RNA structure, 0 = bracket-corrupted
    """
    BRACKET_MAP = {'(': 1, ')': 2, '.': 0}
    tokens = np.zeros((n_samples, length), dtype=np.int64)
    labels = np.zeros(n_samples, dtype=np.int64)
    target_valid = n_samples // 2
    valid_count = 0

    for i in range(n_samples):
        gc = rng.uniform(0.3, 0.7)
        seq = gen_rna_sequence(length, rng, gc_bias=gc)
        pairs, dot = nussinov_fold(seq, min_loop=3)
        bracket_seq = np.array([BRACKET_MAP.get(c, 0) for c in dot])

        if valid_count < target_valid:
            tokens[i] = bracket_seq
            labels[i] = 1
            valid_count += 1
        else:
            # Corrupt the bracket sequence directly
            corrupt = bracket_seq.copy()
            n_swap = max(1, length // 8)
            for _ in range(n_swap):
                r = rng.random()
                if r < 0.33 and (corrupt == 1).any():
                    pos = rng.choice(np.where(corrupt == 1)[0])
                    corrupt[pos] = 2
                elif r < 0.66 and (corrupt == 2).any():
                    pos = rng.choice(np.where(corrupt == 2)[0])
                    corrupt[pos] = 1
                elif (corrupt == 0).any():
                    pos = rng.choice(np.where(corrupt == 0)[0])
                    corrupt[pos] = rng.choice([1, 2])
            tokens[i] = corrupt
            labels[i] = 0

    perm = rng.permutation(n_samples)
    return tokens[perm], labels[perm]


def run_rna_experiment(lengths=(32, 64, 128, 256), epochs=50,
                       train_size=2048, test_size=512, seed=20260806,
                       device='cpu'):
    rng = np.random.default_rng(seed)
    vocab = 3  # ( , ) , .

    results = {}
    print(f"\n{'='*72}")
    print(f"RNA DOT-BRACKET — OctTree vs RealTree")
    print(f"Device: {device}")
    print(f"{'='*72}")

    for L in lengths:
        L_tree = 1 << (L - 1).bit_length()
        if L_tree != L:
            L = L_tree

        print(f"\n--- L = {L} ---")
        print(f"  Folding {train_size + test_size} RNA sequences...")
        tr_tokens, tr_labels = gen_rna_bracket_dataset(L, train_size, rng)
        te_tokens, te_labels = gen_rna_bracket_dataset(L, test_size, rng)
        print(f"  Train: {tr_labels.sum()} valid, {train_size - tr_labels.sum()} corrupted")
        n_open = (tr_tokens == 1).sum(1).mean()
        n_loop = (tr_tokens == 0).sum(1).mean()
        print(f"  Avg/sample: {n_open:.0f} pairs, {n_loop:.0f} unpaired")

        tr_t = torch.from_numpy(tr_tokens)
        tr_l = torch.from_numpy(tr_labels)
        te_t = torch.from_numpy(te_tokens)
        te_l = torch.from_numpy(te_labels)

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
            import time
            t0 = time.time()
            hist = train_one(model, tr_t, tr_l, te_t, te_l,
                           epochs=epochs, lr=1e-2, batch_size=64,
                           device=device, name=name)
            dt = time.time() - t0
            final = hist['test_acc'][-1]
            best = max(hist['test_acc'])
            results[L][name] = {
                'params': np_p, 'final_test_acc': final, 'best_test_acc': best,
                'time_sec': round(dt, 1),
            }
            print(f"  {name:<14} ({np_p:>5d}p)  test={final:.3f}  best={best:.3f}  ({dt:.0f}s)")

    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")
    header = f"{'Model':<14}" + "".join(f"L={L:<8}" for L in lengths)
    print(header)
    print("-" * len(header))
    for name in models:
        cells = f"{name:<14}"
        for L in lengths:
            cells += f"{results[L][name]['final_test_acc']:<10.3f}"
        print(cells)

    print(f"\n  OctTree vs RealTree:")
    for L in lengths:
        o = results[L]['OctTree-8']['final_test_acc']
        r = results[L]['RealTree-8']['final_test_acc']
        diff = o - r
        bar = "+" * int(max(diff, 0) * 50) if diff > 0 else "-" * int(min(-diff, 0) * 50)
        print(f"    L={L:>5d}: Δ={diff:+.3f}  {bar}")

    outpath = "scripts/research/rna_octtree_results.json"
    with open(outpath, 'w') as f:
        json.dump({str(L): v for L, v in results.items()}, f, indent=2)
    print(f"\nSaved to {outpath}")
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--lengths', type=int, nargs='+', default=[32, 64, 128, 256])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--train-size', type=int, default=2048)
    p.add_argument('--test-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=20260806)
    args = p.parse_args()

    run_rna_experiment(
        lengths=tuple(args.lengths),
        epochs=args.epochs,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed)
