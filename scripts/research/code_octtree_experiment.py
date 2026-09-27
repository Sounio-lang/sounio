#!/usr/bin/env python3
"""
OctTree on Code/AST bracketing data.
Code has LITERAL brackets: (), {}, [], and evaluation order.

Task: classify code snippets as "balanced brackets" vs "unbalanced".
This is the real-world Dyck analog — code IS a bracketing language.

Data sources:
  1. Generate from the Sounio repo's own .sio files (extract bracket sequences)
  2. Generate synthetic C-like code with known bracket structure
"""

import os
import re
import glob
import json
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Import everything from the main modules
from ossm_dyck_scaling import (
    oct_mul_fast, train_one, count_params, GRUClassifier, OSSMCell
)
from mpon_dyck_scaling import OctTreeClassifier
import torch
import torch.nn as nn


# ============================================================
# BRACKET EXTRACTION FROM CODE
# ============================================================

# All bracket types in code: (), {}, [], <>
BRACKETS_OPEN = set('([{<')
BRACKETS_CLOSE = set(')]}>')
BRACKET_PAIRS = {'(': ')', '[': ']', '{': '}', '<': '>'}
REVERSE_PAIRS = {v: k for k, v in BRACKET_PAIRS.items()}

# Token mapping: 0=other, 1=(, 2=), 3=[, 4=], 5={, 6=}
BRACKET_TO_INT = {'(': 1, ')': 2, '[': 3, ']': 4, '{': 5, '}': 6}


def extract_bracket_sequence(code_text):
    """Extract the bracket sequence from source code.
    Returns a list of bracket token indices, ignoring all non-bracket characters.
    """
    seq = []
    for ch in code_text:
        if ch in BRACKET_TO_INT:
            seq.append(BRACKET_TO_INT[ch])
    return seq


def load_code_brackets(repo_path='/workspace/sounio', max_files=500, min_len=16, max_len=512):
    """Extract bracket sequences from .sio, .py, .cpp, .ts, .rs files."""
    extensions = ['*.sio', '*.py', '*.cpp', '*.rs', '*.ts', '*.go', '*.java', '*.c', '*.h']
    sequences = []

    for ext in extensions:
        pattern = os.path.join(repo_path, '**', ext)
        for filepath in glob.glob(pattern, recursive=True):
            if 'node_modules' in filepath or '.venv' in filepath or '.git' in filepath:
                continue
            try:
                with open(filepath, 'r', errors='ignore') as f:
                    text = f.read()
                seq = extract_bracket_sequence(text)
                if min_len <= len(seq) <= max_len:
                    sequences.append(seq)
                    if len(sequences) >= max_files:
                        return sequences
            except:
                continue

    return sequences


def make_code_dataset(sequences, length, n_samples, rng):
    """Create balanced dataset from code bracket sequences.

    Valid: actual code bracket sequences (padded/truncated to fixed length)
    Invalid: code sequences with one bracket flipped

    Returns tokens (n_samples, length), labels (n_samples,)
    """
    # Pad/truncate sequences to fixed length
    padded = np.zeros((len(sequences), length), dtype=np.int64)
    for i, seq in enumerate(sequences):
        n = min(len(seq), length)
        padded[i, :n] = seq[:n]

    # Filter to sequences with enough brackets
    bracket_counts = (padded > 0).sum(axis=1)
    valid_pool = padded[bracket_counts >= length // 4]

    if len(valid_pool) < 10:
        # Fall back to using all sequences
        valid_pool = padded

    n_valid = min(n_samples // 2, len(valid_pool))
    n_invalid = n_samples - n_valid

    tokens = np.zeros((n_samples, length), dtype=np.int64)
    labels = np.zeros(n_samples, dtype=np.int64)

    # Valid: sample from real code
    if n_valid > 0:
        idx = rng.choice(len(valid_pool), n_valid, replace=len(valid_pool) < n_valid)
        tokens[:n_valid] = valid_pool[idx]
        labels[:n_valid] = 1

    # Invalid: corrupt real code by flipping one bracket
    for i in range(n_valid, n_samples):
        src = valid_pool[rng.choice(len(valid_pool))]
        corrupt = src.copy()
        bracket_pos = np.where(corrupt > 0)[0]
        if len(bracket_pos) > 0:
            pos = rng.choice(bracket_pos)
            old = corrupt[pos]
            # Flip: open->close or close->open of same type, or change type
            if old % 2 == 1:  # open
                corrupt[pos] = old + 1  # matching close
            else:  # close
                corrupt[pos] = old - 1  # matching open
        tokens[i] = corrupt
        labels[i] = 0

    perm = rng.permutation(n_samples)
    return tokens[perm], labels[perm]


# ============================================================
# EXPERIMENT
# ============================================================

def run_code_experiment(lengths=(32, 64, 128, 256), epochs=50,
                        train_size=2048, test_size=512, seed=20260806,
                        device='cpu'):
    rng = np.random.default_rng(seed)
    vocab = 7  # 0=other, 1-6 = bracket types

    # Load code brackets from the repo
    print("Loading code bracket sequences from repo...")
    sequences = load_code_brackets('/workspace/sounio', max_files=2000, min_len=8, max_len=1024)
    print(f"Found {len(sequences)} code files with bracket sequences")
    if sequences:
        lens = [len(s) for s in sequences]
        print(f"  Length range: {min(lens)}-{max(lens)}, median: {sorted(lens)[len(lens)//2]}")
        print(f"  Sample: {sequences[0][:30]}")

    results = {}
    print(f"\n{'='*72}")
    print(f"CODE BRACKET — OctTree vs RealTree")
    print(f"Device: {device}")
    print(f"{'='*72}")

    for L in lengths:
        L_tree = 1 << (L - 1).bit_length()
        if L_tree != L:
            L = L_tree

        print(f"\n--- L = {L} ---")
        tr_tokens, tr_labels = make_code_dataset(sequences, L, train_size, rng)
        te_tokens, te_labels = make_code_dataset(sequences, L, test_size, rng)
        print(f"  Train: {tr_labels.sum()} valid, {train_size - tr_labels.sum()} corrupted")
        print(f"  Test:  {te_labels.sum()} valid, {test_size - te_labels.sum()} corrupted")

        tr_t = torch.from_numpy(tr_tokens)
        tr_l = torch.from_numpy(tr_labels)
        te_t = torch.from_numpy(te_tokens)
        te_l = torch.from_numpy(te_labels)

        models = {
            'OctTree-8':  OctTreeClassifier(vocab, 8, 2, use_oct=True),
            'RealTree-8': OctTreeClassifier(vocab, 8, 2, use_oct=False),
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

    # Summary
    print(f"\n{'='*72}")
    print("SUMMARY — Code Brackets")
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

    outpath = "scripts/research/code_octtree_results.json"
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

    run_code_experiment(
        lengths=tuple(args.lengths),
        epochs=args.epochs,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.seed)
