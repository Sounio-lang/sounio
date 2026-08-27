#!/usr/bin/env python3
"""
RNA secondary structure with REAL ViennaRNA folding (MFE, not Nussinov).

ViennaRNA uses the Turner energy model — the actual thermodynamic model
used in RNA biology. This produces biologically realistic structures:
  - Hairpin loops (min 3 nt)
  - Interior loops, bulges
  - Multi-loops
  - Pseudoknot-free (but complex nesting)

Task: classify real MFE structure vs corrupted structure.
The dot-bracket sequence is the input; the model must learn whether
the bracket structure is biologically valid vs artificially corrupted.
"""

import numpy as np
import json
import sys, os, time

try:
    import RNA as vrna
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False
    raise SystemExit("ViennaRNA required: pip install ViennaRNA")

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch required")

sys.path.insert(0, os.path.dirname(__file__))
from ossm_dyck_scaling import train_one, count_params, GRUClassifier, OSSMCell
from mpon_dyck_scaling import OctTreeClassifier


# ============================================================
# RNA DATA GENERATION WITH VIENNA RNA
# ============================================================

BASES = 'ACGU'
BASE_TO_INT = {'(': 1, ')': 2, '.': 0}

def gen_rna_sequence(length, rng, gc_bias=0.5):
    """Generate RNA with realistic composition."""
    seq = []
    for _ in range(length):
        if rng.random() < gc_bias:
            seq.append(rng.choice(['G', 'C']))
        else:
            seq.append(rng.choice(['A', 'U']))
    return ''.join(seq)


def fold_rna(seq):
    """Fold with ViennaRNA MFE."""
    fc = vrna.fold_compound(seq)
    ss, mfe = fc.mfe()
    return ss, mfe


def gen_vienna_dataset(length, n_samples, rng):
    """Generate RNA sequences, fold with ViennaRNA, create valid/corrupted labels."""
    tokens = np.zeros((n_samples, length), dtype=np.int64)
    labels = np.zeros(n_samples, dtype=np.int64)
    target_valid = n_samples // 2
    valid_count = 0

    for i in range(n_samples):
        gc = rng.uniform(0.3, 0.7)
        seq = gen_rna_sequence(length, rng, gc_bias=gc)
        ss, mfe = fold_rna(seq)

        bracket_seq = np.array([BASE_TO_INT.get(c, 0) for c in ss])

        if valid_count < target_valid:
            tokens[i] = bracket_seq
            labels[i] = 1
            valid_count += 1
        else:
            # Corrupt the dot-bracket
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


# ============================================================
# EXPERIMENT
# ============================================================

def run(lengths=(32, 64, 128, 256, 512), epochs=50,
        train_size=2048, test_size=512, seed=20260806, device='cpu'):
    rng = np.random.default_rng(seed)
    vocab = 3

    results = {}
    print(f"\n{'='*72}")
    print(f"RNA MFE (ViennaRNA) — OctTree vs RealTree")
    print(f"Device: {device}")
    print(f"{'='*72}")

    for L in lengths:
        L_tree = 1 << (L - 1).bit_length()
        if L_tree != L:
            L = L_tree

        print(f"\n--- L = {L} ---")
        t0 = time.time()
        tr_tokens, tr_labels = gen_vienna_dataset(L, train_size, rng)
        te_tokens, te_labels = gen_vienna_dataset(L, test_size, rng)
        fold_time = time.time() - t0
        n_open = (tr_tokens == 1).sum(1).mean()
        n_loop = (tr_tokens == 0).sum(1).mean()
        print(f"  Folded {train_size + test_size} seqs in {fold_time:.1f}s")
        print(f"  Avg: {n_open:.0f} pairs, {n_loop:.0f} unpaired per seq")

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
    print("SUMMARY — ViennaRNA MFE")
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

    outpath = "scripts/research/rna_vienna_results.json"
    with open(outpath, 'w') as f:
        json.dump({str(L): v for L, v in results.items()}, f, indent=2)
    print(f"\nSaved to {outpath}")
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--lengths', type=int, nargs='+', default=[32, 64, 128, 256, 512])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--train-size', type=int, default=2048)
    p.add_argument('--test-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=20260806)
    args = p.parse_args()

    run(lengths=tuple(args.lengths), epochs=args.epochs,
        train_size=args.train_size, test_size=args.test_size, seed=args.seed)
