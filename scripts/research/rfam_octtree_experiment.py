#!/usr/bin/env python3
"""
OctTree on REAL Rfam RNA secondary structures.
108K sequences from 4,225 families with consensus dot-bracket structures.

Task: classify real RNA structures vs corrupted (bracket-flipped).
This is the definitive test: real biological data, not synthetic folding.
"""

import numpy as np
import json
import sys, os, time

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch required")

sys.path.insert(0, os.path.dirname(__file__))
from ossm_dyck_scaling import train_one, count_params, GRUClassifier, OSSMCell
from mpon_dyck_scaling import OctTreeClassifier

DATA_PATH = '/workspace/sounio/datasets/rna_secondary_structure/rfam_structures.fasta'


def load_rfam(max_seqs=50000, min_len=32, max_len=512):
    """Load Rfam sequences with dot-bracket structures."""
    seqs = []
    structs = []
    families = []

    with open(DATA_PATH) as f:
        lines = f.readlines()

    i = 0
    while i < len(lines) and len(seqs) < max_seqs:
        if lines[i].startswith('>'):
            # Parse header
            header = lines[i].strip()
            fam = 'unknown'
            for part in header.split():
                if part.startswith('family='):
                    fam = part.split('=')[1]
                    break

            if i + 2 < len(lines):
                seq = lines[i+1].strip()
                ss = lines[i+2].strip()

                if min_len <= len(seq) <= max_len and len(seq) == len(ss):
                    # Check it's valid dot-bracket
                    if set(ss) <= set('().'):
                        seqs.append(seq)
                        structs.append(ss)
                        families.append(fam)
            i += 3
        else:
            i += 1

    return seqs, structs, families


def rfam_to_dataset(sequences, structures, length, n_samples, rng):
    """Convert Rfam data to fixed-length tensors.

    Valid: real Rfam dot-bracket (padded with 0=loop)
    Invalid: real dot-bracket with one bracket flipped
    """
    BRACKET_MAP = {'(': 1, ')': 2, '.': 0}

    # Filter to sequences within 2x of target length (pad/truncate)
    suitable = [(s, ss) for s, ss in zip(sequences, structures)
                if len(ss) <= length and len(ss) >= length // 4]

    if not suitable:
        suitable = [(s, ss) for s, ss in zip(sequences, structures) if len(ss) <= length]

    n_valid = min(n_samples // 2, len(suitable))
    n_invalid = n_samples - n_valid

    tokens = np.zeros((n_samples, length), dtype=np.int64)
    labels = np.zeros(n_samples, dtype=np.int64)

    # Valid: real Rfam structures
    idx = rng.choice(len(suitable), n_valid, replace=len(suitable) < n_valid)
    for j, i in enumerate(idx):
        seq, ss = suitable[i]
        bracket = np.array([BRACKET_MAP.get(c, 0) for c in ss])
        tokens[j, :len(bracket)] = bracket[:length]
        labels[j] = 1

    # Invalid: corrupted real structures
    for j in range(n_valid, n_samples):
        src_idx = rng.choice(len(suitable))
        seq, ss = suitable[src_idx]
        bracket = np.array([BRACKET_MAP.get(c, 0) for c in ss])
        # Corrupt: flip brackets
        n_flip = max(1, len(bracket) // 8)
        for _ in range(n_flip):
            r = rng.random()
            if r < 0.33 and (bracket == 1).any():
                pos = rng.choice(np.where(bracket == 1)[0])
                bracket[pos] = 2
            elif r < 0.66 and (bracket == 2).any():
                pos = rng.choice(np.where(bracket == 2)[0])
                bracket[pos] = 1
            elif (bracket == 0).any():
                pos = rng.choice(np.where(bracket == 0)[0])
                bracket[pos] = rng.choice([1, 2])
        tokens[j, :len(bracket)] = bracket[:length]
        labels[j] = 0

    perm = rng.permutation(n_samples)
    return tokens[perm], labels[perm]


def run(lengths=(32, 64, 128, 256, 512), epochs=50,
        train_size=2048, test_size=512, seed=20260806, device='cpu'):
    rng = np.random.default_rng(seed)
    vocab = 3

    print("Loading Rfam data...")
    seqs, structs, fams = load_rfam(max_seqs=50000, min_len=32, max_len=1024)
    print(f"Loaded {len(seqs)} sequences from {len(set(fams))} families")
    lens = np.array([len(s) for s in structs])
    print(f"Length: min={lens.min()}, median={np.median(lens):.0f}, max={lens.max()}")

    results = {}
    print(f"\n{'='*72}")
    print(f"REAL RFAM RNA — OctTree vs RealTree")
    print(f"{'='*72}")

    for L in lengths:
        L_tree = 1 << (L - 1).bit_length()
        if L_tree != L:
            L = L_tree

        print(f"\n--- L = {L} ---")
        tr_tokens, tr_labels = rfam_to_dataset(seqs, structs, L, train_size, rng)
        te_tokens, te_labels = rfam_to_dataset(seqs, structs, L, test_size, rng)
        n_open = (tr_tokens == 1).sum(1).mean()
        print(f"  Train: {tr_labels.sum()} valid, {train_size-tr_labels.sum()} corrupted, avg pairs={n_open:.0f}")

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
    print("SUMMARY — Real Rfam RNA")
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

    outpath = "scripts/research/rfam_octtree_results.json"
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
