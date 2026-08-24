#!/usr/bin/env python3
"""
REAL pseudoknot experiment: Sedenion vs Octonion on actual PK RNA.

RF00008 (hammerhead ribozyme): 54 nt, <> crossing brackets
RF00050 (FMN riboswitch): 127 nt, <> and [] crossing brackets

These are REAL biological pseudoknots, not synthetic.
The WUSS notation preserves the crossing structure with multiple bracket types.

Task: classify valid PK structure vs corrupted.
"""

import numpy as np
import json, sys, os, time

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch required")

sys.path.insert(0, os.path.dirname(__file__))
from ossm_dyck_scaling import train_one, count_params, oct_mul_fast
from mpon_dyck_scaling import OctTreeClassifier
from pseudoknot_experiment import SedenionTreeClassifier, MixedOctQuatTree, sed_mul, quat_mul


def parse_stockholm_pk(path, max_seqs=500):
    """Parse Stockholm with PK annotations preserved."""
    seqs = {}
    ss_cons = None
    with open(path) as f:
        for line in f:
            if line.startswith('#=GC SS_cons'):
                ss_cons = line.split()[-1]
            elif line.startswith('#') or line.startswith('//'):
                continue
            else:
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0]
                    seq = parts[1]
                    seqs[name] = seqs.get(name, '') + seq
    if ss_cons is None:
        return [], []
    results = []
    for name, seq in seqs.items():
        clean_s, clean_ss = [], []
        for s, c in zip(seq, ss_cons):
            if s not in '.-_':
                clean_s.append(s)
                clean_ss.append(c)
        s, ss = ''.join(clean_s), ''.join(clean_ss)
        if len(s) > 20 and len(s) == len(ss):
            pk_chars = set(ss) & set('<>[]{}AaBbCcDdEeFf')
            if pk_chars:
                results.append((s, ss))
        if len(results) >= max_seqs:
            break
    return results


def pk_to_tokens(ss):
    """Convert WUSS dot-bracket to token sequence.
    () = nested brackets → tokens 1,2
    <> = PK layer 1 → tokens 3,4
    [] = PK layer 2 → tokens 5,6
    Other (. , _ : ~) = unpaired → token 0
    """
    TOKEN_MAP = {'(': 1, ')': 2, '<': 3, '>': 4, '[': 5, ']': 6,
                 '{': 3, '}': 4,  # treat {} same as <> (rare)
                 '.': 0, ',': 0, '_': 0, ':': 0, '~': 0,
                 '-': 0, 'A': 3, 'a': 4, 'B': 5, 'b': 6}
    return np.array([TOKEN_MAP.get(c, 0) for c in ss], dtype=np.int64)


def make_pk_dataset(sequences, structures, length, n_samples, rng):
    """Build dataset: valid PK vs corrupted PK."""
    vocab = 7  # 0=loop, 1-6 = bracket types
    tokens = np.zeros((n_samples, length), dtype=np.int64)
    labels = np.zeros(n_samples, dtype=np.int64)

    # Filter sequences that fit
    suitable = [(s, ss) for s, ss in zip(sequences, structures)
                if len(ss) <= length and len(ss) >= length // 3]

    if not suitable:
        suitable = [(s, ss) for s, ss in zip(sequences, structures)]

    nv = n_samples // 2
    for j in range(nv):
        idx = rng.choice(len(suitable))
        ss = suitable[idx][1]
        tok = pk_to_tokens(ss)
        tokens[j, :len(tok)] = tok[:length]
        labels[j] = 1

    for j in range(nv, n_samples):
        idx = rng.choice(len(suitable))
        ss = suitable[idx][1]
        tok = pk_to_tokens(ss)
        # Corrupt: flip one bracket
        bracket_pos = np.where(tok > 0)[0]
        if len(bracket_pos) > 0:
            pos = rng.choice(bracket_pos)
            old = tok[pos]
            if old % 2 == 1:  # open
                tok[pos] = old + 1
            else:  # close
                tok[pos] = old - 1
        tokens[j, :len(tok)] = tok[:length]
        labels[j] = 0

    perm = rng.permutation(n_samples)
    return tokens[perm], labels[perm]


def run():
    rng = np.random.default_rng(20260806)
    device = 'cpu'
    sto_dir = '/workspace/sounio/datasets/rna_secondary_structure'

    print("\n" + "=" * 72)
    print("REAL PSEUDOKNOT RNA — Sedenion vs Octonion vs Mixed")
    print("=" * 72)

    # Load RF00008 (hammerhead, 54 nt, simple PK)
    print("\nLoading RF00008 (hammerhead ribozyme)...")
    rf8 = parse_stockholm_pk(os.path.join(sto_dir, 'RF00008.sto'), max_seqs=1000)
    print(f"  {len(rf8)} sequences, lengths: {sorted([len(s) for s, _ in rf8])[:5]}...")

    # Load RF00050 (FMN riboswitch, 127 nt, complex PK)
    print("Loading RF00050 (FMN riboswitch)...")
    rf50 = parse_stockholm_pk(os.path.join(sto_dir, 'RF00050.sto'), max_seqs=1000)
    print(f"  {len(rf50)} sequences, lengths: {sorted([len(s) for s, _ in rf50])[:5]}...")

    all_results = {}

    for fam_name, fam_data, length in [
        ("RF00008-hammerhead", rf8, 64),
        ("RF00050-FMN", rf50, 128),
    ]:
        seqs = [s for s, _ in fam_data]
        structs = [ss for _, ss in fam_data]
        L = 1 << (length - 1).bit_length()

        print(f"\n--- {fam_name} (L={L}) ---")
        tr_t, tr_l = make_pk_dataset(seqs, structs, L, 1024, rng)
        te_t, te_l = make_pk_dataset(seqs, structs, L, 256, rng)
        print(f"  Train: {tr_l.sum()} valid, {1024-tr_l.sum()} corrupted")

        # Show sample PK structures
        for j in range(2):
            if tr_l[j] == 1:
                pk = [c for c in structs[np.random.default_rng(j).integers(0, len(structs))]]
                print(f"  Sample PK: {''.join(pk[:60])}...")

        tr_t = torch.from_numpy(tr_t)
        tr_l = torch.from_numpy(tr_l)
        te_t = torch.from_numpy(te_t)
        te_l = torch.from_numpy(te_l)

        vocab = 7
        models = {
            'OctTree-8':     OctTreeClassifier(vocab, 8, 2, use_oct=True),
            'RealTree-8':    OctTreeClassifier(vocab, 8, 2, use_oct=False),
            'SedenTree-16':  SedenionTreeClassifier(vocab, 16, 2),
            'RealTree-16':   OctTreeClassifier(vocab, 16, 2, use_oct=False),
            'Mixed-OctQuat': MixedOctQuatTree(vocab, 8, 4, 2),
        }

        all_results[fam_name] = {}
        for name, model in models.items():
            model = model.to(device)
            np_p = count_params(model)
            t0 = time.time()
            hist = train_one(model, tr_t, tr_l, te_t, te_l,
                           epochs=50, lr=1e-2, batch_size=32,
                           device=device, name=name)
            dt = time.time() - t0
            final = hist['test_acc'][-1]
            best = max(hist['test_acc'])
            all_results[fam_name][name] = {'params': np_p, 'acc': final, 'best': best}
            print(f"  {name:<14} ({np_p:>5d}p)  test={final:.3f}  best={best:.3f}  ({dt:.0f}s)")

    print(f"\n{'='*72}")
    print("SUMMARY — Real Pseudoknot RNA")
    print(f"{'='*72}")
    for fam in all_results:
        print(f"\n  {fam}:")
        for name in ['OctTree-8', 'RealTree-8', 'SedenTree-16', 'RealTree-16', 'Mixed-OctQuat']:
            a = all_results[fam].get(name, {})
            print(f"    {name:<14}: {a.get('acc',0):.3f}  ({a.get('params',0)}p)")

    outpath = "scripts/research/real_pk_results.json"
    with open(outpath, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    run()
