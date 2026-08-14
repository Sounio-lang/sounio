#!/usr/bin/env python3
"""
OctTree on Natural Language Constituent Parsing (Universal Dependencies).

TASK: Given a sentence, classify whether its parse tree is "well-formed"
(valid constituency bracketing) vs "corrupted" (bracket violation).

This is the NL analog of Dyck/RNA: sentences have genuine bracketing
structure from constituency parses. The question: does OctTree beat
RealTree on REAL language bracketing?

APPROACH:
1. Parse UD CoNLL-U to extract dependency trees
2. Convert dependency trees to bracket sequences (nested representation)
3. Task: valid tree vs corrupted tree (one edge flipped)
4. Compare OctTree vs RealTree

The hypothesis: natural language has deeper and more complex nesting
than Dyck-1, and the OctTree advantage should appear at shorter lengths.
"""

import numpy as np
import sys, os, time

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch required")

sys.path.insert(0, os.path.dirname(__file__))
from ossm_dyck_scaling import train_one, count_params, GRUClassifier
from mpon_dyck_scaling import OctTreeClassifier


def parse_conllu(path, max_sentences=5000):
    """Parse CoNLL-U file, return list of (tokens, dep_heads, upos_tags)."""
    sentences = []
    tokens, heads, upos = [], [], []
    
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            if not line:
                if tokens:
                    sentences.append((tokens, heads, upos))
                    tokens, heads, upos = [], [], []
                    if len(sentences) >= max_sentences:
                        break
                continue
            
            parts = line.split('\t')
            if len(parts) >= 8:
                # Skip multi-word tokens
                if '-' in parts[0] or '.' in parts[0]:
                    continue
                tokens.append(parts[1])  # word form
                head = int(parts[6]) if parts[6].isdigit() else 0
                heads.append(head)
                upos.append(parts[3])  # UPOS tag
    
    if tokens:
        sentences.append((tokens, heads, upos))
    
    return sentences


def dep_tree_to_brackets(tokens, heads):
    """Convert dependency tree to a bracket sequence.
    
    Uses the tree structure: for each word, its dependents are nested.
    Produces a sequence of open/close brackets representing the tree traversal.
    
    The bracket sequence encodes the TREE STRUCTURE, not the words.
    """
    n = len(tokens)
    # Build children mapping
    children = {i: [] for i in range(n + 1)}  # 0 = ROOT
    for i, h in enumerate(heads):
        children[h].append(i + 1)  # 1-indexed
    
    # DFS traversal producing bracket sequence
    brackets = []
    
    def dfs(node):
        if node > 0:  # not ROOT
            brackets.append(1)  # open
        for child in sorted(children.get(node, [])):
            dfs(child)
        if node > 0:
            brackets.append(2)  # close
    
    dfs(0)  # start from ROOT
    return np.array(brackets, dtype=np.int64)


def make_nl_dataset(sentences, length, n_samples, rng):
    """Build dataset: valid parse brackets vs corrupted."""
    vocab = 3  # 0=pad, 1=open, 2=close
    tokens_arr = np.zeros((n_samples, length), dtype=np.int64)
    labels = np.zeros(n_samples, dtype=np.int64)
    
    # Filter sentences that produce bracket sequences
    suitable = []
    for tokens, heads, upos in sentences:
        brackets = dep_tree_to_brackets(tokens, heads)
        if 8 <= len(brackets) <= length:
            suitable.append(brackets)
    
    if not suitable:
        # Fallback: use all
        for tokens, heads, upos in sentences:
            brackets = dep_tree_to_brackets(tokens, heads)
            if len(brackets) >= 4:
                suitable.append(brackets)
    
    n_valid = min(n_samples // 2, len(suitable))
    
    for j in range(n_valid):
        idx = rng.choice(len(suitable))
        b = suitable[idx]
        tokens_arr[j, :len(b)] = b[:length]
        labels[j] = 1
    
    for j in range(n_valid, n_samples):
        idx = rng.choice(len(suitable))
        b = suitable[idx].copy()
        # Corrupt: flip one bracket
        bracket_pos = np.where(b > 0)[0]
        if len(bracket_pos) > 0:
            pos = rng.choice(bracket_pos)
            b[pos] = 3 - b[pos]  # swap 1↔2
        tokens_arr[j, :len(b)] = b[:length]
        labels[j] = 0
    
    perm = rng.permutation(n_samples)
    return tokens_arr[perm], labels[perm]


def run_nl_experiment(conllu_path, lengths=(16, 32, 64, 128), epochs=50,
                      train_size=2048, test_size=512, seed=20260806):
    rng = np.random.default_rng(seed)
    device = 'cpu'
    vocab = 3
    
    print("\n" + "=" * 72)
    print("OctTree vs RealTree on NATURAL LANGUAGE parse trees")
    print(f"Data: {os.path.basename(conllu_path)}")
    print("=" * 72)
    
    # Load and parse
    sentences = parse_conllu(conllu_path, max_sentences=10000)
    print(f"Loaded {len(sentences)} sentences")
    
    # Show sample bracket sequences
    for i in range(3):
        tokens, heads, upos = sentences[i]
        brackets = dep_tree_to_brackets(tokens, heads)
        bstr = ''.join('(' if b == 1 else ')' for b in brackets[:30])
        print(f"  '{' '.join(tokens[:6])}...' → {bstr}... ({len(brackets)} brackets)")
    
    results = {}
    
    for L in lengths:
        L_tree = 1 << (L - 1).bit_length()
        if L_tree != L:
            L = L_tree
        
        print(f"\n--- L = {L} ---")
        tr_t, tr_l = make_nl_dataset(sentences, L, train_size, rng)
        te_t, te_l = make_nl_dataset(sentences, L, test_size, rng)
        
        n_open = (tr_t == 1).sum(1).mean()
        n_close = (tr_t == 2).sum(1).mean()
        n_pad = (tr_t == 0).sum(1).mean()
        print(f"  Avg per sample: {n_open:.0f} open, {n_close:.0f} close, {n_pad:.0f} pad")
        
        tr_t = torch.from_numpy(tr_t)
        tr_l = torch.from_numpy(tr_l)
        te_t = torch.from_numpy(te_t)
        te_l = torch.from_numpy(te_l)
        
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
            results[L][name] = {'params': np_p, 'acc': final, 'best': best, 'time': round(dt,1)}
            print(f"  {name:<14} ({np_p:>5d}p)  test={final:.3f}  best={best:.3f}  ({dt:.0f}s)")
    
    # Summary
    print(f"\n{'='*72}")
    print("SUMMARY — Natural Language Parse Trees")
    print(f"{'='*72}")
    header = f"{'Model':<14}" + "".join(f"L={L:<10}" for L in lengths)
    print(header)
    print("-" * len(header))
    for name in models:
        cells = f"{name:<14}"
        for L in lengths:
            cells += f"{results[L][name]['acc']:<10.3f}"
        print(cells)
    
    print(f"\n  OctTree vs RealTree:")
    for L in lengths:
        o = results[L]['OctTree-8']['acc']
        r = results[L]['RealTree-8']['acc']
        diff = o - r
        bar = "+" * int(max(diff, 0) * 50) if diff > 0 else "-" * int(min(-diff, 0) * 50)
        print(f"    L={L:>5d}: Δ={diff:+.3f}  {bar}")
    
    outpath = "scripts/research/nl_parsing_results.json"
    with open(outpath, 'w') as f:
        json.dump({str(L): v for L, v in results.items()}, f, indent=2)
    print(f"\nSaved to {outpath}")
    return results


if __name__ == '__main__':
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument('--conllu', type=str, 
                   default='/workspace/sounio/datasets/constituency_parsing/UD_English-EWT-master/en_ewt-ud-train.conllu')
    p.add_argument('--lengths', type=int, nargs='+', default=[16, 32, 64, 128])
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--train-size', type=int, default=2048)
    p.add_argument('--test-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=20260806)
    args = p.parse_args()
    
    run_nl_experiment(args.conllu, tuple(args.lengths), args.epochs,
                     args.train_size, args.test_size, args.seed)
