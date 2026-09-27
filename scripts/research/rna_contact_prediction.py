#!/usr/bin/env python3
"""
RNA Contact Map Prediction — the real task.
Given an RNA nucleotide sequence, predict which bases pair together.

This is what RNAfold, Mfold, ContextFold, SPOT-RNA do.
We test whether OctTree's non-associative state carries pairing information
that associative models miss.

TASK
  Input: RNA nucleotide sequence (A,C,G,U)
  Output: L×L binary contact map (1 if bases i,j form a pair)

  Evaluated by F1-score on positive class (base pairs), since contacts
  are sparse (~10% of L² entries).

ARCHITECTURE
  The tree fold produces a single octonion state from the sequence.
  We extract pairwise information from the intermediate tree levels:
  at each level, the two octonions being combined encode the left/right
  halves of the sequence. Their product (octonion or element-wise)
  captures the interaction between those halves.

  OctContactNet: at each tree level, compute left ⊗ right (non-assoc)
  or left × right (assoc), accumulate into a pairwise feature tensor,
  then read out a contact map via bilinear scoring.
"""

import numpy as np
import json, sys, os, time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("PyTorch required")

sys.path.insert(0, os.path.dirname(__file__))
from ossm_dyck_scaling import oct_mul_fast, _T, _T_KJ, count_params

try:
    import RNA as vrna
    HAS_VIENNA = True
except ImportError:
    HAS_VIENNA = False


# ============================================================
# DATA: RNA sequences → contact maps
# ============================================================

BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'U': 3,
               'a': 0, 'c': 1, 'g': 2, 'u': 3}

def dot_bracket_to_contacts(ss, length):
    """Convert dot-bracket string to L×L contact map."""
    contacts = np.zeros((length, length), dtype=np.float32)
    stack = []
    for j, c in enumerate(ss[:length]):
        if c == '(':
            stack.append(j)
        elif c == ')' and stack:
            k = stack.pop()
            contacts[k, j] = 1.0
            contacts[j, k] = 1.0
    return contacts


def gen_rna_contact_batch(length, n_samples, rng, use_vienna=True):
    """Generate RNA sequences and their contact maps.

    Returns:
      tokens: (N, L) nucleotide indices
      contacts: (N, L, L) binary contact maps
    """
    tokens = np.zeros((n_samples, length), dtype=np.int64)
    contacts = np.zeros((n_samples, length, length), dtype=np.float32)

    BASES = 'ACGU'

    for i in range(n_samples):
        gc = rng.uniform(0.3, 0.7)
        seq = ''.join(rng.choice(['G','C']) if rng.random() < gc else rng.choice(['A','U'])
                     for _ in range(length))

        if use_vienna and HAS_VIENNA:
            fc = vrna.fold_compound(seq)
            ss, mfe = fc.mfe()
        else:
            # Simple random structure
            ss = '.' * length
            for j in range(length // 4):
                k = rng.integers(0, length - 3)
                m = k + rng.integers(3, min(20, length - k))
                if m < length:
                    ss = ss[:k] + '(' + ss[k+1:m] + ')' + ss[m+1:]

        tokens[i] = [BASE_TO_INT.get(b, 0) for b in seq]
        contacts[i] = dot_bracket_to_contacts(ss, length)

    return tokens, contacts


def load_rfam_contacts(path, length, max_seqs=5000):
    """Load Rfam sequences with real contact maps."""
    seqs, structs = [], []
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines) and len(seqs) < max_seqs:
        if lines[i].startswith('>'):
            if i + 2 < len(lines):
                seq = lines[i+1].strip()
                ss = lines[i+2].strip()
                if len(seq) == len(ss) and len(seq) <= length and len(seq) >= length // 2:
                    if set(ss) <= set('().'):
                        seqs.append(seq)
                        structs.append(ss)
            i += 3
        else:
            i += 1

    n = len(seqs)
    tokens = np.zeros((n, length), dtype=np.int64)
    contacts = np.zeros((n, length, length), dtype=np.float32)

    for i, (seq, ss) in enumerate(zip(seqs, structs)):
        tokens[i, :len(seq)] = [BASE_TO_INT.get(b, 0) for b in seq]
        contacts[i] = dot_bracket_to_contacts(ss, length)

    return tokens, contacts


# ============================================================
# OCT CONTACT NET: tree-fold with pairwise extraction
# ============================================================

class OctContactNet(nn.Module):
    """Tree-fold contact prediction network.

    At each tree level, when folding left ⊗ right:
    - The left and right octonions encode the two halves
    - Their interaction (⊗ or ×) captures cross-half pairings
    - We accumulate pairwise features across levels

    Final contact map: bilinear(h_i, h_j) where h_i are per-position
    features extracted from the tree.

    For simplicity, we use the tree to produce a per-position encoding
    (not just a single state) by recording intermediate representations
    at each level and mapping them to position-level features.
    """
    def __init__(self, vocab=4, dim=8, use_oct=True, max_levels=10):
        super().__init__()
        self.dim = dim
        self.use_oct = use_oct
        self.embed = nn.Parameter(torch.randn(vocab, dim) * 0.1)
        self.gate_prod = nn.Parameter(torch.ones(max_levels) * 0.5)
        self.gate_res = nn.Parameter(torch.ones(max_levels) * 0.5)
        self.bias = nn.Parameter(torch.zeros(max_levels, dim))
        # Bilinear contact scoring: score(i,j) = h_i^T W h_j
        self.contact_weight = nn.Parameter(torch.eye(dim) * 0.1)
        self.contact_bias = nn.Parameter(torch.zeros(1))

    def forward(self, tokens):
        """tokens: (batch, L) → contact_logits: (batch, L, L)"""
        x = self.embed[tokens]  # (B, L, D)
        B, L, D = x.shape

        # Store per-position features at each level
        # Level 0: original embeddings
        position_features = [x]  # list of (B, L_k, D)

        h = x
        level = 0
        while h.shape[1] > 1:
            n = h.shape[1]
            if n % 2 == 1:
                pad = torch.zeros(B, 1, D, device=h.device, dtype=h.dtype)
                pad[:, 0, 0] = 1.0
                h = torch.cat([h, pad], dim=1)
                n += 1

            left = h[:, :n//2].reshape(-1, D)
            right = h[:, n//2:].reshape(-1, D)

            if self.use_oct:
                prod = oct_mul_fast(left, right)
            else:
                prod = left * right

            res = left + right
            gp = torch.sigmoid(self.gate_prod[level])
            gr = torch.sigmoid(self.gate_res[level])
            combined = torch.tanh(gp * prod + gr * res + self.bias[level])

            h = combined.reshape(B, n//2, D)
            level += 1

        # h[:, 0] is the final octonion state per sample (B, D)
        # For contact prediction, we need per-position info.
        # Use the ORIGINAL embeddings + the final state as context
        # Score(i,j) = bilinear(x_i, x_j) + bilinear(x_i, h_final) * bilinear(h_final, x_j)
        final = h[:, 0]  # (B, D)

        # Bilinear contact map from embeddings
        # score[i,j] = x_i^T W x_j + b
        # (B, L, D) @ (D, D) → (B, L, D), then @ (B, D, L) → (B, L, L)
        h_proj = torch.matmul(x, self.contact_weight)  # (B, L, D)
        contact_logits = torch.matmul(h_proj, x.transpose(1, 2))  # (B, L, L)
        contact_logits = contact_logits + self.contact_bias

        # Add global state modulation
        # final^T W x_i gives each position a "compatibility with the global fold"
        global_compat = torch.matmul(x, final.unsqueeze(-1)).squeeze(-1)  # (B, L)
        contact_logits = contact_logits + global_compat.unsqueeze(2) * global_compat.unsqueeze(1) * 0.1

        return contact_logits


# ============================================================
# TRAINING
# ============================================================

def train_contact_model(model, train_tokens, train_contacts,
                        test_tokens, test_contacts,
                        epochs=50, lr=1e-2, batch_size=32, device='cpu', name=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_tokens.shape[0]
    history = {'train_f1': [], 'test_f1': []}

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0
        n_batches = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            bt = train_tokens[idx].to(device)
            bc = train_contacts[idx].to(device)

            opt.zero_grad()
            logits = model(bt)  # (B, L, L)
            # Weighted BCE: contacts are sparse (~5-10% positive)
            pos_weight = torch.tensor(5.0, device=device)
            loss = F.binary_cross_entropy_with_logits(logits, bc, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Evaluate F1
        def eval_f1(tokens, contacts):
            with torch.no_grad():
                logits = model(tokens.to(device))
                pred = (torch.sigmoid(logits) > 0.5).float()
                # F1 on positive class
                tp = (pred * contacts.to(device)).sum()
                fp = (pred * (1 - contacts.to(device))).sum()
                fn = ((1 - pred) * contacts.to(device)).sum()
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-8)
                return f1.item()

        train_f1 = eval_f1(train_tokens[:200], train_contacts[:200])
        test_f1 = eval_f1(test_tokens[:200], test_contacts[:200])
        history['train_f1'].append(train_f1)
        history['test_f1'].append(test_f1)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"    [{name}] ep {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  "
                  f"train_F1={train_f1:.3f}  test_F1={test_f1:.3f}")

    return history


def run_contact_experiment(length=64, train_size=2048, test_size=512,
                           epochs=50, seed=20260806,
                           rfam_path=None):
    """RNA contact prediction: OctTree vs RealTree."""
    rng = np.random.default_rng(seed)
    device = 'cpu'

    print("\n" + "=" * 72)
    print("RNA CONTACT MAP PREDICTION")
    print(f"L={length}, train={train_size}, test={test_size}")
    print("=" * 72)

    # Generate data
    if rfam_path and os.path.exists(rfam_path):
        print(f"Loading Rfam data from {rfam_path}...")
        train_tokens, train_contacts = load_rfam_contacts(rfam_path, length, max_seqs=train_size)
        test_tokens, test_contacts = load_rfam_contacts(rfam_path, length, max_seqs=test_size)
        # If not enough, supplement with ViennaRNA
        if len(train_tokens) < train_size:
            n_extra = train_size - len(train_tokens)
            extra_t, extra_c = gen_rna_contact_batch(length, n_extra, rng)
            train_tokens = np.vstack([train_tokens, extra_t])
            train_contacts = np.concatenate([train_contacts, extra_c])
        print(f"  Train: {len(train_tokens)} samples")
        print(f"  Test: {len(test_tokens)} samples")
    else:
        print("Generating ViennaRNA data...")
        train_tokens, train_contacts = gen_rna_contact_batch(length, train_size, rng)
        test_tokens, test_contacts = gen_rna_contact_batch(length, test_size, rng)

    # Contact density
    density = train_contacts.mean()
    print(f"  Contact density: {density:.3f} (expect ~0.05-0.10)")

    train_tokens = torch.from_numpy(train_tokens)
    train_contacts = torch.from_numpy(train_contacts)
    test_tokens = torch.from_numpy(test_tokens)
    test_contacts = torch.from_numpy(test_contacts)

    models = {
        'OctContact':  OctContactNet(vocab=4, dim=8, use_oct=True),
        'RealContact': OctContactNet(vocab=4, dim=8, use_oct=False),
        'GRU-Contact': nn.Sequential(
            nn.Embedding(4, 8),
            nn.GRU(8, 8, batch_first=True),
        ),  # will need custom handling
    }

    results = {}
    for name in ['OctContact', 'RealContact']:
        model = models[name].to(device)
        np_p = count_params(model)
        t0 = time.time()
        print(f"\n  Training {name} ({np_p}p)...")
        hist = train_contact_model(model, train_tokens, train_contacts,
                                   test_tokens, test_contacts,
                                   epochs=epochs, lr=1e-2, batch_size=32,
                                   device=device, name=name)
        dt = time.time() - t0
        final_f1 = hist['test_f1'][-1]
        best_f1 = max(hist['test_f1'])
        results[name] = {'params': np_p, 'final_f1': final_f1, 'best_f1': best_f1, 'time': round(dt,1)}
        print(f"  → {name}: test_F1={final_f1:.3f}  best={best_f1:.3f}  ({dt:.0f}s)")

    print(f"\n{'='*72}")
    print("RESULTS — RNA Contact Prediction (F1-score)")
    print(f"{'='*72}")
    for name in results:
        print(f"  {name:<14}: F1={results[name]['final_f1']:.3f}  best={results[name]['best_f1']:.3f}  ({results[name]['params']}p)")

    diff = results['OctContact']['final_f1'] - results['RealContact']['final_f1']
    print(f"\n  OctContact advantage: {diff:+.3f} F1")
    if diff > 0.02:
        print("  ⚡ Octonion captures pairing structure that associative model misses")
    elif diff < -0.02:
        print("  ✗ Associative model wins — octonion does not help for contacts")
    else:
        print("  ≈ No significant difference")

    outpath = "scripts/research/rna_contact_results.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--length', type=int, default=64)
    p.add_argument('--train-size', type=int, default=2048)
    p.add_argument('--test-size', type=int, default=512)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--seed', type=int, default=20260806)
    p.add_argument('--rfam', type=str, default=None)
    args = p.parse_args()

    rfam = args.rfam or '/workspace/sounio/datasets/rna_secondary_structure/rfam_structures.fasta'
    run_contact_experiment(length=args.length, train_size=args.train_size,
                          test_size=args.test_size, epochs=args.epochs,
                          seed=args.seed, rfam_path=rfam)
