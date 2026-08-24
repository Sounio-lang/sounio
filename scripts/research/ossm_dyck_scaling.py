#!/usr/bin/env python3
"""
OSSM vs Associative SSM on hierarchical sequence tasks.
PyTorch implementation with autograd — correct gradients, GPU-ready.

THE QUESTION: does non-associative state composition learn hierarchical
structure more efficiently than associative composition at matched parameters?

THE DESIGN: four models, same state dimension, same training:
  OSSM-8:   h_t = tanh(A ⊗ h_{t-1} + E[x_t])  — octonion product (non-assoc)
  Diag-8:   h_t = tanh(A ∘ h_{t-1} + E[x_t])  — element-wise (assoc)
  GRU-8:    standard GRU, hidden=8              — associative baseline
  LSTM-8:   standard LSTM, hidden=8             — associative baseline

TASKS:
  T1 — Dyck-1 depth tracking: valid/invalid bracket nesting, lengths 32-1024
  T2 — Bracketing discrimination: scaled from docs/gpu/BRACKETING_TASK.md
       to variable-length sequences where the label depends on evaluation order
"""

import argparse
import json
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    raise SystemExit("PyTorch required. Install: pip install torch")


# ============================================================
# OCTONION OPERATIONS (PyTorch, differentiable)
# ============================================================

_FANO = [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)]

def _build_sign():
    s = torch.zeros(8,8)
    idx = torch.zeros(8,8,dtype=torch.long)
    for i in range(8):
        s[i,0]=1; idx[i,0]=i; s[0,i]=1; idx[0,i]=i
    for i in range(1,8):
        s[i,i]=-1; idx[i,i]=0
    for a,b,c in _FANO:
        for p,q,r in [(a,b,c),(b,c,a),(c,a,b)]:
            s[p,q]=1; idx[p,q]=r
        for p,q,r in [(b,a,c),(c,b,a),(a,c,b)]:
            s[p,q]=-1; idx[p,q]=r
    return s, idx

_S, _I = _build_sign()

# Build a 8x8x8 multiplication tensor T[i,j,k] = _S[i,j] if _I[i,j]==k else 0.
# Then c[k] = sum_{i,j} a[i] * b[j] * T[i,j,k].
# The left-multiplication matrix L(a)[k,j] = sum_i a[i] * T[i,j,k]
# = (a @ T_perm)[k,j] where T_perm = T.reshape(8, 64).reshape(8, 8, 8) appropriately.
_T = torch.zeros(8, 8, 8)
for i in range(8):
    for j in range(8):
        _T[i, j, int(_I[i, j])] = float(_S[i, j])
# T_kj: shape (8_i, 8_k, 8_j) so that a @ T_kj.reshape(8, 64) gives (8_k * 8_j) flat
_T_KJ = _T.permute(0, 2, 1).contiguous()  # (i, k, j)


def oct_mul_fast(a, b):
    """Batched octonion multiply via left-multiplication matrix.

    a, b: (..., 8) -> (..., 8).  Computes L(a) @ b where L(a) is the
    8x8 left-multiplication matrix determined by the Fano plane.

    Fully vectorized — one bmm, no Python loops over elements.
    """
    dev = a.device
    dt = a.dtype
    Tkj = _T_KJ.to(dev, dt)  # (8, 8, 8): (i, k, j)

    leading = a.shape[:-1]
    a_flat = a.reshape(-1, 8)               # (B, 8)
    # L_flat[b, k, j] = sum_i a_flat[b, i] * Tkj[i, k, j]
    # = a_flat @ Tkj.reshape(8, 8*8) -> (B, 8*8) then reshape to (B, 8, 8)
    L_flat = torch.matmul(a_flat, Tkj.reshape(8, 64)).reshape(-1, 8, 8)  # (B, 8_k, 8_j)

    b_flat = b.reshape(-1, 8, 1)            # (B, 8, 1)
    c_flat = torch.matmul(L_flat, b_flat).squeeze(-1)  # (B, 8)

    return c_flat.reshape(*leading, 8)


# ============================================================
# MODELS
# ============================================================

class OSSMCell(nn.Module):
    """Octonion State-Space Model cell.

    h_t = tanh(A ⊗ h_{t-1} + B @ x_t + b)

    A: (8,) octonion weight — applied via octonion product (non-associative)
    B: (8, vocab) embedding lookup
    """
    def __init__(self, vocab_size, dim=8, n_classes=2, use_octonion=True):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.use_oct = use_octonion

        self.A = nn.Parameter(torch.randn(dim) * 0.3)
        self.B = nn.Parameter(torch.randn(vocab_size, dim) * 0.3)
        self.b = nn.Parameter(torch.zeros(dim))
        self.readout = nn.Linear(dim, n_classes)

    def forward(self, tokens):
        """tokens: (batch, length) -> logits: (batch, n_classes)"""
        batch, length = tokens.shape
        h = torch.zeros(batch, self.dim, device=tokens.device)

        for t in range(length):
            x_t = self.B[tokens[:, t]]  # (batch, dim)
            if self.use_oct:
                A = self.A.unsqueeze(0).expand(batch, -1)  # (batch, dim)
                ah = oct_mul_fast(A, h)
            else:
                ah = self.A.unsqueeze(0) * h  # element-wise (associative)
            h = torch.tanh(ah + x_t + self.b)

        return self.readout(h)


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, dim=8, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.readout = nn.Linear(dim, n_classes)

    def forward(self, tokens):
        x = self.embed(tokens)
        _, h = self.gru(x)
        return self.readout(h.squeeze(0))


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, dim=8, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.lstm = nn.LSTM(dim, dim, batch_first=True)
        self.readout = nn.Linear(dim, n_classes)

    def forward(self, tokens):
        x = self.embed(tokens)
        _, (h, _) = self.lstm(x)
        return self.readout(h.squeeze(0))


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# DYCK LANGUAGE DATA GENERATION
# ============================================================

def _gen_valid_dyck_path(length, n, rng):
    """Generate n valid Dyck-1 paths of given length. Vectorized per-timestep.

    Uses the "never go negative" construction: at each step, if depth is 0
    we must open; if remaining steps == depth we must close; otherwise random.
    Also ensures final depth == 0 by forcing closes at the end.
    """
    assert length % 2 == 0, "Dyck paths need even length"
    tokens = np.zeros((n, length), dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)

    for t in range(length):
        remaining = length - t - 1  # steps remaining AFTER this one
        # Must open if depth == 0 (can't go negative)
        must_open = depth == 0
        # Must close if closing this step is needed to reach 0 by the end
        # i.e. remaining < depth means we need to start closing
        must_close = (remaining < depth) & (depth > 0)
        # Random for the rest
        rand = rng.random(n) < 0.5
        do_open = must_open | (~must_close & rand)
        do_open = do_open & ~must_close  # never override must_close
        do_open = do_open | must_open    # always override with must_open
        tokens[:, t] = np.where(do_open, 1, 2)
        depth += np.where(do_open, 1, -1)

    return tokens


def gen_dyck1(length, batch, rng, invalid_frac=0.5):
    """Dyck-1: single bracket type. Balanced 50/50 valid/invalid.

    Valid sequences are constructed by the never-go-negative method.
    Invalid sequences are random walks that violate nesting.

    Vocab: 0=pad, 1='(', 2=')'
    """
    # Ensure even length for valid Dyck paths
    if length % 2 != 0:
        length += 1
    n_valid = batch // 2
    n_invalid = batch - n_valid

    # Generate valid paths (vectorized per-timestep, not per-sample)
    valid_tokens = _gen_valid_dyck_path(length, n_valid, rng)

    # Generate invalid paths: start with a close at position 0 (depth goes to -1)
    # then fill the rest randomly — guarantees invalidity
    invalid_tokens = np.zeros((n_invalid, length), dtype=np.int64)
    raw = rng.random((n_invalid, length))
    invalid_tokens = np.where(raw < 0.5, 1, 2).astype(np.int64)
    # Force position 0 to be a close — this makes depth go to -1 immediately
    invalid_tokens[:, 0] = 2

    tokens = np.vstack([valid_tokens, invalid_tokens])
    labels = np.concatenate([
        np.ones(n_valid, dtype=np.int64),
        np.zeros(n_invalid, dtype=np.int64),
    ])
    perm = rng.permutation(batch)
    return tokens[perm], labels[perm]


def gen_dyck2(length, batch, rng, invalid_frac=0.5):
    """Dyck-2: two bracket types ()[]. Balanced 50/50.

    Valid: Dyck-1 valid path + random bracket type assignment.
    Invalid: random walk with negative depth.

    Vocab: 0=pad, 1='(', 2=')', 3='[', 4=']'
    """
    if length % 2 != 0:
        length += 1
    n_valid = batch // 2
    n_invalid = batch - n_valid

    # Generate valid Dyck paths (open/close structure)
    base = _gen_valid_dyck_path(length, n_valid, rng)
    # Assign random bracket types
    btype = rng.integers(0, 2, size=(n_valid, length))
    valid_tokens = np.zeros((n_valid, length), dtype=np.int64)
    valid_open = base == 1
    valid_close = base == 2
    valid_tokens[valid_open & (btype == 0)] = 1   # (
    valid_tokens[valid_open & (btype == 1)] = 3   # [
    valid_tokens[valid_close & (btype == 0)] = 2  # )
    valid_tokens[valid_close & (btype == 1)] = 4  # ]

    # Invalid: random walks
    raw = rng.random((n_invalid, length))
    btype2 = rng.integers(0, 2, size=(n_invalid, length))
    invalid_tokens = np.zeros((n_invalid, length), dtype=np.int64)
    is_open2 = raw < 0.5
    invalid_tokens[is_open2 & (btype2 == 0)] = 1
    invalid_tokens[is_open2 & (btype2 == 1)] = 3
    invalid_tokens[~is_open2 & (btype2 == 0)] = 2
    invalid_tokens[~is_open2 & (btype2 == 1)] = 4
    # Ensure at least some go negative
    steps = np.where(is_open2, 1, -1)
    depth = np.cumsum(steps, axis=1)
    all_valid = (depth >= 0).all(axis=1) & (depth[:, -1] == 0)
    for b in np.where(all_valid)[0]:
        t = rng.integers(0, max(length // 4, 1))
        invalid_tokens[b, t] = rng.choice([2, 4])

    tokens = np.vstack([valid_tokens, invalid_tokens])
    labels = np.concatenate([
        np.ones(n_valid, dtype=np.int64),
        np.zeros(n_invalid, dtype=np.int64),
    ])
    perm = rng.permutation(batch)
    return tokens[perm], labels[perm]


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_one(model, train_tok, train_lab, test_tok, test_lab,
              epochs=50, lr=1e-2, batch_size=64, device='cpu', name=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    n = train_tok.shape[0]
    history = {'train_acc': [], 'test_acc': [], 'loss': []}

    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            bt = train_tok[idx].to(device)
            bl = train_lab[idx].to(device)
            opt.zero_grad()
            logits = model(bt)
            loss = criterion(logits, bl)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        history['loss'].append(avg_loss)

        with torch.no_grad():
            train_logits = model(train_tok[:512].to(device))
            train_acc = (train_logits.argmax(-1).cpu() == train_lab[:512]).float().mean().item()
            test_logits = model(test_tok[:512].to(device))
            test_acc = (test_logits.argmax(-1).cpu() == test_lab[:512]).float().mean().item()
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)

        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"    [{name}] ep {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  "
                  f"train={train_acc:.3f}  test={test_acc:.3f}")

    return history


def run(lengths=(32,64,128), n_types=1, epochs=50, train_size=2048,
        test_size=512, seed=20260806, device='cpu'):
    rng = np.random.default_rng(seed)
    gen = gen_dyck1 if n_types == 1 else gen_dyck2
    vocab = 3 if n_types == 1 else 5

    results = {}
    print(f"\n{'='*72}")
    print(f"OSSM vs ASSOCIATIVE SSM — Dyck-{n_types} Scaling")
    print(f"Device: {device}")
    print(f"{'='*72}")

    for L in lengths:
        print(f"\n--- L = {L} ---")
        tr_t, tr_l = gen(L, train_size, rng)
        te_t, te_l = gen(L, test_size, rng)
        tr_t = torch.from_numpy(tr_t)
        tr_l = torch.from_numpy(tr_l)
        te_t = torch.from_numpy(te_t)
        te_l = torch.from_numpy(te_l)

        models = {
            'OSSM-8':  OSSMCell(vocab, 8, 2, use_octonion=True),
            'Diag-8':  OSSMCell(vocab, 8, 2, use_octonion=False),
            'GRU-8':   GRUClassifier(vocab, 8, 2),
            'LSTM-8':  LSTMClassifier(vocab, 8, 2),
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
                'time_sec': round(dt,1),
                'loss_tail': [round(x,4) for x in hist['loss'][-3:]],
            }
            print(f"  {name:<10} ({np_p:>5d}p)  test={final:.3f}  best={best:.3f}  ({dt:.0f}s)")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Model':<12}" + "".join(f"L={L:<8}" for L in lengths))
    print("-"*40)
    for name in ['OSSM-8','Diag-8','GRU-8','LSTM-8']:
        cells = f"{name:<12}"
        for L in lengths:
            cells += f"{results[L][name]['final_test_acc']:<10.3f}"
        print(cells)

    # Save
    out = {str(L): v for L,v in results.items()}
    with open("scripts/research/ossm_dyck_results.json", 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to scripts/research/ossm_dyck_results.json")
    return results


def run_length_generalization(train_len=256, test_lens=(256, 512, 1024, 4096),
                              n_types=1, epochs=100, train_size=4096,
                              test_size=1024, seed=20260806, device='cpu'):
    """Train at one length, test at multiple lengths.

    This is the sharpest test: if the OSSM's non-associative state transition
    tracks bracketing depth natively, it should generalize to longer sequences
    better than the associative baselines, which must extrapolate beyond their
    training distribution.
    """
    rng = np.random.default_rng(seed)
    gen = gen_dyck1 if n_types == 1 else gen_dyck2
    vocab = 3 if n_types == 1 else 5

    print(f"\n{'='*72}")
    print(f"OSSM LENGTH GENERALIZATION — Dyck-{n_types}")
    print(f"Train at L={train_len}, test at L={test_lens}")
    print(f"Device: {device}")
    print(f"{'='*72}")

    # Generate training data at train_len
    tr_t, tr_l = gen(train_len, train_size, rng)
    tr_t = torch.from_numpy(tr_t)
    tr_l = torch.from_numpy(tr_l)

    results = {}

    # Build and train 4 models
    models = {
        'OSSM-8':  OSSMCell(vocab, 8, 2, use_octonion=True),
        'Diag-8':  OSSMCell(vocab, 8, 2, use_octonion=False),
        'GRU-8':   GRUClassifier(vocab, 8, 2),
        'LSTM-8':  LSTMClassifier(vocab, 8, 2),
    }

    for name, model in models.items():
        model = model.to(device)
        np_p = count_params(model)
        print(f"\n  Training {name} ({np_p} params) at L={train_len}...")
        hist = train_one(model, tr_t, tr_l, tr_t, tr_l,
                       epochs=epochs, lr=1e-2, batch_size=64,
                       device=device, name=name)

        # Test at each length
        results[name] = {'params': np_p, 'train_acc': hist['test_acc'][-1]}
        for tl in test_lens:
            te_t, te_l = gen(tl, test_size, rng)
            te_t = torch.from_numpy(te_t)
            te_l = torch.from_numpy(te_l)
            with torch.no_grad():
                logits = model(te_t.to(device))
                acc = (logits.argmax(-1).cpu() == te_l).float().mean().item()
            results[name][f'L={tl}'] = acc
            print(f"    L={tl:>5d}: acc={acc:.3f}")

    # Summary table
    print(f"\n{'='*72}")
    print("LENGTH GENERALIZATION TABLE")
    print(f"{'='*72}")
    header = f"{'Model':<12}" + "".join(f"L={tl:<8}" for tl in test_lens)
    print(header)
    print("-" * len(header))
    for name in models:
        cells = f"{name:<12}"
        for tl in test_lens:
            acc = results[name].get(f'L={tl}', 0)
            cells += f"{acc:<10.3f}"
        print(cells)

    # Key comparison: OSSM vs Diag at each test length
    print(f"\n  OSSM advantage over Diag-8:")
    for tl in test_lens:
        o = results['OSSM-8'].get(f'L={tl}', 0)
        d = results['Diag-8'].get(f'L={tl}', 0)
        diff = o - d
        bar = "+" * int(max(diff, 0) * 100) if diff > 0 else "-" * int(min(-diff, 0) * 100)
        print(f"    L={tl:>5d}: OSSM={o:.3f}  Diag={d:.3f}  Δ={diff:+.3f}  {bar}")

    outpath = "scripts/research/ossm_length_generalization_results.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--lengths', type=int, nargs='+', default=[32,64,128])
    p.add_argument('--n-types', type=int, default=1)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--train-size', type=int, default=2048)
    p.add_argument('--test-size', type=int, default=512)
    p.add_argument('--seed', type=int, default=20260806)
    p.add_argument('--gpu', action='store_true')
    # Length generalization mode
    p.add_argument('--generalize', action='store_true',
                   help='Run length-generalization experiment instead')
    p.add_argument('--train-len', type=int, default=256)
    p.add_argument('--test-lens', type=int, nargs='+', default=[256,512,1024,4096])
    args = p.parse_args()

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    if args.generalize:
        run_length_generalization(
            train_len=args.train_len,
            test_lens=tuple(args.test_lens),
            n_types=args.n_types, epochs=args.epochs,
            train_size=args.train_size, test_size=args.test_size,
            seed=args.seed, device=device)
    else:
        run(lengths=tuple(args.lengths), n_types=args.n_types,
            epochs=args.epochs, train_size=args.train_size,
            test_size=args.test_size, seed=args.seed, device=device)
