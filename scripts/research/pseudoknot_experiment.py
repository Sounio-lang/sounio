#!/usr/bin/env python3
"""
Sedenion and Mixed Octonion-Quaternion Networks for RNA pseudoknots.

WHY SEDENIONS?
  Octonions are ALTERNATIVE: a·(a·b) = (a·a)·b. By Artin, any two
  octonions generate an associative subalgebra. Non-associativity
  only activates at the ternary level.

  Sedenions (dim 16) are NOT alternative. They have ZERO DIVISORS:
  a ≠ 0, b ≠ 0, a·b = 0. And their associator [a,a,b] ≠ 0 in general.
  Binary products can be non-associative.

  RNA pseudoknots are CROSSING: they break tree representability.
  Nested RNA = context-free = single tree = octonion territory.
  Pseudoknots = crossing = beyond context-free = sedenion territory?

  The sedenion zero divisors might model the "information disappearance"
  when base pairs cross — two nonzero contributions that cancel.

WHY MIXED OCT/QUAT?
  Quaternions (dim 4, associative) capture local/nested structure.
  Octonions (dim 8, non-assoc) capture global bracketing.
  Mixed: quaternion branch for nested, octonion branch for crossing.
  The network learns which algebra handles which structural feature.

TASK
  RNA with pseudoknots: dot-bracket with MULTIPLE bracket types.
  () for nested, [] for first PK layer, {} for second PK layer.
  This is a 7-symbol alphabet: ( ) [ ] { } .

  Question: does the sedenion or mixed network capture crossing
  structure that octonion (alone) cannot?
"""

import numpy as np
import json, sys, os, time

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch required")

sys.path.insert(0, os.path.dirname(__file__))
from ossm_dyck_scaling import oct_mul_fast, _T, _T_KJ, count_params, train_one
from mpon_dyck_scaling import OctTreeClassifier


# ============================================================
# SEDENION ARITHMETIC (Cayley-Dickson, dim 16)
# ============================================================

def _build_sedenion_sign():
    """Build the 16×16 sign and index table for sedenion multiplication.

    Sedenion = Cayley-Dickson of octonion. a*b = (a0 + a1*e8)(b0 + b1*e8)
    where a0,a1,b0,b1 are octonions.
    a*b = (a0*b0 - conj(b1)*a1, conj(a0)*b1 + b0*a1)... actually standard CD:
    (a,b)*(c,d) = (ac - d̄b, da + bc̄)

    But for the sign table, sedenion multiplication signs follow the
    Cayley-Dickson construction recursively.
    """
    dim = 16
    sign = np.ones((dim, dim))
    idx = np.zeros((dim, dim), dtype=int)

    # Base case: octonion table (dim 8)
    FANO = [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)]
    for i in range(8):
        idx[0, i] = i; idx[i, 0] = i
        idx[i, i] = 0; sign[i, i] = -1
    for a, b, c in FANO:
        for p, q, r in [(a,b,c),(b,c,a),(c,a,b)]:
            sign[p, q] = 1; idx[p, q] = r
        for p, q, r in [(b,a,c),(c,b,a),(a,c,b)]:
            sign[p, q] = -1; idx[p, q] = r

    # Cayley-Dickson doubling: e_{i+8} = e_i * e_{15} (conceptually)
    # For i,j >= 8: write i=8+a, j=8+b
    # e_{8+a} * e_{8+b} = -e_{a XOR b} (negate conjugate of lower part)
    # e_{8+a} * e_b = e_{8 + (a XOR b)} with sign from octonion(a,b)
    # e_a * e_{8+b} = e_{8 + (a XOR b)} with sign from octonion(a,b)
    for a in range(8):
        for b in range(8):
            # e_{8+a} * e_b
            sa = sign[a, b]; ia = idx[a, b]
            sign[8+a, b] = sa
            idx[8+a, b] = 8 + ia if ia != 0 else 8 + a  # actually: 8 + (a XOR b)
            # Simpler: use the recursive CD formula directly
            pass

    # Actually, let me just build it properly from the CD construction
    sign = np.ones((dim, dim))
    idx = np.zeros((dim, dim), dtype=int)

    # Octonion part (0..7)
    for i in range(8):
        idx[0, i] = i; idx[i, 0] = i
        idx[i, i] = 0; sign[i, i] = -1
    for a, b, c in FANO:
        for p, q, r in [(a,b,c),(b,c,a),(c,a,b)]:
            sign[p, q] = 1; idx[p, q] = r
        for p, q, r in [(b,a,c),(c,b,a),(a,c,b)]:
            sign[p, q] = -1; idx[p, q] = r

    # Sedenion doubling: index i = 8+a pairs with index j
    # Using the Cayley-Dickson formula:
    # (a, b)(c, d) = (ac - conj(d)b, da + bc̄)
    # For basis elements e_i = (e_a, 0) for i<8, e_i = (0, e_a) for i=8+a
    # e_i * e_j where both < 8: already set (octonion)
    # e_{8+a} * e_b: = (0, e_a)(e_b, 0) = (0*e_b - conj(0)*0, e_b*e_a + 0*conj(0))...
    # Actually simpler: e_{8+a} * e_b = e_{8 + (a XOR b)} with sign(a,b) from octonion
    # e_a * e_{8+b} = e_{8 + (a XOR b)} with sign(a,b)
    # e_{8+a} * e_{8+b} = -e_{a XOR b} with sign(a,b)

    for a in range(8):
        for b in range(8):
            c = a ^ b  # XOR gives the index in the octonion part
            s = sign[a, b]  # octonion sign

            # e_{8+a} * e_b  → e_{8+c}, sign s (but a special case for a=0)
            if a == 0:
                sign[8, b] = 1; idx[8, b] = 8 + b
            else:
                sign[8+a, b] = s; idx[8+a, b] = 8 + c if c != 0 else 8 + a
                if c == 0:  # a XOR b = 0 means a=b, so e_a * e_a = -1
                    sign[8+a, b] = -1; idx[8+a, b] = 8 + 0  # hmm
                # Let me be more careful
                sign[8+a, b] = s
                idx[8+a, b] = 8 + c

            # e_a * e_{8+b}
            if b == 0:
                sign[a, 8] = 1; idx[a, 8] = 8 + a
            else:
                sign[a, 8+b] = s
                idx[a, 8+b] = 8 + c

            # e_{8+a} * e_{8+b} → -e_c, sign -s (but -1 for a=b)
            if a == b:
                sign[8+a, 8+b] = 1; idx[8+a, 8+b] = 0  # actually e_{8+a}^2 = +e_0 in sedenions?
                # In Cayley-Dickson: e_{n}^2 = -1 for n >= 1 (including sedenion units)
                sign[8+a, 8+b] = -1; idx[8+a, 8+b] = 0
            else:
                sign[8+a, 8+b] = -s
                idx[8+a, 8+b] = c

    # Fix identity: e_0 * anything = that thing
    for i in range(16):
        sign[0, i] = 1; idx[0, i] = i
        sign[i, 0] = 1; idx[i, 0] = i
    # Fix squares
    for i in range(1, 16):
        sign[i, i] = -1; idx[i, i] = 0

    return torch.tensor(sign, dtype=torch.float32), torch.tensor(idx, dtype=torch.long)


_SED_SIGN, _SED_IDX = _build_sedenion_sign()

# Build L-matrix tensor for sedenion: L(a)[k,j] = sum_i a[i] * T[i,k,j]
_SED_T = torch.zeros(16, 16, 16)
for i in range(16):
    for j in range(16):
        _SED_T[i, j, int(_SED_IDX[i, j])] = float(_SED_SIGN[i, j])
_SED_T_KJ = _SED_T.permute(0, 2, 1).contiguous()


def sed_mul(a, b):
    """Batched sedenion multiply via L-matrix. a,b: (...,16) → (...,16)."""
    leading = a.shape[:-1]
    a_flat = a.reshape(-1, 16)
    Tkj = _SED_T_KJ.to(a.device, a.dtype)
    L_flat = torch.matmul(a_flat, Tkj.reshape(16, 256)).reshape(-1, 16, 16)
    b_flat = b.reshape(-1, 16, 1)
    c_flat = torch.matmul(L_flat, b_flat).squeeze(-1)
    return c_flat.reshape(*leading, 16)


def sed_associator(a, b, c):
    """[a,b,c] = (a*b)*c - a*(b*c) for sedenions."""
    ab = sed_mul(a, b)
    ab_c = sed_mul(ab, c)
    bc = sed_mul(b, c)
    a_bc = sed_mul(a, bc)
    return ab_c - a_bc


# ============================================================
# QUATERNION ARITHMETIC (dim 4, associative)
# ============================================================

_QUAT_SIGN = torch.tensor([
    [1,  1,  1,  1],
    [1, -1,  1, -1],
    [1, -1, -1,  1],
    [1,  1, -1, -1],
], dtype=torch.float32)
_QUAT_IDX = torch.tensor([
    [0,1,2,3],
    [1,0,3,2],
    [2,3,0,1],
    [3,2,1,0],
], dtype=torch.long)

_QUAT_T = torch.zeros(4, 4, 4)
for i in range(4):
    for j in range(4):
        _QUAT_T[i, j, int(_QUAT_IDX[i, j])] = float(_QUAT_SIGN[i, j])
_QUAT_T_KJ = _QUAT_T.permute(0, 2, 1).contiguous()


def quat_mul(a, b):
    """Batched quaternion multiply. a,b: (...,4) → (...,4)."""
    leading = a.shape[:-1]
    a_flat = a.reshape(-1, 4)
    Tkj = _QUAT_T_KJ.to(a.device, a.dtype)
    L_flat = torch.matmul(a_flat, Tkj.reshape(4, 16)).reshape(-1, 4, 4)
    b_flat = b.reshape(-1, 4, 1)
    c_flat = torch.matmul(L_flat, b_flat).squeeze(-1)
    return c_flat.reshape(*leading, 4)


# ============================================================
# SEDENION TREE (dim 16)
# ============================================================

class SedenionTreeClassifier(nn.Module):
    """Balanced tree fold with sedenion product (dim 16, non-alternative)."""
    def __init__(self, vocab_size, dim=16, n_classes=2, max_levels=14):
        super().__init__()
        self.dim = dim
        self.embed = nn.Parameter(torch.randn(vocab_size, dim) * 0.1)
        self.gate_prod = nn.Parameter(torch.ones(max_levels) * 0.5)
        self.gate_res = nn.Parameter(torch.ones(max_levels) * 0.5)
        self.bias = nn.Parameter(torch.zeros(max_levels, dim))
        self.readout = nn.Linear(dim, n_classes)

    def forward(self, tokens):
        x = self.embed[tokens]
        h = x
        level = 0
        while h.shape[1] > 1:
            n = h.shape[1]
            if n % 2 == 1:
                pad = torch.zeros(h.shape[0], 1, self.dim, device=h.device, dtype=h.dtype)
                pad[:, 0, 0] = 1.0
                h = torch.cat([h, pad], dim=1)
                n += 1
            left = h[:, :n//2].reshape(-1, self.dim)
            right = h[:, n//2:].reshape(-1, self.dim)
            prod = sed_mul(left, right)
            res = left + right
            gp = torch.sigmoid(self.gate_prod[level])
            gr = torch.sigmoid(self.gate_res[level])
            h = torch.tanh(gp * prod + gr * res + self.bias[level]).reshape(h.shape[0], n//2, self.dim)
            level += 1
        return self.readout(h[:, 0])


# ============================================================
# MIXED OCT/QUAT NETWORK
# ============================================================

class MixedOctQuatTree(nn.Module):
    """Two parallel tree folds: quaternion (associative) + octonion (non-assoc).

    The quaternion branch handles nested/local structure.
    The octonion branch handles global bracketing.
    Concatenated readout learns which to trust.
    """
    def __init__(self, vocab_size, oct_dim=8, quat_dim=4, n_classes=2, max_levels=14):
        super().__init__()
        self.oct_dim = oct_dim
        self.quat_dim = quat_dim
        self.embed_oct = nn.Parameter(torch.randn(vocab_size, oct_dim) * 0.1)
        self.embed_quat = nn.Parameter(torch.randn(vocab_size, quat_dim) * 0.1)
        # Octonion tree gates
        self.oct_gate_prod = nn.Parameter(torch.ones(max_levels) * 0.5)
        self.oct_gate_res = nn.Parameter(torch.ones(max_levels) * 0.5)
        self.oct_bias = nn.Parameter(torch.zeros(max_levels, oct_dim))
        # Quaternion tree gates
        self.quat_gate_prod = nn.Parameter(torch.ones(max_levels) * 0.5)
        self.quat_gate_res = nn.Parameter(torch.ones(max_levels) * 0.5)
        self.quat_bias = nn.Parameter(torch.zeros(max_levels, quat_dim))
        # Readout from concatenated (oct + quat)
        self.readout = nn.Linear(oct_dim + quat_dim, n_classes)

    def _tree_fold(self, x, dim, mul_fn, gate_prod, gate_res, bias):
        h = x
        level = 0
        while h.shape[1] > 1:
            n = h.shape[1]
            if n % 2 == 1:
                pad = torch.zeros(h.shape[0], 1, dim, device=h.device, dtype=h.dtype)
                pad[:, 0, 0] = 1.0
                h = torch.cat([h, pad], dim=1)
                n += 1
            left = h[:, :n//2].reshape(-1, dim)
            right = h[:, n//2:].reshape(-1, dim)
            prod = mul_fn(left, right)
            res = left + right
            gp = torch.sigmoid(gate_prod[level])
            gr = torch.sigmoid(gate_res[level])
            h = torch.tanh(gp * prod + gr * res + bias[level]).reshape(h.shape[0], n//2, dim)
            level += 1
        return h[:, 0]

    def forward(self, tokens):
        x_oct = self.embed_oct[tokens]
        x_quat = self.embed_quat[tokens]

        h_oct = self._tree_fold(x_oct, self.oct_dim, oct_mul_fast,
                                self.oct_gate_prod, self.oct_gate_res, self.oct_bias)
        h_quat = self._tree_fold(x_quat, self.quat_dim, quat_mul,
                                 self.quat_gate_prod, self.quat_gate_res, self.quat_bias)

        combined = torch.cat([h_oct, h_quat], dim=-1)
        return self.readout(combined)


# ============================================================
# PSEUDOKNOT DATA GENERATION
# ============================================================

def gen_pseudoknot_dataset(length, n_samples, rng, n_bracket_types=2):
    """Generate RNA-like sequences with pseudoknot structure.

    Multi-bracket dot-bracket:
      Type 1: () — nested stems
      Type 2: [] — crossing stems (pseudoknot)
      Optional type 3: {} — second PK layer

    Labels: 1 = valid (all brackets close), 0 = corrupted
    """
    vocab = 1 + 2 * n_bracket_types  # 0=loop, then (open, close) pairs
    tokens = np.zeros((n_samples, length), dtype=np.int64)
    labels = np.zeros(n_samples, dtype=np.int64)
    target_valid = n_samples // 2

    for i in range(n_samples):
        # Build a structure with nested + crossing brackets
        structure = np.zeros(length, dtype=np.int64)
        if i < target_valid:
            # Valid: build a proper structure with PKs
            pos = 0
            while pos < length - 4:
                btype = rng.integers(1, n_bracket_types + 1)  # which bracket type
                open_tok = 2 * (btype - 1) + 1  # 1, 3, 5
                close_tok = 2 * btype            # 2, 4, 6
                stem_len = rng.integers(1, max(2, min(5, (length - pos) // 4)))
                # Decide: nested or crossing
                if rng.random() < 0.5 or btype == 1:
                    # Nested: open...open...close...close
                    close_pos = min(pos + stem_len * 2 + rng.integers(3, 10), length - 1)
                    for k in range(stem_len):
                        if pos + k < length and close_pos - k >= 0:
                            structure[pos + k] = open_tok
                            structure[close_pos - k] = close_tok
                    pos = close_pos + 1
                else:
                    # Crossing: open bracket that crosses a previous one
                    # Find an existing open bracket and place a crossing pair
                    open_positions = np.where(structure > 0)[0]
                    if len(open_positions) > 2:
                        mid = rng.choice(open_positions)
                        close_pos = min(mid + rng.integers(3, 15), length - 1)
                        if structure[close_pos] == 0:
                            structure[mid] = open_tok
                            structure[close_pos] = close_tok
                    pos += stem_len + rng.integers(2, 8)
            labels[i] = 1
        else:
            # Invalid: copy a valid structure then corrupt
            structure = np.zeros(length, dtype=np.int64)
            pos = 0
            while pos < length - 4:
                btype = rng.integers(1, n_bracket_types + 1)
                open_tok = 2 * (btype - 1) + 1
                close_tok = 2 * btype
                stem_len = rng.integers(1, 4)
                close_pos = min(pos + stem_len * 2 + rng.integers(3, 10), length - 1)
                for k in range(stem_len):
                    if pos + k < length and close_pos - k >= 0:
                        structure[pos + k] = open_tok
                        structure[close_pos - k] = close_tok
                pos = close_pos + 1
            # Corrupt: flip one bracket
            bracket_pos = np.where(structure > 0)[0]
            if len(bracket_pos) > 0:
                flip_pos = rng.choice(bracket_pos)
                old = structure[flip_pos]
                # Flip open↔close of same type, or change type
                if old % 2 == 1:  # open
                    structure[flip_pos] = old + 1  # matching close
                else:  # close
                    structure[flip_pos] = old - 1  # matching open
            labels[i] = 0

        tokens[i] = structure

    perm = rng.permutation(n_samples)
    return tokens[perm], labels[perm]


# ============================================================
# EXPERIMENT
# ============================================================

def run(lengths=(32, 64, 128), epochs=50, train_size=2048, test_size=512,
        seed=20260806, device='cpu'):
    rng = np.random.default_rng(seed)

    results = {}
    print("\n" + "=" * 72)
    print("PSEUDOKNOT — Sedenion vs Octonion vs Mixed vs Real")
    print("=" * 72)

    for L in lengths:
        L_tree = 1 << (L - 1).bit_length()
        if L_tree != L:
            L = L_tree

        print(f"\n--- L = {L} ---")
        # Generate pseudoknot data with 2 bracket types (nested + PK)
        vocab = 5  # 0=loop, 1=(, 2=), 3=[, 4=]
        tr_tokens, tr_labels = gen_pseudoknot_dataset(L, train_size, rng, n_bracket_types=2)
        te_tokens, te_labels = gen_pseudoknot_dataset(L, test_size, rng, n_bracket_types=2)
        print(f"  Vocab: {vocab} (0=. 1=( 2=) 3=[ 4=])")
        print(f"  Train: {tr_labels.sum()} valid, {train_size-tr_labels.sum()} corrupted")

        tr_t = torch.from_numpy(tr_tokens)
        tr_l = torch.from_numpy(tr_labels)
        te_t = torch.from_numpy(te_tokens)
        te_l = torch.from_numpy(te_labels)

        models = {
            'OctTree-8':     OctTreeClassifier(vocab, 8, 2, use_oct=True),
            'RealTree-8':    OctTreeClassifier(vocab, 8, 2, use_oct=False),
            'SedenTree-16':  SedenionTreeClassifier(vocab, 16, 2),
            'RealTree-16':   SedenionTreeClassifier(vocab, 16, 2),  # placeholder
            'Mixed-OctQuat': MixedOctQuatTree(vocab, 8, 4, 2),
        }
        # Fix RealTree-16 to use element-wise (not sedenion)
        # We'll use a dim-16 OctTreeClassifier with use_oct=False
        from mpon_dyck_scaling import OctTreeClassifier as OTC
        models['RealTree-16'] = OTC(vocab, 16, 2, use_oct=False)

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
    print("SUMMARY — Pseudoknot Structure")
    print(f"{'='*72}")
    header = f"{'Model':<16}" + "".join(f"L={L:<10}" for L in lengths)
    print(header)
    print("-" * len(header))
    for name in models:
        cells = f"{name:<16}"
        for L in lengths:
            cells += f"{results[L][name]['final_test_acc']:<10.3f}"
        print(cells)

    outpath = "scripts/research/pseudoknot_results.json"
    with open(outpath, 'w') as f:
        json.dump({str(L): v for L, v in results.items()}, f, indent=2)
    print(f"\nSaved to {outpath}")
    return results


if __name__ == '__main__':
    run()
