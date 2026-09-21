#!/usr/bin/env python3
"""
C3 — Confirmatory controls (Rfam OctTree lane).

Implements the frozen control ladder from task_freeze.md:

  CountBaseline     logistic on bracket-count features (~10 p) — Task B must
                    defeat it (validity gate), Task A must not (sanity).
  GatedTreeClassifier with product=
    "oct"     octonion product (Fano structure constants, 64 nonzeros) —
              re-exported for paired-seed consistency with OctTreeClassifier.
    "real"    componentwise product (diagonal associative, 8 nonzeros).
    "cliff"   Clifford Cl(3,0) ≅ M₂(ℂ) product — fixed, dense, associative,
              iso-sparse with the octonion table (64 nonzeros), iso-parametric
              (182 p). THE decisive control: separates "any dense algebra
              coupling" from "non-associative coupling specifically".
    "learned" free full-rank bilinear tensor ℝ^{8×8×8} (512 p) + same
              gates/bias/readout (~694 p) — ceiling for "any learnable
              product".

The Cl(3,0) structure tensor is computed from the defining blade rules
(e_i² = +1, e_i e_j = −e_j e_i) and self-tested at import: associativity on
random triples (fail closed), non-commutativity present, exactly 64 nonzero
structure constants.

All tree classifiers share the exact OctTreeClassifier reduction (half-split
levels, per-level sigmoid gates, bias, tanh, readout) so any difference is
attributable to the product and nothing else.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch required")

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ossm_dyck_scaling import oct_mul_fast  # noqa: E402


# ---------------------------------------------------------------
# Clifford Cl(3,0) structure constants, computed from first principles
# ---------------------------------------------------------------

def cl3_structure_tensor() -> np.ndarray:
    """(i, k, j) tensor with out_k = Σ_ij a_i b_j T[i, k, j].

    Basis blades are bitmasks of {e1, e2, e3}: 0=1, 1=e1, 2=e2, 3=e12,
    4=e3, 5=e13, 6=e23, 7=e123. Cl(3,0): e_i² = +1.
    Blade product: sign = (−1)^{#{(p,q): p∈A, q∈B, p>q}}; result blade A XOR B.
    """
    T = np.zeros((8, 8, 8), dtype=np.float64)
    for a in range(8):
        for b in range(8):
            swaps = 0
            for p in range(3):
                if (a >> p) & 1:
                    for q in range(3):
                        if (b >> q) & 1 and p > q:
                            swaps += 1
            sign = -1.0 if swaps % 2 else 1.0
            k = a ^ b
            T[a, k, b] = sign
    return T


def cl3_mul(a: torch.Tensor, b: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """out = Σ_ij a_i b_j T[i, :, j]  (batched)."""
    return torch.einsum("bi,bj,ikj->bk", a, b, T)


def _selftest_clifford() -> None:
    """Fail closed if Cl(3,0) is not what we claim."""
    T = cl3_structure_tensor()
    nz = int((T != 0).sum())
    if nz != 64:
        raise ValueError(f"Cl(3,0) tensor must have 64 nonzeros like the octonion table, got {nz}")
    tt = torch.from_numpy(T)
    rng = np.random.default_rng(7)
    for _ in range(50):
        a = torch.from_numpy(rng.standard_normal(8))[None, :]
        b = torch.from_numpy(rng.standard_normal(8))[None, :]
        c = torch.from_numpy(rng.standard_normal(8))[None, :]
        ab_c = cl3_mul(cl3_mul(a, b, tt), c, tt)
        a_bc = cl3_mul(a, cl3_mul(b, c, tt), tt)
        if not torch.allclose(ab_c, a_bc, atol=1e-10):
            raise ValueError("Cl(3,0) associativity self-test failed")
        if torch.allclose(cl3_mul(a, b, tt), cl3_mul(b, a, tt), atol=1e-12):
            raise ValueError("Cl(3,0) unexpectedly commutative")


_selftest_clifford()


# ---------------------------------------------------------------
# Gated tree classifier with pluggable product
# ---------------------------------------------------------------

class GatedTreeClassifier(nn.Module):
    """Same reduction as OctTreeClassifier (preserved exploratory code);
    only the internal product differs."""

    def __init__(self, vocab_size, dim=8, n_classes=2, product="oct", max_levels=14):
        super().__init__()
        if product not in ("oct", "real", "cliff", "learned"):
            raise ValueError(f"unknown product: {product}")
        self.dim = dim
        self.product = product
        self.max_levels = max_levels

        self.embed = nn.Parameter(torch.randn(vocab_size, dim) * 0.1)
        self.gate_prod = nn.Parameter(torch.ones(max_levels) * 0.5)
        self.gate_res = nn.Parameter(torch.ones(max_levels) * 0.5)
        self.bias = nn.Parameter(torch.zeros(max_levels, dim))
        self.readout = nn.Linear(dim, n_classes)

        if product == "cliff":
            self.register_buffer("cliffT", torch.from_numpy(cl3_structure_tensor()).float())
        elif product == "learned":
            self.bilin = nn.Parameter(torch.randn(dim, dim, dim) * 0.1)

    def _prod(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.product == "oct":
            return oct_mul_fast(left, right)
        if self.product == "real":
            return left * right
        if self.product == "cliff":
            return torch.einsum("bi,bj,ikj->bk", left, right, self.cliffT)
        return torch.einsum("bi,bj,ikj->bk", left, right, self.bilin)

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
            left = h[:, : n // 2].reshape(-1, self.dim)
            right = h[:, n // 2 :].reshape(-1, self.dim)

            prod = self._prod(left, right)
            res = left + right

            gp = torch.sigmoid(self.gate_prod[level])
            gr = torch.sigmoid(self.gate_res[level])
            combined = gp * prod + gr * res + self.bias[level]
            combined = torch.tanh(combined)

            h = combined.reshape(h.shape[0], n // 2, self.dim)
            level += 1

        return self.readout(h[:, 0])


# ---------------------------------------------------------------
# Count baseline — the validity gate for Task B
# ---------------------------------------------------------------

class CountBaseline(nn.Module):
    """Logistic regression on bracket-count features.

    Features: [n_open, n_close, n_open - n_close, n_open + n_close].
    Padding and '.' share token 0, so counts of structural symbols are the
    only count signal available — exactly what Task B destroys.
    """

    def __init__(self, n_classes=2):
        super().__init__()
        self.lin = nn.Linear(4, n_classes)

    def forward(self, tokens):
        n1 = (tokens == 1).sum(1).float()
        n2 = (tokens == 2).sum(1).float()
        feats = torch.stack([n1, n2, n1 - n2, n1 + n2], dim=1)
        return self.lin(feats)


# ---------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------

def main():
    from ossm_dyck_scaling import count_params

    vocab, n_classes = 3, 2
    models = {
        "OctTree-8 (oct)": GatedTreeClassifier(vocab, 8, n_classes, product="oct"),
        "RealTree-8 (real)": GatedTreeClassifier(vocab, 8, n_classes, product="real"),
        "CliffTree-8 (cliff)": GatedTreeClassifier(vocab, 8, n_classes, product="cliff"),
        "LearnedBilinTree (learned)": GatedTreeClassifier(vocab, 8, n_classes, product="learned"),
        "CountBaseline": CountBaseline(n_classes),
    }
    for name, m in models.items():
        n = count_params(m)
        x = torch.randint(0, vocab, (4, 128))
        out = m(x)
        assert out.shape == (4, n_classes), f"{name}: bad output shape {out.shape}"
        print(f"{name:<28} params={n:<5d} forward OK")


if __name__ == "__main__":
    main()
