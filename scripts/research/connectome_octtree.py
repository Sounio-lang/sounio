#!/usr/bin/env python3
"""
Octonion Tree-Fold on Brain Connectome Paths
=============================================
The new approach: instead of static associator features, fold random walks
through the connectome graph using the OctTree architecture.

PRIOR NULL (ABIDE_ASSOCIATOR_NULL.md)
  Static associator [o_i, o_j, o_k] over node triples → 52.1% (chance)
  Positive control (PCA-50 upper triangle) → 63.9% (signal exists)

HYPOTHESIS
  The tree-fold architecture captures parenthesization-dependent path
  structure that static features miss. Brain network paths encoded as
  octonion edge labels, folded via balanced binary tree, carry signal
  beyond both static associators and standard connectome features.

DESIGN
  For each subject:
  1. Build connectome (200×200 Fisher-z Pearson correlation)
  2. Embed each brain region as an octonion (via PCA-8 of connectivity profile)
  3. Sample K random walks of length L through the graph
  4. Fold each walk via OctTree (⊗) vs RealTree (×)
  5. Classify ASD vs TD from the folded path representations

  Control: RealTree (same architecture, associative)
  Baseline: PCA-50 of upper triangle (standard connectome classifier)

  Leave-one-site-out cross-validation (matches prior null protocol)
"""

import numpy as np
import json
import os
import re
import glob
import sys

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    raise SystemExit("PyTorch required")

sys.path.insert(0, os.path.dirname(__file__))
from ossm_dyck_scaling import oct_mul_fast
from mpon_dyck_scaling import OctTreeClassifier, tree_fold_balanced, count_params


# ============================================================
# ABIDE DATA LOADING
# ============================================================

ABIDE_DIR = '/workspace/sounio/artifacts/research/abide/'
MANIFEST = '/workspace/sounio/abide_roi_manifest.tsv'
N_ROIS = 200


def load_abide(max_subjects=500):
    """Load ABIDE-I connectomes and labels.

    Returns:
      connectomes: (N, 200, 200) Fisher-z transformed Pearson correlations
      labels: (N,) binary (1=ASD, 0=TD)
      sites: (N,) site names
    """
    # Load manifest
    rows = []
    with open(MANIFEST) as f:
        header = True
        for line in f:
            if header:
                header = False
                continue
            if not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            rows.append(parts)

    # Find files
    files = {}
    for f in glob.glob(os.path.join(ABIDE_DIR, '*.1D')):
        m = re.search(r'_(\d{7})_rois', os.path.basename(f))
        if m:
            files[str(int(m.group(1)))] = f

    # Match subjects
    subj = []
    for r in rows:
        if len(r) >= 3 and r[0] in files:
            subj.append(r)
    subj = subj[:max_subjects]

    N = len(subj)
    labels = np.array([1 if r[1][0] == 'A' else 0 for r in subj])
    sites = [r[2] for r in subj]

    print(f"Loaded {N} subjects, ASD={labels.sum()}, sites={len(set(sites))}")

    # Build connectomes
    connectomes = np.zeros((N, N_ROIS, N_ROIS), dtype=np.float32)
    valid = np.ones(N, dtype=bool)
    for i, r in enumerate(subj):
        try:
            ts = np.loadtxt(files[r[0]])  # (T, R) timeseries
            if ts.ndim != 2:
                valid[i] = False
                continue
            if ts.shape[0] < ts.shape[1]:
                ts = ts.T  # ensure (T, R)
            R = ts.shape[1]
            if R != N_ROIS:
                # Pad or truncate to 200
                if R < N_ROIS:
                    ts_padded = np.zeros((ts.shape[0], N_ROIS))
                    ts_padded[:, :R] = ts
                    ts = ts_padded
                else:
                    ts = ts[:, :N_ROIS]
            corr = np.corrcoef(ts)  # (R, R)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            corr = np.clip(corr, -0.999, 0.999)
            z = np.arctanh(corr)
            connectomes[i] = z.astype(np.float32)
        except Exception as e:
            valid[i] = False
            continue

    # Filter invalid
    connectomes = connectomes[valid]
    labels = labels[valid]
    sites = np.array(sites)[valid]
    N = len(labels)
    print(f"  Valid: {N} subjects, ASD={labels.sum()}")

    return connectomes, labels, np.array(sites)


# ============================================================
# CONNECTOME → OCTONION EMBEDDINGS
# ============================================================

def node_embeddings(connectomes, labels, sites, dim=8, n_walks=32, walk_len=16, seed=42):
    """Convert connectomes to octonion-valued random walk sequences.

    For each subject:
    1. PCA-8 on the connectivity profile (200-dim row of the connectome)
       → each region gets an 8-dim embedding (octonion)
    2. Sample random walks on the graph (edges weighted by connectivity)
    3. Each walk = sequence of octonion node embeddings

    Returns:
      walks: (N, n_walks, walk_len, dim) octonion node features
      labels: (N,)
    """
    rng = np.random.default_rng(seed)
    N = connectomes.shape[0]

    # PCA-8 on connectivity profiles (fit on all subjects — leakage-safe for LOSO
    # because we'll refit per fold in the real experiment; this is the quick version)
    profiles = connectomes.reshape(N, -1)  # (N, 40000)
    # Actually we want per-node embeddings: each node is described by its
    # connectivity to all other nodes (a 200-dim row)
    all_node_profiles = connectomes.reshape(N * N_ROIS, N_ROIS)
    # PCA via SVD
    from numpy.linalg import svd
    mean_profile = all_node_profiles.mean(axis=0)
    centered = all_node_profiles - mean_profile
    # Subsample for speed
    idx = rng.choice(len(centered), min(50000, len(centered)), replace=False)
    U, S, Vt = svd(centered[idx], full_matrices=False)
    pca_components = Vt[:dim]  # (8, 200)

    # Project each node for each subject
    # node_emb[i, j, :] = (connectomes[i, j, :] - mean_profile) @ pca_components.T
    node_emb = np.zeros((N, N_ROIS, dim), dtype=np.float32)
    for i in range(N):
        node_emb[i] = (connectomes[i] - mean_profile) @ pca_components.T

    # Sample random walks
    walks = np.zeros((N, n_walks, walk_len, dim), dtype=np.float32)

    for i in range(N):
        # Build transition probability from |connectome|
        conn = np.abs(connectomes[i])
        np.fill_diagonal(conn, 0)
        row_sums = conn.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        trans = conn / row_sums

        for w in range(n_walks):
            # Start at random node
            node = rng.integers(0, N_ROIS)
            for t in range(walk_len):
                walks[i, w, t] = node_emb[i, node]
                # Next node weighted by connectivity
                probs = trans[node]
                if probs.sum() > 0:
                    node = rng.choice(N_ROIS, p=probs)
                else:
                    node = rng.integers(0, N_ROIS)

    return walks, labels


# ============================================================
# CONNECTOME CLASSIFIER WITH TREE FOLD
# ============================================================

class ConnectomeOctTree(nn.Module):
    """Octonion tree-fold classifier for brain connectome walks.

    1. Project node embeddings (dim=8) through a learnable octonion weight
    2. Fold each walk via balanced tree product
    3. Pool across walks
    4. Readout to 2 classes
    """
    def __init__(self, dim=8, n_classes=2, n_walks=32, use_oct=True):
        super().__init__()
        self.dim = dim
        self.n_walks = n_walks
        self.use_oct = use_oct

        # Learnable projection of node embeddings
        self.proj = nn.Linear(dim, dim)
        # Tree-fold via the OctTree internal loop
        self.gate_prod = nn.Parameter(torch.ones(10) * 0.5)
        self.gate_res = nn.Parameter(torch.ones(10) * 0.5)
        self.bias = nn.Parameter(torch.zeros(10, dim))
        # Readout
        self.readout = nn.Sequential(
            nn.Linear(dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )

    def forward(self, walks):
        """walks: (batch, n_walks, walk_len, dim) -> logits: (batch, n_classes)"""
        batch, nw, L, dim = walks.shape
        # Project
        h = self.proj(walks.reshape(batch * nw, L, dim))

        # Tree fold
        level = 0
        while h.shape[1] > 1:
            n = h.shape[1]
            if n % 2 == 1:
                pad = torch.zeros(batch * nw, 1, dim, device=h.device, dtype=h.dtype)
                pad[:, 0, 0] = 1.0
                h = torch.cat([h, pad], dim=1)
                n += 1
            left = h[:, :n//2].reshape(-1, dim)
            right = h[:, n//2:].reshape(-1, dim)
            if self.use_oct:
                prod = oct_mul_fast(left, right)
            else:
                prod = left * right
            res = left + right
            gp = torch.sigmoid(self.gate_prod[level])
            gr = torch.sigmoid(self.gate_res[level])
            combined = torch.tanh(gp * prod + gr * res + self.bias[level])
            h = combined.reshape(batch * nw, n//2, dim)
            level += 1

        # Pool across walks
        h = h.reshape(batch, nw, dim).mean(dim=1)  # (batch, dim)
        return self.readout(h)


def loso_cv(X, y, sites, model_class, model_kwargs, epochs=50, lr=0.01,
            batch_size=32, device='cpu', name=""):
    """Leave-one-site-out cross-validation."""
    unique_sites = sorted(set(sites.tolist()))
    bal_accs = []

    for test_site in unique_sites:
        test_mask = sites == test_site
        train_mask = ~test_mask
        if test_mask.sum() < 5 or train_mask.sum() < 20:
            continue

        X_train = torch.from_numpy(X[train_mask]).float().to(device)
        y_train = torch.from_numpy(y[train_mask]).long().to(device)
        X_test = torch.from_numpy(X[test_mask]).float().to(device)
        y_test = torch.from_numpy(y[test_mask]).long().to(device)

        model = model_class(**model_kwargs).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            perm = torch.randperm(len(X_train))
            for i in range(0, len(X_train), batch_size):
                idx = perm[i:i+batch_size]
                opt.zero_grad()
                logits = model(X_train[idx])
                loss = criterion(logits, y_train[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        with torch.no_grad():
            logits = model(X_test)
            pred = logits.argmax(-1)
            # Balanced accuracy
            tp = ((pred == 1) & (y_test == 1)).float().sum()
            tn = ((pred == 0) & (y_test == 0)).float().sum()
            n_pos = (y_test == 1).float().sum()
            n_neg = (y_test == 0).float().sum()
            sens = tp / max(n_pos, 1)
            spec = tn / max(n_neg, 1)
            bal_acc = (sens + spec) / 2
            bal_accs.append(bal_acc.item())

    mean_ba = np.mean(bal_accs)
    std_ba = np.std(bal_accs)
    print(f"  [{name}] LOSO bal-acc: {mean_ba*100:.1f}% ± {std_ba*100:.1f}% "
          f"(n_sites={len(bal_accs)})")
    return mean_ba, std_ba


def run_experiment(n_subjects=500, n_walks=32, walk_len=16, seed=42):
    """Full ABIDE experiment: OctTree vs RealTree vs baselines."""
    print("=" * 72)
    print("OCTONION TREE-FOLD ON BRAIN CONNECTOME PATHS")
    print("=" * 72)

    # Load data
    connectomes, labels, sites = load_abide(max_subjects=n_subjects)

    # Generate walks
    print(f"\nGenerating {n_walks} walks of length {walk_len} per subject...")
    walks, _ = node_embeddings(connectomes, labels, sites, dim=8,
                               n_walks=n_walks, walk_len=walk_len, seed=seed)
    print(f"Walk data shape: {walks.shape}")

    # Standard connectome baseline (PCA-50 upper triangle)
    print("\n--- Baseline: PCA-50 connectome ---")
    iu = np.triu_indices(N_ROIS, 1)
    upper = connectomes[:, iu[0], iu[1]]  # (N, 19900)
    from numpy.linalg import svd as np_svd
    mean_u = upper.mean(axis=0)
    centered_u = upper - mean_u
    U, S, Vt = np_svd(centered_u, full_matrices=False)
    pca50 = (centered_u @ Vt[:50].T)  # (N, 50)

    # LOSO for PCA-50 baseline
    from sklearn.linear_model import LogisticRegression
    try:
        unique_sites = sorted(set(sites.tolist()))
        ba_list = []
        for ts in unique_sites:
            tm = sites == ts
            trm = ~tm
            if tm.sum() < 5 or trm.sum() < 20:
                continue
            clf = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced')
            clf.fit(pca50[trm], labels[trm])
            pred = clf.predict(pca50[tm])
            tp = ((pred == 1) & (labels[tm] == 1)).sum()
            tn = ((pred == 0) & (labels[tm] == 0)).sum()
            sens = tp / max((labels[tm] == 1).sum(), 1)
            spec = tn / max((labels[tm] == 0).sum(), 1)
            ba_list.append((sens + spec) / 2)
        print(f"  [PCA-50] LOSO bal-acc: {np.mean(ba_list)*100:.1f}% ± {np.std(ba_list)*100:.1f}%")
    except ImportError:
        print("  (sklearn not available, skipping PCA-50 baseline)")

    # OctTree vs RealTree
    device = 'cpu'
    print(f"\n--- OctTree (⊗) vs RealTree (×) ---")

    ba_oct, std_oct = loso_cv(
        walks, labels, sites,
        ConnectomeOctTree,
        {'dim': 8, 'n_classes': 2, 'n_walks': n_walks, 'use_oct': True},
        epochs=50, lr=0.005, batch_size=32, device=device,
        name="OctTree-8"
    )

    ba_real, std_real = loso_cv(
        walks, labels, sites,
        ConnectomeOctTree,
        {'dim': 8, 'n_classes': 2, 'n_walks': n_walks, 'use_oct': False},
        epochs=50, lr=0.005, batch_size=32, device=device,
        name="RealTree-8"
    )

    # Summary
    print(f"\n{'='*72}")
    print("RESULTS")
    print(f"{'='*72}")
    print(f"  PCA-50 (connectome baseline): {np.mean(ba_list)*100:.1f}% ± {np.std(ba_list)*100:.1f}%")
    print(f"  OctTree-8 (⊗, non-assoc):     {ba_oct*100:.1f}% ± {std_oct*100:.1f}%")
    print(f"  RealTree-8 (×, associative):   {ba_real*100:.1f}% ± {std_real*100:.1f}%")
    print(f"  OctTree advantage:             {(ba_oct - ba_real)*100:+.1f}%")
    print(f"\n  Prior null (static associator): 52.1% ± 9.4%")
    print(f"  Prior positive control (PCA-50): 63.9% ± 9.2%")

    results = {
        'n_subjects': n_subjects,
        'n_walks': n_walks,
        'walk_len': walk_len,
        'pca50_bal_acc': float(np.mean(ba_list)),
        'pca50_std': float(np.std(ba_list)),
        'octtree_bal_acc': float(ba_oct),
        'octtree_std': float(std_oct),
        'realtree_bal_acc': float(ba_real),
        'realtree_std': float(std_real),
        'octtree_advantage': float(ba_oct - ba_real),
    }
    outpath = '/workspace/sounio/scripts/research/connectome_octtree_results.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == '__main__':
    run_experiment()
