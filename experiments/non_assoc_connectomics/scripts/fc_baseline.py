#!/usr/bin/env python3
"""ABIDE CC200 full functional-connectivity LOSO baseline (numpy-only).

Purpose: establish the INPUT CEILING. If full FC beats chance with a plain
linear classifier, the ~50% of every model on the 8x8 manifest is a
feature-engineering artifact (200->8 pooling), not a model limitation.

Linear kernel ridge (RLS) classifier, leave-one-site-out CV, balanced accuracy.
"""
import os, glob, csv, math
import numpy as np

ABIDE = "/workspace/sounio/artifacts/research/abide"
PHENO = os.path.join(ABIDE, "phenotypic.csv")

# 1) labels: FILE_ID -> (dx, site).  DX_GROUP 1=ASD, 2=control
pheno = {}
with open(PHENO, newline="") as fh:
    for row in csv.DictReader(fh):
        fid = (row.get("FILE_ID") or "").strip()
        dx = (row.get("DX_GROUP") or "").strip()
        site = (row.get("SITE_ID") or "").strip()
        if fid and fid != "no_filename" and dx in ("1", "2"):
            pheno[fid] = (1 if dx == "1" else 0, site)

# 2) build FC features
def fc_upper(ts):
    # ts: (T, 200) -> z-score cols -> corr 200x200 -> fisher-z upper triangle
    ts = ts - ts.mean(0, keepdims=True)
    sd = ts.std(0, keepdims=True); sd[sd < 1e-8] = 1.0
    ts = ts / sd
    c = np.corrcoef(ts.T)
    c = np.clip(c, -0.999999, 0.999999)
    c = np.arctanh(c)  # fisher-z
    iu = np.triu_indices(c.shape[0], k=1)
    return c[iu]

X, y, sites = [], [], []
files = sorted(glob.glob(os.path.join(ABIDE, "*_rois_cc200.1D")))
for f in files:
    fid = os.path.basename(f)[:-len("_rois_cc200.1D")]
    if fid not in pheno:
        continue
    try:
        ts = np.loadtxt(f, comments="#")
    except Exception:
        continue
    if ts.ndim != 2 or ts.shape[1] < 50 or ts.shape[0] < 30:
        continue
    feat = fc_upper(ts)
    if not np.all(np.isfinite(feat)):
        continue
    X.append(feat.astype(np.float32)); y.append(pheno[fid][0]); sites.append(pheno[fid][1])

X = np.asarray(X, dtype=np.float64)
y = np.asarray(y); sites = np.asarray(sites)
print(f"subjects={len(y)}  features={X.shape[1]}  ASD={int((y==1).sum())} ctrl={int((y==0).sum())}  sites={len(set(sites))}")

# 3) LOSO with linear-kernel ridge (RLS): predict via dual form
#    f(x) = k(x, Xtr) (K + lambda I)^-1 ytr ,  K = Xtr Xtr^T
lam = 1.0
usites = sorted(set(sites))
bals = []
for s in usites:
    te = sites == s; tr = ~te
    if te.sum() == 0 or len(set(y[tr])) < 2 or len(set(y[te])) < 2:
        continue
    Xtr, Xte = X[tr], X[te]
    mu = Xtr.mean(0, keepdims=True); sd = Xtr.std(0, keepdims=True); sd[sd < 1e-8] = 1.0
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    ytr = np.where(y[tr] == 1, 1.0, -1.0)
    K = Xtr @ Xtr.T
    n = K.shape[0]
    alpha = np.linalg.solve(K + lam * n * np.eye(n), ytr)
    scores = (Xte @ Xtr.T) @ alpha
    pred = (scores >= 0).astype(int)
    yte = y[te]
    tpr = ((pred == 1) & (yte == 1)).sum() / max(1, (yte == 1).sum())
    tnr = ((pred == 0) & (yte == 0)).sum() / max(1, (yte == 0).sum())
    bal = 50.0 * (tpr + tnr)
    bals.append(bal)
    print(f"  site {s:12s} n={int(te.sum()):3d}  bal={bal:5.1f}%")

bals = np.array(bals)
print(f"\nLOSO mean balanced accuracy (full CC200 FC, linear): {bals.mean():.2f}% +/- {bals.std():.2f}%  over {len(bals)} sites")
print(f"COMPARE: 8x8-manifest models were all ~50% (transformer 50.8, O-SSM raw 49.95)")
