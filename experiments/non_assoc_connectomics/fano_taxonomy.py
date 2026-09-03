#!/usr/bin/env python3
"""
Strategy ι — Fano Taxonomy: Clustering + phenotype agreement

Loads Fano 7-vectors from ABIDE-I connectomes (n=100), clusters in R^7,
and computes Adjusted Rand Index (ARI) against real ABIDE phenotype columns
with a permutation null.

Outputs:
- PCA visualization
- ARI summary with p-values
- Aligned cohort metadata used for the run
"""

import argparse
import json
import os
import struct
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════════
# Load ABIDE frames and compute Fano 7-vectors
# ═══════════════════════════════════════════════════════════════════════

def load_frames_bin(path, limit=None):
    """Load frames from binary file."""
    with open(path, 'rb') as f:
        n_asd = struct.unpack('<q', f.read(8))[0]
        n_td = struct.unpack('<q', f.read(8))[0]
        n_total = n_asd + n_td

        if limit:
            n_total = min(n_total, limit)

        frames = []
        groups = []
        for i in range(n_total):
            frame_data = f.read(7 * 200 * 8)  # 1400 f64 values
            frame = np.frombuffer(frame_data, dtype=np.float64).reshape(7, 200)
            frames.append(frame)
            groups.append(1 if i < n_asd else 2)
    return np.array(frames), np.array(groups), n_asd, n_td


def extract_frame_from_roi(path):
    """Reproduce the ROI -> Laplacian eigenframe extraction used in abide_preprocess.py."""
    ts = np.loadtxt(path)
    if ts.ndim != 2 or ts.shape[0] < 20 or ts.shape[1] < 8:
        raise RuntimeError(f"Unexpected ROI shape in {path}: {ts.shape}")

    n_rois = ts.shape[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.corrcoef(ts.T)
    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 0.0)
    adj = np.maximum(corr, 0.0)
    deg = adj.sum(axis=1)
    lap = np.diag(deg) - adj
    eigenvalues, eigenvectors = np.linalg.eigh(lap)
    if len(eigenvalues) < 8:
        raise RuntimeError(f"Not enough eigenvalues in {path}")

    frame = eigenvectors[:, 1:8].T
    if n_rois < 200:
        padded = np.zeros((7, 200))
        padded[:, :n_rois] = frame
        frame = padded
    elif n_rois > 200:
        frame = frame[:, :200]
    return frame


def iter_cached_subject_rows(pheno_path, roi_dir):
    """Yield phenotypic rows that have a cached ROI file."""
    pheno_df = pd.read_csv(pheno_path)
    roi_dir = Path(roi_dir)
    kept = []
    for _, row in pheno_df.iterrows():
        fid = str(row.get("FILE_ID", ""))
        if not fid or fid in ("no_filename", "nan"):
            continue
        roi_path = roi_dir / f"{fid}_rois_cc200.1D"
        if not roi_path.exists():
            continue
        row = row.copy()
        row["ROI_PATH"] = str(roi_path)
        kept.append(row)
    return pd.DataFrame(kept)


def build_aligned_metadata(pheno_path, roi_dir, n_asd, n_td, frames_index_path=None):
    """
    Reconstruct the exact cohort ordering used by scripts/research/abide_preprocess.py.

    Preferred source of truth:
    1. frames_index.txt (if present)
    2. The preprocessor contract: iterate phenotypic rows, keep cached ROI files,
       append all ASD first, then all TD.
    """
    pheno_df = iter_cached_subject_rows(pheno_path, roi_dir)

    if frames_index_path and Path(frames_index_path).exists():
        ordered = []
        by_fid = pheno_df.set_index("FILE_ID", drop=False)
        with open(frames_index_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                label, fid = line.split("\t", 1)
                if fid not in by_fid.index:
                    raise RuntimeError(f"FILE_ID {fid} from frames_index missing in phenotypic.csv")
                row = by_fid.loc[fid]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                ordered.append(row)
        meta = pd.DataFrame(ordered).reset_index(drop=True)
    else:
        ordered = []
        for dx_group, target_count in ((1, n_asd), (2, n_td)):
            subset = pheno_df[pheno_df["DX_GROUP"] == dx_group]
            kept = []
            for _, row in subset.iterrows():
                kept.append(row)
                if len(kept) == target_count:
                    break
            if len(kept) != target_count:
                raise RuntimeError(
                    f"Unable to reconstruct {target_count} subjects for DX_GROUP={dx_group}; found {len(kept)}"
                )
            ordered.extend(kept)
        meta = pd.DataFrame(ordered).reset_index(drop=True)

    if len(meta) != n_asd + n_td:
        raise RuntimeError(
            f"Aligned metadata length mismatch: expected {n_asd + n_td}, got {len(meta)}"
        )

    if int((meta["DX_GROUP"] == 1).sum()) != n_asd or int((meta["DX_GROUP"] == 2).sum()) != n_td:
        raise RuntimeError("Aligned metadata does not match frames.bin class counts")

    meta = meta.copy()
    meta["site_code"] = pd.Categorical(meta["SITE_ID"]).codes
    meta["sex_code"] = pd.Categorical(meta["SEX"]).codes
    meta["age_quartile"] = pd.qcut(
        meta["AGE_AT_SCAN"],
        q=4,
        labels=False,
        duplicates="drop",
    ).astype(int)
    return meta


def select_site_balanced_metadata(pheno_path, roi_dir, target_subjects):
    """
    Build a diagnosis-balanced, near-uniform per-site cohort from cached ROI files.

    Selection rule:
    - only sites with both ASD and TD cached subjects are eligible
    - allocate one ASD/TD pair per site in round-robin order until target reached
    - within each site/diagnosis bucket, preserve original phenotypic row order
    """
    if target_subjects % 2 != 0:
        raise ValueError("target_subjects must be even for site-balanced selection")

    pheno_df = iter_cached_subject_rows(pheno_path, roi_dir)
    per_site = {}
    for site, site_df in pheno_df.groupby("SITE_ID", sort=True):
        asd_rows = list(site_df[site_df["DX_GROUP"] == 1].iterrows())
        td_rows = list(site_df[site_df["DX_GROUP"] == 2].iterrows())
        pairs = min(len(asd_rows), len(td_rows))
        if pairs == 0:
            continue
        per_site[site] = {
            "asd_rows": asd_rows,
            "td_rows": td_rows,
            "pairs": pairs,
        }

    if not per_site:
        raise RuntimeError("No sites with both ASD and TD cached subjects")

    target_pairs = target_subjects // 2
    capacity_pairs = sum(bucket["pairs"] for bucket in per_site.values())
    if capacity_pairs < target_pairs:
        raise RuntimeError(
            f"Insufficient site-balanced capacity: requested {target_pairs} pairs, have {capacity_pairs}"
        )

    ordered_sites = sorted(
        per_site.keys(),
        key=lambda site: (-per_site[site]["pairs"], site),
    )
    allocated = {site: 0 for site in ordered_sites}
    allocated_pairs = 0
    while allocated_pairs < target_pairs:
        progressed = False
        for site in ordered_sites:
            if allocated[site] >= per_site[site]["pairs"]:
                continue
            allocated[site] += 1
            allocated_pairs += 1
            progressed = True
            if allocated_pairs == target_pairs:
                break
        if not progressed:
            raise RuntimeError("Round-robin allocation stalled before reaching target_pairs")

    selected_rows = []
    for site in ordered_sites:
        n_pairs = allocated[site]
        if n_pairs == 0:
            continue
        selected_rows.extend(row for _, row in per_site[site]["asd_rows"][:n_pairs])
        selected_rows.extend(row for _, row in per_site[site]["td_rows"][:n_pairs])

    meta = pd.DataFrame(selected_rows).reset_index(drop=True)
    if len(meta) != target_subjects:
        raise RuntimeError(
            f"Site-balanced metadata length mismatch: expected {target_subjects}, got {len(meta)}"
        )

    meta = meta.copy()
    meta["site_code"] = pd.Categorical(meta["SITE_ID"]).codes
    meta["sex_code"] = pd.Categorical(meta["SEX"]).codes
    meta["age_quartile"] = pd.qcut(
        meta["AGE_AT_SCAN"],
        q=4,
        labels=False,
        duplicates="drop",
    ).astype(int)
    return meta


def load_frames_for_metadata(meta):
    """Compute eigenframes directly from the cached ROI files for the selected cohort."""
    frames = []
    groups = []
    for i, row in enumerate(meta.itertuples(index=False), start=1):
        if i % 20 == 0:
            print(f"  frame {i}/{len(meta)}")
        frames.append(extract_frame_from_roi(row.ROI_PATH))
        groups.append(int(row.DX_GROUP))
    return np.array(frames), np.array(groups)

def compute_fano_7vector(frame):
    """
    Compute per-subject Fano 7-vector from 7×200 eigenvector frame.

    For each C(7,3)=35 eigenvector triple (a,b,c), compute triple product,
    measure alignment with the 7 Fano basis lines, accumulate strength.
    Normalize to 6-simplex.
    """
    fano_lines = [
        (0, 1, 2), (0, 3, 4), (0, 5, 6),
        (1, 3, 5), (1, 4, 6), (2, 3, 6), (2, 4, 5),
    ]

    fano_strength = np.zeros(7)

    # Iterate over C(7,3)=35 triples
    for a in range(7):
        for b in range(a+1, 7):
            for c in range(b+1, 7):
                # Triple product: Σ_nodes frame[a,n] * frame[b,n] * frame[c,n]
                tp = np.sum(frame[a] * frame[b] * frame[c])
                tp_sq = tp * tp

                # Check alignment with Fano lines
                triple = tuple(sorted([a, b, c]))
                for line_idx, fano_triple in enumerate(fano_lines):
                    fano_sorted = tuple(sorted(fano_triple))
                    if triple == fano_sorted:
                        fano_strength[line_idx] += tp_sq

    # Normalize to simplex
    total = np.sum(fano_strength)
    if total < 1e-12:
        total = 1.0

    return fano_strength / total

def parse_args():
    parser = argparse.ArgumentParser(description="Run Fano taxonomy analysis on ABIDE frames")
    parser.add_argument(
        "--frames-path",
        default="artifacts/research/abide/frames.bin",
        help="Path to frames.bin",
    )
    parser.add_argument(
        "--cohort-mode",
        choices=["frames_bin", "site_balanced"],
        default="frames_bin",
        help="Which cohort construction path to use",
    )
    parser.add_argument(
        "--pheno-path",
        default="/tmp/abide_pilot/phenotypic.csv",
        help="Path to ABIDE phenotypic.csv",
    )
    parser.add_argument(
        "--roi-dir",
        default="/tmp/abide_pilot",
        help="Directory containing cached *_rois_cc200.1D files",
    )
    parser.add_argument(
        "--frames-index-path",
        default="artifacts/research/abide/frames_index.txt",
        help="Optional frames_index.txt written by abide_preprocess.py",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/non_assoc_connectomics",
        help="Directory for summary artifacts",
    )
    parser.add_argument(
        "--output-stem",
        default="fano_taxonomy",
        help="Stem for output artifact filenames",
    )
    parser.add_argument(
        "--target-subjects",
        type=int,
        default=100,
        help="Target cohort size for derived cohort modes such as site_balanced",
    )
    parser.add_argument(
        "--n-perms",
        type=int,
        default=1000,
        help="Number of permutation draws for each phenotype ARI null",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.cohort_mode == "frames_bin":
        print("Loading frames.bin...")
        frames, groups, n_asd, n_td = load_frames_bin(args.frames_path)
        n_total = len(frames)
        print(f"  Loaded {n_total} subjects ({n_asd} ASD, {n_td} TD)")

        print("Aligning phenotype metadata to frames.bin order...")
        meta = build_aligned_metadata(
            args.pheno_path,
            args.roi_dir,
            n_asd,
            n_td,
            args.frames_index_path,
        )
    else:
        print(f"Building site-balanced cohort (target_subjects={args.target_subjects})...")
        meta = select_site_balanced_metadata(
            args.pheno_path,
            args.roi_dir,
            args.target_subjects,
        )
        n_asd = int((meta["DX_GROUP"] == 1).sum())
        n_td = int((meta["DX_GROUP"] == 2).sum())
        n_total = len(meta)
        print(f"  Selected {n_total} subjects ({n_asd} ASD, {n_td} TD)")
        print("Computing eigenframes for selected cohort...")
        frames, groups = load_frames_for_metadata(meta)

    aligned_meta_path = output_dir / f"{args.output_stem}_cohort.csv"
    meta[
        ["SUB_ID", "FILE_ID", "DX_GROUP", "SITE_ID", "AGE_AT_SCAN", "SEX", "age_quartile"]
    ].to_csv(aligned_meta_path, index=False)
    print(f"  Aligned metadata written to {aligned_meta_path}")

    # Compute Fano 7-vectors
    print("Computing Fano 7-vectors...")
    fano_vectors = np.zeros((n_total, 7))
    for i in range(n_total):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{n_total}")
        fano_vectors[i] = compute_fano_7vector(frames[i])

    print(f"Fano 7-vectors computed. Mean simplex values: {fano_vectors.mean(0)}")

    # ═══════════════════════════════════════════════════════════════════
    # Clustering in R^7
    # ═══════════════════════════════════════════════════════════════════

    print("\nClustering analysis...")

    # Standardize vectors (though they're already on simplex)
    scaler = StandardScaler()
    vectors_scaled = scaler.fit_transform(fano_vectors)

    # K-means: find optimal k via silhouette score
    silhouette_scores = []
    kmeans_models = {}

    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(vectors_scaled)
        sil_score = silhouette_score(vectors_scaled, labels)
        silhouette_scores.append(sil_score)
        kmeans_models[k] = (km, labels)
        print(f"  k={k}: silhouette={sil_score:.4f}")

    optimal_k = np.argmax(silhouette_scores) + 2
    print(f"\nOptimal k: {optimal_k} (silhouette={silhouette_scores[optimal_k-2]:.4f})")

    optimal_km, cluster_labels = kmeans_models[optimal_k]

    # ═══════════════════════════════════════════════════════════════════
    # ARI tests with permutation null
    # ═══════════════════════════════════════════════════════════════════

    print(f"\nARI phenotype agreement tests ({args.n_perms} permutations)...")

    # Prepare phenotype data
    phenotype_map = {}

    phenotype_map["DX_GROUP"] = meta["DX_GROUP"].to_numpy()
    phenotype_map["site"] = meta["site_code"].to_numpy()
    phenotype_map["age_quartile"] = meta["age_quartile"].to_numpy()
    phenotype_map["sex"] = meta["sex_code"].to_numpy()

    ari_results = {}
    n_perms = args.n_perms

    for pheno_name, pheno_values in phenotype_map.items():
        observed_ari = adjusted_rand_score(pheno_values, cluster_labels)

        # Permutation null
        null_aris = []
        rng = np.random.default_rng(42)
        for _ in range(n_perms):
            perm_labels = rng.permutation(cluster_labels)
            perm_ari = adjusted_rand_score(pheno_values, perm_labels)
            null_aris.append(perm_ari)

        null_aris = np.array(null_aris)
        p_value = (np.count_nonzero(null_aris >= observed_ari) + 1) / (n_perms + 1)

        ari_results[pheno_name] = {
            'observed_ari': float(observed_ari),
            'p_value': float(p_value),
            'null_mean': float(null_aris.mean()),
            'null_std': float(null_aris.std()),
        }

        print(f"  {pheno_name}: ARI={observed_ari:.4f}, p={p_value:.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # PCA visualization
    # ═══════════════════════════════════════════════════════════════════

    print("\nGenerating PCA visualization...")
    pca = PCA(n_components=2)
    vectors_pca = pca.fit_transform(fano_vectors)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: DX_GROUP
    scatter1 = axes[0].scatter(vectors_pca[:, 0], vectors_pca[:, 1],
                               c=groups, cmap='coolwarm', s=50, alpha=0.6)
    axes[0].set_title('PCA: Colored by DX_GROUP (1=ASD, 2=TD)')
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.colorbar(scatter1, ax=axes[0])

    # Right: Algebra-discovered clusters
    scatter2 = axes[1].scatter(vectors_pca[:, 0], vectors_pca[:, 1],
                               c=cluster_labels, cmap='viridis', s=50, alpha=0.6)
    axes[1].set_title(f'PCA: Colored by Algebra Clusters (k={optimal_k})')
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.colorbar(scatter2, ax=axes[1])

    fig.suptitle('Octonion Fano-Taxonomy: Clinical Labels vs Algebraic Structure')
    plt.tight_layout()
    pca_path = output_dir / f"{args.output_stem}_pca.png"
    plt.savefig(pca_path, dpi=150)
    print(f"  Saved: {pca_path}")

    # ═══════════════════════════════════════════════════════════════════
    # Summary output
    # ═══════════════════════════════════════════════════════════════════

    summary = {
        'n_total': n_total,
        'n_asd': n_asd,
        'n_td': n_td,
        'cohort_mode': args.cohort_mode,
        'optimal_k': int(optimal_k),
        'silhouette_optimal': float(silhouette_scores[optimal_k - 2]),
        'silhouette_scores': [float(s) for s in silhouette_scores],
        'cluster_sizes': [int(np.sum(cluster_labels == i)) for i in range(optimal_k)],
        'ari_results': ari_results,
        'pca_explained_variance': [float(v) for v in pca.explained_variance_ratio_[:2]],
        'mean_simplex': [float(v) for v in fano_vectors.mean(axis=0)],
        'cohort_metadata': {
            'site_counts': {str(k): int(v) for k, v in meta["SITE_ID"].value_counts().sort_index().items()},
            'sex_counts': {str(int(k)): int(v) for k, v in meta["SEX"].value_counts().sort_index().items()},
            'age_quartile_counts': {
                str(int(k)): int(v) for k, v in meta["age_quartile"].value_counts().sort_index().items()
            },
        },
    }

    summary_path = output_dir / f"{args.output_stem}_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("\nResults:")
    print(json.dumps(summary, indent=2))

    print("\n✓ Strategy ι complete.")
    print(f"  PCA figure: {pca_path}")
    print(f"  Summary JSON: {summary_path}")
    print(f"  Cohort CSV: {aligned_meta_path}")

if __name__ == '__main__':
    main()
