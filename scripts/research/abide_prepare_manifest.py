#!/usr/bin/env python3
"""
ABIDE-I Manifest Preparation for brain_ossm_abide.sio

Downloads phenotypic data + CC200 ROI time series from ABIDE-I public S3,
computes Laplacian eigenvectors, compresses to 8×8 feature vectors, and
writes the TSV manifest expected by brain_ossm_abide.sio.

Output: abide_roi_manifest.tsv with schema:
  subject_id<TAB>label<TAB>site<TAB>f0<TAB>f1<TAB>...<TAB>f63

Usage:
  python3 scripts/research/abide_prepare_manifest.py [--output-dir /orangefs/data/abide]
  python3 scripts/research/abide_prepare_manifest.py --output-dir ./artifacts/research/abide --max-subjects 50

For cluster deployment:
  Copy output manifest to /orangefs/data/abide/abide_roi_manifest.tsv
"""

import numpy as np
from scipy import linalg
import os, sys, struct
import argparse

PHENO_URL = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/"
    "ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv"
)

ROI_URL_TEMPLATE = (
    "https://s3.amazonaws.com/fcp-indi/data/Projects/"
    "ABIDE_Initiative/Outputs/cpac/filt_noglobal/"
    "rois_cc200/{file_id}_rois_cc200.1D"
)

N_EIGVECS = 7      # Laplacian eigenvectors (skip trivial e0)
N_ROIS_TARGET = 200 # CC200 atlas
N_FEATURES = 64     # 8 temporal steps × 8 dimensions
N_STEPS = 8
N_DIMS = 8


def download_file(url, path):
    """Download file if not already cached."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
        return True
    except Exception as e:
        print(f"  download failed: {e}", file=sys.stderr)
        return False


def extract_eigenvectors(ts_path):
    """Load time series → correlation → Laplacian → eigenvectors."""
    try:
        ts = np.loadtxt(ts_path)
        if ts.ndim != 2 or ts.shape[0] < 20 or ts.shape[1] < 8:
            return None
        n_rois = ts.shape[1]
        n_timepoints = ts.shape[0]

        # Correlation matrix → threshold → Laplacian
        corr = np.corrcoef(ts.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0)
        adj = np.maximum(corr, 0)  # keep positive correlations
        deg = adj.sum(axis=1)
        L = np.diag(deg) - adj

        # Eigenvectors (Fiedler vectors: skip λ₀)
        eigenvalues, eigenvectors = linalg.eigh(L)
        if len(eigenvalues) < N_EIGVECS + 1:
            return None
        evecs = eigenvectors[:, 1:N_EIGVECS+1].T  # (7, n_rois)

        return evecs, n_rois, n_timepoints
    except Exception:
        return None


def eigvecs_to_features(evecs, n_rois):
    """Compress 7×n_rois eigenvector frame to 64 features (8 steps × 8 dims).

    Strategy: partition the 200 ROIs into 8 blocks of 25, compute the mean
    eigenvector value per block. Then stack with a temporal dimension derived
    from the eigenvector index (7 eigvecs → pad to 8 steps).

    The resulting 8×8 matrix represents:
      - rows = temporal steps (eigenvector spectral scale)
      - cols = spatial blocks (anatomical regions)
    """
    # Pad eigenvectors to 8 (add a zero 8th vector)
    if evecs.shape[0] < N_STEPS:
        padded = np.zeros((N_STEPS, evecs.shape[1]))
        padded[:evecs.shape[0], :] = evecs
        evecs = padded

    # Pad/truncate ROIs to N_ROIS_TARGET
    if n_rois < N_ROIS_TARGET:
        padded = np.zeros((N_STEPS, N_ROIS_TARGET))
        padded[:, :n_rois] = evecs[:, :n_rois]
        evecs = padded
    elif n_rois > N_ROIS_TARGET:
        evecs = evecs[:, :N_ROIS_TARGET]

    # Block average: 200 ROIs → 8 blocks of 25
    block_size = N_ROIS_TARGET // N_DIMS  # 25
    features = np.zeros((N_STEPS, N_DIMS))
    for d in range(N_DIMS):
        start = d * block_size
        end = start + block_size
        features[:, d] = evecs[:, start:end].mean(axis=1)

    # Normalize each feature to [-1, 1] range
    max_abs = np.abs(features).max()
    if max_abs > 1e-12:
        features = features / max_abs

    return features.flatten()  # 64 values: step0_dim0, step0_dim1, ...


def main():
    parser = argparse.ArgumentParser(description="Prepare ABIDE manifest for brain_ossm_abide.sio")
    parser.add_argument("--output-dir", default="./artifacts/research/abide",
                        help="Output directory for manifest and cached files")
    parser.add_argument("--max-subjects", type=int, default=0,
                        help="Max subjects to process (0 = all)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Only use already-cached .1D files")
    args = parser.parse_args()

    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    # 1. Download phenotypic data
    pheno_path = os.path.join(cache_dir, "phenotypic.csv")
    print("Step 1: Phenotypic data")
    if not download_file(PHENO_URL, pheno_path):
        print("FATAL: Cannot download phenotypic data", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    pheno = pd.read_csv(pheno_path)
    asd_df = pheno[pheno['DX_GROUP'] == 1]
    td_df = pheno[pheno['DX_GROUP'] == 2]
    print(f"  {len(asd_df)} ASD, {len(td_df)} TD subjects in phenotypic")

    # 2. Process subjects
    print("\nStep 2: Processing subjects")
    rows = []  # (subject_id, label_str, site, features_64)

    def process_group(df, label_str):
        count = 0
        for _, row in df.iterrows():
            if args.max_subjects > 0 and len(rows) >= args.max_subjects:
                break
            fid = str(row.get('FILE_ID', ''))
            if not fid or fid == 'no_filename' or fid == 'nan':
                continue
            site = str(row.get('SITE_ID', 'unknown'))
            sid = str(row.get('SUB_ID', fid))

            ts_path = os.path.join(cache_dir, f"{fid}_rois_cc200.1D")

            if not args.skip_download:
                url = ROI_URL_TEMPLATE.format(file_id=fid)
                if not download_file(url, ts_path):
                    continue
            elif not os.path.exists(ts_path):
                continue

            result = extract_eigenvectors(ts_path)
            if result is None:
                continue
            evecs, n_rois, n_tp = result
            features = eigvecs_to_features(evecs, n_rois)

            rows.append((sid, label_str, site, features))
            count += 1
            if count % 50 == 0:
                print(f"  {label_str}: {count} processed")
        print(f"  {label_str}: {count} total")

    process_group(asd_df, "ASD")
    process_group(td_df, "TD")

    n_total = len(rows)
    n_asd = sum(1 for r in rows if r[1] == "ASD")
    n_td = sum(1 for r in rows if r[1] == "TD")
    print(f"\n  Total: {n_total} subjects ({n_asd} ASD, {n_td} TD)")

    if n_total == 0:
        print("FATAL: No subjects processed. Check network access to ABIDE S3.", file=sys.stderr)
        sys.exit(1)

    # 3. Write manifest TSV
    manifest_path = os.path.join(args.output_dir, "abide_roi_manifest.tsv")
    print(f"\nStep 3: Writing manifest to {manifest_path}")

    with open(manifest_path, 'w') as f:
        # Header
        cols = ["subject_id", "label", "site"] + [f"f{i}" for i in range(64)]
        f.write("\t".join(cols) + "\n")
        # Data rows
        for sid, label, site, features in rows:
            feat_strs = [f"{v:.8f}" for v in features]
            f.write("\t".join([sid, label, site] + feat_strs) + "\n")

    file_size = os.path.getsize(manifest_path)
    print(f"  Manifest: {manifest_path} ({file_size} bytes, {n_total} rows)")

    # 4. Verification
    print("\nStep 4: Verification")
    with open(manifest_path) as f:
        header = f.readline().strip()
        first_data = f.readline().strip()
    n_header_fields = len(header.split('\t'))
    n_data_fields = len(first_data.split('\t'))
    print(f"  Header fields: {n_header_fields} (expected 67)")
    print(f"  Data fields:   {n_data_fields} (expected 67)")
    assert n_header_fields == 67, f"Header has {n_header_fields} fields, expected 67"
    assert n_data_fields == 67, f"Data has {n_data_fields} fields, expected 67"
    print("  Schema validation: PASS")

    # Site distribution
    sites = {}
    for _, _, site, _ in rows:
        sites[site] = sites.get(site, 0) + 1
    print(f"  Sites: {len(sites)} unique")
    for site, count in sorted(sites.items(), key=lambda x: -x[1])[:5]:
        print(f"    {site}: {count}")

    print(f"\n  To deploy: cp {manifest_path} /orangefs/data/abide/abide_roi_manifest.tsv")
    print("  Done.")


if __name__ == "__main__":
    main()
