#!/usr/bin/env python3
"""
ABIDE-I preprocessor: download all available subjects, compute Laplacian
eigenvectors, export as flat binary for Sounio consumption.

Output format (artifacts/research/abide/frames.bin):
  Header: [n_asd: i64, n_td: i64]
  Per subject: [7 * 200 f64 values] = eigenvector frame (7 vectors × 200 components)
  Order: all ASD subjects first, then all TD subjects.

Also writes frames_index.txt with file IDs for traceability.
"""

import numpy as np
from scipy import linalg
import os, sys, struct
import pandas as pd

CACHE = "artifacts/research/abide"

def download_subject(file_id):
    """Download ROI time series .1D file if not cached."""
    if not file_id or file_id == 'no_filename':
        return None
    path = os.path.join(CACHE, f"{file_id}_rois_cc200.1D")
    if os.path.exists(path):
        return path
    url = (
        f"https://s3.amazonaws.com/fcp-indi/data/Projects/"
        f"ABIDE_Initiative/Outputs/cpac/filt_noglobal/"
        f"rois_cc200/{file_id}_rois_cc200.1D"
    )
    try:
        import urllib.request
        urllib.request.urlretrieve(url, path)
        return path
    except:
        return None


def extract_frame(path):
    """Load time series → correlation → Laplacian → 7 eigenvectors."""
    try:
        ts = np.loadtxt(path)
        if ts.ndim != 2 or ts.shape[0] < 20 or ts.shape[1] < 8:
            return None
        n_rois = ts.shape[1]
        corr = np.corrcoef(ts.T)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 0)
        adj = np.maximum(corr, 0)
        deg = adj.sum(axis=1)
        L = np.diag(deg) - adj
        eigenvalues, eigenvectors = linalg.eigh(L)
        if len(eigenvalues) < 8:
            return None
        frame = eigenvectors[:, 1:8].T  # (7, n_rois)
        return frame, n_rois
    except:
        return None


def main():
    os.makedirs(CACHE, exist_ok=True)

    # Load phenotypic
    pheno_path = os.path.join(CACHE, "phenotypic.csv")
    if not os.path.exists(pheno_path):
        print("Downloading phenotypic data...")
        import urllib.request
        url = ("https://s3.amazonaws.com/fcp-indi/data/Projects/"
               "ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv")
        urllib.request.urlretrieve(url, pheno_path)

    pheno = pd.read_csv(pheno_path)
    asd_df = pheno[pheno['DX_GROUP'] == 1]
    td_df = pheno[pheno['DX_GROUP'] == 2]

    print(f"Phenotypic: {len(asd_df)} ASD, {len(td_df)} TD")

    # Download and extract all
    asd_frames = []
    asd_ids = []
    td_frames = []
    td_ids = []

    def process_group(df, label, frames_list, ids_list):
        count = 0
        total = len(df)
        for idx, row in df.iterrows():
            fid = row['FILE_ID']
            if not fid or fid == 'no_filename':
                continue
            # Use cached files only — skip downloads
            path = os.path.join(CACHE, f"{fid}_rois_cc200.1D")
            if not os.path.exists(path):
                continue
            result = extract_frame(path)
            if result is None:
                continue
            frame, n_rois = result
            # Pad/truncate to exactly 200 components
            if n_rois < 200:
                padded = np.zeros((7, 200))
                padded[:, :n_rois] = frame
                frame = padded
            elif n_rois > 200:
                frame = frame[:, :200]
            frames_list.append(frame)
            ids_list.append(fid)
            count += 1
            if count % 50 == 0:
                print(f"  {label}: {count} extracted")
        print(f"  {label}: {count} total")

    print("\nProcessing ASD subjects...")
    process_group(asd_df, "ASD", asd_frames, asd_ids)

    print("\nProcessing TD subjects...")
    process_group(td_df, "TD", td_frames, td_ids)

    n_asd = len(asd_frames)
    n_td = len(td_frames)
    print(f"\nTotal: {n_asd} ASD, {n_td} TD")

    # Write binary
    bin_path = os.path.join(CACHE, "frames.bin")
    with open(bin_path, 'wb') as f:
        f.write(struct.pack('<qq', n_asd, n_td))
        for frame in asd_frames:
            f.write(frame.astype(np.float64).tobytes())
        for frame in td_frames:
            f.write(frame.astype(np.float64).tobytes())

    print(f"Binary: {bin_path} ({os.path.getsize(bin_path)} bytes)")

    # Write index
    idx_path = os.path.join(CACHE, "frames_index.txt")
    with open(idx_path, 'w') as f:
        for fid in asd_ids:
            f.write(f"ASD\t{fid}\n")
        for fid in td_ids:
            f.write(f"TD\t{fid}\n")

    print(f"Index: {idx_path}")
    print("Done. Sounio can now read frames.bin.")


if __name__ == "__main__":
    main()
