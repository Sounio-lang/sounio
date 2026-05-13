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
  python3 scripts/research/abide_prepare_manifest.py \\
      --output-dir ./artifacts/research/brain_ossm_local_pilot \\
      --phenotypic-csv /tmp/abide_pilot/phenotypic.csv \\
      --roi-dir /tmp/abide_pilot --skip-download --balanced-subjects 100
  python3 scripts/research/abide_prepare_manifest.py \\
      --output-dir ./artifacts/research/brain_ossm_local_site_balanced \\
      --phenotypic-csv /tmp/abide_pilot/phenotypic.csv \\
      --roi-dir /tmp/abide_pilot --skip-download \\
      --sampling-mode site_balanced --balanced-subjects 100

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


def resolve_phenotypic_path(args, cache_dir):
    """Choose a phenotypic CSV source, downloading only when necessary."""
    if args.phenotypic_csv:
        if not os.path.exists(args.phenotypic_csv):
            print(f"FATAL: phenotypic CSV not found: {args.phenotypic_csv}", file=sys.stderr)
            sys.exit(1)
        return args.phenotypic_csv

    pheno_path = os.path.join(cache_dir, "phenotypic.csv")
    if os.path.exists(pheno_path) and os.path.getsize(pheno_path) > 0:
        return pheno_path

    if args.skip_download:
        print(f"FATAL: skip-download set and no cached phenotypic CSV at {pheno_path}", file=sys.stderr)
        sys.exit(1)

    if not download_file(PHENO_URL, pheno_path):
        print("FATAL: Cannot download phenotypic data", file=sys.stderr)
        sys.exit(1)

    return pheno_path


def effective_per_class_limit(args):
    """Return a per-class limit when a balanced cohort was requested."""
    if args.balanced_subjects <= 0:
        return args.max_per_class
    if args.balanced_subjects % 2 != 0:
        print("FATAL: --balanced-subjects must be even", file=sys.stderr)
        sys.exit(1)
    return args.balanced_subjects // 2


def load_candidate_phenotypic(pheno_path, roi_dir, require_cached):
    """Load phenotypic rows with ROI paths, optionally requiring local cache hits."""
    import pandas as pd

    pheno = pd.read_csv(pheno_path)
    pheno = pheno[pheno["DX_GROUP"].isin([1, 2])].copy()
    pheno["FILE_ID"] = pheno["FILE_ID"].astype(str)
    pheno = pheno[~pheno["FILE_ID"].isin(["", "nan", "no_filename"])]
    pheno["ROI_PATH"] = pheno["FILE_ID"].map(
        lambda fid: os.path.join(roi_dir, f"{fid}_rois_cc200.1D")
    )
    pheno["HAS_ROI"] = pheno["ROI_PATH"].map(os.path.exists)
    if require_cached:
        pheno = pheno[pheno["HAS_ROI"]]
    return pheno.copy()


def select_rows_ordered(pheno_df, max_subjects, per_class_limit):
    """Original export rule: cached ASD rows first, then cached TD rows."""
    rows = []
    for dx_group, label_str in ((1, "ASD"), (2, "TD")):
        count = 0
        subset = pheno_df[pheno_df["DX_GROUP"] == dx_group]
        for _, row in subset.iterrows():
            if max_subjects > 0 and len(rows) >= max_subjects:
                break
            if per_class_limit > 0 and count >= per_class_limit:
                break
            rows.append((str(row["SUB_ID"]), label_str, str(row["SITE_ID"]), str(row["FILE_ID"]), row["ROI_PATH"]))
            count += 1
    return rows


def select_rows_site_balanced(pheno_df, target_subjects):
    """
    Diagnosis-balanced, near-uniform per-site export.

    Allocate one ASD/TD pair per eligible site in round-robin order until the
    requested target is reached. Within each site/diagnosis bucket, preserve the
    original phenotypic row order.
    """
    if target_subjects <= 0 or target_subjects % 2 != 0:
        raise ValueError("site_balanced sampling requires an even positive subject target")

    per_site = {}
    for site, site_df in pheno_df.groupby("SITE_ID", sort=True):
        asd_rows = list(site_df[site_df["DX_GROUP"] == 1].iterrows())
        td_rows = list(site_df[site_df["DX_GROUP"] == 2].iterrows())
        pairs = min(len(asd_rows), len(td_rows))
        if pairs <= 0:
            continue
        per_site[site] = {"asd": asd_rows, "td": td_rows, "pairs": pairs}

    if not per_site:
        raise RuntimeError("No eligible sites with both ASD and TD cached subjects")

    target_pairs = target_subjects // 2
    capacity_pairs = sum(bucket["pairs"] for bucket in per_site.values())
    if capacity_pairs < target_pairs:
        raise RuntimeError(
            f"Insufficient site-balanced capacity: requested {target_pairs} pairs, have {capacity_pairs}"
        )

    ordered_sites = sorted(per_site.keys(), key=lambda site: (-per_site[site]["pairs"], site))
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
            raise RuntimeError("Site-balanced allocation stalled before reaching target")

    rows = []
    for site in ordered_sites:
        n_pairs = allocated[site]
        if n_pairs <= 0:
            continue
        for _, row in per_site[site]["asd"][:n_pairs]:
            rows.append((str(row["SUB_ID"]), "ASD", str(row["SITE_ID"]), str(row["FILE_ID"]), row["ROI_PATH"]))
        for _, row in per_site[site]["td"][:n_pairs]:
            rows.append((str(row["SUB_ID"]), "TD", str(row["SITE_ID"]), str(row["FILE_ID"]), row["ROI_PATH"]))
    return rows


def extract_eigenvectors(ts_path):
    """Load time series → correlation → Laplacian → eigenvectors."""
    try:
        ts = np.loadtxt(ts_path)
        if ts.ndim != 2 or ts.shape[0] < 20 or ts.shape[1] < 8:
            return None
        n_rois = ts.shape[1]
        n_timepoints = ts.shape[0]

        # Correlation matrix → threshold → Laplacian
        with np.errstate(divide="ignore", invalid="ignore"):
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
    parser.add_argument("--max-per-class", type=int, default=0,
                        help="Max subjects per diagnosis class (0 = all)")
    parser.add_argument("--balanced-subjects", type=int, default=0,
                        help="Balanced total subject count, split evenly across ASD/TD")
    parser.add_argument("--phenotypic-csv", default="",
                        help="Use an existing phenotypic CSV instead of downloading")
    parser.add_argument("--roi-dir", default="",
                        help="Directory containing cached *_rois_cc200.1D files")
    parser.add_argument("--sampling-mode", choices=["ordered", "site_balanced"], default="ordered",
                        help="ordered = original ASD-then-TD export, site_balanced = round-robin paired sites")
    parser.add_argument("--skip-download", action="store_true",
                        help="Only use already-cached .1D files")
    args = parser.parse_args()

    cache_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    roi_dir = args.roi_dir if args.roi_dir else cache_dir
    per_class_limit = effective_per_class_limit(args)

    # 1. Download phenotypic data
    pheno_path = resolve_phenotypic_path(args, cache_dir)
    print("Step 1: Phenotypic data")
    print(f"  Using: {pheno_path}")

    import pandas as pd
    pheno = pd.read_csv(pheno_path)
    asd_df = pheno[pheno['DX_GROUP'] == 1]
    td_df = pheno[pheno['DX_GROUP'] == 2]
    print(f"  {len(asd_df)} ASD, {len(td_df)} TD subjects in phenotypic")

    # 2. Process subjects
    print("\nStep 2: Processing subjects")
    require_cached = args.skip_download or args.sampling_mode == "site_balanced"
    cached_pheno = load_candidate_phenotypic(pheno_path, roi_dir, require_cached=require_cached)
    if args.sampling_mode == "site_balanced":
        target_subjects = args.balanced_subjects if args.balanced_subjects > 0 else args.max_subjects
        selected = select_rows_site_balanced(cached_pheno, target_subjects)
        print(f"  site_balanced selection: {len(selected)} subjects")
    else:
        selected = select_rows_ordered(cached_pheno, args.max_subjects, per_class_limit)
        print(f"  ordered selection: {len(selected)} subjects")

    rows = []  # (subject_id, label_str, site, features_64)
    for idx, (sid, label_str, site, fid, ts_path) in enumerate(selected, start=1):
        if not os.path.exists(ts_path):
            if args.skip_download:
                continue
            url = ROI_URL_TEMPLATE.format(file_id=fid)
            if not download_file(url, ts_path):
                continue
        result = extract_eigenvectors(ts_path)
        if result is None:
            continue
        evecs, n_rois, _ = result
        features = eigvecs_to_features(evecs, n_rois)
        rows.append((sid, label_str, site, features))
        if idx % 50 == 0:
            print(f"  processed {idx}/{len(selected)}")

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
        f.write("# schema=brain_ossm.abide.v2\n")
        f.write("# seq_len=8\n")
        f.write("# input_dim=8\n")
        f.write("# feature_layout=flat\n")
        f.write("# label_space=asd_vs_control\n")
        f.write("# split_policy=leave_one_site_out\n")
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
        lines = [line.strip() for line in f if line.strip()]
    data_lines = [line for line in lines if not line.startswith("#")]
    header = data_lines[0] if data_lines else ""
    first_data = data_lines[1] if len(data_lines) > 1 else ""
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
