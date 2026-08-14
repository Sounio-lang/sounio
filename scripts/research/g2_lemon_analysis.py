#!/usr/bin/env python3
"""
G₂ 14-generator features on LEMON EEG: test if specific G₂ directions
predict neuroticism/rumination better than the aggregate associator (F1).

APPROACH
========
1. Load cached LEMON preprocessed tensors (204 subjects on /orangefs)
2. Run O-SSM forward pass (untrained, same as original)
3. Extract 14 G₂ features per subject (instead of just F1/F2/F3)
4. Test correlation with neuroticism, rumination, anxiety, depression
5. Compare: which G₂ generators carry signal?

If a SPECIFIC generator (e.g., G3) correlates more strongly than the
aggregate F1, that direction of non-associative structure is the one
that captures rumination — a sharper, more interpretable finding.

We run on the control node (64 cores) with cached preprocessed data.
"""

import numpy as np
import sys, os, csv, json, time
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

# Import G₂ generators
from g2_features import build_g2_generators, compute_g2_features

# Import O-SSM
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'research' / 'ossm_168_dryrun'))
from ossm_168_dryrun import ossm as ossm_mod
from ossm_168_dryrun import _features_fast as feat_fast


def extract_g2_from_epoch(epoch, generators):
    """Extract G₂ features from a single EEG epoch.

    epoch: (7, 1000) — 7 channels, 1000 timesteps
    generators: (14, 7, 7) — G₂ generators

    Returns: (14,) — median ‖D_k(h_t)‖ over trajectory
    """
    # Forward pass through O-SSM (untrained, fixed weights)
    traj = ossm_mod.forward_pass(np.ascontiguousarray(epoch), subject_index=0)
    # traj shape: (T, 2, 8) — T timesteps, 2 states, 8-dim octonion

    # Extract imaginary parts (7-dim) from both states
    T = traj.shape[0]
    traj1 = traj[:, 0, 1:]  # (T, 7) — state 1 imaginary
    traj2 = traj[:, 1, 1:]  # (T, 7) — state 2 imaginary

    # Compute 14 G₂ features for each state, then average
    f1 = compute_g2_features(traj1, generators)
    f2 = compute_g2_features(traj2, generators)
    return (f1 + f2) / 2  # (14,)


def run_lemon_g2(cache_dir='/workspace/data/lemon/preprocessed',
                 endpoints_path='/workspace/data/lemon/endpoints.csv',
                 subjects_path='/workspace/data/lemon/subjects_all.txt'):
    """Run G₂ feature extraction on LEMON EEG."""

    print("=" * 72)
    print("G₂ 14-GENERATOR FEATURES ON LEMON EEG")
    print("=" * 72)

    # Build G₂ generators
    print("\nBuilding G₂ generators...", end=" ")
    generators = build_g2_generators()
    print(f"done ({generators.shape[0]} generators)")

    # Load endpoints
    endpoints = {}
    with open(endpoints_path) as f:
        for row in csv.DictReader(f):
            endpoints[row['subject_id']] = row
    print(f"Loaded {len(endpoints)} endpoints")

    # Load subject list
    with open(subjects_path) as f:
        subjects = [s.strip() for s in f if s.strip()]
    print(f"Subject list: {len(subjects)}")

    # Process each subject
    all_features = {}  # subject_id → (14,) G₂ features
    processed = 0
    failed = 0
    t0 = time.time()

    for sid in subjects:
        # Load cached epoch tensor
        epoch_path = os.path.join(cache_dir, f'{sid}_epochs.npy')
        if not os.path.exists(epoch_path):
            failed += 1
            continue

        try:
            epochs = np.load(epoch_path)
            # epochs shape: (n_epochs, 7, 1000)
            # Process each epoch, then average G₂ features across epochs
            epoch_features = []
            max_epochs = min(10, epochs.shape[0])  # limit for speed

            for i in range(max_epochs):
                feat = extract_g2_from_epoch(epochs[i], generators)
                epoch_features.append(feat)

            # Median across epochs
            all_features[sid] = np.median(epoch_features, axis=0)
            processed += 1

            if processed % 10 == 0:
                elapsed = time.time() - t0
                rate = processed / elapsed
                eta = (len(subjects) - processed) / rate
                print(f"  [{processed}/{len(subjects)}] {sid}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

        except Exception as e:
            print(f"  ⚠ {sid}: {e}")
            failed += 1

    print(f"\nProcessed: {processed}, Failed: {failed}")
    print(f"Time: {time.time() - t0:.0f}s")

    # Merge with endpoints and run correlations
    print(f"\n{'='*72}")
    print("CORRELATION ANALYSIS")
    print(f"{'='*72}")

    def safe_float(v):
        try: return float(v) if v and v.strip() else float('nan')
        except: return float('nan')

    # Build merged dataset
    merged = []
    for sid, feat in all_features.items():
        if sid in endpoints:
            ep = endpoints[sid]
            merged.append({
                'sid': sid,
                'G2': feat,  # (14,)
                'rumination': safe_float(ep.get('cerq_rumination')),
                'neuroticism': safe_float(ep.get('neo_neuroticism')),
                'anxiety': safe_float(ep.get('stai_trait')),
                'hamilton': safe_float(ep.get('hamilton')),
                'bas_drive': safe_float(ep.get('bas_drive')),
            })

    print(f"Merged: {len(merged)} subjects with both G₂ features and endpoints\n")

    if len(merged) < 20:
        print("Too few subjects for reliable analysis")
        return

    # Test each G₂ generator against each endpoint
    endpoints_to_test = [
        ('neuroticism', 'NEO_Neuroticism'),
        ('rumination', 'CERQ_Rumination'),
        ('anxiety', 'STAI_trait'),
        ('hamilton', 'Hamilton'),
        ('bas_drive', 'BAS_Drive'),
    ]

    from scipy.stats import spearmanr

    for ep_key, ep_name in endpoints_to_test:
        values = [m[ep_key] for m in merged]
        valid = [(m, v) for m, v in zip(merged, values) if not np.isnan(v)]

        if len(valid) < 10:
            print(f"  {ep_name}: n={len(valid)} (too few)")
            continue

        print(f"\n  {ep_name} (n={len(valid)}):")
        print(f"    {'Generator':<12} {'rho':<10} {'p':<10} {'Sig':<5}")

        for k in range(14):
            g_vals = [m['G2'][k] for m, _ in valid]
            ep_vals = [v for _, v in valid]

            rho, pval = spearmanr(g_vals, ep_vals)
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""

            if pval < 0.1:  # Only print interesting ones
                print(f"    G{k:<11} {rho:+.3f}    {pval:.4f}   {sig}")

        # Also test the MEAN of all 14 (analogous to original F1)
        g_mean = [np.mean(m['G2']) for m, _ in valid]
        rho_mean, p_mean = spearmanr(g_mean, [v for _, v in valid])
        sig_mean = "***" if p_mean < 0.001 else "**" if p_mean < 0.01 else "*" if p_mean < 0.05 else ""
        print(f"    {'Mean(G)':<12} {rho_mean:+.3f}    {p_mean:.4f}   {sig_mean}")

    # Also compare with original F1 if available
    print(f"\n  Comparison with original features:")
    print(f"    (G₂ mean ≈ aggregate non-associative magnitude, analogous to F1)")

    # Save results
    results = {}
    for m in merged:
        results[m['sid']] = {
            'G2_features': m['G2'].tolist(),
            'rumination': m['rumination'],
            'neuroticism': m['neuroticism'],
            'anxiety': m['anxiety'],
            'hamilton': m['hamilton'],
        }

    outpath = 'scripts/research/g2_lemon_results.json'
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")

    return results


if __name__ == '__main__':
    run_lemon_g2()
