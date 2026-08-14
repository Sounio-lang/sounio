#!/usr/bin/env python3
"""
O-SSM on CHB-MIT Seizure EEG: Octonion vs Sedenion on neural crossing.

The seizure onset is the ultimate "crossing" event in EEG:
- Normal and pathological dynamics COEXIST at the transition zone
- The ictal onset is not A→B (switching), it's A AND B crossing
- This is the neural analog of a pseudoknot

HYPOTHESIS: Sedenion associator spikes at seizure onset where octonion does not.
The non-alternativity captures the crossing between normal and seizure dynamics.

SEGMENTS:
- Pre-ictal: 60s before seizure onset (normal)
- Ictal: during seizure (pathological)
- Post-ictal: 60s after seizure end (recovery)
- Inter-ictal: from non-seizure files (baseline normal)
"""

import numpy as np
import sys, os, struct
sys.path.insert(0, os.path.dirname(__file__))

from nback_sedenion_experiment import (
    oct_mul, oct_assoc_norm, sed_mul, sed_assoc_norm,
    ossm_forward, compute_features
)

# ============================================================
# EDF READER (minimal, for CHB-MIT)
# ============================================================

def load_edf(filepath):
    """Load an EDF file. Returns (data, channels, sample_rate)."""
    with open(filepath, 'rb') as f:
        # EDF header
        version = f.read(8).decode('ascii', errors='replace').strip()
        patient_id = f.read(80).decode('ascii', errors='replace').strip()
        record_id = f.read(80).decode('ascii', errors='replace').strip()
        start_date = f.read(8).decode('ascii', errors='replace').strip()
        start_time = f.read(8).decode('ascii', errors='replace').strip()
        header_bytes = int(f.read(8).decode('ascii').strip())
        reserved = f.read(44).decode('ascii', errors='replace').strip()
        num_records = int(f.read(8).decode('ascii').strip())
        duration = float(f.read(8).decode('ascii').strip())
        n_channels = int(f.read(4).decode('ascii').strip())
        
        # Channel info
        ch_labels = []
        transducers = []
        physical_dims = []
        physical_mins = []
        physical_maxs = []
        digital_mins = []
        digital_maxs = []
        prefilterings = []
        num_samples_per_record = []
        
        for _ in range(n_channels):
            ch_labels.append(f.read(16).decode('ascii').strip())
        for _ in range(n_channels):
            transducers.append(f.read(80).decode('ascii', errors='replace').strip())
        for _ in range(n_channels):
            physical_dims.append(f.read(8).decode('ascii', errors='replace').strip())
        for _ in range(n_channels):
            physical_mins.append(float(f.read(8).decode('ascii').strip()))
        for _ in range(n_channels):
            physical_maxs.append(float(f.read(8).decode('ascii').strip()))
        for _ in range(n_channels):
            digital_mins.append(float(f.read(8).decode('ascii').strip()))
        for _ in range(n_channels):
            digital_maxs.append(float(f.read(8).decode('ascii').strip()))
        for _ in range(n_channels):
            prefilterings.append(f.read(80).decode('ascii', errors='replace').strip())
        for _ in range(n_channels):
            num_samples_per_record.append(int(f.read(8).decode('ascii').strip()))
        
        f.seek(header_bytes)
        
        # Read data records
        total_samples = num_samples_per_record[0] * num_records
        data = np.zeros((total_samples, n_channels), dtype=np.float64)
        
        for rec in range(num_records):
            for ch in range(n_channels):
                n = num_samples_per_record[ch]
                raw_bytes = f.read(n * 2)
                if len(raw_bytes) < n * 2:
                    break
                raw = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float64)
                # Scale to physical
                scale = (physical_maxs[ch] - physical_mins[ch]) / max(digital_maxs[ch] - digital_mins[ch], 1)
                offset = physical_mins[ch] - digital_mins[ch] * scale
                start = rec * n
                end = min(start + n, total_samples)
                data[start:end, ch] = raw[:end-start] * scale + offset
        
        sample_rate = int(num_samples_per_record[0] / duration)
        return data, ch_labels, sample_rate


# ============================================================
# SEIZURE EXPERIMENT
# ============================================================

# Seizure annotations for downloaded files
SEIZURES = {
    'chb01_03.edf': [(2996, 3036)],  # start, end in seconds
    'chb01_04.edf': [(1467, 1494)],
    'chb01_15.edf': [(1732, 1772)],
}
# Non-seizure files
NON_SEIZURE = ['chb01_01.edf', 'chb01_02.edf', 'chb01_06.edf']


def segment_seizure(data, sr, seizure_start, seizure_end,
                    pre_dur=60, post_dur=60, epoch_len=10):
    """Extract pre-ictal, ictal, and post-ictal segments."""
    segments = {'pre': [], 'ictal': [], 'post': []}
    
    # Pre-ictal: 60s before onset
    pre_start = int((seizure_start - pre_dur) * sr)
    pre_end = int(seizure_start * sr)
    if pre_start >= 0:
        for i in range(pre_start, pre_end - int(epoch_len * sr), int(epoch_len * sr)):
            segments['pre'].append(data[i:i + int(epoch_len * sr)])
    
    # Ictal: during seizure
    ict_start = int(seizure_start * sr)
    ict_end = int(seizure_end * sr)
    for i in range(ict_start, ict_end - int(epoch_len * sr), int(epoch_len * sr)):
        segments['ictal'].append(data[i:i + int(epoch_len * sr)])
    
    # Post-ictal: 60s after end
    post_start = int(seizure_end * sr)
    post_end = int((seizure_end + post_dur) * sr)
    if post_end <= data.shape[0]:
        for i in range(post_start, post_end - int(epoch_len * sr), int(epoch_len * sr)):
            segments['post'].append(data[i:i + int(epoch_len * sr)])
    
    return segments


def zscore_epoch(epoch):
    """Z-score each channel."""
    for ch in range(epoch.shape[1]):
        mu = np.mean(epoch[:, ch])
        sigma = np.std(epoch[:, ch]) + 1e-8
        epoch[:, ch] = (epoch[:, ch] - mu) / sigma
    return epoch


def run_seizure_experiment(data_dir='/workspace/sounio/datasets/eeg_seizure'):
    print("\n" + "=" * 72)
    print("O-SSM OCTONION vs SEDENION ON SEIZURE EEG (CHB-MIT)")
    print("=" * 72)
    
    all_features = {'inter': [], 'pre': [], 'ictal': [], 'post': []}
    
    # Process non-seizure files (inter-ictal baseline)
    for fname in NON_SEIZURE:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        print(f"\n  Loading {fname}...", end=" ")
        data, chs, sr = load_edf(path)
        print(f"{data.shape}, {sr}Hz, {len(chs)}ch")
        
        # Use first 7 channels, take random 10s epochs
        n_epochs = min(10, data.shape[0] // (10 * sr))
        rng = np.random.default_rng(42)
        for i in range(n_epochs):
            start = rng.integers(0, data.shape[0] - 10 * sr)
            epoch = data[start:start + 10 * sr, :7].copy()
            epoch = zscore_epoch(epoch)
            
            traj_oct = ossm_forward(epoch, dim=8, use_sedenion=False)
            f1_med, f1_mean, _ = compute_features(traj_oct, use_sedenion=False)
            
            traj_sed = ossm_forward(epoch, dim=16, use_sedenion=True)
            f3_med, f3_mean, _ = compute_features(traj_sed, use_sedenion=True)
            
            all_features['inter'].append({
                'F1': f1_mean, 'F3': f3_mean, 'F3_over_F1': f3_mean / max(f1_mean, 1e-8)
            })
    
    # Process seizure files
    for fname, seizures in SEIZURES.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        print(f"\n  Loading {fname}...", end=" ")
        data, chs, sr = load_edf(path)
        print(f"{data.shape}, {sr}Hz")
        
        for sz_start, sz_end in seizures:
            segments = segment_seizure(data, sr, sz_start, sz_end)
            
            for phase, epochs in segments.items():
                for epoch in epochs[:5]:  # max 5 epochs per phase
                    epoch = epoch[:, :7].copy()
                    epoch = zscore_epoch(epoch)
                    
                    traj_oct = ossm_forward(epoch, dim=8, use_sedenion=False)
                    f1_med, f1_mean, _ = compute_features(traj_oct, use_sedenion=False)
                    
                    traj_sed = ossm_forward(epoch, dim=16, use_sedenion=True)
                    f3_med, f3_mean, _ = compute_features(traj_sed, use_sedenion=True)
                    
                    all_features[phase].append({
                        'F1': f1_mean, 'F3': f3_mean, 'F3_over_F1': f3_mean / max(f1_mean, 1e-8)
                    })
    
    # Summary
    print(f"\n{'='*72}")
    print("RESULTS — Octonion vs Sedenion by Seizure Phase")
    print(f"{'='*72}")
    print(f"{'Phase':<12} {'F1(oct)':<18} {'F3(sed)':<18} {'F3/F1':<10} {'n':<6}")
    print("-" * 64)
    
    for phase in ['inter', 'pre', 'ictal', 'post']:
        feats = all_features[phase]
        if not feats:
            print(f"{phase:<12} (no data)")
            continue
        f1 = np.mean([f['F1'] for f in feats])
        f1_se = np.std([f['F1'] for f in feats]) / np.sqrt(len(feats))
        f3 = np.mean([f['F3'] for f in feats])
        f3_se = np.std([f['F3'] for f in feats]) / np.sqrt(len(feats))
        ratio = np.mean([f['F3_over_F1'] for f in feats])
        print(f"{phase:<12} {f1:.4f}±{f1_se:.4f}   {f3:.4f}±{f3_se:.4f}   {ratio:.1f}   {len(feats)}")
    
    # Discrimination
    print(f"\nDiscrimination (inter-ictal vs ictal):")
    f1_inter = [f['F1'] for f in all_features.get('inter', [])]
    f1_ictal = [f['F1'] for f in all_features.get('ictal', [])]
    f3_inter = [f['F3'] for f in all_features.get('inter', [])]
    f3_ictal = [f['F3'] for f in all_features.get('ictal', [])]
    
    if len(f1_inter) > 2 and len(f1_ictal) > 2:
        def cohens_d(a, b):
            pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            return (np.mean(a) - np.mean(b)) / max(pooled, 1e-8)
        
        d_oct = cohens_d(f1_inter, f1_ictal)
        d_sed = cohens_d(f3_inter, f3_ictal)
        print(f"  Octonion (F1): d = {d_oct:+.3f}")
        print(f"  Sedenion (F3): d = {d_sed:+.3f}")
        if abs(d_sed) > abs(d_oct):
            print(f"  ⚡ SEDENION discriminates seizure vs normal BETTER")
        else:
            print(f"  Octonion discriminates better (or equal)")
    
    return all_features


if __name__ == '__main__':
    results = run_seizure_experiment()
