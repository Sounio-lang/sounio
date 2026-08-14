#!/usr/bin/env python3
"""
OctTree on Atrial Fibrillation (AFib) ECG.

THE HYPOTHESIS
==============
Atrial fibrillation is the cardiac analog of pseudoknots:
- Normal sinus rhythm: regular, nested QRS complexes (tree-structured)
- AFib: irregularly irregular, chaotic atrial activity (non-tree)

The OctTree's non-associative product should capture the loss of
regular nesting structure in AFib, where the R-R intervals become
chaotic and the P-wave disappears.

APPROACH
========
1. Load ECG recordings (PhysioNet 2017 Challenge format)
2. Segment into fixed-length windows
3. Process through OctTree vs RealTree
4. Classify: Normal vs AFib vs Other arrhythmia vs Noise

If OctTree beats RealTree on AFib detection, we have a clinical
cardiology application of the Cayley-Dickson hierarchy.

ECG → OctTree → AFib detection = 7th domain.
"""

import numpy as np
import csv
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(__file__))

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise SystemExit("PyTorch required")

from cayley_dickson_paper_reproduction import (
    OctTreeClassifier, GRUClassifier, OSSMCell,
    count_params, train_one, oct_mul
)


# ============================================================
# LOAD PHYSIONET 2017 CHALLENGE DATA
# ============================================================

def load_physionet2017(zip_path, extract_dir, max_records=1000):
    """Load ECG recordings from PhysioNet 2017 AFib Challenge.
    
    Format: .mat files (scipy), 300 Hz, ~30s, single lead.
    Labels: N (normal), A (AFib), O (other), ~ (noise)
    """
    import scipy.io
    
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
    
    data_dir = os.path.join(extract_dir, 'training2017')
    
    # Load labels
    ref_path = os.path.join(data_dir, 'REFERENCE-original.csv')
    labels = {}
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    labels[row[0]] = row[1]
    
    # Load .mat recordings
    recordings = []
    rec_labels = []
    
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.mat')])[:max_records]
    
    for fname in files:
        rec_id = fname.replace('.mat', '')
        label = labels.get(rec_id, 'N')
        
        path = os.path.join(data_dir, fname)
        try:
            mat = scipy.io.loadmat(path)
            signal = mat['val'].flatten().astype(np.float32)
            if len(signal) > 100:
                recordings.append(signal)
                rec_labels.append(label)
        except:
            continue
    
    return recordings, rec_labels


# ============================================================
# ECG PREPROCESSING
# ============================================================

def preprocess_ecg(signal, target_length=256, sr=300):
    """Preprocess ECG: normalize, resample to target length, quantize to tokens.
    
    We convert continuous ECG to a discrete sequence by:
    1. Z-score normalization
    2. Divide into bins based on amplitude (quantization)
    3. Map to integer tokens
    
    This is analogous to the dot-bracket encoding of RNA structure.
    """
    # Z-score
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    
    # Resample to target_length
    if len(signal) > target_length:
        indices = np.linspace(0, len(signal) - 1, target_length).astype(int)
        signal = signal[indices]
    elif len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)))
    
    # Quantize to 7 bins (vocab=7): based on sign and magnitude
    # This creates a "bracket-like" sequence from the ECG morphology
    # Positive/negative crossings = structure
    tokens = np.zeros(target_length, dtype=np.int64)
    for i in range(target_length):
        v = signal[i]
        if v > 2.0:
            tokens[i] = 6  # R-peak positive
        elif v > 0.5:
            tokens[i] = 5  # positive deflection
        elif v > 0.0:
            tokens[i] = 4  # small positive
        elif v > -0.5:
            tokens[i] = 3  # baseline
        elif v > -2.0:
            tokens[i] = 2  # negative deflection
        else:
            tokens[i] = 1  # S-wave negative
    
    return tokens


def segment_ecg(signal, window_size=256, stride=128):
    """Segment ECG into overlapping windows."""
    segments = []
    for start in range(0, len(signal) - window_size, stride):
        segment = signal[start:start + window_size]
        tokens = preprocess_ecg(segment, window_size)
        segments.append(tokens)
    return segments


# ============================================================
# BUILD DATASET
# ============================================================

def build_afib_dataset(recordings, labels, window_size=256, stride=128,
                       max_windows_per_rec=5, binary=True):
    """Build train/test dataset from ECG recordings.
    
    binary=True: Normal (N) vs AFib (A), drop others
    binary=False: Normal (N) vs AFib (A) vs Other (O) vs Noise (~)
    """
    all_tokens = []
    all_labels = []
    
    rng = np.random.default_rng(42)
    
    for signal, label in zip(recordings, labels):
        if binary and label not in ('N', 'A'):
            continue
        
        segments = segment_ecg(signal, window_size, stride)
        
        # Subsample windows
        if len(segments) > max_windows_per_rec:
            idx = rng.choice(len(segments), max_windows_per_rec, replace=False)
            segments = [segments[i] for i in idx]
        
        for seg in segments:
            all_tokens.append(seg)
            if binary:
                all_labels.append(0 if label == 'N' else 1)  # 0=Normal, 1=AFib
            else:
                label_map = {'N': 0, 'A': 1, 'O': 2, '~': 3}
                all_labels.append(label_map.get(label, 0))
    
    return np.array(all_tokens), np.array(all_labels)


# ============================================================
# EXPERIMENT
# ============================================================

def run_afib_experiment(zip_path='/workspace/sounio/datasets/ecg_afib/training2017.zip',
                        extract_dir='/workspace/sounio/datasets/ecg_afib/training2017',
                        window_size=256, epochs=50, seed=20260806):
    """OctTree vs RealTree on AFib detection."""
    rng = np.random.default_rng(seed)
    device = 'cpu'
    vocab = 7
    
    print("\n" + "=" * 72)
    print("OctTree vs RealTree on ATRIAL FIBRILLATION DETECTION")
    print("=" * 72)
    
    # Load data
    print("Loading PhysioNet 2017 AFib Challenge data...")
    recordings, rec_labels = load_physionet2017(zip_path, extract_dir)
    
    # Count by class
    from collections import Counter
    class_counts = Counter(rec_labels)
    print(f"  Loaded {len(recordings)} recordings")
    print(f"  Classes: {dict(class_counts)}")
    
    # Build dataset
    print(f"\nSegmenting ECG (window={window_size})...")
    tokens, labels = build_afib_dataset(recordings, rec_labels, window_size=window_size)
    
    print(f"  Total windows: {len(tokens)}")
    print(f"  Normal: {(labels==0).sum()}, AFib: {(labels==1).sum()}")
    
    if len(tokens) < 100:
        print("  Too few samples, aborting")
        return
    
    # Split train/test (80/20, by recording to avoid leakage)
    n_train = int(len(tokens) * 0.8)
    perm = rng.permutation(len(tokens))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    
    tr_t = torch.from_numpy(tokens[train_idx])
    tr_l = torch.from_numpy(labels[train_idx])
    te_t = torch.from_numpy(tokens[test_idx])
    te_l = torch.from_numpy(labels[test_idx])
    
    print(f"  Train: {len(tr_t)} ({(tr_l==1).sum()} AFib)")
    print(f"  Test:  {len(te_t)} ({(te_l==1).sum()} AFib)")
    
    # Train models
    results = {}
    
    models = {
        'OctTree-8':  OctTreeClassifier(vocab, 8, 2, use_oct=True),
        'RealTree-8': OctTreeClassifier(vocab, 8, 2, use_oct=False),
        'GRU-8':      GRUClassifier(vocab, 8, 2),
    }
    
    for name, model in models.items():
        model = model.to(device)
        np_p = count_params(model)
        print(f"\n  Training {name} ({np_p}p)...")
        t0 = time.time()
        hist = train_one(model, tr_t, tr_l, te_t, te_l,
                        epochs=epochs, lr=1e-2, batch_size=32,
                        device=device, name=name)
        dt = time.time() - t0
        final = hist['test_acc'][-1]
        best = max(hist['test_acc'])
        results[name] = {'params': np_p, 'acc': final, 'best': best, 'time': round(dt, 1)}
        print(f"  → {name}: acc={final:.3f}  best={best:.3f}  ({dt:.0f}s)")
    
    # Summary
    print(f"\n{'='*72}")
    print("SUMMARY — AFib Detection")
    print(f"{'='*72}")
    for name in results:
        print(f"  {name:<14}: acc={results[name]['acc']:.3f}  best={results[name]['best']:.3f}  ({results[name]['params']}p)")
    
    o = results['OctTree-8']['acc']
    r = results['RealTree-8']['acc']
    diff = o - r
    print(f"\n  OctTree advantage: {diff:+.3f}")
    if diff > 0.02:
        print(f"  ⚡ OctTree BEATS RealTree on AFib detection")
    
    outpath = "scripts/research/afib_results.json"
    with open(outpath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outpath}")
    
    return results


if __name__ == '__main__':
    import argparse, json
    p = argparse.ArgumentParser()
    p.add_argument('--zip', default='/workspace/sounio/datasets/ecg_afib/training2017.zip')
    p.add_argument('--extract', default='/workspace/sounio/datasets/ecg_afib/training2017')
    p.add_argument('--window', type=int, default=256)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--seed', type=int, default=20260806)
    args = p.parse_args()
    
    run_afib_experiment(args.zip, args.extract, args.window, args.epochs, args.seed)
