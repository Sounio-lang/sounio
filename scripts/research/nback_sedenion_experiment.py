#!/usr/bin/env python3
"""
O-SSM on N-back EEG: Octonion vs Sedenion on cognitive load.

The n-back task has increasing cognitive load (1-back → 4-back).
At high load (3-4 back), multiple working memory items coexist and
interfere — this is the EEG analog of crossing structure.

HYPOTHESIS: Sedenion associator (F3) discriminates n-back level better
than octonion associator (F1), because high load creates crossing-like
structure in neural dynamics.

The sedenion's non-alternativity captures the interference that the
octonion's alternativity cannot.
"""

import numpy as np
import sys, os, time
import struct

sys.path.insert(0, os.path.dirname(__file__))

# Octonion arithmetic
_FANO = [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(5,6,1),(6,7,2),(7,1,3)]

def _build_oct():
    sign = np.zeros((8,8)); idx = np.zeros((8,8), dtype=int)
    for i in range(8):
        sign[i,0]=1; idx[i,0]=i; sign[0,i]=1; idx[0,i]=i
        sign[i,i]=-1; idx[i,i]=0
    for a,b,c in _FANO:
        for p,q,r in [(a,b,c),(b,c,a),(c,a,b)]:
            sign[p,q]=1; idx[p,q]=r
        for p,q,r in [(b,a,c),(c,b,a),(a,c,b)]:
            sign[p,q]=-1; idx[p,q]=r
    return sign, idx

_OCT_S, _OCT_I = _build_oct()
_T = np.zeros((8,8,8))
for i in range(8):
    for j in range(8):
        _T[i,j,int(_OCT_I[i,j])] = _OCT_S[i,j]
_T_KJ = np.transpose(_T, (0,2,1)).copy()

def oct_mul(a, b):
    out = np.zeros(8)
    for i in range(8):
        for j in range(8):
            out[int(_OCT_I[i,j])] += _OCT_S[i,j] * a[i] * b[j]
    return out

def oct_assoc_norm(a, b, c):
    return np.linalg.norm(oct_mul(oct_mul(a,b),c) - oct_mul(a,oct_mul(b,c)))


# Sedenion arithmetic
def _build_sed():
    """Build 16×16 sedenion tables via Cayley-Dickson doubling (corrected).

    CD formula: (a,b)(c,d) = (ac - conj(d)b, da + b conj(c))
    """
    dim = 16; sign = np.ones((dim,dim)); idx = np.zeros((dim,dim), dtype=int)
    # Octonion block
    for i in range(8):
        sign[0,i]=1; idx[0,i]=i; sign[i,0]=1; idx[i,0]=i
        sign[i,i]=-1; idx[i,i]=0
    for a,b,c in _FANO:
        for p,q,r in [(a,b,c),(b,c,a),(c,a,b)]:
            sign[p,q]=1; idx[p,q]=r
        for p,q,r in [(b,a,c),(c,b,a),(a,c,b)]:
            sign[p,q]=-1; idx[p,q]=r
    OCT_S = sign[:8,:8].copy(); OCT_I = idx[:8,:8].copy()
    # CD cross-blocks
    for a in range(8):
        for b in range(8):
            if b == 0:
                sign[8+a, b] = 1;  idx[8+a, b] = 8 + a
            else:
                sign[8+a, b] = -OCT_S[a, b]
                idx[8+a, b] = 8 + int(OCT_I[a, b])
            if a == 0:
                sign[a, 8+b] = 1;  idx[a, 8+b] = 8 + b
            else:
                sign[a, 8+b] = OCT_S[b, a]
                idx[a, 8+b] = 8 + int(OCT_I[b, a])
            if b == 0:
                sign[8+a, 8+b] = -1; idx[8+a, 8+b] = a
            else:
                sign[8+a, 8+b] = OCT_S[b, a]
                idx[8+a, 8+b] = int(OCT_I[b, a])
    for i in range(16):
        sign[0,i]=1; idx[0,i]=i; sign[i,0]=1; idx[i,0]=i
    for i in range(1,16):
        sign[i,i]=-1; idx[i,i]=0
    return sign, idx

_SED_S, _SED_I = _build_sed()

def sed_mul(a, b):
    out = np.zeros(16)
    for i in range(16):
        for j in range(16):
            out[int(_SED_I[i,j])] += _SED_S[i,j] * a[i] * b[j]
    return out

def sed_assoc_norm(a, b, c):
    return np.linalg.norm(sed_mul(sed_mul(a,b),c) - sed_mul(a,sed_mul(b,c)))


# ============================================================
# O-SSM FORWARD PASS
# ============================================================

def ossm_forward(signal, dim=8, use_sedenion=False):
    """Process a multi-channel signal through O-SSM, return trajectory.
    
    signal: (T, n_channels) — must have ≤7 channels for octonion, ≤15 for sedenion
    Returns: trajectory (T, 2, dim) — two octonion/sedenion states per timestep
    """
    T_len, n_ch = signal.shape
    d = dim
    
    if use_sedenion:
        mul_fn = sed_mul
    else:
        mul_fn = oct_mul
    
    # Pack channels into state
    def pack(x):
        v = np.zeros(d)
        n = min(n_ch, d-1)
        v[:n] = x[:n]
        v[n] = 1.0  # bias
        return v
    
    # Two-state O-SSM
    h1 = np.zeros(d)
    h2 = np.zeros(d)
    A1 = np.ones(d) * 0.3
    A2 = np.ones(d) * (-0.2)  # different transition for second state
    
    trajectory = np.zeros((T_len, 2, d))
    
    for t in range(T_len):
        x = pack(signal[t])
        
        # State 1: h1 = tanh(A1 ⊗ h1 + x)
        ah1 = mul_fn(A1, h1)
        h1 = np.tanh(ah1 + x)
        
        # State 2: h2 = tanh(A2 ⊗ h2 + h1)  — depends on state 1
        ah2 = mul_fn(A2, h2)
        h2 = np.tanh(ah2 + h1)
        
        trajectory[t, 0] = h1
        trajectory[t, 1] = h2
    
    return trajectory


def compute_features(trajectory, use_sedenion=False):
    """Compute F1 (octonion assoc) and F3 (sedenion assoc) from trajectory."""
    T_len = trajectory.shape[0]
    d = trajectory.shape[2]
    
    mul_fn = sed_mul if use_sedenion else oct_mul
    assoc_fn = sed_assoc_norm if use_sedenion else oct_assoc_norm
    
    assoc_values = []
    
    for t in range(2, T_len):
        # Associator [h_{t-2}, h_{t-1}, h_t] for both states
        for state in range(2):
            a = trajectory[t-2, state]
            b = trajectory[t-1, state]
            c = trajectory[t, state]
            assoc_values.append(assoc_fn(a, b, c))
    
    return np.median(assoc_values), np.mean(assoc_values), np.std(assoc_values)


# ============================================================
# LOAD N-BACK DATA
# ============================================================

def load_brainvision_eeg(vhdr_path):
    """Load BrainVision EEG data."""
    vhdr_dir = os.path.dirname(vhdr_path)
    
    data_file = None
    n_channels = 0
    sample_rate = 0
    data_format = 'INT_16'
    
    with open(vhdr_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('DataFile='):
                data_file = line.split('=', 1)[1].strip()
            elif line.startswith('NumberOfChannels='):
                n_channels = int(line.split('=', 1)[1].strip())
            elif line.startswith('SamplingInterval='):
                interval = float(line.split('=', 1)[1].strip())
                sample_rate = int(1000000.0 / interval)
            elif line.startswith('BinaryFormat='):
                data_format = line.split('=', 1)[1].strip()
    
    if data_file is None:
        return None, 0, 0
    
    eeg_path = os.path.join(vhdr_dir, data_file)
    
    if data_format == 'IEEE_FLOAT_32':
        raw = np.fromfile(eeg_path, dtype=np.float32)
    else:
        raw = np.fromfile(eeg_path, dtype=np.int16)
    
    raw = raw.reshape(-1, n_channels)
    
    return raw.astype(np.float64), n_channels, sample_rate


def load_nback_events(tsv_path):
    """Load n-back events to find trial onsets and levels."""
    events = []
    with open(tsv_path) as f:
        headers = f.readline().strip().split('\t')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= len(headers):
                row = dict(zip(headers, parts))
                events.append(row)
    return events


def segment_by_nback_level(eeg_data, events, sample_rate, n_channels,
                           pre_stim=1.0, post_stim=3.0):
    """Segment EEG into epochs by n-back level.
    
    Returns dict: {level: list of (T, n_ch) epochs}
    """
    pre_samples = int(pre_stim * sample_rate)
    post_samples = int(post_stim * sample_rate)
    
    epochs_by_level = {1: [], 2: [], 3: [], 4: []}
    
    for ev in events:
        trial_type = ev.get('trial_type', '')
        if trial_type not in ('1-back', '2-back', '3-back', '4-back'):
            continue
        
        nback = ev.get('nback_level', '')
        if not nback or nback == 'n/a':
            continue
        
        try:
            level = int(nback)
        except:
            continue
        
        if level not in epochs_by_level:
            continue
        
        onset = float(ev.get('onset', 0))
        onset_sample = int(onset * sample_rate)
        
        start = onset_sample - pre_samples
        end = onset_sample + post_samples
        
        if start >= 0 and end < eeg_data.shape[0]:
            epoch = eeg_data[start:end, :n_channels]
            # Z-score per channel
            for ch in range(epoch.shape[1]):
                mu = np.mean(epoch[:, ch])
                sigma = np.std(epoch[:, ch]) + 1e-8
                epoch[:, ch] = (epoch[:, ch] - mu) / sigma
            
            epochs_by_level[level].append(epoch)
    
    return epochs_by_level


# ============================================================
# EXPERIMENT
# ============================================================

def run_nback_experiment(data_root='/workspace/sounio/datasets/eeg_switching/ds007169_nback',
                         max_subjects=18, max_epochs_per_level=20):
    """Run O-SSM octonion vs sedenion on n-back EEG."""
    
    print("\n" + "=" * 72)
    print("O-SSM OCTONION vs SEDENION ON N-BACK EEG")
    print("=" * 72)
    
    all_features = {1: [], 2: [], 3: [], 4: []}
    
    subjects = sorted([d for d in os.listdir(data_root) if d.startswith('sub-')])
    
    for sub_idx, sub_id in enumerate(subjects[:max_subjects]):
        sub_dir = os.path.join(data_root, sub_id)
        eeg_dir = os.path.join(sub_dir, 'eeg')
        
        vhdr = os.path.join(eeg_dir, f'{sub_id}_task-nback_eeg.vhdr')
        events_tsv = os.path.join(eeg_dir, f'{sub_id}_task-nback_events.tsv')
        
        if not os.path.exists(vhdr) or not os.path.exists(events_tsv):
            continue
        
        print(f"\n[{sub_idx+1}/{min(len(subjects), max_subjects)}] {sub_id}...", end=" ")
        
        # Load data
        eeg_data, n_ch, sr = load_brainvision_eeg(vhdr)
        if eeg_data is None or n_ch < 7:
            print("skip (no data)")
            continue
        
        events = load_nback_events(events_tsv)
        epochs = segment_by_nback_level(eeg_data, events, sr, min(n_ch, 7))
        
        # Count epochs per level
        counts = {l: len(e) for l, e in epochs.items()}
        print(f"epochs: " + " ".join(f"{l}b={counts[l]}" for l in [1,2,3,4]))
        
        # Process epochs through O-SSM
        for level in [1, 2, 3, 4]:
            for epoch in epochs[level][:max_epochs_per_level]:
                # Octonion features
                traj_oct = ossm_forward(epoch, dim=8, use_sedenion=False)
                f1_med, f1_mean, f1_std = compute_features(traj_oct, use_sedenion=False)
                
                # Sedenion features
                traj_sed = ossm_forward(epoch, dim=16, use_sedenion=True)
                f3_med, f3_mean, f3_std = compute_features(traj_sed, use_sedenion=True)
                
                all_features[level].append({
                    'F1_oct_median': f1_med, 'F1_oct_mean': f1_mean,
                    'F3_sed_median': f3_med, 'F3_sed_mean': f3_mean,
                    'F3_over_F1': f3_mean / max(f1_mean, 1e-8),
                })
    
    # Summary
    print(f"\n{'='*72}")
    print("RESULTS — Octonion vs Sedenion by N-back Level")
    print(f"{'='*72}")
    print(f"{'Level':<8} {'F1(oct)':<16} {'F3(sed)':<16} {'F3/F1':<10} {'n':<6}")
    print("-" * 56)
    
    for level in [1, 2, 3, 4]:
        feats = all_features[level]
        if not feats:
            print(f"{level}-back  (no data)")
            continue
        
        f1 = np.mean([f['F1_oct_mean'] for f in feats])
        f1_se = np.std([f['F1_oct_mean'] for f in feats]) / np.sqrt(len(feats))
        f3 = np.mean([f['F3_sed_mean'] for f in feats])
        f3_se = np.std([f['F3_sed_mean'] for f in feats]) / np.sqrt(len(feats))
        ratio = np.mean([f['F3_over_F1'] for f in feats])
        
        print(f"{level}-back  {f1:.3f}±{f1_se:.3f}   {f3:.3f}±{f3_se:.3f}   {ratio:.2f}   {len(feats)}")
    
    # Test: does sedenion discriminate levels better?
    print(f"\nDiscrimination (Cohen's d, 1-back vs 4-back):")
    f1_1 = [f['F1_oct_mean'] for f in all_features.get(1, [])]
    f1_4 = [f['F1_oct_mean'] for f in all_features.get(4, [])]
    f3_1 = [f['F3_sed_mean'] for f in all_features.get(1, [])]
    f3_4 = [f['F3_sed_mean'] for f in all_features.get(4, [])]
    
    if len(f1_1) > 2 and len(f1_4) > 2:
        def cohens_d(a, b):
            pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
            return (np.mean(a) - np.mean(b)) / max(pooled, 1e-8)
        
        d_oct = cohens_d(f1_1, f1_4)
        d_sed = cohens_d(f3_1, f3_4)
        print(f"  Octonion (F1): d = {d_oct:+.3f}")
        print(f"  Sedenion (F3): d = {d_sed:+.3f}")
        if abs(d_sed) > abs(d_oct):
            print(f"  ⚡ Sedenion discriminates 1-back vs 4-back BETTER")
        else:
            print(f"  Octonion discriminates better (or equal)")
    
    return all_features


if __name__ == '__main__':
    results = run_nback_experiment()
