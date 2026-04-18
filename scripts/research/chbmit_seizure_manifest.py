#!/usr/bin/env python3
"""Extract temporal EEG windows from CHB-MIT for O-SSM Hessian seizure analysis.

CHB-MIT Scalp EEG Database:
  - 256 Hz, 23 bipolar channels (10-20 montage)
  - Seizure onset/end annotated in summary files

Labels:
  2 = ictal     (during seizure, from annotated onset to end)
  1 = interictal (same file as seizure, but outside seizure window)
  0 = baseline   (file with zero seizures from the same patient)

We select 8 channels covering bilateral fronto-temporal propagation paths:
  FP1-F7, F7-T7, FP2-F8, F8-T8, FP1-F3, F3-C3, FP2-F4, F4-C4

Pipeline:
  1. Parse summary files for seizure annotations
  2. Read 8 channels from each EDF
  3. Extract non-overlapping 8-timestep windows (8 ch × 8 steps = 64 features)
  4. Z-normalize per channel within each recording
  5. Label each window as ictal / interictal / baseline
  6. Output TSV compatible with Sounio O-SSM Hessian benchmark
"""

import pyedflib
import numpy as np
import os
import re
import sys

DATA_DIR = "data/chbmit"

CHANNEL_NAMES = [
    "FP1-F7", "F7-T7",   # left temporal chain
    "FP2-F8", "F8-T8",   # right temporal chain
    "FP1-F3", "F3-C3",   # left central chain
    "FP2-F4", "F4-C4",   # right central chain
]

WINDOW_SIZE = 8      # 8 timesteps per window
SAMPLE_RATE = 256
MAX_WINDOWS_PER_SEGMENT = 60

SEIZURE_DB = {
    "chb02": {
        "seizure_files": {
            "chb02_16.edf": [(130, 212)],
        },
        "baseline_files": ["chb02_01.edf"],
    },
    "chb03": {
        "seizure_files": {
            "chb03_01.edf": [(362, 414)],
            "chb03_02.edf": [(731, 796)],
            "chb03_03.edf": [(432, 501)],
        },
        "baseline_files": ["chb03_05.edf"],
    },
    "chb05": {
        "seizure_files": {
            "chb05_06.edf": [(417, 532)],
            "chb05_13.edf": [(1086, 1196)],
            "chb05_16.edf": [(2317, 2413)],
        },
        "baseline_files": ["chb05_01.edf"],
    },
    "chb06": {
        "seizure_files": {
            "chb06_04.edf": [(327, 347)],
            "chb06_13.edf": [(506, 519)],
        },
        "baseline_files": ["chb06_02.edf"],
    },
    "chb11": {
        "seizure_files": {
            "chb11_82.edf": [(298, 320)],
        },
        "baseline_files": ["chb11_01.edf"],
    },
}


def find_channel_indices(f):
    labels = [lab.strip() for lab in f.getSignalLabels()]
    indices = []
    for target in CHANNEL_NAMES:
        found = False
        for i, lab in enumerate(labels):
            if lab.upper() == target.upper():
                indices.append(i)
                found = True
                break
            if lab.upper().replace(" ", "").replace(".", "") == target.upper().replace("-", ""):
                indices.append(i)
                found = True
                break
        if not found:
            return None
    return indices


def read_edf_signals(edf_path):
    try:
        f = pyedflib.EdfReader(edf_path, 0, 1)
    except Exception as e:
        print(f"  Warning: cannot read {edf_path}: {e}", file=sys.stderr)
        return None, None, 0

    ch_idx = find_channel_indices(f)
    if ch_idx is None or len(ch_idx) != 8:
        labels = f.getSignalLabels()
        print(f"  Warning: channel mismatch in {edf_path}. Available: {labels}", file=sys.stderr)
        f.close()
        return None, None, 0

    sigs = np.stack([f.readSignal(i) for i in ch_idx])  # (8, T)
    sr = int(f.getSampleFrequency(ch_idx[0]))
    f.close()

    # Detect actual data length (partial downloads leave trailing zeros)
    nonzero_mask = np.any(sigs != 0, axis=0)
    nonzero_idx = np.nonzero(nonzero_mask)[0]
    if len(nonzero_idx) == 0:
        return None, None, 0
    valid_end = int(nonzero_idx[-1]) + 1

    return sigs[:, :valid_end], sr, valid_end


def znormalize(sigs):
    for ch in range(sigs.shape[0]):
        mu = sigs[ch].mean()
        sd = sigs[ch].std()
        if sd > 1e-8:
            sigs[ch] = (sigs[ch] - mu) / sd
    return sigs


def extract_windows_from_range(sigs, start_sample, end_sample, label, subject_id, file_tag, max_windows):
    windows = []
    pos = start_sample
    count = 0
    n_samples = sigs.shape[1]

    while pos + WINDOW_SIZE <= min(end_sample, n_samples) and count < max_windows:
        w = sigs[:, pos:pos + WINDOW_SIZE]  # (8, 8)
        features = w.T.flatten()  # (64,)
        windows.append({
            "subject_id": f"{subject_id}_{file_tag}_{label}_{count}",
            "label": label,
            "site": subject_id,
            "features": features,
        })
        pos += WINDOW_SIZE
        count += 1

    return windows


def process_patient(patient_id, patient_info):
    all_windows = []

    # Process files with seizures
    for edf_name, seizure_ranges in patient_info["seizure_files"].items():
        edf_path = os.path.join(DATA_DIR, edf_name)
        if not os.path.exists(edf_path):
            print(f"  Skipping {edf_name}: not found")
            continue

        sigs, sr, valid_end = read_edf_signals(edf_path)
        if sigs is None:
            continue

        sigs = znormalize(sigs)
        n_samples = sigs.shape[1]
        file_tag = edf_name.replace(".edf", "")

        # Extract ictal windows (during seizure) + pre-ictal + post-ictal
        for (onset_sec, end_sec) in seizure_ranges:
            onset_samp = onset_sec * sr
            end_samp = end_sec * sr

            # Pre-ictal: 30s before onset
            pre_start = max(5 * sr, onset_samp - 30 * sr)
            pre_end = onset_samp
            pre_ictal = extract_windows_from_range(
                sigs, pre_start, pre_end,
                "pre_ictal", patient_id, file_tag,
                MAX_WINDOWS_PER_SEGMENT
            )
            all_windows.extend(pre_ictal)

            ictal = extract_windows_from_range(
                sigs, onset_samp, end_samp,
                "ictal", patient_id, file_tag,
                MAX_WINDOWS_PER_SEGMENT
            )
            all_windows.extend(ictal)

            # Post-ictal: 30s after end
            post_start = end_samp
            post_end = min(n_samples, end_samp + 30 * sr)
            post_ictal = extract_windows_from_range(
                sigs, post_start, post_end,
                "post_ictal", patient_id, file_tag,
                MAX_WINDOWS_PER_SEGMENT
            )
            all_windows.extend(post_ictal)

        # Extract interictal windows (same file, away from seizure)
        # Use first 60 seconds of the file (well before any seizure)
        inter_start = 5 * sr  # skip first 5 seconds
        inter_end = min(65 * sr, n_samples)
        earliest_seizure = min(s[0] for s in seizure_ranges) * sr
        if inter_end > earliest_seizure - 60 * sr:
            inter_end = max(inter_start + WINDOW_SIZE, earliest_seizure - 60 * sr)

        interictal = extract_windows_from_range(
            sigs, inter_start, inter_end,
            "interictal", patient_id, file_tag,
            MAX_WINDOWS_PER_SEGMENT
        )
        all_windows.extend(interictal)

    # Process baseline files (no seizures at all)
    for edf_name in patient_info["baseline_files"]:
        edf_path = os.path.join(DATA_DIR, edf_name)
        if not os.path.exists(edf_path):
            print(f"  Skipping baseline {edf_name}: not found")
            continue

        sigs, sr, valid_end = read_edf_signals(edf_path)
        if sigs is None:
            continue

        sigs = znormalize(sigs)
        file_tag = edf_name.replace(".edf", "")

        # Extract from middle of the recording (most stable)
        mid = sigs.shape[1] // 2
        base_start = max(0, mid - 30 * sr)
        base_end = mid + 30 * sr

        baseline = extract_windows_from_range(
            sigs, base_start, base_end,
            "baseline", patient_id, file_tag,
            MAX_WINDOWS_PER_SEGMENT
        )
        all_windows.extend(baseline)

    return all_windows


def main():
    out_path = "artifacts/research/eeg_hessian/seizure_hessian_manifest.tsv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_windows = []
    for patient_id, patient_info in sorted(SEIZURE_DB.items()):
        print(f"Processing {patient_id}...")
        patient_windows = process_patient(patient_id, patient_info)
        all_windows.extend(patient_windows)
        pc = {}
        for w in patient_windows:
            pc[w["label"]] = pc.get(w["label"], 0) + 1
        parts = " ".join(f"{k}={v}" for k, v in sorted(pc.items()))
        print(f"  -> {len(patient_windows)} windows ({parts})")

    # Encode labels: ictal=2, interictal=1, baseline=0, pre_ictal=3, post_ictal=4
    label_map = {"ictal": 2, "interictal": 1, "baseline": 0, "pre_ictal": 3, "post_ictal": 4}

    header_cols = ["subject_id", "label", "site"] + [f"f{i}" for i in range(64)]

    with open(out_path, "w") as fp:
        fp.write("# schema=seizure_ossm_hessian.v1\n")
        fp.write("# seq_len=8\n")
        fp.write("# input_dim=8\n")
        fp.write("# feature_layout=flat\n")
        fp.write("# label_space=baseline_interictal_ictal_pre_post\n")
        fp.write("# label_encoding=0:baseline,1:interictal,2:ictal,3:pre_ictal,4:post_ictal\n")
        fp.write("\t".join(header_cols) + "\n")

        for w in all_windows:
            row = [
                w["subject_id"],
                str(label_map[w["label"]]),
                w["site"],
            ]
            row += [f"{v:.8f}" for v in w["features"]]
            fp.write("\t".join(row) + "\n")

    counts = {}
    for w in all_windows:
        counts[w["label"]] = counts.get(w["label"], 0) + 1

    print(f"\nTotal: {len(all_windows)} windows")
    for lab in ["ictal", "pre_ictal", "post_ictal", "interictal", "baseline"]:
        print(f"  {lab:12s}: {counts.get(lab, 0)}")
    print(f"  patients:   {len(SEIZURE_DB)}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
