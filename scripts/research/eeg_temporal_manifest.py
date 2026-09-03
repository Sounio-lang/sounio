#!/usr/bin/env python3
"""Extract temporal EEG windows from EEGMMIDB for O-SSM Hessian analysis.

EEGMMIDB runs:
  R01 = baseline eyes open (resting, 1 min)
  R02 = baseline eyes closed (resting, 1 min)
  R03 = motor imagery left/right fist (task, 2 min)
  R04 = motor imagery both fists/feet (task, 2 min)

We label:
  R01 → "rest"    (baseline resting-state)
  R03 → "task"    (motor imagery — active cortical propagation)

Hypothesis: task conditions produce more path-dependent (non-associative)
cortical propagation than rest, measurable via O-SSM Hessian off-diagonal.

Pipeline:
  1. Read 8 frontal/central channels from EDF
  2. Extract non-overlapping windows of 8 timesteps (8 channels × 8 steps = 64 features)
  3. Z-normalize per channel within each recording
  4. Output TSV: subject_id, label, site, f0..f63

Using 8 channels: Fc5, Fc3, Fc1, Fcz, Fc2, Fc4, Fc6, C3
(fronto-central strip, motor-relevant)
"""

import pyedflib
import numpy as np
import os
import sys

DATA_DIR = "data/eegmmidb"
CHANNELS = ["Fc5.", "Fc3.", "Fc1.", "Fcz.", "Fc2.", "Fc4.", "Fc6.", "C3.."]
WINDOW_SIZE = 8  # 8 timesteps per window
SKIP_START = 320  # skip first 2 seconds (160 Hz × 2)
MAX_WINDOWS_PER_RUN = 40


def find_channel_indices(f):
    labels = f.getSignalLabels()
    indices = []
    for target in CHANNELS:
        found = False
        for i, lab in enumerate(labels):
            if lab.strip().lower().startswith(target.rstrip(".").lower()):
                indices.append(i)
                found = True
                break
        if not found:
            return None
    return indices


def extract_windows(edf_path, label, subject_id):
    """Extract windows from one EDF file."""
    try:
        f = pyedflib.EdfReader(edf_path)
    except Exception:
        return []

    ch_idx = find_channel_indices(f)
    if ch_idx is None or len(ch_idx) != 8:
        f.close()
        return []

    sigs = np.stack([f.readSignal(i) for i in ch_idx])  # (8, T)
    f.close()

    n_samples = sigs.shape[1]
    if n_samples < SKIP_START + WINDOW_SIZE:
        return []

    # Z-normalize per channel
    for ch in range(8):
        mu = sigs[ch].mean()
        sd = sigs[ch].std()
        if sd > 1e-8:
            sigs[ch] = (sigs[ch] - mu) / sd

    windows = []
    pos = SKIP_START
    count = 0
    while pos + WINDOW_SIZE <= n_samples and count < MAX_WINDOWS_PER_RUN:
        w = sigs[:, pos:pos + WINDOW_SIZE]  # (8, 8)
        features = w.T.flatten()  # (64,) — row-major: step0_ch0..ch7, step1_ch0..ch7, ...
        windows.append({
            "subject_id": f"{subject_id}_{label}_{count}",
            "label": label,
            "site": subject_id,
            "features": features,
        })
        pos += WINDOW_SIZE
        count += 1

    return windows


def main():
    out_path = "artifacts/research/eeg_hessian/eeg_temporal_manifest.tsv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_windows = []
    subjects_found = 0

    for subj_num in range(1, 31):
        subj_id = f"S{subj_num:03d}"
        r01_path = os.path.join(DATA_DIR, f"{subj_id}R01.edf")  # rest
        r03_path = os.path.join(DATA_DIR, f"{subj_id}R03.edf")  # task

        if not os.path.exists(r01_path) or not os.path.exists(r03_path):
            continue

        rest_windows = extract_windows(r01_path, "rest", subj_id)
        task_windows = extract_windows(r03_path, "task", subj_id)

        if rest_windows and task_windows:
            all_windows.extend(rest_windows)
            all_windows.extend(task_windows)
            subjects_found += 1

    # Write TSV
    header_cols = ["subject_id", "label", "site"] + [f"f{i}" for i in range(64)]

    with open(out_path, "w") as fp:
        fp.write("# schema=eeg_ossm_hessian.v1\n")
        fp.write("# seq_len=8\n")
        fp.write("# input_dim=8\n")
        fp.write("# feature_layout=flat\n")
        fp.write("# label_space=rest_vs_task\n")
        fp.write("\t".join(header_cols) + "\n")

        for w in all_windows:
            row = [w["subject_id"], "1" if w["label"] == "task" else "0", w["site"]]
            row += [f"{v:.8f}" for v in w["features"]]
            fp.write("\t".join(row) + "\n")

    rest_n = sum(1 for w in all_windows if w["label"] == "rest")
    task_n = sum(1 for w in all_windows if w["label"] == "task")

    print(f"Subjects: {subjects_found}")
    print(f"Windows: {len(all_windows)} (rest={rest_n}, task={task_n})")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
