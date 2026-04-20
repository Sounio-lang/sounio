#!/usr/bin/env python3
"""Generate sliding-window manifest for seizure microdinâmica analysis.

Extracts overlapping 8-step windows (stride=4, 50% overlap) across a time range
spanning 60s pre-seizure → seizure → 60s post-seizure for each seizure event.

Output: TSV with columns:
  subject_id, label(0-4), site(patient), time_rel_onset(seconds), f0..f63

time_rel_onset is relative to seizure onset (negative = pre, 0..duration = ictal, >duration = post).
"""

import pyedflib
import numpy as np
import os
import sys

DATA_DIR = "data/chbmit"

CHANNEL_NAMES = [
    "FP1-F7", "F7-T7", "FP2-F8", "F8-T8",
    "FP1-F3", "F3-C3", "FP2-F4", "F4-C4",
]

WINDOW_SIZE = 8
STRIDE = 4  # 50% overlap

SEIZURE_DB = {
    "chb02": [("chb02_16.edf", 130, 212)],
    "chb03": [("chb03_01.edf", 362, 414), ("chb03_02.edf", 731, 796), ("chb03_03.edf", 432, 501)],
    "chb05": [("chb05_06.edf", 417, 532), ("chb05_13.edf", 1086, 1196)],
    "chb06": [("chb06_04.edf", 327, 347), ("chb06_13.edf", 506, 519)],
    "chb11": [("chb11_82.edf", 298, 320)],
}

PRE_SECONDS = 60
POST_SECONDS = 60


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
        if not found:
            return None
    return indices


def process_seizure(patient_id, edf_name, onset_sec, end_sec):
    edf_path = os.path.join(DATA_DIR, edf_name)
    if not os.path.exists(edf_path):
        return []

    try:
        f = pyedflib.EdfReader(edf_path, 0, 1)
    except Exception:
        return []

    ch_idx = find_channel_indices(f)
    if ch_idx is None or len(ch_idx) != 8:
        f.close()
        return []

    sigs = np.stack([f.readSignal(i) for i in ch_idx])
    sr = int(f.getSampleFrequency(ch_idx[0]))
    f.close()

    # Detect valid data
    nonzero_mask = np.any(sigs != 0, axis=0)
    nonzero_idx = np.nonzero(nonzero_mask)[0]
    if len(nonzero_idx) == 0:
        return []
    valid_end = int(nonzero_idx[-1]) + 1
    sigs = sigs[:, :valid_end]

    # Z-normalize
    for ch in range(8):
        mu = sigs[ch].mean()
        sd = sigs[ch].std()
        if sd > 1e-8:
            sigs[ch] = (sigs[ch] - mu) / sd

    onset_samp = onset_sec * sr
    end_samp = end_sec * sr
    duration_sec = end_sec - onset_sec

    range_start = max(0, onset_samp - PRE_SECONDS * sr)
    range_end = min(valid_end, end_samp + POST_SECONDS * sr)

    windows = []
    pos = range_start
    file_tag = edf_name.replace(".edf", "")

    while pos + WINDOW_SIZE <= range_end:
        w = sigs[:, pos:pos + WINDOW_SIZE]
        features = w.T.flatten()

        center_sec = (pos + WINDOW_SIZE // 2) / sr
        time_rel = center_sec - onset_sec

        if pos + WINDOW_SIZE <= onset_samp:
            label = "pre_ictal"
            label_code = 3
        elif pos >= end_samp:
            label = "post_ictal"
            label_code = 4
        else:
            label = "ictal"
            label_code = 2

        windows.append({
            "subject_id": f"{patient_id}_{file_tag}_{len(windows)}",
            "label_code": label_code,
            "site": patient_id,
            "time_rel": round(time_rel, 4),
            "seizure_id": f"{patient_id}_{file_tag}",
            "features": features,
        })
        pos += STRIDE

    return windows


def main():
    out_path = "artifacts/research/eeg_hessian/sliding_seizure_manifest.tsv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_windows = []
    for patient_id, seizures in sorted(SEIZURE_DB.items()):
        for edf_name, onset, end in seizures:
            ws = process_seizure(patient_id, edf_name, onset, end)
            if ws:
                all_windows.extend(ws)
                print(f"  {patient_id}/{edf_name}: {len(ws)} windows ({onset}-{end}s)")

    header = ["subject_id", "label", "site", "time_rel"] + [f"f{i}" for i in range(64)]

    with open(out_path, "w") as fp:
        fp.write("# schema=sliding_seizure.v1\n")
        fp.write("# stride=4\n")
        fp.write("# window=8\n")
        fp.write("# label_encoding=2:ictal,3:pre_ictal,4:post_ictal\n")
        fp.write("\t".join(header) + "\n")

        for w in all_windows:
            row = [w["subject_id"], str(w["label_code"]), w["site"], f"{w['time_rel']:.4f}"]
            row += [f"{v:.8f}" for v in w["features"]]
            fp.write("\t".join(row) + "\n")

    labels = [w["label_code"] for w in all_windows]
    print(f"\nTotal: {len(all_windows)} sliding windows")
    print(f"  pre_ictal:  {labels.count(3)}")
    print(f"  ictal:      {labels.count(2)}")
    print(f"  post_ictal: {labels.count(4)}")
    print(f"Written: {out_path}")


if __name__ == "__main__":
    main()
