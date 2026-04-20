#!/usr/bin/env python3
"""Generate compact sliding-window manifest for Sounio seizure dynamics.

Layout: subject_id, label, site, time_rel_centiseconds, f0..f63
  - label: 2=ictal, 3=pre_ictal, 4=post_ictal
  - site: patient ID (for leave-one-patient-out)
  - time_rel_centiseconds: integer, time relative to seizure onset in 1/100s
  - f0..f63: 8 channels × 8 timesteps

Stride: 128 samples (~0.5s at 256Hz)
Window: 8 samples
"""

import pyedflib
import numpy as np
import os

DATA_DIR = "data/chbmit"
CHANNEL_NAMES = ["FP1-F7","F7-T7","FP2-F8","F8-T8","FP1-F3","F3-C3","FP2-F4","F4-C4"]
WINDOW_SIZE = 8
STRIDE = 128  # ~0.5s at 256Hz
PRE_SEC = 60
POST_SEC = 60

SEIZURES = [
    ("chb02", "chb02_16.edf", 130, 212),
    ("chb03", "chb03_01.edf", 362, 414),
    ("chb03", "chb03_02.edf", 731, 796),
    ("chb03", "chb03_03.edf", 432, 501),
    ("chb05", "chb05_06.edf", 417, 532),
    ("chb05", "chb05_13.edf", 1086, 1196),
    ("chb06", "chb06_04.edf", 327, 347),
    ("chb06", "chb06_13.edf", 506, 519),
    ("chb11", "chb11_82.edf", 298, 320),
]


def find_channels(f):
    labels = [l.strip() for l in f.getSignalLabels()]
    idx = []
    for target in CHANNEL_NAMES:
        for i, l in enumerate(labels):
            if l.upper() == target.upper():
                idx.append(i)
                break
        else:
            return None
    return idx


def process(patient, edf_name, onset_sec, end_sec):
    path = os.path.join(DATA_DIR, edf_name)
    if not os.path.exists(path):
        return []
    try:
        f = pyedflib.EdfReader(path, 0, 1)
    except:
        return []

    ch_idx = find_channels(f)
    if ch_idx is None or len(ch_idx) != 8:
        f.close()
        return []

    sigs = np.stack([f.readSignal(i) for i in ch_idx])
    sr = int(f.getSampleFrequency(ch_idx[0]))
    f.close()

    nz = np.nonzero(np.any(sigs != 0, axis=0))[0]
    if len(nz) == 0:
        return []
    sigs = sigs[:, :int(nz[-1]) + 1]

    for ch in range(8):
        mu, sd = sigs[ch].mean(), sigs[ch].std()
        if sd > 1e-8:
            sigs[ch] = (sigs[ch] - mu) / sd

    onset_samp = onset_sec * sr
    end_samp = end_sec * sr
    range_start = max(0, onset_samp - PRE_SEC * sr)
    range_end = min(sigs.shape[1], end_samp + POST_SEC * sr)

    file_tag = edf_name.replace(".edf", "")
    rows = []
    pos = range_start
    idx = 0
    while pos + WINDOW_SIZE <= range_end:
        w = sigs[:, pos:pos + WINDOW_SIZE]
        features = w.T.flatten()

        center_samp = pos + WINDOW_SIZE // 2
        time_rel_cs = int(round((center_samp / sr - onset_sec) * 100))

        if center_samp < onset_samp:
            label = 3
        elif center_samp >= end_samp:
            label = 4
        else:
            label = 2

        rows.append({
            "id": f"{patient}_{file_tag}_{idx}",
            "label": label,
            "site": patient,
            "time_cs": time_rel_cs,
            "features": features,
        })
        pos += STRIDE
        idx += 1
    return rows


def main():
    out = "artifacts/research/eeg_hessian/dynamics_manifest.tsv"
    os.makedirs(os.path.dirname(out), exist_ok=True)

    all_rows = []
    for patient, edf, onset, end in SEIZURES:
        rows = process(patient, edf, onset, end)
        all_rows.extend(rows)
        print(f"  {edf}: {len(rows)} windows")

    header = ["subject_id", "label", "site", "time_cs"] + [f"f{i}" for i in range(64)]
    with open(out, "w") as fp:
        fp.write("# schema=dynamics.v1\n")
        fp.write("# stride=128\n")
        fp.write("# time_cs=centiseconds_relative_to_onset\n")
        fp.write("\t".join(header) + "\n")
        for r in all_rows:
            row = [r["id"], str(r["label"]), r["site"], str(r["time_cs"])]
            row += [f"{v:.8f}" for v in r["features"]]
            fp.write("\t".join(row) + "\n")

    lc = {2: 0, 3: 0, 4: 0}
    for r in all_rows:
        lc[r["label"]] = lc.get(r["label"], 0) + 1
    print(f"\nTotal: {len(all_rows)} windows")
    print(f"  pre_ictal:  {lc[3]}")
    print(f"  ictal:      {lc[2]}")
    print(f"  post_ictal: {lc[4]}")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
