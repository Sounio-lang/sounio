#!/usr/bin/env python3
"""Door F cohort — per-patient .sio generator.

Extracts 7 EEG windows (II at +10s, FAR at onset-60, PRE30 at -30, PRE10 at -10,
PRE5 at -5, IC at onset, POST at onset+5; each 80 samples at 256 Hz) from a
CHB-MIT EDF file and emits examples/door_f_assoc_preictal_<patient>.sio.

Normalization: z-score by II-train mean/std, then divide by II-train max_abs,
with clip(±5.0) on non-II windows (matches extract_chbmit_chb11_clean.py).

Targets: TGT[t] = CH0[t+1] within each window (next-sample prediction).

Usage:
    python3 scripts/research/door_f_cohort/generate.py            # all cohort
    python3 scripts/research/door_f_cohort/generate.py chb03      # one patient
"""
import os
import sys
import pyedflib
import numpy as np

COHORT = {
    "chb02": dict(edf="data/chbmit/chb02_16.edf", onset=130),
    "chb03": dict(edf="data/chbmit/chb03_01.edf", onset=362),
    "chb05": dict(edf="data/chbmit/chb05_06.edf", onset=417),
    "chb06": dict(edf="data/chbmit/chb06_04.edf", onset=327),
    "chb10": dict(edf="data/chbmit/chb10_12.edf", onset=6313),
}

FS               = 256
N_CH             = 16
WINDOW_LEN       = 80
N_TRAIN          = 64
INTERICTAL_S     = 10.0                 # II window start (seconds)
CH_MAP           = list(range(N_CH))    # first 16 channels (no blanks in our EDFs)

# (name, offset_from_onset_s)  —  order matters: it's the order of emitted data
WINDOWS = [
    ("FAR",   -60),
    ("PRE30", -30),
    ("PRE10", -10),
    ("PRE5",   -5),
    ("IC",      0),
    ("POST",   +5),
]

HERE    = os.path.dirname(os.path.abspath(__file__))
HEADER  = open(os.path.join(HERE, "header.sio.part")).read()
TAIL    = open(os.path.join(HERE, "template.sio.part")).read()


def load_windows(edf_path, onset_s):
    """Return (windows, norm_stats). windows[key] is (N_CH, WINDOW_LEN)."""
    f = pyedflib.EdfReader(edf_path)
    fs = int(f.getSampleFrequency(CH_MAP[0]))
    if fs != FS:
        raise RuntimeError(f"expected fs={FS}, got {fs} for {edf_path}")
    sigs = np.stack([f.readSignal(i) for i in CH_MAP])  # (16, T)
    total = sigs.shape[1]
    f.close()

    def window_at(t0_s):
        t0 = int(round(t0_s * FS))
        if t0 < 0 or t0 + WINDOW_LEN > total:
            raise RuntimeError(f"window @{t0_s}s (samples {t0}..{t0+WINDOW_LEN}) "
                               f"out of bounds (total={total})")
        return sigs[:, t0 : t0 + WINDOW_LEN]

    ii_raw   = window_at(INTERICTAL_S)
    probes   = {name: window_at(onset_s + off) for name, off in WINDOWS}

    means    = ii_raw[:, :N_TRAIN].mean(axis=1, keepdims=True)
    stds     = ii_raw[:, :N_TRAIN].std(axis=1, keepdims=True)
    stds     = np.where(stds < 1e-6, 1.0, stds)
    ii_z     = (ii_raw - means) / stds
    max_abs  = np.abs(ii_z[:, :N_TRAIN]).max(axis=1, keepdims=True)
    max_abs  = np.where(max_abs < 1e-6, 1.0, max_abs)
    ii_norm  = ii_z / max_abs

    probes_norm = {}
    for name, arr in probes.items():
        z = (arr - means) / stds / max_abs
        probes_norm[name] = np.clip(z, -5.0, 5.0)

    return ii_norm, probes_norm


def target_next(ch0_window):
    """TGT[t] = CH0[t+1] for t in 0..WINDOW_LEN-1, last = last (hold)."""
    tgt = np.zeros(WINDOW_LEN)
    tgt[:-1] = ch0_window[1:]
    tgt[-1]  = ch0_window[-1]
    return tgt


def emit_array(name, values):
    lines = []
    for i, v in enumerate(values):
        lines.append(f"    {name}[{i}] = {v:.8f}")
    return "\n".join(lines)


def build_init_data(ii_norm, probes_norm):
    """Emit 'fn init_data() { ... }' body with 7 windows × (16 ch + 1 tgt)."""
    parts = ["fn init_data() with Mut {"]
    # II window
    for c in range(N_CH):
        parts.append(emit_array(f"II_CH{c}", ii_norm[c]))
    parts.append(emit_array("II_TGT", target_next(ii_norm[0])))
    # 6 probe windows
    for name, _ in WINDOWS:
        arr = probes_norm[name]
        for c in range(N_CH):
            parts.append(emit_array(f"{name}_CH{c}", arr[c]))
        parts.append(emit_array(f"{name}_TGT", target_next(arr[0])))
    parts.append("}\n")
    return "\n".join(parts)


def generate(patient):
    cfg     = COHORT[patient]
    edf     = cfg["edf"]
    onset   = cfg["onset"]
    ii, pr  = load_windows(edf, onset)
    init    = build_init_data(ii, pr)
    header  = HEADER
    for k, v in [
        ("{{PATIENT}}", patient), ("{{EDF_PATH}}", edf), ("{{ONSET}}", str(onset)),
        ("{{T_FAR}}", str(onset - 60)), ("{{T_PRE30}}", str(onset - 30)),
        ("{{T_PRE10}}", str(onset - 10)), ("{{T_PRE5}}", str(onset - 5)),
        ("{{T_IC}}",   str(onset)),      ("{{T_POST}}",  str(onset + 5)),
    ]:
        header = header.replace(k, v)
    hdr_str = f"{patient}, seizure at {onset}s, 256Hz, 16ch"
    tail    = TAIL.replace("{{PATIENT_HEADER}}", hdr_str)
    out     = f"examples/door_f_assoc_preictal_{patient}.sio"
    with open(out, "w") as f:
        f.write(header)
        f.write("\n")
        f.write(init)
        f.write("\n")
        f.write(tail)
    nlines = 1 + header.count("\n") + 1 + init.count("\n") + tail.count("\n")
    sys.stderr.write(f"Wrote {out}  ({nlines} lines)\n")


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(COHORT.keys())
    for p in targets:
        if p not in COHORT:
            raise SystemExit(f"unknown patient: {p}  (known: {list(COHORT)})")
        generate(p)


if __name__ == "__main__":
    main()
