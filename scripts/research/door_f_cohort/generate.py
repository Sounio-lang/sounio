#!/usr/bin/env python3
"""Door F cohort — per-patient .sio generator.

Extracts 7 EEG windows (II at +10s, FAR at onset-60, PRE30 at -30, PRE10 at -10,
PRE5 at -5, IC at onset, POST at onset+5; each 80 samples at 256 Hz) from a
CHB-MIT EDF file and emits examples/door_f_assoc_preictal_<patient>.sio.

Normalization: z-score by II-train mean/std, then divide by II-train max_abs,
with clip(±5.0) on non-II windows (matches extract_chbmit_chb11_clean.py).

Targets: TGT[t] = CH0[t+1] within each window (next-sample prediction).

Usage:
    # Legacy (5-patient hardcoded cohort from data/chbmit/):
    python3 scripts/research/door_f_cohort/generate.py            # all
    python3 scripts/research/door_f_cohort/generate.py chb03      # one

    # Manifest-driven (any patient, EDFs anywhere — used by cluster):
    python3 scripts/research/door_f_cohort/generate.py \\
        --patient chb07 --edf /orangefs/.../chb07_12.edf \\
        --onset 4920 --out /path/out.sio
"""
import os
import sys
import argparse
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
NULL_POOL_SIZE   = 100                  # P5.3 — per-patient null-pool of 80-sample slots
CH_MAP_DEFAULT   = list(range(N_CH))    # first 16 channels (no blanks in our EDFs)

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


def load_windows(edf_path, onset_s, ch_map=None,
                 null_pool_size=0, null_seed=0, null_offsets=None):
    """Return (ii_norm, probes_norm, null_pool_norm).

    - ii_norm: (N_CH, WINDOW_LEN) normalised baseline window
    - probes_norm: dict name -> (N_CH, WINDOW_LEN) for the 6 probe windows
    - null_pool_norm: (N, N_CH, WINDOW_LEN) or None if empty.
      Two selection modes for the null pool:
        * null_offsets (explicit): list of sample offsets (int) giving the
          start of each 80-sample epoch; overrides null_pool_size.
        * null_pool_size (random): samples N non-overlapping slices
          uniformly at random, excluding a ±WINDOW_LEN neighbourhood of each
          reserved window.
      Uses the same II-train normalisation (mean/std/max_abs) as the probe
      windows, so null and observed statistics are directly comparable.
    """
    ch_map = list(ch_map) if ch_map is not None else CH_MAP_DEFAULT
    if len(ch_map) != N_CH:
        raise RuntimeError(f"ch_map must have {N_CH} channels, got {len(ch_map)}")
    f = pyedflib.EdfReader(edf_path)
    fs = int(f.getSampleFrequency(ch_map[0]))
    if fs != FS:
        raise RuntimeError(f"expected fs={FS}, got {fs} for {edf_path}")
    sigs = np.stack([f.readSignal(i) for i in ch_map])  # (16, T)
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

    null_pool_norm = None
    chosen_samples = None
    if null_offsets is not None and len(null_offsets) > 0:
        chosen = np.asarray(null_offsets, dtype=np.int64)
        if (chosen < 0).any() or (chosen + WINDOW_LEN > total).any():
            raise RuntimeError(
                f"null_offsets out of range [0, {total - WINDOW_LEN}]: "
                f"min={chosen.min()}, max={chosen.max()}"
            )
        chosen_samples = chosen
    elif null_pool_size > 0:
        rng = np.random.default_rng(null_seed)
        # Forbid any 80-sample slot whose centre is within 2*WINDOW_LEN samples
        # of any reserved window centre (II + 6 probes).
        reserved_centers_s = [INTERICTAL_S] + [onset_s + off for _, off in WINDOWS]
        reserved_samples = [int(round(s * FS)) + WINDOW_LEN // 2
                            for s in reserved_centers_s]
        forbid_radius = 2 * WINDOW_LEN
        candidates = np.arange(0, total - WINDOW_LEN, WINDOW_LEN)
        keep_mask = np.ones_like(candidates, dtype=bool)
        for c in reserved_samples:
            keep_mask &= np.abs(candidates + WINDOW_LEN // 2 - c) >= forbid_radius
        candidates = candidates[keep_mask]
        if len(candidates) < null_pool_size:
            raise RuntimeError(
                f"recording too short for null pool: {len(candidates)} "
                f"candidate slots vs {null_pool_size} requested"
            )
        chosen = rng.choice(candidates, size=null_pool_size, replace=False)
        chosen.sort()
        chosen_samples = chosen

    if chosen_samples is not None:
        null_raw = np.stack([sigs[:, s : s + WINDOW_LEN] for s in chosen_samples])
        null_z = (null_raw - means[None, :, :]) / stds[None, :, :] / max_abs[None, :, :]
        null_pool_norm = np.clip(null_z, -5.0, 5.0)

    return ii_norm, probes_norm, null_pool_norm


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


def build_null_decls(null_pool_norm):
    """Emit top-level `var NULL_CHx: [f64; N*80] = [0.0; N*80]` declarations.

    The NULL arrays must be declared at module scope (not inside init_data)
    because Sounio globals live there, matching the II/FAR/... declarations
    in header.sio.part.
    """
    if null_pool_norm is None:
        return ""
    n_slots = null_pool_norm.shape[0]
    total_len = n_slots * WINDOW_LEN
    lines = ["// ── P5.3 null-pool data arrays (generated) ──────────────────"]
    for c in range(N_CH):
        lines.append(f"var NULL_CH{c}: [f64; {total_len}] = [0.0; {total_len}]")
    return "\n".join(lines) + "\n"


def build_init_data(ii_norm, probes_norm, null_pool_norm=None):
    """Emit 'fn init_data() { ... }' body with 7 windows × (16 ch + 1 tgt).

    If null_pool_norm is provided (shape (N, N_CH, WINDOW_LEN)), also fills
    NULL_CH0..CH15 arrays of length N*WINDOW_LEN, concatenated contiguously
    (slot s occupies indices [s*WINDOW_LEN, (s+1)*WINDOW_LEN)).
    """
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
    # Null pool (concatenated per channel)
    if null_pool_norm is not None:
        # null_pool_norm: (N, N_CH, WINDOW_LEN)
        for c in range(N_CH):
            flat = null_pool_norm[:, c, :].reshape(-1)
            parts.append(emit_array(f"NULL_CH{c}", flat))
    parts.append("}\n")
    return "\n".join(parts)


def generate(patient, edf, onset, out_path,
             ch_map=None, null_pool_size=0, null_seed=0,
             null_offsets=None, tail_override=None):
    ii, pr, null_pool = load_windows(edf, onset,
                                     ch_map=ch_map,
                                     null_pool_size=null_pool_size,
                                     null_seed=null_seed,
                                     null_offsets=null_offsets)
    null_decls = build_null_decls(null_pool)
    init    = build_init_data(ii, pr, null_pool_norm=null_pool)
    effective_null_size = (null_pool.shape[0] if null_pool is not None else 0)
    header  = HEADER
    for k, v in [
        ("{{PATIENT}}", patient), ("{{EDF_PATH}}", edf), ("{{ONSET}}", str(onset)),
        ("{{T_FAR}}", str(onset - 60)), ("{{T_PRE30}}", str(onset - 30)),
        ("{{T_PRE10}}", str(onset - 10)), ("{{T_PRE5}}", str(onset - 5)),
        ("{{T_IC}}",   str(onset)),      ("{{T_POST}}",  str(onset + 5)),
    ]:
        header = header.replace(k, v)
    hdr_str = f"{patient}, seizure at {onset}s, 256Hz, 16ch"
    tail_src = tail_override if tail_override is not None else TAIL
    tail    = tail_src.replace("{{PATIENT_HEADER}}", hdr_str)
    tail    = tail.replace("{{NULL_POOL_SIZE}}", str(effective_null_size))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(header); f.write("\n")
        if null_decls:
            f.write(null_decls); f.write("\n")
        f.write(init);   f.write("\n")
        f.write(tail)
    nlines = (1 + header.count("\n") + 1 + null_decls.count("\n")
              + init.count("\n") + tail.count("\n"))
    sys.stderr.write(f"Wrote {out_path}  ({nlines} lines)\n")


def _parse_channels(raw):
    if raw is None:
        return None
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    return [int(x) for x in parts]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patient", help="patient id, e.g. chb07")
    ap.add_argument("--edf",     help="path to EDF")
    ap.add_argument("--onset",   type=int, help="seizure onset in seconds")
    ap.add_argument("--out",     help="output .sio path")
    ap.add_argument("--channels", help="comma-separated EDF channel indices (length 16). "
                                       "Default: first 16 channels (0..15).")
    ap.add_argument("--null-pool-size", type=int, default=0,
                    help="If > 0, emit an additional NULL_CHx array of "
                         "<size> non-overlapping 80-sample slices sampled "
                         "at random from the recording (forbidden: ±2 window "
                         "lengths around reserved centers).")
    ap.add_argument("--null-seed", type=int, default=20260420,
                    help="RNG seed for null-pool slot selection; default is the "
                         "pre-registration seed.")
    ap.add_argument("--null-offsets",
                    help="Comma-separated sample offsets (integers) specifying "
                         "the start of each 80-sample null epoch. Overrides "
                         "--null-pool-size and --null-seed when set. Use this "
                         "for contiguous-epoch trajectories and chunked runs.")
    ap.add_argument("--tail",
                    help="Path to alternate template (tail) .part file; defaults to "
                         "template.sio.part.")
    ap.add_argument("patients",  nargs="*",
                    help="legacy positional: patient ids from hardcoded COHORT")
    args = ap.parse_args()

    ch_map = _parse_channels(args.channels)
    tail_override = open(args.tail).read() if args.tail else None
    null_offsets = None
    if args.null_offsets:
        null_offsets = [int(x) for x in args.null_offsets.split(",") if x.strip()]

    if args.patient and args.edf and args.onset is not None and args.out:
        generate(args.patient, args.edf, args.onset, args.out,
                 ch_map=ch_map,
                 null_pool_size=args.null_pool_size,
                 null_seed=args.null_seed,
                 null_offsets=null_offsets,
                 tail_override=tail_override)
        return

    targets = args.patients if args.patients else list(COHORT.keys())
    for p in targets:
        if p not in COHORT:
            raise SystemExit(f"unknown patient: {p}  (known: {list(COHORT)})")
        cfg = COHORT[p]
        generate(p, cfg["edf"], cfg["onset"],
                 f"examples/door_f_assoc_preictal_{p}.sio",
                 ch_map=ch_map,
                 null_pool_size=args.null_pool_size,
                 null_seed=args.null_seed,
                 null_offsets=null_offsets,
                 tail_override=tail_override)


if __name__ == "__main__":
    main()
