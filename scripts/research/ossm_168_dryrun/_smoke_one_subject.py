"""Quick smoke-test: run the §5 preprocessing pipeline on a single LEMON
subject and print summary statistics. Used to verify the BrainVision
loader, ICA path, and regional aggregation end-to-end.

    python3 -u scripts/research/ossm_168_dryrun/_smoke_one_subject.py sub-010002
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.research.ossm_168_dryrun import lemon_preprocess as lp  # noqa: E402

DATA_ROOT = "/workspace/data/lemon/raw_eeg"


def main(subject: str) -> None:
    t0 = time.time()
    print(f"[{time.time()-t0:6.1f}s] loading {subject} ...", flush=True)
    raw = lp.load_lemon_raw(DATA_ROOT, subject)
    print(
        f"[{time.time()-t0:6.1f}s]   sfreq={raw.info['sfreq']} "
        f"n_ch={len(raw.ch_names)} dur_s={raw.times[-1]:.1f}",
        flush=True,
    )

    print(f"[{time.time()-t0:6.1f}s] CAR + bandpass + notch + resample ...",
          flush=True)
    lp.preprocess_raw(raw)
    print(
        f"[{time.time()-t0:6.1f}s]   new sfreq={raw.info['sfreq']} "
        f"n_samples={raw.n_times}",
        flush=True,
    )

    print(f"[{time.time()-t0:6.1f}s] Infomax ICA + ICLabel ...", flush=True)
    raw, n_rej, n_tot = lp.run_ica_iclabel(raw)
    print(f"[{time.time()-t0:6.1f}s]   rejected {n_rej}/{n_tot} components",
          flush=True)

    # Mimic tail of preprocess_subject: pick EEG, aggregate, reject, epoch.
    eeg = raw.copy().pick("eeg", verbose=False)
    data_uv = eeg.get_data(units="uV")
    ch = list(eeg.info["ch_names"])
    agg = lp.regional_aggregate(data_uv, ch, strict=False)
    print(f"[{time.time()-t0:6.1f}s] aggregated shape={agg.shape}", flush=True)

    spe = lp.SAMPLES_PER_EPOCH
    n_total = agg.shape[1] // spe
    ep_uv = agg[:, : n_total * spe].reshape(7, n_total, spe).transpose(1, 0, 2)
    keep = lp.reject_epochs_by_amplitude_gradient(ep_uv, lp.TARGET_SFREQ_HZ)
    print(
        f"[{time.time()-t0:6.1f}s] amplitude/gradient reject: "
        f"{int(keep.sum())}/{len(keep)} kept",
        flush=True,
    )
    kept = ep_uv[keep]
    # Z-score per epoch per channel.
    mean = kept.mean(axis=2, keepdims=True)
    std = kept.std(axis=2, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    zepochs = ((kept - mean) / std).astype(np.float32)
    print(
        f"[{time.time()-t0:6.1f}s] final z-scored epoch tensor: "
        f"shape={zepochs.shape} dtype={zepochs.dtype}",
        flush=True,
    )
    print(
        f"[{time.time()-t0:6.1f}s] stats: mean={zepochs.mean():.3g} "
        f"std={zepochs.std():.3g} finite={np.isfinite(zepochs).all()}",
        flush=True,
    )


if __name__ == "__main__":
    sub = sys.argv[1] if len(sys.argv) > 1 else "sub-010002"
    main(sub)
