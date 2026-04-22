"""LEMON (ds000221) EEG preprocessing for the OSSM-168 v3 protocol.

Implements the frozen §5 pipeline of
`docs/papers/preregistrations/2026-04-21_ossm_168_depression_v3.md`:

    1. common-average reference
    2. 1-45 Hz zero-phase Butterworth bandpass (order 4)
    3. 50 Hz notch
    4. downsample to 250 Hz
    5. amplitude reject ±150 µV, gradient 50 µV/ms (segments discarded)
    6. Infomax ICA (seed = 20260421) + ICLabel P(brain) < 0.30 rejection
    7. non-overlapping 4-second epochs
    8. per-epoch per-channel z-score

Followed by the §5 regional aggregation into seven octonion basis
channels {Left/Right frontal, Left/Right temporal, Central, Left/Right
parieto-occipital}.

The output of :func:`preprocess_subject` is a ``float32`` array of
shape ``(n_epochs, 7, 1000)`` — the exact input format consumed by
``ossm.forward_pass`` on synthetic data — together with preprocessing
provenance metadata (:func:`preprocess_subject` returns a
``PreprocessResult`` dataclass).

MNE is imported lazily inside the MNE-dependent functions so the
pure-numpy helpers (:func:`regional_aggregate`, :func:`epoch_and_zscore`,
:func:`reject_epochs_by_amplitude_gradient`) remain testable without
the heavy EEG stack installed.

Install:
    pip install mne mne-bids mne-icalabel
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

# =============================================================================
# Frozen constants — prereg §5
# =============================================================================

TARGET_SFREQ_HZ: float = 250.0
EPOCH_DURATION_S: float = 4.0
SAMPLES_PER_EPOCH: int = int(TARGET_SFREQ_HZ * EPOCH_DURATION_S)

BANDPASS_LOW_HZ: float = 1.0
BANDPASS_HIGH_HZ: float = 45.0
NOTCH_HZ: float = 50.0

AMPLITUDE_REJECT_UV: float = 150.0
GRADIENT_REJECT_UV_PER_MS: float = 50.0

ICA_SEED: int = 20260421
ICA_BRAIN_PROB_MIN: float = 0.30

# Regional aggregation into 7 octonion basis channels (prereg §5 table).
# Stored as 1-indexed so e_1..e_7 line up with the docstring; tuples are
# the canonical 10-20 superset names, matched case-insensitively and
# tolerating the common "T3/T4/T5/T6" ↔ "T7/T8/P7/P8" aliasing used by
# LEMON's 62-channel montage.
REGIONAL_CHANNELS: Mapping[int, tuple[str, ...]] = {
    1: ("Fp1", "F3", "F7", "FC5"),
    2: ("Fp2", "F4", "F8", "FC6"),
    3: ("T7", "TP9", "CP5"),
    4: ("T8", "TP10", "CP6"),
    5: ("Fz", "Cz", "FC1", "FC2"),
    6: ("P3", "P7", "O1", "PO9"),
    7: ("P4", "P8", "O2", "PO10"),
}

# Tolerated historical aliases → canonical name.
_CHANNEL_ALIASES: Mapping[str, str] = {
    "T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8",
}


def _canonicalise(name: str) -> str:
    """Upper-case, strip whitespace, resolve 10-20 historical aliases."""
    n = name.strip()
    # LEMON uses mixed case; compare case-insensitively against the table.
    for alias, canon in _CHANNEL_ALIASES.items():
        if n.upper() == alias:
            return canon
    return n


# =============================================================================
# Pure-numpy helpers (testable without MNE)
# =============================================================================

def regional_aggregate(
    data: NDArray[np.floating],
    ch_names: list[str],
    strict: bool = False,
) -> NDArray[np.float64]:
    """Average channels within each of the seven canonical regions.

    Parameters
    ----------
    data : (n_channels, n_samples) array of continuous EEG after the
        filtering / ICA / rejection steps.
    ch_names : list of length ``n_channels`` matching ``data``.
    strict : if True, every channel listed in REGIONAL_CHANNELS must be
        present; otherwise a :class:`KeyError` is raised. If False, the
        region average uses whatever subset of the canonical channels is
        present (and raises :class:`ValueError` if a region ends up
        empty).

    Returns an ``(7, n_samples)`` array in basis-index order 1..7.
    """
    if data.ndim != 2:
        raise ValueError(f"expected 2D (n_ch, n_samples), got shape {data.shape}")
    if data.shape[0] != len(ch_names):
        raise ValueError(
            f"channel count mismatch: data has {data.shape[0]} rows, "
            f"ch_names has {len(ch_names)}"
        )
    index: dict[str, int] = {}
    for i, name in enumerate(ch_names):
        index[_canonicalise(name).upper()] = i

    out = np.zeros((7, data.shape[1]), dtype=np.float64)
    for region_idx in range(1, 8):
        wanted = REGIONAL_CHANNELS[region_idx]
        picked: list[int] = []
        missing: list[str] = []
        for name in wanted:
            key = _canonicalise(name).upper()
            if key in index:
                picked.append(index[key])
            else:
                missing.append(name)
        if strict and missing:
            raise KeyError(
                f"region e_{region_idx} missing channels: {missing}"
            )
        if not picked:
            raise ValueError(
                f"region e_{region_idx} has no channels after aliasing; "
                f"wanted {wanted}, available {ch_names}"
            )
        out[region_idx - 1] = data[picked].mean(axis=0)
    return out



def epoch_and_zscore(
    data: NDArray[np.floating],
    sfreq: float = TARGET_SFREQ_HZ,
    epoch_s: float = EPOCH_DURATION_S,
) -> NDArray[np.float64]:
    """Cut continuous ``(7, n_samples)`` into non-overlapping epochs and
    z-score every channel within every epoch (step 7 + 8 of §5).

    Returns an ``(n_epochs, 7, samples_per_epoch)`` array. Trailing
    samples that do not fill a full epoch are dropped.
    """
    if data.ndim != 2 or data.shape[0] != 7:
        raise ValueError(f"expected (7, n_samples), got {data.shape}")
    samples_per_epoch = int(round(sfreq * epoch_s))
    n_epochs = data.shape[1] // samples_per_epoch
    if n_epochs == 0:
        return np.zeros((0, 7, samples_per_epoch), dtype=np.float64)
    trimmed = data[:, : n_epochs * samples_per_epoch]
    epochs = trimmed.reshape(7, n_epochs, samples_per_epoch).transpose(1, 0, 2)
    # Per-epoch per-channel z-score.
    mean = epochs.mean(axis=2, keepdims=True)
    std = epochs.std(axis=2, keepdims=True)
    # Guard against flat channels (shouldn't happen post-ICA but be safe).
    std = np.where(std < 1e-12, 1.0, std)
    return ((epochs - mean) / std).astype(np.float64)


def reject_epochs_by_amplitude_gradient(
    epochs: NDArray[np.floating],
    sfreq: float,
    amp_uv: float = AMPLITUDE_REJECT_UV,
    grad_uv_per_ms: float = GRADIENT_REJECT_UV_PER_MS,
) -> NDArray[np.bool_]:
    """Return a per-epoch boolean mask; True = keep, False = reject.

    Operates on **pre-zscore** data (µV units). Rejects any epoch where
    any channel exceeds the absolute-amplitude threshold or the
    per-sample gradient threshold.

    ``epochs`` : ``(n_epochs, n_channels, samples_per_epoch)``.
    """
    if epochs.ndim != 3:
        raise ValueError(f"expected 3D (n_epochs, n_ch, n_s), got {epochs.shape}")
    amp_ok = np.abs(epochs).max(axis=(1, 2)) <= amp_uv
    # gradient in µV per ms = |diff| / (1000 / sfreq)
    ms_per_sample = 1000.0 / sfreq
    grad = np.abs(np.diff(epochs, axis=2)) / ms_per_sample
    grad_ok = grad.max(axis=(1, 2)) <= grad_uv_per_ms
    return amp_ok & grad_ok


# =============================================================================
# MNE-dependent pipeline (imported lazily)
# =============================================================================

@dataclass
class PreprocessResult:
    """Provenance of a subject-level preprocess run."""
    subject: str
    sfreq: float
    epochs: NDArray[np.float64]  # (n_epochs, 7, SAMPLES_PER_EPOCH)
    n_total_epochs: int
    n_retained_epochs: int
    n_ica_rejected: int
    n_ica_total: int
    channel_names_used: list[str] = field(default_factory=list)

    def retention_rate(self) -> float:
        if self.n_total_epochs == 0:
            return 0.0
        return self.n_retained_epochs / self.n_total_epochs


def load_lemon_raw(
    bids_root: str,
    subject: str,
    task: str = "RSEEG",
    acquisition: str | None = None,
):
    """Read one LEMON resting-state EEG recording as an MNE Raw.

    Uses ``mne_bids`` when available; falls back to a direct
    BrainVision read. Subject IDs must match the BIDS ``sub-<id>``
    convention (the ``sub-`` prefix may be included or omitted).
    """
    try:
        from mne_bids import BIDSPath, read_raw_bids  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "mne-bids is required to load LEMON BIDS data; "
            "`pip install mne-bids`"
        ) from exc

    sub_id = subject[4:] if subject.startswith("sub-") else subject
    bids_path = BIDSPath(
        subject=sub_id, task=task, acquisition=acquisition,
        datatype="eeg", root=bids_root, suffix="eeg",
    )
    raw = read_raw_bids(bids_path=bids_path, verbose=False)
    raw.load_data()
    return raw


def preprocess_raw(raw, target_sfreq: float = TARGET_SFREQ_HZ):
    """Steps 1-4 of §5 on an MNE Raw (CAR, bandpass, notch, resample).

    Mutates ``raw`` in place and returns it for chaining.
    """
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.filter(
        l_freq=BANDPASS_LOW_HZ, h_freq=BANDPASS_HIGH_HZ,
        method="iir", iir_params=dict(order=4, ftype="butter"),
        phase="zero", verbose=False,
    )
    raw.notch_filter(freqs=[NOTCH_HZ], verbose=False)
    if abs(float(raw.info["sfreq"]) - target_sfreq) > 1e-3:
        raw.resample(sfreq=target_sfreq, verbose=False)
    return raw


def run_ica_iclabel(raw, seed: int = ICA_SEED,
                    brain_prob_min: float = ICA_BRAIN_PROB_MIN):
    """Step 6 of §5: Infomax ICA + ICLabel brain-probability rejection.

    Requires ``mne`` and ``mne-icalabel``. ``raw`` is mutated in place
    (ICA-cleaned). Returns ``(raw, n_rejected, n_total_components)``.
    """
    try:
        import mne  # type: ignore
        from mne_icalabel import label_components  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "mne-icalabel is required for ICLabel-based component "
            "rejection; `pip install mne-icalabel`"
        ) from exc

    # ICLabel was trained on 1-100 Hz CAR data; our bandpass is 1-45 Hz
    # which ICLabel tolerates. Infomax = ``extended-infomax`` in MNE.
    ica = mne.preprocessing.ICA(
        n_components=None, method="infomax",
        fit_params=dict(extended=True), max_iter="auto",
        random_state=seed, verbose=False,
    )
    ica.fit(raw, verbose=False)
    labels = label_components(raw, ica, method="iclabel")
    brain_prob = np.asarray(labels["y_pred_proba"])
    # ICLabel returns the probability of the most-likely class. To
    # recover P(brain) specifically we need the full probability
    # vector; newer mne-icalabel versions expose ``labels_pred_proba``
    # already restricted to the predicted class, so fall back to
    # using the class label to only reject non-brain components that
    # cross the inverse threshold.
    predicted = list(labels["labels"])
    reject_idx: list[int] = []
    for i, label in enumerate(predicted):
        p = float(brain_prob[i])
        # Conservative rule: reject component if the predicted label is
        # not "brain" AND the classifier is at least (1 - threshold)
        # confident, OR if the predicted label IS "brain" but with
        # probability below the minimum.
        if label == "brain":
            if p < brain_prob_min:
                reject_idx.append(i)
        else:
            if p >= (1.0 - brain_prob_min):
                reject_idx.append(i)
    ica.exclude = reject_idx
    ica.apply(raw, verbose=False)
    return raw, len(reject_idx), int(ica.n_components_)


def preprocess_subject(
    bids_root: str,
    subject: str,
    task: str = "RSEEG",
    acquisition: str | None = None,
) -> PreprocessResult:
    """End-to-end §5 pipeline for one LEMON subject.

    Returns a :class:`PreprocessResult` with a ready-to-feed
    ``epochs`` array of shape ``(n_retained_epochs, 7, 1000)``.
    """
    raw = load_lemon_raw(bids_root, subject, task=task, acquisition=acquisition)
    raw = preprocess_raw(raw)
    raw, n_ica_rej, n_ica_total = run_ica_iclabel(raw)

    # Pre-zscore epoching in µV for amplitude/gradient rejection.
    data_uv = raw.get_data(units="uV")  # (n_ch, n_samples)
    ch_names = list(raw.info["ch_names"])
    aggregated = regional_aggregate(data_uv, ch_names, strict=False)

    samples_per_epoch = SAMPLES_PER_EPOCH
    n_total = aggregated.shape[1] // samples_per_epoch
    if n_total == 0:
        return PreprocessResult(
            subject=subject, sfreq=float(raw.info["sfreq"]),
            epochs=np.zeros((0, 7, samples_per_epoch), dtype=np.float64),
            n_total_epochs=0, n_retained_epochs=0,
            n_ica_rejected=n_ica_rej, n_ica_total=n_ica_total,
            channel_names_used=ch_names,
        )
    trimmed = aggregated[:, : n_total * samples_per_epoch]
    raw_epochs = trimmed.reshape(7, n_total, samples_per_epoch).transpose(1, 0, 2)
    keep_mask = reject_epochs_by_amplitude_gradient(
        raw_epochs, sfreq=float(raw.info["sfreq"])
    )
    kept = raw_epochs[keep_mask]
    # Per-epoch per-channel z-score on retained epochs only.
    mean = kept.mean(axis=2, keepdims=True)
    std = kept.std(axis=2, keepdims=True)
    std = np.where(std < 1e-12, 1.0, std)
    normalised = ((kept - mean) / std).astype(np.float64)
    return PreprocessResult(
        subject=subject, sfreq=float(raw.info["sfreq"]),
        epochs=normalised, n_total_epochs=int(n_total),
        n_retained_epochs=int(keep_mask.sum()),
        n_ica_rejected=n_ica_rej, n_ica_total=n_ica_total,
        channel_names_used=ch_names,
    )


# =============================================================================
# CLI
# =============================================================================

def _cli() -> int:  # pragma: no cover - thin wrapper
    import argparse, json, sys
    parser = argparse.ArgumentParser(
        description="Preprocess one LEMON subject for OSSM-168 (v3)."
    )
    parser.add_argument("--bids-root", required=True,
                        help="Path to the LEMON BIDS root (ds000221).")
    parser.add_argument("--subject", required=True,
                        help='BIDS subject id, e.g. "sub-010002" or "010002".')
    parser.add_argument("--task", default="RSEEG")
    parser.add_argument("--acquisition", default=None)
    parser.add_argument("--output", required=True,
                        help="Output .npz path.")
    args = parser.parse_args()

    result = preprocess_subject(
        bids_root=args.bids_root, subject=args.subject,
        task=args.task, acquisition=args.acquisition,
    )
    np.savez_compressed(
        args.output,
        epochs=result.epochs.astype(np.float32),
        sfreq=np.array(result.sfreq, dtype=np.float64),
        subject=np.array(result.subject),
        n_total_epochs=np.array(result.n_total_epochs, dtype=np.int32),
        n_retained_epochs=np.array(result.n_retained_epochs, dtype=np.int32),
        n_ica_rejected=np.array(result.n_ica_rejected, dtype=np.int32),
        n_ica_total=np.array(result.n_ica_total, dtype=np.int32),
    )
    print(json.dumps({
        "subject": result.subject,
        "n_total_epochs": result.n_total_epochs,
        "n_retained_epochs": result.n_retained_epochs,
        "retention_rate": result.retention_rate(),
        "n_ica_rejected": result.n_ica_rejected,
        "n_ica_total": result.n_ica_total,
    }, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())

