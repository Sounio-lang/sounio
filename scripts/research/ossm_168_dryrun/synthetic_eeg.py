"""Synthetic 7-region EEG for the dry-run.

Generates 7-channel (post-regional-aggregation) bandlimited signals at
250 Hz to stand in for LEMON/MODMA preprocessed data during development.
This module DOES NOT replace real data; it only validates the
implementation pipeline.

Each "subject" is parameterised by:
- a random seed (deterministic per subject_index)
- a latent "rumination index" in [0, 1] that modulates alpha-band (8-12 Hz)
  mass asymmetrically across regions
- a latent "anhedonia index" in [0, 1] that modulates theta-band (4-7 Hz)
  posterior activity
- a latent "valence index" in [-1, 1] that modulates fronto-parietal
  power asymmetry

These latent indices are what F_1, F_2, F_3 should correlate with when
the pipeline is working correctly. They are exposed to the dry-run
harness for the purpose of sanity-checking the direction of the
Spearman correlations on synthetic data.

No real data is touched by this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


SAMPLE_RATE_HZ = 250
EPOCH_DURATION_S = 4.0
SAMPLES_PER_EPOCH = int(SAMPLE_RATE_HZ * EPOCH_DURATION_S)


@dataclass(frozen=True)
class SyntheticSubjectLatents:
    rumination: float
    anhedonia: float
    valence: float

    def __post_init__(self) -> None:
        assert 0.0 <= self.rumination <= 1.0
        assert 0.0 <= self.anhedonia <= 1.0
        assert -1.0 <= self.valence <= 1.0


def generate_latents(n_subjects: int, seed: int = 20260421) -> list[SyntheticSubjectLatents]:
    rng = np.random.default_rng(seed)
    out: list[SyntheticSubjectLatents] = []
    for _ in range(n_subjects):
        out.append(SyntheticSubjectLatents(
            rumination=float(rng.uniform(0.0, 1.0)),
            anhedonia=float(rng.uniform(0.0, 1.0)),
            valence=float(rng.uniform(-1.0, 1.0)),
        ))
    return out


def _bandlimited_noise(
    n_samples: int,
    low_hz: float,
    high_hz: float,
    power: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate Gaussian noise bandpassed to [low_hz, high_hz] with given power."""
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / SAMPLE_RATE_HZ)
    spec = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spec = spec * mask
    sig = np.fft.irfft(spec, n=n_samples).real
    current_power = float(np.mean(sig ** 2)) + 1e-12
    return sig * float(np.sqrt(power / current_power))


def _pink_noise(n_samples: int, power: float, rng: np.random.Generator) -> NDArray[np.float64]:
    """Approximate 1/f pink noise with given target power."""
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / SAMPLE_RATE_HZ)
    amp = 1.0 / np.sqrt(np.maximum(freqs, 1.0))  # 1/f power in magnitude-squared
    phase = rng.uniform(0.0, 2.0 * np.pi, size=len(freqs))
    spec = amp * np.exp(1j * phase)
    spec[0] = 0.0  # no DC
    sig = np.fft.irfft(spec, n=n_samples).real
    current_power = float(np.mean(sig ** 2)) + 1e-12
    return sig * float(np.sqrt(power / current_power))


def generate_subject(
    subject_index: int,
    latents: SyntheticSubjectLatents,
    n_epochs: int = 30,
    seed_offset: int = 20260421,
) -> NDArray[np.float64]:
    """Return a (n_epochs, 7, SAMPLES_PER_EPOCH) array of synthetic EEG.

    Channels 0..6 correspond to the 7 canonical regions pinned in §5
    of the pre-registration (left frontal, right frontal, left temporal,
    right temporal, central, left parieto-occipital, right parieto-occipital).
    """
    rng = np.random.default_rng(seed_offset + subject_index)
    n_total = n_epochs * SAMPLES_PER_EPOCH
    out = np.empty((7, n_total), dtype=np.float64)

    # Region-specific base composition: pink noise (1.0 power) + a
    # physiology-motivated band, plus a latent-modulated modulation.
    for ch in range(7):
        pink = _pink_noise(n_total, power=1.0, rng=rng)

        # Alpha (8-12 Hz): modulated by rumination, stronger in frontal (ch 0, 1)
        alpha_power = 0.3 + 0.9 * latents.rumination * (1.5 if ch in (0, 1) else 0.5)
        alpha = _bandlimited_noise(n_total, 8.0, 12.0, alpha_power, rng)

        # Theta (4-7 Hz): modulated by anhedonia, stronger in posterior (ch 5, 6)
        theta_power = 0.2 + 0.8 * latents.anhedonia * (1.5 if ch in (5, 6) else 0.6)
        theta = _bandlimited_noise(n_total, 4.0, 7.0, theta_power, rng)

        # Valence asymmetry: left-right frontal differential, beta band
        valence_direction = 1.0 if ch in (0, 2, 5) else -1.0 if ch in (1, 3, 6) else 0.0
        beta_power = 0.2 + 0.6 * abs(latents.valence) * (1.0 + 0.5 * valence_direction * latents.valence)
        beta = _bandlimited_noise(n_total, 15.0, 30.0, beta_power, rng)

        out[ch] = pink + alpha + theta + beta

    # Per-channel z-score (per §5 step 8 of the prereg)
    mean = out.mean(axis=1, keepdims=True)
    std = out.std(axis=1, keepdims=True) + 1e-12
    out = (out - mean) / std

    # Reshape into (n_epochs, 7, SAMPLES_PER_EPOCH)
    reshaped = out.reshape(7, n_epochs, SAMPLES_PER_EPOCH).transpose(1, 0, 2)
    assert reshaped.shape == (n_epochs, 7, SAMPLES_PER_EPOCH)
    return reshaped
