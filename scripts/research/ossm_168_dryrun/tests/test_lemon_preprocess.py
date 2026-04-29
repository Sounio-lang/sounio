"""Tests for the LEMON preprocessing pipeline (prereg §5).

Exercises the pure-numpy helpers that do not require MNE:
    * regional_aggregate (channel → 7 basis channels, with aliasing)
    * epoch_and_zscore (epoching + per-epoch per-channel z-score)
    * reject_epochs_by_amplitude_gradient (amplitude + gradient thresholds)

The MNE-dependent functions (load_lemon_raw / preprocess_raw /
run_ica_iclabel / preprocess_subject) are covered by the smoke runner
against real LEMON data and are out of scope for the pytest layer.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.ossm_168_dryrun import lemon_preprocess as lp


# ---------------------------------------------------------------------------
# regional_aggregate
# ---------------------------------------------------------------------------

CANONICAL_62 = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "FC5", "FC6",
    "T7", "T8", "TP9", "TP10", "CP5", "CP6",
    "Fz", "Cz", "FC1", "FC2",
    "P3", "P7", "O1", "PO9", "P4", "P8", "O2", "PO10",
]


def test_regional_aggregate_shape_and_values():
    rng = np.random.RandomState(0)
    data = rng.randn(len(CANONICAL_62), 1000)
    out = lp.regional_aggregate(data, CANONICAL_62)
    assert out.shape == (7, 1000)
    # region e_5 = mean of Fz, Cz, FC1, FC2.
    idx = [CANONICAL_62.index(n) for n in ("Fz", "Cz", "FC1", "FC2")]
    expected = data[idx].mean(axis=0)
    np.testing.assert_allclose(out[4], expected, rtol=0, atol=1e-12)


def test_regional_aggregate_resolves_historical_aliases():
    # LEMON-era BrainVision montages sometimes ship T3/T4/T5/T6.
    aliased = [
        "T3", "T4", "T5", "T6",                                  # aliases
        "Fp1", "F3", "F7", "FC5", "Fp2", "F4", "F8", "FC6",
        "TP9", "TP10", "CP5", "CP6", "Fz", "Cz", "FC1", "FC2",
        "P3", "O1", "PO9", "P4", "O2", "PO10",
    ]
    data = np.random.RandomState(1).randn(len(aliased), 500)
    out = lp.regional_aggregate(data, aliased, strict=False)
    # e_3 = Left temporal should now include T3 (→ T7) in its mean.
    idx_t7 = aliased.index("T3")
    idx_tp9 = aliased.index("TP9")
    idx_cp5 = aliased.index("CP5")
    expected = data[[idx_t7, idx_tp9, idx_cp5]].mean(axis=0)
    np.testing.assert_allclose(out[2], expected, rtol=0, atol=1e-12)


def test_regional_aggregate_strict_raises_on_missing():
    data = np.random.RandomState(2).randn(3, 200)
    with pytest.raises(KeyError):
        lp.regional_aggregate(data, ["Fp1", "F3", "F7"], strict=True)


def test_regional_aggregate_nonstrict_uses_subset():
    # Drop one channel from region e_5 (say FC1) but keep Fz, Cz, FC2.
    reduced = [n for n in CANONICAL_62 if n != "FC1"]
    data = np.random.RandomState(3).randn(len(reduced), 200)
    out = lp.regional_aggregate(data, reduced, strict=False)
    idx = [reduced.index(n) for n in ("Fz", "Cz", "FC2")]
    expected = data[idx].mean(axis=0)
    np.testing.assert_allclose(out[4], expected, rtol=0, atol=1e-12)


def test_regional_aggregate_raises_when_region_has_no_channels():
    data = np.random.RandomState(4).randn(3, 200)
    # "Fp1", "F3", "F7" populate e_1 only; e_2..e_7 should have no
    # channels and raise ValueError.
    with pytest.raises(ValueError):
        lp.regional_aggregate(data, ["Fp1", "F3", "F7"], strict=False)


# ---------------------------------------------------------------------------
# epoch_and_zscore
# ---------------------------------------------------------------------------

def test_epoch_and_zscore_drops_trailing_partial_epoch():
    # 3500 samples at 250 Hz = 14 s = 3 full 4-s epochs + 500-sample tail.
    cont = np.random.RandomState(5).randn(7, 3500) * 10.0
    ep = lp.epoch_and_zscore(cont, sfreq=250.0, epoch_s=4.0)
    assert ep.shape == (3, 7, 1000)


def test_epoch_and_zscore_normalises_per_channel_per_epoch():
    cont = np.random.RandomState(6).randn(7, 4000) * 50.0 + 10.0
    ep = lp.epoch_and_zscore(cont, sfreq=250.0, epoch_s=4.0)
    np.testing.assert_allclose(ep.mean(axis=2), 0.0, atol=1e-12)
    np.testing.assert_allclose(ep.std(axis=2), 1.0, atol=1e-10)


def test_epoch_and_zscore_handles_empty_input():
    cont = np.zeros((7, 100), dtype=np.float64)
    ep = lp.epoch_and_zscore(cont, sfreq=250.0, epoch_s=4.0)
    assert ep.shape == (0, 7, 1000)


def test_epoch_and_zscore_rejects_wrong_shape():
    with pytest.raises(ValueError):
        lp.epoch_and_zscore(np.zeros((6, 1000)), sfreq=250.0)


# ---------------------------------------------------------------------------
# reject_epochs_by_amplitude_gradient
# ---------------------------------------------------------------------------

def test_reject_by_amplitude():
    rng = np.random.RandomState(7)
    ep = rng.randn(4, 7, 1000) * 30.0   # well below 150 µV
    ep[1] = ep[1] * 10.0                # this one crosses 150 µV
    mask = lp.reject_epochs_by_amplitude_gradient(ep, sfreq=250.0)
    assert bool(mask[0]) is True
    assert bool(mask[1]) is False
    assert bool(mask[2]) is True
    assert bool(mask[3]) is True


def test_reject_by_gradient():
    ep = np.zeros((2, 7, 1000), dtype=np.float64)
    # Inject a 100 µV jump between adjacent samples on channel 0, epoch 0.
    # At 250 Hz, 1 sample = 4 ms, so gradient = 25 µV/ms < threshold.
    ep[0, 0, 500] = 100.0
    # On epoch 1, inject a 300 µV jump → 75 µV/ms > 50 threshold.
    ep[1, 0, 500] = 300.0
    mask = lp.reject_epochs_by_amplitude_gradient(ep, sfreq=250.0)
    assert bool(mask[0]) is True
    assert bool(mask[1]) is False


def test_samples_per_epoch_constant_matches_prereg():
    # §5: 4 s * 250 Hz = 1000 samples per epoch.
    assert lp.SAMPLES_PER_EPOCH == 1000
    assert lp.TARGET_SFREQ_HZ == 250.0
    assert lp.EPOCH_DURATION_S == 4.0
