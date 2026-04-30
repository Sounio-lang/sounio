"""Smoke tests for the end-to-end dry-run.

Exercises the full pipeline synthetic-EEG -> O-SSM forward pass
-> F_1 / F_2 / F_3 -> §7.1 invariance.

Kept small (short epochs, few subjects, few labellings) so the
test suite runs in well under a minute.
"""

from __future__ import annotations

import numpy as np

from scripts.research.ossm_168_dryrun import (
    dryrun,
    features,
    ossm,
    synthetic_eeg as syn,
)


def test_synthetic_subject_shape():
    latents_list = syn.generate_latents(1, seed=20260421)
    eeg = syn.generate_subject(0, latents_list[0], n_epochs=2)
    assert eeg.shape == (2, 7, syn.SAMPLES_PER_EPOCH)
    # z-scored per channel over the whole signal, so std is finite.
    assert np.isfinite(eeg).all()


def test_ossm_forward_pass_shape():
    latents_list = syn.generate_latents(1, seed=20260421)
    eeg = syn.generate_subject(0, latents_list[0], n_epochs=1)
    # To keep this test fast use only 100 samples rather than a full 1000-sample epoch.
    epoch = eeg[0][:, :100]
    traj = ossm.forward_pass(epoch, subject_index=0)
    assert traj.shape == (100, ossm.STATE_DIM_OCTONIONS, 8)
    assert np.isfinite(traj).all()


def test_features_are_finite_and_nonzero_on_synthetic():
    latents_list = syn.generate_latents(1, seed=20260421)
    eeg = syn.generate_subject(0, latents_list[0], n_epochs=1)
    # Short forward pass.
    epoch = eeg[0][:, :100]
    traj = ossm.forward_pass(epoch, subject_index=0)
    feats = features.compute_all_features(traj)
    for name, v in feats.items():
        assert np.isfinite(v), f"{name} not finite: {v}"
        assert v >= 0.0, f"{name} is negative: {v}"
    # F_1 and F_3 should be strictly positive on generic states; F_2 is a
    # proximity so can in principle be small but not negative.
    assert feats["F1"] > 0.0, "F_1 collapsed to zero on synthetic data"
    assert feats["F3"] > 0.0, (
        "F_3 collapsed to zero -- v3 L1 mass is supposed to be non-zero"
    )


def test_permuted_forward_pass_preserves_shape():
    latents_list = syn.generate_latents(1, seed=20260421)
    eeg = syn.generate_subject(0, latents_list[0], n_epochs=1)
    epoch = eeg[0][:, :100]
    sigma = (7, 6, 5, 4, 3, 2, 1)  # reverse permutation in S_7
    traj_perm = ossm.permuted_forward_pass(epoch, subject_index=0, sigma_perm=sigma)
    traj_canon = ossm.forward_pass(epoch, subject_index=0)
    assert traj_perm.shape == traj_canon.shape
    # And they are NOT identical (the permutation feeds different channels
    # into different B-matrix rows, so the dynamics diverge).
    assert not np.allclose(traj_perm, traj_canon, atol=1e-10)


def test_invariance_smoke_runs_and_reports_finite_iqr():
    # Small smoke: 2 epochs, 3 labellings.
    # We override the n_epochs via monkey-less patching: use the
    # direct invariance_smoke function with tiny sizes. To keep it
    # really fast we cut each epoch in the synthetic EEG by only
    # asking run_subject for 1 epoch, but run_subject internally
    # uses SAMPLES_PER_EPOCH -- that is 1000 steps, which is the
    # biggest cost. For CI-speed we accept a single epoch here.
    smoke = dryrun.invariance_smoke(subject_index=0, n_epochs=1, n_labellings=3)
    for key in ("F1", "F2", "F3"):
        assert np.isfinite(smoke[key]["canonical"]), f"{key} canonical not finite"
        assert np.isfinite(smoke[key]["iqr"]), f"{key} IQR not finite"
        assert np.isfinite(smoke[key]["mean_delta"]), f"{key} mean_delta not finite"
        assert smoke[key]["iqr"] >= 0.0
    assert smoke["n_labellings_tested"] == 3
