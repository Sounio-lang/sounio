"""End-to-end dry-run of the v3 protocol on synthetic data.

This script does NOT access LEMON or MODMA. Its purpose is to exercise
the full pipeline end-to-end on synthetic 7-channel EEG:

    synthetic EEG  ->  (optional channel permutation)
    ->  untrained O-SSM forward pass  ->  F_1 / F_2 / F_3
    ->  subject-level medians  ->  Spearman against latent targets
    ->  §7.1 invariance across 30 Fano coset representatives.

Usage:
    python3 -m scripts.research.ossm_168_dryrun.dryrun

Intended as an implementation smoke test prior to the confirmatory
analysis on real data.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass

import numpy as np

from . import fano_orbits
from . import features
from . import ossm
from . import synthetic_eeg as syn


@dataclass
class SubjectResult:
    subject_index: int
    latents: syn.SyntheticSubjectLatents
    F1_median: float
    F2_median: float
    F3_median: float


def run_subject(subject_index: int, n_epochs: int,
                 sigma_perm: tuple[int, ...] | None = None) -> SubjectResult:
    """Run one subject through the full pipeline and return subject-level features."""
    latents_list = syn.generate_latents(subject_index + 1, seed=20260421)
    latents = latents_list[subject_index]
    eeg = syn.generate_subject(subject_index, latents, n_epochs=n_epochs)

    per_epoch_f1: list[float] = []
    per_epoch_f2: list[float] = []
    per_epoch_f3: list[float] = []

    for e in range(n_epochs):
        epoch_signal = eeg[e]  # shape (7, SAMPLES_PER_EPOCH)
        if sigma_perm is None:
            traj = ossm.forward_pass(epoch_signal, subject_index)
        else:
            traj = ossm.permuted_forward_pass(epoch_signal, subject_index, sigma_perm)
        feats = features.compute_all_features(traj)
        per_epoch_f1.append(feats["F1"])
        per_epoch_f2.append(feats["F2"])
        per_epoch_f3.append(feats["F3"])

    return SubjectResult(
        subject_index=subject_index,
        latents=latents,
        F1_median=statistics.median(per_epoch_f1),
        F2_median=statistics.median(per_epoch_f2),
        F3_median=statistics.median(per_epoch_f3),
    )


def _spearman(x: list[float], y: list[float]) -> float:
    """Basic Spearman rank correlation (scipy.stats.spearmanr equivalent)."""
    assert len(x) == len(y) and len(x) >= 2
    rx = np.argsort(np.argsort(np.asarray(x)))
    ry = np.argsort(np.argsort(np.asarray(y)))
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denom = float(np.sqrt((dx ** 2).sum() * (dy ** 2).sum()))
    if denom == 0.0:
        return 0.0
    return float((dx * dy).sum() / denom)


def invariance_smoke(subject_index: int, n_epochs: int = 2,
                      n_labellings: int = 30) -> dict[str, object]:
    """Run §7.1 smoke test: features under 30 Fano coset-representative labellings."""
    reps = fano_orbits.coset_representatives_S7_mod_Aut()
    assert len(reps) == 30
    if n_labellings > 30:
        raise ValueError("n_labellings cannot exceed 30")
    used = reps[:n_labellings]

    canonical = run_subject(subject_index, n_epochs, sigma_perm=None)

    f1s = [canonical.F1_median]
    f2s = [canonical.F2_median]
    f3s = [canonical.F3_median]

    for sigma_perm in used:
        # Skip identity (it's the canonical labeling).
        if sigma_perm == (1, 2, 3, 4, 5, 6, 7):
            continue
        res = run_subject(subject_index, n_epochs, sigma_perm=sigma_perm)
        f1s.append(res.F1_median)
        f2s.append(res.F2_median)
        f3s.append(res.F3_median)

    def _iqr(values: list[float]) -> float:
        q25 = float(np.percentile(values, 25))
        q75 = float(np.percentile(values, 75))
        return q75 - q25

    return {
        "subject_index": subject_index,
        "n_labellings_tested": len(f1s),
        "F1": {"canonical": canonical.F1_median, "iqr": _iqr(f1s),
                "mean_delta": float(np.mean(f1s) - canonical.F1_median)},
        "F2": {"canonical": canonical.F2_median, "iqr": _iqr(f2s),
                "mean_delta": float(np.mean(f2s) - canonical.F2_median)},
        "F3": {"canonical": canonical.F3_median, "iqr": _iqr(f3s),
                "mean_delta": float(np.mean(f3s) - canonical.F3_median)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-subjects", type=int, default=5)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--n-labellings", type=int, default=30)
    args = parser.parse_args()

    print(f"Running {args.n_subjects} synthetic subjects x {args.n_epochs} epochs...")
    results = [run_subject(i, args.n_epochs) for i in range(args.n_subjects)]

    r1 = _spearman([r.F1_median for r in results], [r.latents.rumination for r in results])
    r2 = _spearman([-r.F2_median for r in results], [r.latents.anhedonia for r in results])
    r3 = _spearman([r.F3_median for r in results], [abs(r.latents.valence) for r in results])
    print(json.dumps({"rho_F1_rumination": r1, "rho_nF2_anhedonia": r2, "rho_F3_valence": r3}, indent=2))

    print(f"Running §7.1 invariance smoke on subject 0 over {args.n_labellings} labellings...")
    smoke = invariance_smoke(0, n_epochs=min(args.n_epochs, 2), n_labellings=args.n_labellings)
    print(json.dumps(smoke, indent=2))


if __name__ == "__main__":
    main()
