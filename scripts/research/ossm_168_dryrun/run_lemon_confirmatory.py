"""Confirmatory harness for OSSM-168 v3 on real LEMON data.

Pipeline per subject:
    1. load raw EEG  (BrainVision via lemon_preprocess.load_lemon_raw)
    2. §5 preprocessing -> (n_epochs, 7, 1000) z-scored tensor
    3. untrained O-SSM forward pass per epoch -> trajectory (T, 2, 8)
    4. F1, F2, F3 per epoch -> subject-level median
    5. join with phenotypic endpoints (CERQ_Rumination, Hamilton, BAS_Drive, ...)

Aggregate analysis:
    - Spearman rho for each (feature, endpoint) pair in §7 hypothesis table
    - BCa bootstrap 95% CI with 1000 resamples (seed = 20260421)
    - Holm-Bonferroni correction over the three primary H1/H2/H3 tests

Outputs (all under --out-dir):
    features.csv            per-subject F1/F2/F3 medians plus retained epochs
    merged.csv              features + endpoints joined on subject_id
    results.json            Spearman rho + BCa CI + Holm-corrected p values

Preprocessed tensors are cached as ``.npy`` under --cache-dir so re-runs
and ablations are fast.

Example:
    python3 -u scripts/research/ossm_168_dryrun/run_lemon_confirmatory.py \
        --raw-root /workspace/data/lemon/raw_eeg \
        --endpoints /workspace/data/lemon/endpoints.csv \
        --cache-dir /workspace/data/lemon/preprocessed \
        --out-dir /workspace/data/lemon/results_pilot \
        --subjects-list /workspace/data/lemon/subjects_pilot.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.research.ossm_168_dryrun import (  # noqa: E402
    _features_fast as feat_fast,
    features as feat_mod,
    lemon_preprocess as lp,
    ossm as ossm_mod,
)

# -----------------------------------------------------------------------------
# Pinned statistical constants per prereg §7
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 20260421
BCA_CI = 0.95

# Hypothesis table: (label, feature_key, endpoint_column, expected_sign)
# expected_sign is +1 if feature expected to increase with endpoint, -1 otherwise.
HYPOTHESES = [
    # H1: F_1 rumination  <-> CERQ_Rumination (primary)
    ("H1_F1_vs_CERQrumination",  "F1",  "cerq_rumination", +1),
    # H2: -F_2 anhedonia  <-> BAS_Drive reversed  (lower BAS drive = more anhedonia)
    # We test rho(F_2, BAS_Drive); expected negative (higher F_2 means closer to ZD = more anhedonic)
    ("H2_F2_vs_BAS_Drive",       "F2",  "bas_drive",       -1),
    # H3: F_3 valence/depression  <-> Hamilton_Scale (primary clinical)
    ("H3_F3_vs_Hamilton",        "F3",  "hamilton",        +1),
]

# Secondary hypotheses (reported alongside but not Holm-adjusted with the primary set)
# H3s_F3_vs_BSL23 is preregistered as "planned missing": LEMON administered
# BSL-23 (Bohus 2009 Borderline Symptom List) to ~74% of the cohort
# (165/220 of subjects_all.txt), so this secondary runs on the available subset.
SECONDARY = [
    ("H1s_F1_vs_NEO_N",          "F1",  "neo_neuroticism", +1),
    ("H2s_F2_vs_BAS_Reward",     "F2",  "bas_reward",      -1),
    ("H3s_F3_vs_NEO_N",          "F3",  "neo_neuroticism", +1),
    ("H3s_F3_vs_STAI_trait",     "F3",  "stai_trait",      +1),
    ("H3s_F3_vs_BSL23",          "F3",  "bsl23",           +1),
]

# Endpoints where <100% coverage is expected by LEMON study design;
# coverage below the per-endpoint floor triggers a WARN, otherwise an INFO line.
ENDPOINT_COVERAGE_FLOOR = {
    "bsl23": 0.70,      # planned missing, ~74% upstream
    "hamilton": 0.90,
    "cerq_rumination": 0.90,
    "bas_drive": 0.90,
    "bas_reward": 0.90,
    "bas_fun": 0.90,
    "bis": 0.90,
    "neo_neuroticism": 0.90,
    "stai_trait": 0.90,
    "tas_total": 0.90,
}


@dataclass
class SubjectFeatures:
    subject_id: str
    n_epochs: int
    F1: float
    F2: float
    F3: float


def _load_endpoints(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["subject_id"]] = row
    return out


def _iter_subjects(raw_root: Path, subset_list: Path | None) -> list[str]:
    if subset_list is not None and subset_list.is_file():
        return [s.strip() for s in subset_list.read_text().splitlines() if s.strip()]
    return sorted(d.name for d in raw_root.iterdir() if d.is_dir() and d.name.startswith("sub-"))


def preprocess_or_load(
    subject: str, raw_root: Path, cache_dir: Path,
) -> tuple[np.ndarray, dict]:
    """Return ``(epochs_tensor, provenance)``. Caches .npy + .json.

    If the .npy is present but the .json is missing (e.g. an earlier
    run crashed between the two writes), load the epochs and backfill
    a minimal provenance record rather than repeating preprocessing.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy = cache_dir / f"{subject}_epochs.npy"
    meta = cache_dir / f"{subject}_meta.json"
    if npy.is_file():
        epochs = np.load(npy)
        if meta.is_file():
            prov = json.loads(meta.read_text())
        else:
            prov = {
                "subject": subject,
                "n_retained_epochs": int(epochs.shape[0]),
                "n_total_epochs": int(epochs.shape[0]),
                "source": "cached_npy_no_meta",
            }
            meta.write_text(json.dumps(prov, indent=2))
        return epochs, prov
    t0 = time.time()
    res = lp.preprocess_subject(str(raw_root), subject)
    # ``res.epochs`` is already (n_epochs, 7, 1000) per §5.
    epochs = np.asarray(res.epochs, dtype=np.float64)
    prov = {
        "subject": subject,
        "sfreq": res.sfreq,
        "n_total_epochs": res.n_total_epochs,
        "n_retained_epochs": res.n_retained_epochs,
        "retention_rate": res.retention_rate(),
        "n_ica_rejected": res.n_ica_rejected,
        "n_ica_total": res.n_ica_total,
        "channels_used_n": len(res.channel_names_used),
        "wall_time_s": time.time() - t0,
    }
    np.save(npy, epochs)
    meta.write_text(json.dumps(prov, indent=2))
    return epochs, prov


def features_for_subject(
    subject_index: int, epochs: np.ndarray,
    max_epochs: int | None = None,
    subsample_seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Return (F1_median, F2_median, F3_median) across retained epochs.

    ``max_epochs`` subsamples epochs deterministically (seeded with
    ``subsample_seed + subject_index``) to keep wall time tractable on
    the confirmatory run; when None every retained epoch is used.
    F_3 is computed with the vectorised implementation in
    :mod:`_features_fast` which is bit-identical to the reference up
    to floating-point round-off (~1e-13).
    """
    n = epochs.shape[0]
    if max_epochs is not None and n > max_epochs:
        rng = np.random.default_rng(subsample_seed + subject_index)
        idx = np.sort(rng.choice(n, size=max_epochs, replace=False))
        epochs = epochs[idx]
    f1s: list[float] = []
    f2s: list[float] = []
    f3s: list[float] = []
    for epoch in epochs:  # epoch shape: (7, 1000)
        traj = ossm_mod.forward_pass(np.ascontiguousarray(epoch), subject_index)
        f1s.append(feat_mod.feature_f1_from_trajectory(traj))
        f2s.append(feat_mod.feature_f2_from_trajectory(traj))
        f3s.append(feat_fast.feature_f3_fast(traj))
    return (statistics.median(f1s), statistics.median(f2s), statistics.median(f3s))


# -----------------------------------------------------------------------------
# Spearman + BCa bootstrap + Holm
# -----------------------------------------------------------------------------

def _rankdata_avg(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    rx = _rankdata_avg(x)
    ry = _rankdata_avg(y)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denom = float(np.sqrt((dx * dx).sum() * (dy * dy).sum()))
    return float((dx * dy).sum() / denom) if denom > 0 else 0.0


def _norm_cdf(z: float) -> float:
    from scipy.stats import norm  # type: ignore
    return float(norm.cdf(z))


def _norm_ppf(p: float) -> float:
    from scipy.stats import norm  # type: ignore
    return float(norm.ppf(p))


def bca_bootstrap_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = BOOTSTRAP_N,
    ci: float = BCA_CI, seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float]:
    """Return ``(theta_hat, lo, hi)`` BCa CI for Spearman rho(x, y)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    theta = spearman_rho(x, y)
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = spearman_rho(x[idx], y[idx])
    # bias-correction z0
    p0 = float(np.mean(boots < theta))
    p0 = min(max(p0, 1.0 / (n_boot + 1)), 1.0 - 1.0 / (n_boot + 1))
    z0 = _norm_ppf(p0)
    # acceleration via jackknife
    jk = np.empty(n, dtype=np.float64)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        jk[i] = spearman_rho(x[mask], y[mask])
    jk_mean = jk.mean()
    num = ((jk_mean - jk) ** 3).sum()
    den = 6.0 * (((jk_mean - jk) ** 2).sum() ** 1.5) + 1e-12
    a = num / den
    alpha = (1.0 - ci) / 2.0
    z_lo = _norm_ppf(alpha)
    z_hi = _norm_ppf(1.0 - alpha)
    a_lo = _norm_cdf(z0 + (z0 + z_lo) / (1.0 - a * (z0 + z_lo)))
    a_hi = _norm_cdf(z0 + (z0 + z_hi) / (1.0 - a * (z0 + z_hi)))
    lo = float(np.quantile(boots, a_lo))
    hi = float(np.quantile(boots, a_hi))
    return theta, lo, hi


def spearman_p_permutation(
    x: np.ndarray, y: np.ndarray, n_perm: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED + 1,
) -> float:
    """Two-sided permutation p-value for Spearman rho."""
    rng = np.random.default_rng(seed)
    theta = abs(spearman_rho(x, y))
    greater = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        if abs(spearman_rho(x, yp)) >= theta:
            greater += 1
    return (greater + 1) / (n_perm + 1)


def holm_bonferroni(pvals: list[float]) -> list[float]:
    """Return Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank_i, i in enumerate(order):
        adj_p = (m - rank_i) * pvals[i]
        running = max(running, min(1.0, adj_p))
        adj[i] = running
    return adj


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--endpoints", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--subjects-list", type=Path, default=None)
    ap.add_argument("--skip-existing-features", action="store_true",
                    help="skip subjects whose row is already in features.csv")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="deterministically subsample this many epochs per subject "
                    "(saves wall time; subject-level medians remain stable)")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    endpoints = _load_endpoints(args.endpoints)
    subjects = _iter_subjects(args.raw_root, args.subjects_list)
    print(f"[harness] {len(subjects)} subjects queued", flush=True)

    feat_rows: list[SubjectFeatures] = []
    feat_csv = args.out_dir / "features.csv"
    already: dict[str, SubjectFeatures] = {}
    if args.skip_existing_features and feat_csv.is_file():
        with feat_csv.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                already[row["subject_id"]] = SubjectFeatures(
                    subject_id=row["subject_id"],
                    n_epochs=int(row["n_epochs"]),
                    F1=float(row["F1"]),
                    F2=float(row["F2"]),
                    F3=float(row["F3"]),
                )

    # index subject_id -> stable 0-based subject_index (required for seed)
    sub_index_map = {sid: i for i, sid in enumerate(sorted(endpoints.keys()))}
    for pos, sid in enumerate(subjects):
        if sid in already:
            feat_rows.append(already[sid])
            print(f"[{pos + 1}/{len(subjects)}] {sid}  cached features", flush=True)
            continue
        si = sub_index_map.get(sid)
        if si is None:
            print(f"[{pos + 1}/{len(subjects)}] {sid}  SKIP (no endpoint record)", flush=True)
            continue
        t0 = time.time()
        try:
            epochs, prov = preprocess_or_load(sid, args.raw_root, args.cache_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[{pos + 1}/{len(subjects)}] {sid}  preprocess FAILED: {exc!r}",
                  flush=True)
            continue
        if epochs.shape[0] == 0:
            print(f"[{pos + 1}/{len(subjects)}] {sid}  0 epochs retained, skipping",
                  flush=True)
            continue
        F1, F2, F3 = features_for_subject(si, epochs, max_epochs=args.max_epochs)
        n_eff = min(epochs.shape[0], args.max_epochs) if args.max_epochs else epochs.shape[0]
        rec = SubjectFeatures(
            subject_id=sid, n_epochs=int(n_eff),
            F1=F1, F2=F2, F3=F3,
        )
        feat_rows.append(rec)
        print(
            f"[{pos + 1}/{len(subjects)}] {sid}  n_ep={rec.n_epochs} "
            f"F1={F1:.5g} F2={F2:.5g} F3={F3:.5g}  "
            f"wall={time.time() - t0:.1f}s  (prov keeps={prov['n_retained_epochs']}/"
            f"{prov['n_total_epochs']})",
            flush=True,
        )
        # Flush after every subject so crashes don't lose work.
        _write_features(feat_rows, feat_csv)

    if not feat_rows:
        print("[harness] no successful subjects — aborting.", flush=True)
        return

    merged, merged_path = _merge_with_endpoints(feat_rows, endpoints, args.out_dir)
    print(f"[harness] merged {len(merged)} subjects -> {merged_path}", flush=True)
    _log_endpoint_coverage(merged)

    results = _run_hypothesis_tests(merged)
    results_path = args.out_dir / "results.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"[harness] wrote {results_path}", flush=True)
    print(json.dumps(results, indent=2))


def _write_features(rows: list[SubjectFeatures], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["subject_id", "n_epochs", "F1", "F2", "F3"])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def _merge_with_endpoints(
    feat_rows: list[SubjectFeatures], endpoints: dict[str, dict], out_dir: Path,
) -> tuple[list[dict], Path]:
    out: list[dict] = []
    for r in feat_rows:
        ep = endpoints.get(r.subject_id)
        if ep is None:
            continue
        row = {"subject_id": r.subject_id, "n_epochs": r.n_epochs,
               "F1": r.F1, "F2": r.F2, "F3": r.F3}
        for key in ("hamilton", "bsl23", "cerq_rumination",
                    "neo_neuroticism", "bas_drive", "bas_fun",
                    "bas_reward", "bis", "stai_trait", "tas_total"):
            v = ep.get(key, "").strip() if isinstance(ep.get(key), str) else ep.get(key)
            try:
                row[key] = float(v) if v not in (None, "", "nan", "NA") else float("nan")
            except (TypeError, ValueError):
                row[key] = float("nan")
        out.append(row)
    path = out_dir / "merged.csv"
    if out:
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
            w.writeheader()
            w.writerows(out)
    return out, path


def _tests_for_set(merged: list[dict], hyp_table: list[tuple]) -> list[dict]:
    out: list[dict] = []
    for label, feat_key, ep_key, sign in hyp_table:
        x = np.array([r[feat_key] for r in merged], dtype=np.float64)
        y = np.array([r[ep_key] for r in merged], dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 10:
            out.append({"label": label, "n": int(mask.sum()), "note": "n<10"})
            continue
        theta, lo, hi = bca_bootstrap_ci(x[mask], y[mask])
        p = spearman_p_permutation(x[mask], y[mask])
        out.append({
            "label": label, "feature": feat_key, "endpoint": ep_key,
            "expected_sign": sign, "n": int(mask.sum()),
            "rho": theta, "ci95_lo": lo, "ci95_hi": hi, "p_perm": p,
        })
    return out


def _endpoint_coverage(merged: list[dict]) -> dict[str, dict]:
    keys = ("hamilton", "bsl23", "cerq_rumination", "neo_neuroticism",
            "bas_drive", "bas_fun", "bas_reward", "bis",
            "stai_trait", "tas_total")
    n = len(merged)
    out: dict[str, dict] = {}
    for k in keys:
        vals = [r.get(k) for r in merged]
        nn = sum(1 for v in vals if v is not None and np.isfinite(float(v)))
        floor = ENDPOINT_COVERAGE_FLOOR.get(k, 0.90)
        out[k] = {
            "n_nonnull": int(nn), "n_total": int(n),
            "coverage": (nn / n) if n else 0.0,
            "floor": float(floor),
            "below_floor": bool(n and (nn / n) < floor),
        }
    return out


def _log_endpoint_coverage(merged: list[dict]) -> None:
    cov = _endpoint_coverage(merged)
    for k, rec in cov.items():
        tag = "WARN" if rec["below_floor"] else "info"
        print(
            f"[harness] {tag} endpoint coverage {k:>20s}: "
            f"{rec['n_nonnull']}/{rec['n_total']} "
            f"({100 * rec['coverage']:.1f}%, floor={100 * rec['floor']:.0f}%)",
            flush=True,
        )


def _run_hypothesis_tests(merged: list[dict]) -> dict:
    primary = _tests_for_set(merged, HYPOTHESES)
    secondary = _tests_for_set(merged, SECONDARY)
    prim_p = [r.get("p_perm", 1.0) for r in primary]
    holm = holm_bonferroni(prim_p)
    for r, q in zip(primary, holm):
        r["p_holm"] = q
    return {
        "n_subjects_merged": len(merged),
        "n_bootstrap": BOOTSTRAP_N, "bca_ci": BCA_CI,
        "endpoint_coverage": _endpoint_coverage(merged),
        "primary": primary, "secondary": secondary,
    }


if __name__ == "__main__":
    main()
