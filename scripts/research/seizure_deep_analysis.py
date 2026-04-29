#!/usr/bin/env python3
"""Deep analysis of O-SSM Hessian non-associativity in seizure EEG.

Implements the O-SSM model and Hessian computation in NumPy for:
  1. Per-patient breakdown (is ictal>baseline consistent across all 5?)
  2. Sliding-window microdinâmica (curvature curve over seizure time course)
  3. Effect sizes (Cohen's d with bootstrap 95% CI)
  4. Neuroanatomical Fano-line mapping
"""

import numpy as np
import os
import sys


# ═══ Octonion multiplication ═══

def oct_mul(a, b):
    """Multiply two octonions (8-vectors)."""
    r = np.zeros(8)
    r[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3] - a[4]*b[4] - a[5]*b[5] - a[6]*b[6] - a[7]*b[7]
    r[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2] + a[4]*b[5] - a[5]*b[4] - a[6]*b[7] + a[7]*b[6]
    r[2] = a[0]*b[2] + a[2]*b[0] - a[1]*b[3] + a[3]*b[1] + a[4]*b[6] - a[6]*b[4] + a[5]*b[7] - a[7]*b[5]
    r[3] = a[0]*b[3] + a[3]*b[0] + a[1]*b[2] - a[2]*b[1] + a[4]*b[7] - a[7]*b[4] - a[5]*b[6] + a[6]*b[5]
    r[4] = a[0]*b[4] + a[4]*b[0] - a[1]*b[5] + a[5]*b[1] - a[2]*b[6] + a[6]*b[2] - a[3]*b[7] + a[7]*b[3]
    r[5] = a[0]*b[5] + a[5]*b[0] + a[1]*b[4] - a[4]*b[1] - a[2]*b[7] + a[7]*b[2] + a[3]*b[6] - a[6]*b[3]
    r[6] = a[0]*b[6] + a[6]*b[0] + a[1]*b[7] - a[7]*b[1] + a[2]*b[4] - a[4]*b[2] - a[3]*b[5] + a[5]*b[3]
    r[7] = a[0]*b[7] + a[7]*b[0] - a[1]*b[6] + a[6]*b[1] - a[2]*b[5] + a[5]*b[2] + a[3]*b[4] - a[4]*b[3]
    return r


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


# ═══ O-SSM Model ═══

class OSSM:
    def __init__(self, rng_seed=42):
        rng = np.random.RandomState(rng_seed)
        self.A = rng.randn(8)
        self.A /= np.linalg.norm(self.A) + 1e-12
        self.W = rng.randn(10) * 0.1  # 8 projection + 1 assoc weight + 1 bias
        self.mom = np.zeros(10)

    def forward(self, features_64):
        """Run O-SSM on 8 steps × 8 channels, return (logit, assoc_accum)."""
        h = np.zeros(8)
        h[0] = 1.0
        assoc_total = 0.0

        for step in range(8):
            x = features_64[step * 8:(step + 1) * 8]

            left = oct_mul(oct_mul(self.A, h), x)
            right = oct_mul(self.A, oct_mul(h, x))
            assoc_step = np.linalg.norm(left - right)
            assoc_step = min(assoc_step, 1000.0)
            assoc_total += assoc_step

            h = left + 0.2 * right
            h = np.clip(h, -5, 5)
            h = np.nan_to_num(h, 0.0)

        af = assoc_total / 8.0
        logit = self.W[9] + np.dot(h, self.W[:8]) + af * self.W[8]
        return logit, af, h

    def train_step(self, features_64, target, lr=0.02):
        logit, af, h = self.forward(features_64)
        prob = sigmoid(logit)
        grad = np.clip(prob - target, -0.5, 0.5)

        self.mom[:8] = 0.9 * self.mom[:8] + h * grad
        self.W[:8] -= lr * self.mom[:8]
        self.mom[8] = 0.9 * self.mom[8] + af * grad
        self.W[8] -= lr * self.mom[8]
        self.mom[9] = 0.9 * self.mom[9] + grad
        self.W[9] -= lr * self.mom[9]

    def loss(self, features_64, target):
        logit, _, _ = self.forward(features_64)
        prob = sigmoid(logit)
        return -(target * prob + (1.0 - target) * (1.0 - prob))

    def hessian(self, features_64, target):
        """Compute 8×8 Hessian of loss w.r.t. A via finite differences."""
        eps = 0.001
        f0 = self._loss_with_A(self.A.copy(), features_64, target)
        H = np.zeros((8, 8))

        for i in range(8):
            Ap = self.A.copy(); Ap[i] += eps
            Am = self.A.copy(); Am[i] -= eps
            H[i, i] = (self._loss_with_A(Ap, features_64, target) - 2*f0 +
                        self._loss_with_A(Am, features_64, target)) / (eps * eps)

            for j in range(i+1, 8):
                App = self.A.copy(); App[i] += eps; App[j] += eps
                Apm = self.A.copy(); Apm[i] += eps; Apm[j] -= eps
                Amp = self.A.copy(); Amp[i] -= eps; Amp[j] += eps
                Amm = self.A.copy(); Amm[i] -= eps; Amm[j] -= eps
                hij = (self._loss_with_A(App, features_64, target) -
                       self._loss_with_A(Apm, features_64, target) -
                       self._loss_with_A(Amp, features_64, target) +
                       self._loss_with_A(Amm, features_64, target)) / (4 * eps * eps)
                H[i, j] = hij
                H[j, i] = hij

        return H

    def _loss_with_A(self, A_tmp, features_64, target):
        orig = self.A.copy()
        self.A = A_tmp
        l = self.loss(features_64, target)
        self.A = orig
        return l


# ═══ Data loading ═══

def load_main_manifest(path):
    """Load seizure_hessian_manifest.tsv."""
    records = []
    with open(path) as fp:
        for line in fp:
            if line.startswith("#") or line.startswith("subject_id"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 67:
                continue
            records.append({
                "id": parts[0],
                "label": int(parts[1]),
                "patient": parts[2],
                "features": np.array([float(x) for x in parts[3:67]]),
            })
    return records


def load_sliding_manifest(path):
    records = []
    with open(path) as fp:
        for line in fp:
            if line.startswith("#") or line.startswith("subject_id"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 68:
                continue
            records.append({
                "id": parts[0],
                "label": int(parts[1]),
                "patient": parts[2],
                "time_rel": float(parts[3]),
                "features": np.array([float(x) for x in parts[4:68]]),
            })
    return records


# ═══ Fano line decomposition ═══

FANO_LINES = [
    (1, 2, 4, "L0: (e1,e2,e4) FP1-F7/F7-T7/FP2-F4"),
    (2, 3, 5, "L1: (e2,e3,e5) F7-T7/FP2-F8/F3-C3"),
    (3, 4, 6, "L2: (e3,e4,e6) FP2-F8/F8-T8/FP2-F4"),
    (4, 5, 7, "L3: (e4,e5,e7) F8-T8/FP1-F3/F4-C4"),
    (5, 6, 1, "L4: (e5,e6,e1) FP1-F3/F3-C3/FP1-F7"),
    (6, 7, 2, "L5: (e6,e7,e2) F3-C3/FP2-F4/F7-T7"),
    (7, 1, 3, "L6: (e7,e1,e3) FP2-F4/FP1-F7/FP2-F8"),
]

CHANNEL_MAP = {
    0: "e0 (real)",
    1: "e1 = FP1-F7 (L frontal→temporal)",
    2: "e2 = F7-T7  (L temporal chain)",
    3: "e3 = FP2-F8 (R frontal→temporal)",
    4: "e4 = F8-T8  (R temporal chain)",
    5: "e5 = FP1-F3 (L frontal→central)",
    6: "e6 = F3-C3  (L central chain)",
    7: "e7 = FP2-F4 (R frontal→central)",
}


def fano_energy(H, a, b, c):
    return (abs(H[a,b]) + abs(H[b,a]) + abs(H[a,c]) + abs(H[c,a]) +
            abs(H[b,c]) + abs(H[c,b]))


def offdiag_norm(H):
    return np.sum(np.abs(H)) - np.sum(np.abs(np.diag(H)))


def hssm_offdiag_norm(H):
    sub = H[:4, :4]
    return np.sum(np.abs(sub)) - np.sum(np.abs(np.diag(sub)))


# ═══ Analysis ═══

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return 0.0
    pooled_std = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    if pooled_std < 1e-12:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled_std


def bootstrap_ci(a, b, n_boot=2000, alpha=0.05):
    """Bootstrap CI for difference of means."""
    rng = np.random.RandomState(42)
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(sa) - np.mean(sb))
    diffs = np.sort(diffs)
    lo = diffs[int(alpha/2 * n_boot)]
    hi = diffs[int((1-alpha/2) * n_boot)]
    return lo, hi


def main():
    main_path = "artifacts/research/eeg_hessian/seizure_hessian_manifest.tsv"
    sliding_path = "artifacts/research/eeg_hessian/sliding_seizure_manifest.tsv"

    print("Loading main manifest...")
    records = load_main_manifest(main_path)
    print(f"  {len(records)} windows")

    patients = sorted(set(r["patient"] for r in records))
    print(f"  Patients: {patients}")
    print()

    # ═══ 1. Per-patient analysis ═══
    print("=" * 70)
    print("  1. PER-PATIENT HESSIAN ANALYSIS")
    print("=" * 70)

    label_names = {0: "baseline", 1: "interictal", 2: "ictal", 3: "pre_ictal", 4: "post_ictal"}
    all_per_patient = {}

    for pi, holdout in enumerate(patients):
        print(f"\n  Patient {holdout} (fold {pi+1}/{len(patients)}):")
        train_data = [r for r in records if r["patient"] != holdout]
        test_data = [r for r in records if r["patient"] == holdout]

        model = OSSM(rng_seed=42 + pi * 1000)

        # Train
        rng = np.random.RandomState(42 + pi)
        for epoch in range(80):
            for _ in range(300):
                r = train_data[rng.randint(len(train_data))]
                target = 1.0 if r["label"] == 2 else 0.0
                model.train_step(r["features"], target, lr=0.02)

        # Test accuracy
        correct = sum(1 for r in test_data if (model.forward(r["features"])[0] > 0) == (r["label"] == 2))
        print(f"    Test accuracy: {correct}/{len(test_data)} = {100*correct/len(test_data):.1f}%")

        # Compute Hessians per class
        patient_results = {}
        for r in test_data:
            target = 1.0 if r["label"] == 2 else 0.0
            H = model.hessian(r["features"], target)
            od = offdiag_norm(H)
            lab = r["label"]
            if lab not in patient_results:
                patient_results[lab] = []
            patient_results[lab].append(od)

        for lab in sorted(patient_results.keys()):
            vals = patient_results[lab]
            print(f"    {label_names.get(lab, str(lab)):12s}: n={len(vals):3d}, mean_od={np.mean(vals):.4f}, std={np.std(vals):.4f}")

        all_per_patient[holdout] = patient_results

    # Summary: is ictal > baseline consistent?
    print("\n" + "-" * 70)
    print("  PER-PATIENT CONSISTENCY: Ictal vs Baseline")
    print("-" * 70)
    consistent = 0
    for p in patients:
        pr = all_per_patient[p]
        ict = np.mean(pr.get(2, [0]))
        bas = np.mean(pr.get(0, [0]))
        inter = np.mean(pr.get(1, [0]))
        d = cohens_d(pr.get(2, [0]), pr.get(0, [0]))
        status = "ICTAL > BASE" if ict > bas else "BASE > ICTAL"
        if ict > bas:
            consistent += 1
        print(f"    {p}: ictal={ict:.4f} baseline={bas:.4f} interictal={inter:.4f} Cohen's d={d:.2f} [{status}]")

    print(f"\n    Consistent: {consistent}/{len(patients)} patients")

    # ═══ 2. Effect sizes with bootstrap CI ═══
    print("\n" + "=" * 70)
    print("  2. EFFECT SIZES (Cohen's d + Bootstrap 95% CI)")
    print("=" * 70)

    all_od = {lab: [] for lab in range(5)}
    for p, pr in all_per_patient.items():
        for lab, vals in pr.items():
            all_od[lab].extend(vals)

    for lab_a, lab_b, desc in [(2, 0, "Ictal vs Baseline"), (2, 1, "Ictal vs Interictal"),
                                (4, 0, "Post-ictal vs Baseline"), (3, 1, "Pre-ictal vs Interictal")]:
        a, b = np.array(all_od.get(lab_a, [0])), np.array(all_od.get(lab_b, [0]))
        if len(a) > 1 and len(b) > 1:
            d = cohens_d(a, b)
            lo, hi = bootstrap_ci(a, b)
            print(f"  {desc:30s}: d={d:+.3f}, diff_mean={np.mean(a)-np.mean(b):+.6f}, 95%CI=[{lo:+.6f}, {hi:+.6f}]")

    # ═══ 3. Fano line per-patient ═══
    print("\n" + "=" * 70)
    print("  3. FANO LINE DECOMPOSITION (per-patient)")
    print("=" * 70)

    for pi, holdout in enumerate(patients):
        print(f"\n  {holdout}:")
        train_data = [r for r in records if r["patient"] != holdout]
        test_data = [r for r in records if r["patient"] == holdout]

        model = OSSM(rng_seed=42 + pi * 1000)
        rng = np.random.RandomState(42 + pi)
        for epoch in range(80):
            for _ in range(300):
                r = train_data[rng.randint(len(train_data))]
                target = 1.0 if r["label"] == 2 else 0.0
                model.train_step(r["features"], target, lr=0.02)

        fano_by_label = {lab: np.zeros(7) for lab in [0, 1, 2]}
        fano_counts = {lab: 0 for lab in [0, 1, 2]}

        ict_test = [r for r in test_data if r["label"] in [0, 1, 2]]
        for r in ict_test[:100]:  # cap per patient for speed
            target = 1.0 if r["label"] == 2 else 0.0
            H = model.hessian(r["features"], target)
            lab = r["label"]
            for fi, (a, b, c, _) in enumerate(FANO_LINES):
                fano_by_label[lab][fi] += fano_energy(H, a, b, c)
            fano_counts[lab] += 1

        for fi, (a, b, c, desc) in enumerate(FANO_LINES):
            ict_e = fano_by_label[2][fi] / max(1, fano_counts[2])
            bas_e = fano_by_label[0][fi] / max(1, fano_counts[0])
            ratio = ict_e / bas_e if bas_e > 1e-12 else float('inf')
            print(f"    {desc[:35]:35s}  ictal={ict_e:.4f}  base={bas_e:.4f}  ratio={ratio:.1f}x")

    # ═══ 4. O-SSM vs H-SSM per patient ═══
    print("\n" + "=" * 70)
    print("  4. O-SSM vs H-SSM (per-patient, ictal only)")
    print("=" * 70)

    for pi, holdout in enumerate(patients):
        train_data = [r for r in records if r["patient"] != holdout]
        test_ict = [r for r in records if r["patient"] == holdout and r["label"] == 2]

        model = OSSM(rng_seed=42 + pi * 1000)
        rng = np.random.RandomState(42 + pi)
        for epoch in range(80):
            for _ in range(300):
                r = train_data[rng.randint(len(train_data))]
                target = 1.0 if r["label"] == 2 else 0.0
                model.train_step(r["features"], target, lr=0.02)

        o_ods = []
        h_ods = []
        for r in test_ict[:60]:
            H = model.hessian(r["features"], 1.0)
            o_ods.append(offdiag_norm(H))
            h_ods.append(hssm_offdiag_norm(H))

        o_mean = np.mean(o_ods) if o_ods else 0
        h_mean = np.mean(h_ods) if h_ods else 0
        extra_pct = (o_mean - h_mean) / h_mean * 100 if h_mean > 1e-12 else 0
        print(f"  {holdout}: O-SSM={o_mean:.4f}  H-SSM={h_mean:.4f}  octonionic_extra=+{extra_pct:.0f}%")

    # ═══ 5. Sliding window microdinâmica ═══
    print("\n" + "=" * 70)
    print("  5. SEIZURE MICRODINÂMICA (sliding-window curvature)")
    print("=" * 70)

    if os.path.exists(sliding_path):
        sliding = load_sliding_manifest(sliding_path)
        print(f"  Loaded {len(sliding)} sliding windows")

        # Group by patient+seizure, sample for speed
        seizure_groups = {}
        for r in sliding:
            key = r["id"].rsplit("_", 1)[0]  # patient_file
            patient = r["patient"]
            if patient not in seizure_groups:
                seizure_groups[patient] = {}
            skey = key
            if skey not in seizure_groups[patient]:
                seizure_groups[patient][skey] = []
            seizure_groups[patient][skey].append(r)

        # Train one model per patient (leave-one-out)
        for pi, holdout in enumerate(patients):
            if holdout not in seizure_groups:
                continue

            train_data = [r for r in records if r["patient"] != holdout]
            model = OSSM(rng_seed=42 + pi * 1000)
            rng = np.random.RandomState(42 + pi)
            for epoch in range(80):
                for _ in range(300):
                    r = train_data[rng.randint(len(train_data))]
                    target = 1.0 if r["label"] == 2 else 0.0
                    model.train_step(r["features"], target, lr=0.02)

            for skey, windows in sorted(seizure_groups[holdout].items()):
                # Sample every ~2 seconds (512 samples = 2s, stride=4 → ~128 windows)
                step = max(1, len(windows) // 80)
                sampled = windows[::step]

                print(f"\n  {skey} ({len(sampled)} points):")
                print(f"    {'time_rel':>10s}  {'phase':>12s}  {'||H_od||':>12s}")
                print(f"    {'--------':>10s}  {'-----':>12s}  {'--------':>12s}")

                for r in sampled:
                    target = 1.0 if r["label"] == 2 else 0.0
                    H = model.hessian(r["features"], target)
                    od = offdiag_norm(H)
                    phase = {2: "ICTAL", 3: "pre_ictal", 4: "post_ictal"}.get(r["label"], "?")
                    bar = "#" * min(50, int(od * 200))
                    print(f"    {r['time_rel']:>10.1f}s  {phase:>12s}  {od:>12.6f}  {bar}")

    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
