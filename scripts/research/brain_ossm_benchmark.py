#!/usr/bin/env python3
"""
Direction B: Brain O-SSM ABIDE full-cohort benchmark.

Optimized NumPy implementation of the O-SSM and H-SSM models
with leave-one-site-out cross-validation. Faithful port of
brain_ossm_abide.sio.
"""

import numpy as np
import json
import sys
import time
from pathlib import Path

N_SEEDS = 20
N_EPOCHS = 60
LR = 0.03
N_STEPS = 8
N_DIMS = 8

SEEDS_A = [55555,11111,99887,22222,44444,66666,77777,88888,33333,12321,
           24680,13579,54321,67890,11223,44567,77889,99001,31415,27182]
SEEDS_B = [77777,33333,44556,55555,66666,88888,11111,22222,99999,45654,
           86420,97531,12345,98765,33211,76544,98877,10099,92653,58979]


def load_manifest(path):
    sids, labels, sites_list, feats_list = [], [], [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if parts[0] == "subject_id":
                continue
            sids.append(parts[0])
            labels.append(1 if parts[1] in ("ASD", "asd", "1", "AUT") else 0)
            sites_list.append(parts[2])
            feats_list.append([float(x) for x in parts[3:67]])
    return sids, np.array(labels, dtype=np.int32), sites_list, np.array(feats_list, dtype=np.float64)


# Pre-compute octonion multiplication table for speed
# e_i * e_j = sign * e_k encoded as MUL_TABLE[i][j] = (sign, k)
_OCT_TABLE = {}
def _init_oct_table():
    """Build octonion multiplication table using the standard Cayley–Dickson / Fano plane."""
    # Cayley table for imaginary units (1-indexed i,j -> sign, k)
    # Using convention: e_i * e_j = epsilon_{ijk} * e_k for Fano triples
    fano_cycles = [
        (1,2,3), (1,4,5), (2,4,6), (2,5,7), (3,4,7), (1,6,7), (3,5,6)
    ]
    for a, b, c in fano_cycles:
        _OCT_TABLE[(a,b)] = (1, c)
        _OCT_TABLE[(b,c)] = (1, a)
        _OCT_TABLE[(c,a)] = (1, b)
        _OCT_TABLE[(b,a)] = (-1, c)
        _OCT_TABLE[(c,b)] = (-1, a)
        _OCT_TABLE[(a,c)] = (-1, b)

_init_oct_table()


def oct_mul(a, b):
    """Octonion multiplication matching the Sounio stdlib convention."""
    r = np.zeros(8)
    r[0] = a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]-a[4]*b[4]-a[5]*b[5]-a[6]*b[6]-a[7]*b[7]
    r[1] = a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2]+a[4]*b[5]-a[5]*b[4]-a[6]*b[7]+a[7]*b[6]
    r[2] = a[0]*b[2]+a[2]*b[0]-a[1]*b[3]+a[3]*b[1]+a[4]*b[6]-a[6]*b[4]+a[5]*b[7]-a[7]*b[5]
    r[3] = a[0]*b[3]+a[3]*b[0]+a[1]*b[2]-a[2]*b[1]+a[4]*b[7]-a[7]*b[4]-a[5]*b[6]+a[6]*b[5]
    r[4] = a[0]*b[4]+a[4]*b[0]-a[1]*b[5]+a[5]*b[1]-a[2]*b[6]+a[6]*b[2]-a[3]*b[7]+a[7]*b[3]
    r[5] = a[0]*b[5]+a[5]*b[0]+a[1]*b[4]-a[4]*b[1]-a[2]*b[7]+a[7]*b[2]+a[3]*b[6]-a[6]*b[3]
    r[6] = a[0]*b[6]+a[6]*b[0]+a[1]*b[7]-a[7]*b[1]+a[2]*b[4]-a[4]*b[2]-a[3]*b[5]+a[5]*b[3]
    r[7] = a[0]*b[7]+a[7]*b[0]-a[1]*b[6]+a[6]*b[1]-a[2]*b[5]+a[5]*b[2]+a[3]*b[4]-a[4]*b[3]
    return r


def quat_mul(a, b):
    r = np.zeros(4)
    r[0] = a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3]
    r[1] = a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2]
    r[2] = a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1]
    r[3] = a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0]
    return r


class XorShift128:
    def __init__(self, a, b):
        self.a = np.int64(a & 0x7FFFFFFFFFFFFFFF)
        self.b = np.int64(b & 0x7FFFFFFFFFFFFFFF)
    
    def next_val(self):
        x = int(self.a)
        y = int(self.b)
        self.a = np.int64(y)
        x ^= (x << 23) & 0xFFFFFFFFFFFFFFFF
        x ^= (x >> 17)
        x ^= (y ^ (y >> 26))
        x = x & 0xFFFFFFFFFFFFFFFF
        if x >= (1 << 63):
            x -= (1 << 64)
        self.b = np.int64(x)
        r = (x + y)
        if r < 0:
            r = -r
        return r
    
    def randf(self):
        return ((self.next_val() // 100) % 20001 - 10000) / 10000.0


def sigmoid(x):
    x = np.clip(x, -30, 30)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def sigmoid_scalar(x):
    x = max(-30.0, min(30.0, x))
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        ex = np.exp(x)
        return ex / (1.0 + ex)


def run_fold_oct(features, labels, train_mask, test_mask, rng, n_total):
    """Run one O-SSM fold. features: (N, 8, 8), labels: (N,)."""
    W = np.array([rng.randf() / 6.0 for _ in range(10)])
    mom = np.zeros(10)
    
    A = np.array([rng.randf() for _ in range(8)])
    n = np.linalg.norm(A)
    if n > 1e-9:
        A /= n
    else:
        A = np.zeros(8); A[0] = 1.0
    
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    n_train = len(train_idx)
    
    if n_train == 0 or len(test_idx) == 0:
        return None
    
    for ep in range(N_EPOCHS):
        for _ in range(n_train):
            si = rng.next_val() // 100 % n_total
            if not train_mask[si]:
                continue
            x_seq = features[si]  # (8, 8)
            
            h = np.zeros(8); h[0] = 1.0
            assoc_acc = 0.0
            
            for step in range(N_STEPS):
                x = x_seq[step]
                Ah = oct_mul(A, h)
                left = oct_mul(Ah, x)
                hx = oct_mul(h, x)
                right = oct_mul(A, hx)
                diff = left - right
                asq = np.dot(diff, diff)
                assoc_step = min(np.sqrt(max(asq, 0.0)), 1000.0)
                if not np.isfinite(assoc_step):
                    assoc_step = 0.0
                assoc_acc += assoc_step
                h = np.clip(left + 0.2 * right, -5.0, 5.0)
                h = np.where(np.isfinite(h), h, 0.0)
            
            assoc_feature = assoc_acc / 8.0
            if not np.isfinite(assoc_feature):
                assoc_feature = 0.0
            
            logit = W[9] + np.dot(h, W[:8]) + assoc_feature * W[8]
            prob = sigmoid_scalar(logit)
            grad = np.clip(prob - labels[si], -0.5, 0.5)
            
            mom[:8] = mom[:8] * 0.9 + h * grad
            W[:8] -= LR * mom[:8]
            mom[8] = mom[8] * 0.9 + assoc_feature * grad
            W[8] -= LR * mom[8]
            mom[9] = mom[9] * 0.9 + grad
            W[9] -= LR * mom[9]
    
    tp = tn = fp = fn = 0
    pos_n = neg_n = 0
    assoc_sum = 0.0
    
    for si in test_idx:
        x_seq = features[si]
        h = np.zeros(8); h[0] = 1.0
        assoc_acc = 0.0
        
        for step in range(N_STEPS):
            x = x_seq[step]
            Ah = oct_mul(A, h)
            left = oct_mul(Ah, x)
            hx = oct_mul(h, x)
            right = oct_mul(A, hx)
            diff = left - right
            asq = np.dot(diff, diff)
            assoc_step = min(np.sqrt(max(asq, 0.0)), 1000.0)
            if not np.isfinite(assoc_step):
                assoc_step = 0.0
            assoc_acc += assoc_step
            h = np.clip(left + 0.2 * right, -5.0, 5.0)
            h = np.where(np.isfinite(h), h, 0.0)
        
        assoc_feature = assoc_acc / 8.0
        if not np.isfinite(assoc_feature):
            assoc_feature = 0.0
        assoc_sum += assoc_feature
        
        logit = W[9] + np.dot(h, W[:8]) + assoc_feature * W[8]
        prob = sigmoid_scalar(logit)
        pred = 1 if prob >= 0.5 else 0
        label = labels[si]
        
        if pred == 1 and label == 1: tp += 1
        elif pred == 0 and label == 0: tn += 1
        elif pred == 1 and label == 0: fp += 1
        else: fn += 1
        if label == 1: pos_n += 1
        else: neg_n += 1
    
    n_test = len(test_idx)
    if pos_n == 0 or neg_n == 0 or n_test == 0:
        return None
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal = 50.0 * (tpr + tnr)
    return bal


def run_fold_hssm(features, labels, train_mask, test_mask, rng, n_total):
    """Run one H-SSM fold. features: (N, 8, 8), labels: (N,)."""
    W = np.array([rng.randf() / 6.0 for _ in range(10)])
    mom = np.zeros(10)
    
    A = np.array([rng.randf() for _ in range(8)])
    # H-SSM normalizes each quaternion half separately
    n1 = np.linalg.norm(A[:4])
    if n1 > 1e-9: A[:4] /= n1
    else: A[:4] = 0; A[0] = 1.0
    n2 = np.linalg.norm(A[4:])
    if n2 > 1e-9: A[4:] /= n2
    else: A[4:] = 0; A[4] = 1.0
    
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    n_train = len(train_idx)
    
    if n_train == 0 or len(test_idx) == 0:
        return None
    
    for ep in range(N_EPOCHS):
        for _ in range(n_train):
            si = rng.next_val() // 100 % n_total
            if not train_mask[si]:
                continue
            x_seq = features[si]
            
            h = np.zeros(8); h[0] = 1.0
            for step in range(N_STEPS):
                x = x_seq[step]
                t = quat_mul(A[:4], h[:4])
                left0 = quat_mul(t, x[:4])
                t2 = quat_mul(A[4:], h[4:])
                left1 = quat_mul(t2, x[4:])
                h[:4] = np.clip(1.2 * left0, -5.0, 5.0)
                h[4:] = np.clip(1.2 * left1, -5.0, 5.0)
            
            logit = W[9] + np.dot(h, W[:8])
            prob = sigmoid_scalar(logit)
            grad = np.clip(prob - labels[si], -0.5, 0.5)
            
            mom[:8] = mom[:8] * 0.9 + h * grad
            W[:8] -= LR * mom[:8]
            mom[9] = mom[9] * 0.9 + grad
            W[9] -= LR * mom[9]
    
    tp = tn = fp = fn = 0
    pos_n = neg_n = 0
    
    for si in test_idx:
        x_seq = features[si]
        h = np.zeros(8); h[0] = 1.0
        for step in range(N_STEPS):
            x = x_seq[step]
            t = quat_mul(A[:4], h[:4])
            left0 = quat_mul(t, x[:4])
            t2 = quat_mul(A[4:], h[4:])
            left1 = quat_mul(t2, x[4:])
            h[:4] = np.clip(1.2 * left0, -5.0, 5.0)
            h[4:] = np.clip(1.2 * left1, -5.0, 5.0)
        
        logit = W[9] + np.dot(h, W[:8])
        prob = sigmoid_scalar(logit)
        pred = 1 if prob >= 0.5 else 0
        label = labels[si]
        
        if pred == 1 and label == 1: tp += 1
        elif pred == 0 and label == 0: tn += 1
        elif pred == 1 and label == 0: fp += 1
        else: fn += 1
        if label == 1: pos_n += 1
        else: neg_n += 1
    
    n_test = len(test_idx)
    if pos_n == 0 or neg_n == 0 or n_test == 0:
        return None
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    return 50.0 * (tpr + tnr)


def run_benchmark(manifest_path, n_seeds=None):
    if n_seeds is None:
        n_seeds = N_SEEDS
    
    sids, labels, sites_list, feats_flat = load_manifest(manifest_path)
    features = feats_flat.reshape(-1, N_STEPS, N_DIMS)
    n = len(sids)
    sites = sorted(set(sites_list))
    sites_arr = np.array(sites_list)
    n_asd = int(np.sum(labels == 1))
    n_td = n - n_asd
    
    print("=" * 65)
    print("  Brain O-SSM ABIDE Benchmark (Python)")
    print("=" * 65)
    print(f"  Manifest: {manifest_path.name}")
    print(f"  Subjects: {n} (ASD={n_asd}, TD={n_td})")
    print(f"  Sites: {len(sites)}")
    print(f"  Seeds: {n_seeds}")
    print()
    
    o_bals = []
    h_bals = []
    o_site_bals = {s: [] for s in sites}
    h_site_bals = {s: [] for s in sites}
    
    t_start = time.time()
    
    for seed_i in range(n_seeds):
        sa, sb = SEEDS_A[seed_i], SEEDS_B[seed_i]
        o_fold_bals = []
        h_fold_bals = []
        
        for site in sites:
            test_mask = sites_arr == site
            train_mask = ~test_mask
            
            rng_o = XorShift128(sa + hash(site), sb + hash(site) * 17)
            o_bal = run_fold_oct(features, labels, train_mask, test_mask, rng_o, n)
            
            rng_h = XorShift128(sa + hash(site), sb + hash(site) * 17)
            h_bal = run_fold_hssm(features, labels, train_mask, test_mask, rng_h, n)
            
            if o_bal is not None:
                o_fold_bals.append(o_bal)
                o_site_bals[site].append(o_bal)
            if h_bal is not None:
                h_fold_bals.append(h_bal)
                h_site_bals[site].append(h_bal)
        
        o_mean_seed = np.mean(o_fold_bals) if o_fold_bals else 0
        h_mean_seed = np.mean(h_fold_bals) if h_fold_bals else 0
        o_bals.append(o_mean_seed)
        h_bals.append(h_mean_seed)
        
        elapsed = time.time() - t_start
        eta = elapsed / (seed_i + 1) * (n_seeds - seed_i - 1)
        print(f"  Seed {seed_i+1:2d}/{n_seeds}: O={o_mean_seed:.2f}% H={h_mean_seed:.2f}% gap={o_mean_seed-h_mean_seed:+.2f}pp  [{elapsed:.0f}s, ETA {eta:.0f}s]")
    
    o_mean = np.mean(o_bals)
    h_mean = np.mean(h_bals)
    o_std = np.std(o_bals)
    h_std = np.std(h_bals)
    gap = o_mean - h_mean
    
    print(f"\n  O-SSM balanced accuracy: {o_mean:.2f} +/- {o_std:.2f}")
    print(f"  H-SSM balanced accuracy: {h_mean:.2f} +/- {h_std:.2f}")
    print(f"  Gap (O-H): {gap:+.2f} pp")
    
    print(f"\n  Per-site summary:")
    print(f"  {'Site':<20} {'N':>4}  {'O-bal':>7}  {'H-bal':>7}  {'Gap':>7}")
    for site in sites:
        n_site = int(np.sum(sites_arr == site))
        o_s = np.mean(o_site_bals[site]) if o_site_bals[site] else 0
        h_s = np.mean(h_site_bals[site]) if h_site_bals[site] else 0
        print(f"  {site:<20} {n_site:>4}  {o_s:>7.2f}  {h_s:>7.2f}  {o_s-h_s:>+7.2f}")
    
    total_time = time.time() - t_start
    print(f"\n  Total time: {total_time:.1f}s")
    
    results = {
        "n_subjects": n, "n_asd": n_asd, "n_td": n_td,
        "n_sites": len(sites), "n_seeds": n_seeds,
        "o_ssm_bal_mean": float(o_mean), "o_ssm_bal_std": float(o_std),
        "h_ssm_bal_mean": float(h_mean), "h_ssm_bal_std": float(h_std),
        "gap_pp": float(gap),
        "o_bals": [float(x) for x in o_bals],
        "h_bals": [float(x) for x in h_bals],
        "per_site": {
            site: {
                "n": int(np.sum(sites_arr == site)),
                "o_mean": float(np.mean(o_site_bals[site])) if o_site_bals[site] else None,
                "h_mean": float(np.mean(h_site_bals[site])) if h_site_bals[site] else None,
            }
            for site in sites
        },
        "total_time_s": total_time,
    }
    return results


def main():
    repo = Path(__file__).resolve().parent.parent.parent
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--manifest", type=str, default="")
    args = parser.parse_args()
    
    if args.manifest:
        manifest_path = Path(args.manifest)
    else:
        for p in [
            repo / "artifacts" / "research" / "brain_ossm_full_cohort" / "abide_roi_manifest.tsv",
            repo / "artifacts" / "research" / "brain_ossm_local_site_balanced" / "abide_roi_manifest.tsv",
        ]:
            if p.exists():
                manifest_path = p
                break
        else:
            print("FATAL: No manifest found", file=sys.stderr)
            sys.exit(1)
    
    results = run_benchmark(manifest_path, n_seeds=args.seeds)
    
    out_path = repo / "experiments" / "non_assoc_connectomics" / "abide_full_cohort_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {out_path.relative_to(repo)}")


if __name__ == "__main__":
    main()
