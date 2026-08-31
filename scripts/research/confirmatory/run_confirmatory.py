#!/usr/bin/env python3
"""
C4 — Confirmatory runner (Rfam OctTree lane).

Executes the C2 task freeze over the C1 dataset freeze:
  arms A (flip, precedent), B (balance-preserving swap, primary),
  NEG (random Dyck negative control), across L grid, 20 paired seeds,
  6 models (CountBaseline, RealTree-8, CliffTree-8, LearnedBilinTree,
  OctTree-8, GRU-8).

Discipline (frozen):
  - verify_golden() before any training (fail closed)
  - val monitors training (train_one); test is evaluated exactly once,
    on the final model, per model x seed x L x arm
  - paired seeds: data sampling driven by s_i = 2026080900 + i; model init
    torch seed derived per (seed, L, arm) identically for every model
  - hyperparameters frozen from the exploratory run: epochs 50, lr 1e-2,
    batch 64 (no search)
  - every output JSON carries its own sha256 sidecar

Usage: run_confirmatory.py --seed-idx I --Ls 32 64 128 256 512 --outdir OUT
"""

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from corruptions import (  # noqa: E402
    flip_corrupt, swap_corrupt, gen_dyck_word, has_qualifying_pair,
    verify_golden, verify_uniform_dyck,
)
from controls import GatedTreeClassifier, CountBaseline  # noqa: E402
from ossm_dyck_scaling import GRUClassifier, count_params, train_one  # noqa: E402

FASTA = Path("/workspace/sounio/datasets/rna_secondary_structure/rfam_structures.fasta")
FREEZE = HERE / "freeze"

SEEDS = [2026080900 + i for i in range(20)]
TRAIN_N, VAL_N, TEST_N = 16384, 2048, 4096
BRACKET_MAP = {"(": 1, ")": 2, ".": 0}
EPOCHS, LR, BATCH = 50, 1e-2, 64


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_freeze():
    """id -> split, restricted to eligible records."""
    assign = {}
    with gzip.open(FREEZE / "records.tsv.gz", "rt") as f:
        next(f)
        for line in f:
            rid, fam, clan, seq_len, eligible, split, _ = line.rstrip("\n").split("\t")
            if eligible == "1":
                assign[rid] = split
    return assign


def load_structures(assign):
    """Parse the frozen fasta; keep eligible records with their split."""
    pools = {"train": [], "val": [], "test": []}
    with open(FASTA) as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith(">"):
            rid = lines[i][1:].split()[0]
            if i + 2 < len(lines):
                ss = lines[i + 2].strip()
                split = assign.get(rid)
                if split in pools:
                    tokens = np.array([BRACKET_MAP[c] for c in ss], dtype=np.int64)
                    pools[split].append(tokens)
            i += 3
        else:
            i += 1
    return pools


def derive_seed(*parts) -> int:
    h = hashlib.sha256("::".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def _neg_dyck(n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform Dyck word conditioned on admitting a swap (amendment
    2026-08-09, pre-results). Rejection-samples until the word has a
    qualifying swappable pair, so BOTH NEG classes have support exactly on
    swap-admissible words; negatives then always take the swap path. Under
    the uniform sampler the two NEG classes are identical in distribution
    by construction (see gen_dyck_word docstring)."""
    if n_pairs < 2:
        raise ValueError(f"NEG needs n_pairs >= 2, got {n_pairs}")
    t = gen_dyck_word(n_pairs, rng)
    while not has_qualifying_pair(t):
        t = gen_dyck_word(n_pairs, rng)
    return t


def sample_arm(pool, arm, L, n_samples, seed):
    """Build (tokens, labels) for one arm from a split pool."""
    rng = np.random.default_rng(seed)
    suitable = [t for t in pool if L // 4 <= len(t) <= L]
    if not suitable:
        raise ValueError(f"empty pool for L={L}")
    n_valid = n_samples // 2
    replace = len(suitable) < n_valid
    idx_valid = rng.choice(len(suitable), n_valid, replace=replace)
    idx_neg = rng.choice(len(suitable), n_valid, replace=True)

    tokens = np.zeros((n_samples, L), dtype=np.int64)
    labels = np.zeros(n_samples, dtype=np.int64)

    for j, i in enumerate(idx_valid):
        t = suitable[i]
        if arm == "NEG":
            n_pairs = max(1, min(len(t), L) // 2)
            t = _neg_dyck(n_pairs, rng)
        tokens[j, : min(len(t), L)] = t[:L]
        labels[j] = 1

    for k, i in enumerate(idx_neg):
        t = suitable[i]
        if arm == "NEG":
            n_pairs = max(1, min(len(t), L) // 2)
            t = _neg_dyck(n_pairs, rng)
        corrupt = flip_corrupt(t, rng) if arm == "A" else swap_corrupt(t, rng)
        tokens[n_valid + k, : min(len(corrupt), L)] = corrupt[:L]
        labels[n_valid + k] = 0

    perm = rng.permutation(n_samples)
    return tokens[perm], labels[perm]


def build_models(vocab=3, n_classes=2):
    return {
        "CountBaseline": CountBaseline(n_classes),
        "RealTree-8": GatedTreeClassifier(vocab, 8, n_classes, product="real"),
        "CliffTree-8": GatedTreeClassifier(vocab, 8, n_classes, product="cliff"),
        "LearnedBilinTree": GatedTreeClassifier(vocab, 8, n_classes, product="learned"),
        "OctTree-8": GatedTreeClassifier(vocab, 8, n_classes, product="oct"),
        "GRU-8": GRUClassifier(vocab, 8, n_classes),
    }


def test_accuracy(model, tokens, labels, device="cpu", batch=512):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(tokens), batch):
            bt = tokens[i : i + batch].to(device)
            logits = model(bt)
            correct += int((logits.argmax(-1).cpu() == labels[i : i + batch]).sum())
    model.train()
    return correct / len(tokens)


def run_L(pool_bundle, arm, L, seed, outdir):
    rng_data = derive_seed("data", arm, L, seed)
    pools = pool_bundle
    tr_tok, tr_lab = sample_arm(pools["train"], arm, L, TRAIN_N, rng_data)
    va_tok, va_lab = sample_arm(pools["val"], arm, L, VAL_N, derive_seed("val", arm, L, seed))
    te_tok, te_lab = sample_arm(pools["test"], arm, L, TEST_N, derive_seed("test", arm, L, seed))

    tr_t = torch.from_numpy(tr_tok)
    tr_l = torch.from_numpy(tr_lab)
    va_t = torch.from_numpy(va_tok)
    va_l = torch.from_numpy(va_lab)
    te_t = torch.from_numpy(te_tok)
    te_l = torch.from_numpy(te_lab)

    results = {}
    for name, model in build_models().items():
        torch.manual_seed(derive_seed("init", arm, L, seed))  # paired init
        model = model.to("cpu")
        t0 = time.time()
        hist = train_one(model, tr_t, tr_l, va_t, va_l,
                         epochs=EPOCHS, lr=LR, batch_size=BATCH,
                         device="cpu", name=f"{arm}/L{L}/{name}/s{seed}")
        acc = test_accuracy(model, te_t, te_l)
        results[name] = {
            "params": count_params(model),
            "val_acc_final": hist["test_acc"][-1],
            "val_acc_best": max(hist["test_acc"]),
            "test_acc_final": acc,
            "wall_sec": round(time.time() - t0, 1),
        }
        print(f"  [{arm}/L={L}/s{seed}] {name:<18} val={results[name]['val_acc_final']:.3f} "
              f"test={acc:.3f} ({results[name]['wall_sec']:.0f}s)", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-idx", type=int, required=True)
    ap.add_argument("--Ls", type=int, nargs="+", default=[64, 128, 256, 512])
    ap.add_argument("--arms", nargs="+", default=["A", "B", "NEG"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    seed = SEEDS[args.seed_idx]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    verify_golden()
    verify_uniform_dyck()
    print(f"golden vectors + uniform-Dyck self-test OK; seed_idx={args.seed_idx} seed={seed}", flush=True)

    assign = load_freeze()
    pools = load_structures(assign)
    print({k: len(v) for k, v in pools.items()}, flush=True)

    for L in args.Ls:
        doc = {
            "seed": seed,
            "seed_idx": args.seed_idx,
            "L": L,
            "freeze": {
                "manifest_sha256": "50668b60646b02378475e343a74b76d9c5f4a0e2de51433f6ff68c8d600acb18",
                "golden_sha256": "00c4380b30b06080e1d993063562610e397b3813c658e01d68e58ef33932bf77",
            },
            "protocol": {
                "train_n": TRAIN_N, "val_n": VAL_N, "test_n": TEST_N,
                "epochs": EPOCHS, "lr": LR, "batch": BATCH,
                "test_policy": "final model, single evaluation",
            },
            "arms": {},
        }
        for arm in args.arms:
            print(f"=== arm {arm} L={L} seed={seed} ===", flush=True)
            doc["arms"][arm] = run_L(pools, arm, L, seed, outdir)
        raw = json.dumps(doc, indent=2) + "\n"
        out = outdir / f"seed{args.seed_idx:02d}_L{L}.json"
        out.write_text(raw)
        (outdir / f"seed{args.seed_idx:02d}_L{L}.sha256").write_text(
            f"{sha(raw.encode())}  {out.name}\n"
        )
        print(f"wrote {out} sha256={sha(raw.encode())[:16]}…", flush=True)


if __name__ == "__main__":
    main()
