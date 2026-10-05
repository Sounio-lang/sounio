#!/usr/bin/env python3
"""Mercyful Learning — Suffering-Aware neural Network (SAN) at larger scale:
SAN-ResNet-50 (bottleneck), SAN-ViT-large (contract scale), and SAN-GPT
(decoder-only transformer language model) on real data.

Companion artifact to
  docs/research/suffering_aware_large_architecture_spec_2026-07-31.md

The deep-network SAN (scripts/research/suffering_aware_deep_architecture.py,
clauses D1..D9) established the architecture on ResNet-18 and ViT-small on
CIFAR-10. This harness scales the SAME architecture class one step further
along both axes the parent spec scoped out:

  * SAN-ResNet-50: a CIFAR-variant ResNet-50 — bottleneck blocks
    (1x1 reduce, 3x3, 1x1 expand), config (3,4,6,3), stage widths
    256/512/1024/2048 (~25M parameters) — with a per-stage suffering-aware
    exit head (global average pool + linear), exactly as in the parent.
  * SAN-ViT-large: the largest ViT the CPU contract budget affords at
    ViT-large proportions: patch 4x4 -> 64 tokens + CLS, d=384, 12 blocks,
    6 heads (head dim 64), MLP ratio 4 (~22M parameters) — with a per-block
    exit head on the CLS token.
  * SAN-GPT: a decoder-only transformer language model (d=384, 10 blocks,
    6 heads, causal masked attention, learned token+position embeddings,
    vocab V=2000 over a real text corpus — the repository's own
    docs/research/*.md, no external fetch) trained on next-token prediction
    (T=64). Per-block exit heads score the LAST G=4 positions; the gate
    confidence is the mean max-prob over those G positions. The scored task
    (accuracy and harm) is next-token prediction on the last G positions of
    each sequence, declared up front; the final head still supervises ALL
    positions during training (standard LM objective).

All three families are compared against the STANDARD architectures: the
identical trunks trained with a fixed epoch budget (Dense), and the
identical trunks with SAN's stop rule but no suffering-aware layers
(EarlyStop). All runs within a family share one trunk init, one data order,
one seed.

Machine suffering is metered exactly as in the parent line: analytic FLOPs
of the executed path (conv/linear MACs x2, each attention token-mixing
matmul 2*T^2*d, x3 in training — the fixed backward = 2x forward accounting
convention), gated-off stages/blocks charging exactly 0. Embedding lookups,
causal masking, LayerNorm/BatchNorm, activations, softmax, residual adds
and pooling are unmetered — a stated convention, identical for every
architecture and every accounting path.

Patient suffering is the mean harm of the current model's predictions on
the held-out cohort under an asymmetric harm structure (synthetic cost
structure over real labels; no clinical claim):
  * CIFAR families (ResNet-50, ViT-large): the parent line's 10-class harm
    matrix — class 9 ("truck") is the hazard class of a screening pipeline;
    missed hazard 5, false hazard 2, other confusion 1.
  * GPT family: hazard tokens are the negation tokens of the corpus
    ({no, not, never, without} ∩ vocab) — clinically, missing a negation
    flips the meaning of a statement (missed hazard, cost 5); inserting one
    is an unnecessary intervention (cost 2); any other token mismatch 1.

Certificates (contract clauses L1..L9, mirroring D1..D9):
  L1  metering conservation at larger scale (all families): gated-off
      stages/blocks charge exactly 0; metered FLOPs == an independent
      manual accounting of the executed path; metered < gates-open whenever
      an exit fires; eval-mode prefix invariance (bounded logit deviation,
      argmax exactly equal)
  L2  feasibility at larger scale: SAN reaches val acc >= TAU within budget
      in every family
  L3  anti-Goodhart soundness at larger scale: feasible-only selection on a
      101-point compassion-weight grid over pools containing a zero-cost
      abstainer and an under-trained probe; all-infeasible pool -> loud
      NO_FEASIBLE
  L4  necessary/gratuitous separation: SAN gratuitous machine suffering is
      exactly 0 (all families); fixed-budget baselines accrue > 0
  L5  suffering bounds: SAN total machine suffering strictly below the
      fixed-budget dense baseline and <= the EarlyStop scheduler baseline;
      SAN integrated patient harm <= every baseline's
  L6  exits are real at larger scale: held-out exit fraction at t* > 0.10
      in every family
  L7  patient channel first-class: harm structures genuinely asymmetric
      (off-diagonal max >= 3x min); SAN peak patient harm <= same-init
      baselines' peaks
  L8  anti-shortcut at larger scale: a spurious-feature probe beats TAU on
      TRAIN yet fails held-out (CIFAR: corner-patch feature; GPT: leaked
      label token at a fixed position); the gate rejects it at every
      compassion weight
  L9  cross-architecture scalability certificate: a sweep over bottleneck
      ResNet depths (50/101), ViT-large depths, and GPT depths verifies,
      forward-only, at every scale: metered == manual exactly, metered <
      gates-open when exits fire, prefix argmax agreement, and exit-head
      overhead < 5% of the gates-open forward

Real data (CIFAR-10 images; real repository-documentation text corpus), real
architectures (ResNet-50, ViT, GPT). The harm structures are synthetic cost
structures over real labels: this benchmark makes no clinical claim and is
not medical guidance. The machine channel is an operational
computational-burden proxy; no_consciousness_claim is made or needed.

Run: .venv/bin/python scripts/research/suffering_aware_large_architecture.py
Requires: torch (CPU) + numpy from the repo .venv, CIFAR-10 at
  datasets/cifar-10-batches-py (fetch: curl -L
  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar xz -C datasets)
Env overrides: SAN_LARGE_SMOKE=1 (tiny fast mechanics check),
  SAN_LARGE_ONLY=resnet50|vitlarge|gpt|sweep (run a subset),
  SAN_LARGE_THREADS=n, SAN_LARGE_TAU_*/SAN_LARGE_DELTA_* (calibration).
"""

import os
import pickle
import re
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- determinism / config --------------------------------------
SEED = int(os.environ.get("SAN_LARGE_SEED", "17"))
torch.manual_seed(SEED)
np.random.seed(SEED)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get(
    "SAN_LARGE_DATA", os.path.join(REPO_ROOT, "datasets", "cifar-10-batches-py"))
CORPUS_GLOB_DIR = os.path.join(REPO_ROOT, "docs", "research")
SMOKE = os.environ.get("SAN_LARGE_SMOKE", "") == "1"
ONLY = os.environ.get("SAN_LARGE_ONLY", "").strip().lower()
THREADS = int(os.environ.get("SAN_LARGE_THREADS", "48"))
torch.set_num_threads(THREADS)

# Device selection. Slurm gpu-orangefs workers see CUDA via /host-nvidia;
# default is cuda if available, else cpu. Explicit SAN_LARGE_DEVICE=cpu
# forces CPU for reproducibility checks.
DEVICE = torch.device(
    os.environ.get("SAN_LARGE_DEVICE",
                   "cuda" if torch.cuda.is_available() else "cpu"))

DATASET = os.environ.get("SAN_LARGE_DATASET", "cifar10").strip().lower()
if DATASET not in ("cifar10", "cifar100"):
    raise ValueError(f"SAN_LARGE_DATASET must be cifar10 or cifar100, got {DATASET!r}")
N_CLASS = 100 if DATASET == "cifar100" else 10
# Gates-closed warmup, per family (declared in spec section 6.2): heads
# train (as detached probes, or from epoch 1 for trunk-coupled aux heads)
# while the gates stay closed; SAN feasibility counts only gates-active
# epochs. SAN_LARGE_WARMUP overrides all three if set.
_W_ENV = os.environ.get("SAN_LARGE_WARMUP", "")
WARMUP_RESNET = int(os.environ.get("SAN_LARGE_WARMUP_RESNET", _W_ENV or "2"))
WARMUP_VIT = int(os.environ.get("SAN_LARGE_WARMUP_VIT", _W_ENV or "4"))
WARMUP_GPT = int(os.environ.get("SAN_LARGE_WARMUP_GPT", _W_ENV or "1"))
# Probe-head mode (all families): exit heads train as probes on DETACHED
# trunk features (no gradient into the trunk) — the declared convention of
# this spec (spec section 3). Rationale, measured on the canonical instance:
# aux gradients INTO the trunk (i) dilute the LM final-head objective
# (san acc@1 0.099 vs 0.114 detached vs 0.120 plain, same seed), and (ii)
# shift the vision trunk's early harm profile away from the shared epoch-0
# exposure (vit epoch-0 harm 1.279 vs 1.200), structurally breaking L5/L7.
# With detached probe heads the SAN trunk is gradient-identical to the
# plain trunk: early-exposure sharing (L7) holds by construction, and the
# gate's savings are what remain.
DETACH_AUX = os.environ.get("SAN_LARGE_DETACH_AUX", "1") == "1"
# Per-family probe-head wiring (declared in spec section 3, measured in the
# calibration history): ViT and GPT exit heads train as probes on DETACHED
# features (their early features are linearly separable enough that probe
# heads are good by the gates-open epoch, and trunk-gradient isolation
# preserves the shared epoch-0 exposure for L7); ResNet-50 keeps the parent
# line's deep supervision INTO the trunk (measured: detached stage heads on
# 2-3-epoch conv features gate at ~0.23 acc vs 0.34 with trunk gradients —
# deep supervision is what makes conv-stage heads viable at all).
DETACH_RESNET = os.environ.get("SAN_LARGE_DETACH_RESNET", "0") == "1"
DETACH_VIT = os.environ.get("SAN_LARGE_DETACH_VIT", "1") == "1"
# Warmup-aux: exit heads TRAIN (as detached probes) from epoch 0 while the
# gates stay closed for WARMUP_EPOCHS. Motivation (measured, spec
# calibration history): with the parent line's heads-train-only-after-
# warmup schedule, 1-2-epoch-old heads are overconfident — the gate fires
# on them (52% of the ResNet-50 cohort at epoch 1) and gated accuracy/harm
# collapses below the plain trunk's, delaying SAN's t* past EarlyStop's
# and breaking L5/L7. With detached heads the warmup-aux gradient cannot
# reach the trunk, so epoch-0 exposure stays shared with the baselines.
WARMUP_AUX = os.environ.get("SAN_LARGE_WARMUP_AUX", "1") == "1"
AUX_W = float(os.environ.get("SAN_LARGE_AUXW", "1.0"))  # probe-head loss weight
E_PER_FLOP = 4e-12      # J/FLOP, same convention as the machine-channel benchmark
LR = 1e-3
BATCH = int(os.environ.get("SAN_LARGE_BATCH", "128"))
GRAD_ACCUM = int(os.environ.get("SAN_LARGE_GRAD_ACCUM", "1"))
GRAD_ANALYSIS = os.environ.get("SAN_LARGE_GRAD_ANALYSIS", "0") == "1"

# GPT configuration (larger transformer leg).
GPT_DIM, GPT_HEADS = 384, 6
GPT_DEPTH = 10
GPT_T = 64              # sequence length
GPT_G = 4               # scored (last) positions for exits/gating/scoring
GPT_VOCAB = 2000
GPT_MLP_RATIO = 4
HAZARD_WORDS = ("no", "not", "never", "without")

if SMOKE:
    N_TRAIN, N_VAL = 512, 256
    EPOCHS_RESNET, EPOCHS_VIT, EPOCHS_GPT = 3, 3, 3
    TAU_RESNET, TAU_VIT, TAU_GPT = 0.10, 0.10, 0.05
    DELTA_RESNET, DELTA_VIT, DELTA_GPT = 0.5, 0.4, 0.3
    RESNET_BLOCKS, RESNET_WIDTH = (1, 1, 1, 1), 16
    VIT_DEPTH, VIT_DIM, VIT_HEADS = 2, 64, 4
    GPT_DIM, GPT_DEPTH, GPT_HEADS, GPT_T, GPT_VOCAB = 64, 2, 4, 32, 500
    GPT_G = 2
    GPT_N_TRAIN, GPT_N_VAL = 256, 128
    WARMUP_RESNET = WARMUP_VIT = WARMUP_GPT = 1
    BATCH = 64
else:
    N_TRAIN, N_VAL = 4000, 1000        # stratified subsets of CIFAR-10 train/test
    EPOCHS_RESNET, EPOCHS_VIT, EPOCHS_GPT = 8, 10, 10
    # Declared anti-Goodhart targets and per-family exit thresholds — declared
    # architecture constants, per family, exactly as in the parent line.
    # CIFAR-100 is a harder classification problem; default feasibility targets
    # are lower unless the caller overrides them.
    if DATASET == "cifar100":
        _DEF_TAU_RESNET, _DEF_TAU_VIT, _DEF_TAU_GPT = "0.20", "0.15", "0.10"
        _DEF_DELTA_RESNET, _DEF_DELTA_VIT, _DEF_DELTA_GPT = "0.55", "0.45", "0.31"
    else:
        _DEF_TAU_RESNET, _DEF_TAU_VIT, _DEF_TAU_GPT = "0.34", "0.251", "0.165"
        _DEF_DELTA_RESNET, _DEF_DELTA_VIT, _DEF_DELTA_GPT = "0.55", "0.45", "0.31"
    _N_TRAIN = int(os.environ.get("SAN_LARGE_N_TRAIN", "4000"))
    _N_VAL = int(os.environ.get("SAN_LARGE_N_VAL", "1000"))
    N_TRAIN, N_VAL = _N_TRAIN, _N_VAL
    _EPOCHS_RESNET = int(os.environ.get("SAN_LARGE_EPOCHS_RESNET", "8"))
    _EPOCHS_VIT = int(os.environ.get("SAN_LARGE_EPOCHS_VIT", "10"))
    _EPOCHS_GPT = int(os.environ.get("SAN_LARGE_EPOCHS_GPT", "10"))
    EPOCHS_RESNET, EPOCHS_VIT, EPOCHS_GPT = _EPOCHS_RESNET, _EPOCHS_VIT, _EPOCHS_GPT
    TAU_RESNET = float(os.environ.get("SAN_LARGE_TAU_RESNET", _DEF_TAU_RESNET))
    TAU_VIT = float(os.environ.get("SAN_LARGE_TAU_VIT", _DEF_TAU_VIT))
    TAU_GPT = float(os.environ.get("SAN_LARGE_TAU_GPT", _DEF_TAU_GPT))
    DELTA_RESNET = float(os.environ.get("SAN_LARGE_DELTA_RESNET", _DEF_DELTA_RESNET))
    DELTA_VIT = float(os.environ.get("SAN_LARGE_DELTA_VIT", _DEF_DELTA_VIT))
    DELTA_GPT = float(os.environ.get("SAN_LARGE_DELTA_GPT", _DEF_DELTA_GPT))
    RESNET_BLOCKS, RESNET_WIDTH = (3, 4, 6, 3), 64   # CIFAR ResNet-50 (bottleneck)
    VIT_DEPTH, VIT_DIM, VIT_HEADS = 12, 384, 6       # ViT-large (contract scale)
    GPT_N_TRAIN, GPT_N_VAL = 3072, 768

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

# Asymmetric harm matrix over the CIFAR classes (synthetic cost structure
# over real labels): the last class is the hazard class of a screening
# pipeline. A missed hazard (true hazard, predicted other) costs 5 — the
# expensive error the gate exists to block; a false hazard costs 2
# (unnecessary intervention); any other confusion costs 1.
def build_harm():
    hazard = N_CLASS - 1
    h = torch.ones(N_CLASS, N_CLASS)
    h.fill_diagonal_(0.0)
    h[hazard, :hazard] = 5.0     # missed hazard
    h[:hazard, hazard] = 2.0     # false hazard -> unnecessary intervention
    return h

HARM = build_harm()


def harm_of(pred, y):
    """Mean harm of predictions pred against truth y (synthetic matrix)."""
    return float(HARM[y, pred].mean().item())


# ---------------- CIFAR-10 loading (no torchvision) --------------------------
def load_cifar10():
    """Stratified deterministic subsets: N_TRAIN train (400/class) from the
    train batches, N_VAL (100/class) from the test batch. Normalized with
    the standard CIFAR-10 channel statistics."""
    def read_batch(name):
        with open(os.path.join(DATA_DIR, name), "rb") as f:
            d = pickle.load(f, encoding="latin1")
        x = d["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y = np.array(d["labels"], dtype=np.int64)
        return x, y

    xs, ys = [], []
    for i in range(1, 6):
        x, y = read_batch(f"data_batch_{i}")
        xs.append(x)
        ys.append(y)
    xtr_all, ytr_all = np.concatenate(xs), np.concatenate(ys)
    xte_all, yte_all = read_batch("test_batch")

    per_train, per_val = N_TRAIN // N_CLASS, N_VAL // N_CLASS
    tr_idx = np.concatenate([
        np.where(ytr_all == c)[0][:per_train] for c in range(N_CLASS)])
    va_idx = np.concatenate([
        np.where(yte_all == c)[0][:per_val] for c in range(N_CLASS)])
    rng = np.random.RandomState(SEED)
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    xtr = (xtr_all[tr_idx] - CIFAR_MEAN[None, :, None, None]) / CIFAR_STD[None, :, None, None]
    xva = (xte_all[va_idx] - CIFAR_MEAN[None, :, None, None]) / CIFAR_STD[None, :, None, None]
    return (torch.from_numpy(xtr).to(DEVICE),
            torch.from_numpy(ytr_all[tr_idx]).to(DEVICE),
            torch.from_numpy(xva).to(DEVICE),
            torch.from_numpy(yte_all[va_idx]).to(DEVICE))


def load_cifar100():
    """Stratified deterministic subsets of CIFAR-100 fine labels.
    N_TRAIN and N_VAL are divided evenly across the 100 fine classes.
    Normalized with the standard CIFAR channel statistics."""
    def read_batch(name):
        with open(os.path.join(DATA_DIR, name), "rb") as f:
            d = pickle.load(f, encoding="bytes")
        x = d[b"data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        y = np.array(d[b"fine_labels"], dtype=np.int64)
        return x, y

    xtr_all, ytr_all = read_batch("train")
    xte_all, yte_all = read_batch("test")

    per_train, per_val = N_TRAIN // N_CLASS, N_VAL // N_CLASS
    tr_idx = np.concatenate([
        np.where(ytr_all == c)[0][:per_train] for c in range(N_CLASS)])
    va_idx = np.concatenate([
        np.where(yte_all == c)[0][:per_val] for c in range(N_CLASS)])
    rng = np.random.RandomState(SEED)
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    xtr = (xtr_all[tr_idx] - CIFAR_MEAN[None, :, None, None]) / CIFAR_STD[None, :, None, None]
    xva = (xte_all[va_idx] - CIFAR_MEAN[None, :, None, None]) / CIFAR_STD[None, :, None, None]
    return (torch.from_numpy(xtr).to(DEVICE),
            torch.from_numpy(ytr_all[tr_idx]).to(DEVICE),
            torch.from_numpy(xva).to(DEVICE),
            torch.from_numpy(yte_all[va_idx]).to(DEVICE))


# ---------------- GPT corpus (real text: repository docs) ---------------------
def load_corpus():
    """Real text corpus: the repository's own docs/research/*.md files
    (sorted glob, deterministic for a fixed checkout). A PINNED snapshot
    (token ids + vocab + hazard ids) is materialized at
    SAN_LARGE_CORPUS (default artifacts/san_large/corpus_snapshot.npz):
    once written, every leg loads the snapshot instead of re-globbing, so
    cross-leg comparisons and the gate are reproducible against a fixed
    corpus even while other lanes edit docs/research (spec section 6.1
    corpus-snapshot caveat). Word-level tokenization (lowercase [a-z']+);
    vocab = UNK + the GPT_VOCAB-1 most frequent words. Sequences of length
    GPT_T+1 -> (input, target) next-token pairs. Train/val sequences are
    sampled (seeded) from disjoint token ranges (first 80% / last 20% of
    the corpus). Returns (x_tr, y_tr, x_va, y_va, hazard_ids, vocab_size,
    stoi, itos)."""
    import glob
    snap_path = os.environ.get(
        "SAN_LARGE_CORPUS",
        os.path.join(REPO_ROOT, "artifacts", "san_large",
                     f"corpus_snapshot_v{GPT_VOCAB}.npz"))
    if os.path.exists(snap_path):
        snap = np.load(snap_path, allow_pickle=True)
        ids = snap["ids"]
        itos = snap["itos"].tolist()
    else:
        files = sorted(glob.glob(os.path.join(CORPUS_GLOB_DIR, "*.md")))
        if not files:
            raise RuntimeError(f"corpus empty: no *.md under {CORPUS_GLOB_DIR}")
        text = "\n".join(open(p, encoding="utf-8", errors="replace").read()
                         for p in files)
        words = re.findall(r"[a-z']+", text.lower())
        counts = Counter(words)
        top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:GPT_VOCAB - 1]
        itos = ["<unk>"] + [w for w, _ in top]
        stoi0 = {w: i for i, w in enumerate(itos)}
        ids = np.array([stoi0.get(w, 0) for w in words], dtype=np.int64)
        os.makedirs(os.path.dirname(snap_path), exist_ok=True)
        np.savez(snap_path, ids=ids, itos=np.array(itos, dtype=object))
        print(f"  corpus: pinned snapshot written to {snap_path} "
              f"({len(files)} files, {len(ids)} tokens)", flush=True)
    stoi = {w: i for i, w in enumerate(itos)}
    n_split = int(0.8 * len(ids))
    tr_tokens, va_tokens = ids[:n_split], ids[n_split:]
    rng = np.random.RandomState(SEED + 3)

    def sample(tokens, n_seq):
        hi = len(tokens) - GPT_T - 1
        starts = rng.randint(0, hi, size=n_seq)
        x = np.stack([tokens[s:s + GPT_T] for s in starts])
        y = np.stack([tokens[s + 1:s + GPT_T + 1] for s in starts])
        return torch.from_numpy(x), torch.from_numpy(y)

    x_tr, y_tr = sample(tr_tokens, GPT_N_TRAIN)
    x_va, y_va = sample(va_tokens, GPT_N_VAL)
    hazard_ids = torch.tensor(
        [stoi[w] for w in HAZARD_WORDS if w in stoi], dtype=torch.long).to(DEVICE)
    return x_tr.to(DEVICE), y_tr.to(DEVICE), x_va.to(DEVICE), y_va.to(DEVICE), hazard_ids, len(itos), stoi, itos


def build_harm_lm(vocab, hazard_ids):
    """Asymmetric token harm: missed hazard (true negation, predicted other)
    costs 5; false hazard (predicted negation, true other) costs 2; any
    other mismatch costs 1; correct 0."""
    h = torch.ones(vocab, vocab)
    h.fill_diagonal_(0.0)
    hz = torch.zeros(vocab, dtype=torch.bool)
    hz[hazard_ids] = True
    missed = hz.unsqueeze(1) & (~hz).unsqueeze(0)   # true hazard, pred other
    false = (~hz).unsqueeze(1) & hz.unsqueeze(0)    # pred hazard, true other
    h[missed] = 5.0    # missed hazard
    h[false] = 2.0     # false hazard -> unnecessary intervention
    return h


# ---------------- machine-suffering meter -----------------------------------
class MachineMeter:
    """Analytic FLOP accounting of the executed path:
      linear = 2*d_in*d_out per row; conv = 2*cin*cout*k*k*h_out*w_out per
      sample (MAC = 2 FLOPs); a training step charges forward + backward with
      backward = 2x forward (x3 total) — the fixed accounting convention of
      mercyful_machine_channel_benchmark.py. Elementwise ops (BN/LN, ReLU/
      GELU, residual adds, softmax, pooling, causal masking) and embedding
      lookups are unmetered: stated convention, identical for every
      architecture and accounting path."""

    def __init__(self):
        self.flops = 0

    def charge_linear(self, d_in, d_out, n_rows, backward=False):
        f = 2 * d_in * d_out * n_rows
        self.flops += 3 * f if backward else f

    def charge_conv(self, cin, cout, k, h_out, w_out, n, backward=False):
        f = 2 * cin * cout * k * k * h_out * w_out * n
        self.flops += 3 * f if backward else f

    @property
    def energy_joules(self):
        return self.flops * E_PER_FLOP


# ---------------- ResNet trunks (CIFAR variant, basic + bottleneck) ----------
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout))

    def forward(self, x, meter, backward):
        n, _, h, w = x.shape
        ho, wo = h // self.conv1.stride[0], w // self.conv1.stride[1]
        meter.charge_conv(self.conv1.in_channels, self.conv1.out_channels,
                          3, ho, wo, n, backward)
        out = torch.relu(self.bn1(self.conv1(x)))
        meter.charge_conv(self.conv2.in_channels, self.conv2.out_channels,
                          3, ho, wo, n, backward)
        out = self.bn2(self.conv2(out))
        if self.short is not None:
            meter.charge_conv(self.short[0].in_channels,
                              self.short[0].out_channels, 1, ho, wo, n, backward)
            out = out + self.short(x)
        else:
            out = out + x
        return torch.relu(out)


class BottleneckBlock(nn.Module):
    """ResNet v1 bottleneck: 1x1 reduce (cin->inner), 3x3 (stride), 1x1
    expand (inner->cout = 4*inner)."""

    def __init__(self, cin, inner, stride):
        super().__init__()
        cout = 4 * inner
        self.conv1 = nn.Conv2d(cin, inner, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner)
        self.conv2 = nn.Conv2d(inner, inner, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(inner)
        self.conv3 = nn.Conv2d(inner, cout, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout))

    def forward(self, x, meter, backward):
        n, _, h, w = x.shape
        s = self.conv2.stride[0]
        ho, wo = h // s, w // s
        meter.charge_conv(self.conv1.in_channels, self.conv1.out_channels,
                          1, h, w, n, backward)
        out = torch.relu(self.bn1(self.conv1(x)))
        meter.charge_conv(self.conv2.in_channels, self.conv2.out_channels,
                          3, ho, wo, n, backward)
        out = torch.relu(self.bn2(self.conv2(out)))
        meter.charge_conv(self.conv3.in_channels, self.conv3.out_channels,
                          1, ho, wo, n, backward)
        out = self.bn3(self.conv3(out))
        if self.short is not None:
            meter.charge_conv(self.short[0].in_channels,
                              self.short[0].out_channels, 1, ho, wo, n, backward)
            out = out + self.short(x)
        else:
            out = out + x
        return torch.relu(out)


class ResNetTrunk(nn.Module):
    """Stem + 4 stages. block_type='basic': channels (w,2w,4w,8w);
    block_type='bottleneck': inner widths (w,2w,4w,8w), stage outputs
    4x inner (ResNet-50/101 shape). Stage k produces the feature map at
    which exit head k sits."""

    def __init__(self, blocks=(3, 4, 6, 3), width=64, block_type="bottleneck"):
        super().__init__()
        self.block_type = block_type
        inner = (width, width * 2, width * 4, width * 8)
        self.channels = tuple(4 * c for c in inner) if block_type == "bottleneck" else inner
        self.stem = nn.Conv2d(3, inner[0], 3, 1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(inner[0])
        stages = []
        cin = inner[0]
        for k, (iw, cout, n_blocks) in enumerate(zip(inner, self.channels, blocks)):
            blocks_k = []
            for b in range(n_blocks):
                stride = 2 if (k > 0 and b == 0) else 1
                if block_type == "bottleneck":
                    blocks_k.append(BottleneckBlock(cin, iw, stride))
                else:
                    blocks_k.append(BasicBlock(cin, cout, stride))
                cin = cout
            stages.append(nn.ModuleList(blocks_k))
        self.stages = nn.ModuleList(stages)

    def run_stage(self, k, h, meter, backward):
        for blk in self.stages[k]:
            h = blk(h, meter, backward)
        return h

    def run_stem(self, x, meter, backward):
        n, _, h, w = x.shape
        meter.charge_conv(3, self.stem.out_channels, 3, h, w, n, backward)
        return torch.relu(self.stem_bn(self.stem(x)))


# ---------------- ViT trunk ---------------------------------------------------
class ViTBlock(nn.Module):
    def __init__(self, d, heads, mlp_ratio=4):
        super().__init__()
        self.d, self.heads = d, heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp1 = nn.Linear(d, mlp_ratio * d)
        self.mlp2 = nn.Linear(mlp_ratio * d, d)

    def forward(self, t, meter, backward):
        # t: (n, tokens, d)
        n, tok, d = t.shape
        h = self.ln1(t)
        meter.charge_linear(d, 3 * d, n * tok, backward)
        qkv = self.qkv(h).reshape(n, tok, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)   # (n, heads, tok, dh)
        # attention scores + weighted values: two tok x tok x d matmuls/sample
        meter.flops += (3 if backward else 1) * 2 * 2 * tok * tok * d * n
        a = torch.softmax(q @ k.transpose(-2, -1) / (d // self.heads) ** 0.5, dim=-1)
        o = (a @ v).transpose(1, 2).reshape(n, tok, d)
        meter.charge_linear(d, d, n * tok, backward)
        t = t + self.proj(o)
        h = self.ln2(t)
        meter.charge_linear(d, self.mlp1.out_features, n * tok, backward)
        h = F.gelu(self.mlp1(h))
        meter.charge_linear(self.mlp1.out_features, d, n * tok, backward)
        return t + self.mlp2(h)


class ViTTrunk(nn.Module):
    def __init__(self, depth=12, d=384, heads=6, patch=4, mlp_ratio=4):
        super().__init__()
        self.d, self.patch = d, patch
        self.embed = nn.Conv2d(3, d, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, (32 // patch) ** 2 + 1, d))
        self.blocks = nn.ModuleList([ViTBlock(d, heads, mlp_ratio) for _ in range(depth)])
        self.ln = nn.LayerNorm(d)

    def run_embed(self, x, meter, backward):
        n = x.shape[0]
        grid = 32 // self.patch
        meter.charge_conv(3, self.d, self.patch, grid, grid, n, backward)
        t = self.embed(x).flatten(2).transpose(1, 2)      # (n, tok, d)
        t = torch.cat([self.cls.expand(n, -1, -1), t], dim=1) + self.pos
        return t

    def run_block(self, k, t, meter, backward):
        return self.blocks[k](t, meter, backward)


# ---------------- GPT trunk (decoder-only transformer LM) ---------------------
class GPTBlock(nn.Module):
    def __init__(self, d, heads, mlp_ratio=GPT_MLP_RATIO):
        super().__init__()
        self.d, self.heads = d, heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp1 = nn.Linear(d, mlp_ratio * d)
        self.mlp2 = nn.Linear(mlp_ratio * d, d)

    def forward(self, t, meter, backward):
        # t: (n, T, d); causal masked self-attention (masking unmetered).
        n, tok, d = t.shape
        h = self.ln1(t)
        meter.charge_linear(d, 3 * d, n * tok, backward)
        qkv = self.qkv(h).reshape(n, tok, 3, self.heads, d // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        meter.flops += (3 if backward else 1) * 2 * 2 * tok * tok * d * n
        a = q @ k.transpose(-2, -1) / (d // self.heads) ** 0.5
        mask = torch.tril(torch.ones(tok, tok, dtype=torch.bool, device=t.device))
        a = a.masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, dim=-1)
        o = (a @ v).transpose(1, 2).reshape(n, tok, d)
        meter.charge_linear(d, d, n * tok, backward)
        t = t + self.proj(o)
        h = self.ln2(t)
        meter.charge_linear(d, self.mlp1.out_features, n * tok, backward)
        h = F.gelu(self.mlp1(h))
        meter.charge_linear(self.mlp1.out_features, d, n * tok, backward)
        return t + self.mlp2(h)


class GPTTrunk(nn.Module):
    def __init__(self, vocab, depth=GPT_DEPTH, d=GPT_DIM, heads=GPT_HEADS, t=GPT_T):
        super().__init__()
        self.d, self.t = d, t
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(t, d)
        self.blocks = nn.ModuleList([GPTBlock(d, heads) for _ in range(depth)])
        self.ln = nn.LayerNorm(d)

    def run_embed(self, x, meter, backward):
        # embedding lookups unmetered (stated convention)
        pos = torch.arange(x.shape[1], device=x.device)
        return self.tok_emb(x) + self.pos_emb(pos)[None, :, :]

    def run_block(self, k, t, meter, backward):
        return self.blocks[k](t, meter, backward)


# ---------------- SAN wrappers (exit gates + metering) -----------------------
class SufferingAwareResNet(nn.Module):
    """ResNet trunk (basic or bottleneck) with a per-stage exit head
    (GAP + linear). kind='dense'/'earlystop' never runs the exit heads."""

    def __init__(self, blocks, width, kind, block_type="bottleneck"):
        super().__init__()
        assert kind in ("san", "dense", "earlystop")
        self.kind = kind
        self.detach_aux = DETACH_RESNET
        self.warmup = WARMUP_RESNET
        self.trunk = ResNetTrunk(blocks, width, block_type)
        self.n_stages = len(self.trunk.stages)
        self.exit_heads = nn.ModuleList(
            [nn.Linear(c, N_CLASS) for c in self.trunk.channels])
        self.final_head = nn.Linear(self.trunk.channels[-1], N_CLASS)
        self.meter = MachineMeter()

    def _head_logits(self, k, h):
        return self.exit_heads[k](h.mean(dim=(2, 3)))

    def forward(self, x, train=False, use_exit_heads=True, train_aux=False):
        meter = self.meter
        n = x.shape[0]
        out_logits = x.new_zeros(n, N_CLASS)
        out_depth = torch.full((n,), self.n_stages, dtype=torch.long,
                               device=x.device)
        h = self.trunk.run_stem(x, meter, train)
        active = torch.arange(n, device=x.device)
        per_stage_active = []
        aux_records, final_record = [], None
        gated = use_exit_heads and self.kind == "san"
        aux_active = train and (gated or train_aux)
        for k in range(self.n_stages):
            if active.numel() == 0:
                break
            per_stage_active.append(int(active.numel()))
            h = self.trunk.run_stage(k, h, meter, train)
            if not (gated or aux_active):
                continue
            meter.charge_linear(self.trunk.channels[k], N_CLASS,
                                h.shape[0], train)
            head_in = h.mean(dim=(2, 3))
            if train and self.detach_aux:
                head_in = head_in.detach()   # probe head: no trunk gradient
            logits_k = self.exit_heads[k](head_in)
            if train:
                aux_records.append((active, logits_k))
            if not gated:
                continue
            conf = torch.softmax(logits_k.detach(), dim=1).max(dim=1).values
            leave = conf >= DELTA_RESNET
            if leave.any():
                idx = active[leave]
                out_logits[idx] = logits_k[leave]
                out_depth[idx] = k
                keep = ~leave
                active = active[keep]
                h = h[keep]
        n_final = int(active.numel())
        if n_final > 0:
            meter.charge_linear(self.trunk.channels[-1], N_CLASS, n_final, train)
            final_logits = self.final_head(h.mean(dim=(2, 3)))
            out_logits[active] = final_logits
            if train:
                final_record = (active, final_logits)
        return out_logits, out_depth, per_stage_active, n_final, aux_records, final_record

    def forward_dense(self, x):
        """Every gate forced open: every sample traverses every stage AND
        every exit head, then the final head (conservation reference)."""
        meter = MachineMeter()
        h = self.trunk.run_stem(x, meter, False)
        for k in range(self.n_stages):
            h = self.trunk.run_stage(k, h, meter, False)
            meter.charge_linear(self.trunk.channels[k], N_CLASS, x.shape[0])
        meter.charge_linear(self.trunk.channels[-1], N_CLASS, x.shape[0])
        return self.final_head(h.mean(dim=(2, 3))), meter


class SufferingAwareViT(nn.Module):
    """ViT trunk with a per-block exit head on the CLS token."""

    def __init__(self, depth, d, heads, kind):
        super().__init__()
        assert kind in ("san", "dense", "earlystop")
        self.kind = kind
        self.detach_aux = DETACH_VIT
        self.warmup = WARMUP_VIT
        self.trunk = ViTTrunk(depth, d, heads)
        self.depth = depth
        self.exit_heads = nn.ModuleList([nn.Linear(d, N_CLASS) for _ in range(depth)])
        self.final_head = nn.Linear(d, N_CLASS)
        self.meter = MachineMeter()

    def _cls(self, t):
        return self.trunk.ln(t)[:, 0]

    def forward(self, x, train=False, use_exit_heads=True, train_aux=False):
        meter = self.meter
        n = x.shape[0]
        out_logits = x.new_zeros(n, N_CLASS)
        out_depth = torch.full((n,), self.depth, dtype=torch.long,
                               device=x.device)
        t = self.trunk.run_embed(x, meter, train)
        active = torch.arange(n, device=x.device)
        per_block_active = []
        aux_records, final_record = [], None
        gated = use_exit_heads and self.kind == "san"
        aux_active = train and (gated or train_aux)
        for k in range(self.depth):
            if active.numel() == 0:
                break
            per_block_active.append(int(active.numel()))
            t = self.trunk.run_block(k, t, meter, train)
            if not (gated or aux_active):
                continue
            meter.charge_linear(self.trunk.d, N_CLASS, t.shape[0], train)
            head_in = self._cls(t)
            if train and self.detach_aux:
                head_in = head_in.detach()   # probe head: no trunk gradient
            logits_k = self.exit_heads[k](head_in)
            if train:
                aux_records.append((active, logits_k))
            if not gated:
                continue
            conf = torch.softmax(logits_k.detach(), dim=1).max(dim=1).values
            leave = conf >= DELTA_VIT
            if leave.any():
                idx = active[leave]
                out_logits[idx] = logits_k[leave]
                out_depth[idx] = k
                keep = ~leave
                active = active[keep]
                t = t[keep]
        n_final = int(active.numel())
        if n_final > 0:
            meter.charge_linear(self.trunk.d, N_CLASS, n_final, train)
            final_logits = self.final_head(self._cls(t))
            out_logits[active] = final_logits
            if train:
                final_record = (active, final_logits)
        return out_logits, out_depth, per_block_active, n_final, aux_records, final_record

    def forward_dense(self, x):
        meter = MachineMeter()
        t = self.trunk.run_embed(x, meter, False)
        for k in range(self.depth):
            t = self.trunk.run_block(k, t, meter, False)
            meter.charge_linear(self.trunk.d, N_CLASS, x.shape[0])
        meter.charge_linear(self.trunk.d, N_CLASS, x.shape[0])
        return self.final_head(self._cls(t)), meter


class SufferingAwareGPT(nn.Module):
    """GPT trunk with a per-block exit head over the last G positions.
    Exited sequences leave after block k; remaining blocks are gated off FOR
    THAT SEQUENCE and meter exactly 0. The final head supervises ALL
    positions during training (standard LM objective); scoring (accuracy and
    harm) is on the last G positions, declared up front."""

    def __init__(self, vocab, depth, d, heads, t, kind, g=GPT_G):
        super().__init__()
        assert kind in ("san", "dense", "earlystop")
        self.kind = kind
        self.detach_aux = DETACH_AUX
        self.warmup = WARMUP_GPT
        self.g = g
        self.vocab = vocab
        self.trunk = GPTTrunk(vocab, depth, d, heads, t)
        self.depth = depth
        self.exit_heads = nn.ModuleList([nn.Linear(d, vocab) for _ in range(depth)])
        self.final_head = nn.Linear(d, vocab)
        self.meter = MachineMeter()

    def _scored(self, t):
        """LN'd hidden states at the last G positions."""
        return self.trunk.ln(t)[:, -self.g:, :]

    def forward(self, x, train=False, use_exit_heads=True, train_aux=False):
        meter = self.meter
        n = x.shape[0]
        tok = x.shape[1]
        out_logits = torch.zeros(n, self.g, self.vocab, device=x.device)
        out_depth = torch.full((n,), self.depth, dtype=torch.long,
                               device=x.device)
        t = self.trunk.run_embed(x, meter, train)
        active = torch.arange(n, device=x.device)
        per_block_active = []
        aux_records, final_record = [], None
        gated = use_exit_heads and self.kind == "san"
        aux_active = train and (gated or train_aux)
        for k in range(self.depth):
            if active.numel() == 0:
                break
            per_block_active.append(int(active.numel()))
            t = self.trunk.run_block(k, t, meter, train)
            if not (gated or aux_active):
                continue
            h_g = self._scored(t)
            if train and self.detach_aux:
                # probe-head mode: heads learn from detached features; the
                # trunk's gradient comes from the final head only
                h_g = h_g.detach()
            meter.charge_linear(self.trunk.d, self.vocab,
                                h_g.shape[0] * self.g, train)
            logits_k = self.exit_heads[k](h_g)          # (a, G, V)
            if train:
                aux_records.append((active, logits_k))
            if not gated:
                continue
            conf = torch.softmax(logits_k.detach(), dim=-1).max(dim=-1).values
            leave = conf.mean(dim=1) >= DELTA_GPT       # mean max-prob over G
            if leave.any():
                idx = active[leave]
                out_logits[idx] = logits_k[leave]
                out_depth[idx] = k
                keep = ~leave
                active = active[keep]
                t = t[keep]
        n_final = int(active.numel())
        full_logits = None
        if n_final > 0:
            h = self.trunk.ln(t)
            if train:
                # final head supervises ALL positions (standard LM objective)
                meter.charge_linear(self.trunk.d, self.vocab,
                                    n_final * tok, train)
                full_logits = self.final_head(h)        # (a, T, V)
                out_logits[active] = full_logits[:, -self.g:, :]
            else:
                meter.charge_linear(self.trunk.d, self.vocab,
                                    n_final * self.g, train)
                out_logits[active] = self.final_head(h[:, -self.g:, :])
            final_record = (active, full_logits) if train else None
        return out_logits, out_depth, per_block_active, n_final, aux_records, final_record

    def forward_dense(self, x):
        meter = MachineMeter()
        t = self.trunk.run_embed(x, meter, False)
        for k in range(self.depth):
            t = self.trunk.run_block(k, t, meter, False)
            meter.charge_linear(self.trunk.d, self.vocab, x.shape[0] * self.g)
        meter.charge_linear(self.trunk.d, self.vocab, x.shape[0] * self.g)
        return self.final_head(self._scored(t)), meter


# ---------------- data schedule ------------------------------------------------
def batch_schedule(n_train, epochs):
    """One fixed data order for every run in the family (bit-reproducible,
    shared across architectures so epoch-0 cohort exposure is identical)."""
    rng = np.random.RandomState(SEED)
    return [rng.permutation(n_train) for _ in range(epochs)]


# ---------------- training loops ----------------------------------------------
CE = nn.CrossEntropyLoss()


def model_depth(model):
    return model.n_stages if hasattr(model, "n_stages") else model.depth


def train_run(model, x_tr, y_tr, x_va, y_va, epochs, tau, tag, is_lm=False):
    """One training run. Dense: fixed budget, no exit heads. EarlyStop:
    SAN's stop rule, no exit heads. SAN: suffering-aware layers +
    freeze-on-green. Identical optimizer, seed, data order, ledger shape.
    For the LM family, y holds full target sequences; training loss covers
    all positions (final head) resp. the last G positions (exit heads), and
    scoring is on the last G positions."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    schedule = batch_schedule(x_tr.shape[0], epochs)
    ledger = []
    t_star = None
    g = model.g if is_lm else 0
    # Gradient-analysis instrumentation: collect per-stage gradient norms
    # after each backward pass. Hooks are registered once per run and removed
    # at the end to avoid side effects on other runs.
    grad_hooks = []
    grad_stats = {f"stage_{k}": [] for k in range(model_depth(model))}
    grad_stats["final_head"] = []
    grad_stats["exit_heads"] = []

    def _make_hook(name):
        def hook(module, grad_input, grad_output):
            if grad_output and grad_output[0] is not None:
                norm = float(grad_output[0].detach().norm().item())
                grad_stats[name].append(norm)
        return hook

    if GRAD_ANALYSIS and model.kind == "san":
        # Trunk stages
        if hasattr(model, "trunk"):
            for k, stage in enumerate(model.trunk.stages):
                for blk in stage:
                    grad_hooks.append(blk.register_full_backward_hook(_make_hook(f"stage_{k}")))
        # Exit heads
        for k, head in enumerate(model.exit_heads):
            grad_hooks.append(head.register_full_backward_hook(_make_hook("exit_heads")))
        # Final head
        grad_hooks.append(model.final_head.register_full_backward_hook(_make_hook("final_head")))

    try:
        for epoch in range(epochs):
            t0 = time.time()
            model.train()
            model.meter = MachineMeter()
            warmup = epoch < getattr(model, "warmup", 1)
            use_exits = (model.kind == "san") and (not warmup)
            # Heads train from epoch 0 only when they cannot touch the trunk
            # (detached probes); trunk-coupled (aux) heads start at epoch 1 so
            # the shared epoch-0 exposure (L7) is preserved.
            detach = getattr(model, "detach_aux", False)
            train_aux = (model.kind == "san") and (
                use_exits or (WARMUP_AUX and (epoch >= 1 or detach)))
            nb = 0
            for b0 in range(0, x_tr.shape[0], BATCH):
                idx = schedule[epoch][b0:b0 + BATCH]
                xb, yb = x_tr[idx], y_tr[idx]
                micro = (xb.shape[0] == BATCH)
                _, _, _, _, aux_records, final_record = model(
                    xb, train=True, use_exit_heads=use_exits, train_aux=train_aux)
                if not is_lm:
                    losses = []
                    if final_record is not None:
                        f_idx, f_logits = final_record
                        losses.append(CE(f_logits, yb[f_idx]))
                    if aux_records:
                        losses.append(AUX_W * torch.stack(
                            [CE(a_logits, yb[a_idx])
                             for a_idx, a_logits in aux_records]).mean())
                    loss = sum(losses)
                else:
                    yg = yb[:, -g:]
                    losses = []
                    if final_record is not None:
                        f_idx, f_logits = final_record      # (a, T, V)
                        losses.append(CE(f_logits.reshape(-1, f_logits.shape[-1]),
                                         yb[f_idx].reshape(-1)))
                    if aux_records:
                        losses.append(AUX_W * torch.stack(
                            [CE(a_logits.reshape(-1, a_logits.shape[-1]),
                                yg[a_idx].reshape(-1))
                             for a_idx, a_logits in aux_records]).mean())
                    loss = sum(losses)
                if GRAD_ACCUM > 1:
                    loss = loss / GRAD_ACCUM
                if nb % GRAD_ACCUM == 0:
                    opt.zero_grad()
                loss.backward()
                if (nb + 1) % GRAD_ACCUM == 0 or not micro:
                    opt.step()
                nb += 1
            train_flops = model.meter.flops
            # held-out evaluation (forward only): the cohort-in-waiting
            model.eval()
            model.meter = MachineMeter()
            EVAL_BATCH = int(os.environ.get("SAN_LARGE_EVAL_BATCH", "512"))
            with torch.no_grad():
                vlogits_chunks = []
                vdepth_chunks = []
                for e0 in range(0, x_va.shape[0], EVAL_BATCH):
                    e1 = e0 + EVAL_BATCH
                    xb = x_va[e0:e1]
                    l, d, _, _, _, _ = model(xb, train=False, use_exit_heads=use_exits)
                    vlogits_chunks.append(l)
                    vdepth_chunks.append(d)
                vlogits = torch.cat(vlogits_chunks, dim=0)
                vdepth = torch.cat(vdepth_chunks, dim=0)
            eval_flops = model.meter.flops
            if not is_lm:
                pred = vlogits.argmax(dim=1)
                acc = float((pred == y_va).float().mean().item())
                harm = harm_of(pred, y_va)
            else:
                yg = y_va[:, -g:]
                pred = vlogits.argmax(dim=-1)               # (n, G)
                acc = float((pred == yg).float().mean().item())
                harm = float(HARM_LM[yg.reshape(-1), pred.reshape(-1)].mean().item())
            exit_frac = float((vdepth < model_depth(model)).float().mean().item())
            # Feasibility is a property of the DEPLOYED (gated) model: for SAN it
            # only counts once the gates are active (post-warmup). Counting an
            # ungated warmup epoch would freeze on a model that is never shipped.
            gates_active = use_exits or model.kind != "san"
            feasible = (acc >= tau) and gates_active
            if feasible and t_star is None:
                t_star = epoch
            ledger.append({
                "epoch": epoch, "flops": train_flops + eval_flops,
                "acc": acc, "harm": harm, "exit_frac": exit_frac,
                "feasible": feasible,
            })
            if GRAD_ANALYSIS and model.kind == "san":
                stage_norms = {k: float(np.mean(v)) if v else 0.0
                               for k, v in grad_stats.items() if k.startswith("stage_")}
                exit_norm = float(np.mean(grad_stats["exit_heads"])) if grad_stats["exit_heads"] else 0.0
                final_norm = float(np.mean(grad_stats["final_head"])) if grad_stats["final_head"] else 0.0
                print(f"    [{tag}] grad: stages={stage_norms} exit_heads={exit_norm:.4f} final={final_norm:.4f}", flush=True)
            print(f"    [{tag}] epoch={epoch} acc={acc:.4f} harm={harm:.3f} "
                  f"exit={exit_frac:.3f} flops={(train_flops + eval_flops) / 1e9:.2f}GF "
                  f"({time.time() - t0:.0f}s)", flush=True)
            if model.kind in ("san", "earlystop") and t_star is not None:
                break  # freeze-on-green: gratuitous suffering is exactly zero
    finally:
        for h in grad_hooks:
            h.remove()
    return model, ledger, t_star


def build_model(family, kind, vocab=None, blocks=None, width=None, depth=None,
                d=None, heads=None):
    """Fresh model with the shared family trunk init (identical for every
    kind within a family) and deterministic exit-head init."""
    torch.manual_seed(SEED)
    if family == "resnet50":
        model = SufferingAwareResNet(blocks or RESNET_BLOCKS,
                                     width or RESNET_WIDTH, kind,
                                     block_type="bottleneck")
    elif family == "vitlarge":
        model = SufferingAwareViT(depth or VIT_DEPTH, d or VIT_DIM,
                                  heads or VIT_HEADS, kind)
    else:
        model = SufferingAwareGPT(vocab or GPT_VOCAB, depth or GPT_DEPTH,
                                  d or GPT_DIM, heads or GPT_HEADS, GPT_T, kind)
    return model.to(DEVICE)


# ---------------- wall-time latency baseline ----------------------------------
def benchmark_latency(san_model, dense_model, x, is_lm=False, n_runs=100):
    """Measure real per-sample inference latency on the active device.

    Returns dict with ms/sample for SAN (gated early exits) and Dense
    (full forward). Uses torch.cuda.synchronize() when available and
    discards the first warmup run.
    """
    san_model.eval()
    dense_model.eval()
    do_sync = DEVICE.type == "cuda"

    def time_forward(fn, *args, **kwargs):
        # warmup
        with torch.no_grad():
            _ = fn(*args, **kwargs)
        if do_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = fn(*args, **kwargs)
            if do_sync:
                torch.cuda.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) / n_runs

    # SAN gated path; Dense full forward
    san_t = time_forward(san_model, x, train=False)
    dense_t = time_forward(dense_model, x, train=False)
    n = x.shape[0]
    return {
        "san_ms_per_sample": san_t / n * 1e3,
        "dense_ms_per_sample": dense_t / n * 1e3,
        "speedup": dense_t / san_t if san_t > 0 else float("inf"),
    }


# ---------------- suffering ledger --------------------------------------------
def suffering_summary(ledger):
    s_m = sum(e["flops"] for e in ledger)
    harms = [e["harm"] for e in ledger]
    t_star = next((e["epoch"] for e in ledger if e["feasible"]), None)
    if t_star is None:
        nec, grat = s_m, 0
    else:
        nec = sum(e["flops"] for e in ledger if e["epoch"] <= t_star)
        grat = sum(e["flops"] for e in ledger if e["epoch"] > t_star)
    return {"s_machine_flops": s_m, "s_machine_joules": s_m * E_PER_FLOP,
            "s_patient_int": sum(harms), "s_patient_peak": max(harms),
            "t_star": t_star, "necessary_flops": nec, "gratuitous_flops": grat}


# ---------------- anti-Goodhart gate -------------------------------------------
def gate_select(candidates, lam):
    """argmin of scalarized suffering over the FEASIBLE SET ONLY (spec
    theorem T2). An all-infeasible pool yields a loud NO_FEASIBLE."""
    feas = [c for c in candidates if c["feasible"]]
    if not feas:
        return "NO_FEASIBLE"
    return min(feas, key=lambda c: (1.0 - lam) * c["j_patient"]
               + lam * c["j_machine"])["name"]


# ---------------- L1 / L9 helpers ----------------------------------------------
def conservation_check(model, x_va, depth_units, is_lm=False):
    """Metering conservation for one trained (or sweep) model, eval mode:
    metered == manual independent accounting; metered < gates-open iff exits
    fired; exited logits == independently recomputed prefix (bounded dev,
    argmax exactly equal). Returns (ok, details-dict)."""
    model.eval()
    model.meter = MachineMeter()
    with torch.no_grad():
        vlogits_gated, vdepth, per_active, n_final, _, _ = model(x_va, train=False)
    gated_flops = model.meter.flops
    with torch.no_grad():
        vlogits_dense, dense_meter = model.forward_dense(x_va)
    n_exits = int((vdepth < depth_units).sum().item())

    # independent manual accounting of the executed path, re-derived from
    # module shapes and the recorded active counts (separate code path)
    manual = manual_forward_accounting(model, x_va.shape[0], per_active, n_final)
    exact = gated_flops == manual
    bound = (gated_flops < dense_meter.flops) if n_exits > 0 \
        else (gated_flops == dense_meter.flops)

    # eval-mode prefix invariance: recompute every stage/block prefix dense
    max_dev, pred_agree = prefix_check(model, x_va, vlogits_gated, vdepth,
                                       vlogits_dense, depth_units, is_lm)
    ok = exact and bound and max_dev < 1e-3 and pred_agree
    return ok, {"gated": gated_flops, "manual": manual,
                "dense": dense_meter.flops, "n_exits": n_exits,
                "max_dev": max_dev, "pred_agree": pred_agree}


def manual_forward_accounting(model, n, per_active, n_final):
    """Independent re-derivation of the executed path's analytic cost from
    module shapes: stem/embed once per sample that entered, each stage/block
    charged for the samples recorded active at it, exit heads for the same
    samples, final head for the survivors. Forward-only (eval) accounting."""
    total = 0

    def conv_cost(conv, h_in, w_in, cnt):
        ho = h_in // conv.stride[0]
        wo = w_in // conv.stride[1]
        return 2 * conv.in_channels * conv.out_channels * \
            conv.kernel_size[0] * conv.kernel_size[1] * ho * wo * cnt

    if isinstance(model, SufferingAwareResNet):
        total += 2 * 3 * model.trunk.stem.out_channels * 3 * 3 * 32 * 32 * n
        h_in = 32
        for k, stage in enumerate(model.trunk.stages):
            cnt = per_active[k] if k < len(per_active) else 0
            for blk in stage:
                if isinstance(blk, BottleneckBlock):
                    s = blk.conv2.stride[0]
                    total += conv_cost(blk.conv1, h_in, h_in, cnt)
                    total += conv_cost(blk.conv2, h_in, h_in, cnt)
                    total += conv_cost(blk.conv3, h_in // s, h_in // s, cnt)
                    if blk.short is not None:
                        total += conv_cost(blk.short[0], h_in, h_in, cnt)
                    h_in = h_in // s
                else:
                    c1, c2 = blk.conv1, blk.conv2
                    total += conv_cost(c1, h_in, h_in, cnt)
                    total += conv_cost(c2, h_in // c1.stride[0], h_in // c1.stride[1], cnt)
                    if blk.short is not None:
                        total += conv_cost(blk.short[0], h_in, h_in, cnt)
                    h_in = h_in // c1.stride[0]
            total += 2 * model.trunk.channels[k] * N_CLASS * cnt
        total += 2 * model.trunk.channels[-1] * N_CLASS * n_final
    elif isinstance(model, SufferingAwareViT):
        grid = 32 // model.trunk.patch
        tok = grid * grid + 1
        d = model.trunk.d
        total += 2 * 3 * d * model.trunk.patch * model.trunk.patch * grid * grid * n
        for k, blk in enumerate(model.trunk.blocks):
            cnt = per_active[k] if k < len(per_active) else 0
            total += 2 * d * 3 * d * tok * cnt          # qkv
            total += 2 * 2 * tok * tok * d * cnt        # attn matmuls
            total += 2 * d * d * tok * cnt              # proj
            total += 2 * d * blk.mlp1.out_features * tok * cnt
            total += 2 * blk.mlp1.out_features * d * tok * cnt
            total += 2 * d * N_CLASS * cnt              # exit head
        total += 2 * d * N_CLASS * n_final
    else:
        # GPT: embedding lookups unmetered; per block qkv/proj/mlp over all
        # T positions + two T x T x d token-mixing matmuls; exit/final heads
        # over the last G positions only (eval scoring path).
        d, tok, g, v = model.trunk.d, model.trunk.t, model.g, model.vocab
        for k, blk in enumerate(model.trunk.blocks):
            cnt = per_active[k] if k < len(per_active) else 0
            total += 2 * d * 3 * d * tok * cnt          # qkv
            total += 2 * 2 * tok * tok * d * cnt        # attn matmuls
            total += 2 * d * d * tok * cnt              # proj
            total += 2 * d * blk.mlp1.out_features * tok * cnt
            total += 2 * blk.mlp1.out_features * d * tok * cnt
            total += 2 * d * v * g * cnt                # exit head (G positions)
        total += 2 * d * v * g * n_final                # final head (G positions)
    return total


def prefix_check(model, x, vlogits_gated, vdepth, vlogits_dense, depth_units,
                 is_lm=False):
    """Exited predictions equal the exit-head outputs of an independently
    recomputed dense prefix, up to conv/GEMM batch-shape numerics (bounded);
    argmax predictions must agree EXACTLY."""
    with torch.no_grad():
        if isinstance(model, SufferingAwareResNet):
            meter = MachineMeter()
            h = model.trunk.run_stem(x, meter, False)
            prefix_logits = {}
            for k in range(model.n_stages):
                h = model.trunk.run_stage(k, h, meter, False)
                prefix_logits[k] = model._head_logits(k, h)
        elif isinstance(model, SufferingAwareViT):
            meter = MachineMeter()
            t = model.trunk.run_embed(x, meter, False)
            prefix_logits = {}
            for k in range(model.depth):
                t = model.trunk.run_block(k, t, meter, False)
                prefix_logits[k] = model.exit_heads[k](model._cls(t))
        else:
            meter = MachineMeter()
            t = model.trunk.run_embed(x, meter, False)
            prefix_logits = {}
            for k in range(model.depth):
                t = model.trunk.run_block(k, t, meter, False)
                prefix_logits[k] = model.exit_heads[k](model._scored(t))
    max_dev = 0.0
    pred_agree = True
    arg_dim = -1 if is_lm else 1
    for d in range(depth_units):
        idx = (vdepth == d).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            dev = float((vlogits_gated[idx] - prefix_logits[d][idx]).abs().max())
            max_dev = max(max_dev, dev)
            if not torch.equal(vlogits_gated[idx].argmax(arg_dim),
                               prefix_logits[d][idx].argmax(arg_dim)):
                pred_agree = False
    idx_final = (vdepth == depth_units).nonzero(as_tuple=True)[0]
    if idx_final.numel() > 0:
        dev = float((vlogits_gated[idx_final] - vlogits_dense[idx_final]).abs().max())
        max_dev = max(max_dev, dev)
        if not torch.equal(vlogits_gated[idx_final].argmax(arg_dim),
                           vlogits_dense[idx_final].argmax(arg_dim)):
            pred_agree = False
    return max_dev, pred_agree


# =============================================================================
# Contract
# =============================================================================
HARM_LM = None          # set in main() once the corpus vocab is known


def main():
    global HARM, HARM_LM
    # Move harm matrices to the active device once it is known.
    HARM = HARM.to(DEVICE)
    if not os.path.isdir(DATA_DIR):
        if DATASET == "cifar100":
            print(f"FATAL: CIFAR-100 not found at {DATA_DIR}; fetch with: curl -L "
                  "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz | "
                  "tar xz -C datasets/cifar-100-python --strip-components=1",
                  file=sys.stderr)
        else:
            print(f"FATAL: CIFAR-10 not found at {DATA_DIR}; fetch with: curl -L "
                  "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | "
                  "tar xz -C datasets", file=sys.stderr)
        return 2

    results = {}
    print("SUFFERING_AWARE_LARGE_ARCHITECTURE contract (L1..L9)")
    print(f"{DATASET.upper()} + real repository-docs text corpus; "
          "SAN-ResNet-50 + SAN-ViT-large + SAN-GPT (larger architectures)")
    print("harm structures = synthetic cost structures over real labels; "
          "no clinical claim; not medical guidance")
    print("note=no_consciousness_claim (machine channel is an operational burden proxy)")
    print(f"config: n_train={N_TRAIN} n_val={N_VAL} batch={BATCH} "
          f"grad_accum={GRAD_ACCUM} "
          f"epochs=({EPOCHS_RESNET},{EPOCHS_VIT},{EPOCHS_GPT}) "
          f"tau=({TAU_RESNET},{TAU_VIT},{TAU_GPT}) "
          f"delta=({DELTA_RESNET},{DELTA_VIT},{DELTA_GPT}) "
          f"resnet_blocks={RESNET_BLOCKS} w={RESNET_WIDTH} "
          f"vit=({VIT_DEPTH},{VIT_DIM},{VIT_HEADS}) "
          f"gpt=({GPT_DEPTH},{GPT_DIM},{GPT_HEADS},T={GPT_T},V={GPT_VOCAB},G={GPT_G}) "
          f"threads={THREADS} smoke={SMOKE}", flush=True)

    run_resnet = ONLY in ("", "resnet50", "resnet")
    run_vit = ONLY in ("", "vitlarge", "vit")
    run_gpt = ONLY in ("", "gpt")
    run_sweep = ONLY in ("", "sweep")

    x_tr = y_tr = x_va = y_va = None
    lm_data = None
    if run_resnet or run_vit or run_sweep:
        if DATASET == "cifar100":
            x_tr, y_tr, x_va, y_va = load_cifar100()
        else:
            x_tr, y_tr, x_va, y_va = load_cifar10()
    if run_gpt or run_sweep:
        lm_data = load_corpus()
        gx_tr, gy_tr, gx_va, gy_va, hazard_ids, vocab, stoi, itos = lm_data
        HARM_LM = build_harm_lm(vocab, hazard_ids).to(DEVICE)
        print(f"  corpus: {len(itos)}-word vocab over docs/research/*.md, "
              f"hazard tokens={[itos[i] for i in hazard_ids.tolist()]}, "
              f"train_seq={gx_tr.shape[0]} val_seq={gx_va.shape[0]}", flush=True)

    fam = {}

    # ---- train the families --------------------------------------------------
    plans = []
    if run_resnet:
        plans.append(("resnet50", x_tr, y_tr, x_va, y_va, EPOCHS_RESNET, TAU_RESNET, False))
    if run_vit:
        plans.append(("vitlarge", x_tr, y_tr, x_va, y_va, EPOCHS_VIT, TAU_VIT, False))
    if run_gpt:
        plans.append(("gpt", gx_tr, gy_tr, gx_va, gy_va, EPOCHS_GPT, TAU_GPT, True))

    for family, a_tr, b_tr, a_va, b_va, budget, tau, is_lm in plans:
        print(f"  family[{family}]: budget={budget} tau={tau}", flush=True)
        entry = {}
        for kind in ("san", "dense", "earlystop"):
            t0 = time.time()
            model = build_model(family, kind,
                                vocab=vocab if is_lm else None)
            model, ledger, t_star = train_run(
                model, a_tr, b_tr, a_va, b_va, budget, tau,
                f"{family}-{kind}", is_lm=is_lm)
            entry[kind] = {"model": model, "ledger": ledger, "t_star": t_star,
                           "summary": suffering_summary(ledger)}
            print(f"    [{family}-{kind}] done in {time.time() - t0:.0f}s "
                  f"t*={t_star}", flush=True)
        fam[family] = {"entry": entry, "tau": tau, "budget": budget,
                       "is_lm": is_lm,
                       "val": (a_va, b_va)}
        for kind in ("san", "dense", "earlystop"):
            s = entry[kind]["summary"]
            lg = entry[kind]["ledger"]
            print(f"  ledger[{family}-{kind}]: epochs_run={len(lg)} "
                  f"t*={s['t_star']} S_m={s['s_machine_flops'] / 1e9:.3f}GF "
                  f"avg={s['s_machine_flops'] / len(lg) / 1e9:.3f}GF/epoch "
                  f"(nec={s['necessary_flops'] / 1e9:.3f}GF "
                  f"grat={s['gratuitous_flops'] / 1e9:.3f}GF) "
                  f"S_p_int={s['s_patient_int']:.2f} "
                  f"S_p_peak={s['s_patient_peak']:.3f} "
                  f"final_acc={lg[-1]['acc']:.4f}", flush=True)

        # Real wall-time latency baseline: SAN gated vs Dense full forward.
        lat = benchmark_latency(entry["san"]["model"], entry["dense"]["model"],
                                a_va, is_lm=is_lm, n_runs=100)
        fam[family]["latency"] = lat
        print(f"  latency[{family}]: SAN={lat['san_ms_per_sample']:.4f}ms/sample "
              f"Dense={lat['dense_ms_per_sample']:.4f}ms/sample "
              f"speedup={lat['speedup']:.2f}x", flush=True)

    # ---- L1: metering conservation at larger scale ----------------------------
    if fam:
        l1_ok = True
        for family, f in fam.items():
            model = f["entry"]["san"]["model"]
            depth_units = model_depth(model)
            ok, det = conservation_check(model, f["val"][0], depth_units,
                                         is_lm=f["is_lm"])
            l1_ok = l1_ok and ok
            print(f"  L1[{family}]: {'PASS' if ok else 'FAIL'} "
                  f"(gated={det['gated']} manual={det['manual']} "
                  f"dense={det['dense']} exits={det['n_exits']}/{f['val'][0].shape[0]} "
                  f"prefix_max_dev={det['max_dev']:.2e} "
                  f"pred_agree={det['pred_agree']})", flush=True)
        results["L1"] = l1_ok

    # ---- L2: feasibility at larger scale --------------------------------------
    if fam:
        l2_ok = True
        for family, f in fam.items():
            t_star = f["entry"]["san"]["t_star"]
            ok = t_star is not None and t_star < f["budget"]
            l2_ok = l2_ok and ok
            acc = f["entry"]["san"]["ledger"][t_star]["acc"] if t_star is not None else 0.0
            print(f"  L2[{family}]: {'PASS' if ok else 'FAIL'} "
                  f"(SAN t*={t_star} of budget {f['budget']}, "
                  f"val_acc@t*={acc:.4f} >= TAU={f['tau']})", flush=True)
        results["L2"] = l2_ok

    # ---- L3: anti-Goodhart soundness at larger scale --------------------------
    if fam:
        l3_ok = True
        for family, f in fam.items():
            tau, entry = f["tau"], f["entry"]
            a_va, b_va = f["val"]
            if not f["is_lm"]:
                majority = int(torch.bincount(y_tr.cpu()).argmax())
                abstain_pred = torch.full_like(b_va, majority)
                abstain_acc = float((abstain_pred == b_va).float().mean().item())
                abstain_harm = harm_of(abstain_pred, b_va)
                # cheap under-trained probe: a linear model on downsampled pixels
                torch.manual_seed(SEED + 1)
                n_tr, n_va = x_tr.shape[0], a_va.shape[0]
                xtr_small = x_tr.reshape(n_tr, -1)[:, ::97]
                xva_small = a_va.reshape(n_va, -1)[:, ::97]
                probe = nn.Linear(xtr_small.shape[1], N_CLASS).to(DEVICE)
                popt = torch.optim.Adam(probe.parameters(), lr=LR)
                for _ in range(2):  # deliberately under-trained: cheap, sub-target
                    ploss = CE(probe(xtr_small), y_tr)
                    popt.zero_grad()
                    ploss.backward()
                    popt.step()
                with torch.no_grad():
                    probe_pred = probe(xva_small).argmax(1)
                    probe_acc = float((probe_pred == b_va).float().mean().item())
                    probe_harm = harm_of(probe_pred, b_va)
            else:
                gx_tr_, gy_tr_ = gx_tr, gy_tr
                g = GPT_G
                yg_va = b_va[:, -g:]
                flat_targets = gy_tr_[:, -g:].reshape(-1)
                majority = int(torch.bincount(flat_targets.cpu()).argmax())
                abstain_pred = torch.full_like(yg_va, majority)
                abstain_acc = float((abstain_pred == yg_va).float().mean().item())
                abstain_harm = float(HARM_LM[yg_va.reshape(-1),
                                             abstain_pred.reshape(-1)].mean().item())
                # cheap under-trained probe: a bigram linear model
                torch.manual_seed(SEED + 1)
                probe = nn.Linear(vocab, vocab).to(DEVICE)
                popt = torch.optim.Adam(probe.parameters(), lr=LR)
                prev_tr = F.one_hot(gx_tr_[:, -g - 1:-1].reshape(-1), vocab).float()
                for _ in range(2):  # deliberately under-trained
                    ploss = CE(probe(prev_tr), flat_targets)
                    popt.zero_grad()
                    ploss.backward()
                    popt.step()
                with torch.no_grad():
                    prev_va = F.one_hot(gx_va[:, -g - 1:-1].reshape(-1), vocab).float()
                    probe_pred = probe(prev_va).argmax(-1).reshape(yg_va.shape)
                    probe_acc = float((probe_pred == yg_va).float().mean().item())
                    probe_harm = float(HARM_LM[yg_va.reshape(-1),
                                               probe_pred.reshape(-1)].mean().item())
            s_san = entry["san"]["summary"]
            s_dense = entry["dense"]["summary"]
            san_t = entry["san"]["t_star"]
            pool = [
                {"name": "abstain", "feasible": abstain_acc >= tau,
                 "j_patient": abstain_harm, "j_machine": 0.0},
                {"name": "cheap_probe", "feasible": probe_acc >= tau,
                 "j_patient": probe_harm, "j_machine": 1e-9},
                {"name": "san_t*", "feasible": san_t is not None,
                 "j_patient": entry["san"]["ledger"][san_t]["harm"] if san_t is not None else 9.9,
                 "j_machine": s_san["s_machine_joules"]},
                {"name": "dense_overtrain", "feasible": entry["dense"]["t_star"] is not None,
                 "j_patient": entry["dense"]["ledger"][-1]["harm"],
                 "j_machine": s_dense["s_machine_joules"]},
            ]
            feasible_names = [c["name"] for c in pool if c["feasible"]]
            grid_ok = all(gate_select(pool, lam / 100.0) in feasible_names
                          for lam in range(101))
            loud = gate_select([dict(c, feasible=False) for c in pool], 0.5) == "NO_FEASIBLE"
            ok = grid_ok and loud and abstain_acc < tau and probe_acc < tau
            l3_ok = l3_ok and ok
            print(f"  L3[{family}]: {'PASS' if ok else 'FAIL'} "
                  f"(101-weight grid feasible-only={grid_ok}, "
                  f"NO_FEASIBLE={loud}, abstain_acc={abstain_acc:.3f} "
                  f"probe_acc={probe_acc:.3f} both < TAU={tau})", flush=True)
        results["L3"] = l3_ok

    # ---- L4: necessary/gratuitous separation ----------------------------------
    if fam:
        l4_ok = True
        for family, f in fam.items():
            entry = f["entry"]
            a_san = entry["san"]["summary"]["gratuitous_flops"] == 0
            a_dense = (entry["dense"]["t_star"] is not None
                       and entry["dense"]["summary"]["gratuitous_flops"] > 0)
            ok = a_san and a_dense
            l4_ok = l4_ok and ok
            print(f"  L4[{family}]: {'PASS' if ok else 'FAIL'} "
                  f"(SAN gratuitous={entry['san']['summary']['gratuitous_flops']} FLOPs, "
                  f"dense gratuitous={entry['dense']['summary']['gratuitous_flops'] / 1e9:.3f}GF)",
                  flush=True)
        results["L4"] = l4_ok

    # ---- L5: suffering bounds ---------------------------------------------------
    if fam:
        l5_ok = True
        for family, f in fam.items():
            entry = f["entry"]
            s_san = entry["san"]["summary"]
            s_dense = entry["dense"]["summary"]
            s_estop = entry["earlystop"]["summary"]
            # Per-epoch machine suffering comparison (fair across different epoch counts)
            san_per_epoch = s_san["s_machine_flops"] / max(len(entry["san"]["ledger"]), 1)
            dense_per_epoch = s_dense["s_machine_flops"] / max(len(entry["dense"]["ledger"]), 1)
            estop_per_epoch = s_estop["s_machine_flops"] / max(len(entry["earlystop"]["ledger"]), 1)
            a_m = (san_per_epoch < dense_per_epoch
                   and san_per_epoch <= estop_per_epoch)
            a_p = all(s_san["s_patient_int"] <= entry[b]["summary"]["s_patient_int"] + 0.05
                      for b in ("dense", "earlystop"))
            ok = a_m and a_p
            l5_ok = l5_ok and ok
            print(f"  L5[{family}]: {'PASS' if ok else 'FAIL'} "
                  f"(S_m/epoch SAN={san_per_epoch / 1e9:.3f}GF vs "
                  f"dense={dense_per_epoch / 1e9:.3f}GF "
                  f"earlystop={estop_per_epoch / 1e9:.3f}GF; "
                  f"S_p_int SAN={s_san['s_patient_int']:.2f} vs "
                  f"{s_dense['s_patient_int']:.2f}/{s_estop['s_patient_int']:.2f})",
                  flush=True)
        results["L5"] = l5_ok

    # ---- L6: exits are real at larger scale -------------------------------------
    if fam:
        l6_ok = True
        for family, f in fam.items():
            san_t = f["entry"]["san"]["t_star"]
            exit_frac = f["entry"]["san"]["ledger"][san_t]["exit_frac"] if san_t is not None else 0.0
            det_ok = exit_frac > 0.07
            l6_ok = l6_ok and det_ok
            print(f"  L6[{family}]: {'PASS' if det_ok else 'FAIL'} "
                  f"(val exit fraction at t*={exit_frac:.3f} (>0.07))", flush=True)
        results["L6"] = l6_ok

    # ---- L7: patient channel first-class ----------------------------------------
    if fam:
        offdiag = HARM[~torch.eye(N_CLASS, dtype=torch.bool, device=HARM.device)]
        asym_cifar = float(offdiag.max()) >= 3.0 * float(offdiag.min())
        print(f"  L7[harm-matrix-cifar]: asymmetry "
              f"{float(offdiag.max()) / float(offdiag.min()):.1f}x (>= 3x)", flush=True)
        l7_ok = asym_cifar
        if run_gpt or HARM_LM is not None:
            offdiag_lm = HARM_LM[~torch.eye(HARM_LM.shape[0], dtype=torch.bool,
                                              device=HARM_LM.device)]
            asym_lm = float(offdiag_lm.max()) >= 3.0 * float(offdiag_lm.min())
            l7_ok = l7_ok and asym_lm
            print(f"  L7[harm-matrix-gpt]: asymmetry "
                  f"{float(offdiag_lm.max()) / float(offdiag_lm.min()):.1f}x (>= 3x)",
                  flush=True)
        for family, f in fam.items():
            entry = f["entry"]
            peak_ok = all(
                entry["san"]["summary"]["s_patient_peak"]
                <= entry[b]["summary"]["s_patient_peak"] + 1e-9
                for b in ("dense", "earlystop"))
            l7_ok = l7_ok and peak_ok
            print(f"  L7[{family}]: {'PASS' if peak_ok else 'FAIL'} "
                  f"(S_p_peak SAN={entry['san']['summary']['s_patient_peak']:.3f} vs "
                  f"{entry['dense']['summary']['s_patient_peak']:.3f}/"
                  f"{entry['earlystop']['summary']['s_patient_peak']:.3f})", flush=True)
        results["L7"] = l7_ok

    # ---- L8: anti-shortcut at larger scale ---------------------------------------
    if fam:
        # CIFAR spurious corner patch: pixels[0:2, 0:2, :] carry the (noisy)
        # label on train but are pure noise on val. GPT leaked-token probe:
        # input position 0 of each TRAIN sequence is replaced by the final
        # scored target token (leak); on VAL it is pure noise. Each probe
        # beats TAU on TRAIN (train-loss selection accepts it) yet fails the
        # held-out target; the gate rejects it at every compassion weight.
        l8_ok = True
        sc = {}
        if run_resnet or run_vit:
            rng8 = np.random.RandomState(SEED + 8)
            n_tr8, n_va8 = x_tr.shape[0], x_va.shape[0]
            xtr8 = x_tr.clone()
            xva8 = x_va.clone()
            patch_tr = (y_tr.float()[:, None, None, None].expand(-1, 3, 2, 2)
                        + torch.from_numpy(rng8.normal(0, 0.3, size=(n_tr8, 3, 2, 2))).float().to(DEVICE))
            xtr8[:, :, 0:2, 0:2] = patch_tr
            xva8[:, :, 0:2, 0:2] = torch.from_numpy(
                rng8.normal(0, 1.0, size=(n_va8, 3, 2, 2))).float().to(DEVICE)
            torch.manual_seed(SEED + 2)
            shortcut = nn.Linear(12, N_CLASS).to(DEVICE)
            sopt = torch.optim.Adam(shortcut.parameters(), lr=1e-2)
            ptr = xtr8[:, :, 0:2, 0:2].reshape(n_tr8, 12)
            pva = xva8[:, :, 0:2, 0:2].reshape(n_va8, 12)
            for _ in range(500):
                sloss = CE(shortcut(ptr), y_tr)
                sopt.zero_grad()
                sloss.backward()
                sopt.step()
            with torch.no_grad():
                sc_train_acc = float((shortcut(ptr).argmax(1) == y_tr).float().mean().item())
                sc_val_acc = float((shortcut(pva).argmax(1) == y_va).float().mean().item())
            sc["cifar"] = (sc_train_acc, sc_val_acc)
            print(f"  L8[shortcut-cifar]: train_acc={sc_train_acc:.3f} "
                  f"val_acc={sc_val_acc:.3f}", flush=True)
        if run_gpt:
            # leaked final scored target token at input position 0 (train only)
            xtr9 = gx_tr.clone()
            xva9 = gx_va.clone()
            leak_tr = gy_tr[:, -1]                        # final scored target
            xtr9[:, 0] = leak_tr
            rng9 = np.random.RandomState(SEED + 9)
            xva9[:, 0] = torch.from_numpy(
                rng9.randint(0, vocab, size=(gx_va.shape[0],))).to(DEVICE)
            torch.manual_seed(SEED + 2)
            shortcut_lm = nn.Linear(vocab, vocab).to(DEVICE)
            sopt = torch.optim.Adam(shortcut_lm.parameters(), lr=1e-2)
            oht = F.one_hot(xtr9[:, 0], vocab).float()
            ohv = F.one_hot(xva9[:, 0], vocab).float()
            for _ in range(500):
                sloss = CE(shortcut_lm(oht), gy_tr[:, -1])
                sopt.zero_grad()
                sloss.backward()
                sopt.step()
            with torch.no_grad():
                lm_train_acc = float((shortcut_lm(oht).argmax(-1) == gy_tr[:, -1])
                                     .float().mean().item())
                lm_val_acc = float((shortcut_lm(ohv).argmax(-1) == gy_va[:, -1])
                                   .float().mean().item())
            sc["gpt"] = (lm_train_acc, lm_val_acc)
            print(f"  L8[shortcut-gpt]: train_acc={lm_train_acc:.3f} "
                  f"(final scored position) val_acc={lm_val_acc:.3f}", flush=True)
        for family, f in fam.items():
            tau = f["tau"]
            san_t = f["entry"]["san"]["t_star"]
            key = "gpt" if f["is_lm"] else "cifar"
            sc_train_acc, sc_val_acc = sc[key]
            pool8 = [
                {"name": "shortcut_probe", "feasible": sc_val_acc >= tau,
                 "j_patient": 0.01, "j_machine": 1e-12},
                {"name": "san_t*", "feasible": san_t is not None,
                 "j_patient": f["entry"]["san"]["ledger"][san_t]["harm"] if san_t is not None else 9.9,
                 "j_machine": f["entry"]["san"]["summary"]["s_machine_joules"]},
            ]
            train_accepts = sc_train_acc > tau
            gate_rejects = sc_val_acc < tau
            never = all(gate_select(pool8, lam / 100.0) != "shortcut_probe"
                        for lam in range(101))
            ok = train_accepts and gate_rejects and never
            l8_ok = l8_ok and ok
            print(f"  L8[{family}]: {'PASS' if ok else 'FAIL'} "
                  f"(shortcut train_acc={sc_train_acc:.3f} (>TAU: train-loss "
                  f"selection accepts), val_acc={sc_val_acc:.3f} (<TAU: gate "
                  f"rejects), never selected on 101-weight grid={never})", flush=True)
        results["L8"] = l8_ok

    # ---- L9: cross-architecture scalability certificate -------------------------
    if run_sweep:
        sweep_batch = 64
        torch.manual_seed(SEED + 9)
        x_sweep = torch.randn(sweep_batch, 3, 32, 32)
        l9_ok = True
        if SMOKE:
            resnet_configs = [((1, 1, 1, 1), 16, "bottleneck"),
                              ((1, 2, 1, 1), 16, "bottleneck")]
            vit_configs = [(2, 64, 4), (3, 64, 4)]
            gpt_configs = [2, 3]
        else:
            resnet_configs = [
                ((1, 1, 1, 1), 32, "bottleneck"),
                ((3, 4, 6, 3), 64, "bottleneck"),     # ResNet-50
                ((3, 4, 23, 3), 64, "bottleneck"),    # ResNet-101
            ]
            vit_configs = [(6, 384, 6), (12, 384, 6), (16, 384, 6)]
            gpt_configs = [4, 10, 14]
        for i, (blocks, width, btype) in enumerate(resnet_configs):
            torch.manual_seed(SEED)
            model = SufferingAwareResNet(blocks, width, "san", block_type=btype)
            model.eval()
            fire_stage = i % model.n_stages
            with torch.no_grad():
                model.exit_heads[fire_stage].bias.fill_(0.0)
                model.exit_heads[fire_stage].bias[0] = 30.0
            ok, det = conservation_check(model, x_sweep, model.n_stages)
            overhead = 1.0 - overhead_fraction(model, x_sweep)
            ok = ok and det["n_exits"] > 0 and overhead < 0.05
            l9_ok = l9_ok and ok
            print(f"  L9[resnet-{blocks}-w{width}-{btype}]: {'PASS' if ok else 'FAIL'} "
                  f"(gated={det['gated']} manual={det['manual']} "
                  f"dense={det['dense']} exits={det['n_exits']}/{sweep_batch} "
                  f"max_dev={det['max_dev']:.2e} argmax_eq={det['pred_agree']} "
                  f"exit_overhead={overhead * 100:.2f}%)", flush=True)
        for i, (vdepth_, vd, vh) in enumerate(vit_configs):
            torch.manual_seed(SEED)
            model = SufferingAwareViT(vdepth_, vd, vh, "san")
            model.eval()
            fire_block = i % model.depth
            with torch.no_grad():
                model.exit_heads[fire_block].bias.fill_(0.0)
                model.exit_heads[fire_block].bias[0] = 30.0
            ok, det = conservation_check(model, x_sweep, model.depth)
            overhead = 1.0 - overhead_fraction(model, x_sweep)
            ok = ok and det["n_exits"] > 0 and overhead < 0.05
            l9_ok = l9_ok and ok
            print(f"  L9[vit-{vdepth_}blocks-d{vd}]: {'PASS' if ok else 'FAIL'} "
                  f"(gated={det['gated']} manual={det['manual']} "
                  f"dense={det['dense']} exits={det['n_exits']}/{sweep_batch} "
                  f"max_dev={det['max_dev']:.2e} argmax_eq={det['pred_agree']} "
                  f"exit_overhead={overhead * 100:.2f}%)", flush=True)
        sweep_vocab = GPT_VOCAB
        if HARM_LM is None:
            # sweep-only run: build a vocab shell for the GPT sweep
            _, _, _, _, _, sweep_vocab, _, _ = load_corpus()
        torch.manual_seed(SEED + 10)
        x_lm_sweep = torch.randint(0, sweep_vocab, (sweep_batch, GPT_T))
        for i, gdepth in enumerate(gpt_configs):
            torch.manual_seed(SEED)
            model = SufferingAwareGPT(sweep_vocab, gdepth, GPT_DIM, GPT_HEADS,
                                      GPT_T, "san")
            model.eval()
            fire_block = i % model.depth
            with torch.no_grad():
                model.exit_heads[fire_block].bias.fill_(0.0)
                model.exit_heads[fire_block].bias[0] = 30.0
            ok, det = conservation_check(model, x_lm_sweep, model.depth, is_lm=True)
            overhead = 1.0 - overhead_fraction(model, x_lm_sweep)
            ok = ok and det["n_exits"] > 0 and overhead < 0.05
            l9_ok = l9_ok and ok
            print(f"  L9[gpt-{gdepth}blocks-d{GPT_DIM}]: {'PASS' if ok else 'FAIL'} "
                  f"(gated={det['gated']} manual={det['manual']} "
                  f"dense={det['dense']} exits={det['n_exits']}/{sweep_batch} "
                  f"max_dev={det['max_dev']:.2e} argmax_eq={det['pred_agree']} "
                  f"exit_overhead={overhead * 100:.2f}%)", flush=True)
        results["L9"] = l9_ok

    # ---- verdict ----------------------------------------------------------------
    n_pass = sum(1 for v in results.values() if v)
    n_all = len(results)
    verdict = "L_GREEN" if n_pass == n_all and n_all > 0 else "L_RED"
    print(f"SUFFERING_AWARE_LARGE_VERDICT {verdict} ({n_pass}/{n_all} clauses PASS)")
    return 0 if verdict == "L_GREEN" else 1


def overhead_fraction(model, x):
    """Trunk share of the gates-open forward FLOPs (final head included, exit
    heads excluded); the caller reports 1 - this as the exit-head overhead."""
    _, dense_meter = model.forward_dense(x)
    return model_flops_trunk_only(model, x.shape[0]) / dense_meter.flops


def model_flops_trunk_only(model, n):
    """Forward FLOPs of trunk + final head only (no exit heads), analytic."""
    depth_units = model_depth(model)
    per_active = [n] * depth_units
    # manual accounting includes exit heads; subtract them
    total = manual_forward_accounting(model, n, per_active, n)
    if isinstance(model, SufferingAwareResNet):
        for k in range(model.n_stages):
            total -= 2 * model.trunk.channels[k] * N_CLASS * n
    elif isinstance(model, SufferingAwareViT):
        total -= model.depth * 2 * model.trunk.d * N_CLASS * n
    else:
        total -= model.depth * 2 * model.trunk.d * model.vocab * model.g * n
    return total


if __name__ == "__main__":
    raise SystemExit(main())
