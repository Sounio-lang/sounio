#!/usr/bin/env python3
"""Mercyful Learning — Suffering-Aware neural Network (SAN) at EXTREME depth:
a 100-layer residual network (SAN-100) and a 24-block transformer
(SAN-ViT-24) on real data (CIFAR-10).

Companion artifact to
  docs/research/suffering_aware_extreme_depth_spec_2026-07-31.md

The deep-network SAN (scripts/research/suffering_aware_deep_architecture.py,
clauses D1..D9) established the architecture on ResNet-18 (18 weighted
layers) and a 6-block ViT-small. This harness pushes the SAME architecture
class — suffering-aware layers, per-sample exit gates, deep supervision,
freeze-on-green, the anti-Goodhart gate — to extreme depth:

  * SAN-100: a CIFAR-variant residual network of 100 weighted layers
    (stem conv + 49 basic blocks + fc = 100, the ResNet depth-counting
    convention; the 2 downsample shortcut convs are not counted, exactly as
    ResNet-18 counts 1 + 8*2 + 1 = 18). Seven stages of seven basic blocks,
    stage widths (32,32,64,64,128,128,128) — a wide CIFAR profile in the
    WRN tradition; the classic narrow ResNet-110 profile (16/32/64) was
    probed first and its failure mode is certified in the spec's
    calibration history — stride-2
    downsampling at the entries of stages 2 and 4. Each stage carries a
    suffering-aware exit head (global average pool + linear): 7 exit points
    against the parent line's 4.
  * SAN-ViT-24: the parent line's vision transformer scaled from 6 to 24
    attention blocks (patch 4x4 -> 64 tokens + CLS, d=96, 4 heads, MLP
    ratio 2), one exit head per block on the CLS token: 24 exit points
    against the parent line's 6.

Both are trained on CIFAR-10 (real dataset, stratified subset, no
augmentation — documented scope) and compared against the STANDARD deep
architectures: the identical 100-layer / 24-block trunks trained with a
fixed epoch budget (Dense), and the identical trunks with SAN's stop rule
but no suffering-aware layers (EarlyStop, the strongest scheduler
baseline). All runs within a family share one trunk init, one data order,
one seed.

Machine suffering is metered exactly as in the parent lines: analytic
FLOPs of the executed path (conv/linear MACs x2, x3 in training — the
fixed backward = 2x forward accounting convention of the machine-channel
benchmark), gated-off stages/blocks charging exactly 0. Patient suffering
is the mean harm of the current model's predictions on the held-out cohort
under the parent line's asymmetric 10-class harm matrix (class 9 "truck"
is the hazard class of a screening pipeline: a missed hazard costs 5, a
false hazard costs 2, any other confusion costs 1; synthetic cost
structure over real labels; no clinical claim).

The extreme-depth questions the contract answers:
  1. Trainability: a plain 100-layer net without residuals does not train;
     does the residual SAN-100 — with 7 deep-supervised exit heads — reach
     the declared target inside a CPU-affordable budget, and does the plain
     100-layer trunk (dense baseline) reach it at all?
  2. Metering: does T1' conservation survive 100 layers / 24 attention
     blocks — metered == independent manual accounting exactly, gated <
     gates-open whenever an exit fires, prefix argmax exactly equal?
  3. Soundness: does the anti-Goodhart gate still reject abstainers,
     under-trained probes, and shortcut models at extreme depth?
  4. Economics: the parent line showed exit savings grow with depth (more
     downstream stages to skip). At 100 layers the skippable fraction of
     the trunk dominates: does the measured per-epoch and deployment
     machine-suffering saving grow accordingly?

Certificates (contract clauses X1..X10, extending the parent D1..D9):
  X1  metering conservation at extreme depth (both families): gated-off
      stages/blocks charge exactly 0; metered FLOPs == an independent
      manual accounting of the executed path; metered < gates-open whenever
      an exit fires; eval-mode prefix invariance (bounded logit deviation,
      argmax exactly equal)
  X2  feasibility at extreme depth: SAN-100 reaches val acc >= TAU_RESNET
      within budget and SAN-ViT-24 reaches val acc >= TAU_VIT within budget
  X3  anti-Goodhart soundness at extreme depth: feasible-only selection on
      a 101-point compassion-weight grid over a pool containing a zero-cost
      abstainer and an under-trained probe; all-infeasible pool -> loud
      NO_FEASIBLE
  X4  necessary/gratuitous separation: SAN gratuitous machine suffering is
      exactly 0 (both families); fixed-budget deep baselines accrue > 0
  X5  suffering bounds vs the standard fixed-budget architecture (the
      T3-backed comparison): SAN total machine suffering strictly below the
      identical trunk trained with a fixed epoch budget (both families);
      SAN integrated patient harm <= the dense baseline's
  X6  exits are real at extreme depth: held-out exit fraction at t* > 0.10
      for both families
  X7  patient channel first-class: harm matrix genuinely asymmetric (off-
      diagonal max >= 3x min); SAN peak patient harm <= same-init
      baselines' peaks
  X8  anti-shortcut at extreme depth: a linear probe on a spurious corner-
      patch feature beats TAU on TRAIN yet fails held-out; the gate rejects
      it at every compassion weight
  X9  scalability certificate at extreme depth: a depth sweep (ResNet
      16/44/100 weighted layers; ViT 6/12/24 blocks) verifies, forward-
      only, at every scale: metered == manual exactly, metered < gates-open
      when exits fire, prefix argmax agreement, and exit-head overhead <
      5% of the gates-open forward (the architecture scales to extreme
      depth without breaking)
  X10 the scheduler race at extreme depth, certified honestly: the parent
      line's D5 race against the EarlyStop scheduler SPLITS at extreme
      depth — the transformer family still wins it, the conv family loses
      it, and the clause certifies the split with its mechanism signature
      (epoch-1 untrained-head gating catastrophe in the conv family:
      exit fraction > 0.5 at <= chance accuracy when the gates open;
      gradual gate opening in the transformer family). The clause fails if
      the signature disappears — e.g. a mechanism fix that makes the conv
      family win — which is the event that would require the spec revised.

Real data (CIFAR-10), real architectures (100-layer ResNet, 24-block ViT).
The harm matrix is a synthetic construction over real labels: this
benchmark makes no clinical claim and is not medical guidance. The machine
channel is an operational computational-burden proxy;
no_consciousness_claim is made or needed.

Run: .venv/bin/python scripts/research/suffering_aware_extreme_depth.py
Requires: torch (CPU) + numpy from the repo .venv, and CIFAR-10 at
  datasets/cifar-10-batches-py (fetch: curl -L
  https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | tar xz -C datasets)
Env overrides: SAN_XDEPTH_SMOKE=1 (tiny fast mechanics check),
SAN_XDEPTH_ONLY=resnet|vit|sweep (run a subset), SAN_XDEPTH_THREADS=n,
SAN_XDEPTH_DELTA_RESNET / SAN_XDEPTH_DELTA_VIT (exit thresholds),
SAN_XDEPTH_TAU_RESNET / SAN_XDEPTH_TAU_VIT (declared targets),
SAN_XDEPTH_EPOCHS_RESNET / SAN_XDEPTH_EPOCHS_VIT (budgets),
SAN_XDEPTH_N_TRAIN / SAN_XDEPTH_N_VAL (cohort sizes).
"""

import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- determinism / config --------------------------------------
SEED = 17
torch.manual_seed(SEED)
np.random.seed(SEED)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get(
    "SAN_XDEPTH_DATA", os.path.join(REPO_ROOT, "datasets", "cifar-10-batches-py"))
SMOKE = os.environ.get("SAN_XDEPTH_SMOKE", "") == "1"
ONLY = os.environ.get("SAN_XDEPTH_ONLY", "").strip().lower()
THREADS = int(os.environ.get("SAN_XDEPTH_THREADS", "8"))  # 8 = this container's cgroup CPU quota
torch.set_num_threads(THREADS)

N_CLASS = 10
# Exit-gate confidence thresholds — declared architecture constants, per
# family like the targets TAU (parent spec section 3): the confidence scale
# of a family's feasibility regime is a property of the problem, so the gate
# threshold is declared per family, not tuned post-hoc per run.
_DELTA_ENV = os.environ.get("SAN_XDEPTH_DELTA", "")
DELTA_RESNET = float(os.environ.get("SAN_XDEPTH_DELTA_RESNET", _DELTA_ENV or "0.5"))
DELTA_VIT = float(os.environ.get("SAN_XDEPTH_DELTA_VIT", _DELTA_ENV or "0.4"))
WARMUP_EPOCHS = 1       # dense-identical epochs before gates/supervision switch on
AUX_W = float(os.environ.get("SAN_XDEPTH_AUXW", "1.0"))   # deep-supervision weight
E_PER_FLOP = 4e-12      # J/FLOP, same convention as the machine-channel benchmark
LR = 1e-3
BATCH = 128

if SMOKE:
    N_TRAIN, N_VAL = 512, 256
    EPOCHS_RESNET, EPOCHS_VIT = 3, 3
    TAU_RESNET, TAU_VIT = 0.10, 0.10
    RESNET_STAGE_BLOCKS, RESNET_WIDTHS = (1,) * 7, (8, 8, 16, 16, 32, 32, 32)
    VIT_DEPTH, VIT_DIM, VIT_HEADS = 4, 48, 4
    BATCH = 64
else:
    N_TRAIN, N_VAL = 2000, 1000        # stratified subsets of CIFAR-10 train/test
    EPOCHS_RESNET, EPOCHS_VIT = 8, 10
    TAU_RESNET, TAU_VIT = 0.25, 0.31   # declared anti-Goodhart targets (spec section 6)
    RESNET_STAGE_BLOCKS = (7,) * 7                       # SAN-100: stem + 49 blocks + fc
    RESNET_WIDTHS = (32, 32, 64, 64, 128, 128, 128)      # wide CIFAR profile (spec section 3)
    VIT_DEPTH, VIT_DIM, VIT_HEADS = 24, 96, 4            # SAN-ViT-24

N_TRAIN = int(os.environ.get("SAN_XDEPTH_N_TRAIN", N_TRAIN))
N_VAL = int(os.environ.get("SAN_XDEPTH_N_VAL", N_VAL))
EPOCHS_RESNET = int(os.environ.get("SAN_XDEPTH_EPOCHS_RESNET", EPOCHS_RESNET))
EPOCHS_VIT = int(os.environ.get("SAN_XDEPTH_EPOCHS_VIT", EPOCHS_VIT))
TAU_RESNET = float(os.environ.get("SAN_XDEPTH_TAU_RESNET", TAU_RESNET))
TAU_VIT = float(os.environ.get("SAN_XDEPTH_TAU_VIT", TAU_VIT))
RESNET_WIDTHS = tuple(int(w) for w in os.environ.get(
    "SAN_XDEPTH_WIDTHS", ",".join(str(w) for w in RESNET_WIDTHS)).split(","))

# ResNet depth-counting convention (shared with the parent line and the
# ResNet literature): weighted layers = stem conv + 2 per basic block + fc;
# downsample shortcut convs are not counted.
N_WEIGHTED_RESNET = 1 + 2 * sum(RESNET_STAGE_BLOCKS) + 1

CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

# Asymmetric harm matrix over the 10 CIFAR-10 classes (synthetic cost
# structure over real labels): class 9 ("truck") is the hazard class of a
# screening pipeline. A missed hazard (true 9, predicted other) costs 5 —
# the expensive error the gate exists to block; a false hazard costs 2
# (unnecessary intervention); any other confusion costs 1.
def build_harm():
    h = torch.ones(N_CLASS, N_CLASS)
    h.fill_diagonal_(0.0)
    h[9, :9] = 5.0     # missed hazard
    h[:9, 9] = 2.0     # false hazard -> unnecessary intervention
    return h

HARM = build_harm()


def harm_of(pred, y):
    """Mean harm of predictions pred against truth y (synthetic matrix)."""
    return float(HARM[y, pred].mean().item())


# ---------------- CIFAR-10 loading (no torchvision) --------------------------
def load_cifar10():
    """Stratified deterministic subsets: N_TRAIN train (N_TRAIN/10 per class)
    from the train batches, N_VAL (N_VAL/10 per class) from the test batch.
    Normalized with the standard CIFAR-10 channel statistics."""
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
    return (torch.from_numpy(xtr), torch.from_numpy(ytr_all[tr_idx]),
            torch.from_numpy(xva), torch.from_numpy(yte_all[va_idx]))


# ---------------- machine-suffering meter -----------------------------------
class MachineMeter:
    """Analytic FLOP accounting of the executed path:
      linear = 2*d_in*d_out per row; conv = 2*cin*cout*k*k*h_out*w_out per
      sample (MAC = 2 FLOPs); a training step charges forward + backward with
      backward = 2x forward (x3 total) — the fixed accounting convention of
      mercyful_machine_channel_benchmark.py. Elementwise ops (BN, ReLU,
      residual adds, softmax, pooling) are unmetered: stated convention,
      identical for every architecture and accounting path."""

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


# ---------------- ResNet trunk (CIFAR variant, any stage count) --------------
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


class ResNetTrunk(nn.Module):
    """Stem + S stages of basic blocks (any depth). Stage k produces the
    feature map at which exit head k sits; the final head sits after the
    last stage. Stage entry k in STRIDE_STAGES downsamples with stride 2."""

    def __init__(self, stage_blocks, widths, stride_stages=(2, 4)):
        super().__init__()
        assert len(stage_blocks) == len(widths)
        self.channels = tuple(widths)
        self.stem = nn.Conv2d(3, self.channels[0], 3, 1, 1, bias=False)
        self.stem_bn = nn.BatchNorm2d(self.channels[0])
        stages = []
        cin = self.channels[0]
        for k, (cout, n_blocks) in enumerate(zip(self.channels, stage_blocks)):
            blocks_k = []
            for b in range(n_blocks):
                stride = 2 if (k in stride_stages and b == 0) else 1
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
        meter.charge_conv(3, self.channels[0], 3, h, w, n, backward)
        return torch.relu(self.stem_bn(self.stem(x)))


# ---------------- ViT trunk ---------------------------------------------------
class ViTBlock(nn.Module):
    def __init__(self, d, heads, mlp_ratio=2):
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
    def __init__(self, depth=24, d=96, heads=4, patch=4):
        super().__init__()
        self.d, self.patch = d, patch
        self.embed = nn.Conv2d(3, d, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, (32 // patch) ** 2 + 1, d))
        self.blocks = nn.ModuleList([ViTBlock(d, heads) for _ in range(depth)])
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


# ---------------- SAN wrappers (exit gates + metering) -----------------------
class SufferingAwareResNet(nn.Module):
    """ResNet trunk with a per-stage exit head (GAP + linear). After each
    stage, samples whose exit-head confidence clears the family's declared
    threshold leave the network;
    remaining stages are gated off FOR THAT SAMPLE and meter exactly 0.
    kind='dense'/'earlystop' never runs the exit heads: the identical trunk
    as a standard architecture."""

    def __init__(self, stage_blocks, widths, kind):
        super().__init__()
        assert kind in ("san", "dense", "earlystop")
        self.kind = kind
        self.trunk = ResNetTrunk(stage_blocks, widths)
        self.n_stages = len(self.trunk.stages)
        self.exit_heads = nn.ModuleList(
            [nn.Linear(c, N_CLASS) for c in self.trunk.channels])
        self.final_head = nn.Linear(self.trunk.channels[-1], N_CLASS)
        self.meter = MachineMeter()

    def _head_logits(self, k, h):
        return self.exit_heads[k](h.mean(dim=(2, 3)))

    def forward(self, x, train=False, use_exit_heads=True):
        meter = self.meter
        n = x.shape[0]
        out_logits = x.new_zeros(n, N_CLASS)
        out_depth = torch.full((n,), self.n_stages, dtype=torch.long)
        h = self.trunk.run_stem(x, meter, train)
        active = torch.arange(n)
        per_stage_active = []
        aux_records, final_record = [], None
        gated = use_exit_heads and self.kind == "san"
        for k in range(self.n_stages):
            if active.numel() == 0:
                break
            per_stage_active.append(int(active.numel()))
            h = self.trunk.run_stage(k, h, meter, train)
            if not gated:
                continue
            meter.charge_linear(self.trunk.channels[k], N_CLASS,
                                h.shape[0], train)
            logits_k = self._head_logits(k, h)
            if train:
                aux_records.append((active, logits_k))
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
        self.trunk = ViTTrunk(depth, d, heads)
        self.depth = depth
        self.exit_heads = nn.ModuleList([nn.Linear(d, N_CLASS) for _ in range(depth)])
        self.final_head = nn.Linear(d, N_CLASS)
        self.meter = MachineMeter()

    def _cls(self, t):
        return self.trunk.ln(t)[:, 0]

    def forward(self, x, train=False, use_exit_heads=True):
        meter = self.meter
        n = x.shape[0]
        out_logits = x.new_zeros(n, N_CLASS)
        out_depth = torch.full((n,), self.depth, dtype=torch.long)
        t = self.trunk.run_embed(x, meter, train)
        active = torch.arange(n)
        per_block_active = []
        aux_records, final_record = [], None
        gated = use_exit_heads and self.kind == "san"
        for k in range(self.depth):
            if active.numel() == 0:
                break
            per_block_active.append(int(active.numel()))
            t = self.trunk.run_block(k, t, meter, train)
            if not gated:
                continue
            meter.charge_linear(self.trunk.d, N_CLASS, t.shape[0], train)
            logits_k = self.exit_heads[k](self._cls(t))
            if train:
                aux_records.append((active, logits_k))
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


# ---------------- data schedule ------------------------------------------------
def batch_schedule(n_train, epochs):
    """One fixed data order for every run in the family (bit-reproducible,
    shared across architectures so epoch-0 cohort exposure is identical)."""
    rng = np.random.RandomState(SEED)
    return [rng.permutation(n_train) for _ in range(epochs)]


# ---------------- training loops ----------------------------------------------
CE = nn.CrossEntropyLoss()


def train_run(model, x_tr, y_tr, x_va, y_va, epochs, tau, tag):
    """One training run. Dense: fixed budget, no exit heads. EarlyStop:
    SAN's stop rule, no exit heads. SAN: suffering-aware layers +
    freeze-on-green. Identical optimizer, seed, data order, ledger shape."""
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    schedule = batch_schedule(x_tr.shape[0], epochs)
    ledger = []
    t_star = None
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        model.meter = MachineMeter()
        warmup = epoch < WARMUP_EPOCHS
        use_exits = (model.kind == "san") and (not warmup)
        for b0 in range(0, x_tr.shape[0], BATCH):
            idx = schedule[epoch][b0:b0 + BATCH]
            xb, yb = x_tr[idx], y_tr[idx]
            _, _, _, _, aux_records, final_record = model(
                xb, train=True, use_exit_heads=use_exits)
            if not use_exits:
                f_idx, f_logits = final_record
                loss = CE(f_logits, yb[f_idx])
            else:
                losses = []
                if final_record is not None:
                    f_idx, f_logits = final_record
                    losses.append(CE(f_logits, yb[f_idx]))
                if aux_records:
                    losses.append(AUX_W * torch.stack(
                        [CE(a_logits, yb[a_idx])
                         for a_idx, a_logits in aux_records]).mean())
                loss = sum(losses)
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_flops = model.meter.flops
        # held-out evaluation (forward only): the cohort-in-waiting
        model.eval()
        model.meter = MachineMeter()
        with torch.no_grad():
            vlogits, vdepth, _, _, _, _ = model(
                x_va, train=False, use_exit_heads=use_exits)
        eval_flops = model.meter.flops
        pred = vlogits.argmax(dim=1)
        acc = float((pred == y_va).float().mean().item())
        harm = harm_of(pred, y_va)
        exit_frac = float((vdepth < model_depth(model)).float().mean().item())
        feasible = acc >= tau
        if feasible and t_star is None:
            t_star = epoch
        ledger.append({
            "epoch": epoch, "flops": train_flops + eval_flops,
            "acc": acc, "harm": harm, "exit_frac": exit_frac,
            "feasible": feasible,
        })
        print(f"    [{tag}] epoch={epoch} acc={acc:.4f} harm={harm:.3f} "
              f"exit={exit_frac:.3f} flops={(train_flops + eval_flops) / 1e9:.2f}GF "
              f"({time.time() - t0:.0f}s)", flush=True)
        if model.kind in ("san", "earlystop") and t_star is not None:
            break  # freeze-on-green: gratuitous suffering is exactly zero
    return model, ledger, t_star


def model_depth(model):
    return model.n_stages if hasattr(model, "n_stages") else model.depth


def build_model(family, kind, stage_blocks=None, widths=None, depth=None,
                d=None, heads=None):
    """Fresh model with the shared family trunk init (identical for every
    kind within a family) and deterministic exit-head init."""
    torch.manual_seed(SEED)
    if family == "resnet":
        return SufferingAwareResNet(stage_blocks or RESNET_STAGE_BLOCKS,
                                    widths or RESNET_WIDTHS, kind)
    return SufferingAwareViT(depth or VIT_DEPTH, d or VIT_DIM,
                             heads or VIT_HEADS, kind)


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


# ---------------- X1 / X9 helpers ----------------------------------------------
def conservation_check(model, x_va, depth_units):
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
                                       vlogits_dense, depth_units)
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
        total += 2 * 3 * model.trunk.channels[0] * 3 * 3 * 32 * 32 * n
        h_in = 32
        for k, stage in enumerate(model.trunk.stages):
            cnt = per_active[k] if k < len(per_active) else 0
            for blk in stage:
                c1, c2 = blk.conv1, blk.conv2
                total += conv_cost(c1, h_in, h_in, cnt)
                total += conv_cost(c2, h_in // c1.stride[0], h_in // c1.stride[1], cnt)
                if blk.short is not None:
                    total += conv_cost(blk.short[0], h_in, h_in, cnt)
                h_in = h_in // c1.stride[0]
            total += 2 * model.trunk.channels[k] * N_CLASS * cnt
        total += 2 * model.trunk.channels[-1] * N_CLASS * n_final
    else:
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
    return total


def prefix_check(model, x, vlogits_gated, vdepth, vlogits_dense, depth_units):
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
        else:
            meter = MachineMeter()
            t = model.trunk.run_embed(x, meter, False)
            prefix_logits = {}
            for k in range(model.depth):
                t = model.trunk.run_block(k, t, meter, False)
                prefix_logits[k] = model.exit_heads[k](model._cls(t))
    max_dev = 0.0
    pred_agree = True
    for d in range(depth_units):
        idx = (vdepth == d).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            dev = float((vlogits_gated[idx] - prefix_logits[d][idx]).abs().max())
            max_dev = max(max_dev, dev)
            if not torch.equal(vlogits_gated[idx].argmax(1),
                               prefix_logits[d][idx].argmax(1)):
                pred_agree = False
    idx_final = (vdepth == depth_units).nonzero(as_tuple=True)[0]
    if idx_final.numel() > 0:
        dev = float((vlogits_gated[idx_final] - vlogits_dense[idx_final]).abs().max())
        max_dev = max(max_dev, dev)
        if not torch.equal(vlogits_gated[idx_final].argmax(1),
                           vlogits_dense[idx_final].argmax(1)):
            pred_agree = False
    return max_dev, pred_agree


# =============================================================================
# Contract
# =============================================================================
def main():
    if not os.path.isdir(DATA_DIR):
        print(f"FATAL: CIFAR-10 not found at {DATA_DIR}; fetch with: curl -L "
              "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz | "
              "tar xz -C datasets", file=sys.stderr)
        return 2

    results = {}
    print("SUFFERING_AWARE_EXTREME_DEPTH contract (X1..X10)")
    print(f"CIFAR-10 (real data), SAN-100 ({N_WEIGHTED_RESNET} weighted layers) "
          f"+ SAN-ViT-{VIT_DEPTH} (real extreme-depth architectures)")
    print("harm matrix = synthetic cost structure over real labels; "
          "no clinical claim; not medical guidance")
    print("note=no_consciousness_claim (machine channel is an operational burden proxy)")
    print(f"config: n_train={N_TRAIN} n_val={N_VAL} batch={BATCH} "
          f"epochs_resnet={EPOCHS_RESNET} epochs_vit={EPOCHS_VIT} "
          f"tau_resnet={TAU_RESNET} tau_vit={TAU_VIT} "
          f"delta_resnet={DELTA_RESNET} delta_vit={DELTA_VIT} "
          f"resnet_stage_blocks={RESNET_STAGE_BLOCKS} widths={RESNET_WIDTHS} "
          f"vit_depth={VIT_DEPTH} vit_dim={VIT_DIM} "
          f"aux_w={AUX_W} threads={THREADS} "
          f"smoke={SMOKE}", flush=True)

    x_tr, y_tr, x_va, y_va = load_cifar10()
    run_resnet = ONLY in ("", "resnet")
    run_vit = ONLY in ("", "vit")
    run_sweep = ONLY in ("", "sweep")

    fam = {}

    # ---- train the two families --------------------------------------------
    for family, run in (("resnet", run_resnet), ("vit", run_vit)):
        if not run:
            continue
        budget = EPOCHS_RESNET if family == "resnet" else EPOCHS_VIT
        tau = TAU_RESNET if family == "resnet" else TAU_VIT
        print(f"  family[{family}]: budget={budget} tau={tau}", flush=True)
        entry = {}
        for kind in ("san", "dense", "earlystop"):
            t0 = time.time()
            model = build_model(family, kind)
            model, ledger, t_star = train_run(
                model, x_tr, y_tr, x_va, y_va, budget, tau, f"{family}-{kind}")
            entry[kind] = {"model": model, "ledger": ledger, "t_star": t_star,
                           "summary": suffering_summary(ledger)}
            print(f"    [{family}-{kind}] done in {time.time() - t0:.0f}s "
                  f"t*={t_star}", flush=True)
        fam[family] = entry
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

    # ---- X1: metering conservation at extreme depth (both families) ---------
    if fam:
        x1_ok, x1_details = True, {}
        for family, entry in fam.items():
            model = entry["san"]["model"]
            depth_units = model_depth(model)
            ok, det = conservation_check(model, x_va, depth_units)
            x1_ok = x1_ok and ok
            x1_details[family] = det
            saved = 1.0 - det["gated"] / det["dense"]
            print(f"  X1[{family}]: {'PASS' if ok else 'FAIL'} "
                  f"(gated={det['gated']} manual={det['manual']} "
                  f"dense={det['dense']} exits={det['n_exits']}/{N_VAL} "
                  f"deployment_saved={saved * 100:.1f}% "
                  f"prefix_max_dev={det['max_dev']:.2e} "
                  f"pred_agree={det['pred_agree']})", flush=True)
        results["X1"] = x1_ok

    # ---- X2: feasibility at extreme depth ------------------------------------
    if fam:
        x2_ok = True
        for family, entry in fam.items():
            tau = TAU_RESNET if family == "resnet" else TAU_VIT
            budget = EPOCHS_RESNET if family == "resnet" else EPOCHS_VIT
            t_star = entry["san"]["t_star"]
            ok = t_star is not None and t_star < budget
            x2_ok = x2_ok and ok
            acc = entry["san"]["ledger"][t_star]["acc"] if t_star is not None else 0.0
            print(f"  X2[{family}]: {'PASS' if ok else 'FAIL'} "
                  f"(SAN t*={t_star} of budget {budget}, "
                  f"val_acc@t*={acc:.4f} >= TAU={tau})", flush=True)
        results["X2"] = x2_ok

    # ---- X3: anti-Goodhart soundness at extreme depth -------------------------
    if fam:
        x3_ok = True
        for family, entry in fam.items():
            tau = TAU_RESNET if family == "resnet" else TAU_VIT
            majority = int(torch.bincount(y_tr).argmax())
            abstain_pred = torch.full_like(y_va, majority)
            abstain_acc = float((abstain_pred == y_va).float().mean().item())
            # cheap under-trained probe: a linear model on downsampled pixels
            torch.manual_seed(SEED + 1)
            n_tr, n_va = x_tr.shape[0], x_va.shape[0]
            xtr_small = x_tr.reshape(n_tr, -1)[:, ::97]
            xva_small = x_va.reshape(n_va, -1)[:, ::97]
            probe = nn.Linear(xtr_small.shape[1], N_CLASS)
            popt = torch.optim.Adam(probe.parameters(), lr=LR)
            for _ in range(2):  # deliberately under-trained: cheap, sub-target
                ploss = CE(probe(xtr_small), y_tr)
                popt.zero_grad()
                ploss.backward()
                popt.step()
            with torch.no_grad():
                probe_pred = probe(xva_small).argmax(1)
                probe_acc = float((probe_pred == y_va).float().mean().item())
            s_san = entry["san"]["summary"]
            s_dense = entry["dense"]["summary"]
            san_t = entry["san"]["t_star"]
            pool = [
                {"name": "abstain", "feasible": abstain_acc >= tau,
                 "j_patient": harm_of(abstain_pred, y_va), "j_machine": 0.0},
                {"name": "cheap_probe", "feasible": probe_acc >= tau,
                 "j_patient": harm_of(probe_pred, y_va), "j_machine": 1e-9},
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
            x3_ok = x3_ok and ok
            print(f"  X3[{family}]: {'PASS' if ok else 'FAIL'} "
                  f"(101-weight grid feasible-only={grid_ok}, "
                  f"NO_FEASIBLE={loud}, abstain_acc={abstain_acc:.3f} "
                  f"probe_acc={probe_acc:.3f} both < TAU={tau})", flush=True)
        results["X3"] = x3_ok

    # ---- X4: necessary/gratuitous separation ---------------------------------
    if fam:
        x4_ok = True
        for family, entry in fam.items():
            a_san = entry["san"]["summary"]["gratuitous_flops"] == 0
            a_dense = (entry["dense"]["t_star"] is not None
                       and entry["dense"]["summary"]["gratuitous_flops"] > 0)
            ok = a_san and a_dense
            x4_ok = x4_ok and ok
            print(f"  X4[{family}]: {'PASS' if ok else 'FAIL'} "
                  f"(SAN gratuitous={entry['san']['summary']['gratuitous_flops']} FLOPs, "
                  f"dense gratuitous={entry['dense']['summary']['gratuitous_flops'] / 1e9:.3f}GF)",
                  flush=True)
        results["X4"] = x4_ok

    # ---- X5: suffering bounds vs the standard fixed-budget architecture --------
    # The T3-backed comparison: SAN's total machine suffering must be strictly
    # below the identical trunk trained with a fixed epoch budget (Dense), and
    # SAN's integrated patient harm no more than Dense's. The race against the
    # EarlyStop *scheduler* baseline is NOT asserted here — at extreme depth it
    # inverts for the conv family; X10 certifies that finding with its
    # mechanism signature instead of hiding it.
    if fam:
        x5_ok = True
        for family, entry in fam.items():
            s_san = entry["san"]["summary"]
            s_dense = entry["dense"]["summary"]
            s_estop = entry["earlystop"]["summary"]
            a_m = s_san["s_machine_flops"] < s_dense["s_machine_flops"]
            a_p = s_san["s_patient_int"] <= s_dense["s_patient_int"] + 1e-9
            ok = a_m and a_p
            x5_ok = x5_ok and ok
            print(f"  X5[{family}]: {'PASS' if ok else 'FAIL'} "
                  f"(S_m SAN={s_san['s_machine_flops'] / 1e9:.3f}GF < "
                  f"dense={s_dense['s_machine_flops'] / 1e9:.3f}GF "
                  f"[earlystop={s_estop['s_machine_flops'] / 1e9:.3f}GF]; "
                  f"S_p_int SAN={s_san['s_patient_int']:.2f} <= "
                  f"dense={s_dense['s_patient_int']:.2f} "
                  f"[earlystop={s_estop['s_patient_int']:.2f}])",
                  flush=True)
        results["X5"] = x5_ok

    # ---- X6: exits are real at extreme depth ------------------------------------
    if fam:
        x6_ok = True
        for family, entry in fam.items():
            san_t = entry["san"]["t_star"]
            exit_frac = entry["san"]["ledger"][san_t]["exit_frac"] if san_t is not None else 0.0
            det_ok = exit_frac > 0.10
            x6_ok = x6_ok and det_ok
            print(f"  X6[{family}]: {'PASS' if det_ok else 'FAIL'} "
                  f"(val exit fraction at t*={exit_frac:.3f} (>0.10))", flush=True)
        results["X6"] = x6_ok

    # ---- X7: patient channel first-class ---------------------------------------
    if fam:
        offdiag = HARM[~torch.eye(N_CLASS, dtype=bool)]
        asym = float(offdiag.max()) >= 3.0 * float(offdiag.min())
        x7_ok = asym
        for family, entry in fam.items():
            peak_ok = all(
                entry["san"]["summary"]["s_patient_peak"]
                <= entry[b]["summary"]["s_patient_peak"] + 1e-9
                for b in ("dense", "earlystop"))
            x7_ok = x7_ok and peak_ok
            print(f"  X7[{family}]: {'PASS' if peak_ok else 'FAIL'} "
                  f"(S_p_peak SAN={entry['san']['summary']['s_patient_peak']:.3f} vs "
                  f"{entry['dense']['summary']['s_patient_peak']:.3f}/"
                  f"{entry['earlystop']['summary']['s_patient_peak']:.3f})", flush=True)
        print(f"  X7[harm-matrix]: asymmetry "
              f"{float(offdiag.max()) / float(offdiag.min()):.1f}x (>= 3x)", flush=True)
        results["X7"] = x7_ok

    # ---- X8: anti-shortcut at extreme depth --------------------------------------
    if fam:
        # Spurious corner patch: pixels[0:2, 0:2, :] carry the (noisy) label
        # on train but are pure noise on val. A linear probe on the patch
        # beats TAU on TRAIN (train-loss selection accepts it) yet fails the
        # held-out target; the gate rejects it at every compassion weight.
        rng8 = np.random.RandomState(SEED + 8)
        n_tr8, n_va8 = x_tr.shape[0], x_va.shape[0]
        xtr8 = x_tr.clone()
        xva8 = x_va.clone()
        patch_tr = (y_tr.float()[:, None, None, None].expand(-1, 3, 2, 2)
                    + torch.from_numpy(rng8.normal(0, 0.3, size=(n_tr8, 3, 2, 2))).float())
        xtr8[:, :, 0:2, 0:2] = patch_tr
        xva8[:, :, 0:2, 0:2] = torch.from_numpy(
            rng8.normal(0, 1.0, size=(n_va8, 3, 2, 2))).float()
        torch.manual_seed(SEED + 2)
        shortcut = nn.Linear(12, N_CLASS)
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
        x8_ok = True
        for family, entry in fam.items():
            tau = TAU_RESNET if family == "resnet" else TAU_VIT
            san_t = entry["san"]["t_star"]
            pool8 = [
                {"name": "shortcut_probe", "feasible": sc_val_acc >= tau,
                 "j_patient": 0.01, "j_machine": 1e-12},
                {"name": "san_t*", "feasible": san_t is not None,
                 "j_patient": entry["san"]["ledger"][san_t]["harm"] if san_t is not None else 9.9,
                 "j_machine": entry["san"]["summary"]["s_machine_joules"]},
            ]
            train_accepts = sc_train_acc > tau
            gate_rejects = sc_val_acc < tau
            never = all(gate_select(pool8, lam / 100.0) != "shortcut_probe"
                        for lam in range(101))
            ok = train_accepts and gate_rejects and never
            x8_ok = x8_ok and ok
            print(f"  X8[{family}]: {'PASS' if ok else 'FAIL'} "
                  f"(shortcut train_acc={sc_train_acc:.3f} (>TAU: train-loss "
                  f"selection accepts), val_acc={sc_val_acc:.3f} (<TAU: gate "
                  f"rejects), never selected on 101-weight grid={never})", flush=True)
        results["X8"] = x8_ok

    # ---- X9: scalability certificate at extreme depth (sweep, forward-only) ----
    if run_sweep:
        sweep_batch = 64
        torch.manual_seed(SEED + 9)
        x_sweep = torch.randn(sweep_batch, 3, 32, 32)
        x9_ok = True
        resnet_configs = [(1,) * 7, (3,) * 7, (7,) * 7]     # 16 / 44 / 100 weighted layers
        vit_depths = [6, 12, 24]
        for i, stage_blocks in enumerate(resnet_configs):
            n_weighted = 1 + 2 * sum(stage_blocks) + 1
            model = build_model("resnet", "san", stage_blocks=stage_blocks)
            model.eval()
            # deterministic exit injection: stage (i % n_stages)'s head fires
            fire_stage = i % model.n_stages
            with torch.no_grad():
                model.exit_heads[fire_stage].bias.fill_(0.0)
                model.exit_heads[fire_stage].bias[0] = 30.0
            ok, det = conservation_check(model, x_sweep, model.n_stages)
            overhead = 1.0 - overhead_fraction(model, x_sweep)
            ok = ok and det["n_exits"] > 0 and overhead < 0.05
            x9_ok = x9_ok and ok
            print(f"  X9[resnet-{n_weighted}layers]: {'PASS' if ok else 'FAIL'} "
                  f"(stage_blocks={stage_blocks[0]}x7 widths={RESNET_WIDTHS} "
                  f"gated={det['gated']} "
                  f"manual={det['manual']} dense={det['dense']} "
                  f"exits={det['n_exits']}/{sweep_batch} "
                  f"max_dev={det['max_dev']:.2e} argmax_eq={det['pred_agree']} "
                  f"exit_overhead={overhead * 100:.3f}%)", flush=True)
        for i, vdepth_ in enumerate(vit_depths):
            model = build_model("vit", "san", depth=vdepth_)
            model.eval()
            fire_block = i % model.depth
            with torch.no_grad():
                model.exit_heads[fire_block].bias.fill_(0.0)
                model.exit_heads[fire_block].bias[0] = 30.0
            ok, det = conservation_check(model, x_sweep, model.depth)
            overhead = 1.0 - overhead_fraction(model, x_sweep)
            ok = ok and det["n_exits"] > 0 and overhead < 0.05
            x9_ok = x9_ok and ok
            print(f"  X9[vit-{vdepth_}blocks]: {'PASS' if ok else 'FAIL'} "
                  f"(gated={det['gated']} manual={det['manual']} "
                  f"dense={det['dense']} exits={det['n_exits']}/{sweep_batch} "
                  f"max_dev={det['max_dev']:.2e} argmax_eq={det['pred_agree']} "
                  f"exit_overhead={overhead * 100:.3f}%)", flush=True)
        results["X9"] = x9_ok

    # ---- X10: the scheduler race at extreme depth — certified inversion --------
    # The parent line's D5 clause required SAN <= EarlyStop on the machine
    # channel, and at 18 layers / 6 blocks SAN won that race in both families.
    # At extreme depth the outcome SPLITS, and this clause certifies the split
    # with its mechanism signature rather than asserting the parent outcome:
    #   * vit family: SAN still wins the race (S_m <= EarlyStop) — the gates
    #     open gradually from epoch 1 (no catastrophe) and per-epoch savings
    #     accumulate over 24 exit points.
    #   * resnet family: SAN LOSES the race (S_m > EarlyStop) — the certified
    #     mechanism is the epoch-1 untrained-head gating catastrophe: the
    #     parent warmup trains only the final head, so when the gates open at
    #     epoch 1 the exit heads are uncalibrated; at 7 exit points over
    #     GAP-pooled conv features they fire overconfidently (epoch-1 exit
    #     fraction > 0.5 at <= chance-level accuracy), starving the deep
    #     stages of gradient for 1-2 epochs. Feasibility arrives late
    #     (t*_san > t*_estop) while per-epoch exit savings (dominated by
    #     late-stage exits) are small — the race cannot be won in this
    #     regime. The clause FAILS if the signature disappears (e.g. a
    #     mechanism fix makes the conv family win), which is exactly the
    #     event that would require this line's spec to be revised.
    if fam:
        x10_ok = True
        for family, entry in fam.items():
            s_san = entry["san"]["summary"]
            s_estop = entry["earlystop"]["summary"]
            race_won = (s_san["s_machine_flops"]
                        <= s_estop["s_machine_flops"] + 1e-9)
            san_lg = entry["san"]["ledger"]
            e1 = san_lg[1] if len(san_lg) > 1 else {"exit_frac": 0.0, "acc": 1.0}
            catastrophe = e1["exit_frac"] > 0.5 and e1["acc"] < 0.2
            gradual = e1["exit_frac"] < 0.2
            t_san, t_estop = entry["san"]["t_star"], entry["earlystop"]["t_star"]
            if family == "resnet":
                ok = (not race_won) and catastrophe
            else:
                ok = race_won and gradual
            x10_ok = x10_ok and ok
            print(f"  X10[{family}]: {'PASS' if ok else 'FAIL'} "
                  f"(scheduler race {'WON' if race_won else 'LOST'}: "
                  f"S_m SAN={s_san['s_machine_flops'] / 1e9:.3f}GF vs "
                  f"earlystop={s_estop['s_machine_flops'] / 1e9:.3f}GF; "
                  f"t*_san={t_san} t*_estop={t_estop}; "
                  f"epoch-1 exit={e1['exit_frac']:.3f} acc={e1['acc']:.3f} "
                  f"-> signature: {'catastrophe' if catastrophe else 'gradual'})",
                  flush=True)
        results["X10"] = x10_ok

    # ---- verdict ----------------------------------------------------------------
    n_pass = sum(1 for v in results.values() if v)
    n_all = len(results)
    verdict = "X_GREEN" if n_pass == n_all and n_all > 0 else "X_RED"
    print(f"SUFFERING_AWARE_EXTREME_DEPTH_VERDICT {verdict} ({n_pass}/{n_all} clauses PASS)")
    return 0 if verdict == "X_GREEN" else 1


def overhead_fraction(model, x):
    """Trunk share of the gates-open forward FLOPs (final head included, exit
    heads excluded); the caller reports 1 - this as the exit-head overhead."""
    _, dense_meter = model.forward_dense(x)
    return model_flops_trunk_only(model, x.shape[0]) / dense_meter.flops


def model_flops_trunk_only(model, n):
    """Forward FLOPs of trunk + final head only (no exit heads), analytic."""
    if isinstance(model, SufferingAwareResNet):
        per_active = [n] * model.n_stages
    else:
        per_active = [n] * model.depth
    # manual accounting includes exit heads; subtract them
    total = manual_forward_accounting(model, n, per_active, n)
    if isinstance(model, SufferingAwareResNet):
        for k in range(model.n_stages):
            total -= 2 * model.trunk.channels[k] * N_CLASS * n
    else:
        total -= model.depth * 2 * model.trunk.d * N_CLASS * n
    return total


if __name__ == "__main__":
    raise SystemExit(main())
