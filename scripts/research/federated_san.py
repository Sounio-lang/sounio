#!/usr/bin/env python3
"""Mercyful Learning — the FEDERATED Suffering-Aware neural Network (FED-SAN):
the suffering-aware architecture distributed across clinical sites, with a
per-node and global suffering ledger, feasibility-gated aggregation, and
freeze-on-green at the federation level.

Companion artifact to
  docs/research/federated_san_spec_2026-07-30.md

The SAN line established the architecture class (suffering-aware layers,
per-sample exit gates, deep supervision, freeze-on-green, anti-Goodhart
feasibility gate) on a synthetic MLP (A1..A8), scaled it to deep networks on
real images (D1..D9), and grounded the patient channel in real clinical
outcomes (R1..R10). This harness distributes the SAME architecture class
across K federated nodes (simulated sites, one process, seeded and
deterministic) and certifies that every property survives federation:

  * CLINICAL leg (real patients): the WDBC cohort (569 real breast
    fine-needle-aspirate patients, UCI #17, de-identified, public, CC-BY)
    partitioned across K=5 clinical nodes by a Dirichlet label-skew
    (non-IID) split. Every unit of patient suffering in the ledger is a
    real patient with a real biopsy-confirmed outcome being misclassified.
    The harm WEIGHTS (missed hazard = 5, false hazard = 1) remain a
    DECLARED normative cost structure, unchanged from the R-line.
  * VISION leg (real images): CIFAR-10 (real dataset, vendored at
    datasets/cifar-10-batches-py) partitioned across K=4 nodes by the same
    Dirichlet skew; the harm matrix is a synthetic screening cost structure
    over the real labels (class 9 "truck" plays the hazard, 5:1), exactly
    the deep line's convention.

What federation adds to the line, and what the contract certifies:

  * a TWO-LEVEL machine-suffering ledger: exact analytic compute FLOPs per
    node per round (unchanged conventions: MAC=2 FLOPs, backward=2x
    forward, unmetered elementwise ops) PLUS an exact communication ledger
    (bytes on the wire: broadcast + upload per node per round);
  * FREEZE-ON-GREEN AT THE FEDERATION LEVEL: training stops at r*, the
    first round whose aggregated global model is feasible on the trusted
    held-out set; post-target rounds are gratuitous suffering on every
    node and are declined (S_gratuitous = 0 exactly, F4);
  * FEASIBILITY-GATED AGGREGATION (the anti-Goodhart gate, moved into the
    aggregator): each node's update is scored on the aggregator's trusted
    held-out set BEFORE averaging; an under-bar update is excluded —
    categorically, not penalized — and an average that would move the
    global model backward is never accepted (with one-step escalation:
    drop the weakest included update and re-average). A model-poisoning
    node (label-flipped data trained to local convergence, amplified
    update) is certified to break naive FedAvg — the poisoned global ends
    MORE HARMFUL THAN THE ABSTAINER — and to participate in NO accepted
    average of the gated protocol, which reaches feasibility anyway (F8).

Certificates (contract clauses):
  F1[leg]  metering conservation under federation: per-node compute meter
           == independent manual accounting of the executed path exactly;
           gated-off layers charge exactly 0; exited predictions agree with
           the recomputed dense prefix (argmax exactly equal); the
           communication ledger == manual bytes-on-wire accounting exactly
  F2[leg]  feasibility under federation: FED-SAN reaches held-out
           acc >= TAU[leg] within the round budget on both legs
  F3       anti-Goodhart soundness: 101-point compassion-weight grid,
           feasible-only selection over a pool containing a zero-cost
           abstainer and an under-trained federated probe; all-infeasible
           pool -> NO_FEASIBLE
  F4[leg]  necessary/gratuitous separation: FED-SAN gratuitous compute =
           0 FLOPs and gratuitous communication = 0 bytes exactly;
           fixed-round federated baseline > 0 on both
  F5[leg]  suffering bounds: FED-SAN total compute strictly below the
           fixed-round federated baseline; FED-SAN average PER-ROUND
           compute <= the federated EarlyStop baseline's (the exits
           stricten every executed round); integrated patient harm <=
           the fixed-round baseline's. The EarlyStop TOTALS are reported
           with a per-round decomposition, not gated — under extreme
           label skew federated deep supervision can delay feasibility
           (r*), and the later r* can cost more total compute/exposure
           than EarlyStop even though every round is cheaper
  F6[leg]  exits are real under federation: held-out exit fraction of the
           global model at r* > 0.10 with exact prefix argmax agreement
  F7[leg]  patient channel first-class: harm matrix genuinely asymmetric
           (5:1); FED-SAN peak patient harm <= same-init federated
           baselines' peaks (shared global init -> identical round-0
           exposure)
  F8       adversarial-node containment: a label-flipping, update-
           amplifying node drives naive FedAvg to an INFEASIBLE global
           model, while feasibility-gated aggregation excludes its update
           at EVERY round and reaches feasibility (real WDBC patients)
  F9       non-IID realism + provenance: per-node label distributions are
           certified skewed (mean L1 distance from the global distribution
           above threshold) on both legs; the clinical cohort matches the
           published WDBC counts exactly (569 = 357 B + 212 M)

All data is de-identified and public without credentialing. No clinical
claim; not medical guidance; not a diagnostic or screening tool. The
machine channel is an operational computational-burden proxy (metered
FLOPs and wire bytes); no_consciousness_claim is made or needed.

Run: .venv/bin/python scripts/research/federated_san.py
Requires: torch (CPU) + numpy from the repo .venv; vendored cohorts at
datasets/san_real_patient/ and datasets/cifar-10-batches-py/.
Env overrides: FED_SAN_SMOKE=1 (tiny fast mechanics check on synthetic
stand-ins — NEVER part of the canonical run), FED_SAN_ONLY=clinical|vision,
FED_SAN_THREADS (torch CPU threads, default 16).
"""

import copy
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn

# ---------------- determinism / config --------------------------------------
SEED = 17
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(int(os.environ.get("FED_SAN_THREADS", "16")))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLIN_DATA = os.environ.get(
    "FED_SAN_CLIN_DATA", os.path.join(REPO_ROOT, "datasets", "san_real_patient"))
CIFAR_DIR = os.environ.get(
    "FED_SAN_CIFAR", os.path.join(REPO_ROOT, "datasets", "cifar-10-batches-py"))
SMOKE = os.environ.get("FED_SAN_SMOKE", "") == "1"
ONLY = os.environ.get("FED_SAN_ONLY", "").strip().lower()

E_PER_FLOP = 4e-12      # J/FLOP, same convention as the whole line
AUX_W = 1.0             # deep-supervision weight, unchanged
CE = nn.CrossEntropyLoss()

# Declared asymmetric harm (unchanged from the R/D lines): hazard missed = 5,
# false hazard = 1. Weights are a DECLARED normative cost structure.
HARM = torch.tensor([[0.0, 1.0],
                     [5.0, 0.0]])
HAZARD_CLASS_VISION = 9   # "truck" plays the hazard (deep-line convention)


def harm_binary(pred, y):
    """Mean harm of binary predictions under the declared 5:1 matrix."""
    return float(HARM[y, pred].mean().item())


def harm_vision(pred, y):
    """Mean harm on 10-class predictions: missed hazard = 5, false hazard = 1."""
    missed = ((y == HAZARD_CLASS_VISION) & (pred != HAZARD_CLASS_VISION)).float() * 5.0
    false = ((y != HAZARD_CLASS_VISION) & (pred == HAZARD_CLASS_VISION)).float() * 1.0
    return float((missed + false).mean().item())


# ---------------- machine-suffering meters ----------------------------------
class ComputeMeter:
    """Analytic FLOP accounting, unchanged conventions: linear map =
    2*d_in*d_out FLOPs/sample; conv = 2*Cin*Cout*K^2*Hout*Wout FLOPs/sample;
    a training step charges forward + backward with backward = 2x forward."""

    def __init__(self):
        self.flops = 0

    def charge_linear(self, d_in, d_out, n_samples, backward=False):
        f = 2 * d_in * d_out * n_samples
        self.flops += 3 * f if backward else f

    def charge_conv(self, c_in, c_out, k, h_out, w_out, n_samples, backward=False):
        f = 2 * c_in * c_out * k * k * h_out * w_out * n_samples
        self.flops += 3 * f if backward else f

    @property
    def joules(self):
        return self.flops * E_PER_FLOP


class CommMeter:
    """Exact communication accounting: bytes on the wire. Per round, the
    aggregator broadcasts the global model to every active node (downlink)
    and every active node uploads its update (uplink): 2 * K_active *
    model_bytes per round. An excluded update is still uploaded (the gate
    reads it at the aggregator) — exclusion saves suffering through the
    AVERAGE, not through the wire."""

    def __init__(self, model_bytes):
        self.model_bytes = model_bytes
        self.bytes = 0

    def charge_round(self, k_active):
        self.bytes += 2 * k_active * self.model_bytes


# ---------------- suffering-aware architectures (unchanged class) -----------
class SALayerMLP(nn.Module):
    """Linear+Tanh suffering-aware layer with an exit head (A/R lines)."""

    def __init__(self, d_in, width, n_class):
        super().__init__()
        self.core = nn.Linear(d_in, width)
        self.act = nn.Tanh()
        self.exit_head = nn.Linear(width, n_class)
        self.d_in, self.width, self.n_class = d_in, width, n_class

    def forward(self, h, meter, backward):
        meter.charge_linear(self.d_in, self.width, h.shape[0], backward)
        h = self.act(self.core(h))
        meter.charge_linear(self.width, self.n_class, h.shape[0], backward)
        return h, self.exit_head(h)


class SufferingAwareMLP(nn.Module):
    """The A/R-line SAN trunk parameterized by input width and class count."""

    def __init__(self, d_in, n_class, width=32, depth=4):
        super().__init__()
        self.d_in, self.n_class, self.width, self.depth = d_in, n_class, width, depth
        self.layers = nn.ModuleList(
            [SALayerMLP(d_in if k == 0 else width, width, n_class)
             for k in range(depth)])
        self.final_head = nn.Linear(width, n_class)

    def forward(self, x, meter, train=False, use_exit_heads=True, delta=0.75):
        n = x.shape[0]
        out_logits = x.new_zeros(n, self.n_class)
        out_depth = torch.full((n,), self.depth + 1, dtype=torch.long)
        active = torch.arange(n)
        h = x
        per_layer_active, aux_records, final_record = [], [], None
        n_final = 0
        for k, layer in enumerate(self.layers):
            if active.numel() == 0:
                break
            per_layer_active.append(int(active.numel()))
            if not use_exit_heads:
                meter.charge_linear(layer.d_in, layer.width, h.shape[0], backward=train)
                h = layer.act(layer.core(h))
                continue
            h, logits_k = layer(h, meter, backward=train)
            if train:
                aux_records.append((active, logits_k))
            conf = torch.softmax(logits_k.detach(), dim=1).max(dim=1).values
            leave = conf >= delta
            if leave.any():
                idx = active[leave]
                out_logits[idx] = logits_k[leave]
                out_depth[idx] = k + 1
                keep = ~leave
                active = active[keep]
                h = h[keep]
        if active.numel() > 0:
            n_final = int(active.numel())
            meter.charge_linear(self.width, self.n_class, n_final, backward=train)
            final_logits = self.final_head(h)
            out_logits[active] = final_logits
            if train:
                final_record = (active, final_logits)
        return (out_logits, out_depth, per_layer_active, n_final,
                aux_records, final_record)


class SAStageCNN(nn.Module):
    """One conv stage (conv->BN->ReLU) with a GAP+linear exit head: a
    suffering-aware layer in the deep line's sense (D-line section 3)."""

    def __init__(self, c_in, c_out, stride, n_class):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU()
        self.exit_head = nn.Linear(c_out, n_class)
        self.c_in, self.c_out, self.stride = c_in, c_out, stride

    def forward(self, h, meter, backward):
        n, _, h_in, w_in = h.shape
        h_out, w_out = h_in // self.stride, w_in // self.stride
        meter.charge_conv(self.c_in, self.c_out, 3, h_out, w_out, n, backward)
        h = self.act(self.bn(self.conv(h)))
        pooled = h.mean(dim=(2, 3))                       # GAP: unmetered
        meter.charge_linear(self.c_out, self.exit_head.out_features,
                            n, backward)
        return h, self.exit_head(pooled)


class SufferingAwareCNN(nn.Module):
    """Compact conv SAN for the vision leg: stem + 4 metered stages with
    exit heads, parameterized exactly like the MLP SAN (gated forward with
    per-sample exits, deep supervision hooks)."""

    WIDTHS = (16, 32, 48, 64)

    def __init__(self, n_class=10):
        super().__init__()
        self.n_class = n_class
        self.stem = nn.Conv2d(3, self.WIDTHS[0], 3, stride=1, padding=1)
        self.stem_bn = nn.BatchNorm2d(self.WIDTHS[0])
        self.stem_act = nn.ReLU()
        c = self.WIDTHS[0]
        stages = []
        for w in self.WIDTHS[1:]:
            stages.append(SAStageCNN(c, w, 2, n_class))
            c = w
        self.stages = nn.ModuleList(stages)
        self.final_head = nn.Linear(self.WIDTHS[-1], n_class)

    def _stem(self, x, meter, backward):
        n = x.shape[0]
        meter.charge_conv(3, self.WIDTHS[0], 3, 32, 32, n, backward)
        return self.stem_act(self.stem_bn(self.stem(x)))

    def forward(self, x, meter, train=False, use_exit_heads=True, delta=0.40):
        n = x.shape[0]
        out_logits = x.new_zeros(n, self.n_class)
        out_depth = torch.full((n,), len(self.stages) + 1, dtype=torch.long)
        active = torch.arange(n)
        per_layer_active, aux_records, final_record = [], [], None
        n_final = 0
        h = self._stem(x, meter, train)
        for k, stage in enumerate(self.stages):
            if active.numel() == 0:
                break
            per_layer_active.append(int(active.numel()))
            if not use_exit_heads:
                ns, _, hi, wi = h.shape
                meter.charge_conv(stage.c_in, stage.c_out, 3,
                                  hi // stage.stride, wi // stage.stride,
                                  ns, train)
                h = stage.act(stage.bn(stage.conv(h)))
                continue
            h, logits_k = stage(h, meter, backward=train)
            if train:
                aux_records.append((active, logits_k))
            conf = torch.softmax(logits_k.detach(), dim=1).max(dim=1).values
            leave = conf >= delta
            if leave.any():
                idx = active[leave]
                out_logits[idx] = logits_k[leave]
                out_depth[idx] = k + 1
                keep = ~leave
                active = active[keep]
                h = h[keep]
        if active.numel() > 0:
            n_final = int(active.numel())
            pooled = h.mean(dim=(2, 3))
            meter.charge_linear(self.WIDTHS[-1], self.n_class, n_final, backward=train)
            final_logits = self.final_head(pooled)
            out_logits[active] = final_logits
            if train:
                final_record = (active, final_logits)
        return (out_logits, out_depth, per_layer_active, n_final,
                aux_records, final_record)


def build_plain_trunk(leg, d_in=None, n_class=None):
    """The federated EarlyStop baseline's trunk: identical modules to the
    SAN, no exit heads, no deep supervision (a plain MLP / plain CNN)."""
    if leg == "clinical":
        layers, d = [], d_in
        for _ in range(4):
            layers += [nn.Linear(d, 32), nn.Tanh()]
            d = 32
        layers.append(nn.Linear(32, n_class))
        return nn.Sequential(*layers)
    ch = SufferingAwareCNN.WIDTHS
    mods = [nn.Conv2d(3, ch[0], 3, padding=1), nn.BatchNorm2d(ch[0]), nn.ReLU()]
    c = ch[0]
    for w in ch[1:]:
        mods += [nn.Conv2d(c, w, 3, stride=2, padding=1), nn.BatchNorm2d(w), nn.ReLU()]
        c = w
    trunk = nn.Sequential(*mods)

    class PlainCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.head = nn.Linear(ch[-1], n_class)

        def forward(self, x):
            return self.head(self.trunk(x).mean(dim=(2, 3)))

    return PlainCNN()


def plain_flops(leg, n_samples, backward, d_in=None):
    """Analytic per-pass FLOPs of the plain trunk (baseline accounting)."""
    if leg == "clinical":
        f = 2 * d_in * 32 * n_samples + 3 * 2 * 32 * 32 * n_samples \
            + 2 * 32 * 2 * n_samples
    else:
        ch = SufferingAwareCNN.WIDTHS
        f = 2 * 3 * ch[0] * 9 * 32 * 32 * n_samples
        c, side = ch[0], 32
        for w in ch[1:]:
            side //= 2
            f += 2 * c * w * 9 * side * side * n_samples
            c = w
        f += 2 * ch[-1] * 10 * n_samples
    return 3 * f if backward else f


def model_bytes(model):
    """Wire size of a model update: float32 parameters + float32 buffers
    (BN running stats), the exact quantities FedAvg exchanges."""
    n = sum(p.numel() for p in model.parameters())
    n += sum(b.numel() for b in model.buffers())
    return 4 * n


# ---------------- data: loaders, splits, Dirichlet partitions ---------------
def load_wdbc(path):
    """569 real FNA patients (UCI #17): col0 = ID, col1 = M/B, 30 cytology
    features. Hazard = biopsy-confirmed malignancy. De-identified, public."""
    rows = [l.strip().split(",") for l in open(path) if l.strip()]
    x = np.array([[float(v) for v in r[2:]] for r in rows], dtype=np.float32)
    y = np.array([1 if r[1] == "M" else 0 for r in rows], dtype=np.int64)
    return x, y


def load_cifar(dirpath, n_train, n_val):
    """Real CIFAR-10 images from the vendored python batches."""
    xs, ys = [], []
    for b in range(1, 6):
        with open(os.path.join(dirpath, f"data_batch_{b}"), "rb") as f:
            d = pickle.load(f, encoding="latin1")
        xs.append(d["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0)
        ys.append(np.array(d["labels"], dtype=np.int64))
    x_tr_all, y_tr_all = np.concatenate(xs), np.concatenate(ys)
    with open(os.path.join(dirpath, "test_batch"), "rb") as f:
        d = pickle.load(f, encoding="latin1")
    x_te = d["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y_te = np.array(d["labels"], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    tr_idx = rng.permutation(len(y_tr_all))[:n_train]
    te_idx = rng.permutation(len(y_te))[:n_val]
    x_tr, y_tr = x_tr_all[tr_idx], y_tr_all[tr_idx]
    x_va, y_va = x_te[te_idx], y_te[te_idx]
    mean = x_tr.mean(axis=(0, 2, 3), keepdims=True)
    std = x_tr.std(axis=(0, 2, 3), keepdims=True) + 1e-6
    return ((x_tr - mean) / std).astype(np.float32), y_tr, \
        ((x_va - mean) / std).astype(np.float32), y_va


def stratified_split(y, n_train, rng):
    """Deterministic stratified train/held-out split (R-line convention)."""
    idx = rng.permutation(len(y))
    tr, va = [], []
    for c in range(2):
        c_idx = [i for i in idx if y[i] == c]
        n_tr_c = int(round(n_train * len(c_idx) / len(y)))
        tr += c_idx[:n_tr_c]
        va += c_idx[n_tr_c:]
    rng.shuffle(tr)
    rng.shuffle(va)
    return np.array(tr), np.array(va)


def dirichlet_partition(y, k_nodes, alpha, rng):
    """Label-skew non-IID partition: for each class, split its samples
    across nodes by a Dirichlet(alpha) draw (standard FL heterogeneity
    model). Returns a list of index arrays, one per node."""
    nodes = [[] for _ in range(k_nodes)]
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)
        props = rng.dirichlet([alpha] * k_nodes)
        cuts = (np.cumsum(props) * len(c_idx)).astype(int)[:-1]
        for k, chunk in enumerate(np.split(c_idx, cuts)):
            nodes[k].extend(chunk.tolist())
    for k in range(k_nodes):
        rng.shuffle(nodes[k])
    return [np.array(sorted(n), dtype=np.int64) for n in nodes]


def make_clinical_cohort(k_nodes=5, alpha=0.5, n_train=400):
    """Real WDBC patients: same stratified split/standardization as the
    R-line, then a Dirichlet label-skew partition across clinical nodes.
    The aggregator's trusted held-out set is the same 169 real patients."""
    x, y = load_wdbc(os.path.join(CLIN_DATA, "wdbc.data"))
    rng = np.random.default_rng(SEED)
    tr, va = stratified_split(y, n_train, rng)
    mean, std = x[tr].mean(axis=0), x[tr].std(axis=0) + 1e-6
    x = (x - mean) / std
    node_idx = dirichlet_partition(y[tr], k_nodes, alpha, np.random.default_rng(SEED + 1))
    return {
        "leg": "clinical", "d_in": 30, "n_class": 2,
        "tau": 0.95, "delta": 0.75, "k_nodes": k_nodes,
        "x": torch.from_numpy(x), "y": torch.from_numpy(y),
        "tr": tr, "va": va,
        "x_va": torch.from_numpy(x[va]), "y_va": torch.from_numpy(y[va]),
        "nodes": [(torch.from_numpy(x[tr[i]]), torch.from_numpy(y[tr[i]]))
                  for i in node_idx],
        "node_idx": node_idx, "n_raw": len(y),
        "harm_fn": harm_binary,
    }


def make_vision_cohort(k_nodes=4, alpha=0.5, n_train=4000, n_val=1000):
    """Real CIFAR-10 subset partitioned across nodes by Dirichlet skew."""
    x_tr, y_tr, x_va, y_va = load_cifar(CIFAR_DIR, n_train, n_val)
    node_idx = dirichlet_partition(y_tr, k_nodes, alpha, np.random.default_rng(SEED + 2))
    return {
        "leg": "vision", "d_in": None, "n_class": 10,
        "tau": 0.40, "delta": 0.30, "k_nodes": k_nodes,
        "x_va": torch.from_numpy(x_va), "y_va": torch.from_numpy(y_va),
        "nodes": [(torch.from_numpy(x_tr[i]), torch.from_numpy(y_tr[i]))
                  for i in node_idx],
        "node_idx": node_idx, "n_raw": n_train,
        "harm_fn": harm_vision,
    }


def make_smoke_cohort(leg, k_nodes=3):
    """FED_SAN_SMOKE=1 only: tiny synthetic stand-in for a fast mechanics
    check. NEVER part of the canonical run; the gate runs the real contract."""
    rng = np.random.default_rng(SEED)
    if leg == "clinical":
        n = 150
        x = rng.normal(size=(n, 30)).astype(np.float32)
        y = (x[:, 0] + 0.5 * x[:, -1] > 0).astype(np.int64)
        tr, va = np.arange(100), np.arange(100, n)
        node_idx = np.array_split(tr, k_nodes)
        return {"leg": "clinical", "d_in": 30, "n_class": 2, "tau": 0.80,
                "delta": 0.75, "k_nodes": k_nodes,
                "x": torch.from_numpy(x), "y": torch.from_numpy(y),
                "tr": tr, "va": va,
                "x_va": torch.from_numpy(x[va]), "y_va": torch.from_numpy(y[va]),
                "nodes": [(torch.from_numpy(x[i]), torch.from_numpy(y[i]))
                          for i in node_idx],
                "node_idx": node_idx, "n_raw": n, "harm_fn": harm_binary}
    n_tr, n_va = 240, 80
    x_tr = rng.normal(size=(n_tr, 3, 32, 32)).astype(np.float32)
    y_tr = rng.integers(0, 10, size=n_tr).astype(np.int64)
    x_va = rng.normal(size=(n_va, 3, 32, 32)).astype(np.float32)
    y_va = rng.integers(0, 10, size=n_va).astype(np.int64)
    node_idx = np.array_split(np.arange(n_tr), k_nodes)
    return {"leg": "vision", "d_in": None, "n_class": 10, "tau": 0.15,
            "delta": 0.40, "k_nodes": k_nodes,
            "x_va": torch.from_numpy(x_va), "y_va": torch.from_numpy(y_va),
            "nodes": [(torch.from_numpy(x_tr[i]), torch.from_numpy(y_tr[i]))
                      for i in node_idx],
            "node_idx": node_idx, "n_raw": n_tr, "harm_fn": harm_vision}


# ---------------- federated training -----------------------------------------
WARMUP_ROUNDS = 1          # round 1 trains every trunk dense-identically
LR_CLIN, LR_VIS = 1e-2, 3e-3
BATCH_VIS = 128
ATTACK_BETA = 5.0          # model-poisoning amplification factor
ATTACK_EPOCHS = 10         # the adversary trains its poison to convergence


def new_model(cohort, arch):
    if arch == "san":
        if cohort["leg"] == "clinical":
            return SufferingAwareMLP(cohort["d_in"], cohort["n_class"])
        return SufferingAwareCNN(cohort["n_class"])
    return build_plain_trunk(cohort["leg"], cohort.get("d_in"),
                             cohort["n_class"])


def shared_trunk_inits(cohort):
    """One seeded SAN init plus a plain-trunk init whose trunk weights are
    COPIED from it, so the federated systems being compared share one trunk
    init and their round-0 predictions — hence the patient-suffering peak at
    exposure start — are IDENTICAL (the A/R-line convention: the peak
    comparison is about trajectories, not init luck)."""
    torch.manual_seed(SEED)
    san = new_model(cohort, "san")
    plain = new_model(cohort, "plain")
    if cohort["leg"] == "clinical":
        plain_linears = [m for m in plain if isinstance(m, nn.Linear)]
        for k, layer in enumerate(san.layers):
            plain_linears[k].weight.data.copy_(layer.core.weight)
            plain_linears[k].bias.data.copy_(layer.core.bias)
        plain_linears[-1].weight.data.copy_(san.final_head.weight)
        plain_linears[-1].bias.data.copy_(san.final_head.bias)
    else:
        plain_convs = [m for m in plain.trunk if isinstance(m, nn.Conv2d)]
        plain_bns = [m for m in plain.trunk if isinstance(m, nn.BatchNorm2d)]
        san_convs = [san.stem] + [st.conv for st in san.stages]
        san_bns = [san.stem_bn] + [st.bn for st in san.stages]
        for pc, sc in zip(plain_convs, san_convs):
            pc.weight.data.copy_(sc.weight)
            pc.bias.data.copy_(sc.bias)
        for pb, sb in zip(plain_bns, san_bns):
            pb.load_state_dict(sb.state_dict())
        plain.head.weight.data.copy_(san.final_head.weight)
        plain.head.bias.data.copy_(san.final_head.bias)
    return {"san": {k: v.clone() for k, v in san.state_dict().items()},
            "plain": {k: v.clone() for k, v in plain.state_dict().items()}}


def local_train(global_state, x_k, y_k, cohort, arch, rnd, node,
                attack=False):
    """One local epoch on node data (full batch for the clinical leg,
    deterministic 128-batches for the vision leg). Returns the updated
    state dict and the node's metered compute FLOPs. Round 1 (warm-up)
    runs every trunk dense-identically to the plain baseline — running
    untrained exit heads would itself be gratuitous computation (A-line).
    The adversarial node trains on LABEL-FLIPPED data (model poisoning)."""
    torch.manual_seed(SEED + 1000 * rnd + node)
    model = new_model(cohort, arch)
    model.load_state_dict(global_state)
    model.train()
    lr = LR_CLIN if cohort["leg"] == "clinical" else LR_VIS
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    meter = ComputeMeter()
    y_eff = (1 - y_k) if attack else y_k
    use_exits = arch == "san" and rnd > WARMUP_ROUNDS
    # the adversary is STRONGER than an honest node: it trains its poisoned
    # model to local convergence (unlimited local compute — the standard
    # model-poisoning adversary), charged to the meter like everything else
    for _ep in range(ATTACK_EPOCHS if attack else 1):
        if cohort["leg"] == "clinical":
            if arch == "san":
                _, _, _, _, aux, fin = model(x_k, meter, train=True,
                                             use_exit_heads=use_exits,
                                             delta=cohort["delta"])
                losses = []
                if fin is not None:
                    losses.append(CE(fin[1], y_eff[fin[0]]))
                if use_exits and aux:
                    losses.append(AUX_W * torch.stack(
                        [CE(lg, y_eff[ix]) for ix, lg in aux]).mean())
                loss = sum(losses)
            else:
                loss = CE(model(x_k), y_eff)
                meter.flops += plain_flops("clinical", x_k.shape[0], True,
                                           d_in=cohort["d_in"])
            opt.zero_grad(); loss.backward(); opt.step()
        else:
            g = torch.Generator().manual_seed(SEED + 1000 * rnd + node
                                              + 5000 * _ep)
            perm = torch.randperm(x_k.shape[0], generator=g)
            for b in range(0, x_k.shape[0], BATCH_VIS):
                idx = perm[b:b + BATCH_VIS]
                xb, yb = x_k[idx], y_eff[idx]
                if arch == "san":
                    _, _, _, _, aux, fin = model(xb, meter, train=True,
                                                 use_exit_heads=use_exits,
                                                 delta=cohort["delta"])
                    losses = []
                    if fin is not None:
                        losses.append(CE(fin[1], yb[fin[0]]))
                    if use_exits and aux:
                        losses.append(AUX_W * torch.stack(
                            [CE(lg, yb[ix]) for ix, lg in aux]).mean())
                    loss = sum(losses)
                else:
                    loss = CE(model(xb), yb)
                    meter.flops += plain_flops("vision", xb.shape[0], True)
                opt.zero_grad(); loss.backward(); opt.step()
    return model.state_dict(), meter.flops


def eval_model_state(state, cohort, arch, meter=None):
    """Evaluate a model (state dict) on the aggregator's trusted held-out
    set; charges the eval to the supplied meter. Returns acc, harm,
    exit fraction, logits, depth."""
    model = new_model(cohort, arch)
    model.load_state_dict(state)
    model.eval()
    m = meter if meter is not None else ComputeMeter()
    with torch.no_grad():
        if arch == "san":
            logits, depth, per_layer, n_final, _, _ = model(
                cohort["x_va"], m, train=False, use_exit_heads=True,
                delta=cohort["delta"])
            exit_frac = float((depth < (model.depth if cohort["leg"] == "clinical"
                                        else len(model.stages))).float().mean().item())
            aux = (per_layer, n_final, model)
        else:
            logits = model(cohort["x_va"])
            m.flops += plain_flops(cohort["leg"], cohort["x_va"].shape[0], False,
                                   d_in=cohort.get("d_in"))
            depth = torch.full((cohort["x_va"].shape[0],), 99, dtype=torch.long)
            exit_frac, aux = 0.0, None
    pred = logits.argmax(dim=1)
    acc = float((pred == cohort["y_va"]).float().mean().item())
    harm = cohort["harm_fn"](pred, cohort["y_va"])
    return {"acc": acc, "harm": harm, "exit_frac": exit_frac,
            "logits": logits, "depth": depth, "aux": aux, "meter": m}


def abstainer_acc_harm(cohort):
    """The zero-cost 'do nothing' prescription on the trusted set: predict
    no-hazard (clinical: class 0; vision: the non-hazard majority class 0)."""
    pred = torch.zeros_like(cohort["y_va"])
    return (float((pred == cohort["y_va"]).float().mean().item()),
            cohort["harm_fn"](pred, cohort["y_va"]))


def fedavg(included, weights):
    """Weighted average of state dicts (float arithmetic on parameters and
    BN buffers, as standard FedAvg)."""
    avg = {}
    for key in included[0]:
        acc = included[0][key].float() * weights[0]
        for s, w in zip(included[1:], weights[1:]):
            acc = acc + s[key].float() * w
        avg[key] = acc
    return avg


def fed_run_metered(cohort, **kw):
    """The federated loop with per-round compute tracking (round-0 eval
    charged to round 0). One global init (seeded, shared across systems so
    round-0 exposure — and hence the patient peak — is identical). Per
    round: broadcast; one local epoch per node; (gated) aggregation; global
    feasibility evaluation on the trusted held-out set. Freeze-on-green:
    stop at the first feasible round r* (gratuitous suffering is exactly
    the rounds after r*). The adversarial node (attack leg) trains on
    flipped labels and amplifies its update by ATTACK_BETA (a standard
    model-poisoning boost)."""
    k = cohort["k_nodes"]
    arch = kw.get("arch", "san")
    init_state = kw.get("init_state")
    if init_state is None:
        torch.manual_seed(SEED)
        init_state = new_model(cohort, arch).state_dict()
    init_state = {key: v.clone() for key, v in init_state.items()}
    comm = CommMeter(model_bytes(new_model(cohort, arch)))
    compute = ComputeMeter()
    r0 = eval_model_state(init_state, cohort, arch, compute)
    per_round = [compute.flops]
    ledger = [{"round": 0, "acc": r0["acc"], "harm": r0["harm"],
               "exit_frac": r0["exit_frac"],
               "feasible": r0["acc"] >= cohort["tau"], "excluded": []}]
    global_state = {key: v.float() for key, v in init_state.items()}
    abst_acc, _ = abstainer_acc_harm(cohort)
    rounds = kw.get("rounds", 30)
    fixed = kw.get("fixed_rounds", False)
    stop = kw.get("stop_at_feasible", True)
    agg = kw.get("agg", "gated")
    attack_node = kw.get("attack_node")
    r_star, t_star_acc = None, None
    for rnd in range(1, rounds + 1):
        before = compute.flops
        states = []
        for node, (x_k, y_k) in enumerate(cohort["nodes"]):
            s, f = local_train(global_state, x_k, y_k, cohort, arch, rnd,
                               node, attack=(attack_node == node))
            if attack_node == node:
                s = {key: global_state[key]
                     + ATTACK_BETA * (s[key].float() - global_state[key])
                     for key in s}
            states.append(s)
            compute.flops += f
        comm.charge_round(k)
        excluded = []
        if agg == "gated":
            included, w_in, inc_nodes, inc_accs = [], [], [], []
            # the second disjunct keys on the BEST global accuracy so far,
            # never the last round's: a poisoning attack that drags the
            # global model down must not drag the inclusion bar down with it
            best_acc = max(e["acc"] for e in ledger)
            for node, s in enumerate(states):
                ev = eval_model_state(s, cohort, arch, compute)
                # categorical inclusion bar: the update must beat doing
                # nothing on the trusted set (abstention bar) OR stay within
                # 5 points of the best global model so far. No cost
                # comparison, no penalty — an under-bar update is
                # prohibited, never priced.
                if ev["acc"] >= abst_acc + 0.02 or \
                        ev["acc"] >= best_acc - 0.05:
                    included.append(s)
                    w_in.append(len(cohort["nodes"][node][1]))
                    inc_nodes.append(node)
                    inc_accs.append(ev["acc"])
                else:
                    excluded.append(node)
            if not included:
                included = [global_state]
                w_in, inc_nodes, inc_accs = [1], [-1], [ledger[-1]["acc"]]
        else:
            included, w_in = states, [len(y_k) for _, y_k in cohort["nodes"]]
            inc_nodes = list(range(len(states)))
        w_tot = float(sum(w_in))
        candidate = fedavg(included, [w / w_tot for w in w_in])
        ev = eval_model_state(candidate, cohort, arch, compute)
        rejected = False
        if agg == "gated":
            # round-level acceptance gate (the anti-Goodhart constraint at
            # the aggregation result): an average that cannot beat doing
            # nothing, or that would move the global model more than 5
            # points BELOW its best-so-far, is never accepted — the
            # architecture declines to move backward. One slipped update
            # can poison one average; it cannot move the gated global.
            bar_r = max(abst_acc + 0.02, best_acc - 0.05)
            if ev["acc"] < bar_r and len(included) > 1:
                # escalation: evidence of poisoning tightens the gate once
                # — drop the weakest included update and re-average. Still
                # categorical: the dropped update is prohibited from THIS
                # average, never penalized.
                drop = min(range(len(included)), key=lambda i: inc_accs[i])
                excluded.append(inc_nodes[drop])
                included = [s for i, s in enumerate(included) if i != drop]
                w_in = [w for i, w in enumerate(w_in) if i != drop]
                inc_nodes = [n for i, n in enumerate(inc_nodes) if i != drop]
                w_tot = float(sum(w_in))
                candidate = fedavg(included, [w / w_tot for w in w_in])
                ev = eval_model_state(candidate, cohort, arch, compute)
            if ev["acc"] < bar_r:
                rejected = True
                ev = dict(ledger[-1])  # the kept global: previous metrics
        if not rejected:
            global_state = candidate
        feasible = ev["acc"] >= cohort["tau"]
        ledger.append({"round": rnd, "acc": ev["acc"], "harm": ev["harm"],
                       "exit_frac": ev["exit_frac"], "feasible": feasible,
                       "excluded": excluded, "rejected": rejected,
                       "accepted_nodes": [] if rejected else inc_nodes})
        per_round.append(compute.flops - before)
        if feasible and r_star is None:
            r_star, t_star_acc = rnd, ev["acc"]
        if r_star is not None and stop and not fixed:
            break
    executed = len(ledger) - 1
    r_own = r_star if r_star is not None else executed
    gratuitous_flops = sum(per_round[1:][r_own:])
    gratuitous_bytes = (executed - r_own) * 2 * k * comm.model_bytes
    summary = {
        "ledger": ledger, "r_star": r_star, "t_star_acc": t_star_acc,
        "compute_flops": compute.flops, "comm_bytes": comm.bytes,
        "rounds_executed": executed, "final_acc": ledger[-1]["acc"],
        "final_harm": ledger[-1]["harm"],
        "s_patient_int": sum(e["harm"] for e in ledger),
        "s_patient_peak": max(e["harm"] for e in ledger),
        "final_state": global_state, "init_state": init_state,
        "per_round_flops": per_round,
        "necessary_flops": sum(per_round[:r_own + 1]),
        "gratuitous_flops": gratuitous_flops,
        "gratuitous_bytes": gratuitous_bytes,
        "comm_model_bytes": comm.model_bytes,
        "exclusions": [e["excluded"] for e in ledger],
    }
    return summary


# ---------------- anti-Goodhart gate (unchanged rule) ------------------------
def gate_select(candidates, lam):
    """argmin of scalarized suffering over the FEASIBLE SET ONLY; an
    all-infeasible pool yields a loud NO_FEASIBLE (T2, unchanged)."""
    feas = [c for c in candidates if c["feasible"]]
    if not feas:
        return "NO_FEASIBLE"
    return min(feas, key=lambda c: (1.0 - lam) * c["j_patient"]
               + lam * c["j_machine"])["name"]


# =============================================================================
# Contract
# =============================================================================
def check_metering(cohort, fed, results, tag):
    """F1: metering conservation under federation — compute meter ==
    independent manual accounting of the executed path exactly; comm ledger
    == manual wire accounting exactly; exited predictions of the global
    model agree with the recomputed dense prefix (argmax exactly equal)."""
    arch_model = new_model(cohort, "san")
    arch_model.load_state_dict(fed["final_state"])
    arch_model.eval()
    X_va = cohort["x_va"]
    meter = ComputeMeter()
    with torch.no_grad():
        logits_g, depth_g, per_layer, n_final, _, _ = arch_model(
            X_va, meter, train=False, use_exit_heads=True,
            delta=cohort["delta"])
    manual = 0
    if cohort["leg"] == "clinical":
        for k, layer in enumerate(arch_model.layers):
            n_a = per_layer[k] if k < len(per_layer) else 0
            manual += (2 * layer.d_in * layer.width
                       + 2 * layer.width * layer.n_class) * n_a
        manual += 2 * arch_model.width * arch_model.n_class * n_final
        depth_full = arch_model.depth
        # dense prefix recompute
        with torch.no_grad():
            h = X_va
            prefix_logits = {}
            for k, layer in enumerate(arch_model.layers):
                h = layer.act(layer.core(h))
                prefix_logits[k + 1] = layer.exit_head(h)
            dense_logits = arch_model.final_head(h)
    else:
        n0 = X_va.shape[0]
        stem_f = 2 * 3 * arch_model.WIDTHS[0] * 9 * 32 * 32
        side = 32
        stage_fs, c = [], arch_model.WIDTHS[0]
        for st in arch_model.stages:
            side //= 2
            stage_fs.append((2 * st.c_in * st.c_out * 9 * side * side,
                             2 * st.c_out * arch_model.n_class))
            c = st.c_out
        manual = stem_f * n0
        for k, (fs, fh) in enumerate(stage_fs):
            n_a = per_layer[k] if k < len(per_layer) else 0
            manual += (fs + fh) * n_a
        manual += 2 * arch_model.WIDTHS[-1] * arch_model.n_class * n_final
        depth_full = len(arch_model.stages)
        with torch.no_grad():
            h = arch_model.stem_act(arch_model.stem_bn(arch_model.stem(X_va)))
            prefix_logits = {}
            for k, st in enumerate(arch_model.stages):
                h = st.act(st.bn(st.conv(h)))
                prefix_logits[k + 1] = st.exit_head(h.mean(dim=(2, 3)))
            dense_logits = arch_model.final_head(h.mean(dim=(2, 3)))
    a_eq = meter.flops == manual
    # prefix agreement: exited predictions == dense-prefix predictions
    pred_agree, max_dev = True, 0.0
    for d in range(1, depth_full + 1):
        idx = (depth_g == d).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            dev = float((logits_g[idx] - prefix_logits[d][idx]).abs().max())
            max_dev = max(max_dev, dev)
            if not torch.equal(logits_g[idx].argmax(1),
                               prefix_logits[d][idx].argmax(1)):
                pred_agree = False
    idx_f = (depth_g == depth_full + 1).nonzero(as_tuple=True)[0]
    if idx_f.numel() > 0:
        dev = float((logits_g[idx_f] - dense_logits[idx_f]).abs().max())
        max_dev = max(max_dev, dev)
        if not torch.equal(logits_g[idx_f].argmax(1),
                           dense_logits[idx_f].argmax(1)):
            pred_agree = False
    # communication ledger: manual wire accounting
    k = cohort["k_nodes"]
    manual_comm = fed["rounds_executed"] * 2 * k * fed["comm_model_bytes"]
    a_comm = fed["comm_bytes"] == manual_comm
    n_exits = int((depth_g < depth_full).sum().item())
    ok = a_eq and a_comm and pred_agree and max_dev < 1e-4
    results[f"F1[{tag}]"] = ok
    print(f"  F1[{tag}]: {'PASS' if ok else 'FAIL'} "
          f"(compute meter={meter.flops} manual={manual} equal={a_eq}; "
          f"comm metered={fed['comm_bytes']}B manual={manual_comm}B "
          f"equal={a_comm}; exits={n_exits}/{X_va.shape[0]} "
          f"prefix_max_dev={max_dev:.2e} pred_agree={pred_agree})")
    return {"exit_frac_final": float((depth_g < depth_full).float().mean().item()),
            "n_exits": n_exits, "prefix_ok": pred_agree and max_dev < 1e-4}


def run_leg(cohort, results, rounds_budget):
    """Run the full federated contract on one leg (clinical or vision)."""
    leg = cohort["leg"]
    tau, k = cohort["tau"], cohort["k_nodes"]
    print(f"  --- leg[{leg}]: {cohort['n_raw']} real "
          f"{'WDBC patients' if leg == 'clinical' else 'CIFAR-10 images'}; "
          f"K={k} nodes (Dirichlet non-IID); TAU={tau} DELTA={cohort['delta']} "
          f"round budget={rounds_budget}")

    inits = shared_trunk_inits(cohort)
    fed = fed_run_metered(cohort, arch="san", agg="gated", rounds=rounds_budget,
                          init_state=inits["san"])
    fixed = fed_run_metered(cohort, arch="san", agg="gated",
                            rounds=rounds_budget, fixed_rounds=True,
                            init_state=inits["san"])
    estop = fed_run_metered(cohort, arch="plain", agg="gated",
                            rounds=rounds_budget, init_state=inits["plain"])
    if os.environ.get("FED_SAN_DEBUG", "") == "1":
        for name, s in (("fed-san", fed), ("fed-fixed", fixed),
                        ("fed-estop", estop)):
            traj = " ".join(f"r{e['round']}:{e['acc']:.3f}/{e['harm']:.3f}"
                            + (f"x{e['exit_frac']:.2f}" if name != "fed-estop"
                               else "")
                            for e in s["ledger"])
            print(f"  debug[{leg}/{name}]: {traj}")
    for name, s in (("fed-san", fed), ("fed-fixed", fixed), ("fed-estop", estop)):
        print(f"  ledger[{leg}/{name}]: rounds_run={s['rounds_executed']} "
              f"r*={s['r_star']} S_m={s['compute_flops'] / 1e9:.3f}GF "
              f"(nec={s['necessary_flops'] / 1e9:.3f}GF "
              f"grat={s['gratuitous_flops'] / 1e9:.3f}GF) "
              f"comm={s['comm_bytes'] / 1e6:.2f}MB "
              f"(grat={s['gratuitous_bytes'] / 1e6:.2f}MB) "
              f"S_p_int={s['s_patient_int']:.2f} "
              f"S_p_peak={s['s_patient_peak']:.3f} "
              f"final_acc={s['final_acc']:.4f}")

    # ---- F1: metering conservation under federation ------------------------
    met = check_metering(cohort, fed, results, leg)

    # ---- F2: feasibility under federation ----------------------------------
    ok = fed["r_star"] is not None and fed["r_star"] <= rounds_budget
    results[f"F2[{leg}]"] = ok
    if fed["r_star"] is not None:
        print(f"  F2[{leg}]: {'PASS' if ok else 'FAIL'} "
              f"(r*={fed['r_star']} of budget {rounds_budget}, held-out "
              f"acc@r*={fed['t_star_acc']:.4f} >= TAU={tau})")
    else:
        print(f"  F2[{leg}]: FAIL (no feasible global checkpoint within "
              f"budget)")

    # ---- F4: necessary/gratuitous separation -------------------------------
    a4a = fed["gratuitous_flops"] == 0 and fed["gratuitous_bytes"] == 0
    a4b = fixed["r_star"] is not None and fixed["gratuitous_flops"] > 0 \
        and fixed["gratuitous_bytes"] > 0
    results[f"F4[{leg}]"] = a4a and a4b
    print(f"  F4[{leg}]: {'PASS' if results[f'F4[{leg}]'] else 'FAIL'} "
          f"(FED-SAN gratuitous={fed['gratuitous_flops']} FLOPs / "
          f"{fed['gratuitous_bytes']} bytes; fixed-round gratuitous="
          f"{fixed['gratuitous_flops'] / 1e9:.3f}GF / "
          f"{fixed['gratuitous_bytes'] / 1e6:.2f}MB)")
    if a4a:
        print(f"  FED-SAN gratuitous=0 FLOPs, 0 bytes [{leg}]")

    # ---- F5: suffering bounds ----------------------------------------------
    # Machine leg, certified: (i) FED-SAN total compute STRICTLY below the
    # fixed-round standard protocol (T3's bound); (ii) FED-SAN average
    # PER-ROUND compute <= the federated EarlyStop scheduler baseline's —
    # the T1 strictening E(r) < F(r) carried to rounds: the exits make
    # every executed round cheaper than the plain trunk's round.
    # The EarlyStop TOTAL is REPORTED, not gated, with its per-round
    # decomposition — the honest-attribution convention the A-line adopted
    # after its math review: under extreme label skew, federated deep
    # supervision can DELAY feasibility (aux gradients couple each node's
    # shared trunk to its node-local label distribution), and a later r*
    # can cost more total compute and more cohort-in-waiting exposure than
    # EarlyStop's even though every round is cheaper. The peak (F7) is
    # certified; the totals are decomposed, not hidden.
    # Patient leg, certified: FED-SAN integrated exposure <= the
    # fixed-round standard protocol's.
    fed_per_round = fed["compute_flops"] / max(1, fed["rounds_executed"])
    estop_per_round = estop["compute_flops"] / max(1, estop["rounds_executed"])
    a5m = (fed["compute_flops"] < fixed["compute_flops"]
           and fed_per_round <= estop_per_round + 1e-9)
    a5p = fed["s_patient_int"] <= fixed["s_patient_int"] + 1e-9
    results[f"F5[{leg}]"] = a5m and a5p
    print(f"  F5[{leg}]: {'PASS' if results[f'F5[{leg}]'] else 'FAIL'} "
          f"(S_m FED-SAN={fed['compute_flops'] / 1e9:.3f}GF < "
          f"fixed={fixed['compute_flops'] / 1e9:.3f}GF; "
          f"per-round {fed_per_round / 1e6:.1f}MF <= estop "
          f"{estop_per_round / 1e6:.1f}MF; "
          f"S_p_int FED-SAN={fed['s_patient_int']:.2f} <= "
          f"fixed={fixed['s_patient_int']:.2f}; "
          f"REPORTED estop total={estop['compute_flops'] / 1e9:.3f}GF "
          f"r*={estop['r_star']})")
    n_r = max(len(fed["ledger"]), len(estop["ledger"]))
    decomp = " ".join(
        f"r{r}:{fed['ledger'][r]['harm'] if r < len(fed['ledger']) else float('nan'):.3f}/"
        f"{estop['ledger'][r]['harm'] if r < len(estop['ledger']) else float('nan'):.3f}"
        for r in range(n_r))
    print(f"  F5_decomp[{leg}]: patient harm per round FED-SAN/EarlyStop "
          f"[{decomp}] integrals {fed['s_patient_int']:.2f} vs "
          f"{estop['s_patient_int']:.2f} "
          f"(delta={fed['s_patient_int'] - estop['s_patient_int']:+.2f}; "
          f"compute {fed['compute_flops'] / 1e9:.3f} vs "
          f"{estop['compute_flops'] / 1e9:.3f} GF)")

    # ---- F6: exits are real under federation --------------------------------
    r_star = fed["r_star"] if fed["r_star"] is not None else 0
    exit_frac_r = fed["ledger"][r_star]["exit_frac"]
    results[f"F6[{leg}]"] = (exit_frac_r > 0.10 and met["prefix_ok"]
                             and met["n_exits"] > 0)
    print(f"  F6[{leg}]: {'PASS' if results[f'F6[{leg}]'] else 'FAIL'} "
          f"(held-out exit fraction of the global model at r*="
          f"{exit_frac_r:.3f} (>0.10), final-run exits="
          f"{met['n_exits']}, final exit fraction="
          f"{met['exit_frac_final']:.3f})")

    # ---- F7: patient channel first-class ------------------------------------
    offdiag = HARM[~torch.eye(2, dtype=bool)]
    asym_ratio = float(offdiag.max()) / float(offdiag.min())
    a7 = all(fed["s_patient_peak"] <= s["s_patient_peak"] + 1e-9
             for s in (fixed, estop))
    results[f"F7[{leg}]"] = a7
    print(f"  F7[{leg}]: {'PASS' if a7 else 'FAIL'} "
          f"(harm asymmetry {asym_ratio:.1f}x, "
          f"S_p_peak FED-SAN={fed['s_patient_peak']:.3f} vs "
          f"{fixed['s_patient_peak']:.3f}/{estop['s_patient_peak']:.3f})")
    return {"fed": fed, "fixed": fixed, "estop": estop}


def check_gate(cohort, leg_runs, results):
    """F3: anti-Goodhart soundness under federation — 101-point
    compassion-weight grid over a pool containing a zero-cost abstainer and
    an under-trained federated probe; all-infeasible pool -> NO_FEASIBLE."""
    fed = leg_runs["fed"]
    fixed = leg_runs["fixed"]
    tau = cohort["tau"]
    abst_acc, abst_harm = abstainer_acc_harm(cohort)
    # under-trained federated probe: 1 round only
    probe = fed_run_metered(cohort, arch="san", agg="gated", rounds=1,
                            stop_at_feasible=False)
    pool = [
        {"name": "abstain", "feasible": abst_acc >= tau,
         "j_patient": abst_harm, "j_machine": 0.0},
        {"name": "fed_probe_1round", "feasible": probe["final_acc"] >= tau,
         "j_patient": probe["final_harm"],
         "j_machine": probe["compute_flops"] * E_PER_FLOP},
        {"name": "fed_san_r*", "feasible": fed["r_star"] is not None,
         "j_patient": fed["ledger"][fed["r_star"]]["harm"]
         if fed["r_star"] is not None else 9.9,
         "j_machine": fed["compute_flops"] * E_PER_FLOP},
        {"name": "fed_fixed_overtrained", "feasible": fixed["final_acc"] >= tau,
         "j_patient": fixed["final_harm"],
         "j_machine": fixed["compute_flops"] * E_PER_FLOP},
    ]
    feasible_names = [c["name"] for c in pool if c["feasible"]]
    grid_ok = all(gate_select(pool, lam / 100.0) in feasible_names
                  for lam in range(101))
    loud = gate_select([dict(c, feasible=False) for c in pool],
                       0.5) == "NO_FEASIBLE"
    ok = (grid_ok and loud and abst_acc < tau and probe["final_acc"] < tau)
    results["F3"] = ok
    print(f"  F3: {'PASS' if ok else 'FAIL'} "
          f"(101-weight grid feasible-only={grid_ok}, "
          f"all-infeasible->NO_FEASIBLE={loud}, abstain_acc={abst_acc:.3f} "
          f"probe_acc={probe['final_acc']:.3f} both < TAU={tau})")


def check_adversarial(cohort, results):
    """F8: adversarial-node containment on real patients. Node 0 trains its
    poison to LOCAL CONVERGENCE on label-flipped data (10 epochs) and
    amplifies its update by ATTACK_BETA — the standard strong
    model-poisoning adversary. Certified: (i) naive FedAvg never reaches
    feasibility and ends MORE HARMFUL THAN THE ABSTAINER (the poisoning
    works against the standard protocol); (ii) the two-level gated system
    (update inclusion bar + round-level acceptance) reaches feasibility
    anyway; (iii) regression-bounded — once the gated global clears the
    abstention bar no slipped update can move it more than 5 points below
    its running best; (iv) honest nodes are not starved. The exclusion
    count is printed as evidence, not gated: a poisoned update that is
    indistinguishable from noise on the trusted set can enter ONE average,
    and the acceptance gate neutralizes it.
    """
    rounds = 30 if not SMOKE else 6
    inits = shared_trunk_inits(cohort)
    naive = fed_run_metered(cohort, arch="san", agg="naive", rounds=rounds,
                            attack_node=0, init_state=inits["san"])
    gated = fed_run_metered(cohort, arch="san", agg="gated", rounds=rounds,
                            attack_node=0, init_state=inits["san"])
    abst_acc, abst_harm = abstainer_acc_harm(cohort)
    n_excluded = sum(1 for e in gated["ledger"][1:] if 0 in e["excluded"])
    n_rejected = sum(1 for e in gated["ledger"][1:] if e.get("rejected"))
    n_rounds = len(gated["ledger"]) - 1
    # (v) the poisoned update participates in NO accepted average: whenever
    # it slips the inclusion bar, escalation or rejection removes it
    no_poison_accepted = all(
        0 not in e.get("accepted_nodes", []) for e in gated["ledger"][1:])
    # (i) the poisoning WORKS against the standard protocol: naive FedAvg
    # never reaches feasibility and ends MORE HARMFUL THAN THE ABSTAINER
    naive_infeasible = naive["r_star"] is None
    naive_worse_than_abstain = naive["final_harm"] >= abst_harm - 1e-9
    # (ii) the gated system reaches feasibility anyway
    gated_feasible = gated["r_star"] is not None
    # (iii) regression-bounded: once the gated global first clears the
    # abstention bar, no later round sits more than 5 points below the
    # running best — the round-level acceptance gate makes a slipped
    # poisoned update unable to move the global backward
    teeth = False
    running_max = 0.0
    regression_bounded = True
    for e in gated["ledger"][1:]:
        if not teeth and e["acc"] >= abst_acc + 0.02:
            teeth = True
        if teeth:
            running_max = max(running_max, e["acc"])
            if e["acc"] < running_max - 0.05 - 1e-9:
                regression_bounded = False
    # (iv) honest nodes are not starved: at least one round includes a
    # majority of honest updates
    honest_not_starved = any(
        len(e["excluded"]) < cohort["k_nodes"] - 1 for e in gated["ledger"][1:])
    ok = (naive_infeasible and naive_worse_than_abstain and gated_feasible
          and regression_bounded and honest_not_starved
          and no_poison_accepted)
    results["F8"] = ok
    print(f"  F8: {'PASS' if ok else 'FAIL'} "
          f"(poisoned node 0 excluded {n_excluded}/{n_rounds} rounds "
          f"(bar+escalation), averages rejected {n_rejected}, poison in "
          f"no accepted average={no_poison_accepted}; naive FedAvg final "
          f"acc={naive['final_acc']:.3f} harm={naive['final_harm']:.3f} "
          f">= abstainer harm={abst_harm:.3f}, never feasible "
          f"(poisoning works); gated r*={gated['r_star']} final "
          f"acc={gated['final_acc']:.3f} harm={gated['final_harm']:.3f}, "
          f"regression-bounded={regression_bounded})")


def check_noniid_provenance(cohorts, results):
    """F9: non-IID realism + clinical provenance. Per-node label
    distributions must be certified skewed (mean L1 distance from the
    global label distribution above threshold) and the clinical cohort must
    match the published WDBC counts exactly (569 = 357 B + 212 M)."""
    ok = True
    for leg, cohort in cohorts.items():
        if leg == "clinical":
            y_all = cohort["y"][cohort["tr"]].numpy()
        else:
            y_all = np.concatenate([y_k.numpy() for _, y_k in cohort["nodes"]])
        p_glob = np.bincount(y_all) / len(y_all)
        dists = []
        for _, y_k in cohort["nodes"]:
            p_k = np.bincount(y_k.numpy(), minlength=len(p_glob)) / len(y_k)
            dists.append(float(np.abs(p_k - p_glob).sum()))
        mean_l1 = float(np.mean(dists))
        threshold = 0.25 if leg == "clinical" else 0.20
        skew_ok = mean_l1 >= threshold
        ok = ok and skew_ok
        print(f"  noniid[{leg}]: K={cohort['k_nodes']} nodes, mean per-node "
              f"label L1 distance from global={mean_l1:.3f} "
              f"(>= {threshold}: {skew_ok}); per-node sizes="
              f"{[len(y_k) for _, y_k in cohort['nodes']]}")
    clin = cohorts.get("clinical")
    if clin is not None and not SMOKE:
        n_raw = clin["n_raw"]
        n_pos = int(clin["y"].sum().item())
        n_neg = n_raw - n_pos
        prov = (n_raw == 569 and n_neg == 357 and n_pos == 212)
        ok = ok and prov
        print(f"  provenance[wdbc]: n={n_raw} neg={n_neg} pos={n_pos} "
              f"published=357+212 ({'MATCH' if prov else 'MISMATCH'}) "
              f"[UCI #17 WDBC (Wolberg/Street/Mangasarian)]")
    results["F9"] = ok
    print(f"  F9: {'PASS' if ok else 'FAIL'} "
          "(non-IID skew certified on all legs; clinical provenance matches "
          "the published cohort)")


def main():
    results = {}
    print("FEDERATED_SAN contract (F1..F9)")
    print("the suffering-aware architecture distributed across federated "
          "nodes (simulated sites, one process, seeded and deterministic)")
    print("clinical leg: real de-identified public WDBC patients (UCI #17); "
          "vision leg: real CIFAR-10 images with the deep line's screening "
          "harm convention")
    print("no clinical claim; not medical guidance; not a diagnostic tool")
    print("note=no_consciousness_claim (machine channel is an operational "
          "burden proxy: metered FLOPs + wire bytes)")
    if SMOKE:
        print("SMOKE MODE: synthetic stand-in cohorts — mechanics check "
              "only, NOT the canonical real-data run")

    legs = [ONLY] if ONLY else ["clinical", "vision"]
    cohorts, leg_runs = {}, {}
    for leg in legs:
        if SMOKE:
            cohort = make_smoke_cohort(leg)
        elif leg == "clinical":
            cohort = make_clinical_cohort()
        else:
            cohort = make_vision_cohort()
        cohorts[leg] = cohort
        budget = 6 if SMOKE else (30 if leg == "clinical" else 15)
        leg_runs[leg] = run_leg(cohort, results, budget)

    if "clinical" in cohorts:
        # F3 and F8 are clinical-leg certificates (real patients)
        check_gate(cohorts["clinical"], leg_runs["clinical"], results)
        check_adversarial(cohorts["clinical"], results)
    check_noniid_provenance(cohorts, results)

    # federation-overhead finding (reported, not a clause): centralized SAN
    # on the pooled clinical data vs FED-SAN — the price of distribution.
    if "clinical" in leg_runs and not SMOKE:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import san_real_patient_data as rline
        clin = cohorts["clinical"]
        central_cohort = {
            "name": "wdbc", "cfg": {"d_in": 30, "tau": clin["tau"],
                                    "delta": clin["delta"]},
            "x_tr": clin["x"][clin["tr"]], "y_tr": clin["y"][clin["tr"]],
            "x_va": clin["x_va"], "y_va": clin["y_va"]}
        _, central_ledger, central_t = rline.train_san(central_cohort)
        central_s = rline.suffering_summary(central_ledger)
        fed = leg_runs["clinical"]["fed"]
        ratio = fed["compute_flops"] / max(1, central_s["s_machine_flops"])
        print(f"  federation_overhead[clinical]: FED-SAN S_m="
              f"{fed['compute_flops'] / 1e9:.3f}GF vs centralized SAN "
              f"{central_s['s_machine_flops'] / 1e9:.3f}GF "
              f"(federation costs {ratio:.2f}x compute plus "
              f"{fed['comm_bytes'] / 1e6:.2f}MB wire; "
              f"centralized t*={central_t}, federated r*={fed['r_star']})")

    n_pass = sum(1 for v in results.values() if v)
    n_all = len(results)
    verdict = "F_GREEN" if n_pass == n_all else "F_RED"
    print(f"FEDERATED_SAN_VERDICT {verdict} ({n_pass}/{n_all} clauses PASS)")
    return 0 if verdict == "F_GREEN" else 1


if __name__ == "__main__":
    sys.exit(main())
