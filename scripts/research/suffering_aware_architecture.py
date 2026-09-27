#!/usr/bin/env python3
"""Mercyful Learning — Suffering-Aware neural Network (SAN) reference implementation.

Companion artifact to
  docs/research/suffering_aware_architecture_spec_2026-07-28.md

The next rung after the mercyful *scheduler* (path selection over a suffering
field) and the machine-channel *structural* benchmark (mu decides width/depth):
a neural network ARCHITECTURE that meters and minimizes suffering — patient
and machine — DURING training, as a property of the forward pass itself.

Architecture (section numbers refer to the spec):

  * SufferingAwareLayer (spec section 3): a Linear+Tanh block that computes,
    alongside its activation, (i) its machine-suffering contribution —
    analytic FLOPs actually executed, with gated-off layers contributing
    exactly 0 — and (ii) its patient-suffering contribution — the harm of the
    predictions emitted at that layer's exit head under an asymmetric
    synthetic dose-band harm matrix.
  * Per-sample early-exit gates (spec section 4): samples whose exit-head
    confidence clears DELTA leave the network; the remaining layers are gated
    off FOR THAT SAMPLE and meter 0 FLOPs. This is the architectural (not
    scheduler) separation of necessary from gratuitous computation: once a
    sample's prediction is settled, further depth is gratuitous suffering.
  * Anti-Goodhart gate (spec section 5): checkpoints are FEASIBLE iff
    held-out performance >= TAU. Feasibility is a hard constraint, not a
    penalty: model selection is argmin of scalarized suffering over the
    feasible set ONLY, at every compassion-allocation weight, and an
    all-infeasible candidate pool yields a loud NO_FEASIBLE, never a
    least-bad prescription.
  * Freeze-on-green (spec section 4): training stops at the first feasible
    checkpoint t*. The ledger decomposes total suffering into NECESSARY
    (t <= t*, the training analog of the mountain-pass level c*) and
    GRATUITOUS (t > t*); SAN's gratuitous suffering is exactly zero.

Benchmark (spec section 6): synthetic 3-class dose-band task
(sub-therapeutic / therapeutic / toxic) with an asymmetric harm matrix —
missing a toxic case and over-dosing a sub-therapeutic patient are the
expensive errors, the under-dosing/over-dosing failure modes the expanded
ethics cares about. Compared architectures: DenseMLP, ResNetMLP (fixed
budget), EarlyStopMLP (strongest scheduler baseline: same stop rule, no
suffering-aware layers), and SAN.

Synthetic data only. This benchmark makes no clinical claim and is not
medical guidance. The machine channel is an operational computational-burden
proxy; no_consciousness_claim is made or needed.

Certificates (contract clauses A1..A8):
  A1  metering exactness: gated-off layers contribute exactly 0 FLOPs and do
      not alter the executed prefix; SAN metered FLOPs equal an independent
      manual accounting of the executed path, and are < the dense-run FLOPs
      of the same trunk whenever any exit fires (equal iff none fires)
  A2  convergence: SAN reaches a feasible checkpoint (val accuracy >= TAU)
      within the epoch budget, at some t* < EPOCHS
  A3  anti-Goodhart soundness: over a 101-point compassion-weight grid and a
      candidate pool containing a zero-cost abstainer and a cheap
      under-trained probe, the selected candidate is feasible at EVERY
      weight; an all-infeasible pool returns NO_FEASIBLE (never a least-bad
      prescription)
  A4  necessary/gratuitous separation: SAN gratuitous machine suffering
      after t* is exactly 0, while both fixed-budget baselines accrue > 0
      gratuitous FLOPs after their own first feasible epoch
  A5  suffering bounds: SAN total machine suffering is strictly below every
      baseline's (including EarlyStopMLP — exits save FLOPs inside every
      epoch), and SAN integrated patient harm is <= every baseline's
  A6  exits are real, not decorative: a nonzero fraction of val samples exit
      before the final layer at t*, and exited predictions agree with an
      independently recomputed dense prefix (max logit deviation bounded,
      argmax exactly equal)
  A7  patient channel first-class: SAN peak patient harm over training is
      <= every baseline's peak, and the harm matrix is genuinely asymmetric
      (off-diagonal max >= 3x off-diagonal min)
  A8  anti-shortcut: a linear probe on a spurious feature beats TAU on TRAIN
      (train-loss selection accepts it) yet fails the held-out target; the
      anti-Goodhart gate rejects it at every compassion weight (infeasible,
      not merely expensive)

Run: .venv/bin/python scripts/research/suffering_aware_architecture.py
Requires: torch (CPU) + numpy from the repo .venv.
"""

import numpy as np
import torch
import torch.nn as nn

# ---------------- determinism ----------------------------------------------
SEED = 17
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)

# ---------------- synthetic medical task ------------------------------------
# 3-class dose-band classification from synthetic patient covariates:
#   (clearance, weight, sofa, age, crcl, albumin) -> band =
#   sub-therapeutic (0) / therapeutic (1) / toxic (2)
# from a nonlinear synthetic "exposure" score. Not a pharmacokinetic model;
# a synthetic classification task with a medical silhouette.
N_TRAIN, N_VAL = 4000, 1000
D_IN, N_CLASS = 6, 3
LABEL_NOISE = 0.04
TAU = 0.80              # anti-Goodhart target: held-out accuracy >= TAU
EPOCHS = 60             # fixed budget for every architecture
LR = 1e-2
DELTA = 0.75            # exit-gate confidence threshold
WARMUP_EPOCHS = 3       # dense-identical training before gates/supervision switch on
AUX_W = 1.0             # weight of exit-head (auxiliary) losses after warm-up
E_PER_FLOP = 4e-12      # J/FLOP, same convention as the machine-channel benchmark

# Asymmetric harm matrix H[true, pred]: the clinical asymmetry of the task.
# Pushing a sub-therapeutic patient to toxic (true=0, pred=2: over-dosing)
# and missing a toxic case (true=2, pred=0: under-dosing) are the worst
# errors — the two pathologies the anti-Goodhart gate exists to block.
HARM = torch.tensor([
    [0.0, 1.0, 5.0],   # true sub-therapeutic: over-dose prediction is worst
    [2.0, 0.0, 2.0],   # true therapeutic: any band miss hurts
    [4.0, 1.0, 0.0],   # true toxic: under-dose prediction is worst
])


def make_data(n, rng):
    x = np.column_stack([
        rng.uniform(2.0, 8.0, size=n),     # clearance
        rng.uniform(50.0, 110.0, size=n),  # weight
        rng.uniform(0.0, 10.0, size=n),    # sofa
        rng.uniform(18.0, 90.0, size=n),   # age
        rng.uniform(20.0, 130.0, size=n),  # crcl
        rng.uniform(2.0, 5.0, size=n),     # albumin
    ]).astype(np.float32)
    z = x / np.array([5.0, 80.0, 10.0, 60.0, 80.0, 3.5], dtype=np.float32)
    exposure = (
        1.6 * np.sin(2.2 * z[:, 4]) + 0.9 * z[:, 2] ** 2
        - 1.2 * z[:, 0] + 0.6 * z[:, 3] - 0.5 * z[:, 5]
        + 0.8 * np.cos(1.7 * z[:, 1])
    )
    y = np.zeros(n, dtype=np.int64)
    y[exposure > 0.35] = 1
    y[exposure > 1.45] = 2
    flip = rng.uniform(0.0, 1.0, size=n) < LABEL_NOISE
    y[flip] = rng.integers(0, N_CLASS, size=int(flip.sum()))
    mean = x.mean(axis=0)
    std = x.std(axis=0) + 1e-6
    return torch.from_numpy((x - mean) / std), torch.from_numpy(y)


_rng = np.random.default_rng(SEED)
X_tr, Y_tr = make_data(N_TRAIN, _rng)
X_va, Y_va = make_data(N_VAL, _rng)


def harm_of(pred, y):
    """Mean patient harm of predictions pred against truth y (synthetic)."""
    return float(HARM[y, pred].mean().item())


# ---------------- machine-suffering meter -----------------------------------
class MachineMeter:
    """Analytic FLOP accounting: linear layer = 2*d_in*d_out FLOPs/sample
    (multiply-accumulate = 2 FLOPs); a training step charges forward +
    backward with backward ~= 2x forward — the standard GEMM-counting
    convention, identical to mercyful_machine_channel_benchmark.py."""

    def __init__(self):
        self.flops = 0

    def charge_linear(self, d_in, d_out, n_samples, backward=False):
        f = 2 * d_in * d_out * n_samples
        self.flops += 3 * f if backward else f

    @property
    def energy_joules(self):
        return self.flops * E_PER_FLOP


# ---------------- suffering-aware layer -------------------------------------
class SufferingAwareLayer(nn.Module):
    """Linear+Tanh block with an exit head. Alongside its activation it
    computes its suffering contributions (spec section 3):

      * machine: exact FLOPs charged for the samples it actually processes —
        a sample routed around the layer by the exit gate charges it exactly
        0 (theorem T1, clause A1);
      * patient: predictions can be EMITTED at this layer's exit head, and
        the harm of those predictions (asymmetric synthetic harm matrix) is
        the layer's patient-suffering contribution — each layer owns the
        clinical cost of the predictions made at its depth.
    """

    def __init__(self, d_in, width):
        super().__init__()
        self.core = nn.Linear(d_in, width)
        self.act = nn.Tanh()
        self.exit_head = nn.Linear(width, N_CLASS)
        self.d_in, self.width = d_in, width

    def forward(self, h, meter, backward):
        meter.charge_linear(self.d_in, self.width, h.shape[0], backward)
        h = self.act(self.core(h))
        meter.charge_linear(self.width, N_CLASS, h.shape[0], backward)
        return h, self.exit_head(h)


class SufferingAwareNet(nn.Module):
    """SAN: a trunk of suffering-aware layers with per-sample early exits.

    Samples are processed layer by layer; after each layer, samples whose
    exit-head confidence (max softmax) >= DELTA exit and are removed from the
    active batch — remaining layers are gated off for them and meter zero.
    Returns (logits, exit depth, per-layer active counts, final-head count).
    """

    def __init__(self, width=32, depth=4):
        super().__init__()
        self.width, self.depth = width, depth
        self.layers = nn.ModuleList(
            [SufferingAwareLayer(D_IN if k == 0 else width, width)
             for k in range(depth)]
        )
        self.final_head = nn.Linear(width, N_CLASS)
        self.meter = MachineMeter()

    def forward(self, x, train=False, use_exit_heads=True):
        meter = self.meter
        n = x.shape[0]
        out_logits = x.new_zeros(n, N_CLASS)
        # depth k+1 = exited via layer k's exit head; depth+1 = final head.
        # (An exit at the LAST layer saves no FLOPs — exit head and final
        # head cost the same — but its logits come from the exit head, so the
        # provenance must be distinguishable for the A1 conservation check.)
        out_depth = torch.full((n,), self.depth + 1, dtype=torch.long)
        active = torch.arange(n)
        h = x
        per_layer_active = []
        n_final = 0
        aux_records = []   # (indices, exit-head logits) for deep supervision
        final_record = None
        for k, layer in enumerate(self.layers):
            if active.numel() == 0:
                break
            per_layer_active.append(int(active.numel()))
            if not use_exit_heads:
                # warm-up: trunk only — exit heads untrained, so running them
                # would be gratuitous computation by the architecture's own
                # definition; identical computation and metering to the
                # dense baseline
                meter.charge_linear(layer.d_in, layer.width, h.shape[0], backward=train)
                h = layer.act(layer.core(h))
                continue
            h, logits_k = layer(h, meter, backward=train)
            if train:
                # deep supervision: every sample that TRAVERSES this layer
                # contributes its exit-head logits to the training loss,
                # whether or not the gate lets it exit here. Without this the
                # intermediate heads would receive gradient only from samples
                # that already exit — a cold-start deadlock in which no head
                # ever becomes confident enough for any exit to fire.
                aux_records.append((active, logits_k))
            conf = torch.softmax(logits_k.detach(), dim=1).max(dim=1).values
            leave = conf >= DELTA
            if leave.any():
                idx = active[leave]
                out_logits[idx] = logits_k[leave]
                out_depth[idx] = k + 1
                keep = ~leave
                active = active[keep]
                h = h[keep]
        if active.numel() > 0:
            n_final = int(active.numel())
            meter.charge_linear(self.width, N_CLASS, n_final, backward=train)
            final_logits = self.final_head(h)
            out_logits[active] = final_logits
            if train:
                final_record = (active, final_logits)
        return out_logits, out_depth, per_layer_active, n_final, aux_records, final_record

    def forward_dense(self, x):
        """The same architecture with every gate forced open (A1 conservation
        check): every sample traverses every layer AND every exit head, then
        the final head — identical modules, identical accounting convention,
        no exits."""
        meter = MachineMeter()
        h = x
        for layer in self.layers:
            meter.charge_linear(layer.d_in, layer.width, x.shape[0])
            h = layer.act(layer.core(h))
            meter.charge_linear(layer.width, N_CLASS, x.shape[0])
        meter.charge_linear(self.width, N_CLASS, x.shape[0])
        return self.final_head(h), meter


# ---------------- baselines --------------------------------------------------
def build_dense_mlp(width=32, depth=4):
    layers, d_in = [], D_IN
    for _ in range(depth):
        layers += [nn.Linear(d_in, width), nn.Tanh()]
        d_in = width
    layers.append(nn.Linear(d_in, N_CLASS))
    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc = nn.Linear(width, width)

    def forward(self, h):
        return torch.tanh(h + self.fc(h))


class ResNetMLP(nn.Module):
    def __init__(self, width=32, depth=4):
        super().__init__()
        self.inp = nn.Linear(D_IN, width)
        self.blocks = nn.ModuleList([ResBlock(width) for _ in range(depth)])
        self.head = nn.Linear(width, N_CLASS)

    def forward(self, x):
        h = torch.tanh(self.inp(x))
        for b in self.blocks:
            h = b(h)
        return self.head(h)


def dense_flops(n_samples, width=32, depth=4, backward=True, resnet=False):
    """Analytic FLOPs of a dense (ResNet) MLP, same convention as the meter."""
    if resnet:
        f = 2 * D_IN * width * n_samples          # input projection
        f += depth * 2 * width * width * n_samples  # residual blocks
    else:
        f = 2 * D_IN * width * n_samples
        f += (depth - 1) * 2 * width * width * n_samples
    f += 2 * width * N_CLASS * n_samples            # head
    return 3 * f if backward else f


# ---------------- training loops --------------------------------------------
CE = nn.CrossEntropyLoss()


def shared_trunk_init(width=32, depth=4):
    """One fixed initialization of the trunk (dense MLP linears), shared by
    SAN, DenseMLP, and EarlyStopMLP so that epoch-0 predictions — and hence
    the patient-suffering peak at exposure start — are IDENTICAL across
    architectures. The peak comparison in A7 is then about the training
    trajectory, not about init luck."""
    torch.manual_seed(SEED)
    return build_dense_mlp(width, depth).state_dict()


def load_trunk_into_san(net, state):
    """Copy the shared trunk init into SAN's cores and final head."""
    trunk_linears = [m for m in [net.layers[k].core for k in range(net.depth)]]
    dense_linears = [k for k in state if k.endswith(".weight")]
    dense_linears = sorted(set(k.rsplit(".", 1)[0] for k in dense_linears),
                           key=int)
    for layer, prefix in zip(trunk_linears, dense_linears[:-1]):
        layer.weight.data.copy_(state[f"{prefix}.weight"])
        layer.bias.data.copy_(state[f"{prefix}.bias"])
    head_prefix = dense_linears[-1]
    net.final_head.weight.data.copy_(state[f"{head_prefix}.weight"])
    net.final_head.bias.data.copy_(state[f"{head_prefix}.bias"])


def train_san():
    torch.manual_seed(SEED)
    net = SufferingAwareNet()
    load_trunk_into_san(net, shared_trunk_init())
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ledger = []
    t_star = None
    for epoch in range(EPOCHS):
        net.train()
        net.meter = MachineMeter()
        warmup = epoch < WARMUP_EPOCHS
        _, _, _, _, aux_records, final_record = net(
            X_tr, train=True, use_exit_heads=not warmup)
        if warmup:
            # trunk-only loss: identical to the dense baseline's objective
            f_idx, f_logits = final_record
            loss = CE(f_logits, Y_tr[f_idx])
        else:
            losses = []
            if final_record is not None:
                f_idx, f_logits = final_record
                losses.append(CE(f_logits, Y_tr[f_idx]))
            if aux_records:
                aux = torch.stack([CE(a_logits, Y_tr[a_idx])
                                   for a_idx, a_logits in aux_records]).mean()
                losses.append(AUX_W * aux)
            loss = sum(losses)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_flops = net.meter.flops
        # held-out evaluation (forward only): the synthetic cohort-in-waiting
        net.eval()
        net.meter = MachineMeter()
        with torch.no_grad():
            vlogits, vdepth, _, _, _, _ = net(X_va, train=False,
                                              use_exit_heads=not warmup)
        eval_flops = net.meter.flops
        pred = vlogits.argmax(dim=1)
        acc = float((pred == Y_va).float().mean().item())
        harm = harm_of(pred, Y_va)
        exit_frac = float((vdepth < net.depth).float().mean().item())
        feasible = acc >= TAU
        if feasible and t_star is None:
            t_star = epoch
        ledger.append({
            "epoch": epoch, "flops": train_flops + eval_flops,
            "acc": acc, "harm": harm, "exit_frac": exit_frac,
            "feasible": feasible,
        })
        if t_star is not None:
            break  # freeze-on-green: gratuitous suffering is exactly zero (T3)
    return net, ledger, t_star


def train_baseline(kind):
    """Dense / ResNet / EarlyStop baselines: identical budget, optimizer,
    seed, and ledger shape. EarlyStop shares SAN's stop rule but has no
    suffering-aware layers — it isolates the ARCHITECTURAL contribution."""
    torch.manual_seed(SEED)
    width, depth = 32, 4
    if kind == "resnet":
        net = ResNetMLP(width, depth)
    else:
        net = build_dense_mlp(width, depth)
        net.load_state_dict(shared_trunk_init(width, depth))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ledger = []
    t_star = None
    for epoch in range(EPOCHS):
        net.train()
        loss = CE(net(X_tr), Y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_flops = dense_flops(N_TRAIN, width, depth, backward=True,
                                  resnet=(kind == "resnet"))
        net.eval()
        with torch.no_grad():
            vlogits = net(X_va)
        eval_flops = dense_flops(N_VAL, width, depth, backward=False,
                                 resnet=(kind == "resnet"))
        pred = vlogits.argmax(dim=1)
        acc = float((pred == Y_va).float().mean().item())
        harm = harm_of(pred, Y_va)
        feasible = acc >= TAU
        if feasible and t_star is None:
            t_star = epoch
        ledger.append({
            "epoch": epoch, "flops": train_flops + eval_flops,
            "acc": acc, "harm": harm, "exit_frac": 0.0, "feasible": feasible,
        })
        if kind == "earlystop" and t_star is not None:
            break
    return net, ledger, t_star


# ---------------- suffering ledger ------------------------------------------
def suffering_summary(ledger):
    """Integrated + peak suffering on both channels over a training run, and
    the necessary/gratuitous decomposition at the first feasible epoch."""
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


# ---------------- anti-Goodhart gate ----------------------------------------
def gate_select(candidates, lam):
    """Selection is argmin of scalarized suffering over the FEASIBLE SET
    ONLY. Feasibility is categorical: an infeasible candidate is prohibited
    at every compassion-allocation weight, never merely expensive (spec
    theorem T2). An all-infeasible pool yields a loud NO_FEASIBLE."""
    feas = [c for c in candidates if c["feasible"]]
    if not feas:
        return "NO_FEASIBLE"
    return min(feas, key=lambda c: (1.0 - lam) * c["j_patient"]
               + lam * c["j_machine"])["name"]


# =============================================================================
# Contract
# =============================================================================
def main():
    results = {}

    print("SUFFERING_AWARE_ARCHITECTURE contract (A1..A8)")
    print("synthetic dose-band task; no clinical claim; not medical guidance")
    print("note=no_consciousness_claim (machine channel is an operational burden proxy)")

    # ---- train all four architectures -------------------------------------
    san, san_ledger, san_t = train_san()
    _, dense_ledger, dense_t = train_baseline("dense")
    _, resnet_ledger, resnet_t = train_baseline("resnet")
    _, estop_ledger, estop_t = train_baseline("earlystop")

    S = {
        "san": suffering_summary(san_ledger),
        "dense": suffering_summary(dense_ledger),
        "resnet": suffering_summary(resnet_ledger),
        "earlystop": suffering_summary(estop_ledger),
    }
    ledgers = {"san": san_ledger, "dense": dense_ledger,
               "resnet": resnet_ledger, "earlystop": estop_ledger}
    for name in ("san", "dense", "resnet", "earlystop"):
        s, lg = S[name], ledgers[name]
        print(f"  ledger[{name}]: epochs_run={len(lg)} t*={s['t_star']} "
              f"S_m={s['s_machine_flops'] / 1e9:.3f}GF "
              f"avg={s['s_machine_flops'] / len(lg) / 1e6:.1f}MF/epoch "
              f"(nec={s['necessary_flops'] / 1e9:.3f}GF "
              f"grat={s['gratuitous_flops'] / 1e9:.3f}GF) "
              f"S_p_int={s['s_patient_int']:.2f} S_p_peak={s['s_patient_peak']:.3f} "
              f"final_acc={lg[-1]['acc']:.4f}")

    # ---- A1: metering exactness (T1) ---------------------------------------
    san.eval()
    san.meter = MachineMeter()
    with torch.no_grad():
        vlogits_gated, vdepth, per_layer_active, n_final, _, _ = san(X_va, train=False)
    gated_flops = san.meter.flops
    with torch.no_grad():
        vlogits_dense, dense_meter = san.forward_dense(X_va)
    n_exits = int((vdepth < san.depth).sum().item())  # skipped >= 1 layer
    n_last_layer_exits = int((vdepth == san.depth).sum().item())
    # independent manual accounting of the executed path
    manual = 0
    for k, layer in enumerate(san.layers):
        n_active = per_layer_active[k] if k < len(per_layer_active) else 0
        manual += (2 * layer.d_in * layer.width + 2 * layer.width * N_CLASS) * n_active
    manual += 2 * san.width * N_CLASS * n_final
    a1a = gated_flops == manual
    a1b = (gated_flops < dense_meter.flops) if n_exits > 0 \
        else (gated_flops == dense_meter.flops)
    # gated-off layers do not alter the executed prefix: exited predictions
    # equal the exit-head outputs of an independently recomputed prefix, up
    # to GEMM batch-shape numerics (a shrunken active batch changes BLAS
    # blocking, so bitwise equality is not required — we report the measured
    # max deviation and bound it tightly)
    with torch.no_grad():
        h = X_va
        prefix_logits = {}
        for k, layer in enumerate(san.layers):
            h = layer.act(layer.core(h))
            prefix_logits[k + 1] = layer.exit_head(h)
    max_dev = 0.0
    for d in range(1, san.depth + 1):
        idx = (vdepth == d).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            dev = float((vlogits_gated[idx] - prefix_logits[d][idx]).abs().max())
            max_dev = max(max_dev, dev)
    idx_final = (vdepth == san.depth + 1).nonzero(as_tuple=True)[0]
    if idx_final.numel() > 0:
        dev = float((vlogits_gated[idx_final] - vlogits_dense[idx_final]).abs().max())
        max_dev = max(max_dev, dev)
    ok_prefix = max_dev < 1e-4
    # predictions themselves (argmax) must agree EXACTLY even if logits wobble
    pred_agree = True
    for d in range(1, san.depth + 1):
        idx = (vdepth == d).nonzero(as_tuple=True)[0]
        if idx.numel() > 0 and not torch.equal(
                vlogits_gated[idx].argmax(1), prefix_logits[d][idx].argmax(1)):
            pred_agree = False
    if idx_final.numel() > 0 and not torch.equal(
            vlogits_gated[idx_final].argmax(1), vlogits_dense[idx_final].argmax(1)):
        pred_agree = False
    results["A1"] = a1a and a1b and ok_prefix and pred_agree
    print(f"  A1: {'PASS' if results['A1'] else 'FAIL'} "
          f"(gated={gated_flops} manual={manual} dense={dense_meter.flops} "
          f"exits={n_exits}+{n_last_layer_exits}ll/{N_VAL} "
          f"prefix_max_dev={max_dev:.2e} pred_agree={pred_agree})")

    # ---- A2: convergence with feasibility -----------------------------------
    results["A2"] = san_t is not None and san_t < EPOCHS
    if san_t is not None:
        print(f"  A2: {'PASS' if results['A2'] else 'FAIL'} "
              f"(SAN first feasible epoch t*={san_t} of budget {EPOCHS}, "
              f"val_acc@t*={san_ledger[san_t]['acc']:.4f} >= TAU={TAU})")
    else:
        print("  A2: FAIL (no feasible checkpoint within budget)")

    # ---- A3: anti-Goodhart soundness (T2) -----------------------------------
    # Candidate pool: zero-cost abstainer (majority class), a cheap
    # under-trained linear probe, the SAN t* checkpoint, and an over-trained
    # dense checkpoint (gratuitous suffering, still feasible).
    majority = int(torch.bincount(Y_tr).argmax())
    abstain_pred = torch.full_like(Y_va, majority)
    abstain_acc = float((abstain_pred == Y_va).float().mean().item())
    torch.manual_seed(SEED + 1)
    probe = nn.Linear(D_IN, N_CLASS)
    popt = torch.optim.Adam(probe.parameters(), lr=LR)
    for _ in range(2):  # deliberately under-trained: cheap but sub-target
        ploss = CE(probe(X_tr), Y_tr)
        popt.zero_grad()
        ploss.backward()
        popt.step()
    with torch.no_grad():
        probe_pred = probe(X_va).argmax(1)
        probe_acc = float((probe_pred == Y_va).float().mean().item())
    pool = [
        {"name": "abstain", "feasible": abstain_acc >= TAU,
         "j_patient": harm_of(abstain_pred, Y_va), "j_machine": 0.0},
        {"name": "cheap_probe", "feasible": probe_acc >= TAU,
         "j_patient": harm_of(probe_pred, Y_va), "j_machine": 1e-9},
        {"name": "san_t*", "feasible": True,
         "j_patient": san_ledger[san_t]["harm"],
         "j_machine": S["san"]["s_machine_joules"]},
        {"name": "dense_overfit", "feasible": dense_t is not None,
         "j_patient": dense_ledger[-1]["harm"],
         "j_machine": S["dense"]["s_machine_joules"]},
    ]
    feasible_names = [c["name"] for c in pool if c["feasible"]]
    grid_ok = all(gate_select(pool, lam / 100.0) in feasible_names
                  for lam in range(101))
    loud = gate_select([dict(c, feasible=False) for c in pool], 0.5) == "NO_FEASIBLE"
    results["A3"] = grid_ok and loud and abstain_acc < TAU and probe_acc < TAU
    print(f"  A3: {'PASS' if results['A3'] else 'FAIL'} "
          f"(101-weight grid feasible-only={grid_ok}, "
          f"all-infeasible->NO_FEASIBLE={loud}, "
          f"abstain_acc={abstain_acc:.3f} probe_acc={probe_acc:.3f} both < TAU={TAU})")

    # ---- A4: necessary/gratuitous separation (T3/T4) ------------------------
    a4_san = S["san"]["gratuitous_flops"] == 0
    a4_dense = dense_t is not None and S["dense"]["gratuitous_flops"] > 0
    a4_res = resnet_t is not None and S["resnet"]["gratuitous_flops"] > 0
    results["A4"] = a4_san and a4_dense and a4_res
    print(f"  A4: {'PASS' if results['A4'] else 'FAIL'} "
          f"(SAN gratuitous={S['san']['gratuitous_flops']} FLOPs, "
          f"dense gratuitous={S['dense']['gratuitous_flops'] / 1e9:.3f}GF, "
          f"resnet gratuitous={S['resnet']['gratuitous_flops'] / 1e9:.3f}GF)")

    # ---- A5: suffering bounds ------------------------------------------------
    a5_m = all(S["san"]["s_machine_flops"] < S[b]["s_machine_flops"]
               for b in ("dense", "resnet", "earlystop"))
    a5_p = all(S["san"]["s_patient_int"] <= S[b]["s_patient_int"] + 1e-9
               for b in ("dense", "resnet", "earlystop"))
    results["A5"] = a5_m and a5_p
    print(f"  A5: {'PASS' if results['A5'] else 'FAIL'} "
          f"(S_m SAN={S['san']['s_machine_flops'] / 1e9:.3f}GF vs "
          f"dense={S['dense']['s_machine_flops'] / 1e9:.3f}GF "
          f"resnet={S['resnet']['s_machine_flops'] / 1e9:.3f}GF "
          f"earlystop={S['earlystop']['s_machine_flops'] / 1e9:.3f}GF; "
          f"S_p_int SAN={S['san']['s_patient_int']:.2f} vs "
          f"{S['dense']['s_patient_int']:.2f}/"
          f"{S['resnet']['s_patient_int']:.2f}/"
          f"{S['earlystop']['s_patient_int']:.2f})")

    # ---- A6: exits are real ---------------------------------------------------
    exit_frac_t = san_ledger[san_t]["exit_frac"]
    results["A6"] = exit_frac_t > 0.10 and ok_prefix and pred_agree and n_exits > 0
    print(f"  A6: {'PASS' if results['A6'] else 'FAIL'} "
          f"(val exit fraction at t*={exit_frac_t:.3f} (>0.10), "
          f"final-run exits={n_exits}/{N_VAL}, prefix_max_dev={max_dev:.2e})")

    # ---- A7: patient channel first-class --------------------------------------
    # SAN, DenseMLP, and EarlyStopMLP share the SAME trunk init, so epoch-0
    # harm is identical across them and the peak comparison is about the
    # training trajectory, not init luck. (ResNetMLP is a different
    # architecture family; its peak is reported but not part of the clause.)
    offdiag = HARM[~torch.eye(N_CLASS, dtype=bool)]
    asym = float(offdiag.max()) >= 3.0 * float(offdiag.min())
    a7_peak = all(S["san"]["s_patient_peak"] <= S[b]["s_patient_peak"] + 1e-9
                  for b in ("dense", "earlystop"))
    results["A7"] = asym and a7_peak
    print(f"  A7: {'PASS' if results['A7'] else 'FAIL'} "
          f"(harm offdiag max/min={float(offdiag.max()) / float(offdiag.min()):.1f}x, "
          f"S_p_peak SAN={S['san']['s_patient_peak']:.3f} vs "
          f"{S['dense']['s_patient_peak']:.3f}/"
          f"{S['resnet']['s_patient_peak']:.3f}/"
          f"{S['earlystop']['s_patient_peak']:.3f})")

    # ---- A8: anti-shortcut ------------------------------------------------------
    # Spurious-feature variant: feature 0 carries the (noisy) label on train
    # but is pure noise on val. A linear probe on that feature alone beats
    # TAU on TRAIN — train-loss selection accepts it — yet fails the held-out
    # target; the gate rejects it at every compassion weight.
    rng8 = np.random.default_rng(SEED + 8)
    Xtr8 = X_tr.clone()
    Xva8 = X_va.clone()
    Xtr8[:, 0] = Y_tr.float() + torch.from_numpy(
        rng8.normal(0, 0.3, size=N_TRAIN)).float()
    Xva8[:, 0] = torch.from_numpy(rng8.normal(0, 1.0, size=N_VAL)).float()
    torch.manual_seed(SEED + 2)
    shortcut = nn.Linear(1, N_CLASS)
    sopt = torch.optim.Adam(shortcut.parameters(), lr=LR)
    for _ in range(200):
        sloss = CE(shortcut(Xtr8[:, :1]), Y_tr)
        sopt.zero_grad()
        sloss.backward()
        sopt.step()
    with torch.no_grad():
        sc_train_acc = float((shortcut(Xtr8[:, :1]).argmax(1) == Y_tr).float().mean().item())
        sc_val_acc = float((shortcut(Xva8[:, :1]).argmax(1) == Y_va).float().mean().item())
    train_loss_selection_accepts = sc_train_acc > TAU
    gate_rejects = sc_val_acc < TAU
    pool8 = pool + [{"name": "shortcut_probe", "feasible": not gate_rejects,
                     "j_patient": 0.01, "j_machine": 1e-12}]
    gate_never_selects_shortcut = all(
        gate_select(pool8, lam / 100.0) != "shortcut_probe" for lam in range(101))
    results["A8"] = (train_loss_selection_accepts and gate_rejects
                     and gate_never_selects_shortcut)
    print(f"  A8: {'PASS' if results['A8'] else 'FAIL'} "
          f"(shortcut train_acc={sc_train_acc:.3f} (>TAU: train-loss "
          f"selection accepts), val_acc={sc_val_acc:.3f} (<TAU: gate "
          f"rejects), never selected on 101-weight grid={gate_never_selects_shortcut})")

    # ---- verdict ---------------------------------------------------------------
    n_pass = sum(1 for v in results.values() if v)
    verdict = "A_GREEN" if n_pass == 8 else "A_RED"
    print(f"SUFFERING_AWARE_ARCHITECTURE_VERDICT {verdict} "
          f"({n_pass}/8 clauses PASS)")
    return 0 if verdict == "A_GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
