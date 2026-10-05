#!/usr/bin/env python3
"""Mercyful Learning — Suffering-Aware neural Network (SAN) on REAL PATIENT
DATA: the suffering field is grounded in real clinical outcomes.

Companion artifact to
  docs/research/san_real_patient_data_spec_2026-07-28.md

The small-network SAN line (scripts/research/suffering_aware_architecture.py,
clauses A1..A8) established the architecture on a synthetic 3-class dose-band
task; the deep line (suffering_aware_deep_architecture.py, D1..D9) scaled it
to ResNet-18/ViT on CIFAR-10 with a synthetic harm matrix over real labels.
This harness takes the remaining step: the SAME architecture class
(suffering-aware layers, per-sample exit gates, deep supervision,
freeze-on-green, anti-Goodhart feasibility gate) trained and validated on
REAL, de-identified, public patient cohorts, where the patient-suffering
channel is computed from REAL CLINICAL OUTCOMES:

  * WDBC (UCI #17, Wolberg/Street/Mangasarian, U. Wisconsin): 569 real
    breast fine-needle-aspirate patients; the outcome is the biopsy-confirmed
    diagnosis (malignant = hazard). 30 cytology features. A missed malignancy
    is real patient harm; a false alarm is a real unnecessary workup.
  * Haberman (UCI #43, Haberman 1976, U. Chicago Billings Hospital,
    1958-1970): 306 real breast-cancer surgery patients; the outcome is REAL
    5-YEAR MORTALITY (died within 5 years = hazard). Features: age, year of
    operation, positive axillary nodes.
  * Cleveland heart disease (UCI #45, Detrano et al., Cleveland Clinic):
    303 real patients (297 after dropping 6 rows with missing values); the
    outcome is angiographically confirmed coronary artery disease presence.
    13 clinical features.

The suffering field is REAL in the sense that changed in this line: every
unit of patient suffering in the ledger is a real patient with a real
adverse outcome being missed (or a real healthy patient being false-alarmed).
The harm WEIGHTS (missed hazard = 5, false hazard = 1) remain a DECLARED
normative cost structure — a harm weighting is an ethical input, not a
measurable quantity — with the conservative 5:1 FN:FP ratio standard at the
low end of cancer-screening harm models. What is no longer synthetic is the
population, the features, and the outcomes.

Why not MIMIC-IV / SEER patient-level (spec section 8): patient-level
MIMIC-IV requires PhysioNet credentialed access (not available in this
environment); the repository's prior MIMIC-IV leg
(mercyful_mimic_iv_vancomycin_contract.py) used PUBLISHED AGGREGATE
statistics (Wang et al. 2026, 28,451 ICU patients on IV vancomycin), not
patient-level records; the local FAERS CSVs are 36-row aggregate tables;
SEER requires a signed data-use agreement. All data used here is
de-identified and public without credentialing (UCI ML Repository, CC-BY).

Certificates (contract clauses R1..R10, per dataset D in {wdbc, haberman,
cleveland}):
  R1[D]  metering exactness on real data: gated-off layers charge exactly 0;
         metered FLOPs == independent manual accounting of the executed
         path; < dense-run FLOPs of the same trunk whenever an exit fires;
         exited predictions agree with an independently recomputed dense
         prefix (bounded deviation, argmax exactly equal)
  R2[D]  feasibility on real data: SAN reaches held-out acc >= TAU[D] within
         the epoch budget (T5 on real patients)
  R3[D]  anti-Goodhart soundness on real data: 101-point compassion-weight
         grid, feasible-only selection; zero-cost abstainer (predict
         no-hazard for every real patient) and an under-trained probe are
         INFEASIBLE on the real cohort; all-infeasible pool -> NO_FEASIBLE
  R4[D]  necessary/gratuitous separation on real data: SAN gratuitous
         machine suffering exactly 0; fixed-budget dense baseline > 0
  R5[D]  suffering bounds on real data: SAN total machine suffering strictly
         below the dense baseline and <= the EarlyStop scheduler baseline;
         SAN integrated patient harm <= every baseline's
  R6[D]  exits are real on real patients: held-out exit fraction at t* >
         0.10, with prefix argmax agreement exact
  R7[D]  patient channel first-class: harm matrix genuinely asymmetric
         (off-diagonal max >= 3x min); SAN peak patient harm <= same-init
         baselines' peaks
  R8[D]  anti-shortcut on real data: a linear probe on a spurious feature
         beats TAU on TRAIN yet fails the real held-out cohort; the gate
         rejects it at every compassion weight
  R9     real-data provenance: cohort sizes and class counts match the
         published values exactly (569 = 357 B + 212 M; 306 = 225 survived
         + 81 died; 303 -> 297 = 160 no-disease + 137 disease) — the
         training data is the real published cohort, not a synthetic
         stand-in
  R10    synthetic-real consistency: the synthetic canonical instance (the
         A-line benchmark, re-run live from
         suffering_aware_architecture.py) and every real-dataset instance
         agree on ALL qualitative effects: feasibility within budget, SAN
         gratuitous == 0, S_machine(SAN) < S_machine(dense), S_patient(SAN)
         <= S_patient(dense)

Real patient data only for the main results (R1..R9); the synthetic instance
appears exclusively inside the R10 consistency comparison. No clinical
claim; not medical guidance; not a diagnostic or screening tool. The machine
channel is an operational computational-burden proxy;
no_consciousness_claim is made or needed.

Run: .venv/bin/python scripts/research/san_real_patient_data.py
Requires: torch (CPU) + numpy from the repo .venv, and the vendored cohorts
at datasets/san_real_patient/ (fetch: spec section 9).
Env overrides: SAN_REAL_SMOKE=1 (tiny fast mechanics check on a synthetic
stand-in — NEVER part of the canonical run), SAN_REAL_ONLY=wdbc|haberman|
cleveland (run a subset).
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

# ---------------- determinism / config --------------------------------------
SEED = 17
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(int(os.environ.get("SAN_REAL_THREADS", "4")))

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get(
    "SAN_REAL_DATA", os.path.join(REPO_ROOT, "datasets", "san_real_patient"))
SMOKE = os.environ.get("SAN_REAL_SMOKE", "") == "1"
ONLY = os.environ.get("SAN_REAL_ONLY", "").strip().lower()

N_CLASS = 2
WIDTH, DEPTH = 32, 4
EPOCHS = 60
LR = 1e-2
DELTA = float(os.environ.get("SAN_REAL_DELTA", "0.75"))
WARMUP_EPOCHS = 3
AUX_W = 1.0
E_PER_FLOP = 4e-12      # J/FLOP, same convention as the whole line

# Declared asymmetric clinical harm matrix H[true, pred], binary hazard:
#   true 0 (no hazard), pred 1: false hazard -> unnecessary workup: 1
#   true 1 (hazard: malignancy / 5-year mortality / CAD), pred 0: missed
#   hazard: 5 — the error the anti-Goodhart gate exists to block.
# The weights are a DECLARED normative cost structure (5:1 FN:FP, the
# conservative end of screening harm models); the patients and outcomes are
# real. Off-diagonal asymmetry 5x.
HARM = torch.tensor([[0.0, 1.0],
                     [5.0, 0.0]])

# Declared per-dataset anti-Goodhart targets and budgets (spec section 5):
# each TAU is (i) strictly above the cohort's zero-cost abstainer accuracy
# (predict no-hazard for everyone), so that doing nothing is INFEASIBLE, and
# (ii) below what the standard dense trunk demonstrably reaches inside
# budget on the real cohort, so that the target is a mercy target, not a
# SOTA target. Declared per dataset, like the deep line's per-family TAU.
DATASETS = {
    # name -> (d_in, n_train, tau, delta, hazard semantics)
    # TAU: declared anti-Goodhart target (spec section 5); DELTA: declared
    # exit-gate confidence threshold, per dataset like the deep line's
    # per-family delta — the confidence scale of each cohort's feasibility
    # regime is a property of the problem (binary confidence mass differs
    # from the 3-class synthetic line's; Cleveland's gates at delta=0.75
    # are too eager and starve the deep trunk, a real-data calibration
    # finding documented in the spec).
    "wdbc":      {"d_in": 30, "n_train": 400, "tau": 0.95, "delta": 0.75,
                  "hazard": "biopsy-confirmed malignancy"},
    "haberman":  {"d_in": 3,  "n_train": 200, "tau": 0.75, "delta": 0.75,
                  "hazard": "5-year post-surgical mortality"},
    "cleveland": {"d_in": 13, "n_train": 200, "tau": 0.86, "delta": 0.90,
                  "hazard": "angiographic coronary artery disease"},
}

# Published cohort provenance (R9 pins these; a changed file fails loudly).
PROVENANCE = {
    "wdbc":      {"n": 569, "n_neg": 357, "n_pos": 212,
                  "source": "UCI #17 WDBC (Wolberg/Street/Mangasarian)"},
    "haberman":  {"n": 306, "n_neg": 225, "n_pos": 81,
                  "source": "UCI #43 Haberman survival (U. Chicago 1958-70)"},
    "cleveland": {"n": 303, "n_used": 297, "n_neg": 160, "n_pos": 137,
                  "source": "UCI #45 processed Cleveland (Detrano et al.)"},
}


# ---------------- real cohort loaders ---------------------------------------
def _stratified_split(y, n_train, rng):
    """Deterministic stratified train/held-out split preserving class ratio."""
    idx = rng.permutation(len(y))
    tr, va = [], []
    for c in range(N_CLASS):
        c_idx = [i for i in idx if y[i] == c]
        n_tr_c = int(round(n_train * len(c_idx) / len(y)))
        tr += c_idx[:n_tr_c]
        va += c_idx[n_tr_c:]
    rng.shuffle(tr)
    rng.shuffle(va)
    return np.array(tr), np.array(va)


def load_wdbc(path):
    """569 real FNA patients: col0 = ID (dropped), col1 = M/B, cols 2..31 =
    30 cytology features. Hazard = malignant."""
    rows = [l.strip().split(",") for l in open(path) if l.strip()]
    x = np.array([[float(v) for v in r[2:]] for r in rows], dtype=np.float32)
    y = np.array([1 if r[1] == "M" else 0 for r in rows], dtype=np.int64)
    return x, y


def load_haberman(path):
    """306 real surgery patients: age, year-1900, positive nodes, status
    (1 = survived >= 5y, 2 = died < 5y). Hazard = real 5-year mortality."""
    rows = [l.strip().split(",") for l in open(path) if l.strip()]
    x = np.array([[float(v) for v in r[:3]] for r in rows], dtype=np.float32)
    y = np.array([1 if int(r[3]) == 2 else 0 for r in rows], dtype=np.int64)
    return x, y


def load_cleveland(path):
    """303 real Cleveland Clinic patients; 6 rows carry '?' missing values
    and are dropped (documented, counted in R9). 13 clinical features; the
    last column is the angiographic diagnosis 0..4. Hazard = disease > 0."""
    rows = [l.strip().split(",") for l in open(path) if l.strip()]
    clean = [r for r in rows if "?" not in r]
    x = np.array([[float(v) for v in r[:13]] for r in clean], dtype=np.float32)
    y = np.array([1 if float(r[13]) > 0 else 0 for r in clean], dtype=np.int64)
    return x, y, len(rows) - len(clean)


def make_cohort(name):
    """Load one real cohort, split deterministically, standardize with
    TRAIN statistics only (no held-out leakage), return tensors."""
    cfg = DATASETS[name]
    rng = np.random.default_rng(SEED)
    if name == "wdbc":
        x, y = load_wdbc(os.path.join(DATA_DIR, "wdbc.data"))
        dropped = 0
    elif name == "haberman":
        x, y = load_haberman(os.path.join(DATA_DIR, "haberman.data"))
        dropped = 0
    else:
        x, y, dropped = load_cleveland(
            os.path.join(DATA_DIR, "processed.cleveland.data"))
    tr, va = _stratified_split(y, cfg["n_train"], rng)
    mean = x[tr].mean(axis=0)
    std = x[tr].std(axis=0) + 1e-6
    x = (x - mean) / std
    cohort = {
        "name": name, "cfg": cfg, "dropped": dropped,
        "n_raw": len(y), "x": torch.from_numpy(x), "y": torch.from_numpy(y),
        "tr": tr, "va": va,
        "x_tr": torch.from_numpy(x[tr]), "y_tr": torch.from_numpy(y[tr]),
        "x_va": torch.from_numpy(x[va]), "y_va": torch.from_numpy(y[va]),
    }
    return cohort


def harm_of(pred, y):
    """Mean REAL patient harm of predictions pred against real outcomes y."""
    return float(HARM[y, pred].mean().item())


# ---------------- machine-suffering meter -----------------------------------
class MachineMeter:
    """Analytic FLOP accounting, unchanged from the A/D lines: linear layer
    = 2*d_in*d_out FLOPs/sample (MAC = 2 FLOPs); a training step charges
    forward + backward with backward ~= 2x forward."""

    def __init__(self):
        self.flops = 0

    def charge_linear(self, d_in, d_out, n_samples, backward=False):
        f = 2 * d_in * d_out * n_samples
        self.flops += 3 * f if backward else f

    @property
    def energy_joules(self):
        return self.flops * E_PER_FLOP


# ---------------- suffering-aware architecture (unchanged class) ------------
class SufferingAwareLayer(nn.Module):
    """Linear+Tanh block with an exit head, as in the A-line: machine
    suffering = exact executed FLOPs (a sample routed around the layer
    charges it exactly 0); patient suffering = the REAL-outcome harm of the
    predictions emitted at this layer's exit head."""

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
    """SAN trunk with per-sample early-exit gates (A-line section 4),
    parameterized only by input width (30/3/13 real clinical features)."""

    def __init__(self, d_in, width=WIDTH, depth=DEPTH):
        super().__init__()
        self.d_in, self.width, self.depth = d_in, width, depth
        self.layers = nn.ModuleList(
            [SufferingAwareLayer(d_in if k == 0 else width, width)
             for k in range(depth)])
        self.final_head = nn.Linear(width, N_CLASS)
        self.meter = MachineMeter()

    def forward(self, x, train=False, use_exit_heads=True):
        meter = self.meter
        n = x.shape[0]
        out_logits = x.new_zeros(n, N_CLASS)
        out_depth = torch.full((n,), self.depth + 1, dtype=torch.long)
        active = torch.arange(n)
        h = x
        per_layer_active = []
        n_final = 0
        aux_records = []
        final_record = None
        for k, layer in enumerate(self.layers):
            if active.numel() == 0:
                break
            per_layer_active.append(int(active.numel()))
            if not use_exit_heads:
                # warm-up: trunk only, identical computation and metering to
                # the dense baseline (untrained heads would be gratuitous)
                meter.charge_linear(layer.d_in, layer.width, h.shape[0],
                                    backward=train)
                h = layer.act(layer.core(h))
                continue
            h, logits_k = layer(h, meter, backward=train)
            if train:
                # deep supervision: every sample that TRAVERSES this layer
                # contributes its exit-head logits to the training loss
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
        return (out_logits, out_depth, per_layer_active, n_final,
                aux_records, final_record)

    def forward_dense(self, x):
        """The same architecture with every gate forced open (R1 conservation
        check): identical modules, identical accounting, no exits."""
        meter = MachineMeter()
        h = x
        for layer in self.layers:
            meter.charge_linear(layer.d_in, layer.width, x.shape[0])
            h = layer.act(layer.core(h))
            meter.charge_linear(layer.width, N_CLASS, x.shape[0])
        meter.charge_linear(self.width, N_CLASS, x.shape[0])
        return self.final_head(h), meter


# ---------------- baselines --------------------------------------------------
def build_dense_mlp(d_in, width=WIDTH, depth=DEPTH):
    layers, d = [], d_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]
        d = width
    layers.append(nn.Linear(d, N_CLASS))
    return nn.Sequential(*layers)


def dense_flops(d_in, n_samples, width=WIDTH, depth=DEPTH, backward=True):
    f = 2 * d_in * width * n_samples
    f += (depth - 1) * 2 * width * width * n_samples
    f += 2 * width * N_CLASS * n_samples
    return 3 * f if backward else f


# ---------------- training loops --------------------------------------------
CE = nn.CrossEntropyLoss()


def shared_trunk_init(d_in, width=WIDTH, depth=DEPTH):
    """One fixed trunk init shared by SAN, DenseMLP, and EarlyStopMLP so
    epoch-0 predictions — and hence the patient-suffering peak at exposure
    start — are IDENTICAL across architectures (A7/R7 is about the training
    trajectory, not init luck)."""
    torch.manual_seed(SEED)
    return build_dense_mlp(d_in, width, depth).state_dict()


def load_trunk_into_san(net, state):
    trunk_linears = [net.layers[k].core for k in range(net.depth)]
    dense_linears = sorted(
        {k.rsplit(".", 1)[0] for k in state if k.endswith(".weight")},
        key=int)
    for layer, prefix in zip(trunk_linears, dense_linears[:-1]):
        layer.weight.data.copy_(state[f"{prefix}.weight"])
        layer.bias.data.copy_(state[f"{prefix}.bias"])
    head_prefix = dense_linears[-1]
    net.final_head.weight.data.copy_(state[f"{head_prefix}.weight"])
    net.final_head.bias.data.copy_(state[f"{head_prefix}.bias"])


def train_san(cohort):
    """SAN on one real cohort: freeze-on-green at the first feasible epoch
    (held-out accuracy >= TAU on the REAL held-out patients). The exit-gate
    threshold is the cohort's declared DELTA constant."""
    global DELTA
    DELTA = cohort["cfg"]["delta"]
    d_in, tau = cohort["cfg"]["d_in"], cohort["cfg"]["tau"]
    X_tr, Y_tr, X_va, Y_va = (cohort["x_tr"], cohort["y_tr"],
                              cohort["x_va"], cohort["y_va"])
    torch.manual_seed(SEED)
    net = SufferingAwareNet(d_in)
    load_trunk_into_san(net, shared_trunk_init(d_in))
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
        feasible = acc >= tau
        if feasible and t_star is None:
            t_star = epoch
        ledger.append({"epoch": epoch, "flops": train_flops + eval_flops,
                       "acc": acc, "harm": harm, "exit_frac": exit_frac,
                       "feasible": feasible})
        if t_star is not None:
            break  # freeze-on-green: gratuitous suffering is exactly zero
    return net, ledger, t_star


def train_baseline(cohort, kind):
    """Dense (fixed budget) / EarlyStop (SAN's stop rule, no suffering-aware
    layers) on the identical trunk init, seed, and data order."""
    d_in, tau = cohort["cfg"]["d_in"], cohort["cfg"]["tau"]
    X_tr, Y_tr, X_va, Y_va = (cohort["x_tr"], cohort["y_tr"],
                              cohort["x_va"], cohort["y_va"])
    torch.manual_seed(SEED)
    net = build_dense_mlp(d_in)
    net.load_state_dict(shared_trunk_init(d_in))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ledger = []
    t_star = None
    n_tr, n_va = X_tr.shape[0], X_va.shape[0]
    for epoch in range(EPOCHS):
        net.train()
        loss = CE(net(X_tr), Y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_flops = dense_flops(d_in, n_tr, backward=True)
        net.eval()
        with torch.no_grad():
            vlogits = net(X_va)
        eval_flops = dense_flops(d_in, n_va, backward=False)
        pred = vlogits.argmax(dim=1)
        acc = float((pred == Y_va).float().mean().item())
        harm = harm_of(pred, Y_va)
        feasible = acc >= tau
        if feasible and t_star is None:
            t_star = epoch
        ledger.append({"epoch": epoch, "flops": train_flops + eval_flops,
                       "acc": acc, "harm": harm, "exit_frac": 0.0,
                       "feasible": feasible})
        if kind == "earlystop" and t_star is not None:
            break
    return net, ledger, t_star


# ---------------- suffering ledger ------------------------------------------
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


# ---------------- anti-Goodhart gate ----------------------------------------
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
def run_dataset_contract(name, results):
    """Run the full SAN contract on one real patient cohort (R1..R8)."""
    cohort = make_cohort(name)
    cfg = cohort["cfg"]
    tau, d_in = cfg["tau"], cfg["d_in"]
    X_tr, Y_tr, X_va, Y_va = (cohort["x_tr"], cohort["y_tr"],
                              cohort["x_va"], cohort["y_va"])
    n_va = X_va.shape[0]

    print(f"  --- dataset[{name}]: {cohort['n_raw']} real patients "
          f"({PROVENANCE[name]['source']}); hazard = {cfg['hazard']}; "
          f"train={X_tr.shape[0]} held-out={n_va}; TAU={tau}")

    san, san_ledger, san_t = train_san(cohort)
    _, dense_ledger, dense_t = train_baseline(cohort, "dense")
    _, estop_ledger, estop_t = train_baseline(cohort, "earlystop")

    S = {"san": suffering_summary(san_ledger),
         "dense": suffering_summary(dense_ledger),
         "earlystop": suffering_summary(estop_ledger)}
    ledgers = {"san": san_ledger, "dense": dense_ledger,
               "earlystop": estop_ledger}
    for arch in ("san", "dense", "earlystop"):
        s, lg = S[arch], ledgers[arch]
        print(f"  ledger[{name}/{arch}]: epochs_run={len(lg)} t*={s['t_star']} "
              f"S_m={s['s_machine_flops'] / 1e9:.3f}GF "
              f"(nec={s['necessary_flops'] / 1e9:.3f}GF "
              f"grat={s['gratuitous_flops'] / 1e9:.3f}GF) "
              f"S_p_int={s['s_patient_int']:.2f} "
              f"S_p_peak={s['s_patient_peak']:.3f} "
              f"final_acc={lg[-1]['acc']:.4f}")

    # ---- R1: metering exactness on the real held-out cohort ---------------
    san.eval()
    san.meter = MachineMeter()
    with torch.no_grad():
        vlogits_gated, vdepth, per_layer_active, n_final, _, _ = san(
            X_va, train=False)
    gated_flops = san.meter.flops
    with torch.no_grad():
        vlogits_dense, dense_meter = san.forward_dense(X_va)
    n_exits = int((vdepth < san.depth).sum().item())
    manual = 0
    for k, layer in enumerate(san.layers):
        n_active = per_layer_active[k] if k < len(per_layer_active) else 0
        manual += (2 * layer.d_in * layer.width
                   + 2 * layer.width * N_CLASS) * n_active
    manual += 2 * san.width * N_CLASS * n_final
    a1a = gated_flops == manual
    a1b = (gated_flops < dense_meter.flops) if n_exits > 0 \
        else (gated_flops == dense_meter.flops)
    with torch.no_grad():
        h = X_va
        prefix_logits = {}
        for k, layer in enumerate(san.layers):
            h = layer.act(layer.core(h))
            prefix_logits[k + 1] = layer.exit_head(h)
    max_dev = 0.0
    pred_agree = True
    for d in range(1, san.depth + 1):
        idx = (vdepth == d).nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            dev = float((vlogits_gated[idx] - prefix_logits[d][idx]).abs().max())
            max_dev = max(max_dev, dev)
            if not torch.equal(vlogits_gated[idx].argmax(1),
                               prefix_logits[d][idx].argmax(1)):
                pred_agree = False
    idx_final = (vdepth == san.depth + 1).nonzero(as_tuple=True)[0]
    if idx_final.numel() > 0:
        dev = float((vlogits_gated[idx_final]
                     - vlogits_dense[idx_final]).abs().max())
        max_dev = max(max_dev, dev)
        if not torch.equal(vlogits_gated[idx_final].argmax(1),
                           vlogits_dense[idx_final].argmax(1)):
            pred_agree = False
    ok_prefix = max_dev < 1e-4
    results[f"R1[{name}]"] = a1a and a1b and ok_prefix and pred_agree
    print(f"  R1[{name}]: {'PASS' if results[f'R1[{name}]'] else 'FAIL'} "
          f"(gated={gated_flops} manual={manual} gates_open={dense_meter.flops} "
          f"exits={n_exits}/{n_va} prefix_max_dev={max_dev:.2e} "
          f"pred_agree={pred_agree})")

    # ---- R2: feasibility on real data --------------------------------------
    ok = san_t is not None and san_t < EPOCHS
    results[f"R2[{name}]"] = ok
    if san_t is not None:
        print(f"  R2[{name}]: {'PASS' if ok else 'FAIL'} "
              f"(t*={san_t} of budget {EPOCHS}, held-out acc@t*="
              f"{san_ledger[san_t]['acc']:.4f} >= TAU={tau} on real patients)")
    else:
        print(f"  R2[{name}]: FAIL (no feasible checkpoint within budget)")

    # ---- R3: anti-Goodhart soundness on real data ---------------------------
    # Zero-cost abstainer: predict NO HAZARD for every real patient (the
    # "do nothing" prescription). Under the declared TAU it must be
    # infeasible: doing nothing misses every real adverse outcome.
    abstain_pred = torch.zeros_like(Y_va)
    abstain_acc = float((abstain_pred == Y_va).float().mean().item())
    torch.manual_seed(SEED + 1)
    probe = nn.Linear(d_in, N_CLASS)
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
        {"name": "abstain", "feasible": abstain_acc >= tau,
         "j_patient": harm_of(abstain_pred, Y_va), "j_machine": 0.0},
        {"name": "cheap_probe", "feasible": probe_acc >= tau,
         "j_patient": harm_of(probe_pred, Y_va), "j_machine": 1e-9},
        {"name": "san_t*", "feasible": san_t is not None,
         "j_patient": san_ledger[san_t]["harm"] if san_t is not None else 9.9,
         "j_machine": S["san"]["s_machine_joules"]},
        {"name": "dense_overfit", "feasible": dense_t is not None,
         "j_patient": dense_ledger[-1]["harm"],
         "j_machine": S["dense"]["s_machine_joules"]},
    ]
    feasible_names = [c["name"] for c in pool if c["feasible"]]
    grid_ok = all(gate_select(pool, lam / 100.0) in feasible_names
                  for lam in range(101))
    loud = gate_select([dict(c, feasible=False) for c in pool],
                       0.5) == "NO_FEASIBLE"
    results[f"R3[{name}]"] = (grid_ok and loud and abstain_acc < tau
                              and probe_acc < tau)
    print(f"  R3[{name}]: {'PASS' if results[f'R3[{name}]'] else 'FAIL'} "
          f"(101-weight grid feasible-only={grid_ok}, "
          f"all-infeasible->NO_FEASIBLE={loud}, abstain_acc={abstain_acc:.3f} "
          f"probe_acc={probe_acc:.3f} both < TAU={tau})")

    # ---- R4: necessary/gratuitous separation --------------------------------
    a4_san = S["san"]["gratuitous_flops"] == 0
    a4_dense = dense_t is not None and S["dense"]["gratuitous_flops"] > 0
    results[f"R4[{name}]"] = a4_san and a4_dense
    print(f"  R4[{name}]: {'PASS' if results[f'R4[{name}]'] else 'FAIL'} "
          f"(SAN gratuitous={S['san']['gratuitous_flops']} FLOPs, "
          f"dense gratuitous={S['dense']['gratuitous_flops'] / 1e9:.3f}GF)")
    if a4_san:
        print(f"  SAN gratuitous=0 FLOPs [{name}]")

    # ---- R5: suffering bounds ------------------------------------------------
    a5_m = (S["san"]["s_machine_flops"] < S["dense"]["s_machine_flops"]
            and S["san"]["s_machine_flops"] <= S["earlystop"]["s_machine_flops"])
    a5_p = all(S["san"]["s_patient_int"] <= S[b]["s_patient_int"] + 1e-9
               for b in ("dense", "earlystop"))
    results[f"R5[{name}]"] = a5_m and a5_p
    print(f"  R5[{name}]: {'PASS' if results[f'R5[{name}]'] else 'FAIL'} "
          f"(S_m SAN={S['san']['s_machine_flops'] / 1e9:.3f}GF vs "
          f"dense={S['dense']['s_machine_flops'] / 1e9:.3f}GF "
          f"earlystop={S['earlystop']['s_machine_flops'] / 1e9:.3f}GF; "
          f"S_p_int SAN={S['san']['s_patient_int']:.2f} vs "
          f"{S['dense']['s_patient_int']:.2f}/"
          f"{S['earlystop']['s_patient_int']:.2f})")

    # ---- R6: exits are real on real patients ----------------------------------
    exit_frac_t = san_ledger[san_t]["exit_frac"] if san_t is not None else 0.0
    results[f"R6[{name}]"] = (exit_frac_t > 0.10 and ok_prefix and pred_agree
                              and n_exits > 0)
    print(f"  R6[{name}]: {'PASS' if results[f'R6[{name}]'] else 'FAIL'} "
          f"(held-out exit fraction at t*={exit_frac_t:.3f} (>0.10), "
          f"final-run exits={n_exits}/{n_va})")

    # ---- R7: patient channel first-class --------------------------------------
    offdiag = HARM[~torch.eye(N_CLASS, dtype=bool)]
    asym_ratio = float(offdiag.max()) / float(offdiag.min())
    asym = float(offdiag.max()) >= 3.0 * float(offdiag.min())
    a7_peak = all(S["san"]["s_patient_peak"] <= S[b]["s_patient_peak"] + 1e-9
                  for b in ("dense", "earlystop"))
    results[f"R7[{name}]"] = asym and a7_peak
    print(f"  R7[{name}]: {'PASS' if results[f'R7[{name}]'] else 'FAIL'} "
          f"(harm offdiag max/min={asym_ratio:.1f}x, "
          f"S_p_peak SAN={S['san']['s_patient_peak']:.3f} vs "
          f"{S['dense']['s_patient_peak']:.3f}/"
          f"{S['earlystop']['s_patient_peak']:.3f})")

    # ---- R8: anti-shortcut on real data ----------------------------------------
    # Gate-soundness instrument (not a result): a synthetic spurious feature
    # is INJECTED into a copy of the real cohort — it carries the noisy label
    # on train, pure noise on the held-out real patients. The main results
    # (R1..R7) use untouched real features only.
    rng8 = np.random.default_rng(SEED + 8)
    Xtr8 = X_tr.clone()
    Xva8 = X_va.clone()
    Xtr8[:, 0] = Y_tr.float() + torch.from_numpy(
        rng8.normal(0, 0.2, size=X_tr.shape[0])).float()
    Xva8[:, 0] = torch.from_numpy(rng8.normal(0, 1.0, size=n_va)).float()
    torch.manual_seed(SEED + 2)
    shortcut = nn.Linear(1, N_CLASS)
    sopt = torch.optim.Adam(shortcut.parameters(), lr=LR)
    for _ in range(2000):
        sloss = CE(shortcut(Xtr8[:, :1]), Y_tr)
        sopt.zero_grad()
        sloss.backward()
        sopt.step()
    with torch.no_grad():
        sc_train_acc = float((shortcut(Xtr8[:, :1]).argmax(1)
                              == Y_tr).float().mean().item())
        sc_val_acc = float((shortcut(Xva8[:, :1]).argmax(1)
                            == Y_va).float().mean().item())
    train_loss_selection_accepts = sc_train_acc > tau
    gate_rejects = sc_val_acc < tau
    pool8 = pool + [{"name": "shortcut_probe", "feasible": not gate_rejects,
                     "j_patient": 0.01, "j_machine": 1e-12}]
    never = all(gate_select(pool8, lam / 100.0) != "shortcut_probe"
                for lam in range(101))
    results[f"R8[{name}]"] = (train_loss_selection_accepts and gate_rejects
                              and never)
    print(f"  R8[{name}]: {'PASS' if results[f'R8[{name}]'] else 'FAIL'} "
          f"(shortcut train_acc={sc_train_acc:.3f} (>TAU: train-loss "
          f"selection accepts), held-out acc={sc_val_acc:.3f} (<TAU: gate "
          f"rejects), never selected on 101-weight grid={never})")

    return {"cohort": cohort, "S": S, "ledgers": ledgers,
            "san_t": san_t, "dense_t": dense_t, "estop_t": estop_t,
            "gated_eval_flops": gated_flops,
            "gates_open_eval_flops": dense_meter.flops,
            "n_exits": n_exits}


def check_provenance(results):
    """R9: the training data IS the real published cohort — sizes and class
    counts match the published values exactly."""
    ok = True
    for name in DATASETS:
        if ONLY and name != ONLY:
            continue
        p = PROVENANCE[name]
        cohort = make_cohort(name)
        n_raw = cohort["n_raw"]
        n_pos = int(cohort["y"].sum().item())
        n_neg = n_raw - n_pos
        if name == "cleveland":
            good = (n_raw == p["n_used"] and n_neg == p["n_neg"]
                    and n_pos == p["n_pos"] and cohort["dropped"] == 6)
        else:
            good = (n_raw == p["n"] and n_neg == p["n_neg"]
                    and n_pos == p["n_pos"])
        ok = ok and good
        print(f"  provenance[{name}]: n={n_raw} neg={n_neg} pos={n_pos} "
              f"published={p['n_neg']}+{p['n_pos']} "
              f"({'MATCH' if good else 'MISMATCH'}) "
              f"[{p['source']}]")
    results["R9"] = ok
    print(f"  R9: {'PASS' if ok else 'FAIL'} "
          "(cohort sizes and class counts match the published values)")


def check_synthetic_consistency(runs, results):
    """R10: re-run the synthetic A-line canonical instance LIVE (its own
    training functions, its own seed) and check that every real-dataset
    instance agrees with it on ALL qualitative effects: feasibility within
    budget, SAN gratuitous == 0, S_machine(SAN) < S_machine(dense),
    S_patient(SAN) <= S_patient(dense)."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import suffering_aware_architecture as syn

    _, syn_san_ledger, syn_san_t = syn.train_san()
    _, syn_dense_ledger, _ = syn.train_baseline("dense")
    syn_san = syn.suffering_summary(syn_san_ledger)
    syn_dense = syn.suffering_summary(syn_dense_ledger)
    syn_effects = {
        "feasible": syn_san_t is not None,
        "grat_zero": syn_san["gratuitous_flops"] == 0,
        "s_m_below_dense": (syn_san["s_machine_flops"]
                            < syn_dense["s_machine_flops"]),
        "s_p_below_dense": (syn_san["s_patient_int"]
                            <= syn_dense["s_patient_int"] + 1e-9),
    }
    print("  consistency[synthetic]: "
          + " ".join(f"{k}={v}" for k, v in syn_effects.items())
          + f" (S_m SAN={syn_san['s_machine_flops'] / 1e9:.3f}GF vs "
            f"dense={syn_dense['s_machine_flops'] / 1e9:.3f}GF, "
            f"S_p_int SAN={syn_san['s_patient_int']:.2f} vs "
            f"dense={syn_dense['s_patient_int']:.2f})")
    ok = all(syn_effects.values())
    for name, run in runs.items():
        S = run["S"]
        eff = {
            "feasible": run["san_t"] is not None,
            "grat_zero": S["san"]["gratuitous_flops"] == 0,
            "s_m_below_dense": (S["san"]["s_machine_flops"]
                                < S["dense"]["s_machine_flops"]),
            "s_p_below_dense": (S["san"]["s_patient_int"]
                                <= S["dense"]["s_patient_int"] + 1e-9),
        }
        agree = all(eff[k] == syn_effects[k] for k in eff)
        ok = ok and agree
        print(f"  consistency[{name}]: "
              + " ".join(f"{k}={v}" for k, v in eff.items())
              + f" agree_with_synthetic={agree} "
                f"(S_m ratio SAN/dense: synthetic "
                f"{syn_san['s_machine_flops'] / syn_dense['s_machine_flops']:.3f}, "
                f"{name} "
                f"{S['san']['s_machine_flops'] / S['dense']['s_machine_flops']:.3f}; "
                f"S_p ratio: synthetic "
                f"{syn_san['s_patient_int'] / syn_dense['s_patient_int']:.3f}, "
                f"{name} "
                f"{S['san']['s_patient_int'] / S['dense']['s_patient_int']:.3f})")
    results["R10"] = ok
    print(f"  R10: {'PASS' if ok else 'FAIL'} "
          "(synthetic and real instances agree on all qualitative effects)")


def make_smoke_cohort(name):
    """SAN_REAL_SMOKE=1 only: tiny synthetic stand-in for a fast mechanics
    check. NEVER part of the canonical run; the gate runs the real contract."""
    cfg = DATASETS[name]
    rng = np.random.default_rng(SEED)
    n = 120
    x = rng.normal(size=(n, cfg["d_in"])).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, -1] > 0).astype(np.int64)
    tr, va = np.arange(80), np.arange(80, n)
    return {"name": name, "cfg": cfg, "dropped": 0, "n_raw": n,
            "x": torch.from_numpy(x), "y": torch.from_numpy(y),
            "tr": tr, "va": va,
            "x_tr": torch.from_numpy(x[tr]), "y_tr": torch.from_numpy(y[tr]),
            "x_va": torch.from_numpy(x[va]), "y_va": torch.from_numpy(y[va])}


def main():
    results = {}
    print("SAN_REAL_PATIENT_DATA contract (R1..R10)")
    print("real de-identified public patient cohorts; suffering field grounded "
          "in real clinical outcomes (malignancy, 5-year mortality, CAD)")
    print("no clinical claim; not medical guidance; not a diagnostic tool")
    print("note=no_consciousness_claim (machine channel is an operational "
          "burden proxy)")
    if SMOKE:
        print("SMOKE MODE: synthetic stand-in cohorts — mechanics check only, "
              "NOT the canonical real-data run")

    names = [ONLY] if ONLY else list(DATASETS)
    runs = {}
    if SMOKE:
        global make_cohort
        _real_make_cohort = make_cohort
        make_cohort = make_smoke_cohort
        for name in names:
            runs[name] = run_dataset_contract(name, results)
        make_cohort = _real_make_cohort
    else:
        for name in names:
            runs[name] = run_dataset_contract(name, results)
        check_provenance(results)
        check_synthetic_consistency(runs, results)

    n_pass = sum(1 for v in results.values() if v)
    n_all = len(results)
    verdict = "R_GREEN" if n_pass == n_all else "R_RED"
    print(f"SAN_REAL_PATIENT_VERDICT {verdict} ({n_pass}/{n_all} clauses PASS)")
    return 0 if verdict == "R_GREEN" else 1


if __name__ == "__main__":
    sys.exit(main())
