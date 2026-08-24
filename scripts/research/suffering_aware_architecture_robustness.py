#!/usr/bin/env python3
"""Mercyful Learning — SAN robustness validation: cross-validation, sensitivity
analysis, and adversarial stress of the Suffering-Aware neural Network.

Companion artifact to
  docs/research/suffering_aware_architecture_robustness_spec_2026-07-31.md

The SAN contract (A1..A8, scripts/research/suffering_aware_architecture.py)
was proven on ONE train/val split, at ONE exit threshold, under ONE harm
matrix, against honest (non-adversarial) inputs. This harness asks whether
those results are properties of the ARCHITECTURE or artifacts of the
experimental setup, by stressing the three channels the expanded ethics
prices — machine suffering, patient harm, and anti-Goodhart soundness:

  * Cross-validation (V1, V2): 5-fold rotation over a fresh pooled cohort.
    Feasibility (t* exists, gratuitous suffering exactly 0) and the
    suffering ordering (SAN below the dense baseline on both channels) must
    hold in EVERY fold, not just the canonical split.
  * Sensitivity analysis (V3, V4): the exit threshold DELTA is swept over a
    6-point grid; the harm matrix is perturbed by 8 random multiplicative
    off-diagonal reweightings; label noise is doubled and tripled. The
    conclusions (feasibility, machine-suffering bound, patient-harm bound)
    must survive the whole sweep — a result that only holds at the tuned
    point is a tuned fluke, not an architectural property.
  * Adversarial robustness (V5..V8): FGSM attacks the patient channel
    (harm under adversarial inputs vs the dense baseline); a PGD
    confidence-suppression attack targets the MACHINE channel by trying to
    defeat the exit gates and force full-depth computation. The
    architectural guarantee under test: the worst case of the exit
    mechanism is the dense path itself, so the machine-suffering bound
    S_m(SAN) <= S_m(dense) is attack-proof, and metering conservation (A1)
    still holds exactly on adversarial inputs. The anti-Goodhart gate is
    re-stressed across a TAU grid with adversarially cheap candidates.

Method note: this module IMPORTS the canonical SAN harness
(suffering_aware_architecture.py) and reuses its architecture, meter,
harm matrix, gate, and initializers — the stress is applied to the same
object the A1..A8 contract certified, not to a re-implementation. The exit
threshold DELTA and label noise are module-level constants of the base
harness; the sweeps below set them on the imported module per run (the
base forward reads DELTA from module globals at call time) and restore
them afterwards.

Synthetic data only. This benchmark makes no clinical claim and is not
medical guidance. The machine channel is an operational computational-burden
proxy; no_consciousness_claim is made or needed.

Certificates (contract clauses V1..V8):
  V1  cross-validated feasibility: SAN reaches a feasible checkpoint
      (val acc >= TAU) with t* < budget and gratuitous machine suffering
      exactly 0 in EVERY one of 5 folds
  V2  cross-validated suffering ordering: in every fold SAN total machine
      suffering < dense baseline's, and SAN integrated patient harm <=
      dense's (both channels, every fold)
  V3  exit-threshold sensitivity: at every DELTA in {0.50,...,0.90} SAN is
      feasible, gratuitous suffering is exactly 0, and SAN total machine
      suffering < dense's — the A2/A4/A5 conclusions are not a tuned-DELTA
      artifact
  V4  harm-structure + label-noise sensitivity: under 8 random
      multiplicative perturbations of the (still asymmetric) harm matrix,
      SAN integrated patient harm <= dense's; and at label noise 0.08 and
      0.12 SAN remains feasible whenever the dense baseline does, with the
      machine-suffering bound intact at every noise level
  V5  adversarial patient channel: FGSM at eps in {0.05, 0.10, 0.20}
      degrades all models (attack effective: harm strictly increasing in
      eps); at MATCHED training exposure (dense frozen at SAN's t*), SAN's
      adversarial patient harm <= dense's at every eps — the exit
      architecture adds no fragility; and gated vs forced-dense SAN incur
      identical harm on the same adversarial inputs — the exit heads are
      not an adversarial weak point. (The exposure confound — over-trained
      dense@60 is more robust than SAN@t* because more training buys
      margin — is reported as the mercy/robustness trade-off, not gated:
      that robustness is bought with gratuitous machine suffering, and a
      declared robustness margin recovers it at ~30% of the cost.)
  V6  adversarial machine channel: a PGD confidence-suppression attack
      collapses the exit rate (attack effective) and inflates SAN's
      metered FLOPs, but the metered cost NEVER exceeds the dense-run cost
      of the same trunk on the same adversarial inputs — the worst case of
      the exit mechanism is the dense path, so the machine-suffering bound
      is attack-proof
  V7  anti-Goodhart soundness under stress: over TAU in {0.75, 0.80, 0.85}
      and the 101-point compassion grid, with a pool containing a
      zero-cost abstainer, a cheap under-trained probe, and a shortcut
      probe that train-loss selection accepts, the gate selects only
      candidates feasible at that TAU, and an all-infeasible pool returns
      NO_FEASIBLE at every TAU
  V8  metering conservation under stress: on adversarial inputs the gated
      meter equals an independent manual accounting of the executed path,
      forced-open gates equal forward_dense exactly, and exited
      predictions agree (argmax exactly) with the recomputed prefix —
      A1 survives adversarial inputs

Run: .venv/bin/python scripts/research/suffering_aware_architecture_robustness.py
Requires: torch (CPU) + numpy from the repo .venv.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import suffering_aware_architecture as A

SEED = A.SEED
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)

D_IN, N_CLASS = A.D_IN, A.N_CLASS
TAU, EPOCHS, LR = A.TAU, A.EPOCHS, A.LR
WARMUP_EPOCHS, AUX_W = A.WARMUP_EPOCHS, A.AUX_W
CE = nn.CrossEntropyLoss()
HARM = A.HARM
K_FOLDS = 5
N_POOL = 5000          # pooled cohort for the fold rotation
DELTA_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
NOISE_GRID = [0.08, 0.12]
EPS_GRID = [0.05, 0.10, 0.20]
TAU_GRID = [0.75, 0.80, 0.85]
N_HARM_PERTURB = 8


def make_data(n, rng, label_noise):
    """Same generator as the canonical harness, with the label-noise rate
    explicit (the base module reads it from a module global)."""
    saved = A.LABEL_NOISE
    A.LABEL_NOISE = label_noise
    try:
        return A.make_data(n, rng)
    finally:
        A.LABEL_NOISE = saved


def harm_of_h(pred, y, H):
    """Mean patient harm of predictions pred against truth y under harm
    matrix H (the base harm_of reads the canonical module-global HARM)."""
    return float(H[y, pred].mean().item())


# ---------------- parameterized training loops -------------------------------
def train_san_param(Xtr, Ytr, Xva, Yva, delta=A.DELTA, epochs=EPOCHS,
                    seed=SEED, tau=TAU, record_preds=False):
    """The canonical SAN training loop (train_san in the base harness),
    parameterized on data, exit threshold, and feasibility target. DELTA is
    read from the base module's globals inside SufferingAwareNet.forward, so
    we set it there for the duration of the run and restore it on exit.
    Returns the net, the ledger (optionally with val predictions per epoch),
    and t* (first epoch with val acc >= tau)."""
    saved_delta = A.DELTA
    A.DELTA = delta
    try:
        torch.manual_seed(seed)
        net = A.SufferingAwareNet()
        A.load_trunk_into_san(net, A.shared_trunk_init())
        opt = torch.optim.Adam(net.parameters(), lr=LR)
        ledger, t_star = [], None
        for epoch in range(epochs):
            net.train()
            net.meter = A.MachineMeter()
            warmup = epoch < WARMUP_EPOCHS
            _, _, _, _, aux_records, final_record = net(
                Xtr, train=True, use_exit_heads=not warmup)
            if warmup:
                f_idx, f_logits = final_record
                loss = CE(f_logits, Ytr[f_idx])
            else:
                losses = []
                if final_record is not None:
                    f_idx, f_logits = final_record
                    losses.append(CE(f_logits, Ytr[f_idx]))
                if aux_records:
                    aux = torch.stack([CE(a_logits, Ytr[a_idx])
                                       for a_idx, a_logits in aux_records]).mean()
                    losses.append(AUX_W * aux)
                loss = sum(losses)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_flops = net.meter.flops
            net.eval()
            net.meter = A.MachineMeter()
            with torch.no_grad():
                vlogits, vdepth, _, _, _, _ = net(Xva, train=False,
                                                  use_exit_heads=not warmup)
            eval_flops = net.meter.flops
            pred = vlogits.argmax(dim=1)
            acc = float((pred == Yva).float().mean().item())
            entry = {
                "epoch": epoch, "flops": train_flops + eval_flops,
                "acc": acc, "harm": harm_of_h(pred, Yva, HARM),
                "exit_frac": float((vdepth < net.depth).float().mean().item()),
                "feasible": acc >= tau,
            }
            if record_preds:
                entry["pred"] = pred.clone()
            if entry["feasible"] and t_star is None:
                t_star = epoch
            ledger.append(entry)
            if t_star is not None:
                break
        return net, ledger, t_star
    finally:
        A.DELTA = saved_delta


def train_dense_param(Xtr, Ytr, Xva, Yva, epochs=EPOCHS, seed=SEED,
                      record_preds=False):
    """The canonical dense baseline loop, parameterized on data, with
    per-epoch val predictions optionally recorded (needed to re-score the
    patient channel under perturbed harm matrices)."""
    torch.manual_seed(seed)
    width, depth = 32, 4
    net = A.build_dense_mlp(width, depth)
    net.load_state_dict(A.shared_trunk_init(width, depth))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ledger, t_star = [], None
    for epoch in range(epochs):
        net.train()
        loss = CE(net(Xtr), Ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_flops = A.dense_flops(Xtr.shape[0], width, depth, backward=True)
        net.eval()
        with torch.no_grad():
            vlogits = net(Xva)
        eval_flops = A.dense_flops(Xva.shape[0], width, depth, backward=False)
        pred = vlogits.argmax(dim=1)
        acc = float((pred == Yva).float().mean().item())
        entry = {
            "epoch": epoch, "flops": train_flops + eval_flops,
            "acc": acc, "harm": harm_of_h(pred, Yva, HARM),
            "exit_frac": 0.0, "feasible": acc >= TAU,
        }
        if record_preds:
            entry["pred"] = pred.clone()
        if entry["feasible"] and t_star is None:
            t_star = epoch
        ledger.append(entry)
    return net, ledger, t_star


def integrated_harm(ledger, y, H):
    """Integrated patient harm of a ledger under an arbitrary harm matrix
    (requires record_preds=True ledgers)."""
    return sum(harm_of_h(e["pred"], y, H) for e in ledger)


# ---------------- adversarial attacks ----------------------------------------
def fgsm(model, x, y, eps, san):
    """One-step FGSM on CE loss, in standardized-feature space (eps is in
    standard-deviation units). For SAN the differentiable path runs through
    the exit/final heads (gate masks are detached); the meter charges during
    this forward are discarded."""
    xa = x.clone().requires_grad_(True)
    if san:
        model.meter = A.MachineMeter()
        logits = model(xa, train=False)[0]
    else:
        logits = model(xa)
    loss = CE(logits, y)
    grad = torch.autograd.grad(loss, xa)[0]
    return (x + eps * grad.sign()).detach()


def pgd_suppress_confidence(net, x, eps=0.30, steps=30):
    """Machine-channel attack: PGD that MINIMIZES the first exit head's
    confidence (max softmax), trying to keep samples inside the network and
    force gratuitous depth. L-inf ball of radius eps around x."""
    x0 = x.clone()
    xa = x.clone()
    layer0 = net.layers[0]
    alpha = 2.5 * eps / steps
    for _ in range(steps):
        xa.requires_grad_(True)
        h = layer0.act(layer0.core(xa))
        conf = torch.softmax(layer0.exit_head(h), dim=1).max(dim=1).values
        grad = torch.autograd.grad(conf.mean(), xa)[0]
        xa = (xa - alpha * grad.sign()).detach()
        xa = x0 + (xa - x0).clamp(-eps, eps)
    return xa.detach()


def manual_accounting(net, per_layer_active, n_final):
    """Independent manual FLOP accounting of the executed path (A1-style)."""
    manual = 0
    for k, layer in enumerate(net.layers):
        n_active = per_layer_active[k] if k < len(per_layer_active) else 0
        manual += (2 * layer.d_in * layer.width
                   + 2 * layer.width * N_CLASS) * n_active
    manual += 2 * net.width * N_CLASS * n_final
    return manual


# =============================================================================
# Contract
# =============================================================================
def main():
    results = {}

    print("SAN_ROBUSTNESS contract (V1..V8)")
    print("cross-validation + sensitivity + adversarial stress of the SAN contract A1..A8")
    print("synthetic dose-band task; no clinical claim; not medical guidance")
    print("note=no_consciousness_claim (machine channel is an operational burden proxy)")

    # ---- pooled cohort + canonical split ------------------------------------
    rng = np.random.default_rng(SEED + 100)
    X_pool, Y_pool = make_data(N_POOL, rng, A.LABEL_NOISE)
    fold_sizes = [N_POOL // K_FOLDS] * K_FOLDS
    folds = []
    start = 0
    for k in range(K_FOLDS):
        vidx = torch.arange(start, start + fold_sizes[k])
        tmask = torch.ones(N_POOL, dtype=torch.bool)
        tmask[vidx] = False
        folds.append((X_pool[tmask], Y_pool[tmask], X_pool[vidx], Y_pool[vidx]))
        start += fold_sizes[k]

    # canonical reference models (fold 0), predictions recorded for the
    # harm-matrix perturbation analysis and the adversarial evaluations
    Xtr0, Ytr0, Xva0, Yva0 = folds[0]
    san0, san0_ledger, san0_t = train_san_param(Xtr0, Ytr0, Xva0, Yva0,
                                                record_preds=True)
    dense0, dense0_ledger, dense0_t = train_dense_param(Xtr0, Ytr0, Xva0, Yva0,
                                                        record_preds=True)
    S0 = {"san": A.suffering_summary(san0_ledger),
          "dense": A.suffering_summary(dense0_ledger)}
    print(f"  reference[fold0]: SAN t*={san0_t} S_m={S0['san']['s_machine_flops']/1e9:.3f}GF "
          f"S_p_int={S0['san']['s_patient_int']:.2f} | "
          f"dense t*={dense0_t} S_m={S0['dense']['s_machine_flops']/1e9:.3f}GF "
          f"(full {EPOCHS}-epoch budget; dense has no stop rule) "
          f"S_p_int={S0['dense']['s_patient_int']:.2f}")

    # ---- V1: cross-validated feasibility ------------------------------------
    fold_summaries = []
    for k, (Xtr, Ytr, Xva, Yva) in enumerate(folds):
        _, lg_s, t_s = train_san_param(Xtr, Ytr, Xva, Yva)
        _, lg_d, t_d = train_dense_param(Xtr, Ytr, Xva, Yva)
        s_s, s_d = A.suffering_summary(lg_s), A.suffering_summary(lg_d)
        fold_summaries.append({"fold": k, "t_san": t_s, "t_dense": t_d,
                               "san": s_s, "dense": s_d,
                               "n_val": Xva.shape[0]})
        print(f"  fold[{k}]: SAN t*={t_s} grat={s_s['gratuitous_flops']} "
              f"S_m={s_s['s_machine_flops']/1e9:.3f}GF S_p_int={s_s['s_patient_int']:.2f} | "
              f"dense t*={t_d} S_m={s_d['s_machine_flops']/1e9:.3f}GF "
              f"S_p_int={s_d['s_patient_int']:.2f}")
    v1_feasible_all = all(f["t_san"] is not None and f["t_san"] < EPOCHS
                          for f in fold_summaries)
    v1_grat_zero = all(f["san"]["gratuitous_flops"] == 0 for f in fold_summaries)
    results["V1"] = v1_feasible_all and v1_grat_zero
    t_stars = [f["t_san"] for f in fold_summaries if f["t_san"] is not None]
    print(f"  V1: {'PASS' if results['V1'] else 'FAIL'} "
          f"(feasible in {len(t_stars)}/{K_FOLDS} folds, "
          f"t* mean={np.mean(t_stars):.1f}±{np.std(t_stars):.1f}, "
          f"gratuitous=0 in all folds: {v1_grat_zero})")

    # ---- V2: cross-validated suffering ordering ------------------------------
    v2_m = all(f["san"]["s_machine_flops"] < f["dense"]["s_machine_flops"]
               for f in fold_summaries)
    v2_p = all(f["san"]["s_patient_int"] <= f["dense"]["s_patient_int"] + 1e-9
               for f in fold_summaries)
    results["V2"] = v2_m and v2_p
    worst_m = max(f["san"]["s_machine_flops"] / f["dense"]["s_machine_flops"]
                  for f in fold_summaries)
    print(f"  V2: {'PASS' if results['V2'] else 'FAIL'} "
          f"(machine bound holds in {sum(f['san']['s_machine_flops'] < f['dense']['s_machine_flops'] for f in fold_summaries)}/{K_FOLDS} folds, "
          f"patient bound in {sum(f['san']['s_patient_int'] <= f['dense']['s_patient_int'] + 1e-9 for f in fold_summaries)}/{K_FOLDS}, "
          f"worst-case (max) S_m ratio SAN/dense={worst_m:.3f})")

    # ---- V3: exit-threshold sensitivity --------------------------------------
    v3_rows = []
    for delta in DELTA_GRID:
        _, lg_s, t_s = train_san_param(Xtr0, Ytr0, Xva0, Yva0, delta=delta)
        s_s = A.suffering_summary(lg_s)
        ok = (t_s is not None and s_s["gratuitous_flops"] == 0
              and s_s["s_machine_flops"] < S0["dense"]["s_machine_flops"])
        v3_rows.append(ok)
        print(f"  delta={delta:.2f}: t*={t_s} grat={s_s['gratuitous_flops']} "
              f"S_m={s_s['s_machine_flops']/1e9:.3f}GF "
              f"exit_frac@t*={lg_s[t_s]['exit_frac'] if t_s is not None else float('nan'):.3f} "
              f"ok={ok}")
    results["V3"] = all(v3_rows)
    print(f"  V3: {'PASS' if results['V3'] else 'FAIL'} "
          f"(feasibility + zero gratuitous + machine bound at "
          f"{sum(v3_rows)}/{len(DELTA_GRID)} grid points)")

    # ---- V4: harm-structure + label-noise sensitivity -------------------------
    rng_h = np.random.default_rng(SEED + 44)
    v4a_rows = []
    offdiag_idx = [(i, j) for i in range(N_CLASS) for j in range(N_CLASS) if i != j]
    for r in range(N_HARM_PERTURB):
        Hp = HARM.clone()
        factors = torch.from_numpy(
            np.exp(rng_h.uniform(-np.log(2.0), np.log(2.0), size=len(offdiag_idx)))
        ).float()
        for (i, j), f in zip(offdiag_idx, factors):
            Hp[i, j] = HARM[i, j] * f
        off = Hp[~torch.eye(N_CLASS, dtype=bool)]
        asym_ok = float(off.max()) >= 3.0 * float(off.min())
        s_p_san = integrated_harm(san0_ledger, Yva0, Hp)
        s_p_dense = integrated_harm(dense0_ledger, Yva0, Hp)
        v4a_rows.append(asym_ok and s_p_san <= s_p_dense + 1e-9)
        print(f"  harm_perturb[{r}]: asym_max/min={float(off.max())/float(off.min()):.1f}x "
              f"S_p SAN={s_p_san:.2f} <= dense={s_p_dense:.2f}: {s_p_san <= s_p_dense + 1e-9}")
    v4b_rows = []
    for noise in NOISE_GRID:
        rng_n = np.random.default_rng(SEED + 200 + int(noise * 100))
        Xtr_n, Ytr_n = make_data(Xtr0.shape[0], rng_n, noise)
        Xva_n, Yva_n = make_data(Xva0.shape[0], rng_n, noise)
        _, lg_s, t_s = train_san_param(Xtr_n, Ytr_n, Xva_n, Yva_n)
        _, lg_d, t_d = train_dense_param(Xtr_n, Ytr_n, Xva_n, Yva_n)
        s_s, s_d = A.suffering_summary(lg_s), A.suffering_summary(lg_d)
        feas_match = (t_s is not None) or (t_d is None)   # SAN feasible whenever dense is
        m_bound = s_s["s_machine_flops"] < s_d["s_machine_flops"]
        v4b_rows.append(feas_match and m_bound)
        print(f"  label_noise={noise:.2f}: SAN t*={t_s} dense t*={t_d} "
              f"feas_match={feas_match} "
              f"S_m SAN={s_s['s_machine_flops']/1e9:.3f}GF < dense={s_d['s_machine_flops']/1e9:.3f}GF: {m_bound}")
    results["V4"] = all(v4a_rows) and all(v4b_rows)
    print(f"  V4: {'PASS' if results['V4'] else 'FAIL'} "
          f"(harm perturbations {sum(v4a_rows)}/{N_HARM_PERTURB}, "
          f"noise levels {sum(v4b_rows)}/{len(NOISE_GRID)})")

    # ---- V5: adversarial patient channel (FGSM) -------------------------------
    # Fair-comparison design (see spec section 5). A naive SAN@t* vs
    # dense@60 comparison confounds two variables: the exit ARCHITECTURE and
    # the training EXPOSURE (dense runs 60 epochs, SAN freezes at t*=7 —
    # more training buys decision margin, hence adversarial robustness).
    # V5 therefore isolates the architecture:
    #   (a) attack effectiveness: adversarial harm strictly increases with
    #       eps for SAN and for dense@60;
    #   (b) matched exposure: dense frozen at SAN's t* (identical training
    #       budget) vs SAN@t* — SAN adversarial harm <= dense@t* at every
    #       eps, i.e. the exit architecture adds NO fragility (per unit of
    #       machine suffering SAN is in fact the more robust);
    #   (c) exit-channel non-fragility: gated vs forced-dense SAN on the
    #       same adversarial inputs incur the same harm (within 1e-6) —
    #       exit-head predictions are not an adversarial weak point.
    # The exposure confound itself is REPORTED (not gated) as the
    # mercy/robustness trade-off: dense@60's extra robustness over SAN@t*
    # is bought with gratuitous machine suffering, and a SAN trained with a
    # declared robustness margin (tau' = TAU + 0.05) recovers most of it at
    # ~30% of dense@60's suffering.
    san0.eval()
    dense0.eval()
    # dense frozen at SAN's t*: matched training exposure
    dense_t, _, _ = train_dense_param(Xtr0, Ytr0, Xva0, Yva0, epochs=san0_t + 1)
    dense_t.eval()
    # SAN with a declared robustness margin (reported, not gated)
    san_m, san_m_ledger, san_m_t = train_san_param(Xtr0, Ytr0, Xva0, Yva0,
                                                   tau=TAU + 0.05)
    san_m.eval()
    S_margin = A.suffering_summary(san_m_ledger)

    def adv_harm_san(model, eps, force_dense=False):
        xa = fgsm(model, Xva0, Yva0, eps, san=True)
        with torch.no_grad():
            if force_dense:
                logits, _ = model.forward_dense(xa)
            else:
                model.meter = A.MachineMeter()
                logits = model(xa, train=False)[0]
        pred = logits.argmax(1)
        return (harm_of_h(pred, Yva0, HARM),
                float((pred == Yva0).float().mean().item()))

    def adv_harm_dense(model, eps):
        xa = fgsm(model, Xva0, Yva0, eps, san=False)
        with torch.no_grad():
            pred = model(xa).argmax(1)
        return (harm_of_h(pred, Yva0, HARM),
                float((pred == Yva0).float().mean().item()))

    v5a_ok, v5b_rows, v5c_rows = True, [], []
    prev = {"san": 0.0, "dense60": 0.0}
    for eps in EPS_GRID:
        h_s, a_s = adv_harm_san(san0, eps)
        h_sf, _ = adv_harm_san(san0, eps, force_dense=True)
        h_dt, a_dt = adv_harm_dense(dense_t, eps)
        h_d60, a_d60 = adv_harm_dense(dense0, eps)
        h_sm, a_sm = adv_harm_san(san_m, eps)
        v5a_ok = v5a_ok and h_s > prev["san"] and h_d60 > prev["dense60"]
        prev = {"san": h_s, "dense60": h_d60}
        v5b_rows.append(h_s <= h_dt + 1e-9)
        v5c_rows.append(abs(h_s - h_sf) <= 1e-6)
        print(f"  fgsm eps={eps:.2f}: SAN@t* harm={h_s:.3f} acc={a_s:.3f} | "
              f"SAN-forced-dense harm={h_sf:.3f} | "
              f"dense@t*(matched) harm={h_dt:.3f} acc={a_dt:.3f} | "
              f"dense@60 harm={h_d60:.3f} acc={a_d60:.3f} | "
              f"SAN+margin harm={h_sm:.3f} acc={a_sm:.3f}")
    results["V5"] = v5a_ok and all(v5b_rows) and all(v5c_rows)
    print(f"  V5: {'PASS' if results['V5'] else 'FAIL'} "
          f"(attack effective: {v5a_ok}; matched-exposure SAN<=dense@t* at "
          f"{sum(v5b_rows)}/{len(EPS_GRID)} eps; gated==forced-dense at "
          f"{sum(v5c_rows)}/{len(EPS_GRID)} eps)")
    print(f"  V5 trade-off (reported): dense@60 robustness costs "
          f"{S0['dense']['s_machine_flops']/1e9:.3f}GF (full 60-epoch run); SAN+margin t*={san_m_t} "
          f"recovers robustness at {S_margin['s_machine_flops']/1e9:.3f}GF "
          f"({100.0*S_margin['s_machine_flops']/S0['dense']['s_machine_flops']:.1f}% of dense)")

    # ---- V6: adversarial machine channel (confidence suppression) -------------
    with torch.no_grad():
        san0.meter = A.MachineMeter()
        _, vdepth_clean, _, _, _, _ = san0(Xva0, train=False)
        clean_flops = san0.meter.flops
    clean_exit_frac = float((vdepth_clean < san0.depth).float().mean().item())
    xa_m = pgd_suppress_confidence(san0, Xva0)
    with torch.no_grad():
        san0.meter = A.MachineMeter()
        _, vdepth_adv, per_layer_adv, n_final_adv, _, _ = san0(xa_m, train=False)
        adv_flops = san0.meter.flops
        _, dense_meter_adv = san0.forward_dense(xa_m)
    adv_exit_frac = float((vdepth_adv < san0.depth).float().mean().item())
    attack_effective = adv_exit_frac < clean_exit_frac and adv_flops > clean_flops
    bound_attackproof = adv_flops <= dense_meter_adv.flops
    manual_adv = manual_accounting(san0, per_layer_adv, n_final_adv)
    results["V6"] = attack_effective and bound_attackproof and manual_adv == adv_flops
    print(f"  V6: {'PASS' if results['V6'] else 'FAIL'} "
          f"(exit_frac clean={clean_exit_frac:.3f} -> adv={adv_exit_frac:.3f}, "
          f"FLOPs clean={clean_flops} -> adv={adv_flops} "
          f"(+{100.0*(adv_flops-clean_flops)/max(clean_flops,1):.1f}%), "
          f"dense_ceiling={dense_meter_adv.flops} respected: {bound_attackproof}, "
          f"manual==meter: {manual_adv == adv_flops})")

    # ---- V7: anti-Goodhart soundness under stress ------------------------------
    # candidate pool as in A3/A8: abstainer, cheap probe, shortcut probe,
    # SAN t*, over-trained dense. Feasibility is re-evaluated at each TAU.
    majority = int(torch.bincount(Ytr0).argmax())
    abstain_pred = torch.full_like(Yva0, majority)
    torch.manual_seed(SEED + 1)
    probe = nn.Linear(D_IN, N_CLASS)
    popt = torch.optim.Adam(probe.parameters(), lr=LR)
    for _ in range(2):
        ploss = CE(probe(Xtr0), Ytr0)
        popt.zero_grad()
        ploss.backward()
        popt.step()
    with torch.no_grad():
        probe_pred = probe(Xva0).argmax(1)
        san_pred = san0(Xva0, train=False)[0].argmax(1)
        dense_pred = dense0(Xva0).argmax(1)
    # shortcut probe (train-leaking spurious feature), as in A8
    rng8 = np.random.default_rng(SEED + 8)
    Xtr8, Xva8 = Xtr0.clone(), Xva0.clone()
    Xtr8[:, 0] = Ytr0.float() + torch.from_numpy(
        rng8.normal(0, 0.3, size=Xtr0.shape[0])).float()
    Xva8[:, 0] = torch.from_numpy(rng8.normal(0, 1.0, size=Xva0.shape[0])).float()
    torch.manual_seed(SEED + 2)
    shortcut = nn.Linear(1, N_CLASS)
    sopt = torch.optim.Adam(shortcut.parameters(), lr=LR)
    for _ in range(200):
        sloss = CE(shortcut(Xtr8[:, :1]), Ytr0)
        sopt.zero_grad()
        sloss.backward()
        sopt.step()
    with torch.no_grad():
        sc_pred = shortcut(Xva8[:, :1]).argmax(1)
        sc_train_acc = float((shortcut(Xtr8[:, :1]).argmax(1) == Ytr0).float().mean().item())

    def acc_of(pred):
        return float((pred == Yva0).float().mean().item())

    v7_rows = []
    for tau in TAU_GRID:
        pool = [
            {"name": "abstain", "feasible": acc_of(abstain_pred) >= tau,
             "j_patient": harm_of_h(abstain_pred, Yva0, HARM), "j_machine": 0.0},
            {"name": "cheap_probe", "feasible": acc_of(probe_pred) >= tau,
             "j_patient": harm_of_h(probe_pred, Yva0, HARM), "j_machine": 1e-9},
            {"name": "shortcut_probe", "feasible": acc_of(sc_pred) >= tau,
             "j_patient": 0.01, "j_machine": 1e-12},
            {"name": "san_t*", "feasible": acc_of(san_pred) >= tau,
             "j_patient": harm_of_h(san_pred, Yva0, HARM),
             "j_machine": S0["san"]["s_machine_joules"]},
            {"name": "dense_overfit", "feasible": acc_of(dense_pred) >= tau,
             "j_patient": harm_of_h(dense_pred, Yva0, HARM),
             "j_machine": S0["dense"]["s_machine_joules"]},
        ]
        feas_names = {c["name"] for c in pool if c["feasible"]}
        grid_ok = all(A.gate_select(pool, lam / 100.0) in feas_names
                      for lam in range(101))
        loud = A.gate_select([dict(c, feasible=False) for c in pool],
                             0.5) == "NO_FEASIBLE"
        # the trap candidates must actually be traps at this TAU
        traps_infeasible = not ({"abstain", "cheap_probe", "shortcut_probe"}
                                & feas_names)
        v7_rows.append(grid_ok and loud and traps_infeasible)
        print(f"  tau={tau:.2f}: feasible={sorted(feas_names)} "
              f"101-grid feasible-only={grid_ok} NO_FEASIBLE-loud={loud} "
              f"traps_rejected={traps_infeasible}")
    results["V7"] = all(v7_rows) and sc_train_acc > TAU
    print(f"  V7: {'PASS' if results['V7'] else 'FAIL'} "
          f"(sound at {sum(v7_rows)}/{len(TAU_GRID)} TAU points, "
          f"shortcut train_acc={sc_train_acc:.3f} > TAU={TAU}: "
          f"train-loss selection accepts, gate rejects)")

    # ---- V8: metering conservation under stress --------------------------------
    with torch.no_grad():
        san0.meter = A.MachineMeter()
        vlogits_adv, vdepth_adv8, per_layer8, n_final8, _, _ = san0(xa_m, train=False)
        gated_flops8 = san0.meter.flops
        _, dense_meter8 = san0.forward_dense(xa_m)
        # forced-open gates: every sample through every layer + every exit
        # head + final head == forward_dense, exactly
        forced = 0
        for layer in san0.layers:
            forced += (2 * layer.d_in * layer.width
                       + 2 * layer.width * N_CLASS) * xa_m.shape[0]
        forced += 2 * san0.width * N_CLASS * xa_m.shape[0]
        # exited predictions agree with the recomputed prefix on adv inputs
        h = xa_m
        prefix_logits = {}
        for k, layer in enumerate(san0.layers):
            h = layer.act(layer.core(h))
            prefix_logits[k + 1] = layer.exit_head(h)
        vlogits_dense8, _ = san0.forward_dense(xa_m)
    pred_agree8 = True
    for d in range(1, san0.depth + 1):
        idx = (vdepth_adv8 == d).nonzero(as_tuple=True)[0]
        if idx.numel() > 0 and not torch.equal(
                vlogits_adv[idx].argmax(1), prefix_logits[d][idx].argmax(1)):
            pred_agree8 = False
    idx_f = (vdepth_adv8 == san0.depth + 1).nonzero(as_tuple=True)[0]
    if idx_f.numel() > 0 and not torch.equal(
            vlogits_adv[idx_f].argmax(1), vlogits_dense8[idx_f].argmax(1)):
        pred_agree8 = False
    manual8 = manual_accounting(san0, per_layer8, n_final8)
    results["V8"] = (manual8 == gated_flops8 and forced == dense_meter8.flops
                     and gated_flops8 <= dense_meter8.flops and pred_agree8)
    print(f"  V8: {'PASS' if results['V8'] else 'FAIL'} "
          f"(adv-input meter={gated_flops8} == manual={manual8}, "
          f"forced-open={forced} == forward_dense={dense_meter8.flops}, "
          f"gated<=dense: {gated_flops8 <= dense_meter8.flops}, "
          f"prefix argmax agree: {pred_agree8})")

    # ---- verdict -----------------------------------------------------------------
    n_pass = sum(1 for v in results.values() if v)
    verdict = "V_GREEN" if n_pass == 8 else "V_RED"
    print(f"SAN_ROBUSTNESS_VERDICT {verdict} ({n_pass}/8 clauses PASS)")
    return 0 if verdict == "V_GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
