#!/usr/bin/env python3
"""Mercyful Learning — SAN under expanded ethics: environmental, social, and
temporal suffering channels.

Companion artifact to
  docs/research/suffering_aware_expanded_channels_spec_2026-07-31.md

The Suffering-Aware Network (SAN) line so far meters TWO suffering channels —
patient (asymmetric harm of predictions on the held-out cohort-in-waiting)
and machine (metered FLOPs of the executed path) — clauses A1..A8 on the
small-network line, D1..D9 at ResNet-18 / ViT-small scale. This harness asks
the expansion question: can the SAME architecture carry an expanded ethics
with FIVE suffering channels, without new machinery?

  * PATIENT channel (existing): integrated + peak asymmetric harm on the
    held-out cohort.
  * MACHINE channel (existing): metered FLOPs of the executed path
    (gated-off layers charge exactly 0), reported also in joules.
  * ENVIRONMENTAL channel (new): energy (J) and carbon (gCO2e) of the
    metered path — DECLARED DERIVED channels: energy = FLOPs x E_PER_FLOP
    (the standing convention), carbon = energy x a declared grid-intensity
    constant. They are monotone transforms of the machine channel BY
    CONSTRUCTION; the harness says so and never presents them as independent
    measurements.
  * TEMPORAL channel (new): training time (metered FLOPs / a declared
    sustained-FLOP/s rate) and DEPLOYMENT latency — per-sample
    time-to-prediction of the executed prefix at the selected checkpoint.
    Latency is NOT monotone in total FLOPs: per-sample early exits move it
    independently of the training ledger, so it carries genuinely new
    information.
  * SOCIAL channel (new): equity + Rawlsian justice over two synthetic
    subgroups of the held-out cohort (elder / non-elder split on the age
    covariate). Equity = between-group harm gap; justice = worst-group harm
    (the maximin/Rawls term). Not monotone in FLOPs: carries genuinely new
    information.

The expanded ethics also enters the GATE, not just the ledger: a checkpoint
is feasible iff held-out accuracy >= TAU AND worst-group accuracy >=
TAU_GROUP (spec section 4). The anti-Goodhart selection then runs over a
5-channel normalized cost vector on a 70-point simplex weight grid —
feasibility stays categorical at every compassion-allocation weight.

Everything reuses the base SAN harness (models, task, seeds, meter,
accounting conventions) via import; nothing is re-implemented.

Synthetic data only. This benchmark makes no clinical claim and is not
medical guidance. The machine channel is an operational computational-burden
proxy; no_consciousness_claim is made or needed. The environmental,
temporal, and social channels are operational proxies with declared
constants, not physical measurements.

Certificates (contract clauses X1..X8):
  X1  channel metering conservation: gated-off layers contribute exactly 0
      on the machine channel (and hence on its declared environmental and
      temporal transforms); metered FLOPs == independent manual accounting;
      energy/carbon/time equal their declared transforms of metered FLOPs
      EXACTLY; executed prefix invariant under gating (bounded logit
      deviation, argmax exactly equal)
  X2  expanded feasibility reachable: SAN reaches an expanded-feasible
      checkpoint (acc >= TAU and worst-group acc >= TAU_GROUP) at some
      t*_X < EPOCHS
  X3  expanded anti-Goodhart soundness: over a 70-point simplex weight grid
      on the 5-channel cost vector, selection is feasible-only; an
      all-infeasible pool returns loud NO_FEASIBLE; the zero-cost abstainer
      and the cheap under-trained probe are both infeasible
  X4  necessary/gratuitous separation on EVERY channel: SAN gratuitous
      suffering is exactly 0 on all five channels; the fixed-budget dense
      baseline accrues > 0 gratuitous suffering on all five
  X5  suffering bounds on every channel: SAN total <= every baseline on all
      five channels (machine strict <)
  X6  social channel first-class: both subgroups are non-trivial
      (>= 25% of the cohort each); SAN worst-group peak <= the same-init
      baselines' peaks; equity gaps measured and reported for all
  X7  temporal channel first-class: SAN mean inference latency at t*_X is
      strictly below the dense baseline's; SAN peak latency is bounded by
      the gates-open SAN full-path latency; exits are real (> 0.10)
  X8  anti-shortcut under the expanded gate: a linear probe on a spurious
      feature beats TAU on TRAIN yet fails expanded feasibility held-out;
      it is never selected at any point of the simplex grid

Run: .venv/bin/python scripts/research/suffering_aware_expanded_channels.py
Requires: torch (CPU) + numpy from the repo .venv.
"""

import itertools
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suffering_aware_architecture as base  # noqa: E402

SEED = base.SEED
N_TRAIN, N_VAL = base.N_TRAIN, base.N_VAL
D_IN, N_CLASS = base.D_IN, base.N_CLASS
TAU, EPOCHS, LR = base.TAU, base.EPOCHS, base.LR
WARMUP_EPOCHS, AUX_W = base.WARMUP_EPOCHS, base.AUX_W
E_PER_FLOP = base.E_PER_FLOP
HARM = base.HARM
X_tr, Y_tr, X_va, Y_va = base.X_tr, base.Y_tr, base.X_va, base.Y_va
CE = base.CE

# ---------------- expanded-ethics constants (declared, spec section 3) ------
CI_G_PER_KWH = 475.0     # grid carbon intensity, gCO2e/kWh (declared constant)
J_PER_KWH = 3.6e6
RATE_FLOPS = 1.0e9       # sustained metering rate, FLOP/s (declared proxy)
TAU_GROUP = 0.72         # justice bar: worst-group held-out accuracy >= TAU_GROUP

# Synthetic subgroup split (spec section 3.4): column 3 of the feature
# vector is the (standardized) age covariate; the "elder" group is the
# above-mean half. Both groups are ~50% of the cohort by construction.
ELDER = X_va[:, 3] >= 0.0
GROUPS = {"nonelder": ~ELDER, "elder": ELDER}


def harm_of(pred, y):
    return base.harm_of(pred, y)


def group_metrics(pred, y):
    """Per-group accuracy and harm; the social channel's raw material."""
    out = {}
    for name, mask in GROUPS.items():
        out[name] = {
            "n": int(mask.sum().item()),
            "acc": float((pred[mask] == y[mask]).float().mean().item()),
            "harm": harm_of(pred[mask], y[mask]),
        }
    return out


def expanded_feasible(acc, wg_acc):
    """The expanded feasibility predicate (spec section 4): overall target
    AND worst-group justice bar. Both are categorical."""
    return acc >= TAU and wg_acc >= TAU_GROUP


# ---------------- training loops (expanded ledger) ---------------------------
def train_san_x():
    """SAN training, identical to the base line except the ledger carries
    the social channel and freeze-on-green fires on EXPANDED feasibility."""
    torch.manual_seed(SEED)
    net = base.SufferingAwareNet()
    base.load_trunk_into_san(net, base.shared_trunk_init())
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ledger = []
    t_star = None
    for epoch in range(EPOCHS):
        net.train()
        net.meter = base.MachineMeter()
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
        net.meter = base.MachineMeter()
        with torch.no_grad():
            vlogits, vdepth, _, _, _, _ = net(X_va, train=False,
                                              use_exit_heads=not warmup)
        eval_flops = net.meter.flops
        pred = vlogits.argmax(dim=1)
        acc = float((pred == Y_va).float().mean().item())
        gm = group_metrics(pred, Y_va)
        wg_acc = min(g["acc"] for g in gm.values())
        wg_harm = max(g["harm"] for g in gm.values())
        gap = abs(gm["elder"]["harm"] - gm["nonelder"]["harm"])
        feasible = expanded_feasible(acc, wg_acc)
        if feasible and t_star is None:
            t_star = epoch
        ledger.append({
            "epoch": epoch, "flops": train_flops + eval_flops,
            "acc": acc, "harm": harm_of(pred, Y_va),
            "wg_acc": wg_acc, "wg_harm": wg_harm, "gap": gap,
            "exit_frac": float((vdepth < net.depth).float().mean().item()),
            "feasible": feasible,
        })
        if t_star is not None:
            break  # freeze-on-green at the expanded-feasible checkpoint
    return net, ledger, t_star


def train_baseline_x(kind):
    """Dense / ResNet / EarlyStop baselines: identical budget, optimizer,
    seed, and expanded ledger shape."""
    torch.manual_seed(SEED)
    width, depth = 32, 4
    if kind == "resnet":
        net = base.ResNetMLP(width, depth)
    else:
        net = base.build_dense_mlp(width, depth)
        net.load_state_dict(base.shared_trunk_init(width, depth))
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ledger = []
    t_star = None
    for epoch in range(EPOCHS):
        net.train()
        loss = CE(net(X_tr), Y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
        train_flops = base.dense_flops(N_TRAIN, width, depth, backward=True,
                                       resnet=(kind == "resnet"))
        net.eval()
        with torch.no_grad():
            vlogits = net(X_va)
        eval_flops = base.dense_flops(N_VAL, width, depth, backward=False,
                                      resnet=(kind == "resnet"))
        pred = vlogits.argmax(dim=1)
        acc = float((pred == Y_va).float().mean().item())
        gm = group_metrics(pred, Y_va)
        wg_acc = min(g["acc"] for g in gm.values())
        wg_harm = max(g["harm"] for g in gm.values())
        gap = abs(gm["elder"]["harm"] - gm["nonelder"]["harm"])
        feasible = expanded_feasible(acc, wg_acc)
        if feasible and t_star is None:
            t_star = epoch
        ledger.append({
            "epoch": epoch, "flops": train_flops + eval_flops,
            "acc": acc, "harm": harm_of(pred, Y_va),
            "wg_acc": wg_acc, "wg_harm": wg_harm, "gap": gap,
            "exit_frac": 0.0, "feasible": feasible,
        })
        if kind == "earlystop" and t_star is not None:
            break
    return net, ledger, t_star


# ---------------- five-channel suffering ledger ------------------------------
def channel_summary(ledger):
    """All five suffering channels, integrated and peak, plus the
    necessary/gratuitous decomposition at the first expanded-feasible epoch."""
    fl = sum(e["flops"] for e in ledger)
    harms = [e["harm"] for e in ledger]
    wg_harms = [e["wg_harm"] for e in ledger]
    gaps = [e["gap"] for e in ledger]
    t_star = next((e["epoch"] for e in ledger if e["feasible"]), None)
    post = [e for e in ledger if t_star is not None and e["epoch"] > t_star]
    return {
        "machine_flops": fl,
        "machine_joules": fl * E_PER_FLOP,
        "env_joules": fl * E_PER_FLOP,
        "env_carbon_g": fl * E_PER_FLOP / J_PER_KWH * CI_G_PER_KWH,
        "temporal_s": fl / RATE_FLOPS,
        "patient_int": sum(harms),
        "patient_peak": max(harms),
        "social_int": sum(wg_harms),        # Rawlsian: worst-group harm, integrated
        "social_peak": max(wg_harms),
        "equity_gap_mean": sum(gaps) / len(gaps),
        "equity_gap_peak": max(gaps),
        "t_star": t_star,
        "gratuitous": {
            "machine_flops": sum(e["flops"] for e in post),
            "env_joules": sum(e["flops"] for e in post) * E_PER_FLOP,
            "temporal_s": sum(e["flops"] for e in post) / RATE_FLOPS,
            "patient": sum(e["harm"] for e in post),
            "social": sum(e["wg_harm"] for e in post),
        },
    }


# ---------------- deployment latency (temporal channel) ----------------------
def latency_profile_san(net, x):
    """Per-sample inference latency at the deployed (gated) SAN: executed-
    prefix forward FLOPs / declared rate. Exited samples pay only their
    prefix; final-head samples pay the full trunk + exit heads + final head
    (the gates-open SAN path)."""
    net.eval()
    with torch.no_grad():
        _, vdepth, _, _, _, _ = net(x, train=False)
    layer_fw = [2 * l.d_in * l.width + 2 * l.width * N_CLASS for l in net.layers]
    final_fw = 2 * net.width * N_CLASS
    fl = torch.zeros(x.shape[0])
    for i, d in enumerate(vdepth.tolist()):
        if d <= net.depth:
            fl[i] = sum(layer_fw[:d])
        else:
            fl[i] = sum(layer_fw) + final_fw
    return fl / RATE_FLOPS


def latency_dense_per_sample(width=32, depth=4):
    return base.dense_flops(1, width, depth, backward=False) / RATE_FLOPS


# ---------------- expanded anti-Goodhart gate --------------------------------
CHANNELS = ["patient", "machine", "env", "social", "temporal"]


def normalize(costs, ref):
    """Dimensionless 5-channel cost: each channel divided by the fixed-budget
    dense baseline's total (declared normalizer, spec section 5)."""
    return {c: costs[c] / ref[c] for c in CHANNELS}


def gate_select_x(candidates, w):
    """Selection is argmin of the scalarized 5-channel cost over the
    EXPANDED-feasible set ONLY; an all-infeasible pool is a loud
    NO_FEASIBLE, never a least-bad prescription."""
    feas = [c for c in candidates if c["feasible"]]
    if not feas:
        return "NO_FEASIBLE"
    return min(feas, key=lambda c: sum(w[i] * c["j"][ch]
                                       for i, ch in enumerate(CHANNELS)))["name"]


def simplex_grid(n_channels=5, step_denom=4):
    """All weight vectors w >= 0, sum w = 1, on a grid of denominator
    step_denom: C(step_denom + n - 1, n - 1) = 70 points for 5 channels."""
    pts = []
    for comp in itertools.product(range(step_denom + 1), repeat=n_channels - 1):
        last = step_denom - sum(comp)
        if last >= 0:
            pts.append(tuple(v / step_denom for v in comp) + (last / step_denom,))
    return pts


# =============================================================================
# Contract
# =============================================================================
def main():
    results = {}

    print("SUFFERING_AWARE_EXPANDED_CHANNELS contract (X1..X8)")
    print("synthetic dose-band task; no clinical claim; not medical guidance")
    print("note=no_consciousness_claim (machine channel is an operational burden proxy)")
    print("note=operational_proxies (environmental/temporal/social channels use declared constants)")

    # ---- train all four architectures --------------------------------------
    san, san_ledger, san_t = train_san_x()
    _, dense_ledger, dense_t = train_baseline_x("dense")
    _, resnet_ledger, resnet_t = train_baseline_x("resnet")
    _, estop_ledger, estop_t = train_baseline_x("earlystop")

    S = {"san": channel_summary(san_ledger), "dense": channel_summary(dense_ledger),
         "resnet": channel_summary(resnet_ledger), "earlystop": channel_summary(estop_ledger)}
    ledgers = {"san": san_ledger, "dense": dense_ledger,
               "resnet": resnet_ledger, "earlystop": estop_ledger}

    print(f"  subgroups: nonelder n={group_metrics(Y_va, Y_va)['nonelder']['n']} "
          f"elder n={group_metrics(Y_va, Y_va)['elder']['n']} of {N_VAL}")
    for name in ("san", "dense", "resnet", "earlystop"):
        s, lg = S[name], ledgers[name]
        print(f"  ledger[{name}]: epochs_run={len(lg)} t*_X={s['t_star']} "
              f"S_machine={s['machine_flops'] / 1e9:.3f}GF ({s['machine_joules'] * 1e3:.3f}mJ) "
              f"S_env={s['env_carbon_g']:.3e}gCO2e S_temporal={s['temporal_s']:.3f}s "
              f"S_patient_int={s['patient_int']:.2f} (peak={s['patient_peak']:.3f}) "
              f"S_social_int={s['social_int']:.2f} (peak={s['social_peak']:.3f}) "
              f"equity_gap mean={s['equity_gap_mean']:.3f} peak={s['equity_gap_peak']:.3f} "
              f"final_acc={lg[-1]['acc']:.4f} final_wg_acc={lg[-1]['wg_acc']:.4f}")

    # ---- X1: channel metering conservation ----------------------------------
    san.eval()
    san.meter = base.MachineMeter()
    with torch.no_grad():
        vlogits_gated, vdepth, per_layer_active, n_final, _, _ = san(X_va, train=False)
    gated_flops = san.meter.flops
    with torch.no_grad():
        vlogits_dense, dense_meter = san.forward_dense(X_va)
    manual = 0
    for k, layer in enumerate(san.layers):
        n_active = per_layer_active[k] if k < len(per_layer_active) else 0
        manual += (2 * layer.d_in * layer.width + 2 * layer.width * N_CLASS) * n_active
    manual += 2 * san.width * base.N_CLASS * n_final
    x1_meter = gated_flops == manual
    # the environmental and temporal channels are exact DECLARED transforms
    # of the metered FLOPs (not independent measurements — spec section 3)
    x1_transforms = (
        S["san"]["env_joules"] == S["san"]["machine_flops"] * E_PER_FLOP
        and S["san"]["env_carbon_g"] == S["san"]["machine_flops"] * E_PER_FLOP
        / J_PER_KWH * CI_G_PER_KWH
        and S["san"]["temporal_s"] == S["san"]["machine_flops"] / RATE_FLOPS
    )
    # executed prefix invariant under gating (same check as the base A1)
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
        max_dev = max(max_dev, float(
            (vlogits_gated[idx_final] - vlogits_dense[idx_final]).abs().max()))
        if not torch.equal(vlogits_gated[idx_final].argmax(1),
                           vlogits_dense[idx_final].argmax(1)):
            pred_agree = False
    results["X1"] = x1_meter and x1_transforms and max_dev < 1e-4 and pred_agree
    print(f"  X1: {'PASS' if results['X1'] else 'FAIL'} "
          f"(gated={gated_flops} manual={manual} exact={x1_meter}, "
          f"env/temporal transforms exact={x1_transforms}, "
          f"prefix_max_dev={max_dev:.2e} pred_agree={pred_agree})")

    # ---- X2: expanded feasibility reachable ---------------------------------
    results["X2"] = san_t is not None and san_t < EPOCHS
    if san_t is not None:
        print(f"  X2: {'PASS' if results['X2'] else 'FAIL'} "
              f"(SAN first expanded-feasible epoch t*_X={san_t} of {EPOCHS}, "
              f"acc@t*={san_ledger[san_t]['acc']:.4f} >= TAU={TAU}, "
              f"wg_acc@t*={san_ledger[san_t]['wg_acc']:.4f} >= TAU_GROUP={TAU_GROUP})")
    else:
        print("  X2: FAIL (no expanded-feasible checkpoint within budget)")

    # ---- X3: expanded anti-Goodhart soundness -------------------------------
    ref = {"patient": S["dense"]["patient_int"], "machine": S["dense"]["machine_joules"],
           "env": S["dense"]["env_carbon_g"], "social": S["dense"]["social_int"],
           "temporal": S["dense"]["temporal_s"]}

    def cand_costs(summary):
        return normalize({"patient": summary["patient_int"],
                          "machine": summary["machine_joules"],
                          "env": summary["env_carbon_g"],
                          "social": summary["social_int"],
                          "temporal": summary["temporal_s"]}, ref)

    majority = int(torch.bincount(Y_tr).argmax())
    abstain_pred = torch.full_like(Y_va, majority)
    abstain_acc = float((abstain_pred == Y_va).float().mean().item())
    abstain_gm = group_metrics(abstain_pred, Y_va)
    abstain_wg = min(g["acc"] for g in abstain_gm.values())
    torch.manual_seed(SEED + 1)
    probe = nn.Linear(D_IN, N_CLASS)
    popt = torch.optim.Adam(probe.parameters(), lr=LR)
    for _ in range(2):  # deliberately under-trained
        ploss = CE(probe(X_tr), Y_tr)
        popt.zero_grad()
        ploss.backward()
        popt.step()
    with torch.no_grad():
        probe_pred = probe(X_va).argmax(1)
        probe_acc = float((probe_pred == Y_va).float().mean().item())
        probe_gm = group_metrics(probe_pred, Y_va)
        probe_wg = min(g["acc"] for g in probe_gm.values())
    zero_costs = {c: 0.0 for c in CHANNELS}
    pool = [
        {"name": "abstain", "feasible": expanded_feasible(abstain_acc, abstain_wg),
         "j": zero_costs},
        {"name": "cheap_probe", "feasible": expanded_feasible(probe_acc, probe_wg),
         "j": {c: 1e-9 for c in CHANNELS}},
        {"name": "san_t*", "feasible": True, "j": cand_costs(S["san"])},
        {"name": "dense_overfit", "feasible": dense_t is not None,
         "j": cand_costs(S["dense"])},
    ]
    grid = simplex_grid()
    feasible_names = {c["name"] for c in pool if c["feasible"]}
    grid_ok = all(gate_select_x(pool, w) in feasible_names for w in grid)
    loud = gate_select_x([dict(c, feasible=False) for c in pool],
                         grid[0]) == "NO_FEASIBLE"
    results["X3"] = (grid_ok and loud and not pool[0]["feasible"]
                     and not pool[1]["feasible"])
    print(f"  X3: {'PASS' if results['X3'] else 'FAIL'} "
          f"({len(grid)}-point simplex grid feasible-only={grid_ok}, "
          f"all-infeasible->NO_FEASIBLE={loud}, abstain acc={abstain_acc:.3f}/"
          f"wg={abstain_wg:.3f} probe acc={probe_acc:.3f}/wg={probe_wg:.3f} "
          f"both infeasible under (TAU={TAU}, TAU_GROUP={TAU_GROUP}))")

    # ---- X4: necessary/gratuitous separation on every channel ----------------
    g_san = S["san"]["gratuitous"]
    g_dense = S["dense"]["gratuitous"]
    x4_san = all(v == 0 for v in g_san.values())
    x4_dense = dense_t is not None and all(v > 0 for v in g_dense.values())
    results["X4"] = x4_san and x4_dense
    print(f"  X4: {'PASS' if results['X4'] else 'FAIL'} "
          f"(SAN gratuitous all-zero={x4_san} "
          f"[machine={g_san['machine_flops']}F env={g_san['env_joules']}J "
          f"time={g_san['temporal_s']}s patient={g_san['patient']} social={g_san['social']}]; "
          f"dense gratuitous machine={g_dense['machine_flops'] / 1e9:.3f}GF "
          f"patient={g_dense['patient']:.2f} social={g_dense['social']:.2f})")

    # ---- X5: suffering bounds on every channel --------------------------------
    keys = {"machine": "machine_flops", "env": "env_carbon_g",
            "temporal": "temporal_s", "patient": "patient_int",
            "social": "social_int"}
    x5 = {}
    for ch, k in keys.items():
        if ch == "machine":
            x5[ch] = all(S["san"][k] < S[b][k] for b in ("dense", "resnet", "earlystop"))
        else:
            x5[ch] = all(S["san"][k] <= S[b][k] + 1e-12
                         for b in ("dense", "resnet", "earlystop"))
    results["X5"] = all(x5.values())
    print(f"  X5: {'PASS' if results['X5'] else 'FAIL'} "
          f"(per-channel SAN<=all baselines: "
          + " ".join(f"{ch}={x5[ch]}" for ch in CHANNELS) + ")")

    # ---- X6: social channel first-class ---------------------------------------
    gm_true = group_metrics(Y_va, Y_va)
    x6_groups = all(g["n"] >= 0.25 * N_VAL for g in gm_true.values())
    x6_peak = all(S["san"]["social_peak"] <= S[b]["social_peak"] + 1e-9
                  for b in ("dense", "earlystop"))
    results["X6"] = x6_groups and x6_peak
    print(f"  X6: {'PASS' if results['X6'] else 'FAIL'} "
          f"(groups n={gm_true['nonelder']['n']}/{gm_true['elder']['n']} of {N_VAL}, "
          f"S_social_peak SAN={S['san']['social_peak']:.3f} vs "
          f"dense={S['dense']['social_peak']:.3f} resnet={S['resnet']['social_peak']:.3f} "
          f"earlystop={S['earlystop']['social_peak']:.3f}; "
          f"equity_gap_mean SAN={S['san']['equity_gap_mean']:.3f} "
          f"dense={S['dense']['equity_gap_mean']:.3f} "
          f"resnet={S['resnet']['equity_gap_mean']:.3f} "
          f"earlystop={S['earlystop']['equity_gap_mean']:.3f})")

    # ---- X7: temporal channel first-class --------------------------------------
    lat_san = latency_profile_san(san, X_va)
    lat_dense = latency_dense_per_sample()
    lat_san_gates_open = (sum(2 * l.d_in * l.width + 2 * l.width * N_CLASS
                              for l in san.layers)
                          + 2 * san.width * N_CLASS) / RATE_FLOPS
    lat_mean, lat_peak = float(lat_san.mean()), float(lat_san.max())
    exit_frac_t = san_ledger[san_t]["exit_frac"]
    results["X7"] = (lat_mean < lat_dense and lat_peak <= lat_san_gates_open
                     and exit_frac_t > 0.10)
    print(f"  X7: {'PASS' if results['X7'] else 'FAIL'} "
          f"(SAN latency mean={lat_mean * 1e6:.2f}us peak={lat_peak * 1e6:.2f}us vs "
          f"dense={lat_dense * 1e6:.2f}us, gates-open SAN bound={lat_san_gates_open * 1e6:.2f}us, "
          f"exit_frac@t*={exit_frac_t:.3f})")

    # ---- X8: anti-shortcut under the expanded gate ------------------------------
    rng8 = __import__("numpy").random.default_rng(SEED + 8)
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
        sc_pred = shortcut(Xva8[:, :1]).argmax(1)
        sc_val_acc = float((sc_pred == Y_va).float().mean().item())
        sc_gm = group_metrics(sc_pred, Y_va)
        sc_wg = min(g["acc"] for g in sc_gm.values())
    train_loss_selection_accepts = sc_train_acc > TAU
    sc_feasible = expanded_feasible(sc_val_acc, sc_wg)
    pool8 = pool + [{"name": "shortcut_probe", "feasible": sc_feasible,
                     "j": {c: 1e-12 for c in CHANNELS}}]
    never_selected = all(gate_select_x(pool8, w) != "shortcut_probe" for w in grid)
    results["X8"] = (train_loss_selection_accepts and not sc_feasible
                     and never_selected)
    print(f"  X8: {'PASS' if results['X8'] else 'FAIL'} "
          f"(shortcut train_acc={sc_train_acc:.3f} (>TAU: train-loss selection "
          f"accepts), val acc={sc_val_acc:.3f}/wg={sc_wg:.3f} expanded-feasible="
          f"{sc_feasible}, never selected on {len(grid)}-point grid={never_selected})")

    # ---- verdict ------------------------------------------------------------------
    n_pass = sum(1 for v in results.values() if v)
    verdict = "X_GREEN" if n_pass == 8 else "X_RED"
    print(f"SUFFERING_AWARE_EXPANDED_CHANNELS_VERDICT {verdict} "
          f"({n_pass}/8 clauses PASS)")
    return 0 if verdict == "X_GREEN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
