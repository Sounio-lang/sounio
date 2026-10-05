#!/usr/bin/env python3
"""Mercyful Learning — machine-channel structural benchmark (OPUS 5, critique 5).

Companion artifact to
  docs/research/mercyful_machine_channel_benchmark_spec_2026-07-26.md
and a direct answer to OPUS 5's fifth critique of
  docs/papers/mercyful_learning_paradigm_2026-07-26.md:

  "O que de fato exerce o canal máquina é a regra de parada (39 versus 300
  épocas), não o termo rho||theta||^2. Diga isso explicitamente, ou construa
  um benchmark em que mu decida algo estrutural (largura/profundidade, FLOPs
  medidos)."

This benchmark makes mu (the machine-suffering weight) decide STRUCTURE, not
stopping time:

  * A grid of MLP architectures (width x depth) is trained on a synthetic
    2-D nonlinear classification task. ALL candidates are trained with the
    SAME fixed epoch budget — early stopping plays no role in the selection,
    so any decision mu makes is architectural (width, depth, parameters,
    FLOPs, energy), exactly what the critique demanded.
  * S_machine is MEASURED: analytic training FLOPs (6 * P * N * epochs, the
    standard GEMM-counting convention), parameter count, an energy proxy
    (FLOPs * E_PER_FLOP), and wall-clock training time. It is NOT the
    rho||theta||^2 parameter-norm proxy that OPUS 5 showed carries ~0.03% of
    the objective. The norm proxy is computed alongside to quantify how much
    larger the dynamic range of the measured channel is.
  * For each mu in a sweep, the selected architecture is
        argmin_{arch : test_acc >= TAU}  [ L_task + mu * S_machine ]
    The feasibility filter Perf >= TAU is the anti-Goodhart constraint.
  * An "abstain" option (zero parameters, zero FLOPs, majority-class
    accuracy) is included in the UNFILTERED objective. At large mu the
    unfiltered objective prefers abstention (Perf < TAU): the abstention
    trap. The anti-Goodhart filter rejects it and falls back to the smallest
    FEASIBLE architecture — the constraint, not the penalty, guards the
    target.

Synthetic data only. This benchmark makes no clinical claim and is not
medical guidance.

Certificates (contract clauses M1..M8):
  M1  mu = 0 ignores structure entirely: the selection is the best-PERFORMING
      feasible architecture (pure task loss), regardless of parameter count
  M2  selected parameter count is weakly monotone non-increasing in mu
  M3  structural shrink: params(mu=0) / params(smallest feasible selection)
      >= 8 while test accuracy stays >= TAU at every selected point
  M4  measured-FLOPs savings: training FLOPs of the smallest feasible
      selection <= 5% of the mu=0 selection (energy scales identically)
  M5  the decision is structural, not stopping: every candidate used the
      IDENTICAL fixed epoch budget, yet >= 3 distinct architectures are
      selected across the sweep
  M6  abstention trap: at the largest mu the UNFILTERED objective selects
      abstention with Perf < TAU, and the anti-Goodhart filter overrides it
      with a feasible model (Perf >= TAU)
  M7  measured channel is real: Pearson correlation between analytic
      training FLOPs and measured wall-clock training time >= 0.85
  M8  the measured channel is non-decorative: dynamic range (max/min) of
      FLOPs-based S_machine across feasible architectures >= 50 and at
      least 10x the dynamic range of the rho||theta||^2 proxy (the term
      OPUS 5 measured at ~0.03% of the objective)

Run: .venv/bin/python scripts/research/mercyful_machine_channel_benchmark.py
Requires: torch (CPU) + numpy from the repo .venv.
"""

import time

import numpy as np
import torch
import torch.nn as nn

# ---------------- determinism ----------------------------------------------
SEED = 11
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)

# ---------------- synthetic task -------------------------------------------
# 2-D sine-boundary binary classification with label noise. Small MLPs reach
# the target; larger MLPs reach it with more margin — capacity buys little
# accuracy but a lot of FLOPs, which is exactly the regime where a machine
# suffering channel should trade structure against a fixed target.
N_TRAIN, N_TEST = 5000, 1000
NOISE = 0.03
TAU = 0.88            # anti-Goodhart target: test accuracy >= TAU
EPOCHS = 300          # FIXED horizon for every candidate (M5)
LR = 1e-2

E_PER_FLOP = 4e-12    # J/FLOP, order-of-magnitude for modern CPU SIMD GEMMs
RHO = 1e-3            # paper's parameter-norm proxy coefficient (decorative)


def make_data(n, rng):
    x = rng.uniform(-np.pi, np.pi, size=(n, 2)).astype(np.float32)
    boundary = np.sin(2.0 * x[:, 0]) + 0.3 * np.sin(5.0 * x[:, 0])
    y = (x[:, 1] > boundary).astype(np.int64)
    flip = rng.uniform(0.0, 1.0, size=n) < NOISE
    y[flip] = 1 - y[flip]
    return torch.from_numpy(x), torch.from_numpy(y)


_rng = np.random.default_rng(SEED)
X_tr, Y_tr = make_data(N_TRAIN, _rng)
X_te, Y_te = make_data(N_TEST, _rng)

# ---------------- architecture grid ----------------------------------------
WIDTHS = (8, 16, 32, 64, 128)
DEPTHS = (1, 2, 3, 4)


def build_mlp(width, depth):
    layers, d_in = [], 2
    for _ in range(depth):
        layers += [nn.Linear(d_in, width), nn.Tanh()]
        d_in = width
    layers.append(nn.Linear(d_in, 2))
    return nn.Sequential(*layers)


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def train_flops(params, n_samples, epochs):
    """GEMM-convention training FLOPs: fwd = 2P/sample, bwd ~= 2x fwd."""
    return 6.0 * params * n_samples * epochs


# ---------------- train every candidate on the SAME fixed budget -----------
candidates = []  # dicts: width, depth, params, acc, s_machine (J), flops, wall
for depth in DEPTHS:
    for width in WIDTHS:
        torch.manual_seed(SEED * 1000 + depth * 100 + width)
        model = build_mlp(width, depth)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        lossf = nn.CrossEntropyLoss()
        p = count_params(model)
        t0 = time.perf_counter()
        epochs_to_tau = None
        for ep in range(EPOCHS):
            opt.zero_grad()
            loss = lossf(model(X_tr), Y_tr)
            loss.backward()
            opt.step()
            if epochs_to_tau is None and (ep + 1) % 25 == 0:
                with torch.no_grad():
                    tr_acc = (model(X_tr).argmax(1) == Y_tr).float().mean().item()
                if tr_acc >= TAU:
                    epochs_to_tau = ep + 1
        wall = time.perf_counter() - t0
        model.eval()
        with torch.no_grad():
            acc = (model(X_te).argmax(1) == Y_te).float().mean().item()
            norm2 = sum(float((q * q).sum()) for q in model.parameters())
        fl = train_flops(p, N_TRAIN, EPOCHS)
        candidates.append(dict(width=width, depth=depth, params=p, acc=acc,
                               flops=fl, energy=fl * E_PER_FLOP, wall=wall,
                               norm_proxy=RHO * norm2,
                               epochs_to_tau=epochs_to_tau))

# Abstention option: no model, no compute, majority-class accuracy.
majority = float(max((Y_te == 0).float().mean(), (Y_te == 1).float().mean()))
abstain = dict(width=0, depth=0, params=0, acc=majority, flops=0.0,
               energy=0.0, wall=0.0, norm_proxy=0.0, epochs_to_tau=None,
               abstain=True)

feasible = [c for c in candidates if c["acc"] >= TAU]
assert feasible, "no feasible architecture: task/target misconfigured"

# ---------------- mu sweep: mu decides structure ---------------------------
# Objective per candidate: J(mu) = (1 - acc) + mu * S_machine (joules).
# Filtered (anti-Goodhart): minimize over feasible only.
# Unfiltered: minimize over all candidates + abstain (exposes the trap).
s_min = min(c["energy"] for c in candidates)
s_max = max(c["energy"] for c in candidates)
MU_GRID = np.concatenate(([0.0], np.geomspace(1e-4 / s_max, 5.0 / s_min, 24)))

l_task = lambda c: 1.0 - c["acc"]  # noqa: E731


def objective(c, mu):
    """J(mu) = L_task + mu * S_machine, with a deterministic tie-break.

    The 1e-12 * params term only breaks exact task-loss ties (differences
    below ~5e-8), in which case the larger-capacity model is preferred:
    with no machine penalty pressing the choice down, capacity is free.
    """
    return l_task(c) + mu * c["energy"] - 1e-12 * c["params"]


rows = []
for mu in MU_GRID:
    sel = min(feasible, key=lambda c: objective(c, mu))
    unfiltered = min(candidates + [abstain],
                     key=lambda c: objective(c, mu))
    rows.append(dict(mu=mu, sel=sel, unfiltered=unfiltered))

sel0 = rows[0]["sel"]                       # mu = 0
sel_small = min((r["sel"] for r in rows), key=lambda c: c["params"])
last = rows[-1]

# ---------------- report ----------------------------------------------------
print("=" * 78)
print("MERCYFUL MACHINE-CHANNEL BENCHMARK (synthetic, seed=%d, torch CPU)" % SEED)
print("mu decides STRUCTURE (width/depth/params/FLOPs/energy), not stopping")
print("=" * 78)
print(f"task: sine-boundary 2-D, N_train={N_TRAIN}, N_test={N_TEST}, "
      f"noise={NOISE}, tau={TAU}, fixed EPOCHS={EPOCHS} for ALL candidates")
print(f"feasible architectures (test acc >= tau): {len(feasible)}/{len(candidates)}")
print("-" * 78)
print(f"{'arch':>10}{'params':>8}{'test_acc':>9}{'FLOPs':>12}"
      f"{'energy_J':>10}{'wall_s':>8}{'rho|t|^2':>10}{'ep2tau':>7}")
for c in candidates:
    e2t = str(c["epochs_to_tau"]) if c["epochs_to_tau"] else "-"
    tag = " *" if c["acc"] >= TAU else "  "
    print(f"{c['width']:>4}x{c['depth']:<5}{c['params']:>8}{c['acc']:>9.4f}"
          f"{c['flops']:>12.3e}{c['energy']:>10.4f}{c['wall']:>8.2f}"
          f"{c['norm_proxy']:>10.2e}{e2t:>7}{tag}")
print("  (* = feasible;  FLOPs = 6*P*N*epochs GEMM convention;  "
      f"energy = FLOPs * {E_PER_FLOP} J)")
print("-" * 78)
print(f"{'mu (1/J)':>12}{'selected':>10}{'params':>8}{'acc':>7}"
      f"{'FLOPs':>12}{'energy_J':>10}   unfiltered")
for r in rows:
    s, u = r["sel"], r["unfiltered"]
    uname = "ABSTAIN" if u.get("abstain") else f"{u['width']}x{u['depth']}"
    print(f"{r['mu']:>12.3e}{s['width']:>6}x{s['depth']:<3}{s['params']:>8}"
          f"{s['acc']:>7.4f}{s['flops']:>12.3e}{s['energy']:>10.4f}   {uname}")
print("-" * 78)

p0, p_small = sel0["params"], sel_small["params"]
shrink = p0 / p_small
flops_ratio = sel_small["flops"] / sel0["flops"]
distinct = len({id(r["sel"]) for r in rows})
sel_params = [r["sel"]["params"] for r in rows]
mono = all(a >= b for a, b in zip(sel_params, sel_params[1:]))
feas_flops = [c["flops"] for c in feasible]
feas_norm = [c["norm_proxy"] for c in feasible]
range_flops = max(feas_flops) / min(feas_flops)
range_norm = max(feas_norm) / min(feas_norm) if min(feas_norm) > 0 else float("inf")
wall = np.array([c["wall"] for c in candidates])
flops = np.array([c["flops"] for c in candidates])
pearson = float(np.corrcoef(wall, flops)[0, 1])
same_budget = True  # EPOCHS is a loop constant; asserted structurally
trap_unfiltered = last["unfiltered"].get("abstain", False) \
    and last["unfiltered"]["acc"] < TAU
trap_guarded = last["sel"]["acc"] >= TAU

print(f"mu=0 selection            : {sel0['width']}x{sel0['depth']} "
      f"({p0} params, acc {sel0['acc']:.4f})")
print(f"smallest feasible selection: {sel_small['width']}x{sel_small['depth']} "
      f"({p_small} params, acc {sel_small['acc']:.4f})")
print(f"structural shrink factor  : {shrink:.1f}x params, "
      f"FLOPs ratio {flops_ratio:.4f}")
print(f"distinct architectures selected across sweep: {distinct}")
print(f"monotone non-increasing params in mu: {mono}")
print(f"abstention trap at mu_max : unfiltered -> "
      f"{'ABSTAIN' if trap_unfiltered else 'model'} "
      f"(acc {last['unfiltered']['acc']:.4f} < tau), "
      f"anti-Goodhart -> {last['sel']['width']}x{last['sel']['depth']} "
      f"(acc {last['sel']['acc']:.4f} >= tau)")
print(f"channel dynamic ranges    : FLOPs S_machine {range_flops:.1f}x, "
      f"rho||theta||^2 proxy {range_norm:.1f}x")
print(f"wall-clock vs FLOPs Pearson r: {pearson:.3f}")
print("-" * 78)

# ---------------- certificates ----------------------------------------------
checks = []
checks.append(("M1", sel0["acc"] == max(c["acc"] for c in feasible)
               and sel0["params"] >= 10.0 * sel_small["params"],
               "mu=0 ignores structure and selects pure task performance "
               f"(best acc {sel0['acc']:.4f}, {sel0['params']} params vs "
               f"{sel_small['params']} at the small end)"))
checks.append(("M2", mono,
               "selected params weakly monotone non-increasing in mu"))
checks.append(("M3", shrink >= 8.0 and sel_small["acc"] >= TAU,
               f"structural shrink {shrink:.1f}x >= 8x at acc >= tau"))
checks.append(("M4", flops_ratio <= 0.05,
               f"measured FLOPs of smallest feasible <= 5% of mu=0 selection "
               f"(got {flops_ratio:.4f})"))
checks.append(("M5", same_budget and distinct >= 3,
               f"identical fixed epoch budget; {distinct} >= 3 distinct "
               "architectures selected (decision is structural, not stopping)"))
checks.append(("M6", trap_unfiltered and trap_guarded,
               "abstention trap: unfiltered objective abstains below tau at "
               "mu_max; anti-Goodhart filter overrides with feasible model"))
checks.append(("M7", pearson >= 0.85,
               f"measured channel is real: wall-vs-FLOPs Pearson "
               f"{pearson:.3f} >= 0.85"))
checks.append(("M8", range_flops >= 50.0 and range_flops >= 10.0 * range_norm,
               f"measured channel non-decorative: FLOPs range "
               f"{range_flops:.1f}x >= 50 and >= 10x the norm-proxy range "
               f"({range_norm:.1f}x)"))

npass = 0
for cid, ok, desc in checks:
    npass += bool(ok)
    print(f"  {cid}: {'PASS' if ok else 'FAIL'}  {desc}")
verdict = "C_GREEN" if npass == len(checks) else "C_RED"
print(f"MERCYFUL_MACHINE_CHANNEL_VERDICT {verdict} "
      f"({npass}/{len(checks)} clauses PASS)")
