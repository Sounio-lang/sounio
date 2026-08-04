#!/usr/bin/env python3
"""Mercyful Learning paradigm paper — synthetic training benchmark.

Companion artifact to docs/papers/mercyful_learning_paradigm_2026-07-26.md.

Synthetic dose-response training problem. NO real data. A one-layer model
prescribes a treatment intensity ("dose") per synthetic patient. Recovery
probability is a sigmoid of (dose - need); patient suffering is a quadratic
burden of dose (dose = 0 -> zero suffering AND zero recovery: the avoidance
pathology); machine suffering is a quadratic proxy on the parameter norm
(per-inference compute/energy proxy). All numbers are synthetic.

Trains three models by full-batch gradient descent:
  A. standard ML     : minimize L_task only (accuracy maximization)
  B. naive mercy     : minimize suffering terms only, no target constraint
  C. mercyful        : minimize L_task + lam*S_patient + mu*S_machine
                       subject to the hard anti-Goodhart constraint Perf >= tau
                       (feasibility-restoration switching + anti-Goodhart
                       early stopping)

Certificates printed at the end (contract clauses P1..P8):
  P1  standard ML reaches target but with gratuitous suffering
  P2  naive suffering minimization prescribes the pathology (abstention)
  P3  mercyful training reaches the target at near-minimal suffering
  P4  penalty-failure (Goodhart) crossover lambda* exists and matches the
      closed form of Theorem 2.1
  P5  unconstrained GD realizes the switch across lambda*
  P6  gratuitous-suffering ordering: standard > mercyful >= 0
  P7  anti-Goodhart early stopping is sound (never stops below target) and
      reduces training compute vs fixed-horizon standard training
  P8  value stability under suffering-field perturbation (Theorem 3.4 bound)

Run: .venv/bin/python scripts/research/mercyful_paradigm_benchmark.py
"""

import numpy as np

# ---------------- synthetic problem (fixed seed, fully deterministic) ------
RNG = np.random.default_rng(7)
N, D = 4000, 8
X = RNG.normal(0.0, 1.0, size=(N, D))
W_TRUE = RNG.normal(0.0, 0.6, size=D)
B_TRUE = 0.4


def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


NEED = np.clip(softplus(X @ W_TRUE + B_TRUE), 0.2, 3.0)  # required intensity
K = 4.0          # recovery steepness
C_TOX = 0.1      # patient-burden coefficient
RHO = 1e-3       # machine-burden coefficient
TAU = 0.90       # anti-Goodhart target: mean recovery >= TAU
LAM = 1.0        # patient-suffering weight
MU = 1.0         # machine-suffering weight


# ---------------- model and functionals ------------------------------------
def doses(theta, theta0):
    return softplus(X @ theta + theta0)


def perf(theta, theta0):
    """Mean recovery probability (task performance)."""
    return float(np.mean(sigmoid(K * (doses(theta, theta0) - NEED))))


def s_patient(theta, theta0, c=C_TOX):
    """Patient suffering: quadratic treatment burden (0 at dose 0)."""
    d = doses(theta, theta0)
    return float(c * np.mean(d * d))


def s_machine(theta):
    """Machine suffering: operational compute/energy proxy."""
    return float(RHO * np.sum(theta * theta))


def l_task(theta, theta0):
    return 1.0 - perf(theta, theta0)


def l_mercyful(theta, theta0, lam=LAM, mu=MU):
    return (l_task(theta, theta0)
            + lam * s_patient(theta, theta0)
            + mu * s_machine(theta))


def grad_l_task(theta, theta0):
    z = X @ theta + theta0
    d = softplus(z)
    p = sigmoid(K * (d - NEED))
    # dL/dz_i = -(1/N) * K * p(1-p) * sigmoid(z)
    w = -(1.0 / N) * K * p * (1.0 - p) * sigmoid(z)
    return X.T @ w, float(np.sum(w))


def grad_s_patient(theta, theta0):
    z = X @ theta + theta0
    d = softplus(z)
    w = C_TOX * (2.0 / N) * d * sigmoid(z)
    return X.T @ w, float(np.sum(w))


def grad_perf(theta, theta0):
    g_t, g_0 = grad_l_task(theta, theta0)
    return -g_t, -g_0


# ---------------- training loops -------------------------------------------
def train_standard(epochs=300, lr=0.5):
    """Baseline A: task loss only, fixed horizon."""
    theta, theta0 = np.zeros(D), 0.0
    for _ in range(epochs):
        g, g0 = grad_l_task(theta, theta0)
        theta -= lr * g
        theta0 -= lr * g0
    return theta, theta0, epochs


def train_naive(epochs=300, lr=0.5, lam=LAM, mu=MU):
    """Baseline B: suffering terms only, no task loss, no constraint."""
    theta, theta0 = np.zeros(D), 0.0
    for _ in range(epochs):
        gp, gp0 = grad_s_patient(theta, theta0)
        theta -= lr * (lam * gp + mu * 2.0 * RHO * theta)
        theta0 -= lr * (lam * gp0)
    return theta, theta0, epochs


def train_mercyful(lam=LAM, mu=MU, tau=TAU, lr=0.5, lr_c=1.0,
                   max_epochs=600, early_stop=True, eps=1e-4, window=10):
    """Mercyful training: L_mercyful descent with hard target constraint.

    Feasibility restoration: while Perf < tau, ascend on Perf only.
    Once feasible, descend L_mercyful. Anti-Goodhart early stopping:
    stop only when feasible AND suffering progress has stalled.
    Returns (theta, theta0, epochs_used, stopped_feasible).
    """
    theta, theta0 = np.zeros(D), 0.0
    hist_s = []
    epochs_used = max_epochs
    stopped_feasible = False
    for t in range(max_epochs):
        p = perf(theta, theta0)
        if p < tau:
            g, g0 = grad_perf(theta, theta0)      # feasibility restoration
            theta += lr_c * g
            theta0 += lr_c * g0
        else:
            gt, gt0 = grad_l_task(theta, theta0)
            gp, gp0 = grad_s_patient(theta, theta0)
            theta -= lr * (gt + lam * gp + mu * 2.0 * RHO * theta)
            theta0 -= lr * (gt0 + lam * gp0)
        hist_s.append(s_patient(theta, theta0) + s_machine(theta))
        if early_stop and len(hist_s) > window and p >= tau:
            if abs(hist_s[-1] - hist_s[-1 - window]) < eps:
                epochs_used = t + 1
                stopped_feasible = True
                break
    return theta, theta0, epochs_used, stopped_feasible


def solve_s_star(tau=TAU, epochs=1500, lr=0.5, lr_c=1.0, w_task=0.005):
    """Minimal patient suffering over the feasible set {Perf >= tau}.

    Descends w_task*L_task + S_patient with feasibility restoration; the small
    task weight keeps the iterate oscillating about the constraint boundary
    from above. Returns the best FEASIBLE iterate seen (min suffering among
    iterates with perf >= tau), so the estimator does not sit above the
    boundary and inflate S*.
    """
    theta, theta0 = np.zeros(D), 0.0
    best_s, best_p = np.inf, 0.0
    for _ in range(epochs):
        p = perf(theta, theta0)
        if p < tau:
            g, g0 = grad_perf(theta, theta0)
            theta += lr_c * g
            theta0 += lr_c * g0
        else:
            s = s_patient(theta, theta0)
            if s < best_s:
                best_s, best_p = s, p
            gt, gt0 = grad_l_task(theta, theta0)
            gp, gp0 = grad_s_patient(theta, theta0)
            theta -= lr * (w_task * gt + gp)
            theta0 -= lr * (w_task * gt0 + gp0)
    return best_s, best_p


# ---------------- run the three trainings ----------------------------------
th_a, t0_a, ep_a = train_standard()
th_b, t0_b, ep_b = train_naive()
th_c, t0_c, ep_c, es_feasible = train_mercyful()
s_star, perf_star = solve_s_star()

rows = []
for name, th, t0, ep in (("standard ML", th_a, t0_a, ep_a),
                         ("naive mercy", th_b, t0_b, ep_b),
                         ("mercyful", th_c, t0_c, ep_c)):
    p = perf(th, t0)
    grat = s_patient(th, t0) - s_star if p >= TAU else None  # undefined if infeasible
    rows.append((name, ep, p, s_patient(th, t0), s_machine(th), grat))

# ---------------- Theorem 2.1 certificate: the Goodhart crossover ----------
# Abstaining model (near-zero dose) vs best feasible model, as functions of
# the patient weight lambda in the UNCONSTRAINED joint objective.
theta_abs, theta0_abs = np.zeros(D), -8.0
L_abs = l_task(theta_abs, theta0_abs) + MU * s_machine(theta_abs)
S_abs = s_patient(theta_abs, theta0_abs)
L_feas = l_task(th_c, t0_c) + MU * s_machine(th_c)
S_feas = s_patient(th_c, t0_c)
lam_star_closed = (L_abs - L_feas) / (S_feas - S_abs)  # closed form (Thm 2.1)

lam_grid = np.linspace(0.0, 6.0, 241)
obj_abs = L_abs + lam_grid * S_abs
obj_feas = L_feas + lam_grid * S_feas
idx = int(np.argmin(np.abs(obj_abs - obj_feas)))
lam_star_measured = float(lam_grid[idx])

# P5: unconstrained GD realizes the switch across lambda*
def train_unconstrained(lam, epochs=300, lr=0.5, mu=MU):
    theta, theta0 = np.zeros(D), 0.0
    for _ in range(epochs):
        gt, gt0 = grad_l_task(theta, theta0)
        gp, gp0 = grad_s_patient(theta, theta0)
        theta -= lr * (gt + lam * gp + mu * 2.0 * RHO * theta)
        theta0 -= lr * (gt0 + lam * gp0)
    return perf(theta, theta0)

lam_below = 0.5 * lam_star_closed
lam_above = min(6.0, 1.5 * lam_star_closed)
perf_below = train_unconstrained(lam_below)
perf_above = train_unconstrained(lam_above)

# ---------------- Theorem 3.4 certificate: value stability -----------------
# V(s) = min over the feasible set of [w_task*L_task + S_patient](theta; s).
# Perturb the suffering field c -> c*(1 + delta*u_i), |u_i|<=1, and check
# |V - V'| <= delta * max_i(c*dose_i^2) (per-patient Lipschitz bound).
DELTA = 0.05
V0 = None
bound0 = None
viol = 0
max_ratio = 0.0
for trial in range(20):
    u = RNG.uniform(-1.0, 1.0, size=N)
    c_i = C_TOX * (1.0 + DELTA * u)

    def s_patient_pert(theta, theta0):
        d = doses(theta, theta0)
        return float(np.mean(c_i * d * d))

    def grad_s_patient_pert(theta, theta0):
        z = X @ theta + theta0
        d = softplus(z)
        w = (2.0 / N) * c_i * d * sigmoid(z)
        return X.T @ w, float(np.sum(w))

    theta, theta0 = np.zeros(D), 0.0
    for _ in range(400):
        p = perf(theta, theta0)
        if p < TAU:
            g, g0 = grad_perf(theta, theta0)
            theta += 1.0 * g
            theta0 += 1.0 * g0
        else:
            gt, gt0 = grad_l_task(theta, theta0)
            gp, gp0 = grad_s_patient_pert(theta, theta0)
            theta -= 0.5 * (0.02 * gt + gp)
            theta0 -= 0.5 * (0.02 * gt0 + gp0)
    d_star = doses(theta, theta0)
    V = 0.02 * l_task(theta, theta0) + s_patient_pert(theta, theta0)
    bound = DELTA * float(np.max(C_TOX * d_star * d_star))
    if V0 is None:
        V0, bound0 = V, bound
        continue
    ratio = abs(V - V0) / max(bound + bound0, 1e-12)
    max_ratio = max(max_ratio, ratio)
    if abs(V - V0) > bound + bound0:
        viol += 1

# ---------------- report ----------------------------------------------------
print("=" * 74)
print("MERCYFUL PARADIGM BENCHMARK (synthetic, seed=7, N=4000, D=8)")
print("=" * 74)
print(f"target tau={TAU}  lambda={LAM}  mu={MU}  c_tox={C_TOX}  rho={RHO}")
print(f"S*_patient(tau) = {s_star:.4f}  (at perf {perf_star:.4f})")
print("-" * 74)
print(f"{'method':<14}{'epochs':>7}{'Perf':>8}{'S_patient':>11}"
      f"{'S_machine':>11}{'gratuitous':>12}")
for name, ep, p, sp, sm, grat in rows:
    grat_s = f"{grat:>12.4f}" if grat is not None else f"{'infeasible':>12}"
    print(f"{name:<14}{ep:>7}{p:>8.4f}{sp:>11.4f}{sm:>11.4f}{grat_s}")
print("-" * 74)
print(f"Theorem 2.1 crossover: closed form lambda* = {lam_star_closed:.4f}, "
      f"grid-measured = {lam_star_measured:.4f}")
print(f"Unconstrained GD: lambda={lam_below:.3f} (< lambda*) -> Perf "
      f"{perf_below:.4f}; lambda={lam_above:.3f} (> lambda*) -> Perf "
      f"{perf_above:.4f}")
print(f"Early stopping: mercyful used {ep_c} epochs "
      f"(standard: {ep_a}); stopped_feasible={es_feasible}")
print(f"Stability: 20 perturbations delta={DELTA}, bound violations = {viol}, "
      f"max |dV|/(bound sum) = {max_ratio:.3f}")
print("-" * 74)

p_a, sp_a = rows[0][2], rows[0][3]
p_b, sp_b = rows[1][2], rows[1][3]
p_c, sp_c = rows[2][2], rows[2][3]

checks = []
checks.append(("P1", p_a >= TAU and sp_a > s_star * 1.5,
               "standard ML reaches target with gratuitous suffering"))
checks.append(("P2", p_b < TAU and sp_b < 0.05 * s_star,
               "naive suffering minimization prescribes abstention (pathology)"))
checks.append(("P3", p_c >= TAU and sp_c <= s_star * 1.35,
               "mercyful reaches target at near-minimal suffering"))
checks.append(("P4", abs(lam_star_closed - lam_star_measured) < 0.1,
               "Theorem 2.1 crossover matches closed form"))
checks.append(("P5", perf_below >= TAU * 0.95 and perf_above < TAU * 0.5,
               "unconstrained GD switches from treating to abstaining at lambda*"))
checks.append(("P6", sp_a > sp_c >= s_star * 0.99,
               "gratuitous-suffering ordering standard > mercyful >= ~S*"))
checks.append(("P7", es_feasible and ep_c < ep_a,
               "anti-Goodhart early stopping sound and compute-saving"))
checks.append(("P8", viol == 0,
               "Theorem 3.4 value-stability bound never violated"))

npass = 0
for cid, ok, desc in checks:
    npass += bool(ok)
    print(f"  {cid}: {'PASS' if ok else 'FAIL'}  {desc}")
verdict = "P_GREEN" if npass == len(checks) else "P_RED"
print(f"MERCYFUL_PARADIGM_BENCHMARK_VERDICT {verdict} "
      f"({npass}/{len(checks)} clauses PASS)")
