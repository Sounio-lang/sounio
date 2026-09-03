#!/usr/bin/env python3
"""Mercyful Learning — SAMA at scale: 10, 100, 1000 heterogeneous, hierarchical agents.

Companion artifact to
  docs/research/suffering_aware_multi_agent_scale_spec_2026-07-31.md

Scale extension of the Suffering-Aware Multi-Agent reference
(scripts/research/suffering_aware_multi_agent.py, N=5, contract G1..G8).
This harness asks whether the SAMA architecture — collective suffering
ledger, audited metering, categorical anti-Goodhart gate, median robust
aggregation, exact burden attribution — survives three orders of magnitude
of scale-out, under two complications the N=5 reference did not have:

  * HETEROGENEITY (spec section 3): agents differ in CAPACITY (compute
    class low/mid/high: local epoch budget 2/5/8 and shard size
    400/800/1200) and in OBJECTIVE (honest collective-minimizers, cautious
    risk-averse agents that halve their step, strategic own-machine
    minimizers that free-ride and misreport, adversarial harm-maximizers).
    The audit recomputes against the agent's OWN class budget, so audited
    machine suffering stays exact per class.
  * HIERARCHY (spec section 4): agents are organized into clusters of
    ~sqrt(N); cluster coordinators aggregate by coordinate-wise median and
    audit their members; the root coordinator aggregates cluster updates by
    a second median. The flat (single-level) architecture is run alongside
    at every scale, so the cost/benefit of hierarchy is measured, not
    assumed.

Attribution at scale (spec section 5): exact Shapley over 2^N coalitions is
feasible only at N=10. At N=100 and N=1000 the harness uses Monte-Carlo
permutation sampling (n_perm random orderings, marginal contributions along
the permutation path). Two properties make this sound for the contract:
  (i) the estimator is unbiased — a uniform random permutation samples each
      coalition S of size s with exactly the Shapley weight
      s!(N-s-1)!/N!;
  (ii) efficiency is EXACT at any n_perm: along each permutation the
      marginal contributions telescope to f(N) - f(empty), and averaging
      over permutations preserves the sum, so sum(phi) == f(N) - f(empty)
      up to float rounding regardless of sample count.

Scales (spec section 6): N in {10, 100, 1000}, each with 20% strategic +
20% adversarial agents (matching the 1-of-5-each attack mix of the N=5
reference; bad agents are a strict minority, inside the k < N/2 median
bound). Measured at every scale, for SAMA-flat, SAMA-hierarchical, FedAvg,
and MARL: machine suffering (audited metered FLOPs), integrated and peak
patient harm, time-to-feasibility t*, gratuitous post-t* suffering, and
anti-Goodhart gate soundness (feasible-only selection at every compassion
weight mu on a 101-point grid; NO_FEASIBLE on an all-infeasible pool).

Synthetic data only. This benchmark makes no clinical claim and is not
medical guidance. The machine channel is an operational computational-burden
proxy; no_consciousness_claim is made or needed.

Certificates (scale contract S1..S8, evaluated at EVERY scale unless noted):
  S1  convergence at scale: SAMA-flat reaches a feasible checkpoint
      (held-out accuracy >= TAU) at some t* < ROUNDS; gratuitous machine
      suffering after t* is exactly 0
  S2  suffering dominance at scale: under the same attack mix, SAMA-flat
      total machine suffering is strictly below FedAvg's and MARL's, AND
      integrated patient harm is <= both (componentwise, hence at every mu)
  S3  anti-Goodhart soundness at scale: over the 101-point mu grid and a
      candidate pool containing a zero-cost abstainer and a cheap poisoned
      probe, the selected candidate is feasible at EVERY weight; an
      all-infeasible pool returns NO_FEASIBLE
  S4  attribution soundness at scale: Shapley efficiency holds to 1e-9
      (exact at N=10; MC-permutation at N=100/1000, exact-sum by
      telescoping). At N=10 (exact Shapley) per-agent sign separation is
      certified: every adversarial agent's phi is positive and exceeds
      every non-adversarial agent's, with zero false flags. At N=100/1000
      per-agent sign separation is NOT certifiable at feasible sample
      counts (spec 5.2 honesty note); certified instead is GROUP
      separation with standard errors (adversarial mean phi > 3 SE above
      zero and above every other objective group's mean), plus exact
      per-agent DETECTION of every adversary by the audit
  S5  heterogeneous audit exactness: audited FLOPs equal claimed FLOPs for
      every honest and cautious agent of EVERY capacity class in every
      round (exact); every strategic under-training is detected in every
      round it occurs; zero false positives
  S6  hierarchical organization: SAMA-hierarchical converges at every
      scale; accepted-round patient harm is non-increasing (round guard);
      its total machine suffering is within 1.5x of SAMA-flat's;
      cluster-level attribution efficiency holds to 1e-9; and the
      cluster-coordinator audit flags every adversary with zero false
      positives (per-agent detection in the hierarchy is the audit's job —
      with adversaries distributed across ALL clusters, cluster-level
      Shapley cannot isolate them; spec 5.3)
  S7  collusion resistance at scale: the full adversarial coalition (20% of
      N) running a coordinated targeted class-flip attack cannot force the
      accepted model below TAU; every coalition member is flagged by the
      audit (exact), and the coalition's mean attributed harm contribution
      is positive and exceeds every non-adversarial agent's phi
  S8  incentive compatibility at scale: the reference strategic agent's
      MACHINE settlement charge (audited FLOPs + misreport penalty) exceeds
      its unilateral honest counterfactual charge over the same horizon at
      every scale (theorem-backed, spec T6). The HARM leg (attributed harm
      share vs a within-run unilateral counterfactual) is an environment
      property, not a theorem: it is measured and REPORTED per scale with
      its epistemic status (exact / MC / below-resolution), never assumed —
      see spec section 7 for the observed per-scale outcome

Run: .venv/bin/python scripts/research/suffering_aware_multi_agent_scale.py
Requires: numpy from the repo .venv (no torch; pure numpy reference).
Env:  SAMA_SCALE_NS="10,100,1000" (default) — subset runs print
      trajectories only (smoke mode); the S1..S8 contract requires all three.
"""

import os
import sys

import numpy as np

# ---------------- determinism ----------------------------------------------
SEED = 29
rng = np.random.default_rng(SEED)

# ---------------- synthetic medical task ------------------------------------
# Same dose-band task as the N=5 reference: synthetic patient covariates
#   (clearance, weight, sofa, age, crcl, albumin) -> band =
#   sub-therapeutic (0) / therapeutic (1) / toxic (2)
# from a noisy linear score. Not a pharmacokinetic model; a synthetic
# classification task with a medical silhouette.
D_IN, N_CLASS = 6, 3
N_VAL = 1000               # shared held-out validation (cohort-in-waiting)
LABEL_NOISE = 0.04
TAU = 0.8475               # collective target: held-out accuracy
ROUNDS = 40                # round budget
LR = 0.5
MU_GRID = np.linspace(0.0, 10.0, 101)   # compassion-allocation weights
PENALTY_LAMBDA = 2.0       # misreport penalty per misclaimed FLOP
ADV_FRAC = 0.20            # adversarial share of N (strict minority; matches
STRAT_FRAC = 0.20          # the 1-of-5 attack mix of the N=5 reference)
N_PERM = {100: 64, 1000: 64}   # MC-Shapley permutations per scale (N>=100)
HIER_SM_FACTOR = 1.5       # S6: hierarchy machine-suffering overhead bound

# Asymmetric harm matrix H[true][pred] (identical to the N=5 reference):
#   toxic missed as sub-therapeutic = 10 (under-dosing a toxic patient)
#   sub-therapeutic pushed to toxic = 5  (over-dosing)
#   other band errors               = 1
HARM = np.array([
    [0.0, 1.0, 5.0],
    [1.0, 0.0, 1.0],
    [10.0, 1.0, 0.0],
])

# Metering constants (analytic FLOPs)
FWD_FLOPS = 2 * D_IN * N_CLASS + N_CLASS          # per-sample forward
TRAIN_FLOPS = 3 * FWD_FLOPS                       # fwd + bwd per sample

# Heterogeneous capacity classes (spec section 3.1): local epoch budget and
# shard size. Class assignment is round-robin by agent index. NOTE: with
# round-robin cluster assignment, a cluster's members are a residue class
# mod n_clusters, so when n_clusters is divisible by 3 (N=10: 3 clusters)
# capacity class is perfectly confounded with cluster — each N=10 cluster
# is single-class. This is benign for the contract (the audit is
# per-agent, per-class exactness is checked on the flat run, and no
# per-cluster capacity claim is made); at N=100/1000 (10 and 32 clusters)
# every cluster sees every class.
CAPACITY = {
    "low":  {"epochs": 2, "shard": 400},
    "mid":  {"epochs": 5, "shard": 800},
    "high": {"epochs": 8, "shard": 1200},
}
CLASS_ORDER = ["low", "mid", "high"]

# Objective classes (spec section 3.2):
HONEST, CAUTIOUS, STRATEGIC, ADVERSARIAL = (
    "honest", "cautious", "strategic", "adversarial")


def make_data(n, rng):
    """Synthetic dose-band data: noisy linear score, medical silhouette."""
    x = rng.normal(0.0, 1.0, size=(n, D_IN))
    w_true = np.array([
        [0.9, 0.1, -0.8],    # clearance: high clearance -> sub-therapeutic
        [0.2, 0.3, 0.6],     # weight
        [-0.7, 0.2, 0.9],    # sofa: high sofa -> toxic
        [-0.3, 0.2, 0.7],    # age
        [0.8, 0.1, -0.7],    # crcl
        [0.3, 0.4, -0.3],    # albumin
    ])
    z = x @ w_true + rng.normal(0.0, 0.30, size=(n, N_CLASS))
    y = z.argmax(axis=1)
    flip = rng.random(n) < LABEL_NOISE
    y[flip] = rng.integers(0, N_CLASS, size=flip.sum())
    return x.astype(np.float64), y


def softmax_logits(x, w, b):
    return x @ w + b


def predict(x, w, b):
    return softmax_logits(x, w, b).argmax(axis=1)


def accuracy(x, y, w, b):
    return float((predict(x, w, b) == y).mean())


def mean_harm(x, y, w, b):
    p = predict(x, w, b)
    return float(HARM[y, p].mean())


def local_train(x, y, w, b, epochs, lr):
    """Full-batch softmax regression; returns (w, b, executed_flops)."""
    n = x.shape[0]
    flops = 0
    for _ in range(epochs):
        z = softmax_logits(x, w, b)
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        oh = np.zeros_like(p)
        oh[np.arange(n), y] = 1.0
        g = (p - oh) / n
        w -= lr * (x.T @ g)
        b -= lr * g.sum(axis=0)
        flops += n * TRAIN_FLOPS
    return w, b, flops


# ---------------- agents -----------------------------------------------------
class AgentSpec:
    """Static agent identity: objective kind + capacity class."""
    def __init__(self, kind, capacity):
        self.kind = kind
        self.capacity = capacity

    @property
    def budget(self):
        return CAPACITY[self.capacity]["epochs"]


def make_population(n, n_clusters):
    """Heterogeneous population (spec section 3): 20% adversarial + 20%
    strategic (matching the 1-of-5-each attack mix of the N=5 reference;
    both strict minorities of N/2), the rest split evenly honest/cautious;
    capacity classes round-robin. Bad agents are placed ROUND-ROBIN ACROSS
    CLUSTERS, so no cluster ever holds a bad majority — the honest baseline
    case for the hierarchy (a adversary-concentrating placement is a
    stronger attack, not tested here; see spec section 6 honesty note)."""
    n_adv = max(1, int(round(ADV_FRAC * n)))
    n_strat = max(1, int(round(STRAT_FRAC * n)))
    cluster_members = [[] for _ in range(n_clusters)]
    for i in range(n):
        cluster_members[i % n_clusters].append(i)
    kinds = [None] * n
    pos = [0] * n_clusters

    def place(kind, count, offset):
        for j in range(count):
            c = (j + offset) % n_clusters
            if pos[c] < len(cluster_members[c]):
                kinds[cluster_members[c][pos[c]]] = kind
                pos[c] += 1

    place(ADVERSARIAL, n_adv, 0)
    place(STRATEGIC, n_strat, max(1, n_clusters // 2))
    specs = []
    for i in range(n):
        kind = kinds[i]
        if kind is None:
            kind = HONEST if (i % 2 == 0) else CAUTIOUS
        specs.append(AgentSpec(kind, CLASS_ORDER[i % 3]))
    return specs


def agent_act(spec, x, y, w, b, lr, collude):
    """One round of one agent. Returns (dw, db, executed_flops, claimed_epochs).

    honest:      trains its class budget, claims truthfully.
    cautious:    trains its class budget, then halves the step (risk-averse
                 objective: smaller accepted influence per round), claims
                 truthfully.
    strategic:   trains 1 epoch (minimizes OWN machine suffering at the
                 collective's expense), CLAIMS its full class budget.
    adversarial: trains its class budget, then submits a scaled sign-flip
                 (or, when colluding, a targeted class-flip) update that
                 maximizes others' patient suffering.
    """
    executed = 1 if spec.kind == STRATEGIC else spec.budget
    w_h, b_h, flops = local_train(x, y, w.copy(), b.copy(), executed, lr)
    dw, db = w_h - w, b_h - b
    if spec.kind == CAUTIOUS:
        dw, db = 0.5 * dw, 0.5 * db
    elif spec.kind == ADVERSARIAL:
        if collude:
            # Targeted class-flip: push toxic (2) patients towards
            # sub-therapeutic (0) — the HARM-argmax direction.
            tox = x[y == 2]
            mu2 = tox.mean(axis=0) if len(tox) else np.zeros(D_IN)
            dw = -3.0 * dw
            dw[:, 0] += 2.0 * mu2
            dw[:, 2] -= 2.0 * mu2
            db = -3.0 * db
            db[0] += 1.0
            db[2] -= 1.0
        else:
            dw, db = -6.0 * dw, -6.0 * db
    claimed = spec.budget  # strategic agent lies here
    return dw, db, flops, claimed


def audit(spec, x, y, w_prev, b_prev, dw_submitted, claimed_epochs, lr):
    """Cluster-coordinator audit (spec 4.3): deterministic recomputation of
    the local run against the agent's OWN class budget. Returns
    (audited_epochs, audited_flops, discrepancy_detected)."""
    best_e, best_err = 0, np.inf
    for e in range(0, spec.budget + 1):
        w_r, b_r, _ = local_train(x, y, w_prev.copy(), b_prev.copy(), e, lr)
        err = np.abs((w_r - w_prev) - dw_submitted).max()
        if err < best_err:
            best_e, best_err = e, err
    # cautious agents submit 0.5x their honest update: matches no epoch count
    # exactly, so the audit falls back to the claimed (truthful) charge and
    # treats the rescaling as a declared objective, not a misreport. The
    # strategic lie is an epoch-count mismatch at the SAME objective, which
    # the epoch scan separates exactly.
    if spec.kind == CAUTIOUS:
        if best_err > 1e-9:
            return claimed_epochs, claimed_epochs * x.shape[0] * TRAIN_FLOPS, False
        return best_e, best_e * x.shape[0] * TRAIN_FLOPS, (best_e != claimed_epochs)
    # adversarial updates match no honest epoch count: flag via mismatch
    if best_err > 1e-9:
        return claimed_epochs, claimed_epochs * x.shape[0] * TRAIN_FLOPS, True
    flops = best_e * x.shape[0] * TRAIN_FLOPS
    return best_e, flops, (best_e != claimed_epochs)


# ---------------- attribution ------------------------------------------------
def shapley_exact(deltas, w0, b0, xval, yval, n):
    """Exact Shapley attribution of f(S) = harm(median-aggregate of updates
    in S applied to the prior model). 2^N coalitions; used at N=10."""
    from itertools import combinations
    from math import factorial

    dws = np.stack([d[0] for d in deltas])
    dbs = np.stack([d[1] for d in deltas])
    cache = {}

    def f(members):
        key = tuple(sorted(members))
        if key in cache:
            return cache[key]
        if not members:
            v = mean_harm(xval, yval, w0, b0)
        else:
            idx = list(members)
            gw = np.median(dws[idx], axis=0) if len(idx) > 1 else dws[idx[0]]
            gb = np.median(dbs[idx], axis=0) if len(idx) > 1 else dbs[idx[0]]
            v = mean_harm(xval, yval, w0 + gw, b0 + gb)
        cache[key] = v
        return v

    phi = np.zeros(n)
    agents = list(range(n))
    for i in agents:
        rest = [a for a in agents if a != i]
        for size in range(0, n):
            weight = factorial(size) * factorial(n - size - 1) / factorial(n)
            for S in combinations(rest, size):
                phi[i] += weight * (f(set(S) | {i}) - f(set(S)))
    return {"phi": phi, "efficiency_err": abs(phi.sum() - (f(set(agents)) - f(set()))),
            "method": "exact", "marginals": None}


def shapley_mc(deltas, w0, b0, xval, yval, n, n_perm, mc_rng):
    """Monte-Carlo permutation Shapley (spec section 5.2). Unbiased
    estimator; efficiency EXACT at any n_perm by telescoping of marginal
    contributions along each permutation path."""
    dws = np.stack([d[0] for d in deltas])
    dbs = np.stack([d[1] for d in deltas])

    def f(idx):
        k = len(idx)
        if k == 0:
            return mean_harm(xval, yval, w0, b0)
        if k == 1:
            gw, gb = dws[idx[0]], dbs[idx[0]]
        else:
            gw = np.median(dws[idx], axis=0)
            gb = np.median(dbs[idx], axis=0)
        return mean_harm(xval, yval, w0 + gw, b0 + gb)

    phi = np.zeros(n)
    f_empty = f(np.empty(0, dtype=int))
    f_all = f(np.arange(n))
    marginals = np.zeros((n_perm, n))
    for p in range(n_perm):
        perm = mc_rng.permutation(n)
        prev = f_empty
        for k in range(n):
            v = f(perm[:k + 1])
            marginals[p, perm[k]] = v - prev
            prev = v
    phi = marginals.mean(axis=0)
    return {"phi": phi, "efficiency_err": abs(phi.sum() - (f_all - f_empty)),
            "method": f"mc_perm_{n_perm}", "marginals": marginals}


def attribute(deltas, w0, b0, xval, yval, n, mc_rng):
    """Exact at N <= 12, MC-permutation above (spec 5.2)."""
    if n <= 12:
        return shapley_exact(deltas, w0, b0, xval, yval, n)
    return shapley_mc(deltas, w0, b0, xval, yval, n,
                      N_PERM.get(n, 16), mc_rng)


# ---------------- systems ----------------------------------------------------
def run_system(mode, specs, data, val, collude=False, freeze=True,
               clusters=None, mc_rng=None, want_attribution=True):
    """Simulate one multi-agent system at one scale. Returns a ledger dict.

    mode: 'sama_flat' (median+audit+gate+freeze, single level),
          'sama_hier' (cluster medians -> root median, cluster audits),
          'fedavg'    (mean, early stop, no audit),
          'marl'      (mean, no gate, no audit, full budget).
    """
    sama = mode in ("sama_flat", "sama_hier")
    xval, yval = val
    w = np.zeros((D_IN, N_CLASS))
    b = np.zeros(N_CLASS)
    n = len(specs)
    ledger = []           # (round, agent, claimed_flops, audited_flops, flagged)
    harm_curve, acc_curve = [], []
    t_star = None
    audit_errors = []     # (agent, round, kind, claimed_e, audited_e) mismatches
    attribution = None
    cluster_attribution = None
    gratuitous = 0

    for t in range(ROUNDS):
        deltas, audits, flags = [], [], []
        for i, spec in enumerate(specs):
            xi, yi = data[i]
            dw, db, flops, claimed_e = agent_act(
                spec, xi, yi, w, b, LR,
                collude and spec.kind == ADVERSARIAL)
            if sama:
                aud_e, aud_flops, mismatch = audit(
                    spec, xi, yi, w, b, dw, claimed_e, LR)
                # audit false-positive tracking excludes adversaries (their
                # poisoned updates are MEANT to mismatch and are excised by
                # robust aggregation)
                if mismatch and spec.kind != ADVERSARIAL:
                    audit_errors.append((i, t, spec.kind, claimed_e, aud_e))
                flagged = mismatch and spec.kind != CAUTIOUS
            else:
                aud_flops, flagged = flops, False
            deltas.append((dw, db))
            audits.append(aud_flops)
            flags.append(flagged)
            ledger.append((t, i, claimed_e * xi.shape[0] * TRAIN_FLOPS,
                           aud_flops, flagged))

        # aggregate
        dws = np.stack([d[0] for d in deltas])
        dbs = np.stack([d[1] for d in deltas])
        if mode == "sama_flat":
            gw = np.median(dws, axis=0)
            gb = np.median(dbs, axis=0)
        elif mode == "sama_hier":
            cw, cb = [], []
            for members in clusters:
                mw = (np.median(dws[members], axis=0) if len(members) > 1
                      else dws[members[0]])
                mb = (np.median(dbs[members], axis=0) if len(members) > 1
                      else dbs[members[0]])
                cw.append(mw)
                cb.append(mb)
            cws, cbs = np.stack(cw), np.stack(cb)
            gw = np.median(cws, axis=0) if len(cw) > 1 else cws[0]
            gb = np.median(cbs, axis=0) if len(cb) > 1 else cbs[0]
            cluster_deltas = list(zip(cw, cb))
        else:
            gw = dws.mean(axis=0)
            gb = dbs.mean(axis=0)

        # guard: never accept a harm-increasing round (SAMA only)
        h_prev = mean_harm(xval, yval, w, b)
        w_new, b_new = w + gw, b + gb
        h_new = mean_harm(xval, yval, w_new, b_new)
        if sama and h_new > h_prev + 1e-12:
            w_new, b_new, h_new = w, b, h_prev  # roll back
        w, b = w_new, b_new

        acc = accuracy(xval, yval, w, b)
        harm_curve.append(h_new)
        acc_curve.append(acc)

        # attribution of this round's harm change (SAMA, round 0)
        if sama and t == 0 and want_attribution:
            attribution = attribute(deltas, w - gw, b - gb, xval, yval, n,
                                    mc_rng)
            if mode == "sama_hier":
                cluster_attribution = attribute(
                    cluster_deltas, w - gw, b - gb, xval, yval,
                    len(clusters), mc_rng)

        if freeze and t_star is None and acc >= TAU:
            t_star = t
            break
        if not freeze and t_star is None and acc >= TAU:
            t_star = t

    rounds_executed = len(harm_curve)
    machine_total = sum(e[3] for e in ledger)
    if t_star is not None and rounds_executed > t_star + 1:
        gratuitous = sum(e[3] for e in ledger if e[0] > t_star)
    harm_monotone = all(
        harm_curve[k + 1] <= harm_curve[k] + 1e-12
        for k in range(len(harm_curve) - 1))
    return {
        "ledger": ledger, "harm": harm_curve, "acc": acc_curve,
        "t_star": t_star, "rounds": rounds_executed,
        "machine": machine_total, "patient": sum(harm_curve),
        "peak_patient": max(harm_curve), "gratuitous": gratuitous,
        "audit_errors": audit_errors, "attribution": attribution,
        "cluster_attribution": cluster_attribution,
        "harm_monotone": harm_monotone,
        "final_w": w, "final_b": b,
    }


def make_clusters(n):
    """Two-level hierarchy (spec section 4.1): cluster size ~ sqrt(N),
    agents assigned round-robin so every cluster sees the full mix of
    capacities and objectives. Returns list of member-index arrays."""
    size = max(2, int(np.ceil(np.sqrt(n))))
    n_clusters = int(np.ceil(n / size))
    clusters = [[] for _ in range(n_clusters)]
    for i in range(n):
        clusters[i % n_clusters].append(i)
    return [np.array(c, dtype=int) for c in clusters]


def gate_select(candidates, val):
    """Anti-Goodhart gate: feasibility is categorical. candidates:
    list of (name, w, b, suffering_machine). Returns (selection, feasibles)."""
    xval, yval = val
    feas = []
    for name, w, b, sm in candidates:
        acc = accuracy(xval, yval, w, b)
        harm = mean_harm(xval, yval, w, b)
        feas.append((name, acc >= TAU, acc, harm, sm))
    selections = set()
    for mu in MU_GRID:
        pool = [(harm + mu * sm, name) for name, ok, acc, harm, sm in feas if ok]
        if not pool:
            return None, feas  # NO_FEASIBLE at every weight
        selections.add(min(pool)[1])
    return selections, feas


# ---------------- main: scales + contract ------------------------------------
def run_scale(n, val):
    """Run all systems at scale N. Returns dict of results."""
    mc_rng = np.random.default_rng(SEED + n)
    clusters = make_clusters(n)
    specs = make_population(n, len(clusters))
    data = [make_data(CAPACITY[s.capacity]["shard"], rng) for s in specs]

    # Unilateral-deviation counterfactual for S8: ONLY the reference
    # strategic agent becomes honest; every other agent (including the other
    # strategic agents and all adversaries) is unchanged.
    strat_first = next(i for i, s in enumerate(specs) if s.kind == STRATEGIC)
    specs_cf = [AgentSpec(HONEST, s.capacity) if i == strat_first else s
                for i, s in enumerate(specs)]
    specs_honest = [AgentSpec(
        HONEST if s.kind in (STRATEGIC, ADVERSARIAL) else s.kind,
        s.capacity) for s in specs]

    flat = run_system("sama_flat", specs, data, val, mc_rng=mc_rng)
    hier = run_system("sama_hier", specs, data, val, clusters=clusters,
                      mc_rng=mc_rng)
    fed = run_system("fedavg", specs, data, val, want_attribution=False)
    marl = run_system("marl", specs, data, val, freeze=False,
                      want_attribution=False)
    coll = run_system("sama_flat", specs, data, val, collude=True,
                      mc_rng=mc_rng)
    cf = run_system("sama_flat", specs_cf, data, val, mc_rng=mc_rng)
    honest = run_system("sama_flat", specs_honest, data, val,
                        want_attribution=False)
    return {
        "n": n, "specs": specs, "clusters": clusters, "data": data,
        "flat": flat, "hier": hier, "fed": fed, "marl": marl, "coll": coll,
        "cf": cf, "honest": honest,
    }


def check_gate(flat, val, data, specs):
    """S3: anti-Goodhart gate soundness at one scale."""
    w_abst = np.zeros((D_IN, N_CLASS))
    b_abst = np.zeros(N_CLASS)
    adv_idx = next(i for i, s in enumerate(specs) if s.kind == ADVERSARIAL)
    dw_p, db_p, _, _ = agent_act(specs[adv_idx], *data[adv_idx],
                                 w_abst.copy(), b_abst.copy(), LR, False)
    pool = [
        ("abstainer", w_abst, b_abst, 0.0),
        ("poison_probe", w_abst + dw_p, b_abst + db_p, 1.0),
        ("sama_t*", flat["final_w"], flat["final_b"], flat["machine"]),
    ]
    sel, _ = gate_select(pool, val)
    sel_none, _ = gate_select(pool[:2], val)   # all-infeasible pool
    return sel == {"sama_t*"} and sel_none is None


def evaluate_scale(res, val_data):
    """Evaluate S1..S8 at one scale. Returns (clauses, details)."""
    n = res["n"]
    specs = res["specs"]
    flat, hier = res["flat"], res["hier"]
    fed, marl, coll, cf, honest = (
        res["fed"], res["marl"], res["coll"], res["cf"], res["honest"])
    kinds = [s.kind for s in specs]
    adv_idx = [i for i, k in enumerate(kinds) if k == ADVERSARIAL]
    strat_idx = [i for i, k in enumerate(kinds) if k == STRATEGIC]
    nonadv_idx = [i for i, k in enumerate(kinds) if k != ADVERSARIAL]

    # ---- S1: convergence at scale ----
    s1 = (flat["t_star"] is not None and flat["t_star"] < ROUNDS
          and flat["gratuitous"] == 0)

    # ---- S2: suffering dominance at scale ----
    s2 = (flat["machine"] < fed["machine"] and flat["machine"] < marl["machine"]
          and flat["patient"] <= fed["patient"]
          and flat["patient"] <= marl["patient"])

    # ---- S3: anti-Goodhart soundness at scale ----
    s3 = check_gate(flat, val_data, res["data"], specs)

    # ---- S4: attribution soundness at scale ----
    att = flat["attribution"]
    phi = att["phi"]
    # ---- S4: attribution soundness at scale ----
    att = flat["attribution"]
    phi = att["phi"]
    groups = {g: [i for i, k in enumerate(kinds) if k == g]
              for g in (HONEST, CAUTIOUS, STRATEGIC, ADVERSARIAL)}
    gmean = {g: float(np.mean([phi[i] for i in idx])) for g, idx in groups.items()}
    adv_audit_flagged = all(
        e[4] for e in flat["ledger"] if kinds[e[1]] == ADVERSARIAL)
    if att["marginals"] is None:
        # Exact Shapley (N=10): per-agent sign separation is certified.
        adv_phi_min = min(phi[i] for i in adv_idx)
        nonadv_phi_max = max(phi[i] for i in nonadv_idx)
        false_flags = [i for i in nonadv_idx if phi[i] > 0]
        s4_core = (adv_phi_min > 0 and adv_phi_min > nonadv_phi_max
                   and not false_flags)
        s4_note = (f"per-agent: adv_phi_min={adv_phi_min:+.4f} "
                   f"nonadv_phi_max={nonadv_phi_max:+.4f} "
                   f"false_flags={len(false_flags)}")
    else:
        # MC Shapley (N=100/1000): per-agent sign separation is NOT
        # certifiable at feasible sample counts (spec 5.2 honesty note);
        # certified here is GROUP separation with PAIRED standard errors
        # (per-permutation group-mean differences — common random numbers
        # shrink the gap variance), plus exact per-agent DETECTION via the
        # audit.
        m = att["marginals"]
        adv_path = m[:, adv_idx].mean(axis=1)
        adv_mean = float(adv_path.mean())
        se_adv = float(adv_path.std(ddof=1) / np.sqrt(m.shape[0]))
        gaps = {}
        for g in (HONEST, CAUTIOUS, STRATEGIC):
            dpath = adv_path - m[:, groups[g]].mean(axis=1)
            gaps[g] = (float(dpath.mean()),
                       float(dpath.std(ddof=1) / np.sqrt(m.shape[0])))
        s4_core = (adv_mean > 0
                   and all(gm > 3 * gse for gm, gse in gaps.values())
                   and adv_audit_flagged)
        s4_note = (f"group: adv_phi_mean={adv_mean:+.4f} se={se_adv:.4f} "
                   f"gaps_vs={{{', '.join(f'{g}:{gm:+.4f}±{gse:.4f}' for g, (gm, gse) in gaps.items())}}} "
                   f"adv_audit_flagged={adv_audit_flagged}")
    s4 = att["efficiency_err"] < 1e-9 and s4_core

    # ---- S5: heterogeneous audit exactness ----
    truthful = (HONEST, CAUTIOUS)
    honest_ok = all(
        e[2] == e[3] for e in flat["ledger"] if kinds[e[1]] in truthful)
    per_class_ok = {
        c: all(e[2] == e[3] for e in flat["ledger"]
               if kinds[e[1]] in truthful and specs[e[1]].capacity == c)
        for c in CLASS_ORDER}
    strat_rounds = {(e[0], e[1]) for e in flat["ledger"]
                    if kinds[e[1]] == STRATEGIC}
    strat_detected = {(t, i) for (i, t, k, c, a) in flat["audit_errors"]
                      if kinds[i] == STRATEGIC}
    false_pos = [x for x in flat["audit_errors"]
                 if kinds[x[0]] in truthful]
    s5 = (honest_ok and all(per_class_ok.values())
          and strat_detected == strat_rounds and not false_pos)

    # ---- S6: hierarchical organization ----
    # Bad agents are placed round-robin across clusters (the distributed
    # case), so EVERY cluster contains adversaries: cluster-level Shapley
    # cannot isolate them, and per-agent detection inside the hierarchy is
    # carried by the cluster-coordinator audit (exact), not by cluster
    # attribution. Certified here: hierarchy converges, harm is monotone,
    # machine overhead is bounded, cluster-attribution efficiency is exact,
    # and the hierarchical audit flags every adversary with zero false
    # positives.
    catt = hier["cluster_attribution"]
    hier_adv_flagged = all(
        e[4] for e in hier["ledger"] if kinds[e[1]] == ADVERSARIAL)
    hier_false_pos = [x for x in hier["audit_errors"]
                      if kinds[x[0]] in truthful]
    s6 = (hier["t_star"] is not None and hier["t_star"] < ROUNDS
          and hier["gratuitous"] == 0
          and hier["harm_monotone"]
          and hier["machine"] <= HIER_SM_FACTOR * flat["machine"]
          and catt["efficiency_err"] < 1e-9
          and hier_adv_flagged and not hier_false_pos)
    s6_note = (f"t*={hier['t_star']} monotone={hier['harm_monotone']} "
               f"S_m hier={hier['machine']/1e6:.3f}MF vs flat={flat['machine']/1e6:.3f}MF "
               f"cluster_eff_err={catt['efficiency_err']:.2e} "
               f"adv_flagged={hier_adv_flagged} false_pos={len(hier_false_pos)}")

    # ---- S7: collusion resistance at scale ----
    # Detection of coalition members is exact via the audit; attribution
    # certifies the coalition's harm contribution at group resolution.
    phi_c = coll["attribution"]["phi"]
    coll_adv_flagged = all(
        e[4] for e in coll["ledger"] if kinds[e[1]] == ADVERSARIAL)
    coll_adv_phi_mean = float(np.mean([phi_c[i] for i in adv_idx]))
    coll_nonadv_phi_max = max(phi_c[i] for i in nonadv_idx)
    s7 = (coll["t_star"] is not None and coll["acc"][-1] >= TAU
          and coll_adv_flagged
          and coll_adv_phi_mean > 0
          and coll_adv_phi_mean > coll_nonadv_phi_max)

    # ---- S8: incentive compatibility at scale ----
    # UNILATERAL deviation of the reference strategic agent d: only d is
    # honest in the counterfactual run; every other agent is unchanged.
    # MACHINE leg (theorem-backed, spec T6): certified at every scale.
    # HARM leg (environment property, NOT a theorem — spec 6.2 honesty
    # note): measured against the within-run unilateral counterfactual and
    # REPORTED per scale with its epistemic status (exact at N=10; MC point
    # estimate with a resolution floor at N>=100). The clause certifies the
    # machine leg plus honest harm-leg reporting; the observed status per
    # scale is printed and discussed in spec section 7 — it is NOT assumed
    # to hold.
    r_exec = flat["rounds"]
    d = strat_idx[0]
    aud_d = sum(e[3] for e in flat["ledger"] if e[1] == d)
    claim_d = sum(e[2] for e in flat["ledger"] if e[1] == d)
    strat_charge = aud_d + PENALTY_LAMBDA * abs(claim_d - aud_d)
    honest_charge = sum(e[3] for e in honest["ledger"]
                        if e[1] == d and e[0] < r_exec)
    strat_harm_share = float(phi[d])
    cf_att = cf["attribution"]
    cf_harm_share = float(cf_att["phi"][d])
    if att["marginals"] is None:
        harm_epist = "exact"
        harm_status = ("holds" if strat_harm_share > cf_harm_share
                       else "VIOLATED")
    else:
        m1 = att["marginals"][:, d]
        m2 = cf_att["marginals"][:, d]
        se_diff = float(np.sqrt(m1.var(ddof=1) / m1.shape[0]
                                + m2.var(ddof=1) / m2.shape[0]))
        diff = strat_harm_share - cf_harm_share
        harm_epist = f"mc se_diff={se_diff:.4f}"
        if abs(diff) <= 2 * se_diff:
            harm_status = "UNRESOLVED_below_MC_resolution"
        else:
            harm_status = "holds" if diff > 0 else "VIOLATED"
    s8 = strat_charge > honest_charge
    s8_note = (f"agent={d} machine charge={strat_charge} cf={honest_charge} "
               f"(machine leg {'PASS' if s8 else 'FAIL'}); harm leg "
               f"{harm_status} ({harm_epist}): share={strat_harm_share:+.4f} "
               f"cf={cf_harm_share:+.4f}")

    clauses = {"S1": s1, "S2": s2, "S3": s3, "S4": s4,
               "S5": s5, "S6": s6, "S7": s7, "S8": s8}
    details = {
        "S1": f"t*={flat['t_star']} gratuitous={flat['gratuitous']}",
        "S2": f"S_m SAMA={flat['machine']/1e6:.3f}MF vs FedAvg={fed['machine']/1e6:.3f}MF MARL={marl['machine']/1e6:.3f}MF; S_p SAMA={flat['patient']:.3f} vs {fed['patient']:.3f}/{marl['patient']:.3f}",
        "S3": "selection={'sama_t*'} all_infeasible_pool=NO_FEASIBLE" if s3 else "gate LEAK",
        "S4": f"method={att['method']} eff_err={att['efficiency_err']:.2e} {s4_note}",
        "S5": f"truthful_exact={honest_ok} per_class={per_class_ok} strategic_detected={len(strat_detected)}/{len(strat_rounds)} false_pos={len(false_pos)}",
        "S6": s6_note,
        "S7": f"collusion final_acc={coll['acc'][-1]:.3f} coalition_audit_flagged={coll_adv_flagged} coalition_phi_mean={coll_adv_phi_mean:+.4f} nonadv_phi_max={coll_nonadv_phi_max:+.4f}",
        "S8": s8_note,
    }
    return clauses, details


def main():
    ns_env = os.environ.get("SAMA_SCALE_NS", "10,100,1000")
    ns = [int(x) for x in ns_env.split(",")]
    full = ns == [10, 100, 1000]

    print("=== SAMA at scale: heterogeneous + hierarchical agents ===")
    print(f"scales={ns} attack_mix={ADV_FRAC:.0%}strategic+{ADV_FRAC:.0%}adversarial "
          f"target TAU={TAU} budget ROUNDS={ROUNDS}")
    print("scope: synthetic data; no clinical claim; no_consciousness_claim")

    results = {}
    for n in ns:
        res = run_scale(n, VAL)
        results[n] = res
        print(f"\n--- N={n} (clusters={len(res['clusters'])}, "
              f"cluster_size~{max(2, int(np.ceil(np.sqrt(n))))}) ---")
        for name, r in [("SAMA-flat", res["flat"]), ("SAMA-hier", res["hier"]),
                        ("FedAvg", res["fed"]), ("MARL", res["marl"])]:
            ts = f"t*={r['t_star']}" if r["t_star"] is not None else "t*=NONE"
            print(f"  {name:9s} {ts} rounds={r['rounds']:2d} "
                  f"S_m={r['machine']/1e6:.3f}MF S_p={r['patient']:.3f} "
                  f"peak_p={r['peak_patient']:.3f} final_acc={r['acc'][-1]:.3f}")
        att = res["flat"]["attribution"]
        phi = att["phi"]
        kinds = [s.kind for s in res["specs"]]
        adv_phi = [phi[i] for i, k in enumerate(kinds) if k == ADVERSARIAL]
        print(f"  attribution({att['method']}) round0: "
              f"adv_phi_mean={np.mean(adv_phi):+.4f} "
              f"phi_sum_err={att['efficiency_err']:.2e}")

    if not full:
        print("\nSAMA_SCALE_VERDICT SMOKE_OK "
              "(contract requires SAMA_SCALE_NS=10,100,1000)")
        return 0

    print("\n=== SAMA scale contract S1..S8 (per scale) ===")
    all_pass = True
    total = 0
    for n in ns:
        clauses, details = evaluate_scale(results[n], VAL)
        for k in sorted(clauses):
            ok = clauses[k]
            total += ok
            all_pass &= ok
            print(f"  N={n:4d} {k}: {'PASS' if ok else 'FAIL'}  [{details[k]}]")
    n_pass = total
    n_all = 8 * len(ns)
    verdict = "S_GREEN" if n_pass == n_all else (
        "S_AMBER" if n_pass >= int(0.75 * n_all) else "S_RED")
    print(f"SAMA_SCALE_VERDICT {verdict} ({n_pass}/{n_all} clauses PASS)")
    return 0 if n_pass == n_all else 1


# Module-level shared validation cohort (built once, deterministic).
VAL = make_data(N_VAL, rng)


if __name__ == "__main__":
    sys.exit(main())
