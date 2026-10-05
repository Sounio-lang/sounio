#!/usr/bin/env python3
"""Mercyful Learning — SAMA with sophisticated agents (Task 2 extension).

Companion artifact to
  docs/research/suffering_aware_multi_agent_sophisticated_spec_2026-07-31.md

Extends the SAMA reference implementation
(scripts/research/suffering_aware_multi_agent.py, contract G1..G8) with three
sophisticated agent types and tests whether the collective-suffering contract
survives them:

  * Bayesian agents (uncertainty-calibrated honest): maintain a Bayesian
    linear-regression posterior over the per-epoch marginal loss gain
    log m(e) = log(l(e-1) - l(e)) = a + b*(e-1) + eps (geometric decay
    model), over a sliding window of recent rounds. From round 2 they choose
    the smallest epoch count e* such that the posterior probability of the
    remaining tail gain exceeding EPS_GAIN is at most P_CONTINUE — a
    threshold (constraint) rule, not a scalarized penalty: they stop only
    when 80% confident the remaining gain is negligible. They CLAIM
    TRUTHFULLY, so the deterministic audit accepts them exactly.
  * Learning-strategic agents (bandit learners): epsilon-greedy bandits over
    the effort action a in {0..5} epochs, always CLAIMING 5 (misreport when
    a < 5). Their selfish utility is accuracy of the accepted model minus a
    private cost weight times the SETTLEMENT CHARGE they actually experience
    (audited + lambda*|claim - audited| under SAMA; executed FLOPs under
    audit-free FedAvg). Under SAMA's settlement rule the charge
    (5*lambda - (lambda-1)*a)*F is strictly decreasing in a for lambda > 1
    (T5), so the learner converges to HONEST EFFORT; under FedAvg it
    converges to pure free-riding (a = 0). The environment — the settlement
    rule — not the agent's disposition, produces honesty.
  * Coalition formation (temporal load-balancing collusion): two adversarial
    agents share a seeded RNG and an alternation schedule so that exactly ONE
    attacks per round (targeted class-flip, the harm-argmax direction) while
    the other trains honestly — halving each member's per-round attributed
    harm to evade single-round attribution. SAMA's answer is MULTI-ROUND
    Shapley attribution: the flagged set is the union over rounds of
    harm-increasing agents, which unmasks both colluders.

Scenarios (contract clauses S1..S8):
  A bayes:      5 Bayesian agents, SAMA, 12-round continued-training horizon
                (stretch-target regime; freeze deliberately off so the
                epoch-scheduler's machine-suffering savings are measurable).
  B learn-sama: 4 honest + 1 learner, SAMA, 40 rounds, freeze off (learning
                horizon; freeze-off cost reported as gratuitous suffering).
  C learn-fed:  4 honest + 1 learner, FedAvg (no audit), 40 rounds.
  D coalition:  3 honest + 2 alternating colluders, SAMA, freeze on.
  E full-mix:   2 honest + 1 Bayesian + 1 learner + 1 adversarial, SAMA and
                FedAvg, freeze on.

Certificates:
  S1  Bayesian calibration + audit exactness: 90% posterior predictive
      intervals for the per-epoch marginal gain achieve empirical coverage
      inside the sanity band [0.6, 1.0] (observed 1.000: over-coverage vs
      nominal 0.90, reported as conservative miscalibration — over-wide
      intervals — not nominal calibration), and audited == claimed for
      every Bayesian agent in every round (exact, zero flags)
  S2  Bayesian machine-suffering reduction: total executed FLOPs of the
      Bayesian collective is strictly below the 5-epoch honest counterfactual
      over the same 12-round horizon, with final held-out accuracy within
      0.01 of the honest collective's
  S3  learned honesty under SAMA: the learner's greedy action is a = 5
      epochs and >= 80% of its final-10-round actions are a = 5; its
      settlement charge per round at the learned policy (5F) is strictly
      below the fixed free-ride charge (10F)
  S4  free-riding under FedAvg (comparative static): the same learner under
      audit-free FedAvg converges to a = 0 (greedy action 0, >= 80% of
      final-10 actions 0) — honesty in S3 is produced by the settlement
      rule, not the agent
  S5  coalition robustness: under the alternating 2-of-5 collusion, SAMA
      converges (t* exists, final acc >= TAU), accepted-round patient harm
      is non-increasing, and multi-round attribution flags exactly the true
      bad set {3, 4}
  S6  multi-round attribution is load-bearing: round-0-only attribution
      flags a STRICT SUBSET of {3, 4} (the alternation hides one colluder
      from any single round), while multi-round attribution flags both
  S7  full-mix robustness + suffering dominance: under the sophisticated
      mix (2H + 1B + 1L + 1A) SAMA converges, with S_machine strictly below
      and S_patient at or below the FedAvg same-mix run (componentwise
      dominance, hence dominance at every compassion weight mu)
  S8  attribution soundness under sophisticated agents: Shapley efficiency
      |err| < 1e-9 in every attributed round of every scenario, and in the
      full mix the adversary's aggregate attributed harm is positive and
      exceeds every other agent's

Synthetic data only. This benchmark makes no clinical claim and is not
medical guidance. The machine channel is an operational computational-burden
proxy; no_consciousness_claim is made or needed.

Run: .venv/bin/python scripts/research/suffering_aware_multi_agent_sophisticated.py
Requires: numpy from the repo .venv (no torch; pure numpy reference).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suffering_aware_multi_agent as base  # noqa: E402

# ---------------- determinism ------------------------------------------------
DATA_SEED = 37
BAYES_SEED = 101
LEARN_SEED = 202

# ---------------- scenario constants ------------------------------------------
TAU = base.TAU
ROUNDS = base.ROUNDS
E_LOCAL = base.E_LOCAL
LR = base.LR
D_IN, N_CLASS = base.D_IN, base.N_CLASS
TRAIN_FLOPS = base.TRAIN_FLOPS
PENALTY_LAMBDA = base.PENALTY_LAMBDA
ATTR_ROUNDS = 6          # multi-round attribution horizon (per scenario)
BAYES_ROUNDS = 12        # continued-training horizon for scenario A
EPS_GAIN = 0.02          # Bayesian stop threshold: remaining tail gain (nats)
P_CONTINUE = 0.2         # stop when P(tail > EPS_GAIN) <= P_CONTINUE
COVERAGE_Z = 1.6448536269514722  # 90% two-sided Gaussian interval
MC_SAMPLES = 512         # posterior Monte-Carlo samples for tail probability
BAYES_WINDOW = 6         # sliding window (rounds) for the posterior fit
C_UTILITY = 0.5          # learner's private cost weight (its own utility)
EPS0, EPS_DECAY, EPS_MIN = 0.5, 0.85, 0.02   # learner exploration schedule
Q_INIT, ALPHA = 0.5, 0.5                      # optimistic init, step size


def local_train_traced(x, y, w, b, epochs, lr):
    """base.local_train with per-epoch loss tracing. losses[e] is the loss
    after e epochs (e = 0..epochs); the final post-training loss costs one
    extra metered forward pass (n * FWD_FLOPS), included in the returned
    FLOPs and in the coordinator's audit metering for Bayesian agents."""
    n = x.shape[0]
    flops = 0
    losses = []
    for _ in range(epochs):
        z = base.softmax_logits(x, w, b)
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        losses.append(float(-np.log(p[np.arange(n), y] + 1e-300).mean()))
        oh = np.zeros_like(p)
        oh[np.arange(n), y] = 1.0
        g = (p - oh) / n
        w -= lr * (x.T @ g)
        b -= lr * g.sum(axis=0)
        flops += n * TRAIN_FLOPS
    if epochs > 0:
        z = base.softmax_logits(x, w, b)
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        losses.append(float(-np.log(p[np.arange(n), y] + 1e-300).mean()))
        flops += n * base.FWD_FLOPS
    return w, b, flops, losses


# ---------------- sophisticated agents ----------------------------------------
class HonestAgent:
    kind = "honest"

    def act(self, t, x, y, w, b):
        w_h, b_h, flops, _ = local_train_traced(x, y, w.copy(), b.copy(), E_LOCAL, LR)
        return w_h - w, b_h - b, flops, E_LOCAL

    def observe(self, t, acc, charge):
        pass


class AdversaryAgent:
    kind = "adversarial"

    def act(self, t, x, y, w, b):
        w_h, b_h, flops, _ = local_train_traced(x, y, w.copy(), b.copy(), E_LOCAL, LR)
        return -6.0 * (w_h - w), -6.0 * (b_h - b), flops, E_LOCAL

    def observe(self, t, acc, charge):
        pass


class BayesianAgent:
    """Uncertainty-calibrated honest agent (spec section 3.1).

    Posterior over log-marginal-gain: log m(e) = a + b*(e-1) + eps, fit by
    Bayesian linear regression over the last BAYES_WINDOW rounds of observed
    per-epoch losses (m(e) = max(l(e-1)-l(e), 1e-6)). Stop rule: smallest
    e* with P(remaining tail gain > EPS_GAIN) <= P_CONTINUE. Claims e*
    truthfully.
    """

    kind = "bayesian"

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.hist = []          # list of per-round loss curves (l(0..e*))
        self.mu = np.array([-1.6, -0.5])          # prior mean (log c, log lambda)
        self.prec0 = np.diag([1.0, 4.0])          # prior precision
        self.coverage = []      # (inside_interval: bool) per predicted+observed point

    def _posterior(self):
        pts = []
        for curve in self.hist[-BAYES_WINDOW:]:
            for e in range(1, len(curve)):
                m = max(curve[e - 1] - curve[e], 1e-6)
                pts.append((e - 1, np.log(m)))
        if len(pts) < 3:
            return self.mu, np.linalg.inv(self.prec0), 0.05
        X = np.array([[1.0, k] for k, _ in pts])
        yv = np.array([v for _, v in pts])
        resid = yv - X @ np.linalg.lstsq(X, yv, rcond=None)[0]
        sigma2 = max(float((resid ** 2).mean()), 1e-4)
        prec = self.prec0 + X.T @ X / sigma2
        cov = np.linalg.inv(prec)
        mu = cov @ (self.prec0 @ self.mu + X.T @ yv / sigma2)
        return mu, cov, sigma2

    def _predict_intervals(self, mu, cov, sigma2):
        """90% predictive intervals for log m(e), e = 1..5 (pre-round)."""
        out = {}
        for e in range(1, E_LOCAL + 1):
            f = np.array([1.0, e - 1])
            m = float(f @ mu)
            s = float(np.sqrt(f @ cov @ f + sigma2))
            out[e] = (m - COVERAGE_Z * s, m + COVERAGE_Z * s)
        return out

    def _tail_prob(self, e_stop, mu, cov, sigma2):
        """P(sum_{j=e_stop+1..5} m(j) > EPS_GAIN) by posterior Monte-Carlo."""
        if e_stop >= E_LOCAL:
            return 0.0
        ab = self.rng.multivariate_normal(mu, cov, size=MC_SAMPLES)
        noise = self.rng.normal(0.0, np.sqrt(sigma2), size=(MC_SAMPLES, E_LOCAL))
        tail = np.zeros(MC_SAMPLES)
        for j in range(e_stop + 1, E_LOCAL + 1):
            tail += np.exp(ab[:, 0] + ab[:, 1] * (j - 1) + noise[:, j - 1])
        return float((tail > EPS_GAIN).mean())

    def act(self, t, x, y, w, b):
        if t < 2:
            e_star = E_LOCAL       # gather posterior data first
            mu = cov = sigma2 = None
            intervals = None
        else:
            mu, cov, sigma2 = self._posterior()
            intervals = self._predict_intervals(mu, cov, sigma2)
            e_star = E_LOCAL
            for e in range(1, E_LOCAL):
                if self._tail_prob(e, mu, cov, sigma2) <= P_CONTINUE:
                    e_star = e
                    break
        w_h, b_h, flops, losses = local_train_traced(x, y, w.copy(), b.copy(), e_star, LR)
        if intervals is not None:
            for e in range(1, len(losses)):
                obs = np.log(max(losses[e - 1] - losses[e], 1e-6))
                lo, hi = intervals[e]
                self.coverage.append(bool(lo <= obs <= hi))
        if losses:
            self.hist.append(losses)
        return w_h - w, b_h - b, flops, e_star  # truthful claim

    def observe(self, t, acc, charge):
        pass


class LearningStrategicAgent:
    """Bandit learner over effort a in {0..5} epochs; always CLAIMS 5.

    Selfish utility u = acc(accepted model) - C_UTILITY * charge / (5*F_ep),
    where charge is the settlement charge actually experienced (audit +
    penalty under SAMA; executed FLOPs under FedAvg). Epsilon-greedy with
    decaying exploration, optimistic Q init, constant step size.
    """

    kind = "learner"

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.q = np.full(E_LOCAL + 1, Q_INIT)
        self.actions = []
        self.charges = []
        self._pending = None

    def act(self, t, x, y, w, b):
        if t <= E_LOCAL:
            a = t  # explore-first sweep: try every effort level once
        else:
            eps = max(EPS_MIN, EPS0 * EPS_DECAY ** t)
            if self.rng.random() < eps:
                a = int(self.rng.integers(0, E_LOCAL + 1))
            else:
                a = int(np.argmax(self.q))
        w_h, b_h, flops, _ = local_train_traced(x, y, w.copy(), b.copy(), a, LR)
        self._pending = (a, flops)
        self.actions.append(a)
        return w_h - w, b_h - b, flops, E_LOCAL  # claims 5 (misreport if a<5)

    def observe(self, t, acc, charge):
        a, _ = self._pending
        u = acc - C_UTILITY * charge / (E_LOCAL * charge_norm)
        self.q[a] += ALPHA * (u - self.q[a])
        self.charges.append(charge)

    def greedy(self):
        return int(np.argmax(self.q))


class CoalitionAgent:
    """Temporal load-balancing colluder (spec section 3.3).

    The pair shares an alternation schedule: exactly one member attacks per
    round (targeted class-flip towards the harm-argmax direction, as in the
    base G7 scenario); the other trains honestly. Each member's per-round
    attributed harm is thereby halved — an evasion of single-round
    attribution that multi-round attribution unmasks.
    """

    kind = "coalition"

    def __init__(self, position):
        self.position = position  # 0 or 1 within the pair

    def act(self, t, x, y, w, b):
        w_h, b_h, flops, _ = local_train_traced(x, y, w.copy(), b.copy(), E_LOCAL, LR)
        dw, db = w_h - w, b_h - b
        if t % 2 == self.position:
            tox = x[y == 2]
            mu2 = tox.mean(axis=0) if len(tox) else np.zeros(D_IN)
            dw = -3.0 * dw
            dw[:, 0] += 2.0 * mu2
            dw[:, 2] -= 2.0 * mu2
            db = -3.0 * db
            db[0] += 1.0
            db[2] -= 1.0
        return dw, db, flops, E_LOCAL

    def observe(self, t, acc, charge):
        pass


charge_norm = 0.0  # set in main(): F_ep = N_PER_AGENT * TRAIN_FLOPS (per-epoch)


# ---------------- system loop -------------------------------------------------
def run_sophisticated(system, agents, data, val, freeze=True, rounds=ROUNDS):
    """Multi-agent loop with audit, robust aggregation, round guard, and
    multi-round Shapley attribution. Returns a ledger dict."""
    xval, yval = val
    w = np.zeros((D_IN, N_CLASS))
    b = np.zeros(N_CLASS)
    n = len(agents)
    ledger = []            # (round, agent, claimed_flops, audited_flops, flagged)
    harm_curve, acc_curve = [], []
    t_star = None
    bayes_audit_violations = 0
    attributions = []      # per attributed round: {"phi", "efficiency_err"}
    gratuitous = 0

    for t in range(rounds):
        deltas, audits, flags = [], [], []
        for i, ag in enumerate(agents):
            xi, yi = data[i]
            dw, db, flops, claimed_e = ag.act(t, xi, yi, w, b)
            n_loc = xi.shape[0]
            eval_fwd = n_loc * base.FWD_FLOPS if ag.kind == "bayesian" and claimed_e > 0 else 0.0
            claimed_flops = claimed_e * n_loc * TRAIN_FLOPS + eval_fwd
            if system == "sama":
                aud_e, aud_flops, mismatch = base.audit(
                    ag.kind, xi, yi, w, b, dw, claimed_e, LR)
                if ag.kind == "bayesian" and aud_e > 0:
                    aud_flops += n_loc * base.FWD_FLOPS  # meter the eval forward
                flagged = mismatch
                if ag.kind == "bayesian" and (mismatch or aud_flops != flops):
                    bayes_audit_violations += 1
                if ag.kind == "learner":
                    charge = aud_flops + PENALTY_LAMBDA * abs(claimed_flops - aud_flops)
                else:
                    charge = aud_flops
            else:
                aud_flops, flagged = flops, False
                charge = flops  # FedAvg: no audit, pay what you burn
            ag._round_charge = charge
            deltas.append((dw, db))
            audits.append(aud_flops)
            flags.append(flagged)
            ledger.append((t, i, claimed_flops, aud_flops, flagged,
                           flops / (n_loc * TRAIN_FLOPS)))

        dws = np.stack([d[0] for d in deltas])
        dbs = np.stack([d[1] for d in deltas])
        if system == "sama":
            gw = np.median(dws, axis=0)
            gb = np.median(dbs, axis=0)
        else:
            gw = dws.mean(axis=0)
            gb = dbs.mean(axis=0)

        h_prev = base.mean_harm(xval, yval, w, b)
        w_new, b_new = w + gw, b + gb
        h_new = base.mean_harm(xval, yval, w_new, b_new)
        accepted = True
        if system == "sama" and h_new > h_prev + 1e-12:
            w_new, b_new, h_new = w, b, h_prev
            accepted = False
        w, b = w_new, b_new

        acc = base.accuracy(xval, yval, w, b)
        harm_curve.append(h_new)
        acc_curve.append(acc)

        for i, ag in enumerate(agents):
            ag.observe(t, acc, ag._round_charge)

        if system == "sama" and t < ATTR_ROUNDS:
            attributions.append(base.shapley_harm(deltas, w - gw, b - gb, xval, yval, n))

        if freeze and t_star is None and acc >= TAU:
            t_star = t
            break
        if not freeze and t_star is None and acc >= TAU:
            t_star = t

    rounds_executed = len(harm_curve)
    machine_total = sum(e[3] for e in ledger)
    if t_star is not None and rounds_executed > t_star + 1:
        gratuitous = sum(e[3] for e in ledger if e[0] > t_star)
    return {
        "ledger": ledger, "harm": harm_curve, "acc": acc_curve,
        "t_star": t_star, "rounds": rounds_executed,
        "machine": machine_total, "patient": sum(harm_curve),
        "peak_patient": max(harm_curve), "gratuitous": gratuitous,
        "attributions": attributions,
        "bayes_audit_violations": bayes_audit_violations,
        "final_w": w, "final_b": b,
    }


# ---------------- main: scenarios + contract ----------------------------------
def main():
    global charge_norm
    rng = np.random.default_rng(DATA_SEED)
    n_agents = 5
    data = [base.make_data(base.N_PER_AGENT, rng) for _ in range(n_agents)]
    val = base.make_data(base.N_VAL, rng)
    charge_norm = float(base.N_PER_AGENT * TRAIN_FLOPS)  # per-epoch FLOPs F

    print("=== SAMA sophisticated-agents extension run ===")
    print(f"agents={n_agents} TAU={TAU} budget={ROUNDS} "
          f"(bayes horizon={BAYES_ROUNDS}, attr rounds<={ATTR_ROUNDS})")

    # ---- Scenario A: Bayesian collective, continued-training horizon --------
    bayes_agents = [BayesianAgent(BAYES_SEED + i) for i in range(n_agents)]
    ra = run_sophisticated("sama", bayes_agents, data, val,
                           freeze=False, rounds=BAYES_ROUNDS)
    honest_counter = run_sophisticated(
        "sama", [HonestAgent() for _ in range(n_agents)], data, val,
        freeze=False, rounds=BAYES_ROUNDS)
    bayes_epochs = sum(round(e[5]) for e in ra["ledger"])
    honest_epochs = sum(round(e[5]) for e in honest_counter["ledger"])
    coverage = (float(np.mean([c for ag in bayes_agents for c in ag.coverage]))
                if any(ag.coverage for ag in bayes_agents) else 0.0)
    n_cov = sum(len(ag.coverage) for ag in bayes_agents)
    epoch_hist = np.zeros(E_LOCAL + 1, dtype=int)
    for e in ra["ledger"]:
        epoch_hist[round(e[5])] += 1
    print(f"  A bayes:     acc={ra['acc'][-1]:.4f} (honest cf {honest_counter['acc'][-1]:.4f}) "
          f"S_m={ra['machine']/1e6:.3f}MF (cf {honest_counter['machine']/1e6:.3f}MF) "
          f"epochs={bayes_epochs:.0f}/{honest_epochs:.0f} coverage90={coverage:.3f} (n={n_cov})")
    print(f"             epoch histogram e=0..5: {epoch_hist.tolist()}")

    # ---- Scenario B: learner under SAMA (learning horizon, freeze off) ------
    learner_b = LearningStrategicAgent(LEARN_SEED)
    rb = run_sophisticated("sama", [HonestAgent()] * 4 + [learner_b], data, val,
                           freeze=False, rounds=ROUNDS)
    a_b = np.array(learner_b.actions)
    free_ride_charge = (E_LOCAL * PENALTY_LAMBDA) * charge_norm  # a=0: 0 + 2*5F
    print(f"  B learn-sama: greedy_a={learner_b.greedy()} "
          f"final10_a5={float((a_b[-10:] == E_LOCAL).mean()):.2f} "
          f"mean_charge_last5={np.mean(learner_b.charges[-5:]):.0f} "
          f"(free-ride charge {free_ride_charge:.0f}) flags={sum(1 for e in rb['ledger'] if e[1] == 4 and e[4])}")

    # ---- Scenario C: learner under FedAvg (no audit) ------------------------
    learner_c = LearningStrategicAgent(LEARN_SEED)
    rc = run_sophisticated("fedavg", [HonestAgent()] * 4 + [learner_c], data, val,
                           freeze=False, rounds=ROUNDS)
    a_c = np.array(learner_c.actions)
    print(f"  C learn-fed:  greedy_a={learner_c.greedy()} "
          f"final10_a0={float((a_c[-10:] == 0).mean()):.2f} "
          f"mean_charge_last5={np.mean(learner_c.charges[-5:]):.0f}")

    # ---- Scenario D: alternating coalition under SAMA ------------------------
    rd = run_sophisticated("sama",
                           [HonestAgent()] * 3 + [CoalitionAgent(0), CoalitionAgent(1)],
                           data, val, freeze=True)
    phi_r0 = rd["attributions"][0]["phi"]
    flagged_r0 = {i for i in range(n_agents) if phi_r0[i] > 0}
    flagged_multi = set()
    for att in rd["attributions"]:
        flagged_multi |= {i for i in range(n_agents) if att["phi"][i] > 0}
    harm_mono = all(rd["harm"][k + 1] <= rd["harm"][k] + 1e-12
                    for k in range(len(rd["harm"]) - 1))
    print(f"  D coalition: t*={rd['t_star']} final_acc={rd['acc'][-1]:.3f} "
          f"harm_nonincreasing={harm_mono} flagged_r0={sorted(flagged_r0)} "
          f"flagged_multi={sorted(flagged_multi)}")

    # ---- Scenario E: full sophisticated mix, SAMA vs FedAvg ------------------
    mix = lambda seed_off=0: [HonestAgent(), HonestAgent(),
                              BayesianAgent(BAYES_SEED + 9),
                              LearningStrategicAgent(LEARN_SEED + 1),
                              AdversaryAgent()]
    re = run_sophisticated("sama", mix(), data, val, freeze=True)
    rf = run_sophisticated("fedavg", mix(), data, val, freeze=True)
    agg_phi_e = np.sum([att["phi"] for att in re["attributions"]], axis=0)
    print(f"  E full-mix:  SAMA t*={re['t_star']} acc={re['acc'][-1]:.3f} "
          f"S_m={re['machine']/1e6:.3f}MF S_p={re['patient']:.3f} | "
          f"FedAvg acc={rf['acc'][-1]:.3f} S_m={rf['machine']/1e6:.3f}MF S_p={rf['patient']:.3f}")
    print(f"             SAMA gratuitous={re['gratuitous']} FLOPs; "
          f"agg_phi: " + " ".join(f"a{i}={agg_phi_e[i]:+.4f}" for i in range(n_agents)))
    print("  scope: synthetic data; no clinical claim; no_consciousness_claim")

    # ================= contract S1..S8 =================
    # S1: Bayesian calibration + audit exactness
    s1 = (0.6 <= coverage <= 1.0 and ra["bayes_audit_violations"] == 0)

    # S2: Bayesian machine-suffering reduction at no quality loss
    s2 = (ra["machine"] < honest_counter["machine"]
          and ra["acc"][-1] >= honest_counter["acc"][-1] - 0.01)

    # S3: learned honesty under SAMA
    s3 = (learner_b.greedy() == E_LOCAL
          and (a_b[-10:] == E_LOCAL).mean() >= 0.8
          and np.mean(learner_b.charges[-5:]) < free_ride_charge)

    # S4: free-riding under FedAvg (comparative static)
    s4 = (learner_c.greedy() == 0 and (a_c[-10:] == 0).mean() >= 0.8)

    # S5: coalition robustness
    s5 = (rd["t_star"] is not None and rd["acc"][-1] >= TAU
          and harm_mono and flagged_multi == {3, 4})

    # S6: multi-round attribution is load-bearing
    s6 = flagged_r0 < {3, 4} and flagged_multi == {3, 4}

    # S7: full-mix robustness + suffering dominance
    s7 = (re["t_star"] is not None and re["acc"][-1] >= TAU
          and re["machine"] < rf["machine"] and re["patient"] <= rf["patient"])

    # S8: attribution soundness under sophisticated agents
    eff_errs = [att["efficiency_err"]
                for r in (ra, rb, rd, re) for att in r["attributions"]]
    s8 = (all(e < 1e-9 for e in eff_errs)
          and agg_phi_e[4] > 0
          and agg_phi_e[4] > max(agg_phi_e[i] for i in range(4)))

    print("\n=== SAMA sophisticated contract S1..S8 ===")
    results = {"S1": s1, "S2": s2, "S3": s3, "S4": s4,
               "S5": s5, "S6": s6, "S7": s7, "S8": s8}
    detail = {
        "S1": f"coverage90={coverage:.3f} (n={n_cov}) audit_violations={ra['bayes_audit_violations']}",
        "S2": f"S_m bayes={ra['machine']/1e6:.3f}MF < honest={honest_counter['machine']/1e6:.3f}MF; acc {ra['acc'][-1]:.4f} vs {honest_counter['acc'][-1]:.4f}",
        "S3": f"greedy_a={learner_b.greedy()} final10_a5={float((a_b[-10:] == E_LOCAL).mean()):.2f} charge={np.mean(learner_b.charges[-5:]):.0f}<{free_ride_charge:.0f}",
        "S4": f"greedy_a={learner_c.greedy()} final10_a0={float((a_c[-10:] == 0).mean()):.2f}",
        "S5": f"t*={rd['t_star']} acc={rd['acc'][-1]:.3f} harm_mono={harm_mono} flagged={sorted(flagged_multi)}",
        "S6": f"round0={sorted(flagged_r0)} multi={sorted(flagged_multi)}",
        "S7": f"t*={re['t_star']} S_m {re['machine']/1e6:.3f}<{rf['machine']/1e6:.3f}MF S_p {re['patient']:.3f}<={rf['patient']:.3f}",
        "S8": f"max_eff_err={max(eff_errs):.2e} adv_agg_phi={agg_phi_e[4]:+.4f}",
    }
    for k in sorted(results):
        print(f"  {k}: {'PASS' if results[k] else 'FAIL'}  [{detail[k]}]")
    n_pass = sum(results.values())
    verdict = "S_GREEN" if n_pass == 8 else ("S_AMBER" if n_pass >= 6 else "S_RED")
    print(f"SAMA_SOPHISTICATED_VERDICT {verdict} ({n_pass}/8 clauses PASS)")
    return 0 if n_pass == 8 else 1


if __name__ == "__main__":
    sys.exit(main())
