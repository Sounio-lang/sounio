<!-- docs:meta
topic_id: repo.docs.research.suffering-aware-multi-agent-sophisticated-spec-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.suffering-aware-multi-agent-sophisticated-spec-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — SAMA with sophisticated agents: Bayesian uncertainty, learned strategic behavior, and coalition formation under the collective-suffering contract

**Date:** 2026-07-31
**Branch:** research/self-falsifying-compilation-line-20260726
**Status:** `EXECUTABLE` — contract S1..S8, `SAMA_SOPHISTICATED_VERDICT S_GREEN (8/8)`
**Harness:** `scripts/research/suffering_aware_multi_agent_sophisticated.py`
**Gate:** `scripts/ci/suffering_aware_multi_agent_sophisticated_gate.sh` (**SAMA_SOPHISTICATED_GATE_OK**)
**Parent:** `docs/research/suffering_aware_multi_agent_spec_2026-07-30.md`
(SAMA base system; contract G1..G8; audit, median aggregation, round guard,
single-round Shapley attribution, settlement pair)

> **Scope.** All data, patients, and suffering values in this document are
> **synthetic constructions**. This is not medical guidance, not a treatment
> recommendation, and not a clinical decision-support tool. The "machine
> suffering" channel is an **operational computational-burden proxy**
> (metered FLOPs): this work makes **no claim of machine consciousness,
> sentience, or phenomenology**, and no result below depends on one.

---

## 1. Position: does the mercyful society survive smarter members?

The base SAMA contract (G1..G8) was certified against simple fixed-policy
defectors: a strategic agent that always free-rides and always lies, and an
adversarial agent that always attacks with a fixed scaled sign-flip. Fixed
policies are the easy case for a settlement-and-attribution design: the
misreporter never adapts to the penalty, and the colluders never try to hide.
This extension asks the harder question — is the collective-suffering
contract robust to agents that **reason under uncertainty**, **learn** their
strategy against the settlement rule itself, and **coordinate** to evade
attribution?

Three sophisticated agent types (§3):

1. **Bayesian agents** maintain a posterior over the marginal value of one
   more training epoch and stop when the remaining gain is *confidently*
   negligible — a threshold rule on posterior probability, not a scalarized
   cost-benefit penalty. They claim truthfully; the audit must accept them
   exactly, and their uncertainty calibration is measured, not assumed.
2. **Learning-strategic agents** are bandit learners over the effort action,
   always claiming full effort. They experience the settlement charge and
   adapt. The settlement rule, not the agent's disposition, determines what
   they learn: under SAMA the charge is strictly decreasing in effort (T5),
   so a selfish learner converges to **honest effort**; under audit-free
   FedAvg the same learner converges to pure free-riding.
3. **Coalition formation**: two adversarial agents share an alternation
   schedule — exactly one attacks per round while the other trains honestly —
   halving each member's per-round attributed harm to evade single-round
   attribution. SAMA's answer is **multi-round Shapley attribution** (§6):
   the flagged set is the union over rounds of harm-increasing agents.

The design rule of the lineage is unchanged: **constraints and gates, not
penalties in the objective** — with the same single deliberate exception, the
settlement charge on the ledger (who pays), never a term in the training
objective (what gets optimized). The two suffering channels remain
non-scalarized: every comparison below is componentwise, so it holds at every
compassion-allocation weight μ.

## 2. The environment

Identical to the base spec (§2): the synthetic dose-band federation, `N = 5`,
800-patient shards, 1000-patient cohort-in-waiting, asymmetric harm matrix
`H`, `τ = 0.8475`, 40-round budget, `E = 5` honest local epochs, misreport
penalty `λ = 2`. The base harness is imported unchanged; this extension adds
agent classes, a multi-round attribution loop, and five scenarios:

| scenario | mix | system | freeze | horizon |
|---|---|---|---|---|
| A bayes | 5 Bayesian | SAMA | off | 12 rounds (continued-training) |
| B learn-sama | 4 honest + 1 learner | SAMA | off | 40 rounds (learning horizon) |
| C learn-fed | 4 honest + 1 learner | FedAvg (no audit) | off | 40 rounds |
| D coalition | 3 honest + 2 alternating colluders | SAMA | on | budget |
| E full-mix | 2 honest + 1 Bayesian + 1 learner + 1 adversarial | SAMA / FedAvg | on | budget |

Scenarios A and B deliberately run with freeze-on-green **off**: the Bayesian
epoch-scheduler's machine-suffering savings and the learner's adaptation both
require a horizon beyond the (very fast) convergence time. The freeze-off
cost is reported as gratuitous suffering and is a deliberate experiment cost,
not a deployment recommendation — deployment keeps freeze-on-green (G2).

## 3. Sophisticated agents

### 3.1 Bayesian agents (uncertainty-calibrated honest)

Each Bayesian agent records its local per-epoch training-loss curve
`l(0..e*)` every round (a byproduct of training, plus **one extra metered
forward pass** for the post-training loss `l(e*)`, charged at
`n_local · FWD_FLOPs` and included in both the agent's claim and the
coordinator's audit metering — the audit stays exact). It models the
per-epoch marginal gain `m(e) = max(l(e−1) − l(e), 1e-6)` as geometrically
decaying:

```
log m(e) = a + b·(e−1) + ε ,   ε ~ N(0, σ²)
```

fit by Bayesian linear regression over a sliding window of the last 6 rounds
(conjugate normal posterior with fixed prior; `σ²` from residual variance,
floored at 1e-4). From round 2, the agent picks the smallest epoch count
`e*` such that

```
P( Σ_{j > e*} m(j) > EPS_GAIN ) ≤ P_CONTINUE   (EPS_GAIN = 0.02 nats, P_CONTINUE = 0.2)
```

by posterior Monte-Carlo (512 samples): it stops only when **80% confident**
the remaining 5-epoch gain is negligible. This is a threshold rule — a
constraint on posterior probability — not a penalty traded against reward.
The uncertainty is load-bearing: early rounds have wide posteriors (little
data, cross-round drift), so the tail probability stays high and the agent
trains on; as the posterior tightens and the gains decay, it stops earlier.
The agent **claims e* truthfully**, so the deterministic audit accepts it
exactly (S1). Calibration is measured and reported honestly: the 90%
posterior predictive intervals for `log m(e)` achieve empirical coverage
1.000 on n = 225 predicted-and-observed points. Under exact 90% calibration
one would expect 22.5 ± 4.5 misses; observing **zero** misses is a ~5σ
over-coverage event, i.e. the intervals are **over-wide** — conservative
miscalibration driven by cross-round drift inflating the noise estimate
`σ²`, not nominal calibration. The contract band [0.6, 1.0] is a sanity
band; the over-coverage direction is reported, per the lineage honesty
convention.

On scenario A the Bayesian collective executes 275 epoch-equivalents vs 300
for the 5-epoch honest counterfactual over the same 12-round horizon
(`S_machine` 27.612 MF < 28.080 MF) at **equal** final held-out accuracy
(0.8460 = 0.8460): strictly less machine suffering, no quality loss (S2).
Honesty note: the savings are modest (−8%) because the stop rule is
conservative by design and the honest baseline is already cheap per round;
the certificate is the *strict* componentwise improvement at equal accuracy,
not a large factor.

### 3.2 Learning-strategic agents (bandits against the settlement rule)

The learner chooses effort `a ∈ {0,…,5}` epochs per round and **always
claims 5** (misreport whenever `a < 5`). Its private selfish utility is

```
u(a) = acc(accepted model) − C_UTILITY · charge(a) / (5F) ,   C_UTILITY = 0.5
```

where `F = n_local · TRAIN_FLOPs` is one epoch's cost and `charge` is what
the agent actually experiences: the settlement charge
`audited + λ·|claim − audited|` under SAMA, or executed FLOPs under
audit-free FedAvg. (The scalarization lives **inside the defecting agent's
private utility** — a model of selfishness — never in the system's ledger,
gate, or objective, which remain non-scalarized.) Learning is an
epsilon-greedy bandit: an explore-first sweep of all 6 actions in rounds
0–5, then `ε_t = max(0.02, 0.5·0.85^t)`, constant step size `α = 0.5`.

**Theorem T5 (the settlement rule makes honesty the learned optimum).**
*Claim:* under SAMA, the strategic agent's settlement charge after `a`
executed epochs with a fixed claim of `E` is

```
charge(a) = a·F + λ·(E − a)·F = (λE − (λ−1)a)·F ,
```

strictly **decreasing** in `a` for `λ > 1`. *Proof:* `d/da charge =
−(λ−1)·F < 0`. ∎ With `λ = 2, E = 5`: `charge(a) = (10 − a)·F`, minimized
uniquely at `a = 5` — full honest effort — where the misreport term
vanishes. *Corollary (SAMA side):* for any private utility
`u(a) = acc(a) − c·charge(a)/(5F)` with `c > 0`, if `acc` is weakly
increasing in `a` then `u` is **strictly** increasing in `a` — both terms
push the same way — so `u` is uniquely maximized at `a = E` for **every**
`c > 0`; no "small accuracy stake" condition is needed on this side.
*Claim (FedAvg side):* under audit-free FedAvg the charge is `a·F`, so
`u(a) = acc(a) − c·a/5`; this is maximized at `a = 0` **iff** the per-epoch
accuracy gain the agent gets from its own effort is below `c/5` throughout —
a genuine tradeoff, not an identity. On this task one agent of five moves
the accepted accuracy by ≲ 0.002 per own-epoch (observed spread ≲ 0.01
across `a`), well under `c/5 = 0.1`, so the optimum is `a = 0`; the
direction is certified behaviorally by S4, not claimed universally.
*Honesty note:* the corollary is a property of the charge landscape a
learner descends, certified behaviorally by S3/S4 on this seed; it is not a
claim about every possible learner.

Measured: under SAMA (scenario B) the learner's greedy action is `a = 5`
with 100% of its final-10 actions at full effort, and its realized per-round
charge at the learned policy is 468 kFLOP = 5F vs the fixed free-ride charge
936 kFLOP = 10F — deviation stopped paying, so the learner stopped deviating
(S3). Under FedAvg (scenario C) the identical learner converges to `a = 0`
with 100% of final-10 actions at zero effort, charge 0 (S4). **The
environment — the settlement rule — not the agent, produces honesty.** This
is T4's incentive-compatibility clause (G8) made adaptive: the penalty does
not merely make a fixed deviation unprofitable, it makes *learning to
deviate* converge to honest behavior.

### 3.3 Coalition formation (temporal load-balancing collusion)

The base G7 colluders both attack every round. Sophisticated colluders do
worse: they **share an alternation schedule** so exactly one member attacks
per round (targeted class-flip towards the harm-argmax direction, as in G7)
while the other trains honestly. Each member's per-round attributed harm is
halved, and — the point of the attack — **round-0-only attribution flags
just one of them** (the round-0 attacker). This is an evasion of the
attribution layer, not of aggregation: the median bound (base T2) is about
how many bad updates exist per round, and the alternating coalition still
injects only `k = 1 < 5/2` bad update per round.

## 4. Collective suffering ledger and audit — unchanged, and load-bearing

The ledger, deterministic audit, median aggregation, categorical gate, and
round guard are inherited unchanged from the base system. This extension
verifies they absorb the new agent types without modification:

- **Bayesian agents** claim truthfully; audited == claimed in every round
  (exact, zero flags) — including the metered eval forward (S1).
- **Learners** are audited every round; every under-training round is
  flagged (6 flags during exploration in scenario B, then none — the flags
  stop when learning converges to honesty).
- **Alternating colluders** are aggregation-neutralized exactly as the base
  adversary: scenario D converges at t* = 2, final 0.849 ≥ τ, and the round
  guard keeps accepted-round patient harm non-increasing (S5).

## 5. Anti-Goodhart gating — unchanged

The categorical feasibility gate and the median's minority-coalition bound
(base T2) are unaffected by agent sophistication: feasibility is a property
of the checkpoint, and the median bound is about the per-round count of bad
updates, not their intelligence. Scenario E certifies the full sophisticated
mix (2 honest + 1 Bayesian + 1 learner still in its exploration sweep + 1
adversarial): SAMA converges (t* = 1, final 0.849 ≥ τ) with gratuitous
machine suffering exactly 0, while same-mix FedAvg collapses (0.055) — and
the suffering comparison is componentwise (S7):

| system (mix E) | t* | S_machine | S_patient | final acc |
|---|---|---|---|---|
| SAMA | 1 | 3.900 MF | 0.878 | 0.849 |
| FedAvg | — | 82.742 MF | 145.608 | 0.055 |

Componentwise dominance ⇒ dominance at every compassion weight μ; no weight
choice is load-bearing. (SAMA's S_machine is below the base attack-mix run's
5.897 MF because t* = 1 here — the learner's early small updates happen to
accelerate the first feasible round on this seed; the certificate is the
componentwise comparison against the same-mix baseline, not the absolute
number.)

## 6. Multi-round burden attribution

Single-round Shapley attribution (base §6.1) flags the agents whose
attributed harm contribution in *that* round is positive. Against temporal
load-balancing this is evadable: each colluder is innocent-looking on its
off rounds. The extension attributes harm **every round** (for the first
`ATTR_ROUNDS = 6` accepted rounds, exact `2^N` Shapley per round) and flags
the **union**: an agent is flagged iff its attributed contribution is
positive in at least one attributed round.

**Theorem T6 (coalition neutrality, restated from base T2).** *Claim:* with
`k < N/2` bad updates per round, no coalition — adaptive, alternating, or
fixed — can force an accepted update outside the coordinate-wise range
`[h_min, h_max]` of the honest updates. *Proof:* base T2; for odd `N` the
median is order statistic `(N+1)/2`, and pushing it outside the honest range
would require at least `(N+1)/2 > k` bad values on one side. The alternating
schedule changes *which* agent is bad each round, never the per-round count
`k`. ∎ *Clarification (from review):* the bound is **containment**, not
honesty of the median — with `N = 5, k = 1` the median can coincide with an
adversarial value that happens to lie between two honest values (e.g.
honest {1,2,4,5}, bad {3} → median 3); what the coalition cannot do is push
the accepted coordinate outside `[h_min, h_max]`.

**Theorem T7 (multi-round attribution soundness).** *Claim (efficiency):*
per-round Shapley efficiency `Σ_i φ_i(t) = f_t(N) − f_t(∅)` holds exactly
for every attributed round (Shapley's theorem), and by linearity of the
Shapley value the per-round sums equal the Shapley values of the aggregate
game `F(S) = Σ_t f_t(S)`; the harness verifies `|err| < 1e-9` in every
attributed round of every scenario (observed max 8.9e-16). *Claim
(unmasking, this environment):* under the alternation schedule, each
coalition member attacks on a disjoint subset of rounds, so for any
attribution horizon `H ≥ 2` covering both parities the union-flagged set
contains both members; round-0 attribution alone flags only the round-0
attacker. Certified by S5/S6: round-0 flags `{3}`, multi-round flags
`{3, 4}` = true bad set. *Honesty notes:* (i) as in base §6.2, sign
separation of adversaries is a verified property of this synthetic coalition
function, not a consequence of the Shapley axioms; (ii) the union-flag rule
is a binary threshold at `φ > 0` — it does not distinguish tiny from massive
positive contributions, and honest agents can in principle receive a
positive `φ` through harmful interactions with adversarial updates, so its
zero-false-positive status here is a certified environment property
(observed: honest `φ` are strongly negative every attributed round), not a
theorem; (iii) an alternating coalition attacking *below* the
harm-detection floor every round would evade both rules — the certificate
covers the G7-strength targeted class-flip family.

## 7. Benchmark and key results

Answers to the task's three questions:

- **Is the system robust to Bayesian (uncertainty-driven) agents?** Yes —
  and it *profits* from them: truthful claims keep the audit exact,
  interval coverage is inside the sanity band (1.000 on n = 225 —
  over-wide, reported as conservative miscalibration), and the
  collective pays strictly less machine suffering at equal accuracy
  (27.612 < 28.080 MF, S1/S2).
- **Is it robust to agents that learn to be strategic?** Yes — robustness
  here is a *mechanism* result, not just a withstand result: the settlement
  charge is strictly decreasing in effort (T5), so a selfish bandit learner
  converges to honest effort under SAMA (greedy `a = 5`, charge 5F < 10F)
  while the identical learner under audit-free FedAvg converges to pure
  free-riding (`a = 0`) (S3/S4).
- **Is it robust to coalition formation?** Yes, for minority coalitions:
  the alternating 2-of-5 collusion cannot force the model below τ (0.849 ≥
  0.8475), accepted-round harm is non-increasing, and multi-round
  attribution unmasks both members — round-0 attribution alone catches only
  one, which is exactly why the multi-round mechanism is load-bearing
  (S5/S6).

Machine suffering, patient harm, and anti-Goodhart soundness under the full
sophisticated mix: `S_machine` 3.900 MF vs 82.742 MF (21× less than same-mix
FedAvg), `S_patient` 0.878 vs 145.608 (166× less), gratuitous machine
suffering exactly 0, Shapley efficiency error ≤ 8.9e-16, adversary's
aggregate attributed harm +2.0264 > 0 exceeding every other agent's (S7/S8).

## 8. Theorems

T5, T6, T7 are stated in §3.2, §6. **T8 (Bayesian stop rule, empirical
character).** *Statement:* the stop rule is a posterior-probability
threshold (constraint), never a scalarized trade against reward; its
predictive intervals are **over-wide** on this task (empirical coverage
1.000 vs nominal 0.90 on n = 225 — a ~5σ over-coverage event, reported as
conservative miscalibration from cross-round drift inflating `σ²`, within
the sanity band [0.6, 1.0]); and its executed cost — including the metered
eval forward — is accepted exactly by the deterministic audit. *Honesty
note:* no optimality claim is made for EPS_GAIN/P_CONTINUE; they are design
constants of the agent, and the contract certifies the *outcome* (strict
machine-suffering reduction at equal accuracy, exact audit), not the
tuning.

## 9. Contract (executable certificates)

The harness prints and the gate enforces:

- **S1 Bayesian calibration + audit exactness** — 90% predictive-interval
  coverage 1.000 (n = 225) inside the sanity band [0.6, 1.0]; the
  over-coverage (~5σ vs nominal 0.90) is reported as conservative
  miscalibration (over-wide intervals), not nominal calibration; audited ==
  claimed for every Bayesian agent, every round (0 violations).
- **S2 Bayesian machine-suffering reduction** — 275 < 300 epoch-equivalents
  (27.612 MF < 28.080 MF) over the same horizon; final accuracy 0.8460 vs
  0.8460 (within 0.01).
- **S3 learned honesty under SAMA** — greedy action `a = 5`; 100% of
  final-10 actions at `a = 5`; learned-policy charge 468 kFLOP < free-ride
  charge 936 kFLOP.
- **S4 free-riding under FedAvg** — the same learner: greedy action `a = 0`;
  100% of final-10 actions at `a = 0`.
- **S5 coalition robustness** — t* = 2, final 0.849 ≥ τ, accepted-round harm
  non-increasing, multi-round flagged set == {3, 4}.
- **S6 multi-round attribution is load-bearing** — round-0 flags {3} ⊊
  {3, 4}; multi-round flags {3, 4}.
- **S7 full-mix robustness + dominance** — t* = 1, 0.849 ≥ τ; S_machine
  3.900 < 82.742 MF and S_patient 0.878 ≤ 145.608 vs same-mix FedAvg.
- **S8 attribution soundness** — Shapley efficiency |err| < 1e-9 in every
  attributed round of every scenario (max 8.9e-16); adversary's aggregate
  φ = +2.0264 > 0, exceeding every other agent's.

## 10. Limitations

- Bayesian savings are modest (−8% machine suffering) and EPS_GAIN-dependent
  (T8); the certificate is strict improvement at equal accuracy, not a large
  factor.
- The learner result is behavioral (one bandit family, one seed): T5 is a
  property of the charge landscape, certified by S3/S4 for this learner, not
  for every possible learner.
- Multi-round attribution unmasks alternating colluders of G7 attack
  strength; sub-floor attacks would evade any finite-horizon attribution
  (T7 honesty note).
- Scenarios A and B run freeze-off by design (horizon needed for scheduling
  and learning); the gratuitous-suffering guarantee (G2) is a freeze-on
  property and is re-certified under freeze in scenarios D and E.
- The audit still assumes verified recomputation (base §4.2); all suffering
  values are synthetic; the machine channel is an operational
  computational-burden proxy with no phenomenological claim.

## 11. Scope guards

Synthetic data only; not medical guidance; no clinical claim; no claim of
machine consciousness, sentience, or phenomenology; the harness prints
`no_consciousness_claim` in every run.
