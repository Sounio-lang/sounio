<!-- docs:meta
topic_id: repo.docs.research.suffering-aware-multi-agent-scale-spec-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.suffering-aware-multi-agent-scale-spec-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — SAMA at scale: 10, 100, and 1000 heterogeneous, hierarchically organized agents

**Date:** 2026-07-31
**Branch:** research/self-falsifying-compilation-line-20260726
**Status:** `EXECUTABLE` — contract S1..S8, `SAMA_SCALE_VERDICT S_GREEN (24/24)`
**Harness:** `scripts/research/suffering_aware_multi_agent_scale.py`
**Gate:** `scripts/ci/suffering_aware_multi_agent_scale_gate.sh` (**SAMA_SCALE_GATE_OK**)
**Parent:** `docs/research/suffering_aware_multi_agent_spec_2026-07-30.md`
(N=5 reference; collective suffering ledger; audited metering; categorical
anti-Goodhart gate; median robust aggregation; exact Shapley burden
attribution; contract G1..G8)

> **Scope.** All data, patients, and suffering values in this document are
> **synthetic constructions**. This is not medical guidance, not a treatment
> recommendation, and not a clinical decision-support tool. The "machine
> suffering" channel is an **operational computational-burden proxy**
> (metered FLOPs): this work makes **no claim of machine consciousness,
> sentience, or phenomenology**, and no result below depends on one.

---

## 1. The question

The N=5 SAMA reference certified the architecture — collective suffering
ledger, audit, anti-Goodhart gate, median aggregation, Shapley attribution —
on one synthetic task at one size, with homogeneous agents and a flat
coordinator. This document asks the scale-out question, three orders of
magnitude up, under the two complications that real multi-agent systems add:

1. **Does the architecture work at N = 10, 100, 1000?** Measured, per
   scale: machine suffering (audited metered FLOPs), integrated and peak
   patient harm, time-to-feasibility, anti-Goodhart soundness, attribution
   soundness, and incentive compatibility — for SAMA-flat,
   SAMA-hierarchical, FedAvg, and MARL under the SAME attack mix (20%
   strategic + 20% adversarial, matching the 1-of-5-each mix of the
   reference).
2. **Does it survive heterogeneity?** Agents differ in capacity (compute
   class) and in objective (honest / cautious / strategic / adversarial).
3. **Does it survive hierarchical organization?** A two-level hierarchy
   (cluster coordinators + root) is run alongside the flat architecture at
   every scale, so the cost and benefit of hierarchy are measured, not
   assumed.

The expanded ethics stays at the center: the ledger is still a **pair**
(patient suffering, machine suffering), never scalarized; the
compassion-allocation weight μ stays an explicit decision swept on a
101-point grid; componentwise dominance — where it holds — makes the
ranking independent of μ.

## 2. What changes at scale (and what does not)

Unchanged from the reference: the synthetic dose-band task, the asymmetric
harm matrix, metering constants, τ = 0.8475, the round budget of 40, the
categorical feasibility gate, the coordinate-wise median aggregation rule,
the round guard (a harm-increasing round is rolled back), the settlement
pair, and the k < N/2 minority bound (T2 of the reference — a property of
the median, hence scale-free).

New at scale:

- **Heterogeneous capacities.** Three compute classes: `low` (2 local
  epochs, 400-patient shard), `mid` (5, 800), `high` (8, 1200), assigned
  round-robin. The audit recomputes against the agent's OWN class budget,
  so audited machine suffering stays exact per class (S5).
- **Heterogeneous objectives.** Honest (collective suffering minimizer,
  truthful), cautious (risk-averse: trains its budget but halves its step,
  truthful), strategic (own-machine minimizer: trains 1 epoch, claims its
  full budget), adversarial (harm-maximizer: scaled sign-flip, or targeted
  class-flip when colluding). 20% strategic + 20% adversarial.
- **Hierarchical organization.** Cluster size ≈ √N (3 clusters at N=10,
  10 at N=100, 32 at N=1000); cluster coordinators median-aggregate and
  audit their members; the root median-aggregates the cluster updates.
- **Attribution beyond exactness.** Exact Shapley costs 2^N coalition
  evaluations — feasible at N=10 (1024), not at N=1000. Section 5 gives
  the Monte-Carlo replacement and states precisely what is and is not
  certifiable with it.

## 3. Heterogeneous agents

### 3.1 Capacity classes

Machine suffering remains **metered, not proxied**: `m_i(t) =
epochs_executed_i × n_shard,i × 3(2·D·C + C)` with the agent's own class
parameters. The coordinator's audit scans the agent's own epoch range
`e ∈ {0,…,budget_i}` and identifies the unique e matching the submission
to 1e-9. Because honest and cautious agents claim truthfully, audited ==
claimed for every capacity class in every round (S5); the strategic lie
(an epoch-count mismatch at the same objective) is separated exactly, with
zero false negatives and zero false positives at every scale. Cautious
agents submit 0.5× their honest update, which matches no epoch count; the
audit treats the declared rescaling as an objective, not a misreport, and
falls back to the (truthful) claimed charge — this fallback path is
exercised only by cautious agents and is distinguished from the strategic
lie by the truthfulness of the claim, which the epoch scan verifies.

Placement honesty note: with round-robin cluster assignment, cluster
members are a residue class mod n_clusters, and capacity classes are
assigned round-robin mod 3 — so when n_clusters is divisible by 3 (N=10:
3 clusters) capacity class is perfectly confounded with cluster (each
N=10 cluster is single-class). This is benign for everything certified
here — the audit is per-agent, per-class exactness (S5) is checked on the
flat run, and no per-cluster capacity claim is made — but it is stated so
the N=10 hierarchy is not over-read as capacity-heterogeneous. At N=100
(10 clusters) and N=1000 (32 clusters) every cluster sees every class.

### 3.2 Objective classes

The four objectives span the expanded-ethics space: collective (honest),
risk-averse collective (cautious), own-machine-only (strategic), and
anti-collective (adversarial). The cautious class is new at scale: it tests
that the architecture tolerates honest agents whose updates are NOT
gradient-identical — the median does not require identical honest updates,
only that bad updates are a strict minority.

## 4. Hierarchical organization

### 4.1 Two-level median

Cluster update = coordinate-wise median of member updates; root update =
coordinate-wise median of cluster updates. Cluster-coordinator audits are
the same deterministic recomputation as the flat audit, run per member.

**T5 (hierarchical robustness).** *Claim:* if every cluster contains a
strict minority of adversarial members (k_c < |C|/2 for every cluster C),
then the hierarchical aggregate is coordinate-wise bounded by honest
values, exactly as the flat median is. *Proof:* by the reference's T2,
each cluster median lies at or between two of its honest members' values
per coordinate; hence every cluster median lies within the coordinate-wise
honest range; the median of values within a range stays within that range.
∎ *Remark:* the premise is a placement property. Bad agents are placed
round-robin across clusters in the benchmark (the distributed case);
placement that CONCENTRATES adversaries into a majority of one cluster is
a stronger attack not tested here — with ≥ half the clusters adversary-
majority the root median itself corrupts, the same k < N/2 bound one level
up.

### 4.2 What the hierarchy costs and buys

Measured at every scale (S6): hierarchical SAMA converges (t* < ROUNDS,
gratuitous post-t* suffering exactly 0), accepted-round patient harm is
non-increasing, and its total machine suffering is within 1.5× of flat
(observed: exactly 1.00× at all three scales on this seed — both
architectures froze at the same t*; the bound is the certified property,
the equality is this seed's trajectory). What the hierarchy buys is
operational, not statistical: audits and aggregation are O(N) work
distributed over √N coordinators instead of one. What it costs in
attribution is section 5.3.

## 5. Burden attribution at scale

### 5.1 Monte-Carlo permutation Shapley

At N ≥ 100 the exact 2^N sum is infeasible. The harness uses permutation
sampling: draw P uniform random orderings; along each ordering, every
agent's marginal contribution is the harm change when it joins its
predecessor prefix; φ_i is the average over orderings (P = 64).

**T6 (MC-Shapley soundness).** *Claim (unbiasedness):* the permutation
estimator is unbiased for the Shapley value — under a uniform random
ordering, agent i's predecessor set is a coalition S of size s with
probability exactly s!(N−s−1)!/N!, the Shapley weight; averaging marginal
contributions over orderings therefore estimates the same weighted average
as the exact sum. (Standard result; stated for completeness.) *Claim
(exact efficiency at any P):* `Σ_i φ_i = f(N) − f(∅)` up to float
rounding, for ANY P ≥ 1. *Proof:* along one ordering, the marginal
contributions telescope: `Σ_k [f(P_k) − f(P_{k−1})] = f(N) − f(∅)`;
averaging over orderings preserves the sum exactly. ∎ The harness verifies
`|err| < 1e-9` at every scale (observed 0.0 to 2.7e-15).

### 5.2 Honesty note: what MC attribution can and cannot certify

Efficiency is exact at any sample count (T6). **Per-agent sign separation
is not.** Individual φ_i at N=1000 have signal ~0.003 and per-agent MC
standard error of the same order at feasible P; certifying "every
adversary's φ > 0, zero false flags" per agent would need thousands of
orderings at N=1000. The contract therefore certifies, at N ≥ 100:

- **group separation with paired standard errors** — the adversarial
  GROUP's mean φ is positive and exceeds every other objective group's
  mean by > 3 paired SE (per-ordering group-mean differences; common
  random numbers shrink the gap variance), and
- **exact per-agent DETECTION by the audit** — every adversary's poisoned
  update matches no honest epoch count and is flagged, zero false
  positives (S4, S5).

At N=10, exact Shapley certifies full per-agent sign separation (S4),
matching the reference's G4. The division of labor at scale is explicit:
the AUDIT carries per-agent detection exactly; SHAPLEY carries burden
quantification at group resolution. This mirrors the reference's own
honesty note that Shapley sign-separation is environment-specific, not a
Shapley-axiom consequence.

### 5.3 Attribution in the hierarchy

With adversaries distributed across ALL clusters (the benchmark's
placement), cluster-level Shapley cannot isolate them — every cluster
contains adversaries, so there is no clean-cluster reference. The
hierarchical contract (S6) therefore certifies cluster-attribution
**efficiency** (exact, 1e-9) and leaves per-agent detection to the
cluster-coordinator audits (exact). A cluster-level sign test would be
vacuous under distributed placement and is not claimed.

## 6. Benchmark and key results

Synthetic dose-band task; attack mix 20% strategic + 20% adversarial at
every scale, identical across systems. S_m = audited machine suffering
(megaFLOPs), S_p = integrated patient harm, t* = first feasible round.

| N | system | t* | rounds | S_m (MF) | S_p | peak_p | final acc |
|---|---|---|---|---|---|---|---|
| 10 | SAMA-flat | 1 | 2 | 7.582 | 0.891 | 0.446 | 0.848 |
| 10 | SAMA-hier | 1 | 2 | 7.582 | 0.890 | 0.455 | 0.849 |
| 10 | FedAvg | — | 40 | 151.632 | 123.440 | 3.758 | 0.069 |
| 10 | MARL | — | 40 | 151.632 | 123.440 | 3.758 | 0.069 |
| 100 | SAMA-flat | 0 | 1 | 46.238 | 0.467 | 0.467 | 0.850 |
| 100 | SAMA-hier | 0 | 1 | 46.238 | 0.462 | 0.462 | 0.849 |
| 100 | FedAvg | — | 40 | 1849.536 | 129.252 | 3.650 | 0.072 |
| 100 | MARL | — | 40 | 1849.536 | 129.252 | 3.650 | 0.072 |
| 1000 | SAMA-flat | 0 | 1 | 467.111 | 0.431 | 0.431 | 0.855 |
| 1000 | SAMA-hier | 0 | 1 | 467.111 | 0.446 | 0.446 | 0.852 |
| 1000 | FedAvg | — | 40 | 18684.432 | 124.676 | 3.682 | 0.075 |
| 1000 | MARL | — | 40 | 18684.432 | 124.676 | 3.682 | 0.075 |

Answers to the scale questions:

- **Does the architecture work at scale?** Yes at all three scales: SAMA
  reaches τ (t* = 1 at N=10, t* = 0 at N ≥ 100 — at scale the first
  median-aggregated round already crosses τ), gratuitous machine suffering
  after t* is exactly 0, and the round guard keeps accepted-round patient
  harm non-increasing (S1, S6).
- **Do the suffering bounds survive?** Yes, componentwise at every scale:
  20×–40× less machine suffering and ~140×–290× less integrated patient
  harm than either baseline under the same attack mix (S2). Componentwise
  dominance ⇒ the ranking holds at every compassion weight μ.
- **Does anti-Goodhart soundness survive?** Yes: at every scale the gate
  selects `sama_t*` at every μ on the 101-point grid, rejects the
  zero-cost abstainer and the cheap poisoned probe, and returns
  NO_FEASIBLE on an all-infeasible pool (S3).
- **Does heterogeneity break the audit?** No: audited == claimed for every
  truthful agent of every capacity class, every strategic misreport
  detected, zero false positives, at every scale (S5).
- **Does hierarchy break anything measured?** Nothing measured: same t*,
  same machine suffering (1.00× flat), monotone harm, exact cluster-level
  attribution efficiency, exact per-agent audit detection (S6).
- **Does collusion scale?** The full 20% adversarial coalition running a
  coordinated targeted class-flip cannot force the accepted model below τ
  at any scale; every member is audit-flagged, and the coalition's mean
  attributed harm is positive and exceeds every non-adversarial agent's φ
  (S7).

## 7. Incentive compatibility at scale — including a measured violation

**Machine leg (theorem-backed, certified at every scale).** The strategic
agent's machine charge `audited + λ·|claim − audited|` with λ = 2 exceeds
its unilateral honest counterfactual charge over the same horizon at all
three scales (observed: 4.212 MF vs 1.123 MF at N=10, where two rounds
were executed; 2.106 MF vs 1.123 MF at N=100 and N=1000, one round. The
reference agent — the first strategic index, agent 2/20/224 at
N=10/100/1000 — is `high` class at all three scales, hence the
1.123 MF = 8 epochs × 1200 samples × 117 FLOPs honest charge). The
reference's T4 proof of this leg is scale-free: it depends only on
`audited ≤ claim` and λ > 1.

**Harm leg (environment property — measured, not assumed).** The harness
compares the reference strategic agent's attributed harm share against a
WITHIN-RUN UNILATERAL counterfactual (only that agent turns honest; the
other strategic agent(s) and all adversaries unchanged):

| N | status | strategic share | counterfactual share | epistemic basis |
|---|---|---|---|---|
| 10 | **VIOLATED** | −0.6170 | −0.5967 | exact Shapley |
| 100 | unresolved | −0.0068 | −0.0641 | MC, below resolution (se_diff 0.0673) |
| 1000 | unresolved | −0.0015 | +0.0003 | MC, below resolution (se_diff 0.0021) |

At N=10 the harm leg is exactly violated: the free-rider's attributed harm
share is slightly MORE harm-reducing than the honest counterfactual's
(−0.6170 < −0.5967). Mechanism: with 2 adversaries among 10 agents,
small coalitions carry real Shapley weight; in even-size coalitions
containing both this agent and an adversary, the two-point median averages
their updates, and a full-strength honest update averaged with a −6×
adversarial flip can land in a MORE harmful spot than a weak 1-epoch
update averaged with the same flip. Free-riding is then (weakly) rewarded
in the harm currency at small scale. At N ≥ 100 the per-agent MC
resolution floor prevents a certified verdict either way; the point
estimates favor the honest counterfactual at N=100 and are ambiguous at
N=1000.

Consequence, stated plainly: **componentwise incentive compatibility does
NOT hold at N=10 in this environment.** What rescues the settlement at
N=10 is that the pair is never scalarized and the machine leg's penalty
(audited + 2× misreport ≈ 3.75× the honest machine charge) makes the
deviation expensive in the currency the strategic agent actually cares
about — its own machine budget is exactly what it was trying to save. The
reference's G8 harm-leg result should be read as N=5-specific, consistent
with its own "environment-specific" caveat. This violation is reported,
not patched: weakening the adversary or re-weighting coalitions until the
leg goes green would be Goodharting the contract.

## 8. Contract (executable certificates)

The harness prints and the gate enforces, at EVERY scale N ∈ {10, 100,
1000}:

- **S1 convergence at scale** — SAMA-flat reaches τ at t* < 40 (t* = 1 at
  N=10, 0 at N ≥ 100); gratuitous machine suffering after t* exactly 0.
- **S2 suffering dominance** — SAMA-flat S_m strictly below FedAvg and
  MARL, S_p ≤ both, componentwise (hence at every μ).
- **S3 anti-Goodhart soundness** — selection is `sama_t*` at every μ on
  the 101-point grid; abstainer and poisoned probe rejected;
  all-infeasible pool returns NO_FEASIBLE.
- **S4 attribution soundness** — efficiency |err| < 1e-9 at every scale.
  N=10 (exact): per-agent sign separation, adv φ_min = +0.2765 > 0 >
  non-adv φ_max = −0.5451, zero false flags. N=100/1000 (MC, P=64):
  adversarial group mean φ > 0 and > 3 paired SE above every other
  group's mean; every adversary audit-flagged.
- **S5 heterogeneous audit exactness** — audited == claimed for every
  honest/cautious agent of every capacity class, every round; strategic
  under-training detected in every round it occurs; zero false positives.
- **S6 hierarchical organization** — SAMA-hier converges with gratuitous
  0; accepted-round harm non-increasing; S_m within 1.5× of flat
  (observed 1.00×); cluster-attribution efficiency < 1e-9; hierarchical
  audit flags every adversary, zero false positives.
- **S7 collusion resistance** — 20%-of-N targeted class-flip coalition
  cannot force the accepted model below τ (final 0.848–0.849); every
  member audit-flagged; coalition mean φ > 0 exceeds every
  non-adversarial φ.
- **S8 incentive compatibility** — machine leg certified at every scale
  (charge 4.212/2.106/2.106 MF > 1.123 MF counterfactual); harm leg
  measured and reported per scale (VIOLATED at N=10, below MC resolution
  at N ≥ 100 — section 7), never assumed.

## 9. Limitations

- All scale results are single-seed (SEED=29) deterministic trajectories;
  t* values and the exact suffering figures are this seed's. The certified
  properties (bounds, exactness, soundness) are structural; the
  trajectory-relative ones are labeled as such.
- The audit assumes verified recomputation (reference §4.2); at N=1000 the
  coordinator-side recompute is O(N·budget²) per round — the hierarchy
  distributes but does not asymptotically reduce this. Deployment needs
  TEEs or proof-of-training.
- Per-agent Shapley sign separation is certified exactly only at N=10;
  at N ≥ 100 detection is the audit's (exact) and Shapley quantifies at
  group resolution (§5.2).
- The hierarchy is tested under DISTRIBUTED adversarial placement; an
  adversary-concentrating placement that captures ≥ half the clusters
  defeats the root median — the same k < N/2 bound one level up (T5
  remark).
- The S8 harm leg fails exactly at N=10 and is below MC resolution at
  N ≥ 100 (§7); the machine leg is the scale-free incentive result.
- At N ≥ 100, t* = 0: the dominance ratios are large partly because the
  baselines never converge under the fixed attack strength. The baseline
  failure is empirical for the pinned attack strength, as in the
  reference.
- All suffering values are synthetic; the machine channel is an
  operational computational-burden proxy with no phenomenological claim.

## 10. Scope guards

Synthetic data only; not medical guidance; no clinical claim; no claim of
machine consciousness, sentience, or phenomenology; the harness prints
`no_consciousness_claim` in every run.
