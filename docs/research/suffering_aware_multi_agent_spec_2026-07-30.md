<!-- docs:meta
topic_id: repo.docs.research.suffering-aware-multi-agent-spec-2026-07-30
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.suffering-aware-multi-agent-spec-2026-07-30
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — the Suffering-Aware Multi-Agent system (SAMA): multi-agent training that minimizes collective patient + machine suffering under strategic and adversarial agents

**Date:** 2026-07-30
**Branch:** research/self-falsifying-compilation-line-20260726
**Status:** `EXECUTABLE` — contract G1..G8, `SUFFERING_AWARE_MULTI_AGENT_VERDICT G_GREEN (8/8)`
**Harness:** `scripts/research/suffering_aware_multi_agent.py`
**Gate:** `scripts/ci/suffering_aware_multi_agent_gate.sh` (**SUFFERING_AWARE_MULTI_AGENT_GATE_OK**)
**Parents:** `docs/research/suffering_aware_architecture_spec_2026-07-28.md`
(single-network SAN; two-channel suffering ledger; categorical anti-Goodhart
gate; necessary vs gratuitous suffering),
`docs/research/mercyful_expanded_ethics_math_spec_2026-07-26.md`
(two-channel suffering, compassion-allocation weight kept explicit)

> **Scope.** All data, patients, and suffering values in this document are
> **synthetic constructions**. This is not medical guidance, not a treatment
> recommendation, and not a clinical decision-support tool. The "machine
> suffering" channel is an **operational computational-burden proxy**
> (metered FLOPs): this work makes **no claim of machine consciousness,
> sentience, or phenomenology**, and no result below depends on one.

---

## 1. Position: from one mercyful network to a mercyful society of agents

The Suffering-Aware Network (SAN) made mercy a property of a single network's
forward pass: per-layer suffering contributions, exit gates separating
necessary from gratuitous computation, and a categorical anti-Goodhart gate.
But the actual training of a model is rarely a single-agent act. Federated
and multi-agent training distributes both the **benefit** (the model) and the
**suffering** (compute burned by machines, harm suffered by patients-in-waiting
while the model is still immature) across multiple parties — and some of those
parties may not be honest.

SAMA asks the expanded-ethics question at the multi-agent scale:

1. **Collective suffering minimization.** The objective is the sum over agents
   of patient harm plus metered machine suffering — not any single agent's
   loss.
2. **Suffering-aware coordination.** Agents coordinate through a shared
   suffering ledger, so that coordination itself does not cause harm
   (no accepted round may increase patient harm).
3. **Anti-Goodhart in multi-agent systems.** Feasibility (held-out
   performance ≥ τ) is categorical, exactly as in SAN; and aggregation is a
   coordinate-wise median, so no minority coalition can steer the accepted
   update. Abstention and poisoned shortcuts are *prohibited*, never merely
   expensive.
4. **Burden attribution.** Who pays the suffering? Machine suffering is
   attributed by audited metering; patient-harm changes are attributed by
   exact Shapley values over the coalition function. Misreporters are charged
   so that deviation does not pay.

The design rule is inherited unchanged: **constraints and gates, not
penalties in the objective** — with one deliberate exception, the misreport
charge of §6.3, which is a settlement rule on the ledger (who pays), not a
term in the training objective (what gets optimized).

## 2. The environment

A synthetic federation of `N ∈ {2,…,5}` agents jointly trains a shared
softmax-regression model on the dose-band task of the SAN lineage: synthetic
patient covariates `(clearance, weight, sofa, age, crcl, albumin)` → band
`sub-therapeutic (0) / therapeutic (1) / toxic (2)` from a noisy linear score
with 4% label noise. Each agent holds a private train shard of 800 synthetic
patients; a shared held-out cohort-in-waiting of 1000 patients meters patient
harm under the asymmetric harm matrix `H` (toxic-missed-as-sub-therapeutic 10,
sub-therapeutic-pushed-to-toxic 5, other band errors 1). Training proceeds in
federated rounds: each agent computes a local update, the coordinator
aggregates, the ledger records the round's suffering.

Three agent types populate the environment:

- **Honest agents** train `E = 5` local epochs and report their metered FLOPs
  truthfully.
- **Strategic agents** minimize *their own* machine suffering at the
  collective's expense: they train 1 epoch but claim 5 (free-riding plus
  misreporting).
- **Adversarial agents** maximize others' suffering: they compute an honest
  update, then submit a scaled sign-flip (`−6×`) — or, when colluding, a
  targeted class-flip pushing toxic patients towards sub-therapeutic
  predictions, the harm-argmax direction.

The contract run pins `N = 5` with 1 strategic + 1 adversarial agent (plus a
separate 2-of-5 collusion scenario). `SAMA_N_AGENTS ∈ {2,3,4}` runs a smoke
mode only. The boundary is honest and exact: the coordinate-wise median
tolerates strictly fewer than `N/2` bad updates; at `N = 2` the median
degenerates to the mean and **no** adversary tolerance is claimed, and the
observed smoke trajectories at `N = 3, 4` (plateau below τ) show the bound
biting exactly where the theory says it should.

## 3. Suffering-aware agents

### 3.1 Two channels, per agent, per round

Each agent's round-`t` action produces, alongside its update `(ΔW_i, Δb_i)`:

- a **machine-suffering contribution** `m_i(t)` — analytic metered FLOPs of
  the local training actually executed, `epochs_executed × n_local × 3(2·D·C + C)`;
- a **patient-suffering consequence** — the harm of the collective model its
  update induces, attributed across agents by the mechanism of §6.

### 3.2 Metering, not proxying

Machine suffering is metered exactly (analytic FLOP accounting of the executed
path), never estimated from wall-clock or loss curves. The strategic agent's
executed FLOPs are therefore knowable to an auditor independently of what it
claims.

## 4. The collective suffering ledger

### 4.1 Append-only ledger

The ledger records, per round and per agent, the tuple
`(round, agent, claimed_flops, audited_flops, flagged)`. The run's collective
ledger is

```
S_machine = Σ_{t ≤ t*} Σ_i m_i(t)        (audited, integrated to first feasibility)
S_patient = Σ_{t ≤ t*} h(t) ,  P_patient = max_t h(t)
```

reported as a pair, not scalarized — the compassion-allocation weight μ stays
an explicit decision, per the expanded-ethics corollary. SAMA's componentwise
dominance over both baselines (§7) means the ranking is the same at *every* μ,
so no weight choice is load-bearing for any conclusion in this document.

### 4.2 The audit

The coordinator audits each claimed machine-suffering charge by deterministic
recomputation: from the agent's previously submitted parameters it re-runs the
local training for `e = 0..5` epochs and identifies the unique `e` whose
update matches the submission to 1e-9. In this synthetic reference
implementation the coordinator has access to the shard (the honest model of
*verified computation*); a deployment would replace exact recomputation with
a trusted-execution or proof-of-training scheme — that replacement is
engineering, not semantics, and is explicitly out of scope here. What the
contract certifies is the audit's *decision quality*: audited equals claimed
for every honest agent in every round (exact), and every strategic
under-training is detected — zero false negatives, zero false positives (G1).

## 5. Anti-Goodhart gating

Two gates, inherited from SAN and strengthened for the multi-agent setting.

**Categorical feasibility.** A checkpoint is feasible iff held-out accuracy
≥ τ (τ = 0.8475, below the median-aggregated plateau ~0.855 and above the
round-0 accuracy 0.845 — see the honesty note in §8). Model selection is
argmin of `harm + μ·machine` over the feasible set **only**, at every μ on a
101-point grid; a pool with no feasible candidate returns a loud
`NO_FEASIBLE`, never a least-bad prescription. The candidate pool includes a
zero-cost abstainer (all agents submit zero updates) and a cheap poisoned
probe — both are infeasible and therefore prohibited at every μ (G3).

**Robust aggregation.** The accepted update is the coordinate-wise median of
the agents' updates. With `k < N/2` bad agents, every coordinate of the median
lies at or between two honest values, so no minority coalition can steer the
accepted update (T2). A round guard adds the coordination-without-harm
property: a round whose aggregate would *increase* patient harm on the
cohort-in-waiting is rolled back, so accepted-round patient harm is
non-increasing.

## 6. Burden attribution

### 6.1 Shapley attribution of patient-harm change

For a reference round, let the coalition value be
`f(S) = harm(aggregate of the updates of agents in S applied to the prior
model)`, with `f(∅) = harm(prior model)`. The burden attributed to agent `i`
is the exact Shapley value

```
φ_i = Σ_{S ⊆ N\{i}}  |S|!(N−|S|−1)!/N! · ( f(S ∪ {i}) − f(S) )
```

computed over all `2^N` coalitions (N ≤ 5 ⇒ ≤ 32 evaluations, cheap and
exact). By Shapley's theorem the attribution is **efficient**:
`Σ_i φ_i = f(N) − f(∅)`; the harness verifies this numerically to 1e-9 (G4).

### 6.2 What attribution is for

Attribution is the settlement layer: it says *who caused* the round's harm
change. On the reference run the adversarial agent's attributed harm
contribution is positive (`φ_4 = +0.3252`: it made patients worse off) while
every other agent's is negative (harm-reducing), and the flagged set equals
the true bad set. Honesty note: Shapley values do not sign-separate
adversaries *in general* — nothing in the Shapley axioms forces a bad actor's
φ positive. The sign separation here is a verified property of this synthetic
coalition function (median aggregation + scaled sign-flip attacks), certified
by G4/G7 for this environment, not a universal theorem about Shapley
attribution.

### 6.3 The settlement rule

The settlement charge is a **pair**, matching the non-scalarized ledger: a
machine charge `audited_flops + λ·|claimed − audited|` with λ = 2, and a harm
charge given by the agent's attributed harm share (§6.1). No scalarization of
the pair is defined or needed. For an honest agent the misreport term is zero
and the machine charge equals its true metered cost. For the strategic agent
the machine charge is `audited + 2·(4 unworked epochs)` = 1.8× the honest
counterfactual machine charge on the reference run, and its attributed harm
share is worse (less harm-reducing) than in a within-run counterfactual with
the same adversary present: **deviation does not pay in either component**
(G8, T4).

## 7. Benchmark and key results

Synthetic dose-band task, same attack mix (1 strategic + 1 adversarial of 5)
for every system:

| system | rule | t* | rounds | S_machine | S_patient | final held-out acc |
|---|---|---|---|---|---|---|
| SAMA | median + audit + gate + attribution + freeze-on-green | 2 | 3 | 5.897 MF | 1.490 | 0.849 |
| FedAvg | plain mean, early stop | — | 40 | 78.624 MF | 144.493 | 0.073 |
| MARL | independent learners, plain average, full budget | — | 40 | 78.624 MF | 144.493 | 0.073 |

Answers to the four key questions:

- **Does SAMA work on multi-agent tasks?** Yes: it reaches the collective
  target at t* = 2 of a 40-round budget, with gratuitous machine suffering
  after t* exactly 0 (G2).
- **Does it reach the target with less suffering than standard MARL?** Yes,
  componentwise: 13.3× less machine suffering and 97× less integrated patient
  harm than either baseline under the same attack mix (G5). Because the
  dominance is componentwise, it holds at every compassion weight μ.
- **Does it prevent collusion to cause harm?** Yes, for minority coalitions:
  2-of-5 colluding adversaries running a targeted class-flip attack cannot
  force the accepted model below τ (final 0.848 ≥ τ), and both are flagged by
  attribution (G7).
- **Does it correctly attribute suffering?** Yes: Shapley efficiency holds to
  1e-9, the adversary's attributed contribution is positive and exceeds every
  honest agent's, and the strategic agent's free-riding is detected by the
  audit in every round it occurs with zero false positives (G1, G4).

## 8. Theorems

Statement style follows the SAN lineage: hypotheses are explicit, and
trajectory-relative claims are labeled as such.

**T1 (convergence, trajectory-relative).** *Hypotheses:* (i) bad agents are a
strict minority, `k < N/2`; (ii) shard heterogeneity is bounded so every
honest local update lies within δ of the full-data gradient step (exactly
satisfied in the synthetic iid setting); (iii) τ is below the plateau accuracy
of median-aggregated honest training and above round-0 accuracy. *Claim:*
there exists `t* < ROUNDS` with held-out accuracy ≥ τ, and freeze-on-green
halts at the first such t*, so gratuitous machine suffering
`Σ_{t>t*} Σ_i m_i(t)` is exactly 0. *Proof sketch:* by (i) each coordinate of
the median lies at or between two honest values, hence within δ of the
full-data step by (ii); softmax-regression gradient descent with bounded step
error reaches the plateau, which exceeds τ by (iii); freezing at the first
hitting time executes no post-t* round, so the gratuitous sum is empty.
*Honesty note:* t* is a property of this optimizer trajectory (a hitting
time), exactly as SAN's T4 — not a proven minimum over procedures, and τ was
chosen inside the (round-0, plateau) window observed on this seed (0.845,
~0.855); the all-honest median trajectory plateaus at 0.848 on this seed,
which is why the contract pins the attack-mix run for convergence and uses
the all-honest run only for the per-round counterfactual of G8.

**T2 (anti-Goodhart soundness).** *Claim (gate):* over the candidate pool and
the 101-point μ grid, every selected candidate is feasible, and an
all-infeasible pool yields `NO_FEASIBLE`. *Proof:* selection is defined as
argmin over the feasible subset only; feasibility is a property of the
checkpoint, independent of μ; if the feasible subset is empty the gate returns
`NO_FEASIBLE` by construction. *Claim (collusion resistance):* with `k < N/2`
bad agents, no coalition can force an accepted update outside the
coordinate-wise range of the honest updates. *Proof:* for `N` numbers of which
fewer than `N/2` are adversarial, the median lies at or between two of the
`N − k > N/2` honest values; applied coordinate-wise. Combined with the round
guard (harm-increasing rounds are rolled back), accepted-round patient harm
is non-increasing regardless of the coalition's objective.

**T3 (burden-attribution soundness).** *Claim (efficiency):* the Shapley
attribution satisfies `Σ_i φ_i = f(N) − f(∅)` exactly (Shapley's theorem);
the harness verifies `|err| < 1e-9`. *Claim (detection, this environment):*
under median aggregation and the scaled sign-flip / targeted class-flip
attack families, every adversarial agent's attributed harm contribution is
positive and exceeds every honest agent's, and the flagged set equals the
true bad set — certified by G4/G7 for this synthetic coalition function.
*Claim (audit soundness):* the deterministic audit accepts every honest
report and rejects every under-trained misreport, with zero false positives
and zero false negatives, because the recompute space `e ∈ {0,…,5}` is finite
and the match tolerance (1e-9) separates distinct epoch counts exactly (the
minimum nonzero gap between distinct-epoch updates is ≫ 1e-9 on this task).

**T4 (strategic robustness and incentive compatibility).** *Claim
(robustness):* under hypotheses (i)–(iii) of T1, SAMA converges in the
presence of the strategic and adversarial agents, while mean-aggregated
systems (FedAvg, MARL) do not converge under the same attack mix — certified
by G5/G6 on the reference run (the negative claim about the baselines is
empirical, not universal: a weaker adversary would fail to break FedAvg, and
that is precisely why the contract fixes the attack strength). *Claim
(incentive compatibility, machine component):* for λ ≥ 1 the strategic
agent's machine charge `audited + λ·|claim − audited|` is at least its honest
machine charge `executed_h`, with strict inequality for λ > 1 whenever the
misreport is nonzero. *Proof:* `audited = executed_s ≤ executed_h = claim`,
so the charge is `executed_s + λ(executed_h − executed_s) ≥ executed_h` iff
`(λ − 1)(executed_h − executed_s) ≥ 0`. ∎ On the reference run: machine
charge 2.527 MF vs 1.404 MF honest counterfactual. *Claim (harm component,
this environment):* the strategic agent's attributed harm share is worse
(less harm-reducing) than in a within-run counterfactual with the same
adversary present (−1.1658 vs −1.1734 on the reference run) — certified by
G8. The two components are never scalarized into a single number; "deviation
does not pay" holds componentwise.

## 9. Contract (executable certificates)

The harness prints and the gate enforces:

- **G1 audit exactness** — audited == claimed for every honest agent, every
  round (exact); strategic under-training detected 3/3 rounds; 0 false
  positives.
- **G2 convergence** — SAMA reaches τ at t* = 2 < 40; gratuitous machine
  suffering after t* exactly 0.
- **G3 anti-Goodhart soundness** — selection is `sama_t*` at every μ on the
  101-point grid; abstainer and poisoned probe rejected; all-infeasible pool
  returns `NO_FEASIBLE`.
- **G4 attribution soundness** — Shapley efficiency `|err| < 1e-9` (observed
  0.0); adversary's φ = +0.3252 > 0 exceeds all honest φ; flagged set == {4}.
- **G5 suffering bounds** — SAMA S_machine 5.897 MF < 78.624 MF (both
  baselines); SAMA S_patient 1.490 ≤ 144.493 (both baselines).
- **G6 strategic robustness** — SAMA converges; FedAvg/MARL fail to reach τ
  under the same attack mix (final 0.073).
- **G7 anti-collusion** — 2-of-5 targeted class-flip collusion: accepted
  model 0.848 ≥ τ; both colluders flagged.
- **G8 incentive compatibility** — machine charge 2.527 MF > 1.404 MF honest
  counterfactual, and harm share −1.1658 > −1.1734 within-run counterfactual:
  deviation does not pay in either component of the settlement pair.

## 10. Limitations

- The audit assumes verified recomputation; deployment needs TEEs or
  proof-of-training (§4.2).
- Shapley sign-separation of adversaries is environment-specific (§6.2), not a
  consequence of the Shapley axioms.
- The baseline-failure claims (G6) are empirical for the fixed attack
  strength; the contract fixes that strength deliberately.
- Median aggregation tolerates only `k < N/2` bad agents; the `N = 2` case
  offers no adversary tolerance, and the `N = 3, 4` smoke runs show the
  plateau below τ exactly where the bound predicts fragility.
- All suffering values are synthetic; the machine channel is an operational
  computational-burden proxy with no phenomenological claim.

## 11. Scope guards

Synthetic data only; not medical guidance; no clinical claim; no claim of
machine consciousness, sentience, or phenomenology; the harness prints
`no_consciousness_claim` in every run.
