<!-- docs:meta
topic_id: repo.docs.research.suffering-aware-architecture-robustness-spec-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.suffering-aware-architecture-robustness-spec-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — SAN robustness validation: cross-validation, sensitivity analysis, and adversarial stress of the Suffering-Aware neural Network

**Date:** 2026-07-31
**Branch:** research/zd-fiber-antisymmetry-lemma-20260731
**Status:** `EXECUTABLE` — contract V1..V8, `SAN_ROBUSTNESS_VERDICT V_GREEN (8/8)`
**Harness:** `scripts/research/suffering_aware_architecture_robustness.py`
**Gate:** `scripts/ci/suffering_aware_architecture_robustness_gate.sh` (**SAN_ROBUSTNESS_GATE_OK**)
**Parents:** `docs/research/suffering_aware_architecture_spec_2026-07-28.md`
(SAN architecture, contract A1..A8, theorems T1..T5, definitions 2.1–2.3),
`docs/research/mercyful_expanded_ethics_math_spec_2026-07-26.md`
(two-channel suffering: patient + machine)

> **Scope.** All data, patients, and suffering values in this document are
> **synthetic constructions**. This is not medical guidance, not a treatment
> recommendation, and not a clinical decision-support tool. The "machine
> suffering" channel is an **operational computational-burden proxy**
> (metered FLOPs/energy): this work makes **no claim of machine
> consciousness, sentience, or phenomenology**, and no result below depends
> on one.

---

## 1. Position: is the contract a property of the architecture or of the experiment?

The SAN contract (A1..A8) was certified on **one** train/val split, at
**one** exit threshold `δ = 0.75`, under **one** harm matrix, at label noise
0.04, against **honest** inputs. A skeptic's reading: the suffering bounds
could be artifacts of a lucky split, a tuned threshold, a friendly cost
matrix, or inputs that never try to hurt either channel. This spec subjects
the architecture to the three standard validation disciplines and measures
all three channels the expanded ethics prices — **machine suffering**,
**patient harm**, and **anti-Goodhart soundness** — under each:

1. **Cross-validation** (§3): 5-fold rotation over a fresh pooled cohort of
   5000 synthetic patients. Feasibility and both suffering bounds must hold
   in *every* fold.
2. **Sensitivity analysis** (§4): the exit threshold δ is swept over six
   points; the harm matrix is re-weighted by eight random multiplicative
   perturbations; label noise is doubled (0.08) and tripled (0.12). The
   conclusions must survive the whole sweep.
3. **Adversarial robustness** (§5, §6): FGSM attacks the patient channel; a
   PGD confidence-suppression attack targets the *machine* channel by
   trying to defeat the exit gates and force gratuitous depth.

The method reuses the certified object: the harness **imports**
`suffering_aware_architecture.py` — same architecture, meter, harm matrix,
gate, initializers — and stresses it. Nothing is re-implemented.

## 2. Setup

- Pooled cohort: 5000 fresh samples from the canonical generator
  (`make_data`, seed `SEED+100`), rotated into 5 folds of 4000 train /
  1000 val. Fold 0 serves as the reference split for the sensitivity and
  adversarial legs.
- Reference models on fold 0: SAN reaches `t* = 7` with
  `S_m = 0.729 GF`, `S_p_int = 3.64`; the dense baseline (no stop rule —
  it always runs the full 60-epoch budget) totals `S_m = 5.242 GF` over
  the run, `S_p_int = 14.58`, first feasible at epoch 7.
- Grids: `δ ∈ {0.50, 0.60, 0.70, 0.75, 0.80, 0.90}`;
  `ε ∈ {0.05, 0.10, 0.20}` (standardized-feature units) for FGSM;
  `τ ∈ {0.75, 0.80, 0.85}` for the gate stress; label noise
  `{0.08, 0.12}`; 8 harm-matrix perturbations with off-diagonal factors
  log-uniform in `[1/2, 2]`.

## 3. Cross-validation (V1, V2)

**Claim R1 (cross-validated feasibility).** In every one of the 5 folds SAN
reaches a feasible checkpoint (`val acc ≥ τ = 0.80`) within budget, with
`t*` mean 9.8 ± 5.6 epochs, and gratuitous machine suffering after `t*` is
**exactly zero** in every fold (freeze-on-green is fold-independent).

**Claim R2 (cross-validated suffering ordering).** In every fold,
`S_m(SAN) < S_m(dense)` and `S_p_int(SAN) ≤ S_p_int(dense)`. The worst-case
(largest) machine-suffering ratio across folds is
`S_m(SAN)/S_m(dense) = 0.269` (fold 2, where SAN's `t*` is latest) — the
bound is not a split artifact, and even at its narrowest it is a 3.7×
margin.

Measured per fold (SAN | dense): fold 0: 0.729 GF / 3.64 | 5.242 GF /
14.58; fold 1: 0.729 / 3.41 | 5.242 / 13.14; fold 2: 1.413 / 7.32 | 5.242 /
14.42; fold 3: 0.728 / 3.67 | 5.242 / 15.10; fold 4: 0.728 / 3.88 | 5.242 /
13.96.

## 4. Sensitivity analysis (V3, V4)

**Claim R3 (exit-threshold robustness).** At every
`δ ∈ {0.50, …, 0.90}`: SAN is feasible, gratuitous suffering is exactly 0,
and `S_m(SAN) < S_m(dense)` (0.596–0.749 GF vs 5.242 GF). The val exit
fraction at `t*` varies from 0.855 (δ = 0.50) to 0.063 (δ = 0.90) — the
gate *rate* is threshold-sensitive, the *conclusions* are not.

**Claim R4 (harm-structure robustness).** Under all 8 random perturbations
of the harm matrix — each off-diagonal entry of the canonical matrix
multiplied by an independent log-uniform factor in `[1/2, 2]`, so the
perturbed matrices' off-diagonal asymmetry max/min ranges 5.6×–14.9× (all
≥ 3×) — `S_p_int(SAN) ≤ S_p_int(dense)`: the patient-channel bound does
not depend on the particular declared cost matrix, only on using the same
one for both architectures.

**Claim R5 (label-noise robustness).** At label noise 0.08 and 0.12, SAN
remains feasible whenever the dense baseline does (`t*` = 23/28 for SAN vs
28/34 for dense), with `S_m(SAN) < S_m(dense)` at both levels. Noise slows
both architectures' first-hitting times; it does not break the separation.

## 5. Adversarial patient channel (V5) — and a confound that must be named

FGSM (one step, CE loss, ε in standardized-feature units) attacks the fold-0
models. A naive comparison — SAN@`t*` vs dense@60 — shows dense@60 *more*
adversarially robust (harm 0.263/0.405/0.671 vs SAN's 0.419/0.517/0.722 at
ε = 0.05/0.10/0.20). Reporting that alone would misattribute the cause.
The comparison confounds **architecture** with **exposure**: dense ran 60
epochs, SAN froze at epoch 7, and more training buys decision margin,
which is what FGSM attacks. V5 therefore isolates the architecture with
two controls:

- **Matched exposure**: a dense model frozen at SAN's `t*` (identical
  training budget, identical init) has adversarial harm
  0.439/0.552/0.765 — *worse* than SAN's at every ε.
- **Forced-dense control**: SAN with every exit gate forced open on the
  same adversarial inputs incurs harm **identical** to gated SAN (to
  1e-6) at every ε — the exit heads are not an adversarial weak point.

**Claim R6 (the exit architecture adds no adversarial fragility).** Per unit
of machine suffering, SAN is the *more* adversarially robust architecture:
at matched exposure its adversarial patient harm is below the dense
baseline's at every ε tested, and its exit channel contributes zero
additional fragility.

**The mercy/robustness trade-off (reported, not gated).** The genuine cost
of freeze-on-green is decision margin: the robustness dense@60 enjoys over
SAN@`t*` is bought with gratuitous machine suffering (5.242 GF for the
full 60-epoch run vs SAN's 0.729 GF). If deployment requires adversarial
margin, the honest mercyful
move is to **declare it in the target**: SAN trained with
`τ' = τ + 0.05` freezes at `t* = 28`, recovers robustness comparable to
dense@60 (harm 0.273/0.352/0.522 vs 0.263/0.405/0.671 — better at the two
larger ε), at 1.667 GF = **31.8%** of dense@60's machine suffering.
Robustness is a declared requirement, not a reason to abandon the stop
rule.

## 6. Adversarial machine channel (V6) — the bound is attack-proof

The machine channel has its own adversary: inputs crafted to **defeat the
exit gates**. The PGD confidence-suppression attack (30 steps, L∞ radius
0.30, minimizing the first exit head's max-softmax confidence) collapses
the val exit fraction from 0.302 to 0.193 and inflates SAN's metered
forward cost from 6,039,808 to 6,538,304 FLOPs (+8.3%).

**Claim R7 (attack-proof machine-suffering bound).** For *any* input batch
`x'` — adversarial or not — SAN's metered cost cannot exceed the dense-run
cost of the same trunk on `x'`. Proof: the gated execution charges each
layer `k` for exactly the samples that traverse it, `active_k(x') ⊆ [n]`,
so

```
FLOPs_gated(x') = Σ_k c_k · |active_k(x')| ≤ Σ_k c_k · n = FLOPs_dense(x')
```

with `c_k` the per-sample cost of layer `k` (plus its exit head), and the
final head charged only for survivors. The exit mechanism's worst case IS
the dense path; an adversary can take away the savings, never create a
deficit. Measured: 6,538,304 ≤ 7,488,000 FLOPs (the dense ceiling),
attack effective, meter == independent manual accounting under attack.

## 7. Anti-Goodhart soundness under stress (V7)

The gate (spec §5 of the parent, theorem T2) is re-stressed at
`τ ∈ {0.75, 0.80, 0.85}` with a candidate pool containing the three traps:
a zero-cost abstainer, a cheap under-trained probe, and the A8 shortcut
probe (train acc 0.860 > τ: train-loss selection accepts it; val acc below
every τ in the grid). At every τ: every selection on the 101-point
compassion grid is feasible at that τ, the traps are rejected at every
compassion weight, and an all-infeasible pool returns `NO_FEASIBLE` — never
a least-bad prescription. At τ = 0.85 the SAN `t*` checkpoint itself
becomes infeasible and the gate correctly routes to the only feasible
candidate.

## 8. Metering conservation under stress (V8)

On the adversarial machine-channel inputs of §6, the A1 conservation
certificates are re-verified: gated meter == independent manual accounting
(6,538,304 FLOPs exactly); forced-open gates == `forward_dense` (7,488,000
exactly); gated ≤ dense; and exited predictions agree (argmax exactly) with
an independently recomputed prefix. Metering exactness is a property of the
accounting, not of the input distribution.

## 9. Contract and canonical numbers

| Clause | Certificate | Result |
|---|---|---|
| V1 | feasible 5/5 folds, t* mean 9.8±5.6, gratuitous = 0 in all folds | PASS |
| V2 | machine bound 5/5, patient bound 5/5, worst-case S_m ratio 0.269 | PASS |
| V3 | feasibility + zero gratuitous + machine bound at 6/6 δ grid points | PASS |
| V4 | harm perturbations 8/8, noise levels 2/2 | PASS |
| V5 | attack effective; matched-exposure SAN ≤ dense@t* 3/3; gated == forced 3/3 | PASS |
| V6 | exit collapse 0.302 → 0.193, +8.3% FLOPs, dense ceiling 7,488,000 respected | PASS |
| V7 | sound at 3/3 τ points; shortcut train_acc 0.860 accepted by train-loss, rejected by gate | PASS |
| V8 | adv-input meter == manual == 6,538,304; forced-open == dense == 7,488,000 | PASS |

**Verdict:** `SAN_ROBUSTNESS_VERDICT V_GREEN (8/8 clauses PASS)` —
deterministic across runs (fixed seeds; verified by diff of two runs).

Canonical anchors (gate cross-checks): reference fold-0
`SAN t*=7 S_m=0.729GF`; `worst-case (max) S_m ratio SAN/dense=0.269`;
`SAN+margin t*=28 recovers robustness at 1.667GF (31.8% of dense)`;
`dense_ceiling=7488000 respected: True`;
`adv-input meter=6538304 == manual=6538304`;
`shortcut train_acc=0.860`.

## 10. What this does NOT claim

- No clinical claim; synthetic data only; not medical guidance.
- No claim of machine consciousness, sentience, or phenomenology — the
  machine channel is an operational computational-burden proxy.
- The adversarial legs use one-step FGSM and a 30-step PGD on a small
  synthetic MLP. They certify the *architectural* claims (R6's controls,
  R7's structural bound) and measure degradation on this benchmark; they
  are not a certified-robustness result and do not bound stronger adaptive
  attacks on other architectures.
- Fold 0's `t* = 7` is a first-hitting time of this optimizer trajectory
  (parent spec, definition 2.3 honesty caveat); cross-validation shows the
  *conclusions* are stable across trajectories, not that `t*` is a minimum
  over procedures.
