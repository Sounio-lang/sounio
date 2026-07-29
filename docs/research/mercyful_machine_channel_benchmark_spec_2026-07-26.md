<!-- docs:meta
topic_id: repo.docs.research.mercyful-machine-channel-benchmark-spec-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.mercyful-machine-channel-benchmark-spec-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — Machine-Channel Structural Benchmark (OPUS 5, critique 5)

- Date: 2026-07-26
- Status: implemented, C_GREEN (8/8 clauses)
- Harness: `scripts/research/mercyful_machine_channel_benchmark.py`
- CI gate: `scripts/ci/mercyful_machine_channel_gate.sh`
- Parent artifact: `docs/papers/mercyful_learning_paradigm_2026-07-26.md`
- Sibling benchmark: `scripts/research/mercyful_paradigm_benchmark.py` (patient channel)

**This benchmark uses synthetic data only. It makes no clinical claim and is
not medical guidance.**

## 1. The critique being answered

OPUS 5, critique 5 of the Mercyful Learning paradigm paper:

> "Canal máquina: S_machine = 0,0003 nas três linhas — cerca de 0,03% do
> objetivo. A novidade ética central do paper não move o gradiente em lugar
> nenhum do §5. O que de fato exerce o canal máquina é a regra de parada (39
> versus 300 épocas), não o termo ρ‖θ‖². Diga isso explicitamente, ou
> construa um benchmark em que μ decida algo estrutural (largura/profundidade,
> FLOPs medidos). Como está, o segundo sofredor é decorativo na evidência e
> central na retórica."

Two distinct demands: (a) say explicitly that in the §5 dose-response
benchmark the machine channel acted only through early stopping, and
(b) build a benchmark where μ decides something structural. This document and
its harness do (b); the admission of (a) belongs in the paper text itself.

## 2. Design

### 2.1 Problem

Synthetic 2-D sine-boundary binary classification
(`y = 1[x2 > sin(2x1) + 0.3 sin(5x1)]`, 3% label noise), N_train = 5000,
N_test = 1000, fixed seed 11. Anti-Goodhart target τ = 0.88 test accuracy.
The task is chosen so that a wide range of capacities clears τ: capacity buys
margin, not feasibility — the regime where a machine-suffering channel should
trade structure against a fixed target.

### 2.2 Structural decision space

A grid of 20 MLPs: width ∈ {8, 16, 32, 64, 128} × depth ∈ {1, 2, 3, 4},
tanh activations. Parameter counts span 42 → 50 178 (≈ 1200×).

**Every candidate is trained with the identical fixed budget** (300 epochs,
Adam lr = 1e-2, full batch). Early stopping plays no role anywhere in the
selection pipeline. Whatever μ decides here is therefore architectural —
width, depth, parameters, FLOPs, energy — not stopping time. This is the
exact contrast OPUS 5 demanded with the §5 benchmark, where the channel acted
only through 39-vs-300 epochs.

### 2.3 Measured machine suffering (not ρ‖θ‖²)

For each trained candidate, S_machine is measured, not proxied by a
parameter norm:

- **Parameters** P: exact count.
- **Training FLOPs**: `6 · P · N · epochs` — the standard GEMM-counting
  convention (forward = 2 FLOPs per parameter per sample, backward ≈ 2×
  forward). Activation-function and optimizer FLOPs are neglected (percent
  level at these sizes).
- **Energy proxy**: `FLOPs × 4e-12 J`, an order-of-magnitude constant for
  modern CPU SIMD GEMM throughput.
- **Wall-clock training time**: measured with `time.perf_counter`, used to
  validate that the analytic FLOP count tracks a physical quantity
  (certificate M7: Pearson r ≥ 0.85; observed 0.96–0.99).

The paper's `ρ‖θ‖²` term (ρ = 1e-3) is computed alongside purely as a
foil: certificate M8 requires the measured channel's dynamic range across
feasible architectures to be ≥ 50× and ≥ 10× the norm proxy's (observed:
1195× vs 31×). This quantifies the "0.03% of the objective" complaint and
shows the replacement channel has structural teeth.

### 2.4 Objective and selection

For each μ in a 25-point grid (0 … ~5/S_min, geometric):

- **Filtered (anti-Goodhart) selection**:
  `argmin over {arch : test_acc ≥ τ} of [ (1 − acc) + μ · S_machine ]`.
  Exact task-loss ties (< 5e-8) are broken deterministically toward larger
  capacity: with no machine penalty pressing down, capacity is free.
- **Unfiltered selection**: same objective over all candidates **plus an
  abstain option** (0 parameters, 0 FLOPs, majority-class accuracy ≈ 0.50).
  This exposes the abstention trap: at large μ the raw objective prefers to
  not compute at all.

S_machine enters in joules, so μ has units 1/J; the grid is scaled to the
measured energies so the sweep covers the full transition from "capacity is
free" to "abstention wins".

### 2.5 Relation to the full mercyful objective

The paradigm objective is `L_task + λ·S_patient + μ·S_machine`. This
benchmark isolates the machine channel: the patient channel is covered by
`mercyful_paradigm_benchmark.py` (clauses P1–P8) and is held out here so the
structural effect of μ is attributable to S_machine alone. With S_patient
absent, the selection rule reduces to a constrained discrete trade between
task loss and measured compute — the cleanest possible test of whether μ can
move a structural decision.

## 3. What the sweep shows (observed, seed 11)

| μ (1/J)      | selected | params | test acc | training FLOPs | energy |
|--------------|----------|--------|----------|----------------|--------|
| 0            | 32×4     | 3 330  | 0.976    | 3.0e10         | 0.120 J |
| ~1.3e-1      | 8×4      | 258    | 0.967    | 2.3e09         | 0.009 J |
| ~6.5         | 8×2      | 114    | 0.938    | 1.0e09         | 0.004 J |
| ≥ ~1.4e1     | 8×1      | 42     | 0.910    | 3.8e08         | 0.002 J |
| ≥ ~3.2e2 (unfiltered) | ABSTAIN | 0 | 0.503 < τ | 0         | 0       |

- μ = 0: selection is the best-**performing** model, ignoring structure
  entirely (certificate M1). Note it is *not* the largest model — with
  held-out accuracy the mid-size net generalizes best; the honest statement
  is that at μ = 0 the objective is pure task loss.
- Intermediate μ: four distinct architectures are selected as μ grows, with
  parameters weakly monotone non-increasing (M2, M5).
- Large μ: 79× parameter shrink and 1.3% of the μ=0 FLOPs/energy, while
  test accuracy stays ≥ τ at every selected point (M3, M4).
- Very large μ: the **unfiltered** objective abstains (Perf = 0.503 < τ) —
  the abstention trap. The anti-Goodhart filter rejects abstention and
  returns the smallest feasible model, 8×1 at 0.910 ≥ τ (M6). The penalty
  alone would Goodhart into doing nothing; the constraint is what guards
  the target.

## 4. Certificates (contract clauses)

- **M1** μ = 0 ignores structure; selection = argmax test accuracy among
  feasible architectures (and ≥ 10× the parameters of the small-end
  selection).
- **M2** Selected parameter count is weakly monotone non-increasing in μ.
- **M3** Structural shrink ≥ 8× in parameters from the μ = 0 selection to the
  smallest feasible selection, with accuracy ≥ τ maintained (observed 79×).
- **M4** Measured training FLOPs of the smallest feasible selection ≤ 5% of
  the μ = 0 selection (observed 1.3%); energy scales identically.
- **M5** Decision is structural, not stopping: identical fixed epoch budget
  for every candidate, and ≥ 3 distinct architectures selected across the
  sweep (observed 4).
- **M6** Abstention trap: at μ_max the unfiltered objective selects
  abstention with Perf < τ; the anti-Goodhart filter overrides it with a
  feasible model.
- **M7** Measured channel is physical: Pearson(wall-clock, analytic FLOPs)
  ≥ 0.85 across candidates (observed 0.96–0.99).
- **M8** Non-decorative channel: FLOPs-based S_machine dynamic range ≥ 50×
  across feasible architectures and ≥ 10× the ρ‖θ‖² proxy's range
  (observed 1195× vs 31×).

Verdict line: `MERCYFUL_MACHINE_CHANNEL_VERDICT C_GREEN (8/8 clauses PASS)`.

## 5. Answering the critique, point by point

- "μ doesn't move the gradient anywhere in §5" — conceded for §5 (early
  stopping only); this benchmark moves the *selection* over a discrete
  structural space instead, which is the stronger claim the paper's rhetoric
  needs.
- "Build a benchmark where μ decides width/depth/measured FLOPs" — done:
  width, depth, parameters, FLOPs, and energy all vary with μ at a fixed
  training horizon (M2–M5).
- "The second sufferer is decorative in the evidence" — the measured channel
  is not: it has ~38× the dynamic range of the norm proxy and changes the
  selected architecture four times across the sweep (M5, M8).

## 6. Scope and limitations

- Synthetic data only; no clinical claim; not medical guidance.
- The structural space is a discrete grid (selection), not differentiable
  architecture search; μ acts on a choice among trained candidates, which is
  the honest discrete analogue of a penalty on structure.
- FLOPs use the GEMM convention and the energy constant is an
  order-of-magnitude proxy; M7 exists precisely to anchor the analytic count
  to a measured physical quantity (wall clock).
- Patient channel deliberately excluded (isolation); see
  `mercyful_paradigm_benchmark.py` for the combined-objective evidence.

## 7. Reproduction

```bash
.venv/bin/python scripts/research/mercyful_machine_channel_benchmark.py
bash scripts/ci/mercyful_machine_channel_gate.sh
```

Requires the repo `.venv` (torch CPU + numpy). Runtime ≈ 60–90 s on CPU.
Fully deterministic (fixed seeds, fixed epoch budget); wall-clock values and
the M7 correlation vary run-to-run within their threshold margins.
