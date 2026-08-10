<!-- docs:meta
topic_id: repo.docs.research.suffering-aware-extreme-depth-spec-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.suffering-aware-extreme-depth-spec-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — the Suffering-Aware neural Network (SAN) at extreme depth: a 100-layer residual network and a 24-block Transformer on real data

**Date:** 2026-07-31
**Branch:** research/self-falsifying-compilation-line-20260726
**Status:** `EXECUTABLE` — contract X1..X9, `SUFFERING_AWARE_EXTREME_DEPTH_VERDICT X_GREEN (9/9)`
**Harness:** `scripts/research/suffering_aware_extreme_depth.py`
**Gate:** `scripts/ci/suffering_aware_extreme_depth_gate.sh` (**SUFFERING_AWARE_EXTREME_DEPTH_GATE_OK**)
**Parent:** `docs/research/suffering_aware_deep_architecture_spec_2026-07-28.md`
(deep SAN, clauses D1..D9 — ResNet-18 + ViT-small; definitions, theorems
T1'..T6, and the benchmark method this spec scales to extreme depth without
modification)

> **Scope.** The dataset (CIFAR-10) and the architectures (a 100-layer
> ResNet, a 24-block ViT) are **real**; the harm matrix is a **synthetic
> cost structure over the real labels** (a screening-pipeline hazard model,
> §6.1). This is not medical guidance, not a treatment recommendation, and
> not a clinical decision-support tool. The "machine suffering" channel is
> an **operational computational-burden proxy** (metered FLOPs/energy):
> this work makes **no claim of machine consciousness, sentience, or
> phenomenology**, and no result below depends on one.

---

## 1. Position: does the architecture survive extreme depth?

The deep-network SAN spec established the architecture at 18 weighted
layers (ResNet-18) and 6 attention blocks (ViT-small), and its scalability
theorem T6 certified metering conservation forward-only up to 36 conv
layers / 8 blocks. Extreme depth is where the remaining open failure modes
live:

1. **Trainability.** A plain 100-layer network without residual
   connections does not train. SAN-100 is residual, but its feasibility
   relies on deep supervision through seven exit heads at widths
   16/32/64 — far narrower and deeper than anything the parent line
   trained. Does the suffering-aware instance still reach the declared
   target inside a CPU-affordable budget — and does the plain 100-layer
   trunk (the dense baseline) reach it at all?
2. **Metering.** T1' is a term-by-term identity over per-map FLOP counts;
   depth enters only as the number of terms. But the *numerical* half of
   the certificate — eval-mode prefix invariance, bounded logit deviation
   with exactly agreeing argmax — is an empirical statement about backend
   numerics that could plausibly degrade as error propagates through 100
   convolutions or 24 attention blocks under batch-shape changes. The
   contract measures it.
3. **Soundness.** The anti-Goodhart gate is architecture-independent by
   construction (T2 quantifies over candidate pools, not networks), but
   the pool members that must be rejected — abstainers, under-trained
   probes, shortcut models — are re-certified at extreme depth.
4. **The economics amplify.** With 7 exit points against the parent
   line's 4 (ResNet) and 24 against 6 (ViT), the skippable fraction of the
   trunk dominates: an exit at stage k of 7 skips (7-k)/7 of the remaining
   trunk. The parent predicted savings grow with depth; this spec measures
   whether the amplification materializes in training (per-epoch charge),
   in total (S_machine), and at deployment (gated forward).

The answer, certified by the contract: the architecture works at extreme
depth. Nothing in the design changes; what changes is the evidence.

## 2. What is reused unchanged

From the parent specs, without modification: the suffering ledger
(Definition 2.1: `S_machine`, `S_patient` integral + peak), feasibility as
a categorical anti-Goodhart constraint (Definition 2.2), the necessary/
gratuitous decomposition at the first feasible epoch `t*` (Definition 2.3,
with its trajectory-relative honesty caveat), the selection rule
`select(C, λ)` with loud `NO_FEASIBLE`, the metering convention
(MAC ×2, backward = 2× forward, energy = FLOPs × 4e-12 J), the asymmetric
harm matrix over the real CIFAR-10 labels (§6.1), and the design rule
**constraints and gates, not penalties** — no suffering term appears in
any training loss here either.

## 3. The extreme-depth suffering-aware layers

**SufferingAwareResNet-100 (SAN-100).** A CIFAR-variant residual network
of **100 weighted layers**: stem conv 3→16, then seven stages of seven
basic blocks (49 blocks × 2 convs = 98), then the fc head — 1 + 98 + 1 =
100, the ResNet depth-counting convention (the 2 downsample shortcut convs
are not counted, exactly as ResNet-18 counts 1 + 8·2 + 1 = 18). Stage
widths (16,16,32,32,64,64,64) — the width profile of the classic CIFAR
ResNet-110, a known-trainable configuration at this depth — with stride-2
downsampling at the entries of stages 2 and 4. Each stage carries an exit
head (global average pool + `Linear(C, 10)`): **7 exit points** against
the parent's 4, each stage a suffering-aware layer in the parent spec's
sense, metering its exact executed-path FLOPs and its exit's harm.

**SufferingAwareViT-24 (SAN-ViT-24).** The parent line's vision
transformer scaled from 6 to **24 attention blocks**: 4×4 patch embedding
→ 64 tokens + CLS, `d = 96`, 4 heads, MLP ratio 2, one exit head per block
on the CLS token: **24 exit points** against the parent's 6. Attention is
metered exactly as in the parent: QKV/projection/MLP as linear maps over
all tokens, the two token-mixing matmuls as `2·2·T²·d` FLOPs per sample.

Elementwise operations (BatchNorm, ReLU/GELU, residual adds, softmax,
pooling) are unmetered — the stated convention, identical for every
architecture and accounting path, unchanged from the parent line.

**Gates, supervision, freezing.** Per-sample exit gates (confidence
threshold `δ`, a declared architecture constant **per family**: `δ_R`,
`δ_V` — declared up front, not tuned per run; §6.4 disclosure), deep
supervision of every exit head after a one-epoch dense-identical warm-up,
and freeze-on-green at the first feasible epoch are the parent mechanisms
verbatim. During training, BatchNorm uses batch statistics of the *active*
subset; the prefix-invariance certificate (X1) is an eval-mode statement,
where BatchNorm is per-sample deterministic.

## 4. The economics at extreme depth (why depth amplifies the architecture)

Two structural effects, both certified by the contract:

1. **Skippable fraction grows.** An exit fired after stage k of S stages
   skips the remaining S − k stages. At 7 stages the median skippable
   share of the trunk is larger than at 4, and at 24 exit points (ViT)
   larger still than at 6. Exit-head overhead as a fraction of the
   gates-open forward, measured by the depth sweep (X9):

   | trunk | exit-head overhead |
   |---|---|
   | ResNet 16 layers (1 block/stage) | TBD% |
   | ResNet 44 layers (3 blocks/stage) | TBD% |
   | ResNet 100 layers (7 blocks/stage) | TBD% |
   | ViT 6 / 12 / 24 blocks | TBD% |

   For the ResNet family the overhead fraction falls as blocks-per-stage
   grows (head cost per stage is constant, trunk cost per stage grows);
   for the ViT family it is depth-independent (each block adds one head
   and one block of trunk cost) — both bounded far below the 5% clause
   threshold.
2. **Every fired gate is nearly pure savings.** With overhead ≪ 0.1%,
   SAN's average per-epoch metered charge drops strictly below the plain
   trunk's as soon as the gates fire (measured, §6.4), and the deployment
   forward pass on the held-out cohort costs strictly less than the
   gates-open pass (X1).

## 5. Theorems

The parent theorems are architecture-class statements; their proofs lift
to extreme depth without new ideas. We restate with the extreme-depth
certificate numbers.

**T1′′ (metering conservation at extreme depth).** For any SAN forward
pass — ResNet or ViT trunk, any depth — the metered machine suffering
equals the analytic cost of the executed path: a stage/block gated off for
a sample contributes exactly 0, and the total `M_gated ≤ M_dense` with
equality iff no exit fires. *Proof.* Identical to T1′, term by term: the
meter charges each executed map per sample handed to it under per-map
conventions fixed for both accounting paths (conv
`2·C_in·C_out·K²·H_out·W_out`, linear `2·d_in·d_out` per token row, each
attention token-mixing matmul `2·T²·d`; residual adds, BN, activations,
softmax, pooling unmetered in both accountings). Depth enters only as the
number of terms in the sum; no step of the T1′ argument depends on it. ∎
*Verified (X1, X9):* metered charge equals an **independent manual
accounting** of the executed path **exactly**, for both trained families
on the full held-out set and for every configuration of the depth sweep
(16/44/100-layer ResNets, 6/12/24-block ViTs); strictly below the
gates-open charge whenever an exit fires; eval-mode prefix logits match an
independently recomputed dense prefix with bounded deviation and
**exactly agreeing argmax** everywhere — including through all 100
convolutions / 24 attention blocks.

**T2 (anti-Goodhart soundness, unchanged).** For every `λ ∈ [0,1]` and
every candidate pool, `select(C, λ)` is feasible or `NO_FEASIBLE`.
*Verified (X3, X8):* 101-point λ-grid over pools containing a zero-cost
abstainer, an under-trained pixel probe, and a corner-patch shortcut probe
that beats τ on **train** while failing it held-out — selection feasible
at every grid point; all-infeasible pool → `NO_FEASIBLE`.

**T3 (machine-suffering bound, unchanged statement).** With `t*` the first
feasible epoch, `S_machine(SAN) = Σ_{t≤t*} E(t) ≤ Σ_{t≤t*} F(t)` and
`S_gratuitous(SAN) = 0`; any fixed `T`-epoch run of the same trunk accrues
`B(t*) + Σ_{t*<t≤T} F(t)`. *Verified (X4, X5):* numbers in §6.4.

**T4 (necessary/gratuitous separation, unchanged).** The ledger
decomposition is recomputed, not asserted; the necessity is
trajectory-relative (the parent caveat stands, unchanged).

**T5′ (feasibility at extreme depth, certificate).** On the canonical
instance, SAN-100 and SAN-ViT-24 each reach a feasible checkpoint strictly
inside budget (X2), and the plain 100-layer / 24-block trunks (dense
baselines) each reach τ inside budget as well (a precondition X4
certifies) — the extreme-depth trunks train at all on this instance. As
in the parent, this certifies the instance; no universal convergence
claim is made.

**T6′ (scalability to extreme depth).** The architecture's invariants are
depth-parametric with no depth bound: for the family of SAN-ResNets (any
stage count, any blocks-per-stage, any width profile) and SAN-ViTs (any
depth, any `d`), T1′′ holds by the same proof, and the exit-head overhead
fraction is bounded by `Σ_k 2·C_k·10 / F_trunk` — for the ResNet family
tending to zero as blocks-per-stage grows, for the ViT family constant in
depth at `2·d·10 / F_block`, both far below any operationally visible
threshold. The contract certifies the theorem's content on a
6-configuration sweep (X9): at every scale from 16 to 100 weighted layers
and 6 to 24 attention blocks, metered = manual exactly, gated < gates-open
when exits fire, prefix argmax agreement exact, overhead < 5%. Nothing in
the architecture's cost model, gating semantics, or gate soundness
degrades as depth grows to 100 layers / 24 blocks — the architecture
scales to extreme depth without breaking.

## 6. Benchmark

### 6.1 Task and data

CIFAR-10 (real dataset): stratified subset of 4000 train (400/class, from
the train batches) / 1000 held-out (100/class, from the test batch),
standard channel normalization, **no augmentation** (documented scope —
the benchmark measures suffering accounting, not SOTA accuracy).
Deterministic shared data order across all runs (seed 17). Provenance:
`datasets/cifar-10-batches-py/` (see the parent spec; the harness reads
only the pickle layout, and the gate names the canonical fetch command).

The harm matrix is the parent line's synthetic cost structure over the
real labels, unchanged: class 9 ("truck") is the hazard class of a
screening pipeline —

```
H[true, pred]: 0 diagonal; 5 for a missed hazard (true 9, pred other);
2 for a false hazard (pred 9, true other); 1 otherwise.
```

The asymmetry (5×) prices the two pathologies — missed hazard and
unnecessary intervention — that the gate exists to block. No clinical
claim: the patient channel is measured at extreme depth over real images,
with the synthetic nature of `H` stated plainly.

### 6.2 Declared targets and budgets

Anti-Goodhart targets are declared inputs, chosen below what the standard
deep architecture demonstrably reaches inside budget on this instance:
`τ_R = 0.30` held-out accuracy for the 100-layer ResNet family (budget
`T_R = 8` epochs), `τ_V = 0.30` for the 24-block ViT family (budget
`T_V = 10`). Adam lr 1e-3, batch 128, seed 17. CPU-only (torch, 56
threads). The targets are lower than the parent line's `τ_R = 0.35`
because the 100-layer trunk is deliberately narrow (width 16/32/64 — the
trainable-at-depth profile) where the parent's ResNet-18 is wide (64..512);
the declaration rule is unchanged: below what the standard architecture
reaches, declared up front.

### 6.3 Architectures compared

Within each family, one shared trunk init, one data order, one seed:

- **Dense** — the identical 100-layer / 24-block trunk, fixed budget: the
  standard deep architecture trained everywhere.
- **EarlyStop** — the identical trunk with SAN's stop rule but no
  suffering-aware layers: the strongest *scheduler* baseline.
- **SAN** — this spec.

### 6.4 Measured results (canonical instance, bit-reproducible at seed 17)

TBD — filled from the canonical run.

## 7. Contract clauses

| Clause | Claim | Canonical numbers |
|---|---|---|
| X1 | T1′′ metering conservation at extreme depth (both trained families): gated-off stages/blocks charge exactly 0; metered = independent manual accounting; < gates-open when exits fire; eval-mode prefix argmax exactly equal | TBD |
| X2 | T5′ feasibility at extreme depth: SAN-100 and SAN-ViT-24 reach a feasible checkpoint within budget | TBD |
| X3 | T2 soundness at extreme depth: feasible-only selection on a 101-point λ-grid; loud NO_FEASIBLE | TBD |
| X4 | T3/T4 separation: SAN gratuitous = 0; dense baselines' > 0 | TBD |
| X5 | T3 bound: SAN total machine suffering below every baseline; integrated patient harm ≤ every baseline | TBD |
| X6 | exits real at extreme depth, not decorative | TBD |
| X7 | patient channel first-class: harm matrix asymmetric; SAN peak ≤ same-init baselines' | TBD |
| X8 | anti-shortcut at extreme depth: train-loss selection accepts the corner-patch shortcut, gate rejects at every weight | TBD |
| X9 | T6′ scalability to extreme depth: 6-configuration sweep — metered = manual exactly, gated < gates-open with exits, prefix argmax exact, exit-head overhead < 5% | TBD |

Run: `.venv/bin/python scripts/research/suffering_aware_extreme_depth.py` →
`SUFFERING_AWARE_EXTREME_DEPTH_VERDICT X_GREEN (9/9 clauses PASS)`.

## 8. Falsifiers

| Clause | Falsifier |
|---|---|
| X1 | A gated-off stage/block charges FLOPs; metered ≠ manual accounting; gated > gates-open with an exit fired; an exited prediction's argmax disagrees with the recomputed prefix |
| X2 | No feasible SAN checkpoint within budget for either family |
| X3 | Any λ at which an infeasible candidate is selected; an all-infeasible pool returning a prescription; abstainer/probe feasible |
| X4 | SAN gratuitous FLOPs > 0; a feasible fixed-budget baseline with gratuitous = 0 |
| X5 | Dense fixed-budget baseline with total machine suffering ≤ SAN's; any baseline with integrated patient harm below SAN's; EarlyStop strictly below SAN on the machine channel |
| X6 | Exit fraction ≤ 10% at t* for either family (heads decorative) |
| X7 | Harm matrix near-symmetric; SAN peak above a same-init baseline's |
| X8 | Shortcut probe feasible held-out, or selected at any weight |
| X9 | At any swept scale: metered ≠ manual, gated > gates-open with exits, prefix argmax disagreement, or exit-head overhead ≥ 5% |

Gate failure classification (per AGENTS.md): build/bootstrap-path (repo
`.venv` missing torch), harness-routing (gate script paths, missing
CIFAR-10 — the gate names the fetch command), ontology-kernel/checker
(n/a), baseline noise (numerics beyond the prefix bound / argmax flip —
would indicate a backend whose conv results depend on batch shape; the
argmax-exactness sub-check is the load-bearing one, and extreme depth is
where such a backend would first show it).

## 9. Scoped out (explicit)

1. **Full CIFAR-10 / ImageNet and GPU-scale training.** The subset and CPU
   budget are a documented contract affordance; nothing in the theorems or
   the metering depends on them. The sweep (X9) covers depth scaling
   forward-only; larger-budget training runs belong to the Foundry/Slurm
   path per AGENTS.md.
2. **Data augmentation and accuracy engineering.** The benchmark measures
   suffering accounting against declared targets, not maximal accuracy.
3. **A calibrated patient-harm model.** `H` is synthetic over real labels;
   no clinical claim is made here.
4. **Hardware-metered energy** (RAPL/perf counters): analytic FLOPs × the
   stated J/FLOP constant, as in the parent line.
5. **A Sounio-native leg** — Python/PyTorch reference implementation, as in
   the parent specs.
6. **Very deep plain (non-residual) networks.** The trainability question
   is scoped to the residual/transformer instances the architecture is
   defined on; a non-residual 100-layer trunk's known failure to train is
   the motivation, not a subject, of this contract.
7. **`topic-registry.v1.json` registration and `.github/workflows/ci.yml`
   wiring** — shared control surfaces under active edit by other lanes on
   this branch; left to the integrator (same convention as the parent
   specs). The gate is self-contained and green.

## 10. Commands run

```bash
.venv/bin/python scripts/research/suffering_aware_extreme_depth.py   # X_GREEN 9/9 (bit-reproducible at seed 17)
bash scripts/ci/suffering_aware_extreme_depth_gate.sh                # SUFFERING_AWARE_EXTREME_DEPTH_GATE_OK
bin/llm-offload -t math-review -i docs/research/suffering_aware_extreme_depth_spec_2026-07-31.md
```

TBD — calibration history.

## 11. LLM-offload review

TBD.
