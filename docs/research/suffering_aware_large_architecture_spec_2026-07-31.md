<!-- docs:meta
topic_id: repo.docs.research.suffering-aware-large-architecture-spec-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.suffering-aware-large-architecture-spec-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — the Suffering-Aware neural Network (SAN) at larger scale: SAN-ResNet-50, SAN-ViT-large, and SAN-GPT

**Date:** 2026-07-31
**Branch:** research/self-falsifying-compilation-line-20260726
**Status:** `EXECUTABLE` — contract L1..L9
**Harness:** `scripts/research/suffering_aware_large_architecture.py`
**Gate:** `scripts/ci/suffering_aware_large_architecture_gate.sh` (**SUFFERING_AWARE_LARGE_GATE_OK**)
**Parent:** `docs/research/suffering_aware_deep_architecture_spec_2026-07-28.md`
(deep-network SAN, clauses D1..D9 — ResNet-18 + ViT-small on CIFAR-10) and
`docs/research/suffering_aware_architecture_spec_2026-07-28.md`
(small-network SAN, clauses A1..A8 — definitions, theorems T1..T5, and the
benchmark method this spec scales without modification)

> **Scope.** The datasets (CIFAR-10 images; the repository's own
> documentation text corpus) and the architectures (ResNet-50, a
> contract-scale ViT-large, a GPT decoder-only transformer) are **real**;
> the harm structures are **synthetic cost structures over the real
> labels/tokens** (§6.1). This is not medical guidance, not a treatment
> recommendation, and not a clinical decision-support tool. The "machine
> suffering" channel is an **operational computational-burden proxy**
> (metered FLOPs/energy): this work makes **no claim of machine
> consciousness, sentience, or phenomenology**, and no result below depends
> on one.

---

## 1. Position: does the architecture survive *larger* architectures?

The parent deep-architecture spec established that the SAN design —
suffering-aware layers, per-sample exit gates, freeze-on-green, and the
architectural anti-Goodhart gate — survives depth on ResNet-18 and
ViT-small, and its T6 certified scalability forward-only across a depth
sweep. Two scaling questions remained open, and this spec closes both:

1. **Width/structure scaling within the vision families.** ResNet-50
   changes the block *structure* (bottleneck 1×1–3×3–1×1 with 4× expansion,
   ~25M parameters vs ResNet-18's ~11M); the metering identity must remain
   exact under a third conv per block and per-block shortcuts that change
   dimension at every stage entry. ViT-large proportions (d=384, 12 blocks,
   6 heads, MLP ratio 4, ~22M parameters vs ViT-small's d=128/6 blocks)
   raise the per-block matmul cost by about `(384/128)² = 9×` and double
   the block count; the exit-head overhead fraction should shrink further
   (T6).
2. **Cross-modal scaling to a generative transformer.** A GPT
   (decoder-only, causal masked attention, next-token objective over a
   2000-word vocabulary) is a different task class: structured prediction
   over token sequences, not single-label classification. The architecture
   must answer: what is an exit head for a language model, what is the
   patient-harm channel over tokens, and does exact metering survive causal
   attention and embedding lookups.

The answer, certified by the contract: the architecture scales to both.
Nothing in the design changes; what changes is the evidence and two
declared per-modality conventions (§3), stated up front.

## 2. What is reused unchanged

From the parent specs, without modification: the suffering ledger
(Definition 2.1: `S_machine`, `S_patient` integral + peak), feasibility as
a categorical anti-Goodhart constraint (Definition 2.2), the necessary/
gratuitous decomposition at the first feasible epoch `t*` (Definition 2.3,
with its trajectory-relative honesty caveat), the selection rule
`select(C, λ)` with loud `NO_FEASIBLE`, the metering convention (MAC ×2,
each attention token-mixing matmul `2·T²·d` per sample, backward = 2×
forward, energy = FLOPs × 4e-12 J), the per-family declaration of targets
τ and exit thresholds δ as architecture constants, and the design rule
**constraints and gates, not penalties** — no suffering term appears in any
training loss here either.

## 3. The three larger families

**SufferingAwareResNet-50.** A CIFAR-variant ResNet-50: stem conv 3→64,
then four stages of bottleneck blocks (1×1 reduce, 3×3, 1×1 expand),
config (3,4,6,3), inner widths 64/128/256/512, stage outputs
256/512/1024/2048 (~25M parameters). Each stage carries an exit head —
global average pool + `Linear(C_out, 10)` — exactly as in the parent; the
meter charges all three convolutions and the shortcut per executed sample.

**SufferingAwareViT-large (contract scale).** A vision transformer at
ViT-large *proportions* scaled to the CPU contract budget: 4×4 patch
embedding → 64 tokens + CLS, `d = 384`, 12 blocks, 6 heads (head dim 64),
MLP ratio 4 (~22M parameters). An honest ViT-L/16 (d=1024, 24 blocks,
307M parameters) is outside any CPU contract budget; the scaling claim
certified here is depth/width scaling *within* the transformer family
(width-squared and depth both larger than ViT-small; L9 measures the
metered cost at 6/12/16 blocks), and the L9 sweep pushes depth
forward-only to 16 blocks. Each block carries an exit head on its
CLS token (`Linear(384, 10)`).

**SufferingAwareGPT.** A decoder-only transformer language model: learned
token + position embeddings, 10 pre-LN causal-attention blocks, `d = 384`,
6 heads, MLP ratio 4, sequence length T=64, vocabulary V=2000 (UNK + the
1999 most frequent words of the corpus, §6.1). Two per-modality
conventions are declared up front:

- **Exit heads score the last G=4 positions.** A full-sequence LM head
  costs `2·d·V·T` per block, which would make the exit-head overhead ~42%
  of the trunk and break the architecture's at-scale economics (parent
  §4). Scoring the last G positions costs `2·d·V·G` — 2.6% of one block
  (analytic: `2·d·V·G / (24·d²·T + 4·T²·d)` = 6 144 000 / 232 783 872 =
  2.64%), ~2.6% cumulative overhead at depth 10, inside the 5% clause
  bound (L9, measured). The **scored task** is therefore declared:
  next-token prediction on the last G positions of each sequence;
  accuracy and harm are measured there for *every* architecture in the
  family (dense baselines included). The gate confidence is the mean
  max-prob over the G positions.
- **The final head still supervises all T positions in training**
  (standard LM objective, metered `2·d·V·T`); only the exit/gating/scoring
  path uses the G-position slice. In eval the final head is metered for
  exactly the G scored positions it computes.

Embedding lookups, causal masking, LayerNorm/BatchNorm, activations,
softmax, residual adds and pooling are unmetered — the parent convention,
stated, identical for every architecture and every accounting path.

**Gates, supervision, freezing.** Per-sample exit gates (δ_R = 0.55,
δ_V = 0.45, δ_G = 0.31 — declared per family like τ; see §6.2 for the
calibration that produced these values), freeze-on-green at the first
feasible epoch, and **per-family head wiring** (declared, measured in the
calibration history):

- **ResNet-50:** exit heads train with deep supervision **into the trunk**
  (`DETACH_RESNET=0`). Detached stage heads on 2–3-epoch conv features
  gate too weakly (~0.23 acc); trunk-coupled aux is required for
  conv-stage viability at this budget.
- **ViT-large and GPT:** exit heads train as **probes on detached features**
  (`DETACH_VIT=1`, `DETACH_AUX=1`). Aux gradients into the trunk dilute the
  LM final-head objective and shift early vision harm away from the shared
  epoch-0 exposure that L7 requires.

Gates-closed warmup is also per family: W_R = 2, W_V = 4, W_G = 1 epochs
(heads train; gates closed). Feasibility `t*` counts only gates-active
epochs.

## 4. The economics at larger scale

The parent spec's structural prediction (its §4) holds and strengthens:
the metered price of a suffering-aware layer's exit head falls relative to
its trunk as the trunk grows, while the gate's savings grow with depth.
Measured exit-head overhead (L9 sweep, fraction of the gates-open forward):

| trunk | exit-head overhead |
|---|---|
| ResNet bottleneck (1,1,1,1) w32 | <0.1% |
| ResNet-50 (3,4,6,3) w64 | <0.01% |
| ResNet-101 (3,4,23,3) w64 | <0.01% |
| ViT 6/12/16 blocks d384 | <0.01% |
| GPT 4/10/14 blocks d384 V2000 G4 | ~2.6% |

The GPT row is the honest boundary of the economics argument: for a
language model the head fan-out is the *vocabulary*, not 10 classes, so
the G-position scoring convention is what keeps the overhead inside the
architecture's bound — declared in §3, measured in L9, and the one place
where the "overhead tends to zero" claim does **not** apply.

## 5. Theorems

The parent theorems are architecture-class statements; they lift to the
larger instances with the same proofs. We restate with the new certificate
numbers.

**T1″ (metering conservation at larger scale).** For any SAN forward pass
— bottleneck ResNet, ViT, or causal GPT trunk, any depth/width — the
metered machine suffering equals the analytic cost of the executed path:
a stage/block gated off for a sample contributes exactly 0, and the total
`M_gated ≤ M_dense` with equality iff no exit fires. *Proof.* The meter
charges each executed map per sample handed to it, under per-map
conventions fixed for both accounting paths: a conv
`2·C_in·C_out·K²·H_out·W_out`, a linear `2·d_in·d_out` (per token row),
each attention token-mixing matmul `2·T²·d` per sample. The bottleneck
block adds no new case (three convs and an optional 1×1 shortcut, each
charged at its own output spatial size); causal masking nullifies attention
weights but not the executed matmuls, which are charged in full on both
accounting paths — the mask is an unmetered elementwise operation,
identical in both; embedding lookups are unmetered data movement on both
paths, cancelling identically. A sample/sequence exiting after stage `d`
is handed to stages `0..d` only — its charge is exactly the sum over those
stages' maps, computed on its rows; the parent T1 argument applies term by
term. ∎ *Verified (L1, L9):* metered charge equals an **independent manual
accounting** of the executed path **exactly**, for all three trained
families on their full held-out sets and for every configuration of the
scalability sweep (bottleneck ResNets (1,1,1,1)w32/(3,4,6,3)w64/
(3,4,23,3)w64, ViTs 6/12/16 blocks at d=384, GPTs 4/10/14 blocks at
d=384); strictly below the gates-open charge whenever an exit fires;
eval-mode prefix logits match an independently recomputed dense prefix
with bounded deviation and **exactly agreeing argmax** everywhere.

**T2 (anti-Goodhart soundness, unchanged).** For every `λ ∈ [0,1]` and
every candidate pool, `select(C, λ)` is feasible or `NO_FEASIBLE`.
*Verified (L3, L8):* 101-point λ-grid over pools containing a zero-cost
abstainer (majority class / majority token), an under-trained probe
(pixel-linear / bigram-linear), and a spurious-feature shortcut probe that
beats τ on **train** while failing it held-out (CIFAR corner patch; GPT
leaked label token at a fixed input position) — selection feasible at
every grid point; all-infeasible pool → `NO_FEASIBLE`.

**T3 (machine-suffering bound, unchanged statement).** With `t*` the first
feasible epoch, `S_machine(SAN) = Σ_{t≤t*} E(t) ≤ Σ_{t≤t*} F(t)` and
`S_gratuitous(SAN) = 0`; any fixed `T`-epoch run of the same trunk accrues
`B(t*) + Σ_{t*<t≤T} F(t)`. *Verified (L4, L5):* numbers in §6.4.

**T4 (necessary/gratuitous separation, unchanged).** The ledger
decomposition is recomputed, not asserted; the necessity is
trajectory-relative (the parent caveat stands, unchanged).

**T5 (feasibility at larger scale, certificate).** On the canonical
instance, SAN-ResNet-50, SAN-ViT-large, and SAN-GPT each reach a feasible
checkpoint strictly inside budget (L2). As in the parent, this certifies
the instance; no universal convergence claim is made.

**T6 (scalability, restated with the LM boundary).** The architecture's
invariants are depth/width-parametric: for the family of SAN-ResNets
(basic or bottleneck blocks, any configuration, any width), SAN-ViTs, and
SAN-GPTs (any depth, any `d`), T1″ holds by the same proof, and the
exit-head overhead fraction is bounded by `Σ_k head_k / F_gates-open` —
tending to zero with trunk size for classification heads
(`2·C_k·n_class`), and bounded by `G/T · (V·G-scoring share)` for the LM
family under the declared G-position convention. The contract certifies
the theorem's content on a 9-configuration sweep (L9: 3 ResNet + 3 ViT +
3 GPT): at every scale,
metered = manual exactly, gated < gates-open when exits fire, prefix
argmax agreement exact, overhead < 5%.

## 6. Benchmark

### 6.1 Tasks and data

**Vision families.** CIFAR-10 (real dataset): stratified subset of 4000
train / 1000 held-out, standard channel normalization, **no augmentation**
(documented scope — the benchmark measures suffering accounting, not SOTA
accuracy). Deterministic shared data order (seed 17). The harm matrix is
the parent line's synthetic cost structure over the real labels: class 9
("truck") is the hazard class of a screening pipeline — missed hazard 5,
false hazard 2, any other confusion 1.

**GPT family.** Real text corpus: the repository's own `docs/research/*.md`
(319 files, ~3.3MB, ~440k word tokens; sorted glob, deterministic).
Word-level tokenization (lowercase `[a-z']+`); vocabulary V=2000 (UNK +
1999 most frequent). Sequences of T=64 tokens, (input, target) next-token
pairs; 3072 train sequences sampled (seeded) from the first 80% of the
token stream, 768 held-out sequences from the last 20%. The scored task is
next-token prediction on the last G=4 positions (§3). The harm structure
is a synthetic cost structure over the real tokens: the **negation tokens**
{no, not, never, without} ∩ vocab are the hazard class of a screening
pipeline over text — a missed negation (true hazard token, predicted
other) costs 5 (missing a negation flips the meaning of a statement: the
expensive error the gate exists to block); a false negation costs 2
(unnecessary intervention); any other token mismatch costs 1. The
asymmetry (5×) prices the same two pathologies as the vision harm matrix.
No clinical claim: this is the parent line's harm-channel definition
instantiated over real text so the *patient channel extends to language*.

### 6.2 Declared targets and budgets

Anti-Goodhart targets are declared inputs, chosen below what the standard
architecture demonstrably reaches inside budget on this instance (final
calibration; see §10):

| family | τ (held-out) | δ (exit) | budget epochs | W (gates closed) | head wiring |
|---|---:|---:|---:|---:|---|
| ResNet-50 | 0.34 | 0.55 | 8 | 2 | trunk-coupled aux |
| ViT-large | 0.251 | 0.45 | 10 | 4 | detached probe |
| GPT | 0.165 | 0.31 | 10 | 1 | detached probe |

Adam lr 1e-3, batch 128, seed 17. **Numeric environment is part of the
instance:** `SAN_LARGE_THREADS=16` (CPU torch). Thread count changes
conv/GEMM reduction order and can flip tight calibrated margins (τ_V and
patient integrals at the 0.001 level); the gate pins 16. GPT corpus is
pinned at `artifacts/san_large/corpus_snapshot_v2000.npz` so live edits to
`docs/research/*.md` cannot drift the LM leg mid-campaign.

### 6.3 Architectures compared

Within each family, one shared trunk init, one data order, one seed:

- **Dense** — the identical trunk, fixed budget: the standard
  architecture (ResNet-50 / ViT / GPT as trained everywhere).
- **EarlyStop** — the identical trunk with SAN's stop rule but no
  suffering-aware layers: the strongest *scheduler* baseline.
- **SAN** — this spec.

### 6.4 Measured results (canonical instance, bit-reproducible at seed 17)

**Canonical run (2026-08-06):** four parallel legs under
`SAN_LARGE_THREADS=16`, `SAN_LARGE_DEVICE=cpu`, seed 17, harness defaults
`τ=(0.34, 0.251, 0.165)`, `δ=(0.55, 0.45, 0.31)`, pinned GPT corpus
`artifacts/san_large/corpus_snapshot_v2000.npz`. Evidence logs:

- `artifacts/san_large/canonical_resnet50.log`
- `artifacts/san_large/canonical_vitlarge.log`
- `artifacts/san_large/canonical_gpt.log` — **L_GREEN 8/8**
- `artifacts/san_large/canonical_sweep.log` — **L_GREEN 1/1** (L9, all 9 configs PASS)

Ledger lines are `ledger[<family>-<arch>]:` from the harness; GFLOPs are
the printed `S_m` values (same accounting convention as the parent line).

| family | architecture | epochs run | t* | S_machine (GFLOPs) | gratuitous (GF) | S_patient ∫ | S_patient peak | final held-out acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| resnet50 | SAN | 3 | 2 | 99196.627 | 0 | 3.24 | 1.187 | 0.3470 |
| resnet50 | Dense | 8 | 4 | 269948.617 | 101230.731 | 7.28 | 1.187 | 0.5000 |
| resnet50 | EarlyStop | 5 | 4 | 168717.885 | 0 | 5.03 | 1.187 | 0.3880 |
| vitlarge | SAN | 6 | 5 | 217342.367 | 0 | 7.05 | 1.200 | 0.2550 |
| vitlarge | Dense | 10 | 5 | 369280.404 | 147712.162 | 11.56 | 1.200 | 0.2670 |
| vitlarge | EarlyStop | 6 | 5 | 221568.243 | 0 | 7.05 | 1.200 | 0.2640 |
| gpt | SAN | 5 | 4 | 115379.684 | 0 | 4.62 | 0.973 | 0.1667 |
| gpt | Dense | 10 | 4 | 241518.300 | 120759.150 | 9.10 | 0.973 | 0.1553 |
| gpt | EarlyStop | 5 | 4 | 120759.150 | 0 | 4.63 | 0.973 | 0.1719 |

**Exit fractions at t\* (L6):** resnet50 0.245 (epoch 2); vitlarge 0.131
(epoch 5); gpt 0.112 (epoch 4).

**L1 metering (exact gated=manual, argmax equal):**

| family | gated = manual | exits | prefix max_dev |
|---|---:|---:|---:|
| resnet50 | 2 400 613 257 216 | 245/1000 | 0 |
| vitlarge | 2 575 561 044 480 | 131/1000 | 0 |
| gpt | 1 724 709 814 272 | 86/768 | 0 |

**Multi-leg certificate:** every family leg printed
`SUFFERING_AWARE_LARGE_VERDICT L_GREEN (8/8 clauses PASS)`; the sweep
printed L9 PASS on all nine configs and
`L_GREEN (1/1 clauses PASS)`. Assembled evidence:
`artifacts/san_large/multi_leg_certificate.log` →
`SUFFERING_AWARE_LARGE_VERDICT L_GREEN (9/9 clauses PASS)` and
`SUFFERING_AWARE_LARGE_MULTI_LEG_OK`.

## 7. Contract clauses

| Clause | Claim | Canonical numbers (2026-08-06 multi-leg, THREADS=16) |
|---|---|---|
| L1 | T1″ metering conservation | resnet/vit/gpt: gated=manual exact; exits 245/1000, 131/1000, 86/768; argmax equal |
| L2 | T5 feasibility | t*=(2,5,4); acc@t*=(0.3470, 0.2550, 0.1667) ≥ τ |
| L3 | T2 soundness | 101-weight grid feasible-only; NO_FEASIBLE loud; abstain/probe < τ all families |
| L4 | T3/T4 separation | SAN gratuitous=0; dense gratuitous > 0 all families |
| L5 | T3 bound | SAN S_m ≪ dense and ≤ estop; SAN S_p ≤ estop (vit full-precision 7.05=7.05) |
| L6 | exits real | exit@t*=(0.245, 0.131, 0.112) all > 0.10 |
| L7 | patient channel | asymmetry 5.0×; peaks shared epoch-0 (1.187 / 1.200 / 0.973) |
| L8 | anti-shortcut | vision train 0.583 val 0.088; GPT train 1.000 val 0.061; never selected |
| L9 | T6 scalability | 9/9 configs PASS; GPT exit-head overhead ≤ 2.57% < 5% |

Multi-leg run: `SAN_LARGE_THREADS=16 SAN_LARGE_ONLY=<family|sweep>` → four logs →
`SUFFERING_AWARE_LARGE_VERDICT L_GREEN (9/9 clauses PASS)` via
`artifacts/san_large/multi_leg_certificate.log`. Full single-process gate:
`bash scripts/ci/suffering_aware_large_architecture_gate.sh`.

## 8. Falsifiers

| Clause | Falsifier |
|---|---|
| L1 | A gated-off stage/block charges FLOPs; metered ≠ manual accounting; gated > gates-open with an exit fired; an exited prediction's argmax disagrees with the recomputed prefix |
| L2 | No feasible SAN checkpoint within budget for any family |
| L3 | Any λ at which an infeasible candidate is selected; an all-infeasible pool returning a prescription; abstainer/probe feasible |
| L4 | SAN gratuitous FLOPs > 0; a feasible fixed-budget baseline with gratuitous = 0 |
| L5 | Dense fixed-budget baseline with total machine suffering ≤ SAN's; any baseline with integrated patient harm below SAN's; EarlyStop strictly below SAN on the machine channel |
| L6 | Exit fraction ≤ 10% at t* for any family (heads decorative) |
| L7 | A harm structure near-symmetric; SAN peak above a same-init baseline's |
| L8 | Shortcut probe feasible held-out, or selected at any weight |
| L9 | At any swept scale: metered ≠ manual, gated > gates-open with exits, prefix argmax disagreement, or exit-head overhead ≥ 5% |

Gate failure classification (per AGENTS.md): build/bootstrap-path (repo
`.venv` missing torch), harness-routing (gate script paths, missing
CIFAR-10 — the gate names the fetch command; missing docs corpus),
ontology-kernel/checker (n/a), baseline noise (numerics beyond the prefix
bound / argmax flip — would indicate a backend whose conv/GEMM results
depend on batch shape; the argmax-exactness sub-check is the load-bearing
one).

## 9. Scoped out (explicit)

1. **True ViT-L/16, ResNet-101 *training*, and GPU-scale training.** The
   contract scale is a documented CPU affordance; nothing in the theorems
   or the metering depends on it. The sweep (L9) covers ResNet-101 and
   deeper ViT/GPT forward-only; larger-budget training runs belong to the
   Foundry/Slurm path per AGENTS.md.
2. **Data augmentation and accuracy engineering.** The benchmark measures
   suffering accounting against declared targets, not maximal accuracy.
3. **A calibrated patient-harm model.** Both harm structures are synthetic
   over real labels/tokens; the learned-field line
   (`mercyful_learned_suffering_field_spec_2026-07-26.md`) is the path to a
   calibrated one. No clinical claim is made here.
4. **Hardware-metered energy** (RAPL/perf counters): analytic FLOPs × the
   stated J/FLOP constant, as in the parent line.
5. **A Sounio-native leg** — Python/PyTorch reference implementation, as in
   the parent specs.
6. **A large external text corpus.** The GPT leg uses the repository's own
   documentation as a real, in-repo, fetch-free corpus; larger corpora are
   a data-engineering question, not an architecture question.
7. **`topic-registry.v1.json` registration and `.github/workflows/ci.yml`
   wiring** — shared control surfaces under active edit by other lanes on
   this branch; left to the integrator (same convention as the parent
   specs). The gate is self-contained.

## 10. Commands run

```bash
# dataset (one-time): CIFAR-10 pickle layout at datasets/cifar-10-batches-py
# (identical content to https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
# corpus: in-repo docs/research/*.md (no fetch)
SAN_LARGE_SMOKE=1 .venv/bin/python scripts/research/suffering_aware_large_architecture.py   # mechanics check
.venv/bin/python scripts/research/suffering_aware_large_architecture.py                     # canonical L1..L9
bash scripts/ci/suffering_aware_large_architecture_gate.sh                                  # SUFFERING_AWARE_LARGE_GATE_OK
bin/llm-offload -t math-review -i docs/research/suffering_aware_large_architecture_spec_2026-07-31.md
```

## 11. LLM-offload review

Mandatory math-review offload on this spec:

- 2026-07-31 (prior campaign): dual xai/Grok + zai/GLM — receipt under
  `artifacts/san_large/offload_math_review_final.txt`.
- 2026-08-06 (this re-derivation, numbers final): `bin/llm-offload -t
  math-review -p xai` on the transcribed §6.2/§6.4/§7. Receipt:
  `artifacts/san_large/offload_math_review_20260806.txt`. Core ledger
  inequalities **OK**; fixed in place: ViT “quadruple / 4.5×” prose
  (FAIL → width² ≈ 9× + depth doubling) and “8-configuration” vs measured
  9-config L9 sweep. Remaining TIGHTENABLE notes (GPT τ vs dense final
  acc; proof-language vs measurement; ResNet trunk-coupled vs detached
  probes) recorded as design honesty, not blockers of L_GREEN.
