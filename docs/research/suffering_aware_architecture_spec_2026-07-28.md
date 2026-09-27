<!-- docs:meta
topic_id: repo.docs.research.suffering-aware-architecture-spec-2026-07-28
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.suffering-aware-architecture-spec-2026-07-28
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — the Suffering-Aware neural Network (SAN): an architecture that minimizes patient + machine suffering during training

**Date:** 2026-07-28
**Branch:** research/self-falsifying-compilation-line-20260726
**Status:** `EXECUTABLE` — contract A1..A8, `SUFFERING_AWARE_ARCHITECTURE_VERDICT A_GREEN (8/8)`
**Harness:** `scripts/research/suffering_aware_architecture.py`
**Gate:** `scripts/ci/suffering_aware_architecture_gate.sh` (**SUFFERING_AWARE_ARCHITECTURE_GATE_OK**)
**Parents:** `docs/research/mercyful_expanded_ethics_math_spec_2026-07-26.md`
(two-channel suffering, abstention trap T8),
`docs/research/mercyful_machine_channel_benchmark_spec_2026-07-26.md`
(measured machine channel, mu deciding structure),
`docs/research/mercyful-learning.md` (necessary vs gratuitous suffering,
mountain-pass level `c*`)

> **Scope.** All data, patients, and suffering values in this document are
> **synthetic constructions**. This is not medical guidance, not a treatment
> recommendation, and not a clinical decision-support tool. The "machine
> suffering" channel is an **operational computational-burden proxy**
> (measured FLOPs/energy): this work makes **no claim of machine
> consciousness, sentience, or phenomenology**, and no result below depends
> on one.

---

## 1. Position: from a mercyful scheduler to a mercyful architecture

The Mercyful Learning program so far built mercy *around* the network: a
scheduler that picks least-suffering paths over a suffering field (runtime
contract M1..M6), an expanded ethics that prices patient and machine
suffering in two semirings (E1..E8), and a structural benchmark in which the
machine channel decides width/depth (C1..C8). The network itself remained a
standard object trained by a standard loop.

This spec moves mercy **inside** the network. The Suffering-Aware Network
(SAN) is a neural architecture in which:

1. every layer computes, alongside its activation, its **suffering
   contributions** on both channels (§3);
2. **necessary** computation (required to reach the declared target) is
   architecturally separated from **gratuitous** computation (settled
   samples, post-target epochs) by per-sample exit gates and freeze-on-green
   (§4);
3. the **anti-Goodhart constraint is architectural**: feasibility (held-out
   performance ≥ τ) is a categorical property of checkpoints, enforced by a
   gate that filters before any cost comparison — not a penalty in the loss
   (§5);
4. **machine suffering is metered, not proxied**: exact analytic FLOP
   accounting of the executed path, with gated-off layers contributing
   exactly zero (§6).

The design rule throughout is the program's standing one, now applied to the
architecture itself: **constraints and gates, not penalties**. Nothing in
SAN's training loss prices suffering; the suffering reduction comes from the
structure (what gets executed) and the gate (what gets selected). A penalty
relaxation would reintroduce exactly the Goodhart trade the constraint
exists to block.

## 2. Definitions

**Definition 2.1 (suffering ledger).** A training run of epochs `t = 0..T`
produces, per epoch, a machine-suffering charge `m(t) ≥ 0` (metered FLOPs of
the training step plus the held-out evaluation, §6) and a patient-suffering
value `h(t) ≥ 0` (mean harm of the current model's predictions on a held-out
synthetic cohort-in-waiting, under the asymmetric harm matrix `H` of §7.1).
The run's ledger is

```
S_machine   = Σ_t m(t)                      (integrated machine suffering)
S_patient   = Σ_t h(t) ,  P_patient = max_t h(t)   (integral + peak)
```

the training-time analog of the expanded ethics' `A + μP` structure, here
reported as a pair rather than scalarized (the compassion-allocation weight
stays an explicit decision, per the expanded-ethics corollary).

**Definition 2.2 (feasibility, anti-Goodhart).** Given a declared target
`τ`, epoch `t` is **feasible** iff the checkpoint at `t` has held-out
performance `≥ τ`. Feasibility is categorical: an infeasible checkpoint is
prohibited as a prescription at every compassion-allocation weight, never
merely expensive.

**Definition 2.3 (necessary vs gratuitous suffering).** Let
`t* = min{ t : t feasible }`. Then

```
S_necessary  = Σ_{t ≤ t*} m(t)  (+ the patient channel over the same range)
S_gratuitous = Σ_{t > t*} m(t)
```

`S_necessary` is the training-ledger analog of the mountain-pass level `c*`
— with an honesty caveat carried into T4: `c*` is a property of the
*geometry* (min over paths), while `t*` is a property of the *optimizer
trajectory* (first hitting time of this run). `S_necessary` is therefore a
policy-relative necessity: the suffering this training procedure actually
required to first reach the target, not a proven minimum over procedures.

## 3. Suffering-aware layers

A **SufferingAwareLayer** is a `Linear → Tanh` block augmented with an exit
head (`Linear(width, n_classes)`). On every forward pass it computes three
things:

1. **Activation** — the ordinary trunk computation.
2. **Machine-suffering contribution** — exact analytic FLOPs charged to the
   meter for the samples it actually processes:
   `2·d_in·d_out` per sample for the core, `2·width·n_classes` for the exit
   head, `×3` in training — the fixed backward = 2× forward accounting
   convention adopted unchanged from the machine-channel benchmark (a
   stated convention for cross-architecture comparability, not a
   measurement of this backend's true backward cost). A sample routed
   around the layer by an upstream exit gate charges the layer
   **exactly 0**.
3. **Patient-suffering contribution** — predictions can be *emitted* at the
   layer's exit head; the harm of those predictions under `H` is owned by
   the layer at whose depth they were made. Each layer therefore carries the
   clinical cost of the predictions made at its depth, not just the final
   layer.

The loss is ordinary: cross-entropy at the final head plus, after warm-up,
deep supervision through the exit heads (every sample that traverses a layer
contributes that layer's exit-head logits to the loss — without this the
intermediate heads receive gradient only from samples that already exit, a
cold-start deadlock in which no head ever becomes confident enough for any
exit to fire; found empirically, fixed structurally). No suffering term
appears in the loss, by design (§1).

## 4. Necessary vs gratuitous: architectural separation

Two mechanisms, both architectural (properties of the forward pass and the
checkpoint structure, not of an external scheduler):

**Per-sample exit gates.** After each layer, a sample whose exit-head
confidence (max softmax) clears `δ = 0.75` leaves the network; the remaining
layers are gated off *for that sample* and meter zero. Once a sample's
prediction is settled, further depth for it is gratuitous suffering — the
architecture declines to pay it. Exit heads are trained by deep supervision
(§3); a three-epoch warm-up runs the trunk dense-identically to the baseline
first (running untrained exit heads would itself be gratuitous computation).

**Freeze-on-green.** Training stops at `t*`, the first feasible epoch.
Post-target training — margin-chasing past the declared target — is
gratuitous suffering on both channels (more FLOPs for the machine, more
exposure of the cohort-in-waiting to a changing model for the patient), and
the architecture declines it by construction: `S_gratuitous(SAN) = 0`
exactly (T3, clause A4).

## 5. Anti-Goodhart gating as an architectural constraint

Model selection over any candidate pool `C` of checkpoints/models, at any
compassion-allocation weight `λ ∈ [0,1]`:

```
select(C, λ) = argmin_{c ∈ C : feasible(c)}  (1-λ)·J_patient(c) + λ·J_machine(c)
             = NO_FEASIBLE   if the feasible subset is empty
```

Feasibility filters **before** cost comparison. Two Goodhart failure modes
are blocked categorically, not priced:

- **Under-dosing / abstention.** A zero-cost abstainer (majority class,
  0.645 held-out accuracy, `J_machine = 0`) and a cheap under-trained probe
  (0.281) minimize every scalarized suffering objective and are selected by
  it at every `λ` — the expanded ethics' abstention trap (T8 there), now at
  architecture level. The gate never selects them: infeasible is prohibited
  (clause A3, 101-point weight grid).
- **Shortcut prescriptions.** On a spurious-feature variant, a linear probe
  on the shortcut feature reaches **0.866 train accuracy > τ** — train-loss
  selection accepts it — while failing the held-out target (**0.586 < τ**).
  The gate rejects it at every weight (clause A8). Minimizing loss without
  reaching the target is not a prescription, exactly the anti-Goodhart axiom
  of the preprint carried into the architecture.

An all-infeasible pool returns `NO_FEASIBLE`: the architecture abstains
**loudly** rather than prescribing the least-bad failure — it cannot be
traded into pathology by any suffering discount.

## 6. Machine suffering metering

Machine suffering is **measured along the executed path**, not proxied by a
parameter norm (the `ρ‖θ‖²` proxy the machine-channel benchmark showed
carries ~0.03% of the objective). The meter charges per executed linear map
per active sample; exit gates shrink the active set, so the meter records
exactly the computation that happened. Energy is `FLOPs × 4e-12 J` (same
order-of-magnitude convention as the machine-channel benchmark). The
metered channel is real: SAN's total is 0.645 GFLOPs against the dense
baseline's 5.242 — a 8.1× dynamic range the norm proxy could not express.

## 7. Theorems

All theorems are stated for the architecture class and verified numerically
on the canonical benchmark instance by the contract (clauses in parentheses).
Proofs are one-paragraph structural arguments; the certificates are the
executable checks.

**T1 (metering conservation).** For any SAN forward pass, the metered
machine suffering equals the analytic cost of the executed path: a layer
gated off for a sample contributes exactly 0 to that sample's charge, and
the total metered charge `M_gated ≤ M_dense` (the same architecture run with
all gates forced open), with equality iff no exit fires. Moreover gating
does not alter the executed prefix: an exited sample's logits are the exit
head applied to the hidden state of the layers it actually traversed.
*Proof.* The meter charges `2·d_in·d_out` per layer per sample handed to
that layer; a sample exiting at depth `d` is handed to layers `0..d−1` only,
hence charged exactly for those; summing over samples gives the executed
path's cost and the inequality, since the dense run hands every sample to
every layer. The prefix statement is by construction of the forward pass.
∎
*Verified (A1):* metered charge equals an **independent manual accounting**
of the executed path exactly (6 567 040 FLOPs, both), strictly below the
dense-run charge (7 488 000) with 194/1000 samples skipping ≥ 1 layer; an
independently recomputed dense prefix reproduces the gated logits with max
deviation **0.0** (bound 1e-4) and exactly agreeing argmax predictions.

**T2 (anti-Goodhart soundness).** For every `λ ∈ [0,1]` and every candidate
pool, `select(C, λ)` is either feasible (held-out performance ≥ τ) or
`NO_FEASIBLE`. In particular no suffering advantage — including zero total
cost — can make an under-target candidate a prescription, on either channel
or any convex combination of them.
*Proof.* The selection domain is the feasible subset; candidates outside it
are removed before any cost is computed, so their costs never enter a
comparison. If the feasible subset is empty the function returns
`NO_FEASIBLE` by definition. ∎
*Verified (A3, A8):* 101-point λ-grid over a pool containing a zero-cost
abstainer and a cheap under-trained probe — selection feasible at every
grid point; all-infeasible pool → `NO_FEASIBLE`; shortcut probe (train 0.866
> τ, held-out 0.586 < τ) accepted by train-loss selection, rejected by the
gate at every weight.

**T3 (machine-suffering bound).** Let `t*` be SAN's first feasible epoch,
`F(t)` the per-epoch FLOPs of the same architecture with gates forced open,
and `E(t) ≤ F(t)` SAN's metered per-epoch charge (T1). Then

```
S_machine(SAN) = Σ_{t ≤ t*} E(t) ≤ Σ_{t ≤ t*} F(t) =: B(t*)    and    S_gratuitous(SAN) = 0,
```

while any fixed `T`-epoch run of the same trunk accrues
`S_machine(std) = Σ_{t ≤ T} F(t) = B(t*) + Σ_{t* < t ≤ T} F(t) ≥ B(t*)`,
whose gratuitous component `Σ_{t* < t ≤ T} F(t)` is exactly quantified by
the ledger. Freeze-on-green is what makes SAN's gratuitous term vanish; the
exit gates are what strictens the inequality `E(t) ≤ F(t)` inside the
necessary phase.
*Proof.* T1 gives `E(t) ≤ F(t)` per epoch; summing over `t ≤ t*` gives the
bound. `S_gratuitous(SAN) = Σ_{t > t*} E(t)` sums over an empty index set
because training stops at `t*`. The baseline identity is the same sum split
at its own first feasible epoch. ∎
*Verified (A4, A5):* `S_gratuitous(SAN) = 0` exactly; dense baseline
gratuitous 4.368 GFLOPs (83% of its total), ResNet 6.269 (92%);
`S_machine(SAN) = 0.645 GF < 0.874 (earlystop) < 5.242 (dense) < 6.839
(resnet)` — below **every** baseline. Precision, per the math-review
offload (§13): against EarlyStop the win is carried by the epoch count
(`t*` 6 vs 9 via deep supervision, T5) — SAN's average per-epoch charge
(92.2 MFLOPs) sits *between* the plain trunk's (87.4) and its own
gates-open charge (97.3), i.e. exits stricten `E(t) < F(t)` per T1 but at
the measured 19.4% exit rate do not fully amortize the exit-head overhead
within the training phase (they do at deployment, §8.3).

**T4 (necessary/gratuitous separation).** The ledger decomposition of
Definition 2.3 is correctly computed and correctly attributed: SAN's
gratuitous component is exactly zero, and every fixed-budget baseline that
reaches feasibility before its budget accrues a strictly positive gratuitous
component growing linearly in `T − t*`. Caveat, stated plainly (and carried
from Definition 2.3): this necessity is **trajectory-relative** — the
suffering this optimizer required to first reach τ — not the geometric
minimum `c*` of the mountain-pass theorem, which this work does not compute
for training trajectories.
*Verified (A4):* exact numbers in §8; the decomposition is recomputed from
the ledger by the contract, not asserted.

**T5 (convergence, modest and honest).** No universal convergence proof is
offered. What is proved-by-certificate on the canonical instance: with
shared-init warm-up and deep supervision, SAN reaches a feasible checkpoint
strictly inside the budget (`t* = 6 < T = 60`, held-out accuracy 0.802 ≥
τ = 0.80), **earlier** than the identically initialized dense trunk's own
first feasible epoch (`t* = 9`) — deep supervision acts as an auxiliary
gradient path that, on this instance, accelerates feasibility rather than
merely enabling exits. The contract certifies the instance; it does not
claim the acceleration is universal (clause A2).

## 8. Benchmark

### 8.1 Task

Synthetic 3-class dose-band classification: six synthetic patient covariates
(clearance, weight, sofa, age, crcl, albumin) → band ∈ {sub-therapeutic,
therapeutic, toxic} via a nonlinear synthetic exposure score with 4% label
noise; N = 4000 train / 1000 held-out. Not a pharmacokinetic model — a
synthetic classification task with a medical silhouette. The asymmetric harm
matrix prices the two pathologies the gate exists to block:

```
H[true, pred] = [[0,1,5],[2,0,2],[4,1,0]]   (over-dosing a sub-therapeutic
                                             patient: 5; under-dosing a
                                             toxic patient: 4)
```

Anti-Goodhart target τ = 0.80 held-out accuracy; budget T = 60 epochs, Adam
lr 1e-2, full batch, seed 17, bit-reproducible.

### 8.2 Architectures compared

- **DenseMLP** — the same trunk (width 32, depth 4, tanh), fixed 60-epoch
  budget. The standard architecture.
- **ResNetMLP** — residual variant, fixed budget. The standard-architecture
  control.
- **EarlyStopMLP** — the strongest *scheduler* baseline: identical trunk,
  identical stop-at-first-feasible rule as SAN, no suffering-aware layers.
  It isolates the architectural contribution: anything SAN beats it by comes
  from the layers, gates, and deep supervision, not from the stopping rule.
- **SAN** — this spec. SAN, DenseMLP, and EarlyStopMLP share one trunk init,
  so epoch-0 cohort exposure is identical across them and the patient-peak
  comparison (A7) is about trajectories, not init luck.

### 8.3 Measured results (canonical instance, bit-reproducible)

| architecture | epochs run | t* | S_machine (GFLOPs) | necessary | gratuitous | S_patient ∫ | S_patient peak | final held-out acc |
|---|---|---|---|---|---|---|---|---|
| **SAN** | 7 | 6 | **0.645** | 0.645 | **0.000** | **2.92** | 0.602 | 0.802 (≥ τ) |
| DenseMLP | 60 | 9 | 5.242 | 0.874 | 4.368 | 14.06 | 0.602 | 0.926 |
| ResNetMLP | 60 | 4 | 6.839 | 0.570 | 6.269 | 10.13 | 0.379 | 0.932 |
| EarlyStopMLP | 10 | 9 | 0.874 | 0.874 | 0.000 | 3.84 | 0.602 | 0.806 |

Read against the target, not against the margin:

- **Machine channel:** SAN spends **12.3%** of the dense baseline's FLOPs
  (87.7% saved), 9.4% of ResNet's, and **26.2% less than the early-stopping
  scheduler**. The honest decomposition of that last gap (caught and
  corrected by the math-review offload, §13): it is carried by the **epoch
  count**, not the per-epoch cost. SAN's average per-epoch charge is
  **higher** than EarlyStop's — the exit heads cost more per epoch than the
  exits save back at the measured 19.4% exit rate:

  | architecture | epochs run | avg MFLOPs/epoch | gates-open SAN equivalent |
  |---|---|---|---|
  | SAN | 7 | 92.2 | 97.3 (exits save 5.2 MF/epoch, 5.3%) |
  | EarlyStopMLP | 10 | 87.4 | — |
  | DenseMLP | 60 | 87.4 | — |
  | ResNetMLP | 60 | 114.0 | — |

  SAN wins the total **despite** the higher per-epoch price because deep
  supervision moves feasibility itself (t* 9 → 6, T5): `7 × 92.2 < 10 ×
  87.4`. What the exit gates buy is twofold and real: they stricten
  `E(t) < F(t)` against the gates-open SAN (T1), and at **deployment** —
  where the epoch-count term no longer exists — the trained gates make
  inference cheaper than the plain trunk's (gated eval 6.567 MFLOPs vs
  6.720 for the standard architecture, and vs 7.488 gates-open: 12.3% of
  the gates-open pass saved, 19.4% of samples skipping ≥ 1 layer, A6).
  Stated plainly: within this training budget the dominant mercy term is
  *reaching the target sooner*; the exits pay for most of their own
  overhead in training and all of it at deployment, with their margin
  growing with exit confidence.
- **Patient channel:** integrated cohort-in-waiting harm is **20.8%** of the
  dense baseline's (79.2% less), 28.8% of ResNet's, 24.0% less than
  EarlyStop's; the peak equals the shared epoch-0 exposure (0.602), never
  exceeded during training (A7).
- **Gratuitous suffering:** exactly zero for SAN; 4.368 and 6.269 GFLOPs
  (83% and 92% of their totals) for the fixed-budget baselines — the
  quantified price of training past the target (T3, T4).
- **The accuracy rows are the honest cost.** Dense/ResNet reach 0.926/0.932
  against SAN's 0.802. That excess is performance *past the declared target*
  — bought with 8–10× the machine suffering and 3.5–4.8× the patient
  exposure. If the clinically declared target were 0.92, τ must be set to
  0.92: **the target is an ethical input declared by the clinician, not a
  number the architecture may quietly relax or quietly exceed.** Mercy is
  defined relative to the declared target; the architecture enforces both
  directions of it (no under-target prescription, no gratuitous
  over-training).

## 9. Contract clauses

| Clause | Claim | Canonical numbers |
|---|---|---|
| A1 | T1 metering conservation: gated-off layers charge exactly 0; metered = independent manual accounting; < dense-run when exits fire; prefix unaltered | gated = manual = 6 567 040 FLOPs < 7 488 000 dense; 194+76(last-layer)/1000 exits; max prefix deviation 0.0; argmax exactly equal |
| A2 | T5 convergence: SAN reaches feasibility within budget | t* = 6 < 60, acc@t* = 0.802 ≥ 0.80 |
| A3 | T2 soundness: feasible-only selection on a 101-point λ-grid; loud NO_FEASIBLE | abstain 0.645, cheap probe 0.281, both < τ, never selected |
| A4 | T3/T4 separation: SAN gratuitous = 0; baselines' > 0 | SAN 0 FLOPs; dense 4.368 GF; resnet 6.269 GF |
| A5 | T3 bound: SAN total machine suffering below every baseline; integrated patient harm ≤ every baseline | 0.645 < 0.874 < 5.242 < 6.839 GF; 2.92 ≤ 3.84/10.13/14.06 |
| A6 | exits real, not decorative | 19.4% of held-out samples exit early at t* (> 10%); prefix deviation 0.0 |
| A7 | patient channel first-class: harm matrix asymmetric; SAN peak ≤ same-init baselines' | off-diag max/min = 5×; peak 0.602 ≤ 0.602 |
| A8 | anti-shortcut: train-loss selection accepts the shortcut, gate rejects at every weight | shortcut train 0.866 > τ; held-out 0.586 < τ |

Run: `.venv/bin/python scripts/research/suffering_aware_architecture.py` →
`SUFFERING_AWARE_ARCHITECTURE_VERDICT A_GREEN (8/8 clauses PASS)`
(bit-reproducible at seed 17).

## 10. Falsifiers

| Clause | Falsifier |
|---|---|
| A1 | A gated-off layer charges FLOPs, metered ≠ manual accounting, or an exited prediction disagrees with the recomputed prefix |
| A2 | No feasible SAN checkpoint within budget |
| A3 | Any λ at which an infeasible candidate is selected, or an all-infeasible pool returning a prescription |
| A4 | SAN gratuitous FLOPs > 0, or a feasible fixed-budget baseline with gratuitous = 0 |
| A5 | Any baseline with total machine suffering ≤ SAN's, or integrated patient harm below SAN's |
| A6 | Exit fraction ≤ 10% at t* (heads decorative), or prefix disagreement |
| A7 | Harm matrix near-symmetric, or SAN peak above a same-init baseline's |
| A8 | Shortcut probe feasible, or selected at any weight |

Gate failure classification (per AGENTS.md): build/bootstrap-path (repo
`.venv` missing torch), harness-routing (gate script paths),
ontology-kernel/checker (n/a), baseline noise (numerics beyond the 1e-4
prefix bound / argmax flip — would indicate a backend where GEMM results
depend on batch shape; the argmax-exactness sub-check is the load-bearing
one).

## 11. Scoped out (explicit)

1. **A geometric necessary-suffering minimum for training.** `S_necessary`
   is trajectory-relative (T4 caveat). Computing the training analog of the
   mountain-pass `c*` — a minimum over optimization procedures, not paths —
   is open.
2. **Universal convergence/acceleration guarantees** (T5 is a certificate
   for the canonical instance, not a theorem about deep supervision).
3. **Real clinical data and a calibrated patient-harm model.** `H` is a
   synthetic asymmetric matrix; the learned-field line
   (`mercyful_learned_suffering_field_spec_2026-07-26.md`) is the path to a
   calibrated one. No clinical claim is made here.
4. **Hardware-metered energy** (RAPL/perf counters): the machine channel is
   analytic FLOPs × a stated J/FLOP constant, consistent with the
   machine-channel benchmark; wall-clock correlation was established there
   and not re-measured here.
5. **A Sounio-native leg** — this is the Python/PyTorch reference
   implementation; a `.sio` port belongs to the mercyful-sounio line.
6. **`topic-registry.v1.json` registration and `.github/workflows/ci.yml`
   wiring** — both are shared control surfaces under active edit by other
   lanes on this branch; left to the integrator (same convention as the
   companion specs). The gate is self-contained and green.

## 12. Commands run

```bash
.venv/bin/python scripts/research/suffering_aware_architecture.py   # A_GREEN 8/8 (bit-reproducible)
bash scripts/ci/suffering_aware_architecture_gate.sh                # SUFFERING_AWARE_ARCHITECTURE_GATE_OK
bin/llm-offload -t math-review -i docs/research/suffering_aware_architecture_spec_2026-07-28.md
```

## 13. LLM-offload review

Mandatory math-review offload (dual xai/Grok 4.3 + zai/GLM-5.2 per M1
policy) run on this spec. Outcome: **PASS / ADDRESSED** —

- **Z.AI leg** (truncated at token cap, as in prior runs): independently
  recomputed every ratio and accounting identity in §8–§9 — 12.3%/9.4%/
  26.2% machine savings, 20.8%/28.8%/24.0% patient savings, 83%/92%
  gratuitous shares, 8.1× dynamic range, 3.5–4.8× exposure ratios, the
  epoch-indexing of `t*` (epochs 0..t* inclusive), the A1 FLOP identity
  (6 567 040 gated < 7 488 000 dense, 87.7%), and the 194+76 exit
  bookkeeping — all correct. One genuine **[WRONG]** caught and ADDRESSED:
  the first draft attributed the 26.2% gap to EarlyStop to "per-sample
  exits shrink[ing] the executed path inside every epoch" — false as
  stated, because SAN's average per-epoch charge (92.2 MFLOPs) is *higher*
  than EarlyStop's (87.4): the exit heads cost more per epoch than the
  exits save at the measured 19.4% exit rate. The gap is carried by the
  epoch count (t* 6 vs 9, deep supervision, T5). §8.3 and the T3 verified
  note now carry the honest per-epoch decomposition table and the
  corrected attribution (exits stricten `E(t) < F(t)` against gates-open
  SAN and win at deployment; the training-phase win is the earlier t*).
- **Grok leg** (first fan-out leg returned the 34-byte "NO MATHEMATICAL
  CONTENT TO REVIEW" non-response seen in prior runs; retried successfully
  as a single-provider leg on the corrected spec): `[OK]` on every item —
  Defs 2.1–2.3, T1 metering/prefix invariance, T2, T3, the T4
  trajectory-relative caveat, T5's certificate-only scoping, and all A1–A8
  numerical clauses. One [TIGHTENABLE] ADDRESSED: the `×3` training
  multiplier is now explicitly "the fixed backward = 2× forward accounting
  convention adopted unchanged from the machine-channel benchmark", not a
  `≈` approximation claim about this backend.
- Contract `A_GREEN 8/8` and gate `SUFFERING_AWARE_ARCHITECTURE_GATE_OK`
  re-run green after all edits. Full entry in `.claude/llm_offload_log.md`
  (2026-07-28 row). Raw: `/tmp/llm-offload-Wc0UC4/` (zai + grok first leg),
  `/tmp/llm-offload-JJ2486/` (grok retry).
