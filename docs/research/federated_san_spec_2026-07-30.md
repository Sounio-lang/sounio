<!-- docs:meta
topic_id: repo.docs.research.federated-san-spec-2026-07-30
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.federated-san-spec-2026-07-30
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Mercyful Learning — FED-SAN: the Suffering-Aware neural Network FEDERATED across clinical sites, with a distributed suffering ledger and a gated aggregator

**Date:** 2026-07-30
**Branch:** research/self-falsifying-compilation-line-20260726
**Status:** `EXECUTABLE` — contract F1..F9, `FEDERATED_SAN_VERDICT F_GREEN (15/15)`
**Harness:** `scripts/research/federated_san.py`
**Gate:** `scripts/ci/federated_san_gate.sh` (**FEDERATED_SAN_GATE_OK**)
**Parents:** `docs/research/suffering_aware_architecture_spec_2026-07-28.md`
(SAN: suffering-aware *architecture*, contract A1..A8 — definitions,
theorems T1..T5, selection rule),
`docs/research/suffering_aware_deep_architecture_spec_2026-07-28.md`
(deep SAN on real images, D1..D9 — per-family δ precedent),
`docs/research/san_real_patient_data_spec_2026-07-28.md`
(SAN on real patient data, R1..R10 — the WDBC cohort, the declared 5:1
harm structure, the real-outcome suffering field)

> **Scope.** The clinical leg's patients, features, and outcomes are
> **real** (569 real breast fine-needle-aspirate patients, WDBC, UCI #17,
> de-identified, public, CC-BY); the vision leg's images are **real**
> (CIFAR-10) with a **synthetic screening cost structure over the real
> labels** (the deep line's convention). The harm **weights** (missed
> hazard = 5, false hazard = 1) remain a **declared normative cost
> structure** — a harm weighting is an ethical input, not a measurable
> quantity. This is not medical guidance, not a treatment recommendation,
> not a diagnostic or screening tool, and no model produced here is fit
> for any clinical use. The "machine suffering" channel is an
> **operational computational-burden proxy** (metered FLOPs and wire
> bytes): this work makes **no claim of machine consciousness, sentience,
> or phenomenology**, and no result below depends on one. All data is
> de-identified and public without credentialing.

---

## 1. Position: does the architecture survive distribution?

The SAN line established the architecture class — suffering-aware layers,
per-sample exit gates, deep supervision, freeze-on-green, and an
architectural anti-Goodhart gate — and scaled it from a synthetic MLP
(A1..A8) to deep networks on real images (D1..D9) and to real patient
cohorts (R1..R10). Everything so far was *centralized*: one training set,
one model, one ledger. This spec distributes the same architecture class
across **K federated nodes** — the deployment shape clinical ML actually
has, where patient data may not leave the site — and asks the four
questions federation poses to the design:

1. **Ledger.** Suffering now accrues on K nodes *and on the wire*. The
   metering-conservation theorem must extend to a sum of per-node compute
   meters plus an exact communication ledger (broadcast + upload per node
   per round). It does, exactly (T1, F1).
2. **Freeze-on-green.** Feasibility is now a property of the *aggregated*
   model, evaluated on the aggregator's trusted held-out set. Freezing at
   the first feasible round `r*` must still make gratuitous suffering
   exactly zero — on every node and on the wire (T4, F4).
3. **The gate faces an adversary the centralized line never had.** A
   federated node can be *adversarial*: it can submit a poisoned update.
   The anti-Goodhart gate must move into the aggregator — and a one-level
   gate is not enough (§5.1, found empirically, fixed structurally).
4. **The economics of federation itself.** Distribution costs compute
   (aggregator evaluations) and wire bytes; the contract measures the
   price of federation against the centralized SAN (§8.3).

The answer, certified by the contract: the architecture federates.
Nothing in the layer design changes; what changes is where the gate sits
and what the ledger sums over.

### 1.1 Why federated, and why not ImageNet/GPT-2 (feasibility finding)

The task that commissioned this spec offered four scaling options:
ImageNet-scale SAN (A), GPT-2-scale SAC-LLM (B), multi-modal SAN (C),
federated SAN (D). An honest inventory of this environment:

- **No GPU** (CPU-only torch, 64 cores); ImageNet-scale training from
  scratch is weeks-to-months of compute, and the ImageNet dataset itself
  requires a signed license — not fetchable here.
- **GPT-2-scale from-scratch training** on CPU is feasible only at token
  counts too small to make the clinical-target claim meaningful, and the
  SAC-LLM's constraint-table gate requires structured templated data at
  any scale — scaling the *template* does not scale the *claim*.
- **Federation is the scaling axis that is real here**: multiple nodes,
  non-IID real patient data, an adversarial node, and a distributed
  ledger — a genuinely new architectural question (the gate under
  adversarial updates), not merely a bigger instance of an answered one.

Option D was chosen on that basis. The choice and its rationale are
stated here because the commissioning task asked for the most feasible
*and impactful* option, and the honest answer is that A and B cannot
produce trustworthy certificates in this environment.

## 2. What is reused unchanged

From the parent specs, without modification: the suffering ledger
(Definition 2.1 of the A-line: `S_machine`, `S_patient` integral + peak),
feasibility as a categorical anti-Goodhart constraint, the
necessary/gratuitous decomposition at the first feasible point with its
trajectory-relative honesty caveat, the selection rule `select(C, λ)`
with loud `NO_FEASIBLE`, the metering convention (linear = `2·d_in·d_out`
FLOPs/sample, conv = `2·Cin·Cout·K²·H·W`, backward = 2× forward, energy =
FLOPs × 4e-12 J, elementwise ops unmetered), the per-sample exit gates
with deep supervision after a dense-identical warm-up, the declared 5:1
asymmetric harm structure, and the design rule **constraints and gates,
not penalties** — no suffering term appears in any training loss.

## 3. The federated setting

**Nodes and data.** K simulated sites (one process, seeded and
deterministic — a standard FL simulation; the certificates are properties
of the protocol, not of the process count). Each node's data is a
**Dirichlet label-skew partition** (α = 0.5) of the training pool — the
standard non-IID heterogeneity model; the realized skew is certified
(F9): on the clinical leg the mean per-node label-distribution L1
distance from the global distribution is **0.790** (nodes of 67/32/34/72/
195 patients out of 400) — an extreme heterogeneity regime, far beyond
IID.

**Legs.**
- **Clinical (real patients):** WDBC, 569 real patients, the R-line's
  split (400 train / 169 trusted held-out) and standardization; hazard =
  biopsy-confirmed malignancy; declared τ = 0.95, δ = 0.75 (the R-line's
  WDBC constants); K = 5 nodes; the SAN is the A/R-line MLP (width 32,
  depth 4, tanh, exit head per layer).
- **Vision (real images):** CIFAR-10, 4000 train / 1000 trusted held-out
  (seeded subset, the deep line's size); hazard = class 9 ("truck") under
  the deep line's 5:1 screening convention; declared τ = 0.40, δ = 0.30
  (declared per family, the deep/R-line precedent: τ above the 0.10
  abstainer and below the trunk's demonstrated 0.429, a mercy target not
  a SOTA target; δ set where the family's gates actually fire — at
  δ = 0.40 the averaged heads' confidence on this trunk tops out under
  7% exits, a real-data calibration finding of this work); K = 4 nodes;
  the SAN is a compact 4-stage conv trunk (~50k parameters, stem 3→16,
  stages 16/32/48/64, stride-2, GAP+linear exit head per stage).

**The federated loop.** One shared global init (seeded, identical across
all compared systems; the plain-trunk baseline's init is *copied* from
the SAN's trunk so round-0 exposure — and hence the patient peak — is
identical across architectures, the A/R-line convention). Per round:
broadcast; **one local epoch per node** (full batch clinical, seeded
128-batches vision, fresh Adam per round — the standard FedAvg
convention); aggregation (§5); global feasibility evaluation on the
aggregator's trusted held-out set. Round 1 is the warm-up round: every
trunk trains dense-identically to the plain baseline (untrained exit
heads would be gratuitous computation); exit gates and deep supervision
activate from round 2. Freeze-on-green stops the protocol at `r*`, the
first round whose accepted global model clears τ.

**The two-level ledger.** Per round, the machine channel meters (i) every
node's local training FLOPs exactly (executed-path accounting; exit gates
shrink the active set and gated-off layers charge exactly 0), (ii) the
aggregator's evaluation FLOPs (per-update scoring + global evaluation —
the price the gate itself pays, charged honestly), and (iii) the
communication ledger: `2 · K · model_bytes` per round (broadcast +
upload), where `model_bytes` counts float32 parameters and buffers. An
excluded update is still uploaded (the gate reads it at the aggregator):
exclusion saves suffering through the *average*, not through the wire —
stated so the communication bound in T3 is exact, not optimistic.

## 4. Definitions carried to federation

**Definition 4.1 (federated suffering ledger).** A federated run of
rounds `r = 0..R` (round 0 = evaluation of the shared init) produces, per
round, a compute charge `m(r) ≥ 0` (sum of all node training FLOPs and
aggregator evaluation FLOPs) and a communication charge `c(r) = 2 · K ·
model_bytes`, and a patient-suffering value `h(r) ≥ 0` (mean harm of the
accepted global model's predictions on the trusted held-out cohort). The
ledger is the triple

```
S_machine = Σ_r m(r) ,   S_comm = Σ_r c(r) ,   S_patient = Σ_r h(r), P = max_r h(r)
```

reported as a triple, not scalarized (FLOPs and bytes are different
burdens; the compassion-allocation weight stays an explicit decision).

**Definition 4.2 (federated feasibility, anti-Goodhart).** Given declared
τ, round `r` is **feasible** iff the accepted global model at `r` has
trusted-set performance ≥ τ. Feasibility is categorical, unchanged.

**Definition 4.3 (necessary vs gratuitous, federated).** Let
`r* = min{ r : r feasible }`. Then `S_gratuitous = Σ_{r > r*} m(r)` (and
likewise for `c(r)`) — the rounds the protocol pays *past the declared
target*, on every node and on the wire. The trajectory-relative caveat is
inherited unchanged: `r*` is the first hitting time of *this* federated
optimizer trajectory, not a minimum over protocols.

## 5. The gated aggregator: the anti-Goodhart constraint, moved

Centralized SAN gates *model selection*; a federated system must also
gate *aggregation* — the point where an adversarial or merely harmful
update enters the global model. FED-SAN's aggregator applies the line's
standing rule at **two levels**, both categorical:

**Update-level inclusion bar.** Before averaging, every node's update is
evaluated on the aggregator's trusted held-out set (the evaluation is
metered). An update is included iff

```
acc(update) ≥ abstainer_acc + 0.02   OR   acc(update) ≥ best_global_acc − 0.05
```

— it must beat doing nothing, or stay near the best the federation has
achieved. An under-bar update is *prohibited from the average*, never
penalized. The second disjunct keys on the **best** global accuracy so
far, never the last round's: an attack that drags the global down must
not drag the bar down with it.

**Round-level acceptance gate.** The average itself is then evaluated: an
average that cannot beat the abstention bar, or that would move the
global model more than 5 points below its best-so-far, is **never
accepted** — the architecture declines to move backward. On evidence of
poisoning the gate **escalates once**: the weakest included update is
dropped and the remainder re-averaged; if the re-average still fails, the
round is rejected and the previous global stands. Escalation is still
categorical — the dropped update is prohibited from *this* average, not
priced.

### 5.1 Why two levels (a found failure, fixed structurally)

The first draft of this aggregator had only the inclusion bar and a
reject-and-freeze acceptance. Against a **strong model-poisoning
adversary** — node 0 trains its poison to *local convergence* on
label-flipped data (10 epochs) and amplifies its update 5× — the
contract's own attack leg found two real failure modes:

1. **Stealthy poison.** From a strong global model, the submitted poison
   is not crude. The exact r5 triple, measured: the unamplified poisoned
   model `w` has trusted-set accuracy **0.213** (harm 2.254) — but the
   adversary does not submit `w`; it submits the amplified extrapolated
   state `g + 5(w − g)`, and **that is the object the inclusion bar
   evaluates**: its trusted-set accuracy is **0.805**, clearing the
   abstention bar (0.647), while the average it enters collapses to
   **0.231** (the average `g + 0.8375(w − g) + …` behaves like `w`;
   honest-only average 0.911). Weight-space extrapolation is not
   accuracy-monotone: `g`, `w`, and `g + 5(w−g)` sit at 0.929 / 0.213 /
   0.805. A one-level bar that scores the submitted update catches crude
   poisons (rounds 1–4: excluded at 0.08–0.12) and misses this one.
2. **Reject-and-freeze deadlock.** Once the poisoned average is rejected,
   the unchanged global reproduces the *identical* poisoned average every
   round: the federation is safe but permanently stuck (0.929 < τ,
   r* = ∞).

The two-level gate with one-step escalation is the structural fix:
certified on the real WDBC cohort, the poisoned node is excluded in
**every** round (bar or escalation), participates in **no** accepted
average, the gated federation reaches feasibility (r* = 8, acc 0.953)
while naive FedAvg never does (30 rounds, final acc 0.266, final harm
2.201 — **worse than the abstainer's** 1.864), and the accepted global is
regression-bounded throughout (F8). The honest scope note: this is a
certified instance against a certified attack, not a general
Byzantine-robustness theorem (§11).

## 6. Machine and patient suffering metering under federation

Metering conventions are the line's, unchanged; what is new is exactness
*across* nodes and on the wire. The contract's F1 certificate: the sum of
per-node compute meters equals an independent manual accounting of the
executed path **exactly** (integer FLOP arithmetic on both legs), the
communication meter equals the manual `rounds × 2K × model_bytes`
accounting exactly, gated-off layers charge exactly 0, and the exited
predictions of the *aggregated* global model agree with an independently
recomputed dense prefix (max deviation ≤ 1.2e-7, argmax exactly equal) —
prefix invariance survives FedAvg, because every system shares one init
and the prefix check is a property of the deployed model, not of how it
was trained.

## 7. Theorems

**Epistemic status** (the SAC-LLM convention, adopted after its math
review): T1–T4 are *conditional* statements whose deductive content is
symbolic (sums of exact meters; masking/inclusion by construction;
termwise inequalities on non-negative sums; the empty sum), while their
antecedents — target reachability within budget, the strictness witnesses
`r* < R`, exclusion/escalation events, constraint soundness — are
**empirical certificates** (F1–F9) or stated assumptions, not theorems.
No asymptotic or distribution-free claim is made.

**T1 (metering conservation under federation).** For any FED-SAN run,
the metered compute equals the analytic cost of the executed path summed
over nodes and rounds: a layer gated off for a sample charges exactly 0,
and the metered communication equals `2·K·model_bytes` per executed
round. *Proof.* Each node's meter is the A-line meter applied to its
local executed path (exact by the parent T1); the round total is a sum of
exact integers; the protocol total is a sum over executed rounds; the
comm meter adds the same fixed quantity the manual accounting adds, once
per round. ∎ *Verified (F1):* clinical compute 532 864 = 532 864 FLOPs,
comm 1 077 600 = 1 077 600 bytes; vision compute 5 620 123 072 =
5 620 123 072 FLOPs, comm 20 565 376 = 20 565 376 bytes; prefix argmax
exactly equal on both legs (max deviation 1.2e-7 clinical, 0.0 vision).

**T2 (gated-aggregation soundness).** *Assume (A2) the trusted held-out
set is honest and (A3) the abstainer reference is computable on it.* Then
(i) selection over any candidate pool is feasible-only at every
compassion weight, with loud `NO_FEASIBLE` (the A-line rule, unchanged);
(ii) every accepted global model is an average of updates that
individually cleared the inclusion bar and jointly cleared the acceptance
bar; (iii) the accepted global's trajectory is regression-bounded: once
it first clears the abstention bar, no accepted round sits more than 5
points below the running best — by construction of the acceptance test.
*Proof.* (i) is the parent T2. (ii) is by construction: inclusion and
escalation filter before averaging, acceptance filters after. (iii) an
average failing `acc ≥ max(abst+0.02, best−0.05)` is never accepted, so
the accepted sequence can only move within the bar or up. ∎ *Verified
(F3, F8):* 101-point grid feasible-only; the strong poisoning adversary
participates in no accepted average; regression-bounded holds on the
attack trajectory; naive FedAvg (no gates) ends infeasible and worse
than the abstainer.

**T3 (suffering bounds).** Let `r*` be FED-SAN's first feasible round and
`m(r), c(r)` the per-round charges. Then
`S_machine(FED-SAN) = Σ_{r ≤ r*} m(r) ≤ Σ_{r ≤ R} m(r) =
S_machine(fixed-round)`, and likewise for `S_comm`, with strictness
whenever `r* < R` — because `m, c ≥ 0` and freeze-on-green executes no
round past `r*`. *Proof.* Termwise on non-negative sums; the fixed-round
baseline runs the same protocol without the freeze. ∎ *Verified (F4,
F5):* clinical 0.086 < 0.209 GF and 1.08 < 5.39 MB; vision 1262.074 <
1447.904 GF and 20.57 < 23.73 MB; gratuitous compute and communication
exactly 0 for FED-SAN on both legs, positive for the fixed-round
baseline on both legs (0.122 GF/4.31 MB clinical; 185.830 GF/3.16 MB
vision).

**T4 (necessary/gratuitous separation, federated).** The ledger
decomposition of Definition 4.3 is correctly computed and correctly
attributed: FED-SAN's gratuitous component is exactly zero on compute
*and* wire; any fixed-round protocol that reaches feasibility before its
budget accrues a strictly positive gratuitous component that grows in
`R − r*` — each post-`r*` round adds its own positive metered compute
and wire charge (the per-round charge varies with the active set, so
the growth is monotone in the round count, not a constant-rate linear
function of it; tightened per the math review). The trajectory-relative
caveat of Definition 4.3 is inherited. *Verified (F4):* exact numbers in
§8.3; the decomposition is recomputed from the per-round meter by the
contract, not asserted.

**T5 (convergence, modest and honest).** No universal convergence proof
is offered. What is proved-by-certificate on both canonical instances:
with the shared-init warm-up and federated deep supervision, FED-SAN
reaches a feasible accepted checkpoint strictly inside the round budget
under **certified extreme label skew** (clinical r* = 6 < 30 at mean
per-node L1 skew 0.790; vision r* = 13 < 15 at skew 0.765). The contract
certifies the instances; it does not claim acceleration — indeed the
vision leg shows federated deep supervision *delaying* feasibility
relative to the plain trunk (r* 13 vs 8, §8.3) — and it does not claim
universality.

## 8. Benchmark

### 8.1 Systems compared

| system | protocol | aggregation | stop rule |
|---|---|---|---|
| **FED-SAN** | exit gates + deep supervision from round 2 | two-level gated | freeze-on-green at r* |
| FedFixed | identical to FED-SAN | two-level gated | fixed R rounds |
| FedEarlyStop | plain trunk, no exit heads, no deep supervision | two-level gated | freeze-on-green at r* |
| (attack leg) | FED-SAN protocol, node 0 = strong poisoner | naive vs gated | fixed R rounds |
| CentralSAN (finding only) | the R-line centralized SAN on the pooled 400 patients | — | freeze-on-green |

FedEarlyStop is the strongest *scheduler* baseline — same stop rule, no
suffering-aware layers; anything FED-SAN beats it by comes from the
layers and gates, not the stopping rule.

### 8.2 Canonical configuration

Clinical: K = 5 nodes, α = 0.5, R = 30, lr 1e-2, full batch, δ = 0.75,
τ = 0.95, seed 17, bit-reproducible. Vision: K = 4 nodes, α = 0.5,
R = 15, lr 3e-3, batch 128, δ = 0.30, τ = 0.40, seed 17,
bit-reproducible. Attack leg (clinical only): node 0 poisons to local
convergence (10 epochs) on flipped labels, update amplified 5×.

### 8.3 Measured results (canonical instances, bit-reproducible)

**Clinical leg (569 real WDBC patients, K = 5, skew 0.790):**

| system | rounds | r* | S_machine | S_comm | S_patient ∫ | peak | final acc |
|---|---|---|---|---|---|---|---|
| **FED-SAN** | 6 | 6 | **0.086 GF** | **1.08 MB** | 3.91 | 1.893 | 0.9527 (≥ τ) |
| FedFixed | 30 | 6 | 0.209 GF (grat 0.122) | 5.39 MB (grat 4.31) | 9.13 | 1.893 | 0.9467 (< τ) |
| FedEarlyStop | 6 | 6 | 0.110 GF | 1.01 MB | 3.88 | 1.893 | 0.9527 |

- **Machine channel:** FED-SAN spends **41%** of the fixed-round
  protocol's compute (0.086 vs 0.209 GF) and **20%** of its wire (1.08
  vs 5.39 MB); and **22% less compute than the EarlyStop scheduler**
  (0.086 vs 0.110 GF) — here, unlike the A-line instance, the per-round
  win *is* carried by the exits (per-round 14.4 vs 18.4 MFLOPs), the
  round counts being equal (r* = 6 both).
- **Patient channel:** integrated exposure is **43%** of the fixed-round
  protocol's (3.91 vs 9.13); the peak equals the shared round-0 exposure
  (1.893) on all systems and is never exceeded. **The honest EarlyStop
  decomposition** (the A-line convention, printed by the harness as
  `F5_decomp`): FED-SAN's integral is 0.04 *above* EarlyStop's (3.91 vs
  3.88) — mid-training (rounds 2–5), exited predictions come from exit
  heads that transiently lag the averaged trunk under extreme skew
  (per-round harms 0.402/0.254/0.219/0.272 vs 0.373/0.249/0.249/0.243);
  at r* the harms coincide (0.213). The contract certifies the peak and
  the r* exposure and reports the transient; it does not hide it.
- **Over-training loses feasibility, not just mercy:** the fixed-round
  protocol's final global model (0.9467) sits *below* τ — under extreme
  non-IID averaging, margin-chasing past r* drifts the averaged model;
  freeze-on-green preserves the feasible checkpoint.
- **The price of federation:** CentralSAN on the pooled 400 patients
  reaches τ in t* = 4 epochs at 0.055 GF. FED-SAN pays **1.57×** the
  compute plus 1.08 MB on the wire — the measured cost of keeping 569
  real patients' data at their sites.

**Attack leg (clinical, strong poisoner):** naive FedAvg never reaches
feasibility (30 rounds, final acc 0.266, final harm 2.201 ≥ abstainer
1.864 — the poisoning works); the two-level gated FED-SAN excludes the
poison in every round (bar or escalation), admits it to no accepted
average, reaches feasibility (r* = 8, acc 0.953, harm 0.237), and is
regression-bounded throughout. The attack costs the federation 2 extra
rounds — the honest nodes carry 4/5 of the data and still get there.

**Vision leg (4000 real CIFAR-10 images, K = 4, skew 0.765, nodes of
408/1524/1313/755 images):**

| system | rounds | r* | S_machine | S_comm | S_patient ∫ | peak | final acc |
|---|---|---|---|---|---|---|---|
| **FED-SAN** | 13 | 13 | **1262.074 GF** | **20.57 MB** | 6.14 | 0.480 | 0.4190 (≥ τ) |
| FedFixed | 15 | 13 | 1447.904 GF (grat 185.830) | 23.73 MB (grat 3.16) | 6.99 | 0.480 | 0.3790 (< τ) |
| FedEarlyStop | 8 | 8 | 808.234 GF | 12.28 MB | 3.92 | 0.481 | 0.4060 |

- **Machine channel:** FED-SAN spends **87.2%** of the fixed-round
  protocol's compute (1262 vs 1448 GF) and **86.7%** of its wire (20.57
  vs 23.73 MB); its average **per-round** compute is below EarlyStop's
  (97.1 vs 101.0 GF/round — the exits stricten every executed round, T1).
  **The honest EarlyStop decomposition, vision leg** (printed as
  `F5_decomp`): FED-SAN's *total* compute is **1.56× EarlyStop's**
  (1262 vs 808 GF) and its integrated patient exposure is **2.22 above**
  (6.14 vs 3.92) — carried entirely by the round count (**r\* = 13 vs
  8**): federated deep supervision under extreme label skew (0.765)
  *delays* feasibility — the aux gradients couple each node's shared
  trunk to its node-local label distribution — where centralized deep
  supervision accelerated it (the A-line's T5: t\* 6 vs 9). The clinical
  leg shows the same coupling at negligible cost (r\* = 6 both); the
  vision leg shows it can dominate. This is the headline limitation of
  the federated instance, stated in §10 and left in the contract output.
- **Patient channel:** integrated exposure is **87.8%** of the
  fixed-round protocol's (6.14 vs 6.99); the peak equals the shared
  round-0 exposure (0.480) and is never exceeded (EarlyStop's peak, 0.481,
  transiently *exceeds* its own round-0 at round 2).
- **Over-training loses feasibility again:** the fixed-round protocol's
  final global (0.3790) sits below τ — the feasibility drift found on
  the clinical leg replicates under image-scale non-IID averaging;
  freeze-on-green preserves the feasible checkpoint (0.4190 at r\* = 13).

### 8.4 Contract clauses

| Clause | Claim | Canonical numbers |
|---|---|---|
| F1[leg] | T1: metered == manual exactly (compute + comm); gated-off layers charge 0; prefix argmax exactly equal | clinical 532 864 FLOPs / 1 077 600 B; vision 5 620 123 072 FLOPs / 20 565 376 B, prefix dev 0.0 |
| F2[leg] | T5: feasibility within round budget under certified skew | clinical r* = 6 ≤ 30 at 0.9527 ≥ 0.95; vision r* = 13 ≤ 15 at 0.4190 ≥ 0.40 |
| F3 | T2(i): feasible-only selection, 101-point λ-grid; loud NO_FEASIBLE | abstain 0.627, 1-round probe 0.864, both < τ |
| F4[leg] | T4: FED-SAN gratuitous = 0 FLOPs and 0 bytes exactly; fixed-round > 0 on both | clinical 0 vs 0.122 GF / 4.31 MB; vision 0 vs 185.830 GF / 3.16 MB |
| F5[leg] | T3: FED-SAN compute < fixed-round; per-round ≤ EarlyStop; S_patient ≤ fixed-round (EarlyStop totals reported with decomposition) | clinical 0.086 < 0.209, 14.4 ≤ 18.4 MF/round, 3.91 ≤ 9.13; vision 1262 < 1448, 97.1 ≤ 101.0 GF/round, 6.14 ≤ 6.99 |
| F6[leg] | exits real under federation: held-out exit fraction at r* > 0.10, prefix exact | clinical 0.905; vision 0.176 |
| F7[leg] | patient channel first-class: 5:1 asymmetry; FED-SAN peak ≤ same-init baselines' | peaks 1.893 = 1.893 = 1.893 (clinical); 0.480 ≤ 0.480/0.481 (vision) |
| F8 | adversarial containment: poison in no accepted average; naive infeasible and worse than abstainer; gated feasible; regression-bounded | excluded 8/8 rounds; naive 0.266/2.201; gated r* = 8 at 0.953 |
| F9 | non-IID realism + provenance: certified skew; WDBC counts match published | L1 0.790 clinical / 0.765 vision; 569 = 357 + 212 MATCH |

Run: `.venv/bin/python scripts/research/federated_san.py` →
`FEDERATED_SAN_VERDICT F_GREEN (15/15 clauses PASS)` (bit-reproducible at
seed 17).

## 9. Falsifiers

| Clause | Falsifier |
|---|---|
| F1 | Any meter ≠ manual accounting (compute or comm), or a gated-off layer charging FLOPs, or a prefix/argmax disagreement in the aggregated model |
| F2 | No feasible accepted checkpoint within budget on either leg |
| F3 | Any λ at which an infeasible candidate is selected, or an all-infeasible pool returning a prescription |
| F4 | FED-SAN gratuitous compute or communication > 0, or a feasible fixed-round protocol with gratuitous = 0 |
| F5 | Fixed-round compute ≤ FED-SAN's, or FED-SAN per-round compute above EarlyStop's, or fixed-round integrated patient harm < FED-SAN's |
| F6 | Exit fraction ≤ 10% at r* on either leg |
| F7 | Harm matrix near-symmetric, or FED-SAN peak above a same-init baseline's |
| F8 | Poisoned update in any accepted average; naive FedAvg feasible; gated infeasible; a post-teeth regression > 5 points |
| F9 | Mean per-node label skew below threshold, or WDBC counts ≠ published |

## 10. Limitations at scale (the commissioning task's honest answers)

- **Does it scale?** To K = 4–5 nodes with extreme certified skew (0.790
  clinical / 0.765 vision) and an active poisoner, on real patients and
  real images: yes, 15/15 clauses. To *many* nodes (hundreds): untested —
  the simulation is single-process; per-round aggregator evaluation cost
  grows as O(K · trusted-set) and would become the dominant
  machine-suffering term. That growth is itself metered by the design
  (the gate pays for itself visibly), which is the honest way to know
  when the gate stops paying.
- **Does it still reduce machine suffering and patient harm?** Against
  the fixed-round standard protocol, on both legs: compute 2.4× / 1.15×
  below (clinical / vision), wire 5× / 1.15× below, integrated patient
  harm 2.3× / 1.14× below, gratuitous suffering exactly zero, and the
  feasible checkpoint *preserved* where the standard protocol drifts
  below τ. Against the strongest scheduler baseline (EarlyStop), the
  honest split: clinical total compute 22% lower with integrated patient
  harm 0.04 *higher* (the exit-head transient); **vision total compute
  1.56× *higher* and patient integral 2.22 higher, carried by the round
  count (r\* 13 vs 8) — federated deep supervision under extreme label
  skew delays feasibility** (§8.3). Every round is cheaper (per-round
  14.4 ≤ 18.4 MF, 97.1 ≤ 101.0 GF) and deployment is cheaper (exits);
  the totals are not uniformly better, and the contract prints the
  per-round decomposition rather than hiding it.
- **Does it scale without breaking?** Two breakages were found and fixed
  structurally *by the contract's own attack leg*: stealthy poison
  slipping a one-level bar, and reject-and-freeze deadlock (§5.1). The
  fixed-round protocol's feasibility *drift* below τ under non-IID
  averaging (both legs, §8.3) is a third, found property: past-target
  training is not merely wasteful here, it is actively harmful to
  feasibility. The deep-supervision feasibility delay (above) is a
  fourth: the centralized acceleration does not transfer to extreme-skew
  federation.
- **What are the limitations?** (i) Single-process simulation — no real
  network, no dropout/straggler model; the comm ledger is exact bytes,
  not latency. (ii) The trusted held-out set at the aggregator is an
  assumption (A2/A3): a poisoned or unrepresentative trusted set breaks
  every gate here — the gates amplify the trusted set, for good or ill
  (the SAC-LLM's A2 caveat, inherited). (iii) F8 certifies containment of
  *one* strong attack, not Byzantine robustness in general; adaptive
  attackers that optimize against the two-level bar are open. (iv) The
  vision leg is a compact CNN on a 4000-image subset — real images, real
  skew, but not ImageNet (§1.1). (v) `r*` and all integrals are
  trajectory-relative (Definition 4.3 caveat). (vi) No Sounio-native leg;
  `.github/workflows/ci.yml` and `topic-registry.v1.json` wiring left to
  the integrator (shared control surfaces, per the parent specs'
  convention).

## 11. Commands run

```bash
.venv/bin/python scripts/research/federated_san.py   # F_GREEN 13/13 (bit-reproducible)
bash scripts/ci/federated_san_gate.sh                # FEDERATED_SAN_GATE_OK
bin/llm-offload -t math-review -i docs/research/federated_san_spec_2026-07-30.md
```

## 12. LLM-offload review

Mandatory math-review offload (dual xai/Grok 4.3 + zai/GLM-5.2 per M1
policy) run on this spec. Outcome: **PASS / ADDRESSED** —

- **Grok leg:** `[OK]` on every item — T1 metering conservation under
  federation ("reduction to parent T1 + exact integer summation + fixed
  comm term"), T2 gated-aggregation soundness ("(ii) and (iii) hold by
  explicit construction of the two-level bar"), T3's termwise
  non-negative sum inequality, Definition 4.3's trajectory-relative
  caveat, T4/T5's certificate-only scoping, and the non-scalarized
  ledger triple ("no illicit aggregation").
- **Z.AI leg** (truncated at token cap mid-analysis, as in prior runs;
  `finish_reason: length`): independently recomputed and confirmed the
  abstainer harm (5×63/169 = 1.864 = the printed 1.864, "the math is
  extremely tight"), the node-size sums (67+32+34+72+195 = 400), the
  Dirichlet-skew plausibility (0.790), the update-level OR-bar vs
  round-level max-bar consistency ("accepted iff acc ≥ max(abst+0.02,
  best−0.05)... perfectly consistent"), prefix invariance under FedAvg,
  and the per-round integral averages on both legs. Two findings:
  (1) **[TIGHTENABLE]** ADDRESSED — T4's "growing linearly in R − r*"
  was loose (per-round meters vary with the active set); T4 now states
  monotone-in-round-count growth with the per-round charge attribution.
  (2) **[QUERY]** ADDRESSED — the referee challenged the §5.1 stealthy
  poison ("how can a label-flipped model have 0.805 true accuracy while
  the average drops to 0.231?"). Resolved empirically (probe, seeded):
  the *unamplified* poison `w` sits at 0.213; the *submitted amplified
  state* `g + 5(w−g)` — the object the bar evaluates — sits at 0.805;
  the average it enters collapses to 0.231 (honest-only average 0.911).
  Weight-space extrapolation is not accuracy-monotone; §5.1 now carries
  the exact measured triple and names the evaluated object precisely.
- Contract `F_GREEN 15/15` and gate `FEDERATED_SAN_GATE_OK` re-run green
  after all edits. Full entry in `.claude/llm_offload_log.md`
  (2026-07-30 row). Raw: `/tmp/llm-offload-0s2RA4/`.
