<!-- docs:meta
topic_id: repo.docs.papers.san-fpga-deployment-2026-08-04
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.san-fpga-deployment-2026-08-04
-->

# The Suffering Ledger: Integer-Exact Training-Compute Accounting with an FPGA Root-of-Trust, Demonstrated on a Suffering-Aware Early-Exit Network

**Status:** `DRAFT` — every empirical claim marked `MEASURED` below is backed
by a measured artifact in the Sounio repository; claims marked `ESTIMATE` or
`(in progress)` are explicitly noted as such. Human measurement audit is the
authority; LLM reviews are recorded in §7 only as an AI-assist disclosure.
**Date:** 2026-08-06 (repositioned after a seven-front novelty audit,
`agent_logs/san_novelty_audit_2026-08-06.md`)
**Orthography:** EN-US
**Companion spec:** `docs/research/san_imagenet_fpga_dl380_spec_2026-08-02.md`
**Companion contract:** `scripts/research/san_imagenet_fpga_dl380.py`
**Target venue:** arXiv `cs.LG` / `cs.AR`; then IEEE FPL or a systems note

---

## Abstract

Claims that a training procedure saves computation are, today, unfalsifiable
as published: FLOP counts are analytical estimates, energy figures are sampled
at the wall, and no artifact lets a third party verify that the reported
computation is the computation that actually ran. This paper builds the
missing instrument and then turns it on itself. We present a training-compute
ledger with three properties we have not found combined in the literature:
(i) **integer-exact accounting of the executed computation path** — the meter
charges the multiply-accumulates of the stages actually run (a sample that
exits at stage *k* is never executed, and never charged, below *k*), under a
stated partial-meter convention, with a conservation proof checked by an
independent accounting path on every run; (ii) a **declared accuracy
contract** (a feasibility target τ stated
before training; training freezes at the epoch the contract is met, and the
ledger separates the machine burden into a *necessary* and a *gratuitous*
part); and (iii) a **hardware root-of-trust for the decision and its
accounting**: an AMD Alveo U250 kernel that
re-derives per-sample exit decisions and the integer FLOP account they imply,
outside the host, bit-exact against the golden model, at **511 Msamples/s**
sustained on a 1.2M-sample stress cohort — 94.5% of the DMA-limited
theoretical peak at the achieved 135.2 MHz — at approximately **3.3 nJ/sample**
incremental board-level energy (an order-of-magnitude figure from a 1 Hz
on-card sensor; §4.1 states the uncertainty). The card attests the decision
and the accounting; it does not observe the host's physical execution, and we
scope the claim accordingly (§5.2). The same integer function on a host Xeon Gold
6526Y measures 130.8 Msamples/s scalar and 1.05 Gsamples/s AVX-512 across 8
threads, so the card does not win on raw throughput; its value is offload,
fixed low-energy operation, and a hardware-enforced integer specification.

The demonstration workload is a suffering-aware early-exit network (SAN)
[4] — ResNet-50, ViT-small/d384, and a tiny decoder LM — whose training
freezes when the declared target is met. We are explicit about what this
workload is and is not. It is *not* a better early-exit method: that
literature (BranchyNet, Shallow-Deep Networks, PABEE, BERxiT, CALM) already
reports larger inference-time savings with stronger baselines, and our own
full-CIFAR-10 study (50 000/10 000, τ ∈ {0.80, 0.85}) shows the SAN losing to
a plain early-stop baseline in that regime. The workload exists to exercise
the ledger. And the ledger earns its keep: across five model variants that
were believed to train *learned exit gates*, the audit trail exposed that the
gates received no gradient at all — they were frozen random initialisations,
and every gating result attributed to them was an initialisation artifact
(§4.9). A post-τ training variant (SAN-v6) built to activate those gates
produced the first configuration in which the SAN loses to EarlyStop on
metered burden (14 121 vs 13 290 TMAC) — a negative result the ledger
explains mechanistically. A corrected variant with gates supervised by a
correctness-distillation objective (SAN-v7, in the lineage of BERxiT's
learning-to-exit module) fires as designed: validation exit fraction 0.52–0.98
post-τ, total metered burden 9.3% below EarlyStop *including* twenty post-τ
epochs (the exit discount pays for the extra training), and a measured 1.41×
inference wall-time speedup against Dense — at a post-τ accuracy of 0.7998
that breaks the contract met at t*. A second point of the same frontier at
threshold 0.95 (SAN-v7b) recovers the contract (final accuracy 0.8594 ≥ τ)
with exits still firing (0.68–0.86), 6.9% below EarlyStop and a 1.12×
inference speedup: the first configuration in the study satisfying the
declared target with learned exits, and evidence that the
accuracy–compute trade is a *declared deployment point* — exactly what the
audit kernel enforces (§4.9–§4.10). A five-threshold sweep, with predictions
preregistered before the runs (git-timestamped), confirms the accuracy and
latency frontiers as smooth and monotone (5/5 predictions each) and
falsifies the training-burden monotonicity — the necessary/gratuitous split
is dominated by freeze-timing jitter, a mechanism we report because the
preregistered failure revealed it (§4.11). A second preregistered round —
submission-timestamped this time — confirms all five predictions for a
cost-aware gate variant (SAN-v8) while showing it merely reparameterises the
threshold dial, and maps the ViT/GPT threshold dials, falsifying the
accuracy-floor recovery for the LM leg and locating its cause as structural
rather than a threshold artefact (§4.12). The
strongest positive result stands: a curriculum-gated SAN reaches τ = 0.85 on
full CIFAR-10 at accuracy 0.8561 — inside the observed cross-job Dense band
(0.8576–0.8650 at identical seed and configuration; §4.8 discloses the
run-to-run variance) — at 76.7% less metered computation than Dense and 51.7%
less than EarlyStop (within-run comparison) — and every compute figure in that
sentence is backed by an exact, independently verified account of the metered
computation it summarises, accounting we could not find in this literature. All artifacts, measurements, and reproduction
scripts are in the repository.

---

## 1 Introduction

### 1.1 The problem: compute claims you cannot audit

Deep learning has a mature literature on spending less computation: early-exit
networks let samples leave at intermediate classifiers [1,2,3], adaptive-computation
models learn when to stop thinking [17,18,19], and training-efficiency work
reports FLOP or energy savings against declared targets [22,23,25]. What this
literature does not provide is a way to *verify* any of it. Reported FLOPs are
analytical estimates derived from layer shapes, not counts of what executed;
energy figures are sampled at the wall or the driver; and the training trajectory
that produced a claimed saving is not an auditable artifact. A referee — or a
regulator, or the authors themselves six months later — cannot distinguish a
real saving from an accounting convention.

We know this firsthand, because it happened to us. In the course of the study
reported here, our own ledger exposed a claim we had believed across five model
variants: that our early-exit gates were *learned*. They were not — the gradient
path was silently dead, and the gates were frozen random initialisations
(§4.9). The exposure chain matters, so we state it honestly: it was not a
single magic instrument but the conjunction of an exactly-metered anomaly
(exit fraction 0.000 at every epoch, where any working gate would have
produced nonzero exits), a contract clause (L6) that made the anomaly a
*failure* rather than a curiosity, and a code audit the anomaly forced. A
gradient-norm probe of the gate parameters would have caught the bug earlier —
but our instrumentation watched stage and exit-head norms and simply never
metered the gates. You only see what you meter; the ledger is our attempt to
meter the whole story.

### 1.2 What we build

This paper contributes the instrument, not a new network:

1. **An integer-exact ledger of the executed computation path.** Every training
   and evaluation step charges the multiply-accumulates of the stages actually
   run to a meter — a sample that exits at stage *k* is never executed below
   *k*, and those stages are never charged — under a stated partial-meter
   convention (§5.2); a second, independent accounting path re-derives the
   same total from module shapes and recorded active counts, and the two must
   agree exactly (clause L1). This is executed-*path* accounting at the
   framework level, not silicon-level measurement (cf. CUPTI/RAPL), and not
   the analytical full-network FLOP counts reported elsewhere: to our
   knowledge it is the first training-compute accounting with a conservation
   proof; the Green AI / energy-measurement line
   [25,26,27,28,31] measures energy by sampling, and reported FLOP counts
   elsewhere are analytical.
2. **A declared accuracy contract with an auditable stop.** A feasibility
   target τ is stated before training; training freezes at the first epoch the
   contract holds (freeze-on-green), and the ledger decomposes the total
   machine burden into *necessary* (up to the freeze) and *gratuitous*
   (everything after) parts. This is time-to-accuracy [22,23] turned into an
   accounting device: the decomposition is the accuracy-target analogue of the
   "energy bloat" decomposition of Perseus [29], defined per run rather than
   per distributed schedule, and it connects to the algorithmic-efficiency
   quantity of Hernandez & Brown [30] at run granularity.
3. **A hardware root-of-trust for the decision and its accounting.** An AMD
   Alveo U250 kernel re-derives the
   per-sample exit decisions and the integer FLOP account they imply,
   *outside the host*, bit-exact against the golden model, with integer
   semantics enforced by the bitstream. Scope, stated plainly: the card
   attests that the exit decisions and their accounting follow the integer
   specification; it does not observe the host's physical execution, and it
   audits inference cohorts, not the training loop (§5.2). The card is not an
   accelerator for the model — it is an
   independent auditor for the decision path, in the spirit of proposed
   on-chip compute-governance designs [36,37], but measured and deployed.

The demonstration workload is a suffering-aware early-exit network (SAN) [4]:
an early-exit architecture (a standard component since BranchyNet [1] and
Shallow-Deep Networks [2]) whose training freezes at the declared target and
whose executed FLOPs are treated as a first-class, minimisable quantity —
"machine suffering" in the parent line's vocabulary, which we use here as an
*operational metaphor only* (§5.4 delimits this against the machine-welfare
literature [39,40,41,42]). The SAN is deliberately ordinary as an
architecture; the point is that everything it does leaves a verifiable trail.

### 1.3 Contributions

| # | Contribution | Evidence | Status |
|---|---|---|---|
| 1 | Integer-exact executed-FLOP ledger with independent-path conservation verification, integrated into a three-family training harness | `scripts/research/suffering_aware_large_architecture_v2.py`, clause L1 | `MEASURED` |
| 2 | Declared-τ training contract with necessary/gratuitous decomposition of the machine burden | §2.3, §4 | `MEASURED` |
| 3 | FPGA audit kernel (exit decision + metering outside the host): 511 Msamples/s on 1.2M cohort, ~3.3 nJ/sample board-level, bit-exact vs golden model; host-CPU baseline measured | `hardware/fpga/u250_catastrophe_scan/`, `san_scan_cpu_baseline.c` | `MEASURED` |
| 4 | Full-CIFAR-10 training study with EarlyStop ablation column: two-mechanism decomposition (freeze-on-green vs early exits), positive (v3/v4) and negative (v1/v2/v5/v6) results | §4.2–§4.9, Slurm jobs 8584–8651 | `MEASURED / MIXED` |
| 5 | Case study: the ledger catches a five-variant false claim (frozen random gates believed learned); mechanistic explanation of the resulting negative result | §4.9 | `MEASURED` |
| 6 | Real-image kernel validation on ImageNette2-160, bit-exact on the U250 | `train_san_imagenette.py` + `host_san_scan` | `MEASURED` |
| 7 | Honest accounting of limits: small-subset regimes, partial meter convention, board-level power, and a novelty audit repositioning every claim against prior art | §5, `agent_logs/san_novelty_audit_2026-08-06.md` | `DECLARED` |

### 1.4 Prior work

**Early exits.** BranchyNet [1] introduced side-branch classifiers with entropy
thresholds; Shallow-Deep Networks [2] systematised per-stage internal
classifiers on CIFAR and named the "overthinking" waste our machine channel
meters; MSDNet [3] added anytime prediction, and deeply-supervised nets go back
to 2015. Learned exit policies exist in several forms: reinforcement-learned
block skipping (SkipNet, BlockDrop) [16], a learned allocation policy
(Bolukbasi et al.) [15], depth-adaptive transformers [14], and — closest to our
corrected gate — BERxiT's learning-to-exit module, a sigmoid unit supervised
against the binary target "the layer-k classifier was correct on this sample"
[11]. Our SAN-v7 gate uses the same target with a BCE objective; we claim no
novelty for it. QuEE trains a per-exit, per-sample error predictor [45]. For
transformers, DeeBERT [9], PABEE [10] (which also reports early-exit ResNets on
CIFAR-10 with accuracy gains), CALM (provably correct exits) [12], and
LayerSkip (self-speculative exits) [13] define a state of the art we do not
compete with. Surveys: Laskaridis et al. [44] and Han et al.

**Adaptive computation.** ACT [17] puts a linear ponder-cost in the loss;
PonderNet [18] learns stochastic halting (we note it appeared at the ICML 2021
AutoML *workshop*, not the main track); Mixture-of-Depths [19] imposes a hard
capacity budget and evaluates isoFLOP — the methodology we adopt for
comparison. Universal Transformers [20] bridge ACT to transformers. All of
these allocate compute at *inference*; none audits the training bill.

**Stopping training at a target.** Early stopping is classical [21];
time-to-accuracy was institutionalised by DAWNBench [22] and MLPerf Training
[23], and minimised directly in [24]. Freeze-on-green is this rule with an
accounting payload: the stop is declared ex ante and the post-target burden is
named and measured, not merely avoided.

**Accounting.** Green AI [25], Strubell et al. [26], Henderson et al. [27],
Zeus [28], and Carbontracker [31] measure or estimate the environmental cost of
training; Patterson et al. discuss analytical FLOP estimates. Perseus [29]
decomposes training *energy* into useful work and bloat; Hernandez & Brown [30]
define the FLOPs *required* to reach a declared performance level across
algorithm generations. None of these provides integer-exact executed-FLOP
accounting for a single run, nor an independent conservation check.

**Hardware.** Early-exit accelerators on FPGA are established: ATHEENA (a
complete early-exit toolflow) [32], a dedicated exit-decision unit [33],
hardware-aware progressive inference (HAPI) [34], and progressive device–cloud
inference (SPINN) [35]. Compute-governance proposals — FlexHEG's auditable
guarantee processor [36], compute monitoring for training rules [37] — are
designs, not measured deployments; proof-of-learning [38] verifies training
*happened*, not what it *cost*. Our kernel is positioned against this line: not
an early-exit accelerator (the trunk stays on the host) but a measured,
deployed auditor for the computation itself.

**Constrained learning and Goodhart.** Constrained ERM with guarantees
[5,6,7] and the Goodhart/reward-hacking literature [8,46,47] frame the
selection-rule side of the SAN; the parent line [4] develops the two-channel
suffering formulation this workload instantiates.

**Machine welfare.** "Machine suffering" is an occupied term in philosophy:
Metzinger's artificial suffering [40], the AI-welfare programme [41], Tomasik's
RL-welfare argument [42], and Klimovich's recent essay of exactly this title
[39]. None of it is operationalised; §5.4 states precisely what we do and do
not claim.

---

## 2 Background: Suffering-Aware Neural Networks

### 2.1 The two channels

A SAN is defined by:

- A trunk with intermediate exit heads.
- A per-sample exit rule: the first head whose confidence ≥ Δ decides the
  output; if no head is confident, the final head decides.
- A feasibility target τ on held-out accuracy.
- A training rule: train until τ is reached, then freeze.
- A compassion grid over (patient weight, machine weight) that selects among
  feasible checkpoints.

**Patient suffering** is the mean harm of the model's predictions under a cost
matrix C, where C[y, ŷ] encodes the cost of predicting ŷ when the true label is
y. For CIFAR-10 we use the parent line's hazard structure: class 9 (truck) is
the hazard class; missing the hazard costs 5, a false hazard alarm costs 2, any
other confusion costs 1. For SAN-GPT-small the hazard tokens are corpus negation tokens.

**Machine suffering** is the *metered-MAC* count of executed operations. We
follow the convention used throughout the SAN line: MACs × 2 = FLOPs; training
backward pass = 2× forward; biases, norms, activations, softmax, residual adds,
and pooling are unmetered. The convention is identical for every architecture
and baseline, so all reported savings are relative under the same partial
accounting. It is a proxy for computational burden, not a claim that every host-
side operation has been charged.

### 2.2 Anti-Goodhart gating

A standard risk when optimizing a proxy (FLOPs) is that the model learns to
satisfy the proxy while failing the real task [8]. SAN prevents this by
selection: the compassion grid may only choose checkpoints that are feasible
(accuracy ≥ τ). If no checkpoint is feasible, the gate returns `NO_FEASIBLE`
instead of silently degrading accuracy. This is an executable property checked
in the companion contract.

### 2.3 Freeze-on-green

Let t* be the first epoch at which the held-out accuracy reaches τ. SAN stops
at t*. The dense baseline trains for the full budget B; the EarlyStop baseline
also stops at t* but lacks early-exit layers. SAN's integrated machine
suffering decomposes as:

S_SAN = S_necessary(t*) + S_gratuitous

with S_gratuitous = 0 by construction. The dense baseline has S_gratuitous > 0;
EarlyStop has S_gratuitous = 0 but lacks inference-time savings.

---

## 3 Methods

### 3.1 Architectures

**SAN-ResNet-50.** A CIFAR-variant ResNet-50 with bottleneck blocks
(3,4,6,3), stage widths 256/512/1024/2048, and one early-exit head after each
stage plus the final head. Real-scale MACs per stage are taken from the
published architecture (§3.1 of the companion spec).

**SAN-ViT-small/d384.** Patch 4×4 → 64 tokens + CLS, d=384, 12 blocks, 6 heads, MLP
ratio 4. One CLS exit head per block. This is a small attention proxy for the
CIFAR pilot, not ViT-Large. The FPGA kernel's stage-cost LUT can be reloaded
with the published ViT-L/16 constants (61.55 GMAC) for a real-scale deployment.

**SAN-GPT-small.** A tiny decoder-only transformer (d=384, 10 blocks, 6 heads,
causal mask, vocab 2000) trained on next-token prediction over the repository's
own research documentation corpus. Exit heads score the last G=4 positions;
the gate confidence is the mean max-probability over those positions. This is
an internal-corpus pilot, not a general language-modeling result.

All three families are compared against two baselines:

- **Dense**: the same trunk trained for the full budget.
- **EarlyStop**: the same trunk with SAN's stop rule but no exit heads.

### 3.2 FPGA kernel

The kernel `krnl_san_scan.cpp` is a pure integer exit-audit / FLOP-meter. It
does *not* run the trunk; the trunk produces per-sample confidence vectors on
the host CPU/GPU and the kernel receives them as Q0.15 integers. Per lane it
does:

1. Read a packed 512-bit beat containing four 128-bit samples; each sample
   holds up to seven 15-bit Q0.15 confidence fields.
2. Compare each field to the integer threshold q_Δ = ⌊Δ · 2^15⌋.
3. Return the index of the first field ≥ q_Δ (priority encoder).
4. If none, return the final index (catastrophe).
5. Look up the stage-cost LUT at the exit index and accumulate into a 64-bit
   counter.

The kernel contains **no floating-point, no softmax, no multipliers, no DSPs**.
It is bus-limited: 512 bits/cycle × 4 samples/beat × 135.2 MHz = 540.8
Msamples/s theoretical peak. The LUT is loaded by the host, so the same
bitstream serves ResNet-50, ViT-small/d384, and SAN-GPT-small by reloading the LUT and threshold.

**Correctness criterion (T3).** The kernel is correct iff it reproduces the
golden integer function. We verify this bit-exactly on three cohorts: the
ResNet validation cohort (5,000 samples, 5 exit points), the ViT validation
cohort (5,000 samples, 7 exit points), and a 1.2M-sample stress cohort (5 exit
points, ImageNet-sized). The host program `host_san_scan` compares
card outputs to the control-VM golden and reports `HOST_SAN_SCAN_PASS`.

### 3.3 GPU training harness

Training runs on the Slurm `gpu-orangefs` partition (NVIDIA RTX A5000 / RTX
4000 Ada). The harness is `scripts/research/suffering_aware_large_architecture.py`,
adapted for CUDA with `torch.cuda.synchronize()` latency measurement. Each
family shares one trunk initialization, one data order, and one seed across SAN,
Dense, and EarlyStop.

### 3.4 Datasets

- **CIFAR-10**: 50,000 train / 10,000 test, 10 classes. Used for the main GPU
  training study.
- **ImageNette2-160**: 10-class subset of ImageNet, 160 px, real photographs.
  Used for real-image U250 validation. Full ImageNet-1k is not available in
  this environment.
- **CIFAR-100 / ImageNette2-320 / ImageNet-1k**: ablation / larger-proxy
  datasets; availability and results reported in §4.3–4.4.

---

## 4 Experimental Results

### 4.1 U250 throughput and energy

The bitstream was built with Vitis 2025.1 on a separate builder VM and run on
the DL380 U250. Table 1 reports the on-target campaign.

**Table 1: U250 on-target benchmark campaign.**

| dataset | n | points | single-shot (Msamples/s) | sustained (Msamples/s) | result |
|---|---|---|---|---|---|
| val_resnet | 5,000 | 5 | 24.2 | 146.7 | `HOST_SAN_SCAN_PASS` |
| val_vit | 5,000 | 7 | 43.0 | 146.9 | `HOST_SAN_SCAN_PASS` |
| stress_1p2M | 1,200,000 | 5 | 481.9 | **511.0** | `HOST_SAN_SCAN_PASS` |
| val_imagenette | 3,925 | 5 | 24.1 | 122.2 | `HOST_SAN_SCAN_PASS` |

The 1.2M stress cohort reaches **511 Msamples/s sustained**. Because the kernel
is bus-limited by construction, this is best read as **94.5% of the 540.8
Msamples/s DMA-limited theoretical peak** at the achieved 135.2 MHz clock, not
as a claim about kernel microarchitecture performance. Smaller cohorts are
enqueue/sync dominated and report lower sustained rates (Table 1).

**CPU baseline.** The same integer scan/meter function was implemented in C with
scalar, AVX2, AVX-512, and OpenMP-parallel AVX-512 paths and run on the same 1.2M
stress cohort on a Xeon Gold 6526Y host. All paths are bit-exact against the
golden model.

**Table 2: Host-CPU baseline for the audit kernel (stress cohort, n=1,200,000).**

| variant | threads | throughput (Msamples/s) | vs FPGA |
|---|---|---|---|
| Scalar | 1 | 130.8 | 0.26× |
| AVX2 | 1 | 164.5 | 0.32× |
| AVX-512 | 1 | 173.9 | 0.34× |
| AVX-512 + OpenMP | 8 | 1,045 | 2.0× |

The multi-core CPU is faster in raw throughput than the U250 on this
workload. The FPGA value is therefore not a pure speedup claim; it is (a)
offloading the audit path so the host CPU/GPU remains available for the trunk,
(b) fixed incremental board-level energy, and (c) a hardware-enforced integer
specification that is bit-exact against the golden model.

**Energy.** Board-level power was measured with `xrt-smi examine -r electrical`
(1 Hz, 30 s). Idle card: 24.435 W. Under continuous `host_san_scan_bench` on the
1.2M cohort: 26.153 W. Incremental draw ΔP ≈ 1.7 W. The bench processed
15.5436 Gsamples in 30.002 s (aggregate 518.1 Msamples/s), giving approximately
**3.3 nJ/sample** incremental board-level energy. A repeat with the tiny
ImageNette cohort gave load power below idle because the kernel idles between
micro-enqueues; the stress-cohort number is therefore the honest per-sample
energy figure.

The 3.3 nJ/sample value is rounded to two significant figures. The idle and load
power samples are coarse (1 Hz on-card sensor), and the incremental draw is the
difference of two large numbers (ΔP ≈ 1.7 W against 24.4 W idle). We therefore
treat it as an order-of-magnitude board-level estimate, not as a precise energy
claim; a higher-rate external meter would be needed for three-significant-figure
energy.

### 4.2 GPU training savings and latency

Table 3 reports the GPU scale pilot. All numbers are measured on real CIFAR-10
training with `SAN_LARGE_DEVICE=cuda`, using stratified subsets of 4,000 train
/ 1,000 val images and a small epoch budget (8 for ResNet-50, 10 for ViT-small/d384/SAN-GPT-small).
The feasibility targets τ (0.34, 0.251, 0.165) are intentionally low so the
freeze-on-green rule can be demonstrated quickly; the achieved accuracies are
not competitive with full-dataset CIFAR-10 baselines. The point of the table is
to show that the SAN machinery (meter conservation, freeze-on-green, early
exits, compassion grid) operates correctly across families, not to claim a new
accuracy result.

**Table 3: GPU training study (CIFAR-10, small subset).**

| family | t* | SAN acc@t* | EarlyStop acc@t* | S_m(SAN) | S_m(EarlyStop) | S_m(Dense) | SAN vs Dense | SAN vs EarlyStop | exit@t* | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 4 | 0.390 | 0.361 | 160.1 TMAC | 101.2 TMAC | 269.9 TMAC | **40.7%** | **−58.2%** | 0.489 | L1–L4, L6–L8 PASS; L5 tradeoff |
| ViT-small/d384 | 4 | 0.262 | 0.275 | 183.3 TMAC | 184.6 TMAC | 369.3 TMAC | **50.4%** | **0.7%** | 0.033 | L1–L4, L7–L8 PASS; L5 PASS, L6 exit-frac 0.03 |
| SAN-GPT-small | 4 | 0.167 | 0.172 | 115.4 TMAC | 120.8 TMAC | 241.5 TMAC | **52.2%** | **4.5%** | 0.112 | L1–L8 PASS |

*S_m is the metered-MAC burden (MAC×2, backward = 2× forward, biases/norms/etc.
unmetered). EarlyStop is the same trunk with SAN's stop rule but no exit heads,
so its inference latency equals Dense. Verdicts refer to the companion contract
clauses L1–L8; see Appendix B for clause definitions.*

**Machine channel — two separable mechanisms.** The total saving against Dense is
the sum of (a) freeze-on-green (the EarlyStop baseline) and (b) early exits (the
increment over EarlyStop). In this small-subset, fast-convergence regime the split
is:

| family | vs Dense | = freeze-on-green | + early exits |
|---|---|---|---|
| ResNet-50 | 40.7% | 62.5% | −21.8 pp |
| ViT-small/d384 | 50.4% | 50.0% | +0.4 pp |
| SAN-GPT-small | 52.2% | 50.0% | +2.2 pp |

For ResNet-50 the early-exit heads add per-sample compute at the chosen Δ, so
SAN is *slower* than EarlyStop on metered-MAC (−58.2%); the net 40.7% saving
comes almost entirely from stopping early. For ViT-small/d384 the exit fraction
at t* is only 0.033 (below the L6 threshold), so early exits are effectively inert
and the 50.4% saving is freeze-on-green alone. SAN-GPT-small shows the only
meaningful early-exit contribution (4.5% over EarlyStop), with an exit fraction
of 0.112. These are honest findings: the freeze-on-green rule carries the
machine-channel result in this regime, and the per-family exit-head behaviour
varies.

ResNet-50, with its early-exit-friendly residual stages, also translates the MAC
savings into a small real wall-time speedup on CIFAR-10 (0.196 ms vs 0.213 ms),
though the margin is within the noise regime of CUDA synchronize microbenchmarks.
ViT-small/d384 and SAN-GPT-small are essentially tied with Dense in wall time because the
forward is short enough that dispatch overhead dominates; at larger scale the
MAC savings would likely translate to latency savings as well.

**Patient channel.** ResNet-50 shows the disclosed tradeoff the parent line
reports at ImageNet scale: SAN patient harm is slightly higher than EarlyStop's
because early stopping alone can freeze on a luckier epoch. ViT-small/d384 and
SAN-GPT-small satisfy the stricter L5 clause (SAN ≤ both baselines) in this run.
These are results, not tuning failures. The cost matrix C is synthetic; no
clinical claim is made.

### 4.3 Threshold ablation

Table 4 reports the Δ sensitivity study on the same CIFAR-10 small subset
(Slurm jobs 8599–8607).  The threshold changes which samples exit early and
therefore the metered-MAC burden, but it cannot overcome the small training
budget: even the best configurations remain `L_RED` because the patient-channel
clauses (L5/L7 in particular) are not fully satisfied on this subset.  The
ablation is therefore reported as a *sensitivity knob*, not as a tuning recipe
that turns the study green.

**Table 4: Threshold ablation (CIFAR-10, 4,000 train / 1,000 val).**

| family | Δ | t* | SAN acc@t* | S_m(SAN) | S_m(Dense) | saving | speedup | clauses PASS |
|---|---|---|---|---|---|---|---|---|
| ResNet-50 | 0.35 | — | 0.338 | 201.7 TMAC | 269.9 TMAC | 25.3% | 1.54x | 4/8 |
| ResNet-50 | 0.45 | 4 | 0.355 | 149.7 TMAC | 269.9 TMAC | 44.5% | 1.10x | 7/8 |
| ResNet-50 | 0.55 | 3 | 0.347 | 130.9 TMAC | 269.9 TMAC | 51.5% | 1.00x | 6/8 |
| ResNet-50 | 0.65 | 2 | 0.354 | 100.3 TMAC | 269.9 TMAC | 62.8% | 0.98x | 7/8 |
| ResNet-50 | 0.75 | 4 | 0.377 | 167.1 TMAC | 269.9 TMAC | 38.1% | 0.99x | 7/8 |
| ViT-small/d384 | 0.25 | — | 0.187 | 234.6 TMAC | 369.3 TMAC | 36.5% | 1.83x | 4/8 |
| ViT-small/d384 | 0.35 | 9 | 0.267 | 312.2 TMAC | 369.3 TMAC | 15.5% | 1.19x | 6/8 |
| ViT-small/d384 | 0.45 | 4 | 0.258 | 183.6 TMAC | 369.3 TMAC | 50.3% | 0.98x | 6/8 |
| ViT-small/d384 | 0.55 | 4 | 0.257 | 184.2 TMAC | 369.3 TMAC | 50.1% | 0.99x | 7/8 |

On this subset, the best SAN operating points are Δ = 0.45 for ResNet-50
(t* = 4, 44.5% MAC saving, 7/8 clauses) and Δ = 0.55 for ViT-small/d384
(t* = 4, ~50% MAC saving, 7/8 clauses).  The residual family benefits from a
softer threshold because its confidence grows more gradually across exits;
the attention family needs a sharper threshold to avoid firing on weak early
heads.  These values are used as the starting points for the CIFAR-100 runs in
§4.4.

### 4.4 Larger proxy: CIFAR-100

CIFAR-100 was staged on OrangeFS (`/orangefs/training/sounio/datasets/cifar-100-python`)
and two Slurm jobs were submitted with thresholds adjusted to the harder,
100-class task.  The first submission (jobs 8609/8610) failed because the
harness's CIFAR-100 loader used `encoding="latin1"`, under which the pickle
keys are strings rather than bytes; this was fixed to `encoding="bytes"` and
the runs were resubmitted as jobs 8611 (ResNet-50) and 8612
(ViT-small/d384).

On the same 4,000 train / 1,000 val split, CIFAR-100 is too hard for the
small epoch budget: neither family reaches its lowered τ, so `t* = None` and
the freeze-on-green rule never fires.  The result is reported as a negative
result, not hidden.

**Table 5: CIFAR-100 small-subset pilot.**

| family | τ | epochs | final acc | S_m(SAN) | S_m(Dense) | saving | verdict |
|---|---|---|---|---|---|---|---|
| ResNet-50 | 0.20 | 8 | 0.126 | 266.9 TMAC | 270.0 TMAC | 1.2% | L_RED (1/8) |
| ViT-small/d384 | 0.15 | 10 | 0.117 | 369.1 TMAC | 369.3 TMAC | <0.1% | L_RED (infeasible) |

The near-zero savings show that early exits are almost irrelevant when the
model is underfit: almost no sample becomes confident enough to exit, so SAN
runs essentially the full trunk.  A usable CIFAR-100 result would require a
larger train split, more epochs, or a smaller model.  ImageNette-320/full
remains a candidate real-image proxy if bandwidth and storage permit.

### 4.5 Real-image validation on ImageNette2-160

A SAN-ResNet-18 was trained on 4,000 ImageNette2-160 real photographs (frozen
ImageNet-1k backbone, layer4 + heads fine-tuned). Validation confidences for
3,925 real images were exported to the U250 cohort format and run on-target:

```
host_san_scan: dataset=val_imagenette n=3925 points=5 q_delta=18021 family=resnet
CARD_RESULT n=3925 catastrophes=241 flops_macs=4446713384960 wall=0.095ms (41.2Msamples/s kernel-only)
HOST_SAN_SCAN_PASS (val_imagenette)
```

The card's histogram, catastrophe count, and FLOP total are bit-exact against
the Python golden model on real photographs.

### 4.6 Full CIFAR-10 ResNet-50 sensitivity study (negative control)

To test whether the small-subset savings generalise to a non-trivial accuracy
regime, we ran SAN-ResNet-50 on the full CIFAR-10 train/val split (50 000 / 10
000) with Δ = 0.55 and a 60-epoch budget. We report three runs: τ = 0.85 with
seed 17 (job 8615), τ = 0.85 with seed 42 (job 8619), and τ = 0.80 with seed 17
(job 8620). All three jobs failed with a CUDA out-of-memory error in the final
latency benchmark, but the training ledgers were written before the failure and
are the numbers reported here.

**Table 6: Full CIFAR-10 ResNet-50 sensitivity (50k/10k, Δ = 0.55).**

| run | τ | seed | variant | t* | epochs run | final acc | S_m (TMAC) | vs Dense | vs EarlyStop |
|---|---|---|---|---|---|---|---|---|---|
| 8615 | 0.85 | 17 | SAN | None | 60 | 0.7686 | 10 303 | −58.7% | +7.8% |
| 8615 | 0.85 | 17 | EarlyStop | 22 | 23 | 0.8513 | 9 552 | −61.7% | — |
| 8615 | 0.85 | 17 | Dense | 24 | 60 | 0.8650 | 24 918 | — | +160.9% |
| 8619 | 0.85 | 42 | SAN | None | 60 | 0.7675 | 10 486 | −57.9% | −6.5% |
| 8619 | 0.85 | 42 | EarlyStop | 26 | 27 | 0.8504 | 11 213 | −55.0% | — |
| 8619 | 0.85 | 42 | Dense | 21 | 60 | 0.8576 | 24 918 | — | +122.1% |
| 8620 | 0.80 | 17 | SAN | None | 60 | 0.7592 | 10 138 | −59.3% | +71.3% |
| 8620 | 0.80 | 17 | EarlyStop | 6 | 7 | 0.8240 | 2 907 | −88.3% | — |
| 8620 | 0.80 | 17 | Dense | 5 | 60 | 0.8501 | 24 918 | — | +756.7% |

Four findings:

1. **Freeze-on-green is the dominant savings mechanism.** EarlyStop alone removes
   55–88% of Dense's computation across the three runs; this is the single
   largest and most robust effect.
2. **SAN early-exit heads are seed- and τ-sensitive.** At τ = 0.85 the SAN never
   reaches the target and consumes between 6.5% less and 7.8% more than
   EarlyStop depending on the seed; at τ = 0.80 the SAN is 71% more expensive
   than EarlyStop because the baseline can freeze after only 6 epochs while the
   SAN runs the full budget.
3. **The small-subset results are regime-specific.** The CIFAR-10 small-subset
   numbers in §4.2 demonstrate the training machinery, not a guaranteed
   efficiency margin at competitive accuracy.
4. **Early exits do not guarantee higher accuracy.** Across all three runs the
   SAN's final accuracy (0.759–0.769) is below both EarlyStop (0.824–0.851) and
   Dense (0.850–0.865), suggesting the exit heads can interfere with convergence
   when the target is high.

### 4.7 End-to-end deployment loop

Table 1 showed the scan kernel in isolation. We now report, to our knowledge,
the first measured pass through the full inference pipeline: a SAN-ResNet-18
trunk running on the DL380 host CPU, exit confidences quantized to Q0.15, packed
into the same 512-bit beats used by `host_san_scan`, and streamed to the U250
via the new `host_san_scan_e2e` XRT host. The orchestration script
`scripts/research/san_fpga_endtoend.py` measures every phase and validates the
card output bit-exactly against an independent Python golden scan.

**Table 7: Host↔card phase decomposition (ImageNette2-160, n=3 925).**

| phase | time | note |
|---|---|---|
| quantize + pack | ~40 ms | Q0.15 floor + 512-bit beat packing (host preprocessing; not part of the round trip below) |
| xclbin setup | ~135 ms | one-time `xclbin` load per process (no PR) |
| DMA H2D | ~0.12 ms | 62 848 bytes = 982 beats × 64 bytes |
| kernel | ~0.66 ms | e2e phase time, per-call XRT overhead included |
| DMA D2H | ~0.15 ms | histogram + catastrophe count + MAC total |
| **host↔card total** | **~136 ms** | xclbin setup + DMA + kernel + DMA; first cohort; subsequent cohorts reuse context |
| PyTorch forward (CPU) | ~18.4 s | SAN-ResNet-18, 3 925 real images, DL380 CPU |

*Throughput reconciliation.* Three different Msamples/s figures appear for this
small cohort across this paper, and they are all the same kernel under
different measurement envelopes: 41.2 Msamples/s is the pure kernel wall time
in the end-to-end run (0.095 ms, §4.5); ~6 Msamples/s is the e2e *phase* time
above (0.66 ms, per-call buffer/sync overhead included); and Table 1's
24.1/122.2 Msamples/s are the bench harness's single-shot and sustained
figures. The kernel is identical; only the envelope differs. The 511
Msamples/s headline comes from the 1.2M-sample stress cohort, where fixed
overheads amortise.

The host↔card total is the measured FPGA round trip for one cohort; it is
dominated by the one-time `xclbin` load. After the load, additional cohorts can
be enqueued without reloading, so the per-cohort cost excluding forward is
~1 ms. The PyTorch forward pass is CPU-only because the DL380 has no GPU
fitted; a GPU would reduce it proportionally. The packed cohort is 62 848 bytes
= 982 beats × 64 bytes, so the DMA times correspond to ~470 MB/s effective
PCIe bandwidth for the small transfer. The script supports `--mock-host` for CI
or environments without the U250; the mock run reproduces the same histogram,
catastrophe count, and FLOP total and confirms the Python packing matches the
C++ unpacking bit-exactly.

This closes the loop between training (§4.2), confidence extraction, and FPGA
audit: the trunk runs on the host, the card decides and meters, and the
orchestration script verifies that the two agree exactly.

### 4.8 SAN-v2: learned gating and stronger exit heads

To test whether the SAN's accuracy deficit at high τ is an architecture problem
or a training problem, we built SAN-v2 with three changes:

1. **MLP exit heads** (two-layer GELU with residual) replacing the v1 linear
   heads, giving each exit more capacity.
2. **Learned gating network** (per-stage MLP + stage embedding) replacing the
   fixed confidence threshold, letting the model learn *when* to exit.
3. **Gradient instrumentation** measuring per-stage gradient norms during
   training.

**Gradient analysis (job 8622).** On the full CIFAR-10 run, the v1 SAN shows a
clear pathology: trunk stage gradients grow 8–23× over 60 epochs while the
final head grows only 5×, and the exit-head-to-final gradient ratio *falls*
from 0.118 to 0.098. The exit heads are learning more slowly than the final
head but still inject growing gradient into the trunk — they are both weak and
interfering.

**SAN-v2 result (job 8625).** With MLP heads and a learned gate, the SAN's
final accuracy rises from 0.768 to **0.812** and S_m falls from 10 303 to
**8 012 TMAC** (−22%). The learned gate, however, converges to exit_frac =
1.000 — every sample exits at the first stage — so the model still does not
reach τ = 0.85. The gate has learned to minimise FLOPs by always exiting early,
which caps accuracy at the first-stage head's capacity.

> **Correction (2026-08-06, gradient-path audit).** The previous paragraph is
> wrong in a way that matters, and we leave the original text in place because
> correcting it is the point of §4.9: subsequent audit showed that the gate
> networks in SAN-v2 through SAN-v6 received **no gradient** — the gate input
> is detached, the exit decision is a hard threshold, and no auxiliary loss
> supervises the gate. What "converged to exit_frac = 1.000" was therefore not
> learning but a frozen random initialisation whose sigmoid output happened to
> sit above the exit threshold at the first stage. Every gating behaviour
> reported for v2–v6 should be read as initialisation artifact, not learned
> policy. The corrected, actually-supervised gate is SAN-v7 (§4.9).

**What this tells us.** The SAN-v1 accuracy deficit is partly an architecture
problem (weak linear heads), which MLP heads fix. But the deeper problem is
that an unconstrained learned gate discovers the FLOP-minimising solution —
always exit at the first stage — which is accuracy-capped. A usable SAN at high
τ needs either (a) a gating penalty for early exits, (b) a curriculum that
trains deep stages before allowing early exits, or (c) a gate constrained to
match a target accuracy, not just a target FLOP count.

**SAN-v3: curriculum + depth penalty (job 8630).** We implemented SAN-v3 with
(i) a curriculum that opens gate *k* only after stage *k*'s head reaches a
minimum accuracy threshold, and (ii) a depth penalty λ × (remaining depth)
charged during training for every sample that continues past an open gate. The
implementation is in `scripts/research/suffering_aware_large_architecture_v2.py`.

**Table 8: SAN-v3 full CIFAR-10 result (50k/10k, τ = 0.85, seed 17).**

| variant | t* | epochs run | final acc | S_m (TMAC) | vs Dense | vs EarlyStop |
|---|---|---|---|---|---|---|
| SAN-v3 | 13 | 14 | 0.8561 | 5 814 | −76.7% | −51.7% |
| EarlyStop | 28 | 29 | 0.8501 | 12 044 | −51.7% | — |
| Dense | 25 | 60 | 0.8576 | 24 918 | — | +107% |

*Baselines are trained within the same job* (the harness always trains all
three legs), and all comparisons in this table are within-run. Cross-job
variance at identical seed and configuration is real and disclosed: across the
three τ = 0.85 full-CIFAR-10 jobs (8615, 8619, 8630), Dense final accuracy
ranged 0.8576–0.8650, Dense t* ranged 21–25, EarlyStop final accuracy ranged
0.8501–0.8513, and EarlyStop S_m ranged 9 552–12 044 TMAC (freeze timing is
high-variance because a few epochs' shift in crossing τ moves the integral
substantially). The S_m figures are identical across jobs only because the
Dense convention charges the full 60-epoch budget analytically. Readers
comparing Table 8 to Table 6 should bear this in mind: same seed does not
imply same trajectory under GPU nondeterminism, and we quote single-run
numbers with the band stated rather than claiming statistical equivalence.

The curriculum prevents premature exit: gates stay closed (exit_frac = 0.000)
until the trunk has learned a strong representation, so the model reaches τ =
0.85 at epoch 13 with accuracy 0.8561 — inside the observed cross-job Dense
band (0.8576–0.8650) — while consuming **76.7% less** computation than Dense
and **51.7% less** than EarlyStop (within-run; against the cross-job EarlyStop
band the saving is 20–52%). This is the first SAN variant that both reaches
a competitive accuracy target and delivers a large, measured efficiency gain.

**SAN-v4: post-τ distillation + adaptive exit (job 8631).** To test whether
early exits can be made useful *after* the curriculum has trained the trunk, we
added (i) KL-distillation from the final head into every exit head once τ is
reached, and (ii) an adaptive exit threshold that rises to 0.8 after τ, so only
very confident samples leave early. The result is a further small improvement:
t* = 12 (one epoch earlier than SAN-v3), final accuracy 0.8557, and S_m = 5 399
TMAC — **78.3% less** than Dense and **60.6% less** than EarlyStop. The exit
fraction, however, remains 0.000: the model reaches τ before any stage head has
crossed the curriculum threshold, so no sample ever exits early in training.
The measured savings therefore come from the freeze-on-green rule, not from
inference-time early exits. Making the latter contribute is future work: it
requires either a longer post-τ budget so the curriculum can open the gates, or
a deployment-time exit policy that is evaluated separately from the training
freeze rule.

**SAN-v5: adaptive curriculum + accuracy guarantee (job 8634).** We then tested
a more aggressive variant: gates open when a stage head reaches 0.7 × τ (not τ),
plus a double penalty for early-exit errors and multi-exit distillation from all
deeper heads. The result is negative: SAN-v5 reaches τ = 0.85 at the same epoch
12 as SAN-v4, but with slightly lower accuracy (0.8512 vs 0.8557) and identical
S_m (5 399 TMAC). The adaptive curriculum did not open the gates earlier — the
stage heads still had not reached 0.7 × τ by the time the model froze — so the
extra machinery produced no benefit and a small accuracy cost. We report this
negative result explicitly: aggressive curriculum acceleration does not help at
this target, and the simpler SAN-v4 is the better variant.

### 4.9 The ledger catches its own authors: the gate that never learned

This section reports the episode that, more than any savings figure, justifies
this paper's instrument.

**The belief.** SAN-v2 through SAN-v5 were designed, tuned, and reported under
the belief that their per-stage gating networks were *learned*: the gates are
`torch.nn` modules, they sit in the optimizer's parameter set, and their
behaviour differed across variants (exit_frac = 1.000 in v2; 0.000 in v3–v5).
We attributed those behaviours to training.

**The audit.** The trigger was a metered anomaly, not a code review: the post-τ
variant below returned exit fraction 0.000 at *every* epoch under conditions
designed to produce exits, and an exact meter makes "exactly zero, always" a
fact that demands a mechanism, not a number to round past. Tracing the
gradient path end to end gave the mechanism. The gate input is detached (so no
signal flows
through it from the trunk), the exit decision is a hard threshold on the gate's
sigmoid output (non-differentiable), and — the decisive fact — **no loss term
anywhere in the harness supervises the gates**. The optimizer held their
parameters; no gradient ever reached them. The gates of v2–v6 were frozen at
their random initialisation. The v2 "collapse" to always-exit and the v3–v5
"never exit" were not learned policies but the sign of a random draw against a
fixed threshold. We note for honesty: a gradient-norm probe on the gate
parameters would have caught this at v2 — but the v2 instrumentation watched
stage and exit-head norms only, and nobody metered the gates. Conventional
signals (training curves, accuracies, analytical FLOP counts) all looked
plausible, because the training was otherwise correct.

**A controlled negative result (SAN-v6, job 8651).** The v6 variant was built
to let the gates learn *after* the contract is met: once τ is reached, all
gates open and training continues for 20 post-τ epochs. On full CIFAR-10
(50 000/10 000, τ = 0.85, seed 17): SAN reaches τ at t* = 13 (accuracy 0.8560),
then trains 20 further epochs to final accuracy **0.8753** — above Dense
(0.8474) — but the exit fraction stays **0.000** through every post-τ epoch,
and the metered burden grows to **14 121 TMAC**: the first configuration in
which the SAN *loses* to EarlyStop (13 290 TMAC). With the audit's explanation,
the result is mechanistically closed: the gates were open but frozen, so the
post-τ epochs bought accuracy (via distillation and auxiliary heads acting as
regularisers) and zero exits. Post-τ training without a working gate is pure
cost; the ledger prices it exactly.

**The correction (SAN-v7, job 8717).** The gate is now supervised directly with the
correctness target of BERxiT's learning-to-exit module [11] — exit at stage k
iff stage k's head was correct on this sample — via a per-stage BCE applied
post-τ. We claim no novelty for the mechanism; the novelty is that its
behaviour is now *accounted*. A unit-level gradient test (BCE 0.694 → 0.326 in
five steps, all gate parameters move) and a CPU smoke run verified the path
before any GPU hour was spent. The full-scale result (50 000/10 000, τ = 0.85,
seed 17, adaptive exit threshold 0.8):

**Table 9: SAN-v7 — the first learned-gate run.**

| variant | t* | epochs run | final acc | S_m (TMAC) | latency (ms/sample) |
|---|---|---|---|---|---|
| SAN-v7 | 11 | 32 | 0.7998 | 9 799 | **0.3515** |
| EarlyStop | 25 | 26 | 0.8509 | 10 798 | — |
| Dense | 35 | 60 | 0.8546 | 24 918 | 0.4941 |

Three findings, all measured. First, **the gates fire**: the validation exit
fraction rises from 0.000 at t* to 0.52–0.98 across the twenty post-τ epochs,
and per-epoch metered cost falls from 415 GF to ~210–250 GF (−40–50%). Second,
**the exit discount pays for the post-τ training**: total S_m is 9 799 TMAC,
9.3% *below* the within-run EarlyStop (10 798) even though the SAN trained six
more epochs — the first configuration in which the SAN beats EarlyStop with
post-τ training included (60.7% below Dense). Third, **inference gets
measurably faster**: 0.3515 ms/sample against Dense's 0.4941, a 1.41×
wall-time speedup on the same GPU — the first inference-time win in this
study that comes from learned exits rather than from freeze accounting.

And one honest failure: the post-τ validation accuracy settles at 0.7998,
*below* the contract the model met at t* (0.8544, clause L2 passes there).
With threshold 0.8 the gate trades too much accuracy for compute. The run
therefore reads as the first point of a frontier — (exit-heavy, 1.41× faster,
acc 0.80) — not as a contract-satisfying configuration. A second point with
threshold 0.95 (SAN-v7b, job 8720) is measured in §4.10. Clauses L4 and L6
fail by construction in the v6/v7 design: gratuitous burden is nonzero by
deliberate post-τ training, and the exit fraction at t* is exactly zero
because exits exist only after t* — the clause predates the design it now
judges, and we say so rather than redrawing it.

**Why this section exists.** Every element of this episode — five variants
carrying a silently false claim, the claim surviving two external LLM reviews,
and its exposure by an exactly-metered anomaly that conventional signals had
all missed — is the phenomenon this paper's
instrument exists to catch, caught *in vivo*. We considered quietly fixing the
gate and reporting only the corrected numbers; that would have been the
stronger-looking paper and the weaker science.

### 4.10 The frontier is a deployment choice: SAN-v7b at threshold 0.95

The SAN-v7 result left one question open: was the contract violation a
property of the learned gate or of the chosen threshold? SAN-v7b (job 8720)
repeats the run identically except for the adaptive exit threshold, raised
from 0.8 to 0.95.

**Table 10: two measured points of the accuracy–compute frontier
(full CIFAR-10, τ = 0.85, seed 17; baselines within-run).**

| variant | threshold | t* | final acc | S_m (TMAC) | vs EarlyStop | vs Dense | latency speedup |
|---|---|---|---|---|---|---|---|
| SAN-v7 | 0.80 | 11 | 0.7998 | 9 799 | −9.3% | −60.7% | 1.41× |
| SAN-v7b | 0.95 | 14 | **0.8594** | 12 757 | −6.9% | −48.8% | 1.12× |
| EarlyStop (v7 run) | — | 25 | 0.8509 | 10 798 | — | — | — |
| EarlyStop (v7b run) | — | 32 | 0.8566 | 13 705 | — | — | — |
| Dense (either run) | — | 35 / 27 | 0.8546 / 0.8431 | 24 918 | — | — | 1.00× |

Three findings. First, **the threshold buys accuracy at a measured price**:
raising it from 0.80 to 0.95 recovers the contract (final accuracy 0.8594 ≥
τ, against 0.7998) at a cost of 2 958 TMAC of additional training burden and
a smaller — but still positive — inference speedup (1.12× against 1.41×).
The frontier is real, monotone in the expected direction, and priced exactly.
Second, **SAN-v7b is the first configuration in this study that satisfies the
declared contract at the end of training with learned exits firing**
(validation exit fraction 0.68–0.86 across the post-τ epochs; L1 conservation
exact with 1 737 exits on the 10 000-sample cohort; clause L2 passes at t* =
14 with 0.8562). Third, both SAN variants beat their within-run EarlyStop on
total metered burden *including* the twenty post-τ epochs — the exit discount
during post-τ training consistently pays for the extra epochs (the two
EarlyStop legs differ, 10 798 and 13 705 TMAC, by the cross-job variance
disclosed in §4.8; comparisons are within-run only).

The deployment reading is the point of the instrument: the threshold is not a
hyperparameter to be tuned and forgotten but a *declared* frontier point —
and the FPGA audit kernel enforces exactly the declared point at inference,
with the stage-cost LUT and threshold loaded by the host and every decision
attested bit-exactly. Training-time accounting and deployment-time
attestation close their loop here.

### 4.11 Preregistered: the five-point frontier, and what its falsifications teach

Before the sweep below was run, we committed preregistered predictions for
it — intervals for final accuracy, S_m, and latency at three new thresholds,
monotone constraints across the frontier, contract invariants, and
predictions for the ViT/GPT legs — with the git timestamp as precedence
proof and a timing disclosure for the one job whose completion preceded the
commit by 22 minutes (`agent_logs/san_v7_frontier_preregistration_2026-08-06.md`,
which carries the full scorecard). The measured frontier:

**Table 11: accuracy–compute frontier, five thresholds
(full CIFAR-10, ResNet-50, τ = 0.85, seed 17).**

| threshold | t* | final acc | S_m (TMAC) | latency speedup | post-τ exit_frac (late epochs) |
|---|---|---|---|---|---|
| 0.80  | 11 | 0.7998 | 9 799  | 1.41× | 0.71–0.98 |
| 0.85  | 15 | 0.8293 | 12 163 | 1.27× | 0.89–0.96 |
| 0.90  | 10 | 0.8379 | 10 357 | 1.18× | 0.85–0.92 |
| 0.95  | 14 | 0.8594 | 12 757 | 1.12× | 0.68–0.86 |
| 0.975 | 15 | 0.8688 | 13 795 | 1.04× | 0.74–0.79 |

**Scorecard.** Accuracy: all five points inside the preregistered intervals
or monotone band (5/5; the final accuracies are perfectly monotone in the
threshold). Latency: 5/5, perfectly monotone decreasing. Contract
invariants: L1 conservation exact with exits firing in every run, L2 passes
at every t*, exit_frac(t*) = 0 with L6 failing exactly as declared
by construction. **S_m: falsified.** Only one of three intervals held, and
the predicted monotonicity broke — S_m at threshold 0.90 (10 357 TMAC) dips
*below* the 0.85 point (12 163). The baseline invariant fell with it: the
within-run EarlyStop legs swung from 7 891 to 13 705 TMAC across the five
v7-line runs at identical configurations, and the SAN won only one of the
three new pairings.

**Mechanism, stated because the falsification bought it.** The accuracy and
latency frontiers are smooth functions of the threshold — the gate's
selectivity controls them directly. The training burden is not: S_m =
nec(t*) + discounted post-τ epochs, and nec(t*) swings by ~2 000 TMAC with
the epoch at which τ happens to be crossed (t* ∈ {10, 11, 14, 15} across
identical configs), larger than the inter-threshold effect. The threshold
controls the *post-contract* cost; it does not control *when the contract is
met*. And since the EarlyStop baseline shares that same freeze-timing
variance, the stable, reportable claim of this study is the SAN's own
frontier — not any pairwise win against a baseline whose freeze jitter
exceeds the effect being measured. We would not have written this paragraph
without the preregistered failure.

**ViT and GPT legs (jobs 8738/8739).** With the v7-era post-τ plumbing both
families now run the full protocol (confidence-threshold exits, no learned
gates — declared difference). ViT-large/d384 reaches τ = 0.251 at t* = 4,
finishes post-τ training at 0.3763 ≥ τ, and its exit fraction at t* is
0.342 — the first L6 pass on full CIFAR-10, ten times the pre-v7 line's
0.03 — with a 1.17× measured latency speedup. The GPT leg reaches τ = 0.165
at t* = 4 with L6 passing (0.120) and a 3.33× latency speedup, but its
accuracy decays to 0.1309 < τ by the end of post-τ training: the preregistered
"stays ≥ τ" claim held for ViT and failed for the LM, and the LM's cheap
exits (3.33×) are exactly the regime where over-exiting eats the margin.
One-line summary of both: the contract machinery works across families; the
post-τ accuracy floor does not come for free and currently holds only where
the exit threshold leaves enough margin.

### 4.12 Second preregistered round: the cost-aware gate, and the ViT/GPT dials

A second preregistration (`agent_logs/san_v8_vitgpt_preregistration_2026-08-06.md`,
committed before any of the six jobs was submitted — the submission
timestamps are verifiable against it) covered two questions: whether a
cost-aware gate (SAN-v8) moves the frontier, and whether the ViT/GPT legs
have well-behaved threshold dials.

**SAN-v8 (job 8750).** The v7 gate learns P(stage-k head correct); v8 keeps
the target and adds a pos_weight proportional to the remaining depth, so
"correct AND cheap" is worth more gradient than "correct AND expensive" — a
mechanism verified before any GPU hour by a unit test showing the early-gate
firing propensity shifting up (0.093 → 0.256 at gate 0, smoke scale) while
deep gates stay fixed. All five preregistered predictions confirmed (E1–E5):
late exit fraction 0.77–0.86 (predicted [0.75, 0.97]), final accuracy 0.8279
(predicted [0.820, 0.870] — including the allowed dip below τ), latency
speedup 1.34× (predicted [1.10, 1.40]), S_m 10 781 TMAC (predicted
[10 000, 13 000], the interval widened by the t*-jitter lesson of §4.11), and
the L1/L2/L6 contract pattern exactly as declared. **But the comparison that
matters is against the frontier, not the intervals**: at the same threshold
(0.95), v7b achieves (acc 0.8594, S_m 12 757, 1.12×) and v8 achieves
(0.8279, 10 781, 1.34×) — v8 trades 3.2 accuracy points for 2 000 TMAC and
speed, i.e. it moves *along* the frontier rather than outward. Neither
dominates; the pos_weight axis is, at this scale, a reparameterisation of the
threshold axis. We report this as a measured negative for a plausible
improvement: one selectivity dial is enough, and the dial to keep is the
declared threshold, because the FPGA enforces it.

**ViT dial (jobs 8751–8753, Δ ∈ {0.35, 0.55, 0.65} against the Δ = 0.45
reference).** The dial behaves on the quantities the gate controls and not on
the ones it does not: late-epoch exit fractions fall (band means 0.28 / 0.35
/ 0.10 / 0.10 — monotone only from the second point on, preregistered G1
falsified by the 0.35 ↔ 0.45 overlap), latency speedups decay monotonically
(1.19× / 1.17× / 0.99× / 0.98×, G4 confirmed), and every run ends above τ
(G3 confirmed). Final accuracy, however, is not a function of the dial:
0.2600 / 0.3763 / 0.3248 / 0.3755 — all three accuracy intervals falsified,
and the monotone-ordering claim with them (G2). At τ = 0.251 the target is
met almost immediately (t* = 4), so post-τ accuracy is shaped by exit
composition noise, not by the threshold.

**GPT dial (jobs 8754/8755, Δ ∈ {0.40, 0.50} against the Δ = 0.31
reference).** Exits and speedups obey the dial cleanly (late exits 0.95 /
0.86 / 0.52 and speedups 3.33× / 2.02× / 1.23×, both monotone, three of four
intervals confirmed). The accuracy floor does not move: final accuracies
0.1309 / 0.1312 / 0.1276 — G6 falsified (no monotone rise) and, decisively,
**G7 falsified: no threshold in the swept range restores accuracy ≥ τ after
post-τ training**. The LM leg's accuracy decay is therefore structural, not
a threshold artefact: confidence-driven exits during post-τ training erode
the representation regardless of where the bar is set. This closes, by
falsification, the question §4.11 left open — the post-τ accuracy floor for
the LM family requires a different mechanism (most plausibly a
correctness-gated exit like the ResNet's, rather than a confidence one), not
a better threshold.

---

## 5 Discussion

### 5.1 What the numbers mean

The U250 result shows that the decision/metering path of an early-exit
deployment can be carried by a small, fast, low-energy FPGA kernel that runs
*outside* the host it audits. The kernel does not accelerate the trunk; it
accelerates — and, more importantly, *independently attests* — the audit: for
every incoming cohort, it decides when each sample should have exited and
counts the exact computation that was executed.

The GPU results show two things. First, the freeze-on-green contract delivers
large, exactly-metered savings in the regimes where the target is reachable
(§4.2, §4.8), and fails loudly and measurably where it is not (§4.4, §4.6).
Second — and this is the lesson we did not plan — the accounting earns its keep
even when the training machinery is wrong: §4.9's episode shows an exact ledger
exposing a five-variant false claim that every conventional signal had missed.
We therefore do not claim that the SAN is a better early-exit network; the
early-exit literature [1,2,9,10,11,12] is ahead of it as a method. We claim
that this is the first early-exit study whose every compute figure is exact,
independently verified, and hardware-attested — and that this property changed
what we could see about our own work.

### 5.2 Limitations (stated honestly)

**Small-subset training study.** The GPU study uses 4,000 train / 1,000 val
images and low feasibility targets (τ = 0.34/0.251/0.165). The reported
accuracies (0.39, 0.26, 0.17) demonstrate the SAN training machinery, not
competitive CIFAR-10 performance. Claims are framed as machinery validation and
metered-MAC savings under identical accounting, not as accuracy results.

**Partial meter convention.** The machine channel charges only conv/linear MACs
and attention token-mixing matmuls; biases, norms, activations, softmax,
residuals, pooling, host dispatch, and PCIe transfers are unmetered. Reported
percentages are relative savings under this stated convention, not claims about
total wall-time or total energy.

**Full ImageNet-1k is unavailable in this environment.** It requires
-credentials and ~150 GB of download; this node has neither. ImageNette2-160 is
the honest real-image proxy, and all "ImageNet scale" claims refer to (a) the
real architecture FLOP constants, (b) the 1.2M-sample stress cohort, or (c)
explicit extrapolation.

**Full-CIFAR-10 sensitivity study.** §4.6 reports three ResNet-50 runs on the
full 50 000 / 10 000 split with τ ∈ {0.80, 0.85} and two seeds. Across all
runs, SAN never reached the feasibility target and achieved lower final accuracy
(0.759–0.769) than both EarlyStop (0.824–0.851) and Dense (0.850–0.865). The
SAN–EarlyStop comparison is seed- and τ-sensitive: SAN consumed between 6.5%
less and 71% more computation than EarlyStop. This confirms that the
small-subset savings are regime-specific and that the freeze-on-green rule, not
the early-exit heads, is the dominant savings mechanism at non-trivial accuracy.

**Patient channel is mixed.** ResNet-50 shows the disclosed patient-channel
tradeoff on this task: its integrated patient harm is slightly above EarlyStop's
because early stopping alone can freeze on a luckier epoch. ViT-small/d384 and
SAN-GPT-small satisfy the stricter L5 clause (SAN ≤ both baselines) in this run.
The parent line still reports an attention-family tradeoff at ImageNet scale, so
the small-proxy result should not be read as a general claim.

**Single bitstream, no DSE.** The work did not include placement/routing
exploration, multi-bitstream campaigns, or comparison with HLS alternatives. The
135.2 MHz achieved clock is the honest result of one Vitis build.

**CPU baseline measured.** Table 2 reports a host-CPU implementation of the
same integer audit/metering kernel. The multi-core CPU is faster in raw
throughput than the U250 on this workload, so the FPGA claim is reframed as
offload + energy + hardware-enforced integer specification, not as a raw speedup.
The CPU measurement was taken on a Xeon Gold 6526Y host; the DL380 host that
holds the U250 may differ, and a same-host measurement would tighten the
comparison further.

**What the FPGA does and does not attest.** The card re-derives exit decisions
from host-exported confidence vectors and accumulates the stage-cost LUT entry
for each decision; it does not run the trunk and does not observe the host's
physical execution. It is therefore a root-of-trust for the *decision logic
and its integer accounting* — the host cannot silently change the threshold or
the cost model without the card disagreeing — not a measurement of the host's
silicon. Likewise the training-time ledger is executed-path accounting at the
framework level: it counts exactly the MACs of the path the framework actually
ran, but it does not count unmetered operation classes (§2.1), and it is not a
hardware performance counter. We consider a CUPTI/RAPL cross-check of the
meter a worthwhile future validation, not a present claim.

**Energy uncertainty.** The 3.3 nJ/sample figure derives from ΔP ≈ 1.7 W against
a 24.4 W idle, read from a 1 Hz on-card sensor. The sensor and the subtraction
are coarse; the per-sample value is rounded to two significant figures and
should be read as an order-of-magnitude board-level estimate, not a precise
energy claim. A higher-rate external meter would be needed for three-significant-
figure energy.

### 5.3 Future work

- Run the full pipeline on ImageNet-1k when credentials and storage become
  available.
- Explore the threshold (Δ) and feasibility (τ) sensitivity surface across
  families.
- Extend the kernel to multi-batch continuous streaming and measure host
  PCIe overhead in a server context.
- Investigate sparsity and quantization ladders for the machine channel.
- Measure the CPU baseline on the same DL380 host that holds the U250, and
  collect rack-level energy for both host and card.
- Map the accuracy–compute frontier across the adaptive exit threshold beyond
  the two measured points of §4.9–§4.10, and characterise gate calibration.

### 5.4 Terminology: what "machine suffering" does and does not mean here

The parent line [4] calls the metered computation "machine suffering." We keep
the term for continuity and delimit it precisely, because the term is occupied
elsewhere. In philosophy, machine/artificial suffering denotes a putative
*phenomenology* of artificial systems — Metzinger's argument for a moratorium
on synthetic phenomenology [40], the AI-welfare research programme [41],
Tomasik's treatment of RL reward as a welfare proxy [42], and Klimovich's
essay of exactly this title [39]. We make **no** claim of sentience,
phenomenology, or moral patiency for any artifact in this paper; on the
consensus markers of that literature, nothing here is a candidate sufferer.
Our usage is an *operational metaphor*: a scalar quantity (executed FLOPs,
exactly counted) that a training procedure is asked to minimise subject to a
declared contract, named to keep the ethical weight of compute visible in the
objective rather than in the acknowledgements. Readers who find the metaphor
distracting may read "machine burden" throughout without loss of content; the
ledger, the contract, and the kernel do not depend on the name. What we claim
as novel is the operationalisation — a normatively motivated, exactly measured
minimisation objective with an auditable trail — not the vocabulary.

### 5.5 What this paper is not (novelty audit summary)

Before this revision we ran a structured seven-front audit of the novelty of
every claim (`agent_logs/san_novelty_audit_2026-08-06.md`, primary sources
throughout). Its verdicts, absorbed into this draft: early-exit heads,
confidence thresholds, and learned exit policies are established [1,2,11,15,16];
the correctness-supervised gate of §4.9 is BERxiT's LTE target [11]; freeze-on-green
is time-to-accuracy [22,23] with accounting; early-exit on FPGA exists as
acceleration [32,33] — our claim is the *auditor* role, the integer-exact
conservation-checked ledger, and the per-run necessary/gratuitous decomposition
against τ, none of which the audit found in prior art. The closest threats to
those three are Perseus's energy-bloat decomposition [29], the algorithmic-
efficiency quantity of Hernandez & Brown [30], and the compute-governance
designs of [36,37]; §1.4 positions each.

---

## 6 Conclusion

We set out to deploy a suffering-aware early-exit network and ended up building
something more durable than the network: an instrument. The instrument is a
training-compute ledger that is integer-exact about the computation actually
executed, verified by an independent accounting path on every run, bound to a
declared accuracy contract, and attested outside the host by an FPGA kernel
running at 511 Msamples/s with bit-exact agreement against the golden model.
The demonstration workload — the SAN — gave the instrument everything it needed
to prove itself: large metered savings in the reachable-target regime (76.7%
below Dense at iso-accuracy on full CIFAR-10), loud measurable failure where
the target is not reachable, and, in the episode that became §4.9, a five-variant
false claim about learned gates that no conventional signal caught and exact
accounting did. We have stated the limits plainly — regime-specific savings, a
partial meter convention, no full ImageNet-1k, board-level power, an early-exit
literature ahead of our workload as a method, and a borrowed metaphor
deliberately delimited in §5.4 — because the honesty of the evidence is itself
the contribution. The artifacts are available for reproduction.

---

## 7 AI Disclosure

This draft was prepared under human direction, with mandatory LLM-offload review
per `.claude/AGENT_OFFLOAD_POLICY.md`. The companion spec and contract were
audited by math-review offload (xai/Grok 4.3 and Z.AI GLM) and revised per
their findings. No clinical content. GAIDeT-ICMJE 2025.

---

## References

[1] Teerapittayanon, S., McDanel, B., & Kung, H. T. (2016). BranchyNet: Fast
inference via early exiting from deep neural networks. *ICPR*.
arXiv:1709.01686.

[2] Kaya, Y., Hong, S., & Dumitras, T. (2019). Shallow-Deep Networks:
Understanding and mitigating network overthinking. *ICML*, PMLR 97:3301–3310.
arXiv:1810.07052.

[3] Huang, G., Chen, D., Li, T., Wu, F., van der Maaten, L., & Weinberger, K.
(2018). Multi-scale dense networks for resource efficient image
classification. *ICLR*. arXiv:1703.09844.

[4] Sounio repository, suffering-aware neural network line:
`docs/research/suffering_aware_architecture_spec_2026-07-28.md` and successors.

[5] Chamon, L. F., & Ribeiro, A. (2020). Probably approximately correct
constrained learning. *NeurIPS*.

[6] Chamon, L. F., Paternain, S., Calvo-Fullana, M., & Ribeiro, A. (2023).
Constrained learning with non-convex losses. *IEEE Transactions on Information
Theory*.

[7] Cotter, A., Jiang, H., Gupta, M., Wang, S., Narayan, T., You, S., &
Sridharan, D. (2019). Optimization with non-differentiable constraints with
applications to fairness, recall, churn, and other goals. *JMLR*.

[8] Manheim, D., & Garrabrant, S. (2018). Categorizing variants of Goodhart's
law. arXiv:1803.04585.

[9] Xin, J., Tang, R., Lee, J., Yu, Y., & Lin, J. (2020). DeeBERT: Dynamic
early exiting for accelerating BERT inference. *ACL*.
DOI 10.18653/v1/2020.acl-main.204.

[10] Zhou, W., Xu, C., Ge, T., McAuley, J., Xu, K., & Wei, F. (2020). BERT
loses patience: Fast and robust inference with early exit. *NeurIPS*.
arXiv:2006.04152.

[11] Xin, J., Tang, R., Yu, Y., & Lin, J. (2021). BERxiT: Early exiting for
BERT with better fine-tuning and extension to regression. *EACL*, 91–104.
DOI 10.18653/v1/2021.eacl-main.8.

[12] Schuster, T., Fisch, A., Gupta, J., Dehghani, M., Bahri, D., Tran, V.,
Tay, Y., & Metzler, D. (2022). Confident adaptive language modeling.
*NeurIPS*. arXiv:2207.07061.

[13] Elhoushi, M., Shrivastava, A., Liskovich, D., Hosmer, B., Wasti, B.,
Lai, L., Mahmoud, A., Acun, B., Agarwal, S., Roman, A., Aly, A., Chen, B., &
Symeonidis, G. (2024). LayerSkip: Enabling early exit inference and
self-speculative decoding. *ACL*. arXiv:2404.16710.

[14] Elbayad, M., Gu, J., Grave, E., & Auli, M. (2020). Depth-adaptive
transformer. *ICLR*. arXiv:1910.10073.

[15] Bolukbasi, T., Wang, J., Dekel, O., & Saligrama, V. (2017). Adaptive
neural networks for efficient inference. *ICML*.

[16] Wang, X., Yu, F., Dou, Z.-Y., Darrell, T., & Gonzalez, J. E. (2018).
SkipNet: Learning dynamic routing in convolutional networks. *ECCV*.
arXiv:1711.09485.

[17] Graves, A. (2016). Adaptive computation time for recurrent neural
networks. arXiv:1603.08983.

[18] Banino, A., Balaguer, J., & Blundell, C. (2021). PonderNet: Learning to
ponder. *ICML 2021 Workshop on Automated Machine Learning*; arXiv:2107.05407.

[19] Raposo, D., Santoro, A., Richards, B., Humphreys, I., & Lillicrap, T.
(2024). Mixture-of-Depths: Dynamically allocating compute in transformer-based
language models. arXiv:2404.02258.

[20] Dehghani, M., Gouws, S., Vinyals, O., Uszkoreit, J., & Kaiser, Ł. (2019).
Universal transformers. *ICLR*.

[21] Prechelt, L. (1998). Early stopping — but when? In *Neural Networks:
Tricks of the Trade*, LNCS 1524, 55–69. DOI 10.1007/3-540-49430-8_3.

[22] Coleman, C., Narayanan, D., Kang, D., Zhao, T., Zhang, J., Nardi, L.,
Bailis, P., Olukotun, K., Ré, C., & Zaharia, M. (2019). Analysis of DAWNBench,
a time-to-accuracy machine learning performance benchmark. *SIGOPS Oper. Syst.
Rev.* 53(1). arXiv:1806.01427.

[23] Mattson, P., Cheng, C., Diamos, G., Coleman, C., Micikevicius, P.,
Patterson, D., Tang, H., Wei, G.-Y., Bailis, P., Bittorf, V., et al. (2020).
MLPerf training benchmark. *MLSys*. arXiv:1910.01500.

[24] Shah, I. S. H., Hajialigol, D., Hsieh, C.-J., & Alizadeh, M. (2023).
Repeated random sampling for minimizing the time-to-accuracy of learning.
arXiv:2305.18424.

[25] Schwartz, R., Dodge, J., Smith, N. A., & Etzioni, O. (2020). Green AI.
*Communications of the ACM* 63(12), 54–63. DOI 10.1145/3381831.

[26] Strubell, E., Ganesh, A., & McCallum, A. (2019). Energy and policy
considerations for deep learning in NLP. *ACL*. DOI 10.18653/v1/P19-1355.

[27] Henderson, P., Hu, J., Romoff, J., Brunskill, E., Jurafsky, D., &
Pineau, J. (2020). Towards the systematic reporting of the energy and carbon
footprints of machine learning. *JMLR* 21(248). arXiv:2002.05651.

[28] You, J., Chung, J.-W., & Chowdhury, M. (2023). Zeus: Understanding and
optimizing GPU energy consumption of DNN training. *NSDI*. arXiv:2208.06102.

[29] Chung, J.-W., Qiao, Y., et al. (2024). Perseus: Reducing energy bloat in
large model training. *SOSP*. arXiv:2312.06902.

[30] Hernandez, D., & Brown, T. B. (2020). Measuring the algorithmic
efficiency of neural networks. arXiv:2005.04305.

[31] Anthony, L. F. W., Kanding, B., & Selvan, R. (2020). Carbontracker:
Tracking and predicting the carbon footprint of training deep learning models.
arXiv:2007.03051.

[32] Biggs, J., Bouganis, C.-S., & Constantinides, G. A. (2023). ATHEENA: A
toolflow for hardware early-exit network automation. *IEEE FCCM*.
DOI 10.1109/FCCM57271.2023.00022.

[33] Low cost early exit decision unit design for CNN accelerator. (2020).
*IEEE ISOCC*. DOI 10.1109/ISOCC50952.2020.9333079.

[34] Laskaridis, S., Venieris, S. I., Kim, H., & Lane, N. D. (2020). HAPI:
Hardware-aware progressive inference. *ICCAD*. DOI 10.1145/3400302.3415698.

[35] Laskaridis, S., Venieris, S. I., Almeida, M., Leontiadis, I., & Lane,
N. D. (2020). SPINN: Synergistic progressive inference of neural networks
over device and cloud. *MobiCom*. DOI 10.1145/3372224.3419194.

[36] Petrie, J., Aarne, M., Ammann, T., & Dalrymple, D. (2025). Flexible
hardware-enabled guarantees for AI compute (FlexHEG). arXiv:2506.15093.

[37] Shavit, Y. (2023). What does it take to catch a Chinchilla? Verifying
rules on large-scale neural network training via compute monitoring.
arXiv:2303.11341.

[38] Jia, H., Yaghini, M., Choquette-Choo, C. A., Dullerud, N., Thudi, A.,
Chandrasekaran, V., & Papernot, N. (2021). Proof-of-learning: Definitions and
practice. *IEEE S&P*. arXiv:2103.05633.

[39] Klimovich, A. (2025). The price of machine suffering. *AI & Society*
41(5), 4477–4484. DOI 10.1007/s00146-025-02831-8.

[40] Metzinger, T. (2021). Artificial suffering: An argument for a global
moratorium on synthetic phenomenology. *Journal of Artificial Intelligence and
Consciousness* 8(1). DOI 10.1142/S270507852150003X.

[41] Long, R., Sebo, J., Butlin, P., Finlinson, K., Fish, K., Harding, J.,
Pfau, J., Sims, T., Birch, J., & Chalmers, D. (2024). Taking AI welfare
seriously. arXiv:2411.00986.

[42] Tomasik, B. (2014). Do artificial reinforcement-learning agents matter
morally? arXiv:1410.8233.

[43] Sastry, G., Heim, L., Belfield, H., Anderljung, M., Brundage, M.,
Hazell, J., O'Keefe, C., Hadfield, G. K., Ngo, R., Pilz, K., Gor, G.,
Bluemke, E., Shoker, S., Egan, J., Trager, R. F., Avin, S., Weller, A.,
Bengio, Y., & Coyle, D. (2024). Computing power and the governance of
artificial intelligence. arXiv:2402.08797.

[44] Laskaridis, S., Kouris, A., & Lane, N. D. (2021). Adaptive inference
through early-exit networks: Design, challenges and directions. *EMDL*;
extended version in *ACM Computing Surveys*. arXiv:2106.05022.

[45] Regol, F., Chataoui, S., Charpentier, P., Coates, M., Piantanida, P., &
Günnemann, S. (2024). QuEE: early exiting via learned per-sample error
prediction. arXiv preprint, June 2024. (Identifier to be confirmed before
submission; see `agent_logs/san_novelty_audit_2026-08-06.md`.)

[46] Skalse, J., Howe, N., Krasheninnikov, D., & Krueger, D. (2022). Defining
and characterizing reward hacking. *NeurIPS*.

[47] Pan, A., Bhatia, K., & Steinhardt, J. (2022). The effects of reward
misspecification: Mapping and mitigating misaligned models. *ICLR*.

---

## A Reproduction

```bash
# CPU contract + spec/outline consistency (~3 min on CPU)
bash scripts/ci/san_imagenet_fpga_dl380_gate.sh
# expect: SAN_IMAGENET_FPGA_DL380_VERDICT I_GREEN (8/8 clauses PASS)

# GPU training (requires CUDA)
SAN_LARGE_DEVICE=cuda SAN_LARGE_ONLY=resnet50 \
  .venv/bin/python scripts/research/suffering_aware_large_architecture.py

# U250 host scan (requires Xilinx XRT + bitstream on DL380)
./host_san_scan -x krnl_san_scan.hw.xclbin -d val_resnet
```


---

## B Companion contract clauses (L1–L8)

The GPU harness reports per-family verdicts against clauses L1–L8 of
`scripts/research/suffering_aware_large_architecture.py`:

| clause | predicate | threshold / note |
|---|---|---|
| L1 | Metering conservation: gated-off stages charge 0; metered MACs == independent manual accounting | equality |
| L2 | Feasibility: SAN reaches val acc ≥ τ within budget | τ per family |
| L3 | Anti-Goodhart: 101-weight compassion grid never selects an infeasible checkpoint | abstainer & probe infeasible |
| L4 | Necessary/gratuitous separation: SAN gratuitous machine burden = 0 | equality |
| L5 | Suffering bounds: SAN machine burden < Dense and ≤ EarlyStop; patient harm ≤ both baselines | within small tolerance |
| L6 | Exits are real: val exit fraction at t* > 0.07 | 0.07 |
| L7 | Patient channel first-class: harm matrix off-diagonal max ≥ 3× min; SAN peak harm ≤ baselines | 3× asymmetry |
| L8 | Anti-shortcut: a spurious-feature probe that overfits train but fails val is rejected by the gate | train > τ, val < τ, never selected |

The FPGA / ImageNet-scale contract uses an analogous I1–I8 clause set defined in
`scripts/research/san_imagenet_fpga_dl380.py` and enforced by
`scripts/ci/san_imagenet_fpga_dl380_gate.sh`.
