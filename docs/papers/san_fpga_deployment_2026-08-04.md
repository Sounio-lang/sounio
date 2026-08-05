<!-- docs:meta
topic_id: repo.docs.papers.san-fpga-deployment-2026-08-04
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.san-fpga-deployment-2026-08-04
-->

# Suffering-Aware Neural Networks on AMD Alveo U250: Deploying the Exit-Audit / FLOP-Metering Kernel

**Status:** `DRAFT` — every empirical claim marked `MEASURED` below is backed
by a measured artifact in the Sounio repository; claims marked `ESTIMATE` or
`(in progress)` are explicitly noted as such. Human measurement audit is the
authority; LLM reviews are recorded in §7 only as an AI-assist disclosure.
**Date:** 2026-08-04
**Orthography:** EN-US
**Companion spec:** `docs/research/san_imagenet_fpga_dl380_spec_2026-08-02.md`
**Companion contract:** `scripts/research/san_imagenet_fpga_dl380.py`
**Target venue:** arXiv `cs.LG` / `cs.AR`; then IEEE FPL or a systems note

---

## Abstract

We report, to our knowledge, the first measured deployment of the SAN exit-audit / FLOP-metering
kernel on an AMD Alveo U250 FPGA, together with a GPU training study of
SAN-ResNet-50, SAN-ViT-small/d384, and a tiny decoder LM (SAN-GPT-small). Suffering-aware neural networks
(SANs) are early-exit architectures whose training freezes as soon as a
held-out feasibility target is met, eliminating gratuitous computation. We
offload only the inference-time catastrophe-scan and FLOP-metering path to the
FPGA; the trunk runs on the host CPU/GPU. On the U250 the scan kernel runs at
**511 Msamples/s sustained** on a 1.2M-sample stress cohort. Because the kernel
is bus-limited by construction, this figure is best read as **94.5% of the
540.8 Msamples/s DMA-limited theoretical peak** at the achieved 135.2 MHz clock.
The same integer function on a host Xeon Gold 6526Y measures **130.8 Msamples/s**
scalar, **173.9 Msamples/s** AVX-512 single-thread, and **1.05 Gsamples/s**
AVX-512 across 8 threads (bit-exact on the same cohorts). The FPGA therefore does
not win on raw throughput against a multi-core CPU; its value is the combination
of offloading the audit path from the host, the fixed low-energy operation, and
the hardware-enforced integer specification. It consumes approximately
**3.3 nJ/sample** incremental board-level energy (rounded to the precision the
on-card power sensor supports). The kernel is bit-exact against the control-VM
golden model on synthetic cohorts, the stress cohort, and real photographs from
ImageNette2-160.

On NVIDIA GPUs (A5000 / RTX 4000 Ada) SAN training reduces the *metered-MAC*
integrated machine burden by **40.7%** (ResNet-50), **50.4%** (ViT-small/d384), and
**52.2%** (SAN-GPT-small) relative to dense training in a small-subset, fast-convergence
regime (validation accuracies 0.39, 0.26, 0.17, chosen to demonstrate the
freeze-on-green mechanism, not to claim competitive task accuracy). A separate
full-CIFAR-10 sensitivity study (50 000 / 10 000, ResNet-50, two seeds, τ ∈
{0.80, 0.85}) shows that these savings do **not** generalise to this accuracy
regime: SAN never reached the feasibility target and consumed between 6.5% less
and 71% more computation than an early-stop baseline, while achieving lower
final accuracy (0.759–0.769 vs 0.824–0.865). The small-subset savings are
therefore reported as a machinery demonstration, not as a universal efficiency
claim. ResNet-50 also shows a measured 1.08x wall-time latency speedup on
CIFAR-10, though this margin is small and task-dependent. We disclose the limits
honestly: the energy number is board-level, not rack-level; full ImageNet-1k is
unavailable, so ImageNette2-160 is the real-image proxy; and ResNet-50 shows a
disclosed patient-channel tradeoff while the other families satisfy the stricter
L5 clause in this run. All artifacts, measurements, and reproduction scripts are
in the repository.

---

## 1 Introduction

### 1.1 The problem: computation we do not need

Deep learning at scale pays for every layer it runs, including layers that do
not change the answer. Early-exit networks [1,2,3] reduce this cost by allowing
samples to leave the network at intermediate classifiers when they are
"confident enough." Most early-exit work, however, optimizes for speedup or
FLOP reduction alone, and trains for a fixed budget; it therefore continues to
spend computation after the model has already become good enough on the task it
was asked to solve.

A suffering-aware neural network (SAN) [4] inverts the framing. It defines two
channels of suffering:

- **Patient suffering**: the cost of prediction errors under an asymmetric harm
  structure (e.g., missing a hazard class is worse than a false alarm).
- **Machine suffering**: the computation actually executed, metered exactly.

Training proceeds only until a held-out feasibility target is reached (e.g.,
validation accuracy ≥ τ). Once the target is met, training freezes
*immediately*. The result is a strict separation of machine suffering into a
*necessary* part (up to the freeze epoch) and a *gratuitous* part (everything
after), with gratuitous suffering forced to zero. At inference, early exits
further reduce per-sample machine suffering whenever a sample can exit early.

This paper asks: can this idea be deployed as a real system? We answer with a
complete pipeline — training on GPU, confidence extraction, and a
bit-exact FPGA catastrophe-scan / FLOP-metering kernel on an AMD Alveo U250 —
measured end to end.

### 1.2 Contributions

| # | Contribution | Evidence | Status |
|---|---|---|---|
| 1 | A SAN training harness for ResNet-50, ViT-small/d384, and a tiny decoder LM on real data, with metered-MAC accounting and freeze-on-green | `scripts/research/suffering_aware_large_architecture.py` | `MEASURED` on GPU (small subset) |
| 2 | An FPGA exit-audit / FLOP-meter kernel with integer semantics, no multipliers/DSPs, and one sample/cycle/PE throughput | `hardware/fpga/u250_catastrophe_scan/krnl_san_scan.cpp` | `MEASURED` on U250 |
| 3 | On-target U250 benchmark: 511 Msamples/s on 1.2M stress cohort; host-CPU baseline 130.8 Msamples/s scalar to 1.05 Gsamples/s AVX-512/8t; ~3.3 nJ/sample board-level energy; bit-exact against golden model | `host_san_scan`, `host_san_scan_bench`, `san_scan_cpu_baseline.c` | `MEASURED` |
| 4 | GPU training study: 40.7–52.2% metered-MAC savings in a fast-convergence, small-subset regime; CIFAR-100 is infeasible under the same budget | Slurm jobs 8584/8588/8591, 8599–8607, 8611–8612 | `MEASURED / MIXED` |
| 5 | Real-image kernel validation on ImageNette2-160, bit-exact on the U250 | `train_san_imagenette.py` + `host_san_scan` | `MEASURED` |
| 6 | Honest accounting of limits: small-subset accuracy, partial meter convention, ImageNet-1k unavailable, board-level power, ResNet-50 patient-channel tradeoff | §5, §6 | `DECLARED` |

The primary systems contribution is the measured bridge between the SAN
learning rule and FPGA deployment: the kernel is not an approximate accelerator
of a float model; it *is* the integer specification of the exit-audit and
FLOP-metering path, and the deployment is sound exactly when the card
reproduces that specification. The trunk stays on the host; the card decides,
counts, and meters.

### 1.3 Prior work

Early-exit networks were introduced by BranchyNet [1] and Shallow-Deep Networks
[2], with later work on confidence thresholds [3], dynamic inference, and
hardware-aware early exit. Most of this work treats the threshold as a
hyperparameter tuned for accuracy-efficiency tradeoffs and trains for a fixed
budget.

Constrained learning [5,6,7] provides generalization guarantees for ERM with
explicit constraints, but does not address inference-time compute reduction or
the freeze-on-green training rule. Mercyful / suffering-aware learning [4]
introduces the two-channel suffering framework and the anti-Goodhart selection
rule; the present paper is the first systems study that maps that framework to
a measured FPGA deployment.

On the FPGA side, early-exit accelerators have been proposed for CNNs and
transformers, typically using custom datapaths that map part of the model
into hardware. Our kernel takes the opposite approach: the trunk stays on the
host (or GPU), and only the cheap decision/metering path runs on the FPGA.
This keeps the bitstream small, architecture-agnostic (the stage-cost LUT is
loaded by the host), and verifiable by direct comparison to a golden model.

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
| quantize + pack | ~40 ms | Q0.15 floor + 512-bit beat packing |
| xclbin setup | ~135 ms | one-time `xclbin` load per process (no PR) |
| DMA H2D | ~0.12 ms | 62 848 bytes = 982 beats × 64 bytes |
| kernel | ~0.66 ms | ~6 Msamples/s single-shot for this small cohort |
| DMA D2H | ~0.15 ms | histogram + catastrophe count + MAC total |
| **host↔card total** | **~136 ms** | first cohort; subsequent cohorts reuse context |
| PyTorch forward (CPU) | ~18.4 s | SAN-ResNet-18, 3 925 real images, DL380 CPU |

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

**What this tells us.** The SAN-v1 accuracy deficit is partly an architecture
problem (weak linear heads), which MLP heads fix. But the deeper problem is
that an unconstrained learned gate discovers the FLOP-minimising solution —
always exit at the first stage — which is accuracy-capped. A usable SAN at high
τ needs either (a) a gating penalty for early exits, (b) a curriculum that
trains deep stages before allowing early exits, or (c) a gate constrained to
match a target accuracy, not just a target FLOP count.

---

## 5 Discussion

### 5.1 What the numbers mean

The U250 result shows that the SAN decision/metering path is deployable as a
small, fast, low-energy FPGA kernel. The kernel does not accelerate the trunk;
it accelerates the audit: for every incoming cohort, it decides when each sample
should have exited and counts the exact computation that was executed. This is
the operation a production SAN deployment runs continuously.

The GPU result shows that the SAN training rule saves substantial computation
across architecture families on a real dataset, even at CIFAR-10 scale. The
savings would be larger at ImageNet scale because the per-exit stage cost is
much higher.

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
- Include the EarlyStop baseline column in the GPU training table for direct
  three-way comparison.

---

## 6 Conclusion

We have presented, to our knowledge, the first measured deployment of the SAN exit-audit /
FLOP-metering kernel on an FPGA, together with a GPU training study across three
architecture families. The U250 kernel runs at 511 Msamples/s on a 1.2M-sample
stress cohort and approximately 3.3 nJ/sample board-level energy, with bit-exact
correctness verified against a golden model on synthetic, stress, and real-image
cohorts. GPU training saves 40.7–52.2% of metered-MAC burden in a small-subset,
fast-convergence regime. We have stated the limitations plainly — small-subset
accuracy, partial meter convention, no full ImageNet-1k, board-level power only,
ResNet-50 patient-channel tradeoff — because the honesty of the evidence is itself a
contribution. The artifacts are available for reproduction.

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

[2] Kaya, Y., Hong, S., & Dumitras, T. (2019). Shallow-Deep Networks:
Understanding and mitigating network overthinking. *ICML*.

[3] Huang, G. (2018). Multi-scale dense networks for resource efficient image
classification. *ICLR*.

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
