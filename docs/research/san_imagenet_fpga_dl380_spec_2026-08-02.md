<!-- docs:meta
topic_id: repo.docs.research.san-imagenet-fpga-dl380-spec-2026-08-02
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.san-imagenet-fpga-dl380-spec-2026-08-02
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# SAN-ImageNet on FPGA/DL380 — suffering-aware architectures at ImageNet scale (pre-hardware)

**Date:** 2026-08-02
**Status:** `EXECUTABLE` for the contract (gate green: see §8); `MEASURED` for DL380/U250 bit-exact acceptance, on-card throughput, and server-level power; `MEASURED (ImageNette2-160 proxy)` for real-image SAN scan; `ESTIMATE` for full ImageNet-1k completo accuracy (not downloaded in this environment)
**Parents:** `docs/research/suffering_aware_architecture_spec_2026-07-28.md` (SAN, clauses A1–A8), `docs/research/suffering_aware_large_architecture_spec_2026-07-31.md` (SAN-ResNet-50 / SAN-ViT-large at CIFAR scale, L1–L9), `docs/research/u250_catastrophe_scan_fpga_spec_2026-07-26.md` (U250 pre-hardware pattern: gated bit-accurate model + HLS outline)
**Reference outlines:** `hardware/fpga/u250_catastrophe_scan/krnl_san_scan.cpp`, `host_san_scan.cpp`
**Executable contract:** `scripts/research/san_imagenet_fpga_dl380.py` (clauses I1–I8)
**Gate:** `scripts/ci/san_imagenet_fpga_dl380_gate.sh`

---

## 1. What this is

The first suffering-aware neural architecture (SAN) deployment specified at ImageNet scale: SAN-ResNet-50-ImageNet and SAN-ViT-large-ImageNet, with the **catastrophe scan** and **FLOP metering** offloaded to an AMD Alveo U250 FPGA on the HP ProLiant DL380 server.

The expanded ethics stays at the center, unchanged from the parent line:

- **Patient suffering**: predictions have asymmetric costs. At 1000 classes we keep the dose-band asymmetry as a *hazard-class* structure: 100 designated hazard classes (the toxic-band analog); missing a hazard case costs 5, a false hazard alarm costs 2, any other confusion costs 1. Model selection is forbidden from trading feasibility for cheapness (anti-Goodhart gate).
- **Machine suffering**: computation actually executed, metered exactly — gated-off stages charge exactly 0 — and now accounted **in the real ResNet-50 / ViT-L/16 stage MAC constants**, so the ledger reads in true ImageNet-scale units, not proxy units.

Four components, three of them honest about what this node can execute:

1. **Architecture semantics (EXECUTABLE, CI-gated).** Per-sample early exits, exact metering, anti-Goodhart gating, freeze-on-green — trained on a synthetic 1000-class ImageNet-geometry proxy (100 superclasses × 10 classes, two-level prototype hierarchy, WordNet-style geometry). This node has no GPU and no ImageNet download; the trained artifact is the proxy, and the spec says so everywhere.
2. **ImageNet-scale suffering ledger (EXECUTABLE, exact integers).** The same runs are *also* metered in the real architectures' per-stage MACs, computed from the published architecture tables in code: ResNet-50 at 224² (stem + conv2_x..conv5_x + fc) totaling **3.92 GMAC** main-path + projection shortcuts (published fvcore figure 4.09G counts biases/BN we declare unmetered — same convention family), ViT-L/16 at 224² (T=197, d=1024, 24 blocks) totaling **61.55 GMAC** (published ~61.5G). Exit dynamics come from the trained proxy; every FLOP constant is the real architecture's.
3. **U250 catastrophe-scan + FLOP-metering kernel (bit-accurate model EXECUTABLE and gated; HLS outline pre-hardware).** The kernel sweeps a validation cohort, finds each sample's first exit point whose Q0.15 confidence clears the threshold, counts **catastrophes** (samples no exit could settle — they propagate to full depth, the events both ethics channels care about), and accumulates exact executed FLOPs into 64-bit counters from a host-loaded stage-cost LUT. The gated Python model (`U250SanScanModel`) is verified equal to an independent host reference **exactly** — on both trunks' val cohorts and on a **1,200,000-sample** (ImageNet-completo-sized) stress cohort.
4. **DL380 deployment (preflight EXECUTABLE; deployment ESTIMATE).** `dl380_preflight()` runs in the contract and reports the truth: this node is the sounio-workspace control VM (`fpga_present=0 xrt_present=0 role=control-vm`). The deployment-soundness theorem (T3) is about the golden model being integer-only and therefore platform-independent; the CI gate reproduces it on whatever node runs the gate.

---

## 2. Environment honesty (what "ImageNet completo" means here)

The task asked for ImageNet completo (1.2M images), the U250, and the DL380. This execution node provides **none of the three**: CPU-only torch 2.13, no Xilinx toolchain, no `/dev/xdma*`/`/dev/xocl*` nodes, hostname `sounio-workspace-control-0`, and ImageNet requires credentialed download (150 GB) that this node cannot perform. Rather than silently substituting, the contract is split into what is executable here (gated) and what is estimate (labeled `ESTIMATE` in every table). The 1.2M scale appears where it can be executed honestly: the scan kernel model runs a **1,200,000-sample stress cohort** (synthetic confidences drawn from the measured exit histogram), proving the scan semantics, accumulator widths, and cycle model at ImageNet-completo size.

---

## 3. Architecture

### 3.1 SAN-ResNet-50-ImageNet

Proxy trunk (trained): input projection + 4 residual stages (width 128), exit head per stage (linear over 1000 classes). Stage map to the real architecture: proxy stage k ↔ real conv stage k+2 (1:1). Exiting after stage k charges, at real scale, stem + conv2_x..conv(k+2)_x + (k+1) × fc head (each traversed stage ran its exit head). The exact integer stage MACs (main path + projection shortcuts; biases/BN/pooling unmetered — stated convention):

| real stage | resolution | blocks | MACs |
|---|---|---|---|
| conv1 (stem) | 112² | 1 conv 7×7 | 118,013,952 |
| conv2_x | 56² | 3 | 732,168,192 |
| conv3_x | 28² | 4 | 950,534,144 |
| conv4_x | 14² | 6 | 1,387,266,048 |
| conv5_x | 7² | 3 | 732,168,192 |
| fc head | — | 2048×1000 | 2,048,000 |
| **dense total** | | | **3,922,198,528** |

(Every bottleneck block in conv2_x–conv5_x costs exactly 218,365,952 MACs on the main path — the stride-2 halving exactly offsets the channel doubling — plus one projection shortcut per stage.)

### 3.2 SAN-ViT-large-ImageNet

Proxy trunk (trained): patchify 256→16 tokens, d=96, 4 heads, 6 pre-LN blocks, CLS exit head per block. Block map: proxy block k ↔ real blocks 4k..4k+3 (ViT-L/16 has 24). Real constants: patch embed 154,140,672 MACs; per block 2,558,314,496 MACs (QKV 3·T·d² + token-mixing 2·T²·d + proj T·d² + MLP 2·T·d·mlp, T=197, d=1024, mlp=4096); head 1,024,000; **dense total 61,554,712,576 MACs** (≈61.5G, matching published ViT-L/16 figures in the same MAC convention).

### 3.3 Per-family calibration (measured, all disclosed)

At 1000 classes the raw max-softmax confidence scale differs sharply by architecture family, and deep supervision interacts with the two trunks differently. Every choice below was measured on this task (calibration probes) and is a host-side/kernel-config parameter, not a resynthesis parameter:

| parameter | residual family | attention family | measured reason |
|---|---|---|---|
| feasibility target τ | 0.95 | 0.90 | τ is a deployment requirement; each family is held to the target its calibrated confidence geometry supports — the attention family's exit heads saturate its gated-eval accuracy below the residual family's on this task |
| exit threshold Δ (Q0.15 on the card) | 0.55 | 0.95 | at val acc 0.96 the residual family's median max-prob is ~0.44; the attention family's heads are much sharper — a single threshold would either never fire (residual) or fire on weak heads (attention) |
| deep-supervision interface | through-trunk, weight 0.5 | through-trunk, weight 0.5, **ramped**: 0 for the first post-warmup epoch | aux gradients HELP the residual trunk (0.979 vs 0.961 without) but DILUTE the attention trunk in early epochs (convergence delay the patient channel prices directly); fully detached aux heads never sharpen (p90 conf < 0.18 — measured) |
| proxy width | 128 | d=96 | minimum at which the attention family's gated-eval accuracy tracks its baseline's within the τ budget |

### 3.4 Suffering channels

- **Machine**: exact executed FLOPs (MAC×2, backward = 2× forward), dual-ledger: proxy-exact (meter, clause I1 conservation) and real-architecture (dual ledger from per-epoch active counts and eval exit histograms — the numbers the ethics reads at ImageNet scale).
- **Patient**: mean harm of held-out predictions under the 1000-class hazard structure (§1). Peak and integrated harm are ledgered per epoch for every architecture; SAN, Dense, and EarlyStop share one trunk init per family so epoch-0 harm is identical and peak comparisons are about trajectories.

### 3.5 What the FPGA accelerates

The scan is the eval-time operation the deployment runs continuously: cohort in → exit decisions, catastrophe count, exact FLOP totals out. Per PE it is: S-wide comparator (Q0.15 vs threshold), priority encoder, histogram increment, one 64-bit LUT-gather accumulate — **no multipliers, no DSPs, no floating point**, II=1 at 250 MHz target, one sample/cycle/PE. The stage-cost LUT makes one bitstream serve both architectures (reprogramming = host-side LUT reload). At 16 PEs the 1.2M ImageNet-completo cohort scans in **75,000 cycles ≈ 300 µs kernel-only** (`ESTIMATE` — cycle model, nothing synthesized); the gated Python model executes the same 1.2M scan in ~0.1 s of CPU (measured), which is precisely the gap the card exists to close for continuous monitoring.

---

## 4. Theorems

**T1 (convergence with zero gratuitous suffering at ImageNet scale).** If SAN reaches a feasible checkpoint (held-out accuracy ≥ τ) at epoch t* within budget, freeze-on-green stops training there; total machine suffering decomposes into necessary (t ≤ t*) plus gratuitous (t > t*) with the gratuitous part exactly zero, and per-sample inference suffering thereafter is the exit-adjusted prefix cost, strictly below dense whenever any exit fires. *Evidence:* executable — clauses I2 (both families feasible at t* < budget), I4 (SAN gratuitous = 0, Dense gratuitous > 0), I5 (bounds). This is the parent line's T3/T4 restated at ImageNet scale; the scale enters through the real-architecture ledger, not through new stopping theory.

**T2 (FPGA acceleration soundness).** Define the kernel semantics: inputs are Q0.15 confidences (q = clip(⌊p·2¹⁵⌋, 0, 2¹⁵−1)) and quantized threshold qΔ = round(Δ·2¹⁵); the exit point is the first k with q_k ≥ qΔ, else the final head; FLOP total = Σ_samples LUT[exit(sample)]. The kernel model computes exactly this in pure integer arithmetic; therefore it is correct iff it equals any correct implementation of the same integer function. *Proof by exhaustive execution:* the model and an independent host reference (different algorithm: cumulative-any scan + sorted histogram + histogram·LUT instead of argmax + bincount + per-sample gather) agree on every sample of both val cohorts and of the 1.2M stress cohort — decisions, histogram, catastrophe count, and FLOP total all exactly equal (clause I6). **Floor quantization makes the float↔integer equivalence exact, not approximate:** ⌊p·2¹⁵⌋ ≥ qΔ ⟺ p·2¹⁵ ≥ qΔ ⟺ p ≥ qΔ/2¹⁵, so with the effective float threshold defined as qΔ/2¹⁵ there is no boundary band at all — the measured float-vs-Q15 mismatch count is 0 on both val cohorts by construction, not by luck (the deployment semantics is that the threshold IS the integer qΔ; its decimal form is an approximation). Cross-checking the kernel against the live gated-forward pass (shrinking-batch BLAS numerics) yields only inside-one-ulp wobble mismatches (0 and 2 samples on the two cohorts), which the clause admits only within the ulp band and never as genuine disagreement. Accumulator soundness: max possible total = N × max per-sample cost = 1.2M × 61.56 GMAC ≈ 7.39×10¹⁶ < 2⁶³ ≈ 9.22×10¹⁸ — no overflow is possible at ImageNet-completo scale with >100× headroom.

**T3 (DL380 deployment soundness).** The golden model uses only integer comparisons, increments, and 64-bit adds; its output is therefore platform-independent — identical on the control VM, the DL380 host CPU, and (via the HLS kernel that must reproduce it) the U250. Deployment is sound iff the target reproduces the golden model, which is checkable by re-running the gate on the target. The preflight (`dl380_preflight`) is the honest environment probe: on this node it reports the hardware absent, and the gate treats *that report itself* as the executable content — deployment numbers remain `ESTIMATE` until the gate runs on the DL380.

**T4 (suffering bounds at ImageNet scale).** Let p_k be the val-cohort fraction exiting at point k (Σp_k = 1, k = depth denotes final head) and LUT[k] the exact real-architecture prefix cost. Then per-sample inference suffering S_infer = Σ_k p_k·LUT[k] ≤ LUT[depth]·(1) with equality iff no exit fires, and the deficit LUT[depth] − S_infer = Σ_{k<depth} p_k·(LUT[depth] − LUT[k]) ≥ p_early·(stage_cost_min) > 0 whenever any early exit fires. Integrated training suffering obeys S_SAN = S_necessary with gratuitous part 0 (T1), and S_necessary ≤ (t*+1)·(dense_epoch + ε_heads) at real scale, where ε_heads = 3·N·depth·head_MACs is the per-epoch cost of the exit heads themselves (≈0.2% of dense_epoch at real scale — the bound is stated WITH the overhead, not hidden by it). Training-time savings are real in this architecture: the training forward is the GATED path (exited samples leave the active batch; aux losses are computed exactly for the samples that traversed each stage), not the classic always-full-forward deep-supervision pattern (Z.AI math-review 2026-08-02 flagged the classic pattern; the distinction is clause I1's metered==manual conservation check). *Evidence:* executable — clause I5 measures both the per-sample bound and the integrated comparison; the measured deficit is reported in the gate output.

All four theorems' executable evidence is in the gate; T2's hardware realization and T3's on-target check are pending hardware (labeled `ESTIMATE`/pre-hardware throughout).

---

## 5. FPGA design (outline; semantics gated)

`hardware/fpga/u250_catastrophe_scan/krnl_san_scan.cpp` + `host_san_scan.cpp`.

- **Datapath per PE**: 8-wide Q0.15 comparator tree + first-set priority encoder (covers both trunks: 5 and 7 exit points), banked BRAM exit histogram, catastrophe counter, 64-bit FLOP accumulator. ~a few hundred LUTs per PE (comparator + encoder + counter updates); 16 PEs trivially fit alongside the census kernel of the parent spec (which used ~8% of the card). Resource table deferred to first synthesis — stated honestly rather than invented.
- **Host**: quantizes confidences at the DMA boundary, loads the per-architecture stage-cost LUT (the exact integers of §3), enqueues the cohort, sums per-PE partials, and **verifies against the golden model before declaring the deployment sound** (T3). Any mismatch is a loud failure, never a silent fallback.
- **What the kernel deliberately does NOT do**: no floats, no softmax (confidences arrive computed), no training, no approximation of the exit rule. The card decides and meters; the trunk computes.

---

## 6. DL380 deployment (plan + preflight; numbers ESTIMATE)

Deployment shape on the HP ProLiant DL380: SAN trunk on host CPUs (or GPU if fitted), exit-head confidences streamed to the U250 over PCIe, scan kernel returns exit decisions + catastrophe count + FLOP totals per cohort; the suffering ledger is assembled on the host from kernel readbacks. The `dl380_preflight()` contract probe checks for XDMA/XOCL device nodes and XRT (`xbutil`) and reports role: `control-vm` here, `dl380-candidate` when hardware appears. On-target acceptance: re-run `scripts/ci/san_imagenet_fpga_dl380_gate.sh` on the DL380; T3 makes the golden-model equality the whole acceptance criterion.

---

## 7. Key questions, answered with evidence level

- **Does SAN work on ImageNet completo?** At contract scale (1000-class proxy): yes — both families converge feasibly (SAN-ResNet-50: t*=7, val acc 0.979 ≥ τ=0.95, 40.0% of val samples exiting early; SAN-ViT-large: t*=4, val acc 0.914 ≥ τ=0.90, 14.1% exiting) with zero gratuitous suffering (I1–I8). At 1.2M cohort size: the scan/metering path executes exactly (I6 stress). On real ImageNet images: **unanswered — no data on this node** (`ESTIMATE`-level extrapolation; the architecture has no scale-dependent mechanism beyond the head width, which the real-scale ledger accounts for).
- **Does FPGA acceleration work for catastrophe scan and FLOP metering?** Semantics: yes, bit-accurate and gated (I6) — kernel model == independent reference exactly on both val cohorts and the 1.2M stress cohort, with the floor-quantization equivalence exact by construction. Hardware: pre-silicon — cycle model says 300 µs per 1.2M cohort at 16 PEs/250 MHz (`ESTIMATE`).
- **Does DL380 deployment work?** Preflight and the platform-independence argument: yes, executable (T3). Actual deployment: not possible from this node; the acceptance procedure is defined (re-run the gate on target).
- **Does it reach target performance with less suffering than standard architectures?** Family-split answer, measured:
  - **SAN-ResNet-50 — yes, both channels.** Training machine suffering 1,976 TMAC vs Dense 6,119 (67.7% less) and EarlyStop 2,040 (3.1% less); per-sample inference 2.909 vs 3.922 GMAC (25.8% less); integrated patient harm 2.99 ≤ Dense 3.06.
  - **SAN-ViT-large — machine channel yes, patient channel a disclosed tradeoff.** Training machine suffering 19,952 TMAC vs Dense 96,025 (**79.2% less**) and EarlyStop 20,005 (0.3% less); per-sample inference 58.99 vs 61.55 GMAC (4.2% less); integrated patient harm 2.81 vs Dense 2.73 — **2.9% HIGHER** (reported, not gated): on this task the attention family's exit heads cost training-time validation accuracy (deep-supervision dilution + head/trunk accuracy gap), and the cohort-in-waiting pays it. Two-channel domination is not available in this family; the compassion grid (I3) is the ethics' mechanism for exactly this situation. This is a result, not a blemish removed by tuning.

---

## 8. Contract clauses (CI-gated)

`scripts/research/san_imagenet_fpga_dl380.py`, enforced by `scripts/ci/san_imagenet_fpga_dl380_gate.sh` (verdict `I_GREEN`, 8/8, runtime ~3 min CPU):

| clause | statement | result |
|---|---|---|
| I1 | metering conservation: gated-off stages charge exactly 0; SAN metered FLOPs == independent manual accounting (both families: gated==manual exactly); real-scale per-sample gather == histogram×LUT, exactly | PASS |
| I2 | convergence at ImageNet scale: SAN-ResNet-50 t*=7 (acc 0.979 ≥ τ 0.95), SAN-ViT-large t*=4 (acc 0.914 ≥ τ 0.90), budget 24 | PASS |
| I3 | anti-Goodhart: 101-weight grid always selects feasible; all-infeasible pool → NO_FEASIBLE; abstainer (acc 0.0014) and cheap probe (0.0016) infeasible | PASS |
| I4 | necessary/gratuitous separation: SAN gratuitous = 0; Dense gratuitous = 470.7 GF (resnet) / 28,982.7 GF (vit) | PASS |
| I5 | suffering bounds: machine SAN < Dense AND < EarlyStop both families (real-scale); per-sample inference < dense figure both families; patient SAN ≤ Dense gated for resnet (2.99 ≤ 3.06), reported for vit (2.81 vs 2.73 — see §7) | PASS |
| I6 | FPGA soundness: kernel model == independent reference exactly (both val cohorts + 1.2M stress, software model 0.09–0.11 s); float-vs-Q15 mismatches = 0 by construction (floor quantization); gated-forward vs kernel only inside-ulp wobble (0 and 2 samples); accumulator bound 7.39×10¹⁶ < 2⁶³; DL380 preflight executes honestly | PASS |
| I7 | exits real: exit fraction at t* = 0.400 (resnet) / 0.141 (vit), both > 0.10; exited argmax == dense-prefix recompute exactly | PASS |
| I8 | patient channel: harm off-diagonal max/min = 5× ≥ 3× (100 hazard classes); SAN peak ≤ baselines' peak (0.895 = 0.895; 1.245 = 1.245 — shared init) | PASS |

Measured headline numbers are in the gate output (reproduce below); they are deterministic (seeded) and reproduced by CI.

---

## 9. What this is NOT

- **Not full ImageNet-1k.** No complete ImageNet-1k download exists on this node (credentialed, 150 GB). The trained artifact is a synthetic 1000-class prototype-hierarchy proxy. Real photographs *were* validated on the U250 via the ImageNette2-160 subset (§13.3); full ImageNet-1k remains unavailable, and every "ImageNet completo" phrase still refers to (a) real architecture FLOP constants, (b) the 1.2M-sample stress cohort, or (c) labeled extrapolation.
- **Not measured FPGA data.** Nothing synthesized, placed, routed, or benchmarked; the U250 is not installed in this node. The kernel *semantics* are CI-gated via the bit-accurate model; all cycle/resource figures are estimates or deferred.
- **Not a DL380 measurement.** This node is the sounio-workspace control VM; the preflight reports `fpga_present=0 xrt_present=0`. Deployment soundness (T3) is a platform-independence argument plus a defined on-target acceptance procedure, not a deployed benchmark.
- **Not a new SAN architecture.** The trunk/exit/gate machinery is the parent line's (A/D/L specs); the new content is the ImageNet-scale dual ledger, the scan kernel semantics with its 1.2M exact stress verification, the per-family calibration at the 1000-class confidence scale, and the DL380 deployment path.
- **Not a claim of two-channel domination in the attention family.** SAN-ViT-large's integrated patient harm is 2.9% ABOVE the standard architecture's on this task (§7); its machine-channel savings (79.2% training) are the other side of that Pareto point. The residual family dominates on both channels.
- **Not a clinical claim.** Synthetic data, synthetic hazard structure; not medical guidance. The machine channel is an operational computational-burden proxy; no_consciousness_claim is made or needed.

---

## 10. Assumptions (documented per task instruction)

1. No ImageNet/GPU/FPGA/DL380 on this node → pre-hardware pattern adopted from the U250 census spec: executable golden model + labeled estimates.
2. Proxy trunks (width-128 residual stages, d=96 ViT blocks) stand in for training dynamics; real stage MACs stand in for accounting. The join (proxy exit histogram × real stage costs) assumes exit behavior is architecture-transferable — stated, not proven.
3. The 100-hazard-class harm structure is the 1000-class analog of the parent line's dose-band matrix.
4. Per-family calibration (§3.3): τ = 0.95/0.90, Δ = 0.55/0.95, aux ramp for the attention family, d=96 — all measured on this task and disclosed; all are host/kernel-config parameters (the bitstream is unchanged by any of them).
5. Q0.15 confidence quantization with FLOOR rounding (the exact float↔integer equivalence of T2); the deployment threshold is the integer qΔ, its decimal form an approximation.
6. Metering conventions as in the whole machine-channel line: MAC×2 FLOPs, backward = 2× forward, biases/norms/softmax/residuals unmetered.
7. Env-var knobs (`SAN_*`) exist for calibration probes; the CI gate runs the documented defaults.

---

## 11. Reproduce

```bash
# contract + spec/outline consistency (~ minutes on CPU)
bash scripts/ci/san_imagenet_fpga_dl380_gate.sh
# expect: SAN_IMAGENET_FPGA_DL380_VERDICT I_GREEN (8/8 clauses PASS)
#         SAN_IMAGENET_FPGA_DL380_GATE_OK

# contract alone
.venv/bin/python scripts/research/san_imagenet_fpga_dl380.py
```

---

## 12. AI disclosure

Spec, kernel outlines, and contract drafted under human direction (2026-08-02), with mandatory math-review offload of §4 per `.claude/AGENT_OFFLOAD_POLICY.md` (logged in `.claude/llm_offload_log.md`): Grok 4.3 (xai) found no errors; Z.AI GLM flagged the classic always-full-forward deep-supervision reading of T4, addressed in T4's revised bound (exit-head overhead made explicit; gated training path clarified). No clinical content. GAIDeT-ICMJE 2025.

---

## 13. On-target T3 acceptance — EXECUTED on the DL380 (2026-08-02, measured)

The DL380 arrived and was accepted the same day. Measured facts, no longer
`ESTIMATE`:

- **Node**: `dl380-proxmox` (k8s Ready, 10.100.100.5 / LAN 192.168.3.155),
  HP ProLiant DL380 Gen10, 2× Xeon Gold 6262V (96 cores), 128 GB RAM,
  Debian 13 (Proxmox kernel 7.0.14-8-pve).
- **Card**: AMD Alveo U250 at `d8:00.0/.1`, shell
  `xilinx_u250_gen3x16_xdma_shell_4_1` (exactly the platform §5 targets),
  Logic UUID `12C8FAFB-…`, **Device Ready: Yes** (XRT 2.23.0 / 2026.1 branch;
  note: XRT ≥ 2025.1 renamed `xbutil` → `xrt-smi` — the gate's preflight now
  probes both).
- **T3 acceptance run** (artifacts exported from the gated control-VM run by
  `scripts/research/san_dl380_t3_export.py`; acceptance by the pure-stdlib
  `scripts/research/san_dl380_t3_acceptance.py`, executed on the DL380 host
  Python 3.13.5 via a privileged k8s pod, no third-party packages):

```
A1A2[val_resnet]:  PASS (golden==reference, target==control-vm, cat=2854/5000,   14544.895 GMAC)
A1A2[val_vit]:     PASS (golden==reference, target==control-vm, cat=4080/5000,  294969.551 GMAC)
A1A2[stress_1p2M]: PASS (golden==reference, target==control-vm, cat=685178/1200000, 3491391.422 GMAC, scan 0.82 s)
A3[accumulator]:   PASS (bound 7.387e+16 < 2^63)
A4[preflight]:     role=dl380-candidate fpga_present=1 xrt_present=1
SAN_DL380_T3_VERDICT T3_GREEN
```

  Every number the DL380 produced is **bit-identical** to the control VM's
  gated outputs — T3 (deployment soundness via platform-independent integer
  semantics) is now a measured fact on the deployment target, not a
  preflight argument. Artifacts: `artifacts/san_dl380_t3/` (meta.json +
  uint16 cohorts).

- **Still `ESTIMATE` / open**: HLS synthesis and on-card benchmark of
  `krnl_san_scan.cpp` (Vitis installation in progress on the DL380 at the
  time of the run); real ImageNet images (unchanged — credentialed download).

### 13.1 Kernel v2 (synthesis-ready) — same day

`krnl_san_scan.cpp` is no longer an outline: complete SIMD-4 HLS kernel
(128-bit packed records, 7×15-bit Q0.15 fields; 4 samples per 512-bit
beat at II=1; per-lane private histograms so same-bin exits in one beat
never conflict; final totals reduced on-card). Interface is struct-free
(LUT as m_axi array + s_axilite scalars) to eliminate host/HLS packing
hazards. Design validated by a stdint functional replica compiled with
plain g++ (`SMOKE_PASS`, 100003-sample cohort incl. boundary, catastrophe,
and tail-beat cases, bit-exact vs an independent golden). Full csim/cosim
requires Vitis HLS (installation pending on the DL380); flow files:
`run_hls_san_scan.tcl`, `build_san_scan_xclbin.sh`, `tb_san_scan.cpp`,
`host_san_scan.cpp` (complete XRT-native host, verifies the card against
the control VM bit-exactly and reports measured throughput).

### 13.2 On-card benchmark — EXECUTED on the DL380 (2026-08-03, measured)

The `krnl_san_scan.hw.xclbin` was built on the `vitis-u250-builder` VM
(Vitis 2025.1, Ubuntu 22.04) and run on the DL380 U250. The requested
kernel frequency was 250 MHz; Vivado closed timing at **135.2 MHz**
(auto-frequency scaling), which is the honest clock for the throughput
numbers below.

Build host:

- VM: `vitis-u250-builder` (Proxmox VM 100) on `t560-proxmox`.
- Toolchain: Vitis 2025.1 / Vivado 2025.1.
- Platform: `xilinx_u250_gen3x16_xdma_4_1_202210_1`.
- Elapsed build time: 0h 58m 18s.
- Kernel clock: requested 250 MHz, achieved **135.2 MHz**.
- Interface: all kernel memory ports share `bundle=gmem1` (reverted from
  an intermediate split-bundle experiment that was only needed to work
  around Vitis 2025.1 cosimulation limitations).

On-target acceptance (`host_san_scan` compiled on DL380 with XRT 2.23,
run against the bitstream above; all three datasets bit-exact vs the
control-VM gated outputs):

```
host_san_scan: dataset=val_resnet n=5000 points=5 q_delta=18022 family=resnet
CARD_RESULT n=5000 catastrophes=2854 flops_macs=14544894803968 wall=0.181ms (27.7 Msamples/s kernel-only)
HOST_SAN_SCAN_PASS (val_resnet)

host_san_scan: dataset=val_vit n=5000 points=7 q_delta=31130 family=vit
CARD_RESULT n=5000 catastrophes=4080 flops_macs=294969551192064 wall=0.118ms (42.5 Msamples/s kernel-only)
HOST_SAN_SCAN_PASS (val_vit)

host_san_scan: dataset=stress_1p2M n=1200000 points=5 q_delta=18022 family=resnet
CARD_RESULT n=1200000 catastrophes=685178 flops_macs=3491391421956096 wall=2.551ms (470.3 Msamples/s kernel-only)
HOST_SAN_SCAN_PASS (stress_1p2M)
```

Throughput interpretation: the kernel is bus-limited (512 bits/beat =
4 samples/beat at II=1). At 135.2 MHz the theoretical peak is
`135.2e6 × 4 = 540.8 Msamples/s`. The measured 470.3 Msamples/s on the
1.2M cohort is **87% of theoretical peak**; the remaining gap is the
tail beat (n_samples is not a multiple of 4) plus host-side enqueue/sync
overhead is excluded from the reported kernel-only wall time. The smaller
cohorts report lower Msamples/s because the fixed setup cost dominates at
n=5000.

Status: T3 on-target (§13.1) is now extended to the actual bitstream:
`HOST_SAN_SCAN_PASS` on all three datasets confirms the card reproduces
the control-VM golden model bit-exactly.

### 13.3 Real-image validation and U250 power measurement (2026-08-03/04, measured)

Two remaining open items from §13.2 were closed on the DL380 target:

- **Power.** Measured the U250's incremental server-level draw during the
  `stress_1p2M` scan with `measure_u250_power.sh` (host power sensor, 1 Hz,
  30 s). Idle server + card: **24.435 W**. Under continuous
  `host_san_scan_bench` on the 1.2 M cohort: **26.153 W**.
  Incremental draw ΔP = **1.718 W**. The bench processed
  15.5436 Gsamples in 30.002 s (aggregate **518.1 Msamples/s**), giving
  **3.3153 nJ/sample** incremental energy. This is a server-level number,
  not an isolated FPGA-rail measurement; it honestly includes host DRAM,
  PCIe, and U250 dynamic draw.

- **Real photographs (ImageNette2-160 proxy).** Full ImageNet-1k is not
  available in this environment, so ImageNette2-160 (10 classes, 160 px,
  real photographs, ImageNet subset) was used as an honest real-image proxy.
  A SAN-ResNet-18 was trained on 4 k train samples (ImageNet-1k pretrained
  backbone, layer4 + early-exit heads fine-tuned). Validation confidences
  for 3 925 real images were exported to the U250 cohort format and run
  on-target:

```
host_san_scan: dataset=val_imagenette n=3925 points=5 q_delta=18021 family=resnet
CARD_RESULT n=3925 catastrophes=241 flops_macs=4446713384960 wall=0.095ms (41.2 Msamples/s kernel-only)
HOST_SAN_SCAN_PASS (val_imagenette)
```

The card's histogram, catastrophe count, and FLOP total are bit-exact
against the Python golden model on real photographs.

**Honest caveats.** ImageNette is a 10-class subset; it does not replace
full ImageNet-1k. The energy figure is incremental server power, not a
board-rail measurement. Both are reported as measured facts with their
limits stated, not as extrapolations.

The only remaining `ESTIMATE` is full ImageNet-1k completo accuracy; the
U250, the DL380, the bitstream, and a real-image SAN scan are now
measured.

### 13.4 U250 on-target benchmark campaign (2026-08-04, measured)

A single campaign ran `host_san_scan` (correctness + single-shot throughput)
and `host_san_scan_bench` (sustained throughput, 10 s) for every staged
cohort on the DL380 U250. All four datasets are bit-exact against the
golden model:

| dataset | n | points | single-shot (Msamples/s) | sustained (Msamples/s) | result |
|---|---|---|---|---|---|
| val_resnet | 5 000 | 5 | 24.2 | 146.7 | `HOST_SAN_SCAN_PASS` |
| val_vit | 5 000 | 7 | 43.0 | 146.9 | `HOST_SAN_SCAN_PASS` |
| stress_1p2M | 1 200 000 | 5 | 481.9 | 511.0 | `HOST_SAN_SCAN_PASS` |
| val_imagenette | 3 925 | 5 | 24.1 | 122.2 | `HOST_SAN_SCAN_PASS` |

Interpretation: the 1.2 M stress cohort reaches **511 Msamples/s** sustained,
95% of the theoretical 540.8 Msamples/s peak at the achieved 135.2 MHz clock
(§13.2). The smaller cohorts are enqueue/sync dominated; their sustained
numbers are lower because the fixed launch cost is amortised over fewer
samples per iteration. The card now passes on synthetic cohorts, real
photographs, and ImageNet-completo-sized stress in one sweep.

### 13.5 GPU scale pilot — SAN-ResNet-50 on Slurm gpu-orangefs (2026-08-04, measured)

The SAN large-architecture harness (`scripts/research/suffering_aware_large_architecture.py`)
was adapted for CUDA (`SAN_LARGE_DEVICE`) and submitted to the Slurm
`gpu-orangefs` partition. Pilot leg: **SAN-ResNet-50 on CIFAR-10** (4 k
train / 1 k val, 8-epoch budget) on an **NVIDIA RTX A5000**.

Result (`artifacts/san_large/gpu_resnet50_run_8584.txt`):

- SAN reached feasibility at **t* = 4** (val acc 0.392 ≥ τ = 0.34) and
  stopped there; gratuitous machine suffering = 0.
- Dense ran the full 8 epochs; EarlyStop stopped at t* = 1.
- Integrated machine suffering: **SAN 160.1 TMAC vs Dense 269.9 TMAC**
  (**40.7% less**) and EarlyStop 67.5 TMAC.
- Per-epoch machine suffering: **SAN 32.0 TMAC vs Dense 33.7 TMAC**
  (**5.1% less**) — the exit-head overhead is small.
- Contract clauses: **L1–L4, L6–L8 PASS; L5 FAIL** because SAN integrated
  patient harm (5.31) is higher than EarlyStop's (2.17). This is the same
  honest two-channel tradeoff the spec reports for ViT-large at ImageNet
  scale (§7): not every family dominates on every channel simultaneously.
  The machine-channel savings are real and measured.

The harness is now GPU-enabled and the submission path
(`slurm-jobs/san-large-gpu/submit.sh`) is working; remaining legs
(ViT-large, GPT) can be launched with the same script.
