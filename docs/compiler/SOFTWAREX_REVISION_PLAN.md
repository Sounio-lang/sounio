<!-- docs:meta
topic_id: repo.docs.compiler.softwarex-revision-plan
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.softwarex-revision-plan
-->

# SoftwareX Reviews: Actionable Revision Plan (SOFTX-D-26-00069)

This document turns the SoftwareX editor/reviewer feedback for the preprint
**"Native GPU Octonion Support for Deep Learning: A Compiler-First Approach"**
into concrete, testable, repo-backed work items.

Status note (repo reality check, 2026-02-12):
- We currently have strong *CPU-side* validation and microbenchmarks.
- The PTX/MSL paths are primarily *code generation + static validation* today.
- Cross-device GPU *execution* and *performance* evidence is not yet presented in a
  reproducible, end-to-end way in the manuscript.

The reviewers are essentially asking us to (1) narrow/justify claims, and
(2) ship a reproducible benchmarking + profiling story that matches those claims.

## Reviewer #1: Major Points -> Work Items

### 1) Novelty + Positioning (“first”, “gap”, “why a new language?”)
Work items:
- Replace absolute novelty language (“first”, “no production-ready GPU implementation exists”)
  with qualified phrasing unless we can cite and demonstrate it.
- Expand “Related Work” to include:
  - library-based octonion/hypercomplex tooling (CPU and GPU, if any),
  - kernel/DSL approaches (Triton, TVM, MLIR, etc.),
  - what users can do today in major ecosystems (PyTorch/JAX) and what is missing.
- Add a short “Why compiler-level?” subsection that is specific and falsifiable:
  - what optimizations are enabled (fusion, layout, whole-program specialization),
  - what safety guarantees exist (effects, memory model constraints),
  - what is *not* provided yet (autodiff integration, full training stack).

Deliverables:
- Updated manuscript (`docs/compiler/TECHNICAL_REPORT.md`, `docs/compiler/TECHNICAL_REPORT.tex`)

### 2) Performance evaluation (methodology, scaling, baselines, and NVIDIA results)
Work items:
- Replace “single small case” framing with a scaling study:
  - include multiple matrix sizes and, if applicable, batched sizes
  - report variance + confidence intervals
- Add meaningful baselines on the *same hardware*:
  - CPU: compare naive vs blocked vs SIMD-friendly layouts
  - GPU: compare against a simple baseline kernel (even if non-octonion)
- Add methodology details:
  - compiler flags, warmup, measurement policy
  - kernel launch overhead isolation (especially for small sizes)
  - memory layout and alignment, precision mode, and any fast-math settings
- If we claim “PTX backend works on NVIDIA GPUs”, we must include:
  - an end-to-end “generate PTX -> run on NVIDIA -> verify correctness” path
  - at least one NVIDIA datapoint, ideally across sizes

Deliverables:
- A reproducible benchmark runner script (see “Reproducibility” section below)
- Roofline analysis/plot (Reviewer #2 requirement) for at least one representative kernel
- Manuscript section updates with the new results and methodology

### 3) Reproducibility + Usability
Work items:
- Add a minimal, verified reproduction path:
  - exact commit hash / tag
  - exact build commands
  - benchmark runner(s) + expected artifacts
- Add “what is validated” detail:
  - what tests cover (properties), tolerances, seeds
  - cross-backend comparisons (CPU interp vs JIT; GPU when available)
- Add “language surface” and “GPU execution model” clarity:
  - data layout for octonions/tensors
  - kernel invocation mechanism
  - supported devices and explicit fallbacks
- Add “compiler maturity / self-host” evidence:
  - strict self-host corpus gate (`scripts/selfhost/selfhost_zero_fallback_gate.sh`)
  - strict driver-output smoke gate for the bootstrap subset (`scripts/ci/selfhost_driver_output_gate.sh`)
  - machine-readable report generator for CI artifacts (`scripts/selfhost/selfhost_reproducibility_report.py`)
- Add a minimal end-to-end example relevant to DL:
  - forward pass of a toy octonion layer (already doable)
  - training demo if/when autodiff integration is real and verifiable

Deliverables:
- `scripts/paper/reproduce_octonion_preprint.sh` (tests + benches)
- `scripts/selfhost/selfhost_zero_fallback_gate.sh` + `scripts/ci/selfhost_driver_output_gate.sh` (self-host maturity gates)
- Manuscript appendix that names the scripts, commit/tag, and outputs

### 4) Presentation / Credibility Details
Work items:
- Provide derivations or references for FLOP counts where we use them as evidence.
- Explain any “sign rules” / multiplication-table encoding clearly:
  - what data structure encodes the algebra
  - how correctness is verified/maintained
- Ensure claims match evidence:
  - if GPU numbers aren’t included, do not imply GPU performance has been measured

## Reviewer #2: Roofline + Scientific Impact

### 1) Scientific impact / what does the reader learn?
Work items:
- Add a “Design lessons” section:
  - what worked in the compiler pipeline for this algebra
  - what did not (non-associativity effects on transformations, etc.)
  - backend-specific constraints encountered

### 2) Roofline plot requirement
Work items:
- Add a roofline plot for a representative kernel:
  - operational intensity assumptions stated
  - measured achieved throughput (GFLOP/s)
  - peak compute + peak bandwidth source stated (spec or measured)
- Include a script that regenerates the roofline CSV/figure from raw benchmark output.

Deliverables:
- `docs/compiler/TECHNICAL_REPORT.tex`: roofline figure
- `scripts/benchmarks/roofline_octonion_matmul.py`: emits CSV for pgfplots (no external Python deps)

## Execution Order (Recommended)
1. Fix reproducibility paths and scripts (so every future claim is verifiable).
2. Improve CPU benchmarking + baselines + roofline (immediate, hardware-agnostic).
3. Add real NVIDIA execution path + results (requires CUDA hardware).
4. Update novelty/positioning text *after* we know what we can demonstrate.
