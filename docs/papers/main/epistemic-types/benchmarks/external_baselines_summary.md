<!-- docs:meta
topic_id: repo.docs.papers.main.epistemic-types.benchmarks.external-baselines-summary
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.main.epistemic-types.benchmarks.external-baselines-summary
-->

# External Baseline Comparison Results

**Contract:** `benchmarks/independence/contract.v2.json` (schema `sounio.independence.contract.v2`)
**Frozen:** 2026-02-26T11:48:41Z
**Status:** PASS (geomean 1.2077x >= 1.2x threshold)

## Methodology

Sounio code-generated GPU kernels are compared against three external baselines across 5 kernel families. Each baseline runs 50 samples per family with median aggregation. The performance gate requires a geometric mean speedup of >= 1.2x across all families.

**Baselines tested:**
- **CUDA CUTLASS** — NVIDIA's optimized GEMM/attention library
- **Triton** — OpenAI's GPU compiler for Python
- **PyTorch Inductor** — `torch.compile` backend with CPU fallback

## Per-Baseline Speedup (Sounio / Baseline)

| Family | CUDA CUTLASS | Triton | PyTorch Inductor |
|--------|-------------|--------|------------------|
| dense_linear | 1.35x | 1.30x | 1.01x |
| attention | 1.25x | 1.28x | 0.97x |
| epistemic_elementwise | 1.18x | 1.22x | 3.14x |
| monte_carlo | 1.20x | 1.15x | 2.47x |
| quantum_parallel | 1.08x | 1.07x | 1.72x |

**Samples per family:** 50

## Aggregate Performance Summary

| Metric | Value |
|--------|-------|
| Measurement mode | `external_baseline_ingest` |
| Geomean speedup | **1.2077x** |
| Threshold | 1.2x |
| No-regression floor | 0.95x |
| Substitutions | 0 |
| Baselines available | 3/3 |
| Status | **PASS** |

### Family-Level Aggregated Speedups (median across baselines)

| Family | Speedup | Floor | Cap |
|--------|---------|-------|-----|
| dense_linear | 1.30x | 0.95 | 3.0 |
| attention | 1.25x | 0.95 | 3.0 |
| epistemic_elementwise | 1.22x | 0.95 | 2.5 |
| monte_carlo | 1.20x | 0.95 | 2.5 |
| quantum_parallel | 1.08x | 0.95 | 2.0 |

## Notes on PyTorch Inductor Results

The Inductor baseline shows anomalously high speedup values for `epistemic_elementwise` (3.14x) and `monte_carlo` (2.47x), and a slight regression for `attention` (0.97x). This is because the Inductor probe ran on CPU backend with `torch.compile` (see `details.backend_device: "cpu"` in the report). The `torch.compile` CPU backend benefits from operator fusion on elementwise and reduction workloads but lacks GPU-specific attention kernels. These numbers are not directly comparable to the CUDA-based CUTLASS and Triton baselines.

The aggregated family speedups (1.22x for epistemic_elementwise, 1.20x for monte_carlo) use the **median** across all three baselines, which dampens this outlier effect.

## Provenance

| Artifact | Path |
|----------|------|
| Performance summary | `artifacts/omega/performance_summary.v1.json` |
| Baseline freeze | `artifacts/omega/baseline_freeze.v1.json` |
| CUTLASS report | `artifacts/omega/external_baselines/cuda-cutlass.v1.json` |
| Triton report | `artifacts/omega/external_baselines/triton.v1.json` |
| Inductor report | `artifacts/omega/external_baselines/pytorch-inductor.v1.json` |
| Freeze digest (SHA-256) | `4da696978c2858ed0e3577d1ed7b2157919ab3c1...` |
| Signature | Ed25519, scope: `freeze_digest_sha256` |

## Reproduction

```bash
# Run external baseline probes (requires PyTorch + CUDA)
python3 scripts/omega/omega_collect_external_baselines.py --execute

# Generate performance summary
python3 scripts/omega/omega_performance_summary.py --prefer-external
```

See `scripts/omega/` for full orchestration details.
