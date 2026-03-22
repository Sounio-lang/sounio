<!-- docs:meta
topic_id: repo.docs.papers.main.epistemic-types.benchmarks.l4-gemm-summary
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.main.epistemic-types.benchmarks.l4-gemm-summary
-->

# L4 GPU Epistemic GEMM Benchmark Results

**Generated from:** `artifacts/omega/l4_*.{txt,log}` (27 raw log files)
**Date range:** 2026-02-25 to 2026-02-26
**Kernel:** Sounio-generated PTX — epistemic GEMM with `Knowledge<T>` shadow registers

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA L4 |
| Memory | 23,034 MiB |
| Driver | 590.48.01 |
| Compute Capability | 8.9 (Ada Lovelace) |
| Host | `gpu-appliance-l4` (10.100.100.215) |
| Toolchain | rustc 1.93.1, cargo 1.93.1 |
| Grid | (64, 64, 1), Block: (16, 16, 1) |

## Results

All runs use `scripts/cuda_gemm_dispatch.py` with Sounio-generated PTX loaded via CUDA driver API. Threshold: >= 500 GFLOPS.

### 4096x4096x4096 (primary benchmark)

| Case | PTX | GFLOPS | ms/iter | Status | Source |
|------|-----|--------|---------|--------|--------|
| sm7_4096 (best) | 15,647 chars | **5,605.8** | 24.52 | PASS | `l4_4096_perf_pass_2026-02-25.log` |
| crystallize | sm7 | 5,473.8 | — | PASS | `l4_runner_crystallize_2026-02-26T102238Z.log` |
| direct dispatch | sm7 | 5,275.4 | — | PASS | `l4_real_gpu_dispatch_4096.log` |
| direct sm7 (v2 report) | sm7 | 5,270.7 | 26.08 | PASS | `l4_real_gpu_dispatch_direct_sm7_4096_2026-02-26T101117Z.log` |
| dispatch 2026-02-25 | sm7 | 5,229.8 | — | PASS | `l4_real_gpu_dispatch_4096_2026-02-25.log` |
| sm7 generic | 15,647 chars | 5,202.5 | — | PASS | `l4_4096_case_sm7_generic.log` |
| sm7_4096_i12 (12 iters) | 15,647 chars | 5,154.9 | 26.66 | PASS | `l4_4096_perf_pass_2026-02-25.log` |
| sm89 fix verify | sm89 | 4,526.9 | — | PASS | `l4_sm89_fix_verify_2026-02-25T093836Z.log` |
| sm89 revalidate | sm89 | 4,462.9 | — | PASS | `l4_sm89_revalidate_2026-02-25T125220Z.log` |
| onhost gen | 22,580 chars | 4,443.6 | 30.93 | PASS | `l4_4096_perf_pass_2026-02-25.log` |
| remote toolchain | sm7 | 4,440.1 | — | PASS | `l4_real_gpu_dispatch_4096_remote_toolchain_2026-02-25.log` |
| ptx_22580 | 22,580 chars | 4,367.7 | 31.47 | PASS | `l4_4096_perf_pass_2026-02-25.log` |
| sm89 initial | sm89 | **FAIL** | — | FAIL | `l4_4096_case_sm89.log` |

### 2048x2048x2048

| Case | GFLOPS | Status | Source |
|------|--------|--------|--------|
| dispatch 2026-02-25 | **7,584.6** | PASS | `l4_real_gpu_dispatch_2048_2026-02-25.log` |
| dispatch (undated) | 7,200.7 | PASS | `l4_real_gpu_dispatch_2048.log` |
| remote retry | 6,786.0 | PASS | `l4_real_gpu_dispatch_2048_remote_toolchain_retry_2026-02-25.log` |

### 1024x1024x1024

| Case | GFLOPS | Status | Source |
|------|--------|--------|--------|
| remote retry | 6,061.1 | PASS | `l4_real_gpu_dispatch_1024_remote_toolchain_retry_2026-02-25.log` |

### 8192x8192x8192 (scale stability)

| Case | GFLOPS | Status | Source |
|------|--------|--------|--------|
| sm7 stress | 4,956.8 | PASS | `l4_real_gpu_dispatch_8192_stress_2026-02-25.log` |
| sm7 deep | 4,876.3 | PASS | `l4_real_gpu_dispatch_8192_sm7_deep_2026-02-25.log` |
| onhost ptx | 4,355.4 | PASS | `l4_real_gpu_dispatch_8192_onhost_ptx_2026-02-25.log` |

**8192 stability:** mean 4,729.5 GFLOPS, stddev 266.6, min 4,355.4, max 4,956.8

Source: `l4_scale_stability_report.v1.txt`

## Summary Statistics

| Metric | Value |
|--------|-------|
| Peak GFLOPS (any size) | 7,584.6 (2048x2048) |
| Best at 4096x4096 | 5,605.8 |
| Median at 4096x4096 (sm7 only) | 5,215 |
| All runs vs 500 threshold | **22/23 PASS** (1 FAIL: sm89 initial) |
| 8192 stability stddev | 266.6 GFLOPS (5.6% of mean) |

## Known Issues

**SM89 `cuModuleLoadData` failure:** The initial SM89-targeted PTX (22,664 chars) failed with error code `0x000000c8` (CUDA_ERROR_INVALID_PTX). This was resolved by using SM7 generic PTX which runs correctly on the SM 8.9 hardware via forward compatibility. Subsequent SM89 runs after PTX regeneration passed (4,463-4,527 GFLOPS).

## Reproduction

```bash
# Requires SSH access to gpu-appliance-l4
GPU_HOST=gpu-appliance-l4 GEMM_M=4096 GEMM_N=4096 GEMM_K=4096 \
  bash scripts/gpu_test_runner.sh
```

See `scripts/gpu_test_runner.sh` for full configuration options.
